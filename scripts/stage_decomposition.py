"""The stage decomposition over the M3c cells: which stage of imagination fails?

M3e ruled memorisation out and M3f ruled checkpoint timing out, and the
quantity both were built around -- the absolute `embedding` loss -- is
confounded by the embedding's own scale (spec 2026-09-19 M3g, section 1).
What is not known is which STAGE of imagination fails at step 20,000:
whether the posterior carries the frame (encode), whether one prior step
from the true state predicts the next posterior (predict), whether that
survives fifteen open-loop steps on the prior's own samples (carry), or
whether a latent that tracks is lost in the rendering the gate scores
(decode). This script measures the four in the model's own 32 x 32
categorical latent, where the posterior on the validation windows is
exactly known:

  evaluate  per cell: `prepare_cell` on the reference study (12, 14), the
            trust pass with `keep_latents` -- the rollout the gate scored,
            plus the posterior over the window, the teacher-forced and the
            open-loop prior -- the bitwise self-check against the cell's
            diagnostic (30), the step-1 identity (30), the statistics of
            spec 2.3, one stages_<arm>_seed<n>.json. Then the CONTROL cells
            -- known-blind rung-4000 checkpoints from the M3f ladder, loaded
            as rung cells -- through the same pass, into
            stages_control_<arm>_seed<n>.json.
  read      pool the records, read the control (34), decide Reading S,
            print the tables, write stages.txt and stages_curves.png.

LOADING IS `trust_horizon.py`'S and `checkpoint_ladder.py`'S: `Cell`,
`load_cell`, `self_check`, `prepare_cell` and `rung_cell` are imported by
path, so a cell is refused here for the reasons and in the words the other
tools refuse it.

THE CHECKS, BY PHASE, each with its own status:

  evaluate: EXIT_NO_CHECKPOINTS (11)     a requested cell or control lacks its
                                          checkpoint or record (a cell, also its
                                          diagnostic); judged for every one
                                          before any pass runs.
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   a cell's trust pass is not bitwise its
                                          diagnostic, or (cell or control) the
                                          open-loop prior at step 1 is not the
                                          teacher-forced prior at step 1.
  read:     EXIT_NO_CHECKPOINTS (11)     a requested record is missing.
            EXIT_SELF_CHECK_FAILED (30)   a record's recorded self-check is not
                                          ok, or its identity flag is not set.
            EXIT_CONTROL_MISREAD (34)     NEW. the known-blind control reads
                                          anything but ENCODE_FAILS: no arm's
                                          reading is printed, no stages.txt.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 34 is in
no other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10,
diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, argparse
2, a traceback 1).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import LatentIdentityError, reference_trajectories
from mbfps.eval.split_gap import decision_horizon
from mbfps.eval.stages import (
    SEEDS_REQUIRED,
    entropy_by_group,
    information,
    marginal_accuracy,
    marginal_classes,
    open_accuracy,
    open_marginal,
    open_persistence,
    persistence_accuracy,
    teacher_accuracy,
    teacher_nll,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.eval.trust import moved_mask
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_stage_decomposition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
_ladder = _sibling("checkpoint_ladder")
rung_cell = _ladder.rung_cell

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_CONTROL_MISREAD = 34
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 34 is new and in no other tool's range."""

PHASES: tuple[str, ...] = ("evaluate",)
CONTROL_ARM: str = "frozen_ssl"
CONTROL_SEEDS: tuple[int, ...] = (1, 2)
CONTROL_DIR: Path = Path("runs/m3f_ladder/step4000")
"""Spec 2.1: the known-blind control -- frozen_ssl seeds 1 and 2 at rung 4000
of the M3f ladder (reconstruction R^2 ~ 0, posterior equal to prior)."""
CONTROL_LABEL: str = "control"
"""The arm label the control's records pool under: a fourth arm to the
pooling, never one of the three."""
KINDS: tuple[str, ...] = ("cell", "control")


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------


def stages_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"stages_{arm}_seed{seed}.json"


def control_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"stages_control_{arm}_seed{seed}.json"


def record_path(out_dir: Path, kind: str, arm: str, seed: int) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return (stages_record_path if kind == "cell" else control_record_path)(out_dir, arm, seed)


# ---------------------------------------------------------------------------
# evaluate: one cell's pass, statistics and record.
# ---------------------------------------------------------------------------


_cell_args = _ladder._cell_args
"""`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
args; a cell is read with `out` pointed at the directory it lives in -- the
ladder's helper, imported rather than copied."""


def cell_statistics(traj, *, context: int) -> dict:
    """Spec 2.3 over one cell's kept latents and rows: every per-window
    series `read` pools, the components the tables print, the companions.
    The decode inputs are stored as the two distances of each channel so
    `read` takes the margin at its own h through `stages.decode_margin`."""
    post, teacher, opened = traj.post_logits, traj.prior_teacher_logits, traj.prior_open_logits
    classes = marginal_classes(post, context)
    true_positions = np.asarray(traj.true_positions, dtype=np.float64)
    persistence_position = np.linalg.norm(
        np.asarray(traj.positions_at_context, dtype=np.float64)[:, None, :] - true_positions, axis=-1
    )
    model_position = np.linalg.norm(np.asarray(traj.positions, dtype=np.float64) - true_positions, axis=-1)
    return {
        "latent": {"groups": int(post.shape[-2]), "classes": int(post.shape[-1])},
        "information": information(post, teacher, context),
        "accuracy": {
            "teacher": teacher_accuracy(post, teacher, context),
            "persistence": persistence_accuracy(post, context),
            "marginal": marginal_accuracy(post, classes, context),
        },
        "open": {
            "accuracy": open_accuracy(post, opened, context),
            "persistence": open_persistence(post, context),
            "marginal": open_marginal(post, classes, context),
        },
        "decode": {
            "persistence_distance": np.asarray(traj.embedding_persistence_distance, dtype=np.float64),
            "distance_to_truth": np.asarray(traj.embedding_distance_to_truth, dtype=np.float64),
            "probe_persistence": persistence_position,
            "probe_model": model_position,
            "moved": moved_mask(traj.true_positions, traj.true_at_context),
        },
        "companions": {
            "entropy": entropy_by_group(post[:, context:]),
            "marginal_classes": classes,
            "nll": teacher_nll(post, teacher, context),
            "rendering_median": np.median(traj.posterior_rendering_distance, axis=0),
            "jitter_median": np.median(traj.true_step_displacement, axis=0),
        },
    }


def stages_record(cell: Cell, traj, stats: dict, *, kind: str, source: Path, context: int,
                  horizon: int, h: int, device, check) -> dict:
    """The LIVE record for one cell or control: numpy arrays and real NaNs;
    `write_record` sanitises it. `label` is what the pooling groups by -- the
    arm for a cell, `CONTROL_LABEL` for a control -- so a control at the same
    arm and seed as a cell never pools with it."""
    return {
        "arm": cell.arm,
        "seed": int(cell.seed),
        "kind": kind,
        "label": cell.arm if kind == "cell" else CONTROL_LABEL,
        "source": str(source),
        "step": int(cell.record["steps"]),
        "record_git_sha": cell.record["git_sha"],
        "context": int(context),
        "horizon": int(horizon),
        "decision_h": int(h),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "episodes": {"val": list(cell.record["episodes"]["val"])},
        "windows": {
            "total": int(traj.windows_total),
            "episode": [int(e) for e in traj.window_episode],
            "clusters": int(np.unique(np.asarray(traj.window_episode)).size),
        },
        "self_check": None if check is None else check.record(),
        "identity": True,
        **stats,
    }


def evaluate_cell(args, cell: Cell, device, train, val, *, kind: str, source: Path) -> tuple[int, dict | None]:
    """One cell or control: 12, the protocol (14), the refit, the
    reproduction (14), the trust pass with its latents, the identity (30), a
    cell's self-check (30), the statistics, the record."""
    status, prepared = prepare_cell(_cell_args(args, source), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    try:
        traj = reference_trajectories(
            prepared.model, val, prepared.embedding_probe, keep_latents=True, **prepared.common,
        )
    except LatentIdentityError as error:
        print(
            f"\nSELF-CHECK FAILED for {cell.arm} seed {cell.seed} ({kind}): {error}. The "
            "open-loop and teacher-forced passes did not start from the same state, so nothing "
            "read off them is one rollout. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None
    check = None
    if kind == "cell":
        check = self_check(traj, cell.diagnostic)
        if not check.ok:
            print(
                f"\nSELF-CHECK FAILED for {cell.arm} seed {cell.seed}: " + "; ".join(check.failures())
                + ". Same windows, same rollout, same refit probe -- or this is not the rollout "
                "the gate scored. No record written."
            )
            return EXIT_SELF_CHECK_FAILED, None
    h, _ = decision_horizon(prepared.horizon)
    stats = cell_statistics(traj, context=prepared.context)
    record = stages_record(
        cell, traj, stats, kind=kind, source=source, context=prepared.context,
        horizon=prepared.horizon, h=h, device=device, check=check,
    )
    path = record_path(args.out, kind, cell.arm, cell.seed)
    write_record(path, record)
    accuracy = stats["accuracy"]
    print(
        f"{cell.arm} seed {cell.seed} ({kind}, step {record['step']}): information "
        f"{float(np.mean(stats['information'])):.3f} nats; teacher {float(np.mean(accuracy['teacher'])):.3f} / "
        f"persistence {float(np.mean(accuracy['persistence'])):.3f} / marginal "
        f"{float(np.mean(accuracy['marginal'])):.3f}; open({h}) "
        f"{float(np.mean(stats['open']['accuracy'][:, h - 1])):.3f}; wrote {path}"
    )
    return EXIT_OK, record


def load_requested(args, cells, controls) -> list[tuple[str, Path, Cell]]:
    """Every requested cell and control, loaded BEFORE any pass runs, so a
    missing one is reported (11) before minutes are spent."""
    loaded = []
    for arm, seed in cells:
        loaded.append(("cell", args.reference, load_cell(args.reference, arm, seed)))
    for arm, seed in controls:
        loaded.append(("control", args.control, rung_cell(args.control, arm, seed)))
    return loaded


def evaluate_phase(args, cells, controls, device, train, val) -> int:
    try:
        loaded = load_requested(args, cells, controls)
    # `rung_cell` is `checkpoint_ladder.py`'s own, and that script loads
    # `trust_horizon.py` by path itself -- a second, independent execution of
    # the same file, so its `CellMissing` is a distinct class object from
    # this module's own `_trust.CellMissing` even though both are raised for
    # the same reason. A control's missing checkpoint must be caught too.
    except (CellMissing, _ladder.CellMissing) as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.out.mkdir(parents=True, exist_ok=True)
    for kind, source, cell in loaded:
        status, _ = evaluate_cell(args, cell, device, train, val, kind=kind, source=source)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3g_stages"))
    parser.add_argument("--reference", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: the nine 20,000-step cells")
    parser.add_argument("--control", type=Path, default=CONTROL_DIR,
                        help="the rung directory holding the known-blind control cells")
    parser.add_argument("--control-arm", default=CONTROL_ARM, choices=list(ARMS))
    parser.add_argument("--control-seeds", nargs="+", type=int, default=list(CONTROL_SEEDS))
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the cell's diagnostic (a control: its record) says what it was
    # written at; a value that disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=PHASES, default=PHASES[-1])
    parser.add_argument("--figure", type=Path, default=None,
                        help="the curves figure; default <out>/stages_curves.png")
    return parser


def _check_usage(parser: argparse.ArgumentParser, args) -> None:
    for name, other in (("--reference", args.reference), ("--control", args.control)):
        if args.out.resolve() == Path(other).resolve():
            parser.error(
                f"--out and {name} are the same directory; a record written into a study "
                "directory is a study directory changed"
            )
    if len(set(args.control_seeds)) < SEEDS_REQUIRED:
        parser.error(
            f"--control-seeds needs at least {SEEDS_REQUIRED} distinct seeds; with fewer no stage "
            "can pass, so the control would read ENCODE_FAILS vacuously"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    controls = [(args.control_arm, int(seed)) for seed in args.control_seeds]
    if args.phase in ("evaluate", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        status = evaluate_phase(args, cells, controls, device, train, val)
        if status != EXIT_OK:
            return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
