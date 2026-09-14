"""The trust horizon over the M3c cells, part 1: load, check, one reference pass, one record per cell.

M3c's ladder records store per-step MEAN error curves. Whether `random_vit`
loses least at h=45 because its prior drifts slowly -- stays near persistence,
which the metric rewards -- or because `frozen_ssl` moves the wrong way cannot
be read off a mean: magnitude and direction are per-window quantities. This
script re-runs the ladder's canonical reference pass on each frozen checkpoint
with the per-window trajectories KEPT (`reference_trajectories`), and writes
one `trust_<arm>_seed<n>.json` per cell holding, per window and per horizon
step, the crossing step in both channels, the persistence margin, the three
displacement ratios, the direction cosine, and the held-out scale-corrected
error with its fold alphas -- the quantities of the M3d design, section 2.2.
The pooling over the nine records and the two readings are the second part
of this script and come after the per-cell loop in `main`.

LOADING IS `diagnose_dynamics.py`'S, NOT A SECOND COPY. The checkpoint path
and its arm/seed validation (`load_checkpoint_model`), the diagnostic's path
and the probe-measurability rule are imported from that script by path -- the
way every script test loads a script -- because each is a rule the ladder
already applied to these nine cells, and a duplicate here could drift from
the ladder's while both kept passing, leaving the two tools reading one cell
through two loaders. Everything else on the loading path (`episode_split`,
`fit_probes`, `evaluate_rollout`, the record I/O) lives in `src/` and is
imported from there. The probe is REFIT, exactly as the ladder refits it --
`fit_probes` on the train split at the cell's seed and protocol -- because no
probe weights exist on disk; the refit is part of what the self-check pins.

THE CHECKS RUN IN A FIXED ORDER, PER CELL, BEFORE ITS RECORD IS WRITTEN,
and each has its own status:

  EXIT_NO_CHECKPOINTS (11)    -- a REQUESTED cell lacks its checkpoint, its
    study record or its diagnostic. Every requested cell is loaded before any
    probe refit, so a run asking for nine and finding eight stops in a
    second, not twenty minutes in. (Stricter than `diagnose_dynamics.py`,
    which runs whatever cells it finds: the readings this script exists for
    need every arm.)
  EXIT_SPLIT_MISMATCH (12)    -- the split by name is not the record's.
  EXIT_RECORD_MISMATCH (14)   -- `evaluate_rollout` no longer reproduces the
    study record's `curves.rssm_position` (an ENVIRONMENT difference: on cpu
    the mps-trained cells miss by 6-12 map units). A `--context` or
    `--horizon` that disagrees with the protocol the diagnostic records is
    refused under this status too, UP FRONT: a rollout at another protocol
    cannot reproduce the record, which is how `diagnose_dynamics.py` surfaces
    the same mismatch -- after its refit, twenty seconds later.
  EXIT_SELF_CHECK_FAILED (30) -- the trust pass's own mean curves are not
    bitwise the diagnostic's `curves.reference_position` and
    `curves.persistence_position` (`max |delta| == 0.0`, the ladder's own
    `record_reproduction` rule), or its windows are not the diagnostic's.
    Same windows, same rollout, same refit probe, or this is not measuring
    what the ladder measured -- and the cell's record is NOT written, because
    the second part pools every record it finds.

11, 12 and 14 carry `diagnose_dynamics.py`'s meanings ON PURPOSE -- a wrapper
reading the status learns the same thing from either tool -- and 30 is new,
distinct from every other tool's (1-23) and from argparse's own 2. A
mislabelled checkpoint (that script's 16) and an unsupported device (its 17)
have no status here and surface as the library's own exceptions.
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import Trajectories, reference_trajectories
from mbfps.eval.probe import fit_probes
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.study import (
    SPLIT_SEED,
    StudyJob,
    git_sha,
    job_record_path,
    load_record,
    write_record,
)
from mbfps.eval.trust import (
    crossing_step,
    displacement_decomposition,
    embedding_ratio,
    moved_mask,
    persistence_margin,
    scale_corrected_error,
)
from mbfps.models.encoders import encoder_backbone
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script.

    Resolved against THIS file, so it works run as `python scripts/...` and
    loaded by a test alike; a bare `import diagnose_dynamics` would need the
    scripts directory on `sys.path`, which only the first of those provides.
    """
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_trust_horizon", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_diagnose_dynamics = _sibling("diagnose_dynamics")
checkpoint_path = _diagnose_dynamics.checkpoint_path
load_checkpoint_model = _diagnose_dynamics.load_checkpoint_model
diagnostic_record_path = _diagnose_dynamics.diagnostic_record_path
probe_is_measurable = _diagnose_dynamics.probe_is_measurable

EXIT_OK = 0
EXIT_NO_CHECKPOINTS = 11
EXIT_SPLIT_MISMATCH = 12
EXIT_RECORD_MISMATCH = 14
EXIT_SELF_CHECK_FAILED = 30
"""11, 12 and 14 are `diagnose_dynamics.py`'s, with its meanings. 30 is new
and is in no other tool's range: run_study 1/3-6/23, report_study 7-10,
spike 10, diagnose 11-17, pool 18-22, argparse 2, a traceback 1."""


class CellMissing(FileNotFoundError):
    """A requested cell lacks its checkpoint, its study record or its
    diagnostic -- the ONE error `main` maps to `EXIT_NO_CHECKPOINTS`.

    A type of its own so `main` catches exactly this: `load_record` raises the
    base `FileNotFoundError` for a file that vanished between the existence
    check and the read, and that is a race worth a traceback, not a status.
    """


@dataclass(frozen=True)
class Cell:
    """One cell's three files, read. The model is built from `checkpoint` by
    `load_checkpoint_model` once the device is known."""

    arm: str
    seed: int
    checkpoint: Path
    record: dict
    diagnostic: dict


def load_cell(out_dir: Path, arm: str, seed: int) -> Cell:
    """The checkpoint, the study record and the diagnostic of one cell, or
    `CellMissing` naming the first of the three that is not there."""
    out_dir = Path(out_dir)
    checkpoint = checkpoint_path(out_dir, arm, seed)
    record_path = job_record_path(out_dir, StudyJob(arm=arm, seed=seed))
    diagnostic_path = diagnostic_record_path(out_dir, arm, seed)
    for kind, path in (
        ("checkpoint", checkpoint),
        ("study record", record_path),
        ("diagnostic", diagnostic_path),
    ):
        if not path.exists():
            raise CellMissing(f"{arm} seed {seed}: no {kind} at {path}")
    return Cell(
        arm=arm,
        seed=seed,
        checkpoint=checkpoint,
        record=load_record(record_path),
        diagnostic=load_record(diagnostic_path),
    )


def _max_delta(ours, theirs) -> tuple[float, int]:
    """`max |ours - theirs|` and the 1-based step it is at; `(inf, 0)` when
    the two are not even the same shape -- a curve written at another horizon
    is a mismatch to report, not a broadcasting traceback."""
    ours = np.asarray(ours, dtype=float)
    theirs = np.asarray(theirs, dtype=float)
    if ours.shape != theirs.shape:
        return float("inf"), 0
    delta = np.abs(ours - theirs)
    return float(delta.max()), int(delta.argmax()) + 1


@dataclass(frozen=True)
class SelfCheck:
    """Does the trust pass reproduce the diagnostic it is about to be read
    beside. Two exact curve equalities (the ladder's `record_reproduction`
    rule, `max |delta| == 0.0`) and two window identities."""

    reference_position_max_delta: float
    reference_position_step: int
    persistence_position_max_delta: float
    persistence_position_step: int
    windows_total_match: bool
    windows_episode_match: bool

    def failures(self) -> list[str]:
        out = []
        if self.reference_position_max_delta != 0.0:
            out.append(_curve_failure(
                "reference_position", self.reference_position_max_delta, self.reference_position_step
            ))
        if self.persistence_position_max_delta != 0.0:
            out.append(_curve_failure(
                "persistence_position", self.persistence_position_max_delta, self.persistence_position_step
            ))
        if not self.windows_total_match:
            out.append("windows.total is not the diagnostic's")
        if not self.windows_episode_match:
            out.append("windows.episode is not the diagnostic's")
        return out

    @property
    def ok(self) -> bool:
        return not self.failures()

    def record(self) -> dict:
        return {
            "reference_position_max_delta": self.reference_position_max_delta,
            "persistence_position_max_delta": self.persistence_position_max_delta,
            "windows_total_match": self.windows_total_match,
            "windows_episode_match": self.windows_episode_match,
            "ok": self.ok,
        }


def _curve_failure(name: str, delta: float, step: int) -> str:
    if step == 0:
        return f"{name} has a different length from the diagnostic's"
    return f"{name} differs from the diagnostic's by max |delta| {delta:.3e} at step {step}"


def self_check(traj: Trajectories, diagnostic: dict) -> SelfCheck:
    """The two curves are recomputed HERE from the rows the record is built
    from -- the mean over windows of |p_hat(h) - p(h)| and of |p_hat(0) -
    p(h)|, spec 2.3 -- and compared with the diagnostic's. They are NOT read
    off `traj.reference_position` / `traj.persistence_position`: those are
    the pass's own reduction of the same rows (Task 3 pins them bitwise
    equal to these on a clean run), and judging them would let a regression
    that kept the curve right and the rows wrong write a record."""
    curves = diagnostic["curves"]
    reference_rows = _distance(traj.positions, traj.true_positions)
    persistence_rows = _distance(traj.positions_at_context[:, None, :], traj.true_positions)
    reference, reference_step = _max_delta(
        reference_rows.mean(axis=0), curves["reference_position"]
    )
    persistence, persistence_step = _max_delta(
        persistence_rows.mean(axis=0), curves["persistence_position"]
    )
    windows = diagnostic["windows"]
    # `null` when the ladder carried no clustering: a trust record cannot be
    # clustered on nothing, so that is a mismatch, never `range(n)`.
    episode = windows["episode"]
    return SelfCheck(
        reference_position_max_delta=reference,
        reference_position_step=reference_step,
        persistence_position_max_delta=persistence,
        persistence_position_step=persistence_step,
        windows_total_match=int(traj.windows_total) == int(windows["total"]),
        windows_episode_match=(
            episode is not None
            and [int(e) for e in traj.window_episode] == [int(e) for e in episode]
        ),
    )


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-D Euclidean distance over the last axis -- `position_error`'s norm."""
    return np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)


def trust_record(arm, seed, traj: Trajectories, diagnostic: dict, *, context, horizon, device) -> dict:
    """The LIVE record for one cell: numpy arrays and real NaNs. `write_record`
    sanitises it (`null` plus the `nonfinite` map) on the way to disk, so
    this dict must not carry a `nonfinite` key of its own.

    Every quantity goes through the pure functions of `mbfps.eval.trust`;
    this function only names which of the pass's arrays is which. Both
    channels' errors are the same shape and both crossings are the same
    call, so the wiring -- the probe's positions to the probe crossing, the
    embedding distances to the free crossing, `positions_real` as the third
    argument of the decomposition and never `positions` -- is what the
    fabricated-trajectory tests pin by value.
    """
    moved = moved_mask(traj.true_positions, traj.true_at_context)
    model_err = _distance(traj.positions, traj.true_positions)
    persist_err = _distance(traj.positions_at_context[:, None, :], traj.true_positions)
    decomposition = displacement_decomposition(
        traj.positions, traj.true_positions, traj.positions_at_context,
        traj.true_at_context, traj.positions_real, moved,
    )
    scale = scale_corrected_error(
        traj.positions, traj.true_positions, traj.positions_at_context, moved, traj.window_episode,
    )
    check = self_check(traj, diagnostic)
    curves = diagnostic["curves"]
    return {
        "arm": arm,
        "seed": seed,
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "episodes": {"val": list(diagnostic["episodes"]["val"])},
        "windows": {
            "total": int(traj.windows_total),
            "episode": [int(e) for e in traj.window_episode],
        },
        "probe": {
            "selection_r2": float(diagnostic["probe"]["embedding_selection_r2"]),
            # The ladder's rule, asked with no null band: the persistence-to-
            # floor band at the FINAL step must be positive, and it is read off
            # the persistence curve, never the reference.
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": curves["persistence_position"][-1],
                    "floor": curves["floor_position"][-1],
                },
                widest_se=0.0,
            )),
        },
        "self_check": check.record(),
        "crossing": {
            "probe": crossing_step(model_err, persist_err, moved),
            "free": crossing_step(
                traj.embedding_distance_to_truth, traj.embedding_persistence_distance, moved
            ),
        },
        "margin": persistence_margin(model_err, persist_err),
        "ratio_probe": decomposition.ratio_probe,
        "ratio_raw": decomposition.ratio_raw,
        "ratio_free": embedding_ratio(
            traj.embedding_displacement, traj.true_embedding_displacement, moved
        ),
        "cosine": decomposition.cosine,
        # The numerator and denominator series of the two ratios, kept
        # beside the per-window ratios: the pooling's estimand is a ratio of
        # MEDIANS (spec 3.1), which the ratios alone cannot recover.
        "displacement": {
            "probe_hat": _distance(traj.positions, traj.positions_at_context[:, None, :]),
            "probe_real": _distance(traj.positions_real, traj.positions_at_context[:, None, :]),
            "free_hat": np.asarray(traj.embedding_displacement, dtype=float),
            "free_true": np.asarray(traj.true_embedding_displacement, dtype=float),
        },
        "scale": {
            "alpha_a": scale.alpha_a,
            "alpha_b": scale.alpha_b,
            "score_a": scale.score_a,
            "score_b": scale.score_b,
            "held_out": scale.held_out,
            "boundary": scale.boundary,
            "folds_available": bool(scale.folds_available),
        },
        "counts": {
            "not_moved": (~moved).sum(axis=0),
            "zero_displacement": decomposition.zero_displacement.sum(axis=0),
            # Windows that never move within the horizon: their crossing is
            # NaN and they are counted here, not pooled.
            "never_moved": int((~moved.any(axis=1)).sum()),
        },
    }


def trust_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"trust_{arm}_seed{seed}.json"


def write_trust_record(out_dir: Path, record: dict) -> Path:
    """Atomic and strict-JSON, through `write_record`: a NaN ratio is `null`
    plus its token, never a bare `NaN` the pooling reader would choke on."""
    path = trust_record_path(out_dir, record["arm"], record["seed"])
    write_record(path, record)
    return path


def protocol_mismatch(args, diagnostic: dict) -> str | None:
    """The flag that disagrees with the protocol the diagnostic was written
    at, or None. A flag left at None takes the diagnostic's value."""
    for flag in ("context", "horizon"):
        asked = getattr(args, flag)
        recorded = int(diagnostic[flag])
        if asked is not None and asked != recorded:
            return f"--{flag} {asked} is not the {flag} the diagnostic was written at ({recorded})"
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study_v2"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    # None -> each cell's diagnostic says what it was written at. A value
    # that disagrees with the diagnostic is refused, see `protocol_mismatch`.
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser


def _run_cell(args, cell: Cell, device, train, val) -> tuple[int, dict | None]:
    """One cell: 12, the protocol (14), the refit, the reproduction (14), the
    reference pass, 30, the record. Returns `(EXIT_OK, record)` once the
    record is written, `(status, None)` on any refusal: `main` keeps the live
    records for the pooling that follows the loop."""
    arm, seed = cell.arm, cell.seed
    names = [p.name for p in val]
    if cell.record["episodes"]["val"] != names:
        print(
            f"\nSPLIT MISMATCH for {arm} seed {seed}: the held-out episodes are not "
            "the ones the record was scored on.\n"
            f"  split:  {names}\n  record: {cell.record['episodes']['val']}"
        )
        return EXIT_SPLIT_MISMATCH, None

    mismatch = protocol_mismatch(args, cell.diagnostic)
    if mismatch is not None:
        print(
            f"\nRECORD MISMATCH for {arm} seed {seed}: {mismatch}. A rollout at "
            "another protocol cannot reproduce the record's curve, so nothing is refit."
        )
        return EXIT_RECORD_MISMATCH, None
    context = int(cell.diagnostic["context"]) if args.context is None else args.context
    horizon = int(cell.diagnostic["horizon"]) if args.horizon is None else args.horizon

    cfg = get_config(arm, seed=seed, device=args.device)
    model = load_checkpoint_model(args.out, arm, seed, cfg, device)
    backbone = encoder_backbone(cfg.encoder)
    # The probe is REFIT at the rollout's own context/horizon and at the
    # cell's seed, exactly as the ladder refit it: it is applied to latents
    # filtered from a zero state for exactly `context` real frames.
    _, embedding_probe = fit_probes(
        model, train, backbone, device, context=context, horizon=horizon, seed=seed,
    )
    common = dict(
        context=context, horizon=horizon, seed=seed, device=device, feature_backbone=backbone,
    )
    reference = evaluate_rollout(model, val, embedding_probe, **common)
    reproduction, step = _max_delta(
        reference.rssm_position, cell.record["curves"]["rssm_position"]
    )
    if reproduction != 0.0:
        print(
            f"\nRECORD MISMATCH for {arm} seed {seed}: evaluate_rollout no longer "
            f"reproduces the study record's curves.rssm_position (max abs "
            f"{reproduction:.3e} at step {step}). Measured, the records reproduce "
            f"bitwise on mps and miss on cpu by an arm-dependent 6-12 map units. "
            f"This run used device={device} torch={torch.__version__}."
        )
        return EXIT_RECORD_MISMATCH, None

    traj = reference_trajectories(model, val, embedding_probe, **common)
    check = self_check(traj, cell.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {arm} seed {seed}: " + "; ".join(check.failures())
            + ". Same windows, same rollout, same refit probe -- or this is not "
            "measuring what the ladder measured. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None

    record = trust_record(
        arm, seed, traj, cell.diagnostic, context=context, horizon=horizon, device=device,
    )
    path = write_trust_record(args.out, record)
    episodes = len(set(int(e) for e in traj.window_episode))
    folds = (
        "available" if record["scale"]["folds_available"] else
        "UNAVAILABLE (one episode cannot make two folds: alpha, the held-out "
        "error and every episode-clustered SE are NaN)"
    )
    print(
        f"{arm} seed {seed}: {traj.windows_total} windows from {episodes} validation "
        f"episode{'' if episodes == 1 else 's'}; folds {folds}; self-check max|delta| "
        f"reference {check.reference_position_max_delta:.1e} persistence "
        f"{check.persistence_position_max_delta:.1e}; wrote {path}"
    )
    return EXIT_OK, record


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = get_device(prefer=args.device)
    # EVERY requested cell, before any refit: the readings need all of them,
    # and a missing ninth cell found after eight refits is twenty minutes late.
    try:
        cells = [load_cell(args.out, arm, seed) for arm in args.arms for seed in args.seeds]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    # The study's own split: the shared VAL_FRACTION, and SPLIT_SEED -- fixed
    # at 0 and deliberately NOT the cell's seed.
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    # The live records, keyed (arm, seed), and the run's horizon -- the same
    # on every cell, since the protocol check refuses a cell at another one.
    # The pooling after the loop reads both.
    records: dict[tuple[str, int], dict] = {}
    horizon = 0
    for cell in cells:
        status, record = _run_cell(args, cell, device, train, val)
        if status != EXIT_OK:
            return status
        records[(cell.arm, cell.seed)] = record
        horizon = int(record["horizon"])
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
