"""The checkpoint ladder over the M3c cells: retrain each cell to 5,000 steps saving five rungs, evaluate every rung and the 20,000-step reference, then pool and decide Reading T.

M3e ruled memorisation out and read, off the study records' learning curves,
that the ViT arms' one-step `embedding` loss reaches its minimum at step
1,610-4,535 of 20,000 and rises afterwards. Every rollout number M3b-M3e
recorded was measured at step 20,000. This script asks whether the model at
its own loss minimum rolls out better (spec 2026-09-18 M3f):

  train     retrain each requested cell from the study's seed and config to
            --steps (5,000), saving a labelled checkpoint into step{N}/ at
            each rung -- a study directory in the study's layout -- and
            anchor the retrain to the reference run: its per-step losses
            against the reference record's history.loss, exact.
  evaluate  per cell: the REFERENCE first (prepare_cell on --reference, the
            bitwise self-check against its diagnostic, the trust pass, the
            validation objective), then every rung: the study's own
            evaluation half (`evaluate_job`) writes step{N}/result_*.json,
            prepare_cell reads it back, the trust pass, the objective. One
            ladder_<arm>_seed<n>.json per cell holding all six rungs.
  read      pool the ladder records, decide Reading T, print the tables,
            write ladder.txt and ladder_curves.png (Task 6).

LOADING IS `trust_horizon.py`'S: `Cell`, `load_cell`, `self_check`,
`prepare_cell` and the checkpoint path are imported by path, and the trust
pass is `split_gap.py`'s `stratum_summary`. A rung's `Cell` carries the rung
record's own protocol in the diagnostic's slot -- a rung has no diagnostic,
and that is all `prepare_cell` reads there.

THE CHECKS, BY PHASE, each with its own status:

  train:    EXIT_NO_CHECKPOINTS (11)     a requested reference cell lacks its
                                          checkpoint, record or diagnostic;
                                          judged for every cell before any trains.
            EXIT_ANCHOR_MISMATCH (32)     NEW. --anchor hard and the retrain's
                                          losses are not the reference's.
  evaluate: EXIT_NO_CHECKPOINTS (11)     the reference cell, the train record
                                          or a rung checkpoint is missing.
            EXIT_ANCHOR_MISMATCH (32)     --anchor hard and the train record's
                                          anchor is not 0.0 (a `train` run under
                                          `report` cannot be read under `hard`).
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   the reference's trust pass is not
                                          bitwise its diagnostic -- judged BEFORE
                                          any rung of the cell is evaluated.
            EXIT_RUNG_MISLABELLED (33)    NEW. a rung checkpoint's arm, seed or
                                          step is not the rung's.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 32 and 33
are in no other tool's range (run_study 1/3-6/23, report_study 7-10, spike
10, diagnose 11-17, pool 18-22, trust 30, split_gap 31, argparse 2, a
traceback 1).
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.ladder import (
    CURVE_WINDOW,
    OBJECTIVE_BATCHES,
    REFERENCE_RUNG,
    RUNGS,
    STEPS,
    anchor_delta,
    primary_rung,
)
from mbfps.eval.objective import val_objective
from mbfps.eval.split_gap import learning_curve_summary, stratum_summary
from mbfps.eval.study import (
    SPLIT_SEED,
    StudyJob,
    evaluate_job,
    git_sha,
    history_record,
    job_record_path,
    load_record,
    write_record,
)
from mbfps.training.world_model import history_at, train_world_model
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_checkpoint_ladder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
probe_is_measurable = _trust.probe_is_measurable
checkpoint_path = _trust.checkpoint_path

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_ANCHOR_MISMATCH = 32
EXIT_RUNG_MISLABELLED = 33
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 32 and 33 are new and in no other tool's range."""

PHASES: tuple[str, ...] = ("train", "evaluate", "all")
ANCHOR_POLICIES: tuple[str, ...] = ("hard", "report")
ANCHOR_DEFAULT: str = "report"
"""The smoke run's measurement pins this (spec 2.4, Task 7); until then the
default is the policy that cannot stop a run."""


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------


def rung_dir(out_dir: Path, step: int) -> Path:
    """`step{N}/` under `--out`: a study directory, holding this rung's
    checkpoint and its study record under the study's own file names."""
    return Path(out_dir) / f"step{int(step)}"


def train_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"train_{arm}_seed{seed}.json"


def ladder_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"ladder_{arm}_seed{seed}.json"


# ---------------------------------------------------------------------------
# train: one cell retrained, its rungs saved, its anchor measured.
# ---------------------------------------------------------------------------


def train_record(arm: str, seed: int, history: dict, rungs, delta: float, first: int | None, *,
                 steps: int, policy: str, reference: Cell, device) -> dict:
    """The LIVE train record: the whole retrain history (so `evaluate` can
    slice it per rung without retraining), the elapsed seconds at each rung,
    the anchor, and provenance."""
    return {
        "arm": arm,
        "seed": int(seed),
        "steps": int(steps),
        "rungs": [int(r) for r in rungs],
        "seq_len": int(reference.record["seq_len"]),
        "history": history_record(history),
        "checkpoint_seconds": {str(k): float(v) for k, v in history["checkpoint_seconds"].items()},
        "seconds": float(history["seconds"]),
        "kl_dyn_max": float(history["kl_dyn_max"]),
        "kl_rate_above_free_bits": float(history["kl_rate_above_free_bits"]),
        "anchor": {
            "policy": policy,
            "steps": int(steps),
            "max_delta": float(delta),
            "first_step": None if first is None else int(first),
            "reference_git_sha": str(reference.record.get("git_sha", "unknown")),
        },
        "git_sha": git_sha(),
        "device": str(device),
        "torch_version": torch.__version__,
    }


def history_from_train_record(record: dict) -> dict:
    """The trainer's history dict, rebuilt from a train record, in the shape
    `history_at` reads (JSON turned the `checkpoint_seconds` keys into str)."""
    return {
        "arm": record["arm"],
        "steps": int(record["steps"]),
        "loss": [float(v) for v in record["history"]["loss"]],
        "parts": [dict(p) for p in record["history"]["parts"]],
        "seconds": float(record["seconds"]),
        "kl_dyn_max": float(record["kl_dyn_max"]),
        "kl_rate_above_free_bits": float(record["kl_rate_above_free_bits"]),
        "checkpoint_seconds": {int(k): float(v) for k, v in record["checkpoint_seconds"].items()},
    }


def anchor_failure(arm: str, seed: int, delta: float, first, steps: int) -> str:
    return (
        f"\nANCHOR MISMATCH for {arm} seed {seed}: the retrain's per-step loss is not the "
        f"reference record's (max abs {delta:.3e}, first at step {first} of {steps}). Under "
        "--anchor hard the rungs must be states the original run passed through, and they "
        "are not. No rung is evaluated."
    )


def train_cell(args, buffer, arm: str, seed: int, reference: Cell, device, rungs) -> tuple[int, dict | None]:
    """Retrain one cell from the study's seed and configuration -- `seq_len`
    the reference record's, `steps` the ladder's -- saving a rung at each of
    `rungs`, then anchor it. The train record is written whatever the anchor
    says: it is the evidence."""
    cfg = get_config(
        arm, steps=args.steps, seq_len=int(reference.record["seq_len"]), seed=seed, device=args.device,
    )
    history = train_world_model(cfg, buffer, out_dir=args.out, checkpoint_steps=tuple(rungs))
    delta, first = anchor_delta(history["loss"], reference.record["history"]["loss"], args.steps)
    record = train_record(
        arm, seed, history, rungs, delta, first,
        steps=args.steps, policy=args.anchor, reference=reference, device=device,
    )
    path = train_record_path(args.out, arm, seed)
    write_record(path, record)
    where = f" (first at step {first})" if first is not None else ""
    print(
        f"{arm} seed {seed}: trained {args.steps} steps in {history['seconds']:.0f}s, rungs "
        f"{list(rungs)}; anchor max|delta| {delta:.1e}{where}; wrote {path}"
    )
    if args.anchor == "hard" and delta != 0.0:
        print(anchor_failure(arm, seed, delta, first, args.steps))
        return EXIT_ANCHOR_MISMATCH, None
    return EXIT_OK, record


# ---------------------------------------------------------------------------
# evaluate: the reference first, then every rung.
# ---------------------------------------------------------------------------


def rung_checkpoint_label(path: Path) -> tuple:
    """`(arm, seed, step)` as the checkpoint labels itself."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("arm"), payload.get("seed"), payload.get("step")


def rung_cell(directory: Path, arm: str, seed: int) -> Cell:
    """A rung's `Cell`: its checkpoint and study record, and in the
    diagnostic's slot -- a rung has no diagnostic -- the record's own
    protocol, which is all `prepare_cell` reads there (`protocol_mismatch`).
    `self_check` is never run on a rung."""
    directory = Path(directory)
    checkpoint = checkpoint_path(directory, arm, seed)
    record_path = job_record_path(directory, StudyJob(arm=arm, seed=seed))
    for kind, path in (("checkpoint", checkpoint), ("study record", record_path)):
        if not path.exists():
            raise CellMissing(f"{arm} seed {seed}: no {kind} at {path}")
    record = load_record(record_path)
    return Cell(
        arm=arm, seed=seed, checkpoint=checkpoint, record=record,
        diagnostic={"context": int(record["context"]), "horizon": int(record["horizon"])},
    )


def _cell_args(args, out_dir: Path) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
    args; each rung is read with `out` pointed at its own directory."""
    return types.SimpleNamespace(
        out=Path(out_dir), device=args.device, context=args.context, horizon=args.horizon,
    )


def smoothed_at(parts, term: str, step: int, window: int) -> float:
    """The `window`-step moving mean of `term` ending at 1-based `step` (the
    window shrinks to the prefix) -- the training-side number printed beside
    the validation objective at that rung."""
    values = np.asarray([float(p[term]) for p in list(parts)[:int(step)]], dtype=float)
    if values.size == 0:
        raise ValueError(f"no training steps at or before step {step}")
    w = min(int(window), values.size)
    return float(values[-w:].mean())


def rung_entry(step: int, record: dict, traj, prepared, objective: dict, train_parts, window: int,
               check=None) -> dict:
    """One rung of one cell: the gate's own metric off the study record, the
    probe's measurability, the validation objective, the smoothed training
    embedding at that step, the reference's self-check (None on a rung), and
    the trust pass reduced by `stratum_summary`."""
    curves = record["curves"]
    r2 = record["probe"]["embedding_selection_r2"]
    return {
        "step": int(step),
        "record_git_sha": str(record.get("git_sha", "unknown")),
        "gate": {
            "gap_final": float(record["position"]["gap_final"]),
            "steps_degenerate": int(record["position"]["steps_degenerate"]),
        },
        "probe": {
            "selection_r2": float(r2) if r2 is not None else float("nan"),
            "measurable": bool(probe_is_measurable(
                {"persistence": curves["persistence_position"][-1], "floor": curves["floor_position"][-1]},
                widest_se=0.0,
            )),
        },
        "objective": {name: float(value) for name, value in objective.items()},
        "train_embedding": smoothed_at(train_parts, "embedding", step, window),
        "self_check": None if check is None else check.record(),
        "summary": stratum_summary(traj, prepared.horizon),
    }


def evaluate_reference(args, reference: Cell, device, train, val, buffer) -> tuple[int, dict | None, tuple]:
    """Rung 20000: `prepare_cell` on `--reference` (12, 14), the trust pass,
    the bitwise self-check against the diagnostic (30), the objective.
    Returns the entry and the resolved `(context, horizon)` every rung is
    evaluated at."""
    status, prepared = prepare_cell(_cell_args(args, args.reference), reference, device, train, val)
    if status != EXIT_OK:
        return status, None, ()
    traj = reference_trajectories(prepared.model, val, prepared.embedding_probe, **prepared.common)
    check = self_check(traj, reference.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {reference.arm} seed {reference.seed} at the reference "
            f"(step {REFERENCE_RUNG}): " + "; ".join(check.failures())
            + ". The ruler M3d validated does not reproduce, so no rung is evaluated against it."
        )
        return EXIT_SELF_CHECK_FAILED, None, ()
    cfg = get_config(
        reference.arm, seq_len=int(reference.record["seq_len"]), seed=reference.seed, device=args.device,
    )
    objective = val_objective(
        prepared.model, buffer, val, cfg, device, batches=args.objective_batches, seed=reference.seed,
    )
    entry = rung_entry(
        REFERENCE_RUNG, reference.record, traj, prepared, objective,
        reference.record["history"]["parts"], args.window, check=check,
    )
    return EXIT_OK, entry, (prepared.context, prepared.horizon)


def evaluate_rung(args, buffer, arm: str, seed: int, step: int, history: dict, device, train, val, *,
                  context: int, horizon: int, seq_len: int) -> tuple[int, dict | None]:
    """One rung: the checkpoint's label (11, 33), `evaluate_job` into the rung
    directory, `prepare_cell` on it (12, 14 -- proving the two rollouts
    agree), the trust pass, the objective."""
    directory = rung_dir(args.out, step)
    checkpoint = checkpoint_path(directory, arm, seed)
    if not checkpoint.exists():
        print(f"NO CELL: {arm} seed {seed}: no checkpoint at {checkpoint}")
        return EXIT_NO_CHECKPOINTS, None
    label = rung_checkpoint_label(checkpoint)
    if label != (arm, int(seed), int(step)):
        print(
            f"\nRUNG MISLABELLED: {checkpoint} is arm={label[0]!r} seed={label[1]!r} "
            f"step={label[2]!r}, not this rung's arm={arm!r} seed={seed!r} step={step!r}; it "
            "would be reported under a step it was never trained to."
        )
        return EXIT_RUNG_MISLABELLED, None
    evaluate_job(
        StudyJob(arm=arm, seed=seed), buffer, directory, history=history_at(history, step),
        steps=int(step), seq_len=seq_len, context=context, horizon=horizon, device=args.device,
    )
    cell = rung_cell(directory, arm, seed)
    status, prepared = prepare_cell(_cell_args(args, directory), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    traj = reference_trajectories(prepared.model, val, prepared.embedding_probe, **prepared.common)
    cfg = get_config(arm, seq_len=seq_len, seed=seed, device=args.device)
    objective = val_objective(
        prepared.model, buffer, val, cfg, device, batches=args.objective_batches, seed=seed,
    )
    return EXIT_OK, rung_entry(step, cell.record, traj, prepared, objective, history["parts"], args.window)


def ladder_record(arm: str, seed: int, entries: dict, *, primary: int, min_step: int, train_rec: dict,
                  args, device, val, rungs, context: int, horizon: int) -> dict:
    """The LIVE ladder record for one cell: every rung's entry under its step,
    the primary rung and the minimum it was read from, and provenance."""
    return {
        "arm": arm,
        "seed": int(seed),
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "reference_git_sha": entries[REFERENCE_RUNG]["record_git_sha"],
        "train_git_sha": str(train_rec["git_sha"]),
        "anchor": dict(train_rec["anchor"]),
        "rungs": [int(r) for r in rungs],
        "reference_rung": REFERENCE_RUNG,
        "primary_rung": int(primary),
        "embedding_min_step": int(min_step),
        "curve_window": int(args.window),
        "objective_batches": int(args.objective_batches),
        "episodes": {"val": [p.name for p in val]},
        "entries": {str(step): entries[step] for step in sorted(entries)},
    }


def evaluate_cell(args, buffer, arm: str, seed: int, device, train, val, rungs) -> tuple[int, dict | None]:
    """One cell: 11 (reference, train record), 32 under `hard`, the reference
    (12, 14, 30) BEFORE any rung, then each rung (11, 33, 12, 14), then the
    primary rung off the reference's history, and the ladder record."""
    try:
        reference = load_cell(args.reference, arm, seed)
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS, None
    train_path = train_record_path(args.out, arm, seed)
    if not train_path.exists():
        print(f"NO CELL: {arm} seed {seed}: no train record at {train_path}; run --phase train first")
        return EXIT_NO_CHECKPOINTS, None
    train_rec = load_record(train_path)
    anchor = train_rec["anchor"]
    if args.anchor == "hard" and float(anchor["max_delta"]) != 0.0:
        print(anchor_failure(arm, seed, float(anchor["max_delta"]), anchor["first_step"], int(anchor["steps"])))
        return EXIT_ANCHOR_MISMATCH, None

    # The reference FIRST: its self-check is the ruler every rung is read against.
    status, ref_entry, protocol = evaluate_reference(args, reference, device, train, val, buffer)
    if status != EXIT_OK:
        return status, None
    context, horizon = protocol
    seq_len = int(reference.record["seq_len"])
    history = history_from_train_record(train_rec)
    entries = {REFERENCE_RUNG: ref_entry}
    for step in rungs:
        status, entry = evaluate_rung(
            args, buffer, arm, seed, int(step), history, device, train, val,
            context=context, horizon=horizon, seq_len=seq_len,
        )
        if status != EXIT_OK:
            return status, None
        entries[int(step)] = entry

    min_step = learning_curve_summary(reference.record["history"], window=args.window)["terms"]["embedding"]["smoothed_min_step"]
    primary = primary_rung(min_step, rungs)
    record = ladder_record(
        arm, seed, entries, primary=primary, min_step=min_step, train_rec=train_rec, args=args,
        device=device, val=val, rungs=rungs, context=context, horizon=horizon,
    )
    path = ladder_record_path(args.out, arm, seed)
    write_record(path, record)
    check = ref_entry["self_check"]
    print(
        f"{arm} seed {seed}: reference self-check max|delta| reference "
        f"{check['reference_position_max_delta']:.1e} persistence "
        f"{check['persistence_position_max_delta']:.1e}; {len(rungs)} rungs evaluated; "
        f"embedding minimum at step {min_step} -> primary rung {primary}; wrote {path}"
    )
    return EXIT_OK, record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3f_ladder"))
    parser.add_argument("--reference", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: rung 20000, and the anchor's history")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the reference diagnostic says what it was written at; a value
    # that disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--steps", type=int, default=STEPS, help="the retrain's length")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--anchor", choices=ANCHOR_POLICIES, default=ANCHOR_DEFAULT,
                        help="hard: refuse a retrain whose losses are not the reference's (32); "
                             "report: print the delta and continue")
    parser.add_argument("--objective-batches", type=int, default=OBJECTIVE_BATCHES,
                        help="validation draws per rung for the objective")
    parser.add_argument("--window", type=int, default=CURVE_WINDOW,
                        help="moving-mean window (steps) the embedding minimum is read at")
    parser.add_argument("--figure", type=Path, default=None,
                        help="the curves figure; default <out>/ladder_curves.png")
    return parser


def main(argv: list[str] | None = None, *, rungs=RUNGS) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    rungs = tuple(sorted({int(r) for r in rungs}))
    if args.steps < max(rungs):
        parser.error(f"--steps {args.steps} is below the last rung {max(rungs)}")
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)

    references: dict[tuple[str, int], Cell] = {}
    if args.phase in ("train", "all"):
        # Every requested reference cell is loaded before anything trains.
        try:
            references = {(arm, seed): load_cell(args.reference, arm, seed) for arm, seed in cells}
        except CellMissing as error:
            print(f"NO CELL: {error}")
            return EXIT_NO_CHECKPOINTS
        args.out.mkdir(parents=True, exist_ok=True)

    for arm, seed in cells:
        if args.phase in ("train", "all"):
            status, _ = train_cell(args, buffer, arm, seed, references[(arm, seed)], device, rungs)
            if status != EXIT_OK:
                return status
        if args.phase in ("evaluate", "all"):
            status, _ = evaluate_cell(args, buffer, arm, seed, device, train, val, rungs)
            if status != EXIT_OK:
                return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
