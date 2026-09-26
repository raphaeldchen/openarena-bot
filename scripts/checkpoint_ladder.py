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
            self-check against its diagnostic within the reproduction bound
            of spec 2.4 -- 64 ULPs of the stored value -- the trust pass, the
            validation objective), then every rung: the study's own
            evaluation half (`evaluate_job`) writes step{N}/result_*.json,
            prepare_cell reads it back, the trust pass, the objective. One
            ladder_<arm>_seed<n>.json per cell holding all six rungs.
  read      pool the ladder records, decide Reading T, print the tables,
            write ladder.txt and ladder_curves.png.

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
            EXIT_SELF_CHECK_FAILED (30)   the reference's trust pass does not
                                          reproduce its diagnostic within the
                                          bound of spec 2.4 -- judged BEFORE
                                          any rung of the cell is evaluated.
            EXIT_RUNG_MISLABELLED (33)    NEW. a rung checkpoint's arm, seed or
                                          step is not the rung's.
  read:     EXIT_NO_CHECKPOINTS (11)     a requested ladder record is missing.
            EXIT_SELF_CHECK_FAILED (30)   a ladder record's recorded reference
                                          self-check is not ok.

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
import mbfps.eval.pooling as pooling
from mbfps.eval.ladder import (
    CURVE_WINDOW,
    DECISION_H,
    FAMILY,
    OBJECTIVE_BATCHES,
    R2_SENSITIVITY,
    REFERENCE_RUNG,
    REPORTED_H,
    RUNGS,
    STEPS,
    ArmInputs,
    TimingInputs,
    anchor_delta,
    format_reading_timing,
    gate_passes,
    primary_rung,
    reading_timing,
)
from mbfps.eval.objective import val_objective
from mbfps.eval.split_gap import (
    CHANNELS,
    TERMS,
    StratumContrast,
    cell_series,
    decision_horizon,
    learning_curve_summary,
    margin_at,
    q_key,
    stratum_summary,
    survival_indicator,
)
from mbfps.eval.trust import survival, trust_horizon
from mbfps.eval.trust_readings import ARMS_ORDER, Q_REPORTED
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
trustworthy = _trust.trustworthy

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_ANCHOR_MISMATCH = 32
EXIT_RUNG_MISLABELLED = 33
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 32 and 33 are new and in no other tool's range."""

_split = _sibling("split_gap")
SURVIVAL_STEPS = _split.SURVIVAL_STEPS
"""The survival table's columns, M3d's and M3e's, filtered to <= the run's horizon."""

PHASES: tuple[str, ...] = ("train", "evaluate", "read", "all")
ANCHOR_POLICIES: tuple[str, ...] = ("hard", "report")
ANCHOR_DEFAULT: str = "hard"
"""Pinned by the smoke run (spec 2.4): random_vit/s0 retrained 5,000 steps on
mps under torch 2.13.0 reproduced the M3c record's per-step loss with max
|delta| exactly 0.0 at every step, so a retrain that does not is refused."""


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
    the self-check against the diagnostic within the reproduction bound (30),
    the objective.
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
# read: pooling glue -- ladder records -> the inputs Reading T is decided on.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s. What is decided here is only WHICH
# series goes in under WHICH mask (spec 3.1): the survival indicator over the
# windows that moved within the horizon; the margin over the windows moved AT
# h. The rung contrast is the PAIRED one -- both rungs score the same val
# windows -- with the two rung groups labelled as two arms so
# `paired_contrast` reads them as such.


def _entry(record: dict, step: int) -> dict:
    return record["entries"][str(int(step))]


def _series(record: dict, step: int, channel: str, values, changed, label: str) -> pooling.CellSeries:
    """One per-window series of one cell at one rung, through
    `split_gap.cell_series`. `arm` is the GROUP label (`<arm>@primary` /
    `<arm>@20000`): the two rung groups of one arm are two "arms" on the same
    windows to `paired_contrast`, which refuses an arm against itself. `rung`
    is the stratum (val); `channel` the reading."""
    return cell_series(
        _entry(record, step)["summary"], values, changed,
        arm=label, seed=record["seed"], rung="val", channel=channel,
        val=record["episodes"]["val"], horizon=record["horizon"], context=record["context"],
        device=record["device"], torch_version=record["torch_version"],
    )


def survival_series(record: dict, step: int, channel: str, h: int, label: str) -> pooling.CellSeries:
    """`split_gap.survival_indicator` on this rung's summary: `1[h_x > h]`
    over the windows that moved within the horizon."""
    return _series(record, step, f"S/{channel}", *survival_indicator(_entry(record, step)["summary"], channel, h), label)


def margin_series(record: dict, step: int, channel: str, h: int, label: str) -> pooling.CellSeries:
    """`split_gap.margin_at` on this rung's summary: `Delta(h)` over the
    windows moved AT h."""
    return _series(record, step, f"margin/{channel}", *margin_at(_entry(record, step)["summary"], channel, h), label)


_NO_CONTRAST = StratumContrast(estimate=float("nan"), se=float("nan"), z=float("nan"), clusters=0)


def _paired(treatment, control) -> StratumContrast:
    """`pooling.paired_contrast`, or the NaN contrast when there is nothing to
    pair -- no cell, or no window every cell of both groups changed."""
    if not treatment or not control:
        return _NO_CONTRAST
    if not np.logical_and.reduce([c.changed for c in list(treatment) + list(control)]).any():
        return _NO_CONTRAST
    c = pooling.paired_contrast(treatment, control)
    return StratumContrast(estimate=c.mean, se=c.se, z=c.z, clusters=c.clusters)


def _probe_ok(record: dict, step: int, min_r2: float | None) -> bool:
    entry = _entry(record, step)
    if not entry["probe"]["measurable"]:
        return False
    if min_r2 is None:
        return True
    r2 = float(entry["probe"]["selection_r2"])
    return bool(np.isfinite(r2) and r2 >= min_r2)


def _included(record: dict, channel: str, min_r2: float | None) -> bool:
    """Spec 3.1: the free channel pools every cell; the probe channel the
    cells measurable (and, on the sensitivity line, at or above `min_r2`) at
    BOTH rungs of the pair."""
    if channel == "free":
        return True
    return _probe_ok(record, record["primary_rung"], min_r2) and _probe_ok(record, REFERENCE_RUNG, min_r2)


def arm_inputs(records: dict, arm: str, seeds, *, h: int, series=survival_series,
               min_r2: float | None = None) -> ArmInputs:
    """One arm's `ArmInputs` at step `h`: each seed's primary rung paired
    against its reference, pooled over `seeds`, and the same within each seed
    alone."""
    def contrast(channel, seed_list):
        kept = [records[(arm, s)] for s in seed_list if _included(records[(arm, s)], channel, min_r2)]
        treatment = [series(r, int(r["primary_rung"]), channel, h, f"{arm}@primary") for r in kept]
        control = [series(r, REFERENCE_RUNG, channel, h, f"{arm}@{REFERENCE_RUNG}") for r in kept]
        return _paired(treatment, control)

    primary = {int(s): int(records[(arm, s)]["primary_rung"]) for s in seeds}
    per_seed = {
        int(s): ArmInputs(
            t_free=contrast("free", [s]), t_probe=contrast("probe", [s]),
            primary_rungs={int(s): primary[int(s)]}, per_seed=None,
        )
        for s in seeds
    }
    return ArmInputs(
        t_free=contrast("free", list(seeds)), t_probe=contrast("probe", list(seeds)),
        primary_rungs=primary, per_seed=per_seed,
    )


def _clusters(records: dict) -> int:
    return int(_entry(next(iter(records.values())), REFERENCE_RUNG)["summary"]["windows"]["clusters"])


def timing_inputs(records: dict, *, arms, seeds, h: int, series=survival_series,
                  min_r2: float | None = None) -> TimingInputs:
    """Every arm's inputs at `h`, and `z_fam` over the val stratum's cluster
    count (spec 3.1)."""
    ordered = sorted(arms, key=lambda a: ARMS_ORDER.index(a) if a in ARMS_ORDER else len(ARMS_ORDER))
    return TimingInputs(
        arms={arm: arm_inputs(records, arm, seeds, h=h, series=series, min_r2=min_r2) for arm in ordered},
        z_fam=pooling.cluster_threshold(FAMILY, _clusters(records)),
        h=h,
    )


# `decision_horizon` is `split_gap`'s, imported: one decision horizon (M3e's
# DECISION_H) for both readings.


# ---------------------------------------------------------------------------
# The printed tables.
# ---------------------------------------------------------------------------


def _num(value, spec: str = ".3f") -> str:
    value = float(value)
    return format(value, spec) if np.isfinite(value) else "n/a"


def _cells_in_order(records: dict) -> list[tuple[str, int]]:
    return sorted(records, key=lambda k: (ARMS_ORDER.index(k[0]) if k[0] in ARMS_ORDER else 9, k[1]))


def _steps_in_order(rungs) -> list[int]:
    return [*sorted(int(r) for r in rungs), REFERENCE_RUNG]


def _anchor_table(records: dict) -> str:
    lines = [
        "--- anchor: the retrain's per-step loss against the reference run's (spec 2.4; exact equality) ---",
        f"  {'arm':<12}{'seed':>5}{'steps':>7}{'policy':>8}{'max|delta|':>12}{'first_step':>11}  train git_sha",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        a = r["anchor"]
        first = "-" if a["first_step"] is None else str(a["first_step"])
        lines.append(
            f"  {arm:<12}{seed:>5}{a['steps']:>7}{a['policy']:>8}{float(a['max_delta']):>12.1e}"
            f"{first:>11}  {r['train_git_sha'][:12]}"
        )
    return "\n".join(lines) + "\n"


def _primary_table(records: dict, rungs) -> str:
    first = next(iter(records.values()))
    lines = [
        f"--- primary rung per cell: the rung nearest the reference run's smoothed embedding minimum "
        f"({first['curve_window']}-step window; ties to the earlier; rungs {list(rungs)}) ---",
        f"  {'arm':<12}{'seed':>5}{'min_step':>10}{'primary':>9}",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        lines.append(f"  {arm:<12}{seed:>5}{r['embedding_min_step']:>10}{r['primary_rung']:>9}")
    return "\n".join(lines) + "\n"


def _self_check_table(records: dict) -> str:
    lines = [
        f"--- reference self-check (step {REFERENCE_RUNG}'s trust pass against its diagnostic; exact) ---",
        f"  {'arm':<12}{'seed':>5}{'ref max|d|':>12}{'pers max|d|':>13}{'windows':>9}{'probe R2':>10}{'measurable':>12}  ok",
    ]
    for arm, seed in _cells_in_order(records):
        e = _entry(records[(arm, seed)], REFERENCE_RUNG)
        c = e["self_check"]
        lines.append(
            f"  {arm:<12}{seed:>5}{c['reference_position_max_delta']:>12.1e}"
            f"{c['persistence_position_max_delta']:>13.1e}"
            f"{str(c['windows_total_match'] and c['windows_episode_match']):>9}"
            f"{_num(e['probe']['selection_r2']):>10}{str(e['probe']['measurable']):>12}  {c['ok']}"
        )
    return "\n".join(lines) + "\n"


def _gate_table(records: dict, arms, seeds, rungs, horizon: int) -> str:
    lines = [
        f"--- the gate at every rung: gap_closed({horizon}) on position per seed (spec 4.1's metric; "
        "NaN = non-positive band); GATE PASSES = > 0 in every seed. Reported, not decided on. ---",
        f"  {'arm':<12}{'step':>7}" + "".join(f"{f's{s}':>10}" for s in seeds)
        + f"{'nanmean':>10}{'degen(max)':>11}  gate",
    ]
    for arm in arms:
        for step in _steps_in_order(rungs):
            finals = {int(s): float(_entry(records[(arm, s)], step)["gate"]["gap_final"]) for s in seeds}
            degenerate = max(int(_entry(records[(arm, s)], step)["gate"]["steps_degenerate"]) for s in seeds)
            values = list(finals.values())
            mean = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            flag = f"GATE PASSES at step {step}" if gate_passes(finals) else ""
            lines.append(
                f"  {arm:<12}{step:>7}" + "".join(f"{_num(finals[int(s)], '+.4f'):>10}" for s in seeds)
                + f"{_num(mean, '+.4f'):>10}{degenerate:>11}  {flag}"
            )
    return "\n".join(lines) + "\n"


def _stacked(records: dict, arm: str, step: int, channel: str, seeds, key: str) -> np.ndarray:
    """One per-window array of `key` -- `crossing` (per channel) or
    `first_moved` -- concatenated over the seeds included in `channel` at this
    rung: every cell in the free channel, the measurable cells in the probe
    one."""
    rows = []
    for s in seeds:
        if channel != "free" and not _probe_ok(records[(arm, s)], step, None):
            continue
        block = _entry(records[(arm, s)], step)["summary"]
        rows.append(np.asarray(block["crossing"][channel] if key == "crossing" else block[key], dtype=float))
    return np.concatenate(rows) if rows else np.zeros(0)


def _survival_table(records: dict, arms, seeds, rungs, horizon: int) -> str:
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- survival per rung: S(h) = fraction of moved draws with h_x > h "
        "(a window not yet moved at h survives vacuously, as in M3d); seeds stacked ---",
        f"  {'arm':<12}{'step':>7} {'channel':<7}" + "".join(f"{f'S({h})':>7}" for h in steps)
        + "".join(f"{'H*' + q_key(q)[1:]:>8}" for q in Q_REPORTED),
    ]
    for arm in arms:
        for step in _steps_in_order(rungs):
            for channel in CHANNELS:
                crossings = _stacked(records, arm, step, channel, seeds, "crossing")
                s = survival(crossings, horizon) if crossings.size else np.full(horizon + 1, np.nan)
                lines.append(
                    f"  {arm:<12}{step:>7} {channel:<7}" + "".join(f"{_num(s[h], '.2f'):>7}" for h in steps)
                    + "".join(f"{trust_horizon(s, q):>8d}" if crossings.size else f"{'n/a':>8}" for q in Q_REPORTED)
                )
    return "\n".join(lines) + "\n"


def _conditional_table(records: dict, arms, seeds, rungs, horizon: int) -> str:
    """M3d's `u(h)` and `S_c(h)` beside S(h), through `trust_horizon.py`'s own
    functions imported by path; printed, never decided on."""
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- conditional survival per rung: u(h) = unmoved fraction, S_c(h) = (S(h) - u(h)) / (1 - u(h)) "
        "(M3d's series; printed, not decided on) ---",
        f"  {'arm':<12}{'step':>7} {'channel':<7}" + "".join(f"{f'u({h})':>7}" for h in steps)
        + "".join(f"{f'Sc({h})':>8}" for h in steps),
    ]
    for arm in arms:
        for step in _steps_in_order(rungs):
            for channel in CHANNELS:
                crossings = _stacked(records, arm, step, channel, seeds, "crossing")
                first = _stacked(records, arm, step, channel, seeds, "first_moved")
                if not crossings.size:
                    lines.append(f"  {arm:<12}{step:>7} {channel:<7}" + "".join(f"{'n/a':>7}" for _ in steps)
                                 + "".join(f"{'n/a':>8}" for _ in steps))
                    continue
                s = (survival(crossings, horizon) if np.isfinite(crossings).any()
                     else np.full(horizon + 1, np.nan))
                u = _trust.unmoved_fraction(crossings, first, horizon)
                sc = _trust.conditional_survival(s, u)
                lines.append(
                    f"  {arm:<12}{step:>7} {channel:<7}"
                    + "".join(f"{_num(u[h], '.2f'):>7}" for h in steps)
                    + "".join(f"{_num(sc[h], '.2f'):>8}" for h in steps)
                )
    return "\n".join(lines) + "\n"


def _objective_table(records: dict, rungs) -> str:
    first = next(iter(records.values()))
    lines = [
        f"--- the validation objective per rung (val_objective over {first['objective_batches']} draws; "
        "descriptive, no verdict) beside the smoothed training embedding at that step ---",
        f"  {'arm':<12}{'seed':>5}{'step':>7}{'val loss':>10}{'val emb':>9}{'train emb':>10}"
        + "".join(f"{term:>10}" for term in TERMS if term != "embedding"),
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        best_emb = best_loss = None
        for step in _steps_in_order(rungs):
            e = _entry(r, step)
            o = e["objective"]
            lines.append(
                f"  {arm:<12}{seed:>5}{step:>7}{_num(o['loss'], '.4f'):>10}{_num(o['embedding'], '.4f'):>9}"
                f"{_num(e['train_embedding'], '.4f'):>10}"
                + "".join(f"{_num(o[term], '.4f'):>10}" for term in TERMS if term != "embedding")
            )
            if np.isfinite(o["embedding"]) and (best_emb is None or o["embedding"] < best_emb[1]):
                best_emb = (step, o["embedding"])
            if np.isfinite(o["loss"]) and (best_loss is None or o["loss"] < best_loss[1]):
                best_loss = (step, o["loss"])
        smallest_at = lambda best: "n/a" if best is None else str(best[0])
        lines.append(
            f"  {arm:<12}{seed:>5}  val embedding smallest at step {smallest_at(best_emb)}; val loss smallest at "
            f"step {smallest_at(best_loss)}; primary rung {r['primary_rung']}"
        )
    return "\n".join(lines) + "\n"


def _reported_table(records: dict, arms, seeds, horizon: int) -> str:
    steps = sorted({h for h in REPORTED_H if h <= horizon} | {horizon})
    lines = [
        f"--- T at the reported horizons (primary - {REFERENCE_RUNG}; z only; not decided on) ---",
        f"  {'arm':<12}{'channel':<8}" + "".join(f"{f'z@{h}':>9}" for h in steps) + f"{'dFree@dec':>11}",
    ]
    dec, _ = decision_horizon(horizon)
    for arm in arms:
        by_h = {h: arm_inputs(records, arm, seeds, h=h) for h in steps}
        margin = arm_inputs(records, arm, seeds, h=dec, series=margin_series).t_free
        for channel in CHANNELS:
            zs = [getattr(by_h[h], f"t_{channel}").z for h in steps]
            extra = (f"{_num(margin.estimate, '+.4f')} (z {_num(margin.z, '+.2f')})"
                     if channel == "free" else "")
            lines.append(
                f"  {arm:<12}{channel:<8}" + "".join(f"{_num(z, '+.2f'):>9}" for z in zs) + f"  {extra}"
            )
    lines.append("  dFree@dec: the paired rung difference of the embedding-space margin Delta_free "
                 "at the decision horizon -- the continuous companion, in the loss's units.")
    return "\n".join(lines) + "\n"


def _sensitivity_text(records: dict, arms, seeds, h: int, z_fam: float) -> str:
    dropped = [
        f"{a} s{s}" for a, s in _cells_in_order(records)
        if any(not _probe_ok(records[(a, s)], step, R2_SENSITIVITY)
               and _probe_ok(records[(a, s)], step, None)
               for step in (records[(a, s)]["primary_rung"], REFERENCE_RUNG))
    ]
    inputs = timing_inputs(records, arms=arms, seeds=seeds, h=h, min_r2=R2_SENSITIVITY)
    lines = [
        f"--- sensitivity (changes no verdict): probe channel with selection R2 < {R2_SENSITIVITY} at "
        f"either rung excluded -- dropped: {', '.join(dropped) if dropped else 'none'} ---",
    ]
    for arm, a in inputs.arms.items():
        lines.append(
            f"  {arm:<12}T_probe estimate {_num(a.t_probe.estimate, '+.4f')} se "
            f"{_num(a.t_probe.se, '.4f')} z {_num(a.t_probe.z, '+.2f')} "
            f"(bar {z_fam:.2f}; clusters {a.t_probe.clusters})"
        )
    return "\n".join(lines) + "\n"


def _per_seed_text(inputs: TimingInputs) -> str:
    lines = ["--- per seed (the same contrast within one seed alone; no seed averaging) ---"]
    for arm, a in inputs.arms.items():
        for seed, leaf in sorted((a.per_seed or {}).items()):
            leaf_reading = reading_timing(TimingInputs(arms={arm: leaf}, z_fam=inputs.z_fam, h=inputs.h)).arms[arm]
            lines.append(
                f"  {arm:<12}s{seed}: primary rung {leaf.primary_rungs[seed]}; T_free z "
                f"{_num(leaf.t_free.z, '+.2f')}, T_probe z {_num(leaf.t_probe.z, '+.2f')} -> "
                f"{leaf_reading.status.name.replace('_', ' ')}"
            )
    return "\n".join(lines) + "\n"


def write_curves(records: dict, figure: Path, rungs) -> str:
    """Three panels against the rung (the reference at the right): H*_0.75 on
    the free channel, gap_closed at the horizon, and the validation
    embedding; arms coloured, seeds as thin lines. A missing or broken
    matplotlib, or an unwritable path, costs the FIGURE and nothing else
    (report_study's guard)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    colours = {"pixel_ae": "tab:orange", "frozen_ssl": "tab:blue", "random_vit": "tab:gray"}
    steps = _steps_in_order(rungs)
    x = np.arange(len(steps))
    panels = (
        ("H*_0.75, free channel", lambda e: e["summary"]["trust_horizon"]["free"][q_key(0.75)]),
        ("gap_closed at the horizon (position)", lambda e: e["gate"]["gap_final"]),
        ("validation embedding loss", lambda e: e["objective"]["embedding"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    try:
        for ax, (title, pick) in zip(axes.flat, panels):
            for (arm, seed), record in records.items():
                y = [float(pick(_entry(record, step))) for step in steps]
                ax.plot(x, y, marker="o", markersize=3, color=colours.get(arm, "black"),
                        linewidth=0.9, alpha=0.85, label=f"{arm} s{seed}")
            if title.startswith("gap_closed"):
                ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in steps], rotation=45)
            ax.set_xlabel("training step (rung)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def readings_text(records: dict, *, arms, seeds, rungs, horizon: int, figure_line: str) -> str:
    """Everything `read` prints, in `split_gap.txt`'s style; written to
    `ladder.txt` byte-identical."""
    h, clamped = decision_horizon(horizon)
    inputs = timing_inputs(records, arms=arms, seeds=seeds, h=h)
    reading = reading_timing(inputs)
    note = (f"  NOTE: decision horizon clamped to the run's horizon h={h} "
            f"(pre-registered DECISION_H = {DECISION_H}).\n" if clamped else "")
    return "".join([
        _anchor_table(records),
        _primary_table(records, rungs),
        _self_check_table(records),
        _gate_table(records, arms, seeds, rungs, horizon),
        _survival_table(records, arms, seeds, rungs, horizon),
        _conditional_table(records, arms, seeds, rungs, horizon),
        _objective_table(records, rungs),
        note,
        f"  pooling: z_fam = cluster_threshold({FAMILY}, {_clusters(records)}) = {inputs.z_fam:.2f}; "
        "each seed's primary rung is paired against its reference on the same val windows; the "
        "survival indicator pools windows that moved within the horizon; the probe channel pools "
        "cells measurable at both rungs.\n",
        format_reading_timing(reading, inputs),
        _reported_table(records, arms, seeds, horizon),
        _sensitivity_text(records, arms, seeds, h, inputs.z_fam),
        _per_seed_text(inputs),
        f"  {figure_line}\n",
    ])


def write_readings(out_dir: Path, text: str) -> Path:
    path = Path(out_dir) / "ladder.txt"
    path.write_text(text)
    return path


def read_phase(args, cells, rungs) -> int:
    """Load every requested ladder record (11), refuse one whose reference
    self-check is not ok (30), then the figure, the text, `ladder.txt`."""
    records: dict[tuple[str, int], dict] = {}
    for arm, seed in cells:
        path = ladder_record_path(args.out, arm, seed)
        if not path.exists():
            print(f"NO CELL: {arm} seed {seed}: no ladder record at {path}; run --phase evaluate first")
            return EXIT_NO_CHECKPOINTS
        records[(arm, seed)] = load_record(path)
    for (arm, seed), record in records.items():
        check = _entry(record, REFERENCE_RUNG)["self_check"]
        if not trustworthy(check):
            print(
                f"\nSELF-CHECK FAILED for {arm} seed {seed}: the ladder record's reference "
                f"self-check is {check!r}; the record is not read against a ruler that reproduced."
            )
            return EXIT_SELF_CHECK_FAILED
        if record["rungs"] != [int(r) for r in rungs]:
            raise ValueError(
                f"{arm} seed {seed}: the ladder record holds rungs {record['rungs']}, not the "
                f"requested {list(rungs)}; a different ladder is a different --out"
            )
    horizon = int(next(iter(records.values()))["horizon"])
    figure = args.figure if args.figure is not None else args.out / "ladder_curves.png"
    figure_line = write_curves(records, figure, rungs)
    text = readings_text(
        records, arms=list(args.arms), seeds=[int(s) for s in args.seeds],
        rungs=rungs, horizon=horizon, figure_line=figure_line,
    )
    print(text, end="")
    write_readings(args.out, text)
    return EXIT_OK


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
    if args.out.resolve() == args.reference.resolve():
        parser.error(
            "--out and --reference are the same directory; the train phase would overwrite the "
            "reference study's checkpoints"
        )
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]

    if args.phase in ("train", "evaluate", "all"):
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
    if args.phase in ("read", "all"):
        return read_phase(args, cells, rungs)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
