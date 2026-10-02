"""M3m: does the one-step map predict motion, or did it never learn it? -- measure and read.

M3l refuted the bottleneck lever and left the objective lever standing by
elimination, but the hypothesis it inherited -- "the loss never asks for motion"
-- is false as stated: the dynamics prior trained on 75-98% of steps. What
survives is narrower. The loss asks for ONE-step latent agreement; the gate reads
45-step rollout error. So M3m asks whether the 45-step failure is error that
COMPOUNDS over a rollout, or a one-step map that never learned motion at all, by
reading the re-grounding ladder against a baseline that is GROUND TRUTH.
`mbfps.eval.burden` is the pure half -- the decomposition, the baseline, the
interval, Reading H and its table -- and owns no torch, device or record schema.
This script is the impure half.

TWO AXES, NEVER CONFLATED. `k` is the RE-GROUNDING PERIOD, over
`diagnostics.REGROUNDING_KS`; `h` is the HORIZON STEP.

PER CELL, THE RECORD CARRIES:

  curves          the canonical position curves of the reference `prepare_cell`
                  VERIFIED against the study record, and every rung of the
                  ladder (`regrounding_sweep`), plus the baseline's mean.
  burden_by_k     `rung(k) - floor`, per horizon step, for every k.
  compounding_by_k `rung(k) - rung(1)`, per horizon step, for every k.
  margin          `(point, ci_low, ci_high)` of the mean `motion_margin` at every
                  `REPORTED_H`, episode-clustered, drawn at the cell's own seed.
  window_margin   the per-window margins the intervals are drawn from, so a
                  reader can redraw them.
  controls        four numbers with known answers, MEASURED and never asserted:
                  the identity residual, the k=horizon rung's divergence from
                  that verified reference (0.0), whether k=1 collapsed onto the floor
                  (False), and the k=1 rung's compounding (0.0).
  base_control    the median true one-step displacement against the median floor
                  error at `DECISION_H`: if the agent barely moved, no method
                  could detect motion prediction in that cell.
  protocol        the confidence level AND the resample count, from `burden`'s
                  constants. M3l shipped a headline interval whose draw count the
                  permanent artefact could not be audited for.

NOTHING IN THE MEASURE PHASE DECIDES ANYTHING. Reading H is the read phase's, from
the pooled records. A control that missed its known answer is RECORDED, not
raised, and the run carries on: the records are the artefact, and a measure that
raised would discard the evidence of which cell broke. 45 and 46 are returned by
the read phase from `reading_burden`'s two refusal statuses and never by the
measure; the numbered exits report what was found in the data, and 45/46 are in
no other tool's range (39/40
M3j, 41/42 M3k, 43/44 M3l, 38 M3i, run_study 1/3-6/23, report_study 7-10, diagnose
11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages 34, sharper_latent
35-37, argparse 2, a traceback 1). 0 / 11 / 12 / 14 are `trust_horizon.py`'s, on
purpose: `prepare_cell` refuses a cell for the same reasons in the same words as
every other diagnostic.

THE READ PHASE (`--phase read`) pools the nine records and prints Reading H, from
the records alone and with no GPU. It refuses what it cannot read, in three
different ways that are kept apart on purpose:

  * BY NAME, with no number (`SystemExit` carrying the message, status 1): a plan
    narrower than `ARMS_REQUIRED` arms, records that disagree on any
    `_PROTOCOL_FIELDS` entry (`git_sha` among them), a record filed under another
    cell's name. These are the operator's, not the data's.
  * BY NUMBER: 45 when a control with a known answer was missed, 46 when the data
    cannot be read (a cell where the agent barely moved, an arm short of
    `SEEDS_MINIMUM` seeds). 11 names a cell whose record is missing.
  * BY TRACEBACK: `reading_burden`'s two `ValueError`s, which are shape errors and
    are never caught into a status or an exit.

`burden.txt` is written only for a reading, and is the same bytes stdout carries.

THE ONE ALIGNMENT RISK. `regrounding_sweep` retains its per-window rows in the
order IT walked, and `baseline_rows` walks the validation episodes in a second,
model-free loop. Row `w` of each must be the same window, and NOTHING TIES THEM
BUT THE SHARED WINDOW RULE in `eval.windows` -- there is no bitwise tie between
the two loops, and none is claimed. That rule lives in one module precisely
because, as `diagnostics.py`'s header says, its failure mode is silent drift with
no loud test available. So `baseline_rows` consumes `window_starts` and iterates
`val_paths` in the order GIVEN (never `sorted()`, never a set), `measure_cell`
refuses unless the shapes agree, and the tests distinguish the orders.

THE BASELINE IS MODEL-FREE and touches no generator, so it cannot perturb the
sampling stream `regrounding_sweep`'s matched arms depend on. It is run AFTER the
sweep regardless, and never between the sweep's arms.

LOADING IS `trust_horizon.py`'s: `load_cell` and `prepare_cell` are imported by
path through `_sibling`, so the checkpoint is loaded with
`trust_horizon.load_checkpoint_model`, the embedding probe is refit with
`fit_probes` at the rollout's own context/horizon and at the cell's seed (exactly
as `diagnose_dynamics.py` does), the split and the protocol are checked, and the
val rollout `prepare_cell` runs is shown to reproduce the study record -- all
before a pass of this script's own is paid for. This module holds its own copy of
every class `_sibling` defines; never catch another importer's copy of
`CellMissing`.

THE REFERENCE IS THE ONE THAT PASSED THAT CHECK, NOT A SECOND PASS. `prepare_cell`
refuses (`EXIT_RECORD_MISMATCH`) unless its val rollout reproduces the study record
within the bound of spec 2.4, and it hands that rollout back as
`Prepared.reference`. `measure_phase` passes it to `measure_cell`, which reads the
floor and the canonical curves off it and runs `evaluate_rollout` itself NOWHERE.
A second pass would be identical in practice -- `evaluate_rollout` seeds its
sampler -- but it would be a pass nothing had checked, and the curves written into
every record would no longer be provably the ones that reproduced the study. This
module therefore does not import `evaluate_rollout`.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.burden import (
    ARMS_REQUIRED, CONFIDENCE, DECISION_H, IDENTITY_TOLERANCE, REPORTED_H, RESAMPLES,
    BurdenArm, BurdenInputs, burden, compounding, format_reading_burden, identity_residual,
    margin_interval, one_step_persistence, reading_burden,
)
from mbfps.eval.diagnostics import REGROUNDING_KS, regrounding_sweep
from mbfps.eval.probe import probe_targets
from mbfps.eval.rollout import RolloutResult
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.eval.windows import window_starts
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_prediction_burden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
prepare_cell = _trust.prepare_cell

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH

EXIT_CONTROL_BROKEN: int = 45
"""A control with a known answer was missed: the identity residual, the k=45
rung's bitwise reproduction of the record, or k=1 collapsing onto the floor.

RAISED IN THE READ PHASE from `reading_burden`'s `UNRESOLVED_CONTROL`, never
here: the measure phase records the control and carries on."""

EXIT_UNREADABLE: int = 46
"""The reading cannot be taken from this data: a cell where the agent barely
moved, or an arm short of SEEDS_MINIMUM seeds.

RETURNED BY THE READ PHASE from `reading_burden`'s `UNREADABLE`, never here.
Two refusals look like it and are NOT it: records that disagree on the protocol
and a plan narrower than ARMS_REQUIRED arms are raised BY NAME (a `SystemExit`
carrying the message, so status 1) and have no number of their own. The numbered
exits report what was found in the DATA; those two are the operator's."""

PHASES: tuple[str, ...] = ("all", "measure", "read")
"""The phases this script runs. `all` is `measure` and then `read`.

THE DEFAULT IS THE LAST, `read`: it spends no GPU time, and without records it
refuses by name (11). A default of `measure` would make a bare invocation pay for
nine cells."""


def burden_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"burden_{arm}_seed{seed}.json"


# ---------------------------------------------------------------------------
# The model-free baseline.
# ---------------------------------------------------------------------------


def baseline_rows(val_paths, *, context: int, horizon: int):
    """Per-window true one-step displacement, plus one episode label per window.

    MODEL-FREE: this touches no model and no generator, so it cannot perturb
    the sampling stream `regrounding_sweep`'s matched arms depend on.

    THE TRAVERSAL ORDER IS THE CALLER'S. `regrounding_sweep` retains its
    per-window rows in the order IT walked, and row `w` here must be the same
    window. Nothing ties the two loops but the shared window rule from
    `eval.windows` -- which is exactly why that rule lives in one module, as
    `diagnostics.py`'s header says, because its failure mode is silent drift
    with no loud test available. Never `sorted(val_paths)`, never a set.

    The label is the index into `val_paths` of the episode a window was cut
    from. `diagnostics` numbers episodes AFTER skipping a too-short one, so the
    two disagree on the numbers once such an episode precedes another; clustering
    reads only which windows share a label, and a test compares the two as
    partitions.
    """
    rows, labels = [], []
    need = context + horizon
    for index, path in enumerate(val_paths):
        episode = load_episode(path)
        for start in window_starts(episode.length, context, horizon):
            targets = probe_targets(
                episode.privileged[start + context : start + need + 1],
                episode.privileged_keys,
            )
            rows.append(one_step_persistence(targets))
            labels.append(index)
    if not rows:
        raise SystemExit(
            f"no validation window reached {need + 1} frames; lower "
            "--context/--horizon or check the split"
        )
    return np.stack(rows), np.asarray(labels)


# ---------------------------------------------------------------------------
# measure: one cell, then every cell.
# ---------------------------------------------------------------------------


def _cell_args(args, source: Path) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
    args, and its `out` is the STUDY directory the checkpoint is loaded from
    (`trust_horizon.load_checkpoint_model(args.out, ...)`). This script's own
    `--out` holds the burden records and nothing else, so handing `args`
    through unchanged would look for the nine M3c checkpoints in the output
    directory and refuse EVERY cell -- a real run blocker on M3j, fixed with
    this same shim. Same shim, same reason, as `scripts/latent_capacity.py`.
    """
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context,
        horizon=args.horizon,
    )


def _require_a_readable_protocol(arm: str, seed: int, horizon: int, ks) -> None:
    """The status is read at `DECISION_H` and every margin off the k=1 rung.

    Refused by name BEFORE any pass is paid for: asked afterwards, the first is
    an IndexError out of the base control and the second a KeyError out of the
    sweep's retained rows.
    """
    if horizon < max(DECISION_H, *REPORTED_H):
        raise SystemExit(
            f"{arm} seed {seed}: a rollout horizon of {horizon} cannot be read at "
            f"DECISION_H={DECISION_H} and REPORTED_H up to {max(REPORTED_H)}"
        )
    if 1 not in tuple(ks):
        raise SystemExit(
            f"{arm} seed {seed}: ks={tuple(ks)!r} omits k=1, the rung every margin "
            "and every compounding is read against"
        )


def measure_cell(
    model, val_paths, probe, *, reference: RolloutResult, arm: str, seed: int,
    context: int, horizon: int, ks, device, feature_backbone, study_record: dict,
) -> dict:
    """One cell: the floor, every rung, the baseline, the intervals, the controls.

    `reference` IS `Prepared.reference`, THE ROLLOUT `prepare_cell` VERIFIED. It
    supplies the floor and the canonical curves, and this function runs no
    `evaluate_rollout` of its own: `prepare_cell` proved THAT pass reproduces the
    study record (and refused with `EXIT_RECORD_MISMATCH` when it did not), so
    the floor and the curves in the record are provably the ones that passed the
    provenance check, where a second pass would be one nothing had checked.

    IN THIS ORDER: `regrounding_sweep` for every rung; `baseline_rows` LAST, so
    it cannot sit between the sweep's arms; the alignment refusal; the finiteness
    refusal; the per-window margin; the intervals; the controls; the base control.

    `probe` IS THE ONE `fit_probes` returned, and `reference` was read through
    that same probe (`prepare_cell` fits it and passes it to its own rollout), so
    the floor and the rungs are read through the same embedding probe, or the
    margin compares two differently fitted pipelines -- the drift
    `evaluate_rollout`'s own comments record as having destroyed a signal once.

    THE FLOOR IS READ FROM THREE PLACES, two passes and no control between them:
    `reference.floor_position` (the record's curve, every burden and the identity
    residual), `sweep.reference.floor_position` (what `k_one_is_floor` compares
    against) and `sweep.window_floor_position` (the base control's median).

    THE K=1 RUNG IS THE ONE-STEP ARM. `regrounding_sweep`'s `k=1` re-grounds
    after every step and imagines one prior step from a posterior-grounded state,
    which is exactly what the margin asks about; re-implementing it here would be
    a second definition of it.

    THE STACKED MARGIN `rows - one` IS `motion_margin`'s definition, subtracted
    over every window at once. `motion_margin` has a shape and finiteness guard
    behind it and the subtraction has none, and `margin_interval` would take a
    NaN straight through to `(nan, nan, nan)` were it not for its own guard, so
    both operands are refused BY NAME here first. A test pins the stacked result
    to `motion_margin` window by window, which is what keeps the sign
    convention -- `rows - one`, positive when one prior step beats assuming the
    agent did not move -- in one place that is tested.

    `study_record` is the cell's own study record, read STRICTLY: `steps`,
    `kl_rate_above_free_bits`, `kl_dyn_max` and `git_sha` are carried from the
    training history beside the checkpoint, and a record without them is not a
    study record. `git_sha` is as strict as the other three because the read phase
    compares it across records as a protocol field: a default would let a record
    that merely lacks it pass for one written where git could not answer.

    Keys of every per-`k` and per-`h` mapping are STRINGS, the form JSON gives
    them back in, so the record is identical before and after it is written.
    """
    _require_a_readable_protocol(arm, seed, horizon, ks)
    common = dict(
        context=context, horizon=horizon, seed=seed, device=device,
        feature_backbone=feature_backbone,
    )
    sweep = regrounding_sweep(model, val_paths, probe, ks=ks, **common)
    rows, labels = baseline_rows(val_paths, context=context, horizon=horizon)

    one = sweep.window_position[1]
    if rows.shape != one.shape:
        raise SystemExit(
            f"{arm} seed {seed}: the baseline does not align with the sweep -- "
            f"{rows.shape} rows against {one.shape}. Both must be cut by "
            "`window_starts` over the same --source episodes in the same order"
        )
    for name, array in (
        ("the baseline's per-window displacement", rows),
        ("the sweep's k=1 rung", one),
    ):
        if not np.isfinite(array).all():
            raise SystemExit(
                f"{arm} seed {seed}: {name} carries non-finite values, and the "
                "stacked margin would carry them into every interval"
            )
    window_margin = rows - one

    floor = reference.floor_position
    burden_k = {k: burden(sweep.curve(k), floor) for k in ks}
    compounding_k = {k: compounding(sweep.curve(k), sweep.curve(1)) for k in ks}
    controls = {
        "identity_residual": max(
            identity_residual(sweep.curve(k), sweep.curve(1), floor) for k in ks
        ),
        "open_loop_divergence": float(sweep.open_loop_divergence(reference)),
        "k_one_is_floor": bool(sweep.is_bitwise_the_floor(1, "position")),
        "compounding_at_k_one_max_abs": float(np.max(np.abs(compounding_k[1]))),
    }
    margin = {}
    for h in REPORTED_H:
        point, low, high = margin_interval(
            window_margin, labels, h=h, resamples=RESAMPLES, seed=seed,
        )
        margin[str(h)] = {"point": point, "ci_low": low, "ci_high": high}
    base_control = {
        "displacement_median": float(np.median(rows[:, DECISION_H - 1])),
        "floor_median": float(np.median(sweep.window_floor_position[:, DECISION_H - 1])),
    }

    return {
        "arm": arm,
        "seed": seed,
        "step": int(study_record["steps"]),
        "kl_rate_above_free_bits": float(study_record["kl_rate_above_free_bits"]),
        "kl_dyn_max": float(study_record["kl_dyn_max"]),
        "record_git_sha": study_record["git_sha"],
        "git_sha": git_sha(),
        "device": str(device),
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "torch_version": torch.__version__,
        "confidence": CONFIDENCE,
        "resamples": RESAMPLES,
        "decision_h": DECISION_H,
        "reported_h": list(REPORTED_H),
        "ks": [int(k) for k in ks],
        "identity_tolerance": IDENTITY_TOLERANCE,
        "curves": {
            "rssm_position": reference.rssm_position.tolist(),
            "persistence_position": reference.persistence_position.tolist(),
            "floor_position": floor.tolist(),
            "one_step_persistence": rows.mean(axis=0).tolist(),
            "rungs": {str(k): sweep.curve(k).tolist() for k in ks},
        },
        "burden_by_k": {str(k): burden_k[k].tolist() for k in ks},
        "compounding_by_k": {str(k): compounding_k[k].tolist() for k in ks},
        "margin": margin,
        "window_margin": window_margin.tolist(),
        "controls": controls,
        "base_control": base_control,
        "windows": {"total": int(rows.shape[0]), "episode": labels.tolist()},
        "episodes": {"val": [Path(p).name for p in val_paths]},
    }


def _cell_line(record: dict, path: Path) -> str:
    """One line per measured cell: the margin at the decision horizon with its
    interval, the burden at the last rung, and the identity residual.

    The three controls are RECORDED here, not judged: the read phase decides what
    a missed one means (45), from the pooled records."""
    margin = record["margin"][str(record["decision_h"])]
    last = str(record["ks"][-1])
    controls = record["controls"]
    return (
        f"{record['arm']} seed {record['seed']}: margin at h={record['decision_h']} "
        f"{margin['point']:+.4f} [{margin['ci_low']:+.4f}, {margin['ci_high']:+.4f}]; "
        f"burden(k={last}) {record['burden_by_k'][last][record['decision_h'] - 1]:+.4f}; "
        f"identity {controls['identity_residual']:.2e}; open-loop "
        f"{controls['open_loop_divergence']:.1e}; k=1 is floor "
        f"{controls['k_one_is_floor']}; {record['windows']['total']} windows; wrote {path}"
    )


def measure_phase(args) -> int:
    """Every requested cell: `prepare_cell`'s refusals, then `measure_cell`, then
    one record per cell through `write_record`.

    EVERY CELL IS LOADED BEFORE ANY IS MEASURED, so a missing ninth cell is
    found before eight have been paid for (11). Then the split -- the study's own
    `VAL_FRACTION` at `SPLIT_SEED`, which is fixed and deliberately NOT the
    cell's seed -- and, per cell, `prepare_cell` (12, 14), the measurement, and
    the write. The first non-`EXIT_OK` status is returned and nothing is written
    for the cell that earned it.

    THE FLOOR AND THE CURVES COME FROM `prepared.reference`, the rollout
    `prepare_cell` verified against the study record, and are not recomputed: a
    second pass would be one the provenance check never saw.

    CELLS ARE LOADED FROM `args.source`, NOT `args.out`. `--out` is this script's
    own record directory and holds none of the nine checkpoints.

    A MISSED CONTROL DOES NOT STOP THE RUN. 45 belongs to the read phase, which
    decides it from the pooled records.
    """
    arms, seeds = _plan(args)
    device = get_device(prefer=args.device)
    try:
        loaded = [load_cell(args.source, arm, seed) for arm in arms for seed in seeds]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, prepared = prepare_cell(
            _cell_args(args, args.source), cell, device, train, val
        )
        if prepared is None:
            return status
        record = measure_cell(
            prepared.model, val, prepared.embedding_probe,
            reference=prepared.reference, arm=cell.arm, seed=cell.seed,
            context=prepared.context, horizon=prepared.horizon,
            ks=REGROUNDING_KS, device=device,
            feature_backbone=prepared.common["feature_backbone"],
            study_record=cell.record,
        )
        path = burden_record_path(args.out, cell.arm, cell.seed)
        write_record(path, record)
        print(_cell_line(record, path))
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: what pooling these records requires, and Reading H.
# ---------------------------------------------------------------------------


_PROTOCOL_FIELDS = (
    ("git_sha", lambda r: str(r["git_sha"])),
    ("torch_version", lambda r: str(r["torch_version"])),
    ("device", lambda r: str(r["device"])),
    ("step", lambda r: int(r["step"])),
    ("context", lambda r: int(r["context"])),
    ("horizon", lambda r: int(r["horizon"])),
    ("split_seed", lambda r: int(r["split_seed"])),
    ("ks", lambda r: [int(k) for k in r["ks"]]),
    ("decision_h", lambda r: int(r["decision_h"])),
    ("reported_h", lambda r: [int(h) for h in r["reported_h"]]),
    ("confidence", lambda r: float(r["confidence"])),
    ("resamples", lambda r: int(r["resamples"])),
    ("identity_tolerance", lambda r: float(r["identity_tolerance"])),
    ("episodes.val", lambda r: list(r["episodes"]["val"])),
    ("windows.episode", lambda r: list(r["windows"]["episode"])),
)
"""Every field `require_one_protocol` compares, as `(name, pick)`, in one
module-level table that the function ITERATES and a test reads -- so a field
cannot be added to the comparison without the test noticing it has no
disagreement case, and one cannot be dropped without it noticing the required
set shrank.

`git_sha` IS NOT OPTIONAL. `latent_capacity.require_one_protocol`'s docstring
records that an earlier version without it let records from different torch
builds and code versions pool into one finding with no refusal at all.
`torch_version` is the other half of that sentence.

`step`, `context`, `horizon` and `split_seed` say WHICH checkpoints, over which
windows, on which split; `ks`, `decision_h`, `reported_h` say what was read off
them; `confidence`, `resamples` and `identity_tolerance` are what each interval
and control was taken at. All are written from the measure phase's own
constants, so two records of one code version cannot disagree on them and a
disagreement is a mixed pool. `episodes.val` and `windows.episode` say the rows
were the same: nine cells pooled into one reading must describe the same
windows, and two that differ were scored on a union nothing measured.

NOT COMPARED, on purpose: `kl_rate_above_free_bits`, `kl_dyn_max` and
`record_git_sha`. They are properties of each CELL'S TRAINING, which differ
between cells by construction, and they are carried for the reader, not for the
verdict.

A record WITHOUT a field is refused by name (`_pick`), not defaulted: a default
would make a record that merely lacks the field pass for one that agrees."""


def _pick(cell, field: str, pick, record: dict):
    """`pick(record)`, or the named refusal for a record that lacks `field`.

    Every record here was written by `measure_cell`, so a missing field is a
    file that is not one -- a bare `KeyError` would not say which."""
    try:
        return pick(record)
    except (KeyError, TypeError) as error:
        raise SystemExit(
            f"{cell[0]} seed {cell[1]} lacks {field}: it is not a burden record "
            "this script wrote"
        ) from error


def _brief(value) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _require_agreement(records: dict, fields) -> list:
    """The sorted `(cell, record)` pairs, after refusing any two that disagree on
    one of `fields` -- named by cell and field.

    EVERY record is compared with the first, not only the second, so a record
    that sorts last is held to the same value as one that sorts second."""
    items = sorted(records.items())
    if not items:
        raise SystemExit("no burden record to read")
    (first_cell, first), rest = items[0], items[1:]
    picks = dict(_PROTOCOL_FIELDS)
    for field in fields:
        reference = _pick(first_cell, field, picks[field], first)
        for cell, record in rest:
            mine = _pick(cell, field, picks[field], record)
            if mine != reference:
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]} and {first_cell[0]} seed "
                    f"{first_cell[1]} disagree on {field}: {_brief(mine)} vs "
                    f"{_brief(reference)}; they are not one measurement and "
                    "cannot be read as one"
                )
    return items


def require_one_protocol(records: dict) -> None:
    """Every record reports the same protocol on the same windows, or the two that
    disagree are named with the field.

    Nine cells pooled into one reading must describe the same rows: two
    protocols pooled as one would be a reading over a union nothing measured.
    Mirrors `scripts/latent_capacity.py`'s namesake, and `git_sha` is among the
    fields for the reason that docstring gives."""
    _require_agreement(records, [name for name, _ in _PROTOCOL_FIELDS])


def require_readable_plan(arms) -> None:
    """Refuse a plan with fewer than `ARMS_REQUIRED` DISTINCT arms.

    Reading H needs `ARMS_REQUIRED` arms to reach a decisive status, and with
    fewer `reading_burden` does not refuse: it falls through to INDETERMINATE
    under a sentence about "0 of 1 arms", a status the PLAN produced and the data
    did not. That is the distinction M3j's `require_readable_plan` introduced,
    after a run printed `clears = up` beside a verdict of NO DIFFERENCE. Asked
    here it costs nothing; asked after `--phase all` it is nine cells of GPU time.

    THE SEED COUNT IS NOT CHECKED HERE, deliberately. An arm carrying fewer than
    `SEEDS_MINIMUM` seeds is `reading_burden`'s own refusal, and comes back as
    `UNREADABLE` (46) naming the arm: a plan check on seeds would stand in front
    of it and make that status unreachable from the read phase.

    The caller passes the DISTINCT arms (`_plan`): `--arms a a a` is one arm."""
    if len(arms) < ARMS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(arms)} arm(s) cannot be read: Reading H needs "
            f"ARMS_REQUIRED={ARMS_REQUIRED} arms to reach a decisive status, and "
            "would otherwise print INDETERMINATE for a reason the data did not give"
        )


def load_burden(out_dir: Path, arms, seeds) -> dict:
    """Every planned cell's record, keyed by `(arm, seed)` -- or `CellMissing`
    naming the first that is not on disk.

    Named, never skipped: a pool over the cells that happen to be present,
    printed under the nine cells' names, is exactly the failure
    `pooling.MissingCell` exists for. A record whose own `arm`/`seed` disagree
    with the file it was read under is refused too -- a swapped pair of files
    would pool one cell under another's name with every count still right."""
    records: dict[tuple[str, int], dict] = {}
    for arm in arms:
        for seed in seeds:
            path = burden_record_path(out_dir, arm, int(seed))
            if not path.exists():
                raise CellMissing(
                    f"{arm} seed {int(seed)}: no burden record at {path}; "
                    "run --phase measure first"
                )
            record = load_record(path)
            if record.get("arm") != arm or int(record.get("seed", -1)) != int(seed):
                raise SystemExit(
                    f"{path.name} was read for {arm} seed {int(seed)} but its "
                    f"record says arm={record.get('arm')!r} seed={record.get('seed')!r}"
                )
            records[(arm, int(seed))] = record
    return records


def burden_inputs(records: dict) -> BurdenInputs:
    """The pooled records as `BurdenInputs`, every cell at ONE decision horizon on
    ONE ladder.

    The decision horizon and the ladder are the two values `BurdenInputs` holds
    once for nine cells, so a disagreement on either is refused HERE, at the point
    the pick is made, rather than depending on `require_one_protocol` having run
    first: a first-record pick would otherwise read every other cell at a horizon
    its own record does not name.

    Each cell's margin is read at the decision horizon and each rung at the
    decision horizon's step -- `[h - 1]`, because the curves are 0-indexed and
    the horizon is counted from 1 (`burden.at_horizon`). The record's keys are
    strings, the form JSON gives them back in; `BurdenArm` takes integer rungs.
    `clusters` is the number of distinct episodes the windows were cut from and
    `rows` the number of windows; neither is read by the verdict."""
    items = _require_agreement(records, ("decision_h", "ks"))
    decision_h = int(items[0][1]["decision_h"])
    ks = tuple(int(k) for k in items[0][1]["ks"])
    cells = {}
    for (arm, seed), record in items:
        margin = record["margin"][str(decision_h)]
        controls, base = record["controls"], record["base_control"]
        cells[(arm, seed)] = BurdenArm(
            arm=arm, seed=seed,
            margin=float(margin["point"]),
            margin_low=float(margin["ci_low"]),
            margin_high=float(margin["ci_high"]),
            burden_by_k={
                k: float(record["burden_by_k"][str(k)][decision_h - 1]) for k in ks
            },
            compounding_by_k={
                k: float(record["compounding_by_k"][str(k)][decision_h - 1]) for k in ks
            },
            identity_residual=float(controls["identity_residual"]),
            open_loop_divergence=float(controls["open_loop_divergence"]),
            k_one_is_floor=bool(controls["k_one_is_floor"]),
            displacement_median=float(base["displacement_median"]),
            floor_median=float(base["floor_median"]),
            clusters=len(set(record["windows"]["episode"])),
            rows=int(record["windows"]["total"]),
        )
    return BurdenInputs(cells=cells, decision_h=decision_h, ks=ks)


READ_EXITS: dict[str, int] = {
    "UNRESOLVED_CONTROL": EXIT_CONTROL_BROKEN,
    "UNREADABLE": EXIT_UNREADABLE,
}
"""Reading H's two refusal statuses, each to its own exit.

Keyed on exactly the two refusal statuses `reading_burden` returns; every other
status falls through to `EXIT_OK`. `PREDICTS_MOTION`, `COPIES` and
`INDETERMINATE` are READINGS, and a milestone that exited non-zero on a finding
would make "the run worked" and "the news was good" the same signal.

The numbered statuses report what was found in the DATA. A narrowed plan is the
operator asking for something no data can answer, so it raises rather than
adding a status -- the distinction M3j's `require_readable_plan` introduced."""


def _plan(args) -> tuple[list[str], list[int]]:
    """The DISTINCT arms and seeds, in the order given. `--arms a a` is one arm
    and `--seeds 0 0 0` one seed: counting the lists would let a plan through
    that the records dict, keyed by cell, cannot honour -- and measuring from the
    lists as given would measure a cell twice and write its record twice."""
    return (
        list(dict.fromkeys(args.arms)),
        list(dict.fromkeys(int(s) for s in args.seeds)),
    )


def read_phase(args) -> int:
    """Refuse a plan that cannot be read, pool every requested cell (11 names the
    first missing), refuse records that are not one measurement, decide Reading H,
    and return `READ_EXITS.get(reading.status, EXIT_OK)`: 0 for every reading, 45
    or 46 for the two refusals.

    A READING IS WRITTEN AND PRINTED AS ONE STRING: `burden.txt` is the artefact
    and stdout the log, `format_reading_burden` already ends its text in a
    newline, and this function prints it with `end=""`. `print(text)` would put
    a second newline on stdout and none in the file. The text is written as it
    is -- never `rstrip()`ped.

    A REFUSAL WRITES NOTHING. Its status and the rule that names the offending
    cells are printed, and no `burden.txt` is created, so there is no file to be
    mistaken for a reading.

    NOTHING CATCHES `reading_burden` HERE, and that is deliberate. It raises
    `ValueError` for an empty cell set and for two decisive statuses that both
    reach the bar; neither is something found in the data, and a `try` would turn
    an arm count that has outgrown the bar into an exit number. The first is
    unreachable from here (the plan check refuses before any record is read); the
    second is reachable only with four or more arms."""
    arms, seeds = _plan(args)
    require_readable_plan(arms)
    try:
        records = load_burden(args.out, arms, seeds)
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    require_one_protocol(records)
    inputs = burden_inputs(records)
    reading = reading_burden(inputs)
    code = READ_EXITS.get(reading.status, EXIT_OK)
    if code != EXIT_OK:
        print(
            f"{reading.status}: {reading.rule}\n"
            "no reading was taken, so no burden.txt was written"
        )
        return code
    text = format_reading_burden(reading, inputs)
    (Path(args.out) / "burden.txt").write_text(text)
    print(text, end="")
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M3m: does the one-step map predict motion, or did it never learn it? "
            "-- measure the nine cells, read Reading H"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/m3m_burden"))
    parser.add_argument("--source", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: the nine 20,000-step cells")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--phase", choices=PHASES, default=PHASES[-1])
    return parser


def main(argv: list[str] | None = None) -> int:
    """`measure`, `read`, or `all` -- the measure and then the read, which does
    not run when the measure stopped with a status: a cell that was refused wrote
    no record, and a read after it would be over a pool missing that cell.

    `--phase all` REFUSES A PLAN `read` COULD NOT READ before any cell is
    measured; `--phase measure` is allowed one, because the milestone's smoke is
    one cell."""
    args = _parser().parse_args(argv)
    if args.phase == "all":
        require_readable_plan(_plan(args)[0])
    if args.phase in ("measure", "all"):
        status = measure_phase(args)
        if status != EXIT_OK:
            return status
    if args.phase in ("read", "all"):
        return read_phase(args)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
