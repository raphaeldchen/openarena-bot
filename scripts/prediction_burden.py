"""M3m's prediction-burden ladder -- the measure phase, and a read phase that is retired.

M3l refuted the bottleneck lever and left the objective lever standing by
elimination, but the hypothesis it inherited -- "the loss never asks for motion"
-- is false as stated: the dynamics prior trained on 75-98% of steps. What
survives is narrower. The loss asks for ONE-step latent agreement; the gate reads
45-step rollout error. So M3m asked whether the 45-step failure is error that
COMPOUNDS over a rollout, or a one-step map that never learned motion at all, by
reading the re-grounding ladder against a baseline that is GROUND TRUTH.
`mbfps.eval.burden` is the pure half -- the decomposition and the baseline -- and
owns no torch, device or record schema. This script is the impure half.

READING H IS RETIRED, AND THIS SCRIPT NO LONGER COMPUTES THE STATISTIC IT WAS READ
FROM. M3m's `motion_margin` subtracted the k=1 rung, a model error measured through
the probe (a median floor error of 88-224 map units), from the median true one-step
displacement, a ground-truth quantity (3.97), so the readout error dominated the
difference 22.2x-56.5x, the base control failed 9 of 9 and Reading H shipped
UNREADABLE. M3n replaced it with `mbfps.eval.headroom` and `scripts/motion_headroom.py`.
The nine records at `runs/m3m_burden` and `runs/m3m_burden_rulers` stay on disk, because
the M3m spec cites them, and the read phase still parses them: it prints the margin each
one STORED under a legend saying it is superseded and why, and then refuses. A record
this script writes NOW carries no margin.

TWO AXES, NEVER CONFLATED. `k` is the RE-GROUNDING PERIOD, over
`diagnostics.REGROUNDING_KS`; `h` is the HORIZON STEP.

PER CELL, THE RECORD CARRIES:

  curves          the canonical position curves of the reference `prepare_cell`
                  VERIFIED against the study record, and every rung of the
                  ladder (`regrounding_sweep`), plus the baseline's mean.
  burden_by_k     `rung(k) - floor`, per horizon step, for every k.
  compounding_by_k `rung(k) - rung(1)`, per horizon step, for every k.
  controls        five numbers with known answers, MEASURED and never asserted:
                  the identity residual, the k=horizon rung's divergence from
                  that verified reference (0.0), whether k=1 collapsed onto the floor
                  (False), the k=1 rung's compounding (0.0), and the divergence
                  between the floor the reference carries and the floor the sweep's
                  own pass computed (0.0).
  base_control    the median true one-step displacement against the median floor
                  error at `DECISION_H`: if the agent barely moved, no method
                  could detect motion prediction in that cell. RECORDED, and used
                  by no statistic.
  rulers          two PAIRED standard errors at `DECISION_H`, off the sweep's own
                  per-window rows: the ruler on `burden(1)` and the ruler on
                  `compounding(ks[-1])`. The curves in the record are means, and the
                  per-window rows they came from are not kept, so without these no
                  interval on a share read off the ladder could be computed from the
                  records afterwards.
  protocol        `decision_h`, `reported_h`, `ks` and `identity_tolerance`, from this
                  module's and `burden`'s constants, beside the device, the git sha
                  and the windows' episode labels.

NOT IN THE RECORD, since M3n: `margin`, `window_margin`, `confidence` and `resamples`.
The first two are the retired statistic and the windows it was drawn from; the last two
name the level and the draw count of an interval nothing draws, and a record that names a
level for an interval it does not hold is worse than one that names none.

NOTHING IN THIS SCRIPT DECIDES ANYTHING. A control that missed its known answer is
RECORDED, not raised, and the run carries on: the records are the artefact, and a
measure that raised would discard the evidence of which cell broke. 45 and 46 are
HISTORICAL: Reading H returned them from its two refusal statuses and M3m reported 46,
so the constants stay and the numbers are not reissued (39/40 M3j, 41/42 M3k, 43/44
M3l, 47/48 M3n, 38 M3i, run_study 1/3-6/23, report_study 7-10, diagnose 11-17, pool
18-22, trust 30, split_gap 31, ladder 32-33, stages 34, sharper_latent 35-37, argparse 2,
a traceback 1). 0 / 11 / 12 / 14 are `trust_horizon.py`'s, on purpose: `prepare_cell`
refuses a cell for the same reasons in the same words as every other diagnostic.

THE READ PHASE (`--phase read`) IS RETIRED. It loads the records the plan names, refuses
records that do not describe one protocol, prints the margin each stored (a dash for a
record that carries none) under the superseded legend, and refuses by name, pointing at
`scripts/motion_headroom.py`. It takes no verdict, writes no `burden.txt` and REMOVES
NOTHING: it used to unlink the `burden.txt` an earlier read left, and a retired read that
deleted a file from a records directory shared with the M3m spec would be destroying an
artefact to protect a reader from a reading it can no longer take. Two refusals, kept
apart on purpose:

  * BY NAME, with no number (`SystemExit` carrying the message, status 1): the
    retirement itself, records that disagree on any `_PROTOCOL_FIELDS` entry (`git_sha`
    among them), a record filed under another cell's name, and `--phase all`, which was
    the measure and then the read and is refused before any cell is measured. These
    are the operator's, not the data's.
  * BY NUMBER: 11 names a cell whose record is missing.

THE ONE ALIGNMENT RISK. `regrounding_sweep` retains its per-window rows in the
order IT walked, and `baseline_rows` walks the validation episodes in a second,
model-free loop. Row `w` of each must be the same window, and NOTHING TIES THEM
BUT THE SHARED WINDOW RULE in `eval.windows` -- there is no bitwise tie between
the two loops, and none is claimed. That rule lives in one module precisely
because, as `diagnostics.py`'s header says, its failure mode is silent drift with
no loud test available. So `baseline_rows` consumes `window_starts` and iterates
`val_paths` in the order GIVEN (never `sorted()`, never a set), `measure_cell`
refuses unless the shapes agree, and the tests distinguish the orders. THAT REFUSAL IS
NOW THE ONLY TIE: nothing subtracts the two arrays any more, so a baseline cut on another
window rule would no longer raise out of a subtraction -- it would be the base control's
medians, taken over windows that are not the sweep's.

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
    IDENTITY_TOLERANCE, REPORTED_H, at_horizon, burden, compounding, identity_residual,
    one_step_persistence,
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
"""HISTORICAL. A control with a known answer was missed: the identity residual, the
k=45 rung's bitwise reproduction of the record, or k=1 collapsing onto the floor.

M3m's read phase returned it from `reading_burden`'s `UNRESOLVED_CONTROL`. That
reading is retired and NOTHING RETURNS THIS NUMBER NOW; the constant stays because
the exit-code registry holds it and a number a milestone reported must not be
reissued to another failure. The control itself is still MEASURED, into every
record's `controls`, and `scripts/motion_headroom.py` refuses a missed control by
name."""

EXIT_UNREADABLE: int = 46
"""HISTORICAL. The reading cannot be taken from this data: a cell where the agent
barely moved, or an arm short of SEEDS_MINIMUM seeds.

M3m's read phase returned it from `reading_burden`'s `UNREADABLE`, and it is the
status M3m shipped: the base control failed in 9 of 9 cells. That reading is
retired and NOTHING RETURNS THIS NUMBER NOW; the constant stays for the reason
`EXIT_CONTROL_BROKEN` gives. 47 and 48 are `scripts/motion_headroom.py`'s."""

DECISION_H: int = 45
"""The horizon the base control and the two rulers are read at -- the M3 gate's own
horizon, and the horizon the ladder's headline share is read at.

THIS IS NOT `headroom.DECISION_H`, which is 1 and which M3n's verdict is read at.
It was `burden.DECISION_H` until M3n removed the status that was read there; what
survives of it is the horizon of two recorded controls, held HERE, by the one module
that reads them. It is recorded as `decision_h` beside the controls."""

PHASES: tuple[str, ...] = ("all", "measure", "read")
"""The phases the parser accepts. `all` was `measure` and then `read`; the read is
retired, so `main` REFUSES `all` by name before any cell is measured, and the parser
still accepts it so that an operator who types it is told why rather than shown
argparse's list.

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
    """The base control and the rulers are read at `DECISION_H`, the record's grid
    reaches the largest step of `REPORTED_H`, and every compounding and every ruler
    is read off the k=1 rung.

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
            f"{arm} seed {seed}: ks={tuple(ks)!r} omits k=1, the rung every "
            "compounding and every ruler is read against"
        )


def measure_cell(
    model, val_paths, probe, *, reference: RolloutResult, arm: str, seed: int,
    context: int, horizon: int, ks, device, feature_backbone, study_record: dict,
) -> dict:
    """One cell: the floor, every rung, the baseline, the controls and the rulers.

    `reference` IS `Prepared.reference`, THE ROLLOUT `prepare_cell` VERIFIED. It
    supplies the floor and the canonical curves, and this function runs no
    `evaluate_rollout` of its own: `prepare_cell` proved THAT pass reproduces the
    study record (and refused with `EXIT_RECORD_MISMATCH` when it did not), so
    the floor and the curves in the record are provably the ones that passed the
    provenance check, where a second pass would be one nothing had checked.

    IN THIS ORDER: `regrounding_sweep` for every rung; `baseline_rows` LAST, so
    it cannot sit between the sweep's arms; the alignment refusal; the finiteness
    refusal; the ladder; the controls; the base control; the rulers.

    NO MARGIN IS COMPUTED, and none is recorded. M3m drew an episode-clustered
    interval on `baseline - the k=1 rung` at every `REPORTED_H`, and that statistic
    is retired (see the module docstring). Both refusals below survive it: the
    baseline's rows are still reduced to the base control's median and the recorded
    `one_step_persistence` curve, and the sweep's k=1 rows to the rulers.

    `probe` IS THE ONE `fit_probes` returned, and `reference` was read through
    that same probe (`prepare_cell` fits it and passes it to its own rollout), so
    the floor and the rungs are read through the same embedding probe, or every
    burden compares two differently fitted pipelines -- the drift
    `evaluate_rollout`'s own comments record as having destroyed a signal once.

    THE FLOOR IS READ FROM THREE PLACES, from two passes: `reference.floor_position`
    (the record's curve, every burden and the identity residual),
    `sweep.reference.floor_position` (what `k_one_is_floor` compares against) and
    `sweep.window_floor_position` (the base control's median). What ties the two
    passes together is `controls["floor_divergence"]`, the largest absolute
    difference between the first two curves over the horizon, which must be 0.0.
    `open_loop_divergence == 0.0` is NOT that control: it shows the two passes agree
    on the RSSM curve, and says nothing about the floor curve. It is RECORDED, not
    judged -- nothing in this script reads it, so a record that lacks it (the nine that
    predate it) is still readable.

    THE RULERS ARE THE SWEEP'S PAIRED ONES, never its unpaired spread. `burden(1)`
    is the k=1 rung minus the floor and `compounding(ks[-1])` is the last rung minus
    the k=1 rung, both on the same windows from the same per-window RNG snapshot, so
    the ruler each has to clear is the spread of the per-window DIFFERENCE:
    `floor_margin_standard_error(1)` and `paired_standard_error(ks[-1], 1)`.
    `curve_standard_error` is the between-window spread of ONE curve, dominated by
    window difficulty that every rung shares, and `diagnostics` measured it
    overstating the bar for an adjacent-k difference by 1.7x to 3.9x. Recorded at
    `DECISION_H`, the horizon the headline share is read at, because the per-window
    rows they come from are not kept.

    THE K=1 RUNG IS THE ONE-STEP ARM. `regrounding_sweep`'s `k=1` re-grounds
    after every step and imagines one prior step from a posterior-grounded state,
    which is the rung every compounding is read against; re-implementing it here
    would be a second definition of it.

    BOTH PER-WINDOW ARRAYS ARE REFUSED BY NAME if they carry a non-finite value:
    the baseline's rows and the sweep's k=1 rows. They are reduced to a median, a
    mean and two standard errors, and a NaN or an inf in either would be written into
    the record, where `study.write_record` nulls it and the read meets a hole instead
    of the cause.

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
                "medians, curves and rulers computed from it would carry them into "
                "the record"
            )

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
        "floor_divergence": float(np.max(np.abs(
            floor - sweep.reference.floor_position
        ))),
    }
    base_control = {
        "displacement_median": float(np.median(
            [at_horizon(window, DECISION_H) for window in rows]
        )),
        "floor_median": float(np.median(
            [at_horizon(window, DECISION_H) for window in sweep.window_floor_position]
        )),
    }
    rulers = {
        "floor_margin_standard_error": at_horizon(
            sweep.floor_margin_standard_error(1), DECISION_H
        ),
        "paired_standard_error": at_horizon(
            sweep.paired_standard_error(ks[-1], 1), DECISION_H
        ),
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
        "controls": controls,
        "base_control": base_control,
        "rulers": rulers,
        "windows": {"total": int(rows.shape[0]), "episode": labels.tolist()},
        "episodes": {"val": [Path(p).name for p in val_paths]},
    }


def _cell_line(record: dict, path: Path) -> str:
    """One line per measured cell: the burden at the last rung, read at the decision
    horizon, and the controls.

    The controls are RECORDED here, not judged: nothing in this script decides what
    a missed one means, and `scripts/motion_headroom.py` refuses it by name."""
    last = str(record["ks"][-1])
    controls = record["controls"]
    return (
        f"{record['arm']} seed {record['seed']}: "
        f"burden(k={last}) {at_horizon(record['burden_by_k'][last], record['decision_h']):+.4f}; "
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

    A MISSED CONTROL DOES NOT STOP THE RUN, and nothing here decides what it means:
    the record carries it, and the reading that refuses it is `motion_headroom.py`'s.
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
# read: what pooling these records requires, and the retired reading.
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
them; `identity_tolerance` is what the identity control was taken at. All are
written from the measure phase's own constants, so two records of one code version
cannot disagree on them and a disagreement is a mixed pool. `episodes.val` and
`windows.episode` say the rows were the same: nine cells pooled into one table must
describe the same windows, and two that differ were scored on a union nothing
measured.

`confidence` AND `resamples` ARE NOT HERE since M3n, because they described the
interval on the retired margin and a new record carries neither. The nine M3m records
still do, and nothing compares them: they were written by one run and agree. Keeping
them in the table would refuse every record this script writes now as "not a burden
record this script wrote".

NOT COMPARED, on purpose: `kl_rate_above_free_bits`, `kl_dyn_max` and
`record_git_sha`. They are properties of each CELL'S TRAINING, which differ
between cells by construction, and they are carried for the reader, not for a
comparison.

A record WITHOUT a field is refused by name (`_pick`), not defaulted: a default
would make a record that merely lacks the field pass for one that agrees."""


def _pick(cell, field: str, pick, record: dict):
    """`pick(record)`, or the named refusal for a record that lacks `field`.

    Every record here was written by `measure_cell`, so a missing field is a
    file that is not one -- a bare `KeyError` would not say which."""
    try:
        return pick(record)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"{cell[0]} seed {cell[1]} lacks {field}: it is not a burden record "
            "this script wrote"
        ) from error


_LINE = 60
"""How much of one value a refusal prints before it stops being a line."""


def _brief(value) -> str:
    """`repr(value)` cut to a line. For ONE value that is all there is to say; two
    long values that differ somewhere in the middle need `_versus`, which says where."""
    text = repr(value)
    return text if len(text) <= _LINE else text[:_LINE - 3] + "..."


def _versus(mine, reference) -> str:
    """`mine vs reference`, in a form a person can act on.

    A value that fits on a line is printed as it is. Two long lists -- the 150
    window labels of `windows.episode`, the episode names of `episodes.val` -- are
    the case a bare truncation fails: they are cut at the same character and read
    as one text, however they differ. So they are told apart by LENGTH when the
    lengths differ (`151 vs 150 rows`) and by the FIRST INDEX where they differ when
    they do not, with the two values found there.

    THERE IS NO FALLBACK, because no `_PROTOCOL_FIELDS` pick reaches one: the only
    values longer than a line are those two lists (the longest scalar, `git_sha`,
    prints at 42 characters against a line of 60), and the caller compares only
    values that differ, so two long lists of one length differ at SOME index. A
    value outside that set raises (`TypeError`, `StopIteration`) instead of being
    printed as two truncations that read as one.

    The count is of rows, not of anything the lists hold; the previous milestone
    printed `(N vs M rows)` alone, which is the same number twice when the lists
    are as long as each other."""
    shown = f"{_brief(mine)} vs {_brief(reference)}"
    if max(len(repr(mine)), len(repr(reference))) <= _LINE:
        return shown
    if len(mine) != len(reference):
        return f"{len(mine)} vs {len(reference)} rows"
    at = next(i for i, (a, b) in enumerate(zip(mine, reference)) if a != b)
    return (
        f"{len(mine)} rows, first differing at index {at}: "
        f"{_brief(mine[at])} vs {_brief(reference[at])}"
    )


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
                    f"{first_cell[1]} disagree on {field}: {_versus(mine, reference)}; "
                    "they are not one measurement and cannot be read as one"
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


def _plan(args) -> tuple[list[str], list[int]]:
    """The DISTINCT arms and seeds, in the order given. `--arms a a` is one arm
    and `--seeds 0 0 0` one seed: counting the lists would let a plan through
    that the records dict, keyed by cell, cannot honour -- and measuring from the
    lists as given would measure a cell twice and write its record twice."""
    return (
        list(dict.fromkeys(args.arms)),
        list(dict.fromkeys(int(s) for s in args.seeds)),
    )


STORED_MARGIN_COLUMNS: tuple[str, ...] = ("arm", "seed", "margin", "ci_low", "ci_high")
STORED_MARGIN_WIDTHS: tuple[int, ...] = (13, 6, 11, 11, 11)
"""The superseded table's columns and their widths, right-aligned and unseparated.

`margin`, `ci_low` and `ci_high` are the record's `margin[str(decision_h)]` triple
as M3m's measure phase stored it: the mean of `baseline - the k=1 rung` over the
windows, and the 95% episode-clustered interval drawn around it."""

RETIRED: str = (
    "Reading H is retired, so this read takes no verdict, writes no burden.txt and "
    "removes nothing. M3n replaced it: run scripts/motion_headroom.py and read ITS "
    "verdict and its rule line. The table above is the margin these records stored, "
    "printed under the legend that says why it is superseded, so that it is not "
    "mistaken for a result"
)
"""The message `read_phase` refuses with, and so what a nonzero exit carries."""

PHASE_ALL_RETIRED: str = (
    "--phase all was the measure and then the read, and the read is retired: it would "
    "pay for every cell and then refuse. Run --phase measure for the ladder records "
    "this script still writes, and scripts/motion_headroom.py for the reading that "
    "replaced Reading H"
)
"""The message `main` refuses `--phase all` with, BEFORE any cell is measured."""


def _row(values, widths) -> str:
    return "".join(f"{str(v):>{w}}" for v, w in zip(values, widths, strict=True))


def format_superseded_margin(records: dict) -> str:
    """The margin each record stored, at the records' decision horizon, under the
    legend that says it is superseded -- as a string ending in one newline.

    EVERY CELL IS LISTED, in `sorted()` order. A record that carries no `margin` --
    one this script wrote after M3n retired it -- is listed with a dash in each of
    the three columns and the legend says what the dash means; it is not a `KeyError`
    and it is not skipped. `record.get("margin")`, never `record["margin"]`: the
    measure phase writes none now, so the latter would crash on every record it
    produces.

    THE DECISION HORIZON IS THE RECORDS' ONE, and a pool that disagrees on it is
    refused here, where the pick is made, rather than depending on
    `require_one_protocol` having run first: a first-record pick would read every
    other cell at a horizon its own record does not name.

    THE LEGEND IS THE POINT of printing a number nothing may read. Its two figures
    are M3m's own, from its nine records, and it says so; they are not a claim about
    any other pool. What it must not do is what M3m's did: tell a reader that a
    margin of -116 means the one-step map copies, the sentence that shipped in three
    places the same milestone's results refuted.
    """
    items = _require_agreement(records, ("decision_h",))
    decision_h = int(items[0][1]["decision_h"])
    lines = [
        f"--- SUPERSEDED: M3m's motion_margin at horizon {decision_h}, as its records stored it",
        _row(STORED_MARGIN_COLUMNS, STORED_MARGIN_WIDTHS),
    ]
    carries_none = False
    for (arm, seed), record in items:
        stored = record.get("margin")
        if stored is None:
            carries_none = True
            numbers = ("-", "-", "-")
        else:
            entry = stored[str(decision_h)]
            numbers = tuple(
                f"{float(entry[key]):+.4f}" for key in ("point", "ci_low", "ci_high")
            )
        lines.append(_row((arm, seed, *numbers), STORED_MARGIN_WIDTHS))
    lines += [
        "  superseded by M3n's headroom, and printed only so that a stored number is "
        "not mistaken for a result: no reading was taken from it",
        "  margin = the true one-step displacement minus the k=1 rung. The "
        "displacement is ground truth, a median of 3.97 map units on the shipped "
        "split; the rung is read through the probe, and on M3m's nine cells the "
        "median floor error alone was 88.29-224.36, so readout error dominated the "
        "difference 22.2x-56.5x",
        "  so a negative margin here is that readout error and not the model's: it "
        "does not mean the one-step map copies, and on those nine cells it sat within "
        "0.1% to 5.8% of what a perfect one-step predictor would have read",
        "  read scripts/motion_headroom.py instead: headroom, skill and deficit, three "
        "probe-space differences that share one denominator",
    ]
    if carries_none:
        lines.append(
            "  - = this record carries no margin: it was written after the statistic "
            "was retired"
        )
    return "\n".join(lines) + "\n"


def read_phase(args) -> int:
    """The retired reading: pool the records, print the margin each stored under the
    superseded legend, and refuse BY NAME.

    A missing cell is still found in the data and returned as a NUMBER, 11
    (`NO CELL`, naming the first the plan reaches), and nothing is printed. Records
    that do not describe one protocol are refused by name, as before. Past both, the
    table is printed to stdout and `RETIRED` is raised as a `SystemExit`: status 1,
    the message on stderr, and nothing returned. This function therefore returns an
    int only for the missing cell; every other way out is a `SystemExit`.

    NOTHING IS WRITTEN AND NOTHING IS REMOVED. A refusal is not a reading, so no
    `burden.txt` is created; and the old `unlink` of the `burden.txt` an earlier read
    left is gone with the reading that wrote it. That directory is a records
    directory shared with the M3m spec, and a read that deleted from it, to spare a
    reader a reading this script can no longer take, would be destroying an artefact
    for nothing.

    THE PLAN IS NOT NARROWED. `--arms pixel_ae` prints one arm's stored margins, which
    is a legitimate thing to want from an old record; the checks that a plan must
    carry `ARMS_REQUIRED` arms and `SEEDS_MINIMUM` seeds belonged to a verdict, and
    there is none.
    """
    arms, seeds = _plan(args)
    try:
        records = load_burden(args.out, arms, seeds)
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    require_one_protocol(records)
    print(format_superseded_margin(records), end="")
    raise SystemExit(RETIRED)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M3m's prediction-burden ladder: measure the nine cells. The read phase "
            "is retired -- it prints the margin the records stored, superseded, and "
            "points at scripts/motion_headroom.py"
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
    """`measure` or `read`. `all` is refused.

    `--phase all` was the measure and then the read. The read is retired and always
    refuses, so `all` would pay for every planned cell and then fail: it is refused BY
    NAME, before the first cell is measured, with the two things that still work
    (`PHASE_ALL_RETIRED`).

    `--phase measure` RETURNS `measure_phase`'s status as it is, so a refused cell
    (11, 12, 14 from the shared cell checks, 30 from the self-check) is the exit
    status and not a success printed after a refused 30-minute run. `--phase read`
    returns `read_phase`'s, which is 11 or a `SystemExit` -- see there."""
    args = _parser().parse_args(argv)
    if args.phase == "all":
        raise SystemExit(PHASE_ALL_RETIRED)
    if args.phase == "measure":
        return measure_phase(args)
    return read_phase(args)


if __name__ == "__main__":
    sys.exit(main())
