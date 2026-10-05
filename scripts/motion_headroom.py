"""M3n: is the one-step map closer to copying, or to perfect? -- measure and read.

M3m asked whether the one-step map predicts motion, and could not say: its
`motion_margin` subtracted a probe-space model error of 88-224 map units from a
ground-truth displacement of 3.97, and the readout error dominated the difference
22.2x-56.5x. `mbfps.eval.headroom` is the pure half -- the three differences, the
gate, the placement, the reading and its table -- and owns no torch, device or
record schema. This script is the impure half.

TWO AXES, NEVER CONFLATED. `k` is the RE-GROUNDING PERIOD, over
`diagnostics.REGROUNDING_KS`; `h` is the HORIZON STEP. The verdict is read at ONE
cell, `(DECISION_K, DECISION_H)`; everything else is recorded for a reader.

PER CELL, THE RECORD CARRIES (every constant it was taken at is stored, so none is
a literal that survives the tests):

  curves        the canonical position curves of the reference `prepare_cell`
                VERIFIED against the study record, and, per k, the sweep's rung
                and hold-baseline means.
  intervals     `(point, ci_low, ci_high)` of `headroom`, `skill` and `deficit`
                at every k and every `REPORTED_H`, episode-clustered, drawn at the
                cell's own `cell_bootstrap_seed`, which `cell_bootstrap_seed()`
                derives from `--bootstrap-seed` and the cell's `(arm, seed)`: the
                nine cells must not resample the same episode draws, because the
                verdict counts them as nine independent readings. Both seeds are
                recorded. No per-window rows are kept: they are 13 x 45
                x 15 numbers, and the intervals are the record.
  share         `skill / headroom` as a POINT ESTIMATE, or null where `headroom`'s
                interval is not above zero. Presentation: the verdict reads the
                two differences, and a ratio is ill-defined where the gate fails.
  secondary     `SECONDARY_SIGMAS` times one iid standard error per difference per
                horizon step. It decides nothing.
  controls      five numbers with known answers, MEASURED and never asserted:
                the k=horizon hold against the persistence curve (0.0), the
                across-k spread at h=1 (0.0), the k=horizon rung against the
                verified reference (0.0), the sweep's floor against the verified
                reference's (0.0), and the identity residual (below
                `IDENTITY_TOLERANCE`) -- plus `negative_headroom_steps`, a list the
                gate reads, not a value checked against a constant.
  nonfinite     NOT a field `measure_cell` sets. `study.write_record` owns the
                top-level key `nonfinite` (the map from the dotted path of each
                non-finite value it nulled to the token it came from) and RAISES
                for a record that already carries one, so the schema's
                `"nonfinite": 0` -- a count under the same name -- would have
                crashed the first write of the run. On disk the key is `{}` for
                every record this script wrote, because `measure_cell` refuses a
                non-finite row before it writes; the read phase refuses a record
                where it is not.

NOTHING IN THE MEASURE PHASE DECIDES ANYTHING. A control that missed its known
answer is RECORDED, and flagged in the cell's printed line, and the run carries on:
the records are the artefact, and a measure that raised would discard the evidence
of which cell broke. The READ phase refuses it, by name.

ONE TRAVERSAL. `regrounding_sweep` computes the rungs, the shared floor, the hold
baseline and the episode labels on the same windows in the same order, so no
alignment between two passes is needed and none is assumed. M3m had to walk the
validation episodes a second time for the labels, with a documented numbering
mismatch; M3n does not.

THE READ PHASE (`--phase read`) pools the records and prints the reading, from the
records alone and with no GPU. It refuses what it cannot read, in three different
ways that are kept apart on purpose:

  * BY NAME, with no number (`SystemExit` carrying the message, status 1): a plan
    narrower than `ARMS_REQUIRED` arms, records that disagree on any
    `_PROTOCOL_FIELDS` entry (`git_sha` among them), a record filed under another
    cell's name, a record with no episode labels or a non-finite interval, a
    control that missed its known answer, and `--phase all` over fewer than
    `SEEDS_MINIMUM` seeds (refused before any cell is measured). These are the
    operator's or the measurement's, not the model's.
  * BY NUMBER: 47 when a strict majority of cells read UNREADABLE, 48 when no
    placement held a strict majority. 11 names a cell whose record is missing.
    47 and 48 are FINDINGS, so the table is printed and written for both. THE
    `rule` LINE IS PRINTED, not only the verdict, because 48 has two causes -- no
    placement reached the bar, or one did and its cells spanned fewer than
    `ARMS_REQUIRED` arms -- and only the rule line says which. The four model
    placements, AMBIGUOUS included, exit 0: each is an answer.
  * BY TRACEBACK: `reading_headroom`'s `ValueError` for an arm short of
    `SEEDS_MINIMUM` seeds (`--phase read --seeds 0 1`) and for fewer than
    `ARMS_REQUIRED` arms. They are shape errors, never caught into a status.

`headroom.txt` is written only for a reading, and is the same bytes stdout carries.
A read that takes no reading also removes the one an earlier read of the same
`--out` left, so the file is the current reading or it is absent.

EXIT NUMBERS. 47 and 48 are returned by the read phase and in no other tool's range
(39/40 M3j, 41/42 M3k, 43/44 M3l, 45/46 M3m, 38 M3i, run_study 1/3-6/23, report_study
7-10, diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages 34,
sharper_latent 35-37, argparse 2, a traceback 1). 0 / 11 / 12 / 14 are
`trust_horizon.py`'s, on purpose: `prepare_cell` refuses a cell for the same reasons
in the same words as every other diagnostic. This script RETURNS its statuses, as
`prediction_burden.py` does, and an exception from a cell propagates.

LOADING IS `trust_horizon.py`'s: `load_cell` and `prepare_cell` are imported by path
through `_sibling`, so the checkpoint is loaded with
`trust_horizon.load_checkpoint_model`, the embedding probe is refit with
`fit_probes` at the rollout's own context/horizon and at the cell's seed, the split
and the protocol are checked, and the val rollout `prepare_cell` runs is shown to
reproduce the study record -- all before a pass of this script's own is paid for.
This module holds its own copy of every class `_sibling` defines; never catch
another importer's copy of `CellMissing`.

THE REFERENCE IS THE ONE THAT PASSED THAT CHECK, NOT A SECOND PASS.
`Prepared.reference` is the rollout `prepare_cell` verified against the study
record. `measure_cell` writes its three position curves into the record and reads
the two reference controls against it, and runs `evaluate_rollout` itself NOWHERE.
Those controls are ALSO the `keep_trajectories=True` control: the verified
reference was measured with the flag OFF, so a bitwise match on
`open_loop_divergence` and `floor_divergence` proves the flag moved neither the
arms nor the reference, with no second traversal.
"""

import argparse
import importlib.util
import math
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import REGROUNDING_KS, RegroundingSweep, regrounding_sweep
from mbfps.eval.headroom import (
    ARMS_REQUIRED, CONFIDENCE, DECISION_H, DECISION_K, DISPLACEMENT_RECORDED_ONLY,
    IDENTITY_TOLERANCE, NO_MAJORITY, REPORTED_H, RESAMPLES, SECONDARY_SIGMAS, SEEDS_MINIMUM,
    HeadroomCell, HeadroomInputs, Interval, deficit, format_reading_headroom, headroom,
    reading_headroom, skill, triple_residual,
)
from mbfps.eval.pooling import clustered_interval
from mbfps.eval.rollout import RolloutResult
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_motion_headroom", path)
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

EXIT_UNREADABLE_HEADROOM: int = 47
"""RETURNED BY THE READ PHASE when a strict majority of cells read UNREADABLE.

The INSTRUMENT's verdict, not the model's: a perfect one-step predictor is
indistinguishable from a copying one in most cells, so no statement about the
model is available at this (k, h). It is a finding rather than a failure --
every candidate change to the objective would be graded by the same probe --
and it redirects the next milestone to the readout rather than to training.
"""

EXIT_NO_MAJORITY: int = 48
"""RETURNED BY THE READ PHASE when no placement holds a strict majority.

Distinct from 47: there, most cells agreed that nothing is readable; here the
cells disagree about where the map sits. It has TWO causes -- no placement
reached the bar, or one did and its cells spanned fewer than `ARMS_REQUIRED`
arms -- and the `rule` line is what says which.
"""

READ_EXITS: dict[str, int] = {
    "UNREADABLE": EXIT_UNREADABLE_HEADROOM,
    NO_MAJORITY: EXIT_NO_MAJORITY,
}
"""Keyed on exactly the statuses that are not a placement of the model. The
four model placements -- BETWEEN, AT_PERFECT, AT_COPYING, AMBIGUOUS -- exit 0:
each is an answer, and AMBIGUOUS is an answer about the ruler at a readable
cell, which is a different claim from 47's about the instrument.
"""

PHASES: tuple[str, ...] = ("all", "measure", "read")
"""The phases this script runs. `all` is `measure` and then `read`.

THE DEFAULT IS THE LAST, `read`: it spends no GPU time, and without records it
refuses by name (11). A default of `measure` would make a bare invocation pay for
nine cells."""

ZERO_CONTROLS: tuple[str, ...] = (
    "persistence_divergence", "k_invariance_at_h1", "open_loop_divergence",
    "floor_divergence",
)
"""The controls whose known answer is EXACTLY 0.0. Each is a max of absolute
values, and the first two propagate NaN through `np.max` and `np.ptp` by design,
so a bad row surfaces as NaN rather than as a large number."""


def headroom_record_path(out: Path, arm: str, seed: int) -> Path:
    """One file per CELL. The seed is in the name, not only in the payload.

    M3m's fixtures all used seed 0, which hid the collision this prevents: in
    the real run, three seeds of an arm would have been written to one
    filename, destroying 6 of 9 records and surfacing only at the read phase
    after the GPU time had been spent.
    """
    return Path(out) / f"headroom_{arm}_seed{seed}.json"


def cell_bootstrap_seed(base_seed: int, arm: str, seed: int) -> int:
    """The seed one cell's bootstrap intervals are drawn at: a pure function of
    the run's base seed and the cell's identity.

    WHY IT EXISTS. The verdict is a strict majority across nine cells, and a
    majority count reads the cells as independent. Drawn at ONE shared seed,
    every cell and every `(k, h)` within it resamples the same 2000 draws of the
    same 24 episodes, so their resampling noise is perfectly correlated: a draw
    that happens to favour one side tilts all nine the same way, and the count
    reads one coin flip nine times. `pooling.clustered_interval` documents this
    hazard against a DEFAULTED seed, and one explicit seed for all nine is the
    same failure with the default spelled out. The point estimate is seed-free,
    so only the bounds move, and a reader cannot tell nine identical draws from
    nine independent ones by looking.

    Within a cell every `(k, h)` keeps that cell's ONE seed. The independence
    that matters is between cells, and the cell's own intervals are then
    reproducible from the record alone: `bootstrap_seed` is the run's base,
    `cell_bootstrap_seed` is what `clustered_interval` was handed.

    WHY NOT `hash()`. Python salts `hash(str)` per process (`PYTHONHASHSEED`), so
    the same cell would draw different resamples on every run and no record would
    reproduce. `zlib.crc32` over the UTF-8 bytes of `"base:arm:seed"` is the same
    unsigned 32-bit integer on every process and platform, and `numpy`'s
    `default_rng` mixes it through `SeedSequence`, so adjacent integers still
    start unrelated streams. Collisions among nine cells are possible in
    principle at 32 bits; a test pins that none occurs at the shipped base.
    """
    return zlib.crc32(f"{int(base_seed)}:{arm}:{int(seed)}".encode("utf-8"))


def broken_controls(controls: dict) -> list[str]:
    """The controls that missed their known answer, each as `name=value`.

    EVERY ZERO CONTROL IS READ WITH `!= 0.0`, NEVER `> 0`. `persistence_divergence`
    and `k_invariance_at_h1` propagate NaN by design, and `nan > 0` is False: a
    control checked with `>` reports a broken measurement as passing, and the read
    goes on to print a verdict about a model from rows nobody could trust.

    THE IDENTITY RESIDUAL IS HELD TO `IDENTITY_TOLERANCE`, not to zero -- it is
    rounding noise, 1.421e-14 on the shipped curves -- and is written `not
    residual <= tolerance` for the same reason: `nan > tolerance` is also False.
    """
    broken = [
        f"{name}={controls[name]!r}" for name in ZERO_CONTROLS if controls[name] != 0.0
    ]
    residual = controls["identity_residual"]
    if not residual <= IDENTITY_TOLERANCE:
        broken.append(f"identity_residual={residual!r}")
    return broken


# ---------------------------------------------------------------------------
# measure: one cell, then every cell.
# ---------------------------------------------------------------------------


def _cell_args(args, source: Path) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
    args, and its `out` is the STUDY directory the checkpoint is loaded from
    (`trust_horizon.load_checkpoint_model(args.out, ...)`). This script's own
    `--out` holds the headroom records and nothing else, so handing `args`
    through unchanged would look for the nine M3c checkpoints in the output
    directory and refuse EVERY cell -- a real run blocker on M3j, fixed with
    this same shim. Same shim, same reason, as `scripts/prediction_burden.py`.
    """
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context,
        horizon=args.horizon,
    )


def _require_a_readable_protocol(arm: str, seed: int, horizon: int, ks) -> None:
    """The verdict is read at `(DECISION_K, DECISION_H)`, every interval at
    `REPORTED_H`, and the sweep's own self-check needs the k == horizon rung.

    Refused by name BEFORE any pass is paid for: asked afterwards, the first is an
    IndexError out of the 45-step interval, the second a KeyError out of the
    decision cell, and the third a ValueError out of the sweep.
    """
    if horizon < max(DECISION_H, *REPORTED_H):
        raise SystemExit(
            f"{arm} seed {seed}: a rollout horizon of {horizon} cannot be read at "
            f"DECISION_H={DECISION_H} and REPORTED_H up to {max(REPORTED_H)}"
        )
    if DECISION_K not in tuple(ks):
        raise SystemExit(
            f"{arm} seed {seed}: ks={tuple(ks)!r} omits k={DECISION_K}, the rung "
            "the verdict is read at"
        )
    if horizon not in tuple(ks):
        raise SystemExit(
            f"{arm} seed {seed}: ks={tuple(ks)!r} omits the horizon {horizon}, and "
            "the k == horizon pass is the self-check that the sweep reproduces "
            "`evaluate_rollout`"
        )


def measure_cell(
    model, val_paths, probe, *, reference: RolloutResult, arm: str, seed: int,
    context: int, horizon: int, ks, device, feature_backbone, study_record: dict,
    bootstrap_seed: int,
) -> dict:
    """One cell: one sweep, then every interval and control from its rows.

    ONE TRAVERSAL. The sweep computes the rungs, the shared floor and the hold
    baseline on the same windows in the same order and carries the episode labels
    too, so no alignment between two passes is needed and none is assumed.

    `reference` IS `Prepared.reference`, THE ROLLOUT `prepare_cell` VERIFIED. Its
    three position curves are written into the record, and the two reference
    controls are read against it: `open_loop_divergence` (the k=horizon rung) and
    `floor_divergence` (the sweep's own floor). Together they are ALSO the
    `keep_trajectories=True` control -- the verified reference was measured with
    the flag OFF, so a bitwise match proves the flag moved neither the arms nor
    the reference. A second traversal would double the run to establish what these
    two establish for free.

    `probe` IS THE ONE `fit_probes` returned, and `reference` was read through that
    same probe, so the floor, the hold and the rungs are all read through one
    embedding probe.

    `bootstrap_seed` IS THE RUN'S BASE, NOT THE SEED THE INTERVALS ARE DRAWN AT.
    Every interval of this cell, at every `(k, h)`, is drawn at
    `cell_bootstrap_seed(bootstrap_seed, arm, seed)`, so the nine cells resample
    independent episode draws; the record carries both, and the base is the one
    `_PROTOCOL_FIELDS` compares across records.

    THE THREE DIFFERENCES GO THROUGH `eval.headroom`'s OWN FUNCTIONS, which hold the
    sign convention and the shape and finiteness guard in one place. The guard
    raises a bare ValueError that names neither the cell nor the array, so the
    rows are refused BY NAME here first.

    THE IDENTITY IS TAKEN ON THE PER-WINDOW ROWS, which is strictly stronger than on
    the mean curves: a compensating pair of errors that cancels in the mean does
    not cancel window by window. It is the MAX over the ks (`np.max`, which
    propagates NaN, where Python's `max` skips it according to its position).

    `share` IS NULL, never a number, where `headroom`'s interval is not above zero.
    `headroom` crosses zero inside the reported grid on a real cell --
    pixel_ae_seed1 reads 0.575649 at h=1 and -0.641206 at h=2 -- and the ratio
    there prints +339.5% at h=2 then -335.7% at h=3.

    `study_record` is the cell's own study record, read STRICTLY: `steps`,
    `kl_rate_above_free_bits`, `kl_dyn_max` and `git_sha` are carried from the
    training history beside the checkpoint, and a record without them is not a
    study record. `git_sha` is as strict as the other three because the read phase
    compares it across records as a protocol field.

    Keys of every per-`k` and per-`h` mapping are STRINGS, the form JSON gives
    them back in, so the record is identical before and after it is written.
    """
    _require_a_readable_protocol(arm, seed, horizon, ks)
    sweep = regrounding_sweep(
        model, val_paths, probe, ks=ks, context=context, horizon=horizon,
        seed=seed, device=device, feature_backbone=feature_backbone,
    )
    groups = np.asarray(sweep.window_episode)
    floor_rows = sweep.window_floor_position
    cell_seed = cell_bootstrap_seed(bootstrap_seed, arm, seed)

    named = [("the floor", floor_rows)]
    for k in ks:
        named += [
            (f"the k={k} hold", sweep.window_hold_position[k]),
            (f"the k={k} rung", sweep.window_position[k]),
        ]
    bad = [
        name for name, array in named if not np.isfinite(np.asarray(array)).all()
    ]
    if bad:
        raise SystemExit(
            f"{arm} seed {seed}: {', '.join(bad)} carry non-finite values, and every "
            "interval and control read from them would carry them too"
        )

    intervals: dict = {}
    share: dict = {}
    secondary: dict = {}
    negative: list = []
    residuals: list = []
    for k in ks:
        hold_rows = sweep.window_hold_position[k]
        rung_rows = sweep.window_position[k]
        rows = {
            "headroom": headroom(hold_rows, floor_rows),
            "skill": skill(hold_rows, rung_rows),
            "deficit": deficit(rung_rows, floor_rows),
        }
        residuals.append(triple_residual(hold_rows, rung_rows, floor_rows))
        secondary[k] = {
            name: (SECONDARY_SIGMAS * RegroundingSweep.standard_error(array)).tolist()
            for name, array in rows.items()
        }
        intervals[k] = {}
        share[k] = {}
        for h in REPORTED_H:
            cell: dict = {}
            for name, array in rows.items():
                point, low, high = clustered_interval(
                    array, groups, h=h, resamples=RESAMPLES, seed=cell_seed,
                )
                cell[name] = {"point": point, "ci_low": low, "ci_high": high}
            intervals[k][h] = cell
            hd = cell["headroom"]
            share[k][h] = (
                cell["skill"]["point"] / hd["point"] if hd["ci_low"] > 0.0 else None
            )
            if hd["point"] < 0.0:
                negative.append([k, h])

    return {
        "arm": arm,
        "seed": seed,
        "step": int(study_record["steps"]),
        "git_sha": git_sha(),
        "record_git_sha": study_record["git_sha"],
        "torch_version": torch.__version__,
        "device": str(device),
        "context": int(context),
        "horizon": int(horizon),
        "ks": [int(k) for k in ks],
        "reported_h": list(REPORTED_H),
        "decision_k": DECISION_K,
        "decision_h": DECISION_H,
        "confidence": CONFIDENCE,
        "resamples": RESAMPLES,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "secondary_sigmas": SECONDARY_SIGMAS,
        "bootstrap_seed": bootstrap_seed,
        "cell_bootstrap_seed": cell_seed,
        "split_seed": SPLIT_SEED,
        "displacement_median": DISPLACEMENT_RECORDED_ONLY,
        "episodes": {"val": [Path(p).name for p in val_paths]},
        "windows": {"total": int(sweep.windows_total), "episode": groups.tolist()},
        "curves": {
            "floor_position": reference.floor_position.tolist(),
            "persistence_position": reference.persistence_position.tolist(),
            "rssm_position": reference.rssm_position.tolist(),
            "rungs": {str(k): sweep.curve(k).tolist() for k in ks},
            "holds": {str(k): sweep.hold_position[k].tolist() for k in ks},
        },
        "intervals": {
            str(k): {str(h): v for h, v in by_h.items()} for k, by_h in intervals.items()
        },
        "share": {
            str(k): {str(h): v for h, v in by_h.items()} for k, by_h in share.items()
        },
        "secondary": {"iid_2se": {str(k): v for k, v in secondary.items()}},
        "controls": {
            "persistence_divergence": sweep.persistence_divergence(),
            "k_invariance_at_h1": sweep.k_invariance_at_h1(),
            "open_loop_divergence": float(sweep.open_loop_divergence(reference)),
            "floor_divergence": float(np.max(np.abs(
                sweep.reference.floor_position - reference.floor_position
            ))),
            "identity_residual": float(np.max(residuals)),
            "negative_headroom_steps": negative,
        },
        "kl_dyn_max": float(study_record["kl_dyn_max"]),
        "kl_rate_above_free_bits": float(study_record["kl_rate_above_free_bits"]),
    }


def _triple(interval: dict) -> str:
    return f"{interval['point']:+8.3f} [{interval['ci_low']:+8.3f},{interval['ci_high']:+8.3f}]"


def _cell_line(record: dict, path: Path) -> str:
    """One line per measured cell: the three differences at the decision cell with
    their intervals, the identity residual, and whether every control held.

    The controls are RECORDED here, not judged -- but a missed one is NAMED in the
    line, because the measure phase carries on and this is the only place an
    operator sees it before the read refuses it."""
    k, h = str(record["decision_k"]), str(record["decision_h"])
    at = record["intervals"][k][h]
    controls = record["controls"]
    broken = broken_controls(controls)
    held = "controls hold" if not broken else "CONTROLS BROKEN: " + ", ".join(broken)
    return (
        f"{record['arm']} seed {record['seed']}: at k={k}, h={h} "
        + "; ".join(f"{name} {_triple(at[name])}" for name in ("headroom", "skill", "deficit"))
        + f"; identity {controls['identity_residual']:.2e}; {held}; "
        f"{record['windows']['total']} windows from "
        f"{len(set(record['windows']['episode']))} episodes; wrote {path}"
    )


def measure_phase(args) -> int:
    """Every requested cell: `prepare_cell`'s refusals, then `measure_cell`, then
    one record per cell through `write_record`.

    EVERY CELL IS LOADED BEFORE ANY IS MEASURED, so a missing ninth cell is found
    before eight have been paid for (11). Then the split -- the study's own
    `VAL_FRACTION` at `SPLIT_SEED`, which is fixed and deliberately NOT the cell's
    seed -- and, per cell, `prepare_cell` (12, 14), the measurement, and the write.
    THE FIRST NON-`EXIT_OK` STATUS IS RETURNED and nothing is written for the cell
    that earned it: dropping it would let `--phase measure` exit 0 after a refused
    cell, the worst available outcome of a run that costs GPU time, because the
    operator reads success and the records are not there.

    AN EXCEPTION FROM A CELL IS NOT CAUGHT. It reaches the caller as a traceback,
    the cells before it keep their records and the cells after it are not measured.

    CELLS ARE LOADED FROM `args.source`, NOT `args.out`. `--out` is this script's
    own record directory and holds none of the nine checkpoints.

    A MISSED CONTROL DOES NOT STOP THE RUN. It is recorded and named in the cell's
    line, and the read phase refuses it.
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
            ks=args.ks, device=device,
            feature_backbone=prepared.common["feature_backbone"],
            study_record=cell.record, bootstrap_seed=args.bootstrap_seed,
        )
        path = headroom_record_path(args.out, cell.arm, cell.seed)
        write_record(path, record)
        print(_cell_line(record, path))
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: what pooling these records requires, and the reading.
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
    ("reported_h", lambda r: [int(h) for h in r["reported_h"]]),
    ("decision_k", lambda r: int(r["decision_k"])),
    ("decision_h", lambda r: int(r["decision_h"])),
    ("confidence", lambda r: float(r["confidence"])),
    ("resamples", lambda r: int(r["resamples"])),
    ("identity_tolerance", lambda r: float(r["identity_tolerance"])),
    ("secondary_sigmas", lambda r: int(r["secondary_sigmas"])),
    ("bootstrap_seed", lambda r: int(r["bootstrap_seed"])),
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
windows, on which split; `ks`, `reported_h`, `decision_k` and `decision_h` say what
was read off them; `confidence`, `resamples`, `secondary_sigmas`,
`identity_tolerance` and `bootstrap_seed` (the run's BASE; each cell's own
`cell_bootstrap_seed` differs between cells by construction and is not compared)
are what each interval, figure and control was taken at. All are written from the measure phase's own constants and
arguments, so two records of one run cannot disagree on them and a disagreement is
a mixed pool. `episodes.val` and `windows.episode` say the rows were the same: the
cells pooled into one reading must describe the same windows.

NOT COMPARED, on purpose: `kl_rate_above_free_bits`, `kl_dyn_max` and
`record_git_sha`, which are properties of each CELL'S TRAINING and differ between
cells by construction; and `displacement_median`, which is recorded and used by no
statistic.

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
            f"{cell[0]} seed {cell[1]} lacks {field}: it is not a headroom record "
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

    A value that fits on a line is printed as it is. Two long lists -- the window
    labels of `windows.episode`, the episode names of `episodes.val` -- are the case
    a bare truncation fails: they are cut at the same character and read as one
    text, however they differ. So they are told apart by LENGTH when the lengths
    differ and by the FIRST INDEX where they differ when they do not, with the two
    values found there.

    THERE IS NO FALLBACK, because no `_PROTOCOL_FIELDS` pick reaches one: the only
    values longer than a line are those two lists, and the caller compares only
    values that differ, so two long lists of one length differ at SOME index. A
    value outside that set raises (`TypeError`, `StopIteration`) instead of being
    printed as two truncations that read as one."""
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

    EVERY record is compared with the first, not only the second, so a record that
    sorts last is held to the same value as one that sorts second."""
    items = sorted(records.items())
    if not items:
        raise SystemExit("no headroom record to read")
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

    Cells pooled into one reading must describe the same rows: two protocols pooled
    as one would be a reading over a union nothing measured. Mirrors
    `scripts/prediction_burden.py`'s namesake, and `git_sha` is among the fields
    for the reason `_PROTOCOL_FIELDS` gives."""
    _require_agreement(records, [name for name, _ in _PROTOCOL_FIELDS])


def require_sound_records(records: dict) -> None:
    """Refuse a record whose measurement is not to be trusted, by name and before
    any statistic is read from it.

    A control that missed its known answer means the measurement is wrong and no
    statistic in the record may be reported. It is asked FIRST, so a NaN control is
    named as the control it is. A record that carries ANY non-finite value -- the
    sanitiser's `nonfinite` map is not empty -- is not one `measure_cell` wrote (it
    refuses a non-finite row before it writes), and is refused for whichever
    value it is. Both are refused BY NAME, with no number of their own: 47 and 48
    are findings about the data, and neither is one."""
    for cell, record in sorted(records.items()):
        broken = _pick(cell, "controls", lambda r: broken_controls(r["controls"]), record)
        if broken:
            raise SystemExit(
                f"{cell[0]} seed {cell[1]}: control(s) missed their known answer -- "
                f"{', '.join(broken)}. The measurement is wrong, and no statistic in "
                "this record may be reported"
            )
        found = _pick(cell, "nonfinite", lambda r: dict(r["nonfinite"]), record)
        if found:
            raise SystemExit(
                f"{cell[0]} seed {cell[1]} carries non-finite value(s) at "
                f"{', '.join(sorted(found))}, which `measure_cell` refuses to write; "
                "this is not a record it wrote"
            )


def require_readable_plan(arms) -> None:
    """Refuse a plan with fewer than `ARMS_REQUIRED` DISTINCT arms.

    A verdict needs `ARMS_REQUIRED` arms, and `reading_headroom` raises for fewer.
    Asked here it costs nothing and is a refusal by name; asked after `--phase all`
    it is nine cells of GPU time. One arm is the shape M3j's `--arms random_vit`
    built.

    THE SEED COUNT IS NOT CHECKED HERE, deliberately: an arm short of
    `SEEDS_MINIMUM` seeds is `reading_headroom`'s own refusal, and a plan check on
    seeds would stand in front of it. `--phase all` asks the seed question
    separately, in `main`, before it measures (`require_seeds_for_a_verdict`).

    The caller passes the DISTINCT arms (`_plan`): `--arms a a a` is one arm."""
    if len(arms) < ARMS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(arms)} arm(s) cannot be read: a verdict needs "
            f"ARMS_REQUIRED={ARMS_REQUIRED} arms, since one arm's finding is not a "
            "finding about the model"
        )


def require_seeds_for_a_verdict(seeds) -> None:
    """Refuse a plan with fewer than `SEEDS_MINIMUM` DISTINCT seeds.

    CALLED BY `main` FOR `--phase all` ONLY, never by `read_phase`. Every arm of a
    plan carries the plan's seeds, so a plan short of the minimum leaves every arm
    short of it and `reading_headroom` can only raise. Asked after `--phase all`
    that is every planned cell of GPU time spent to learn something the command
    line already said; asked here it costs nothing.

    It is not in `read_phase` because a `read` over a short plan must REACH
    `reading_headroom`, whose `ValueError` names the arm and its seeds; and a
    `--phase measure` over a short plan is the milestone's smoke.

    The caller passes the DISTINCT seeds (`_plan`): `--seeds 0 0 0` is one seed."""
    if len(seeds) < SEEDS_MINIMUM:
        raise SystemExit(
            f"a plan of {len(seeds)} seed(s) cannot reach a verdict: every arm would "
            f"carry fewer than SEEDS_MINIMUM={SEEDS_MINIMUM} seeds, and "
            "`reading_headroom` refuses such an arm once the cells are already "
            f"measured. Pass at least {SEEDS_MINIMUM} distinct seeds, or run "
            "--phase measure for a smoke"
        )


def load_headroom(out_dir: Path, arms, seeds) -> dict:
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
            path = headroom_record_path(out_dir, arm, int(seed))
            if not path.exists():
                raise CellMissing(
                    f"{arm} seed {int(seed)}: no headroom record at {path}; "
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


def _require_finite_intervals(cell, record: dict) -> None:
    """Refuse a record whose interval values are not ALL finite, naming the one.

    `Interval.resolvably_positive` is `ci_low > 0.0`, which is False for NaN, so a
    NaN deficit beside a resolvable headroom and skill reads `AT_PERFECT` -- a
    verdict about a model derived from a broken number. `clustered_interval`
    refuses non-finite rows, so this should be unreachable; this makes it provably
    so, rather than leaving it to that refusal. Every cell of the table, not only
    the decision cell: a NaN anywhere is a record this script did not write."""
    try:
        table = record["intervals"]
        for k, by_h in table.items():
            for h, entries in by_h.items():
                for name, interval in entries.items():
                    for field, value in interval.items():
                        if not (isinstance(value, (int, float)) and math.isfinite(value)):
                            raise SystemExit(
                                f"{cell[0]} seed {cell[1]}: intervals[{k}][{h}]."
                                f"{name}.{field} is {value!r}, not a finite number; "
                                "refusing a record that could hand a NaN to "
                                "`reading_headroom`"
                            )
    except (KeyError, TypeError, AttributeError) as error:
        raise SystemExit(
            f"{cell[0]} seed {cell[1]} lacks a readable intervals table: it is not a "
            "headroom record this script wrote"
        ) from error


def headroom_inputs(records: dict) -> HeadroomInputs:
    """The pooled records as `HeadroomInputs`, every cell at ONE (k, h).

    REFUSES a null `windows.episode` rather than falling back to `arange(n)`. A
    record with no clustering must stop the read; treating every window as its own
    episode converts the verdict's ruler into the window-level bootstrap
    `pooling.clustered_interval`'s first paragraph rules out, and it would do so
    silently, printing a narrower interval and a more confident verdict.

    REFUSES a record whose interval values are not all finite
    (`_require_finite_intervals`).

    REFUSES records that disagree on the decision cell or the ladder, HERE, at the
    point the pick is made, rather than depending on `require_one_protocol` having
    run first: a first-record pick would otherwise read every other cell at a
    (k, h) its own record does not name. A table whose rows were read at different
    (k, h) is not a table.

    `clusters` is the number of DISTINCT episodes the windows were cut from. It is
    carried for the reader of the table and read by no statistic."""
    items = _require_agreement(records, ("decision_k", "decision_h", "ks"))
    decision_k = int(items[0][1]["decision_k"])
    decision_h = int(items[0][1]["decision_h"])
    ks = tuple(int(k) for k in items[0][1]["ks"])
    cells = {}
    for (arm, seed), record in items:
        labels = record["windows"]["episode"]
        if labels is None:
            raise SystemExit(
                f"{arm} seed {seed} carries no episode labels, so no episode-clustered "
                "interval can be drawn from it; refusing rather than clustering by window"
            )
        _require_finite_intervals((arm, seed), record)
        try:
            at = record["intervals"][str(decision_k)][str(decision_h)]
            drawn = {
                name: Interval(
                    point=float(at[name]["point"]), ci_low=float(at[name]["ci_low"]),
                    ci_high=float(at[name]["ci_high"]),
                )
                for name in ("headroom", "skill", "deficit")
            }
        except (KeyError, TypeError) as error:
            raise SystemExit(
                f"{arm} seed {seed} lacks intervals[{decision_k}][{decision_h}]: it is "
                "not a headroom record this script wrote"
            ) from error
        cells[(arm, seed)] = HeadroomCell(
            arm=arm, seed=seed, headroom=drawn["headroom"], skill=drawn["skill"],
            deficit=drawn["deficit"], clusters=len(set(labels)),
        )
    return HeadroomInputs(
        cells=cells, decision_k=decision_k, decision_h=decision_h, ks=ks,
    )


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
    first missing), refuse records that are not one measurement or not sound,
    place the one-step map, and return `READ_EXITS.get(reading.verdict, EXIT_OK)`:
    0 for a model placement, 47 or 48 for the two verdicts that are not one.

    THE READING IS PRINTED AND WRITTEN AS ONE STRING, for every verdict, 47 and 48
    included: they are findings, and the table is what a reader needs to see them.
    The text carries the verdict AND THE RULE -- the rule is the only line that
    tells the two causes of `NO_MAJORITY` apart. `format_reading_headroom` already
    ends its text in a newline and this function prints it with `end=""`;
    `print(text)` would put a second newline on stdout and none in the file. The
    text is written as it is, never `rstrip()`ped.

    A REFUSAL WRITES NOTHING, AND LEAVES NOTHING. The `headroom.txt` an EARLIER
    read of the same `--out` left is removed FIRST, before any refusal is reached:
    left in place it would sit beside a refusal, and a later reader could not tell a
    stale reading from a current one. It is removed before the plan is even
    checked, so that every way out of this function that does not reach the write
    -- a returned 11, a named `SystemExit`, an exception from `reading_headroom` --
    leaves the directory without a reading the current invocation did not produce.
    Only `<--out>/headroom.txt` is touched.

    NOTHING CATCHES `reading_headroom` HERE, and that is deliberate. It raises
    `ValueError` for an arm short of `SEEDS_MINIMUM` seeds and for fewer than
    `ARMS_REQUIRED` arms; neither is something found in the data, and a `try` would
    turn a malformed plan into an exit number. The second is unreachable from here
    (`require_readable_plan` refuses first); the first is `--phase read --seeds 0 1`."""
    reading_path = Path(args.out) / "headroom.txt"
    reading_path.unlink(missing_ok=True)
    arms, seeds = _plan(args)
    require_readable_plan(arms)
    try:
        records = load_headroom(args.out, arms, seeds)
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    require_one_protocol(records)
    require_sound_records(records)
    inputs = headroom_inputs(records)
    reading = reading_headroom(inputs)
    text = format_reading_headroom(reading, inputs)
    reading_path.write_text(text)
    print(text, end="")
    return READ_EXITS.get(reading.verdict, EXIT_OK)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class _Csv(argparse.Action):
    """A list flag that reads commas and spaces alike: `--seeds 0,1,2` and
    `--seeds 0 1 2` are one plan, and so is `--seeds 0,1 2`.

    The run's own command line passes comma-joined lists; `prediction_burden.py`
    takes space-separated ones. Taken as one token, `"0,1,2"` is not an integer
    and `"frozen_ssl,pixel_ae,random_vit"` is not an arm, so a parser that took
    only one spelling would refuse the command after the checkpoints were loaded.
    Repeats are dropped, in the order given."""

    cast = staticmethod(str)

    def check(self, parser, option, items) -> None:
        """Hook for a subclass that restricts the values."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            items = [
                self.cast(part) for token in values for part in token.split(",") if part
            ]
        except ValueError:
            parser.error(f"{option_string}: cannot read {' '.join(values)!r}")
        if not items:
            parser.error(f"{option_string}: no value given")
        items = list(dict.fromkeys(items))
        self.check(parser, option_string, items)
        setattr(namespace, self.dest, items)


class _Arms(_Csv):
    def check(self, parser, option, items) -> None:
        unknown = [arm for arm in items if arm not in ARMS]
        if unknown:
            parser.error(f"{option}: {unknown!r} are not among the arms {list(ARMS)!r}")


class _Ints(_Csv):
    cast = staticmethod(int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M3n: is the one-step map closer to copying, or to perfect? "
            "-- measure the cells, read the placement"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/m3n_motion"))
    parser.add_argument("--source", type=Path, default=Path("runs/m3_study_v2"),
                        help="the M3c study directory: the nine 20,000-step cells")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--device", default="mps")
    # None -> the cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", action=_Arms, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", action=_Ints, default=list(SEEDS))
    parser.add_argument("--ks", nargs="+", action=_Ints, default=list(REGROUNDING_KS),
                        help="the re-grounding periods; must include 1 and the horizon")
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                        help="the run's base seed; each cell draws its intervals at "
                             "cell_bootstrap_seed(base, arm, seed), and both are "
                             "recorded in the cell")
    parser.add_argument("--phase", choices=PHASES, default=PHASES[-1])
    return parser


def main(argv: list[str] | None = None) -> int:
    """`measure`, `read`, or `all` -- the measure and then the read, which does
    not run when the measure stopped with a status: a cell that was refused wrote
    no record, and a read after it would be over a pool missing that cell.

    `--phase all` REFUSES A PLAN `read` COULD NOT TAKE A READING FROM before any
    cell is measured: fewer than `ARMS_REQUIRED` arms, or fewer than
    `SEEDS_MINIMUM` seeds. `--phase measure` is allowed either, because the
    milestone's smoke is one cell; `--phase read` is allowed a short SEED plan,
    because that is where `reading_headroom`'s ValueError names the short arm."""
    args = _parser().parse_args(argv)
    if args.phase == "all":
        arms, seeds = _plan(args)
        require_readable_plan(arms)
        require_seeds_for_a_verdict(seeds)
    if args.phase in ("measure", "all"):
        status = measure_phase(args)
        if status != EXIT_OK:
            return status
    if args.phase in ("read", "all"):
        return read_phase(args)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
