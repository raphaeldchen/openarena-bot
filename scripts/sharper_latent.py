"""The sharper latent over the M3c cells: sweep the rollout's sampling temperature, and -- only if the rollout is noise-limited -- retrain the nine cells at the temperature the sweep names.

M3g localised the M3 failure to the prior: from the true state, one prior step
predicts the next posterior's mode less often than "the mode did not change"
does. Its own disclosure, and the measurements in spec 2026-09-23 M3h section
1, say that failure is a near-tie inside a much larger problem -- the
posterior carries 0.71-0.87 nats per group over 32 groups, so every imagined
step injects ~23-28 nats of fresh entropy against ~0.4-0.6 nats of dynamics
information, and from h = 5 onward two draws of the SAME model from the SAME
state separate by more than that model's entire imagined displacement. So:

  sweep     re-run the canonical pass on each shipped cell at every rollout
            temperature of TAU_GRID, sharpening the prior inside `imagine`
            and nothing else. tau = 1.0 runs first and must reproduce the
            cell's diagnostic bitwise, or nothing below it is read.
  train     read the sweep's records, decide Reading N, refuse (35) unless it
            says NOISE_LIMITED, check the tau = 1.0 identity (37), then
            retrain nine cells at tau*.
  evaluate  score each retrained cell through the study's own evaluation half
            and the trust pass, rung-style -- a retrained cell has no prior
            diagnostic to self-check against, exactly as M3f's rungs had none.
  read      pool both phases, decide Reading M, print the tables, write
            sweep.txt / sharper.txt and the figures.

LOADING IS `trust_horizon.py`'S: `Cell`, `load_cell`, `self_check` and
`prepare_cell` are imported by path, so a cell is refused here for the reasons
and in the words the other tools refuse it. `prepare_cell` reads the
temperature off the args it is handed (M3h), so the model it returns samples
the way the checkpoint it loaded was trained to.

THE CHECKS, BY PHASE, each with its own status:

  sweep:    EXIT_NO_CHECKPOINTS (11)     a requested cell lacks its checkpoint,
                                          record or diagnostic; judged for every
                                          cell before any pass runs.
            EXIT_SPLIT_MISMATCH (12)      the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)     --context/--horizon disagree with the
                                          protocol, or evaluate_rollout no longer
                                          reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30)   the tau = 1.0 pass is not bitwise the
                                          cell's diagnostic -- judged BEFORE any
                                          other temperature of that cell runs.
  train:    EXIT_NO_CHECKPOINTS (11)     a sweep record is missing.
            EXIT_NOT_NOISE_LIMITED (35)   NEW. Reading N does not say the rollout
                                          is noise-limited: the retrain is gated
                                          in code, not by a person reading a table.
            EXIT_IDENTITY_CHECK_FAILED (37) NEW. the tau = 1.0 retrain's losses are
                                          not the M3c record's prefix exactly.
  evaluate: EXIT_NO_CHECKPOINTS (11)     a retrained checkpoint is missing.
            EXIT_TEMPERATURE_MISMATCH (36) NEW. a checkpoint's recorded temperature
                                          is not the one it is being evaluated at.
  read:     EXIT_NO_CHECKPOINTS (11)     a requested record is missing.
            EXIT_SELF_CHECK_FAILED (30)   a sweep record's self-check is not ok.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 35, 36 and
37 are in no other tool's range (run_study 1/3-6/23, report_study 7-10, spike
10, diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages
34, argparse 2, a traceback 1).

TWO CORRECTIONS TO THE M3H TASK-6 BRIEF:

  * `tau_key` does not format as `f"{tau:.3f}"` (e.g. "1.000"): that string
    contains a '.', and `study.write_record` refuses a record key that does
    -- the non-finite map addresses fields by dotted path, so a dotted key
    would alias a real nested field on `load_record` (`split_gap.q_key`'s
    docstring states the identical rule, for the same reason). The key is
    `q_key`'s own idiom instead, `0.7 -> "tau70"`, and `tau_value` is its
    inverse.

The brief also assumed `Trajectories` carried the pass's noise reference. It
did not, and reading it from a second `_diagnose` call would have doubled
every temperature's rollout compute; `diagnostics.Trajectories` now carries
`noise_embedding` (M3h), which is where this script reads it.

FOUR CORRECTIONS TO THE M3H TASK-7 BRIEF, found running its own tests:

  * `history_from_record` is NOT `checkpoint_ladder.history_from_train_record`
    bound under a new name. That function reads `kl_dyn_max`,
    `kl_rate_above_free_bits` and `checkpoint_seconds` off its OWN script's
    `train_record`'s top level, where they live because that record is built
    straight from a live `train_world_model` history that still carries them.
    `train_cell`'s retrain record below carries none of the three (`history`
    holds only `loss` and `parts`, `study.history_record`'s policy), so
    binding the ladder's function raises `KeyError('kl_dyn_max')` the first
    time `evaluate_cell` reads a retrain record back. `history_from_record`
    here is its own function, recomputing the two KL summaries from the
    stored per-step `parts` exactly as `train_world_model` computes them.
  * A retrained cell's protocol is not `--context 5 --horizon 45` by a bare
    literal default: the retrain record carries no context or horizon of its
    own, and the real nine cells' protocol is whatever the M3c reference was
    scored at, not the study's own signature default. `evaluate_cell` reads
    it off the M3c reference's own record (`job_record_path(args.reference,
    ...)`), the same None-means-the-cell's-own-value rule every other
    protocol flag in this file follows.
  * `train_phase` printed `format_reading_noise(reading, inputs)`
    unconditionally. That function indexes `reading.cells` by every
    `(arm, tau)` pair `inputs.arms` carries, which a REAL `reading_noise`
    result always populates -- but the gate's own tests fabricate the
    reading directly (deliberately, so the gate is tested and not the
    fixture's luck) and never populate `cells`, so the unconditional print
    raised `KeyError` before the gate's refusal ever printed. Guarded on
    `reading.cells` being non-empty; a real reading is unaffected.
  * `evaluate_cell` cannot hand `_trust.trust_record` the rung-style
    `cell.diagnostic` `rung_cell` builds (`{context, horizon}` only):
    `trust_record` unconditionally calls `self_check`, which needs a
    diagnostic's `curves.reference_position`, `curves.persistence_position`
    and `windows` -- fields a rung's reduced diagnostic never carries, so
    the call raised `KeyError('curves')`. A retrained cell has no earlier
    diagnose_dynamics.py pass to self-check against in the first place (the
    docstring already says so), so `evaluate_cell` builds the comparison
    from THIS SAME pass -- exact against itself by construction, the way a
    check with nothing earlier to reproduce should read -- and takes
    `curves.floor_position` / `probe.embedding_selection_r2` (which
    `self_check` never compares) from the study record `evaluate_job` just
    wrote.

ONE CORRECTION TO THE M3H TASK-8 BRIEF, found running its own tests:

  * The brief read Reading M's control arm, `old_trust`, as
    `_load_trust(args.reference, cells)` -- a `trust_<arm>_seed<n>.json` per
    M3c cell from an earlier `trust_horizon.py` pass over `--reference`. The
    shipped M3c directory does carry those (M3d's own `--out` was the same
    `runs/m3_study_v2`), but this file's fixtures never write them, so the
    tests would have exercised an empty control while the real run read the
    files: two paths, with the untested one shipping.

    The control is now ALWAYS `_reference_trust`, built from the sweep's own
    REFERENCE_TAU entry. It is also the better control of the two:
    `load_sweep` has proven that entry bitwise identical to the M3c
    diagnostic in THIS run (exit 30 otherwise), and it scores the same
    windows at the same protocol as the retrain it is compared against,
    rather than being a file another milestone happened to leave behind. No
    exit code, message or record key changed.
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
from mbfps.eval.sharper import (
    ARMS_REQUIRED,
    DECISION_H,
    IDENTITY_STEPS,
    REFERENCE_TAU,
    RETRAIN_FAMILY,
    SEEDS_REQUIRED,
    STEPS,
    SWEEP_FAMILY,
    TAU_CANDIDATES,
    TAU_GRID,
    RetrainArm,
    RetrainInputs,
    SweepArm,
    SweepInputs,
    SweepStatus,
    TauInputs,
    format_reading_noise,
    format_reading_sharper,
    reading_noise,
    reading_sharper,
)
from mbfps.eval.split_gap import StratumContrast, cell_series, decision_horizon, fmt_z, survival_indicator
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
from mbfps.eval.trust_readings import ARMS_ORDER
from mbfps.eval.ladder import anchor_delta
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_sharper_latent", path)
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

_ladder = _sibling("checkpoint_ladder")
rung_cell = _ladder.rung_cell

_diagnose = _sibling("diagnose_dynamics")

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_NOT_NOISE_LIMITED = 35
EXIT_TEMPERATURE_MISMATCH = 36
EXIT_IDENTITY_CHECK_FAILED = 37
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 35, 36 and 37 are new and in no other tool's range."""

_split = _sibling("split_gap")
stratum_summary = _split.stratum_summary

PHASES: tuple[str, ...] = ("sweep", "train", "evaluate", "read", "all")


# ---------------------------------------------------------------------------
# Paths and keys.
# ---------------------------------------------------------------------------


def sweep_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"sweep_{arm}_seed{seed}.json"


def tau_key(tau: float) -> str:
    """`0.7 -> "tau70"`, `split_gap.q_key`'s idiom: a record key may not
    contain '.', since `write_record` addresses non-finite fields by dotted
    path. Hundredths are the grid's own resolution, so 0.3 and 0.30 are one
    entry and a float's repr never becomes part of the record's shape."""
    return f"tau{int(round(float(tau) * 100))}"


def tau_value(key: str) -> float:
    """`"tau70" -> 0.7`: `tau_key`'s inverse, so a reader of a record can get
    back the temperature an entry was scored at without parsing the entry."""
    return int(key.removeprefix("tau")) / 100.0


# ---------------------------------------------------------------------------
# sweep: one cell at every temperature.
# ---------------------------------------------------------------------------


def _cell_args(args, source: Path, tau: float = REFERENCE_TAU) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context`, `horizon` and (M3h)
    `sample_temperature` off its args. The sweep asks for 1.0 -- it sharpens
    the ROLLOUT of a model trained at 1.0, not the model."""
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context, horizon=args.horizon,
        sample_temperature=float(tau),
    )


def tau_entry(traj, record: dict, horizon: int) -> dict:
    """One temperature of one cell: the gate's own metric off the pass's own
    band (not the study record's -- every temperature has its own rollout),
    the probe's measurability, and the trust pass reduced by
    `stratum_summary`, the same reduction M3e and M3f read. The probe's
    selection R^2 is the STUDY RECORD's, which the reproduction discipline
    keeps equal to the probe `prepare_cell` refit, and is shared by every
    temperature because the probe does not depend on the rollout; its measurability is judged on THIS temperature's
    band, since the persistence-to-floor band is what makes the probe
    channel readable and the floor does not move but the model's curve does.
    `noise` is filled by the caller from the pass's noise reference."""
    summary = stratum_summary(traj, horizon)
    position = summary["band"]["position"]
    r2 = record["probe"]["embedding_selection_r2"]
    curves = summary["curves"]
    return {
        "gate": {
            "gap_final": position["gap_final"],
            "degenerate": position["steps_degenerate"],
        },
        "probe": {
            "selection_r2": float(r2) if r2 is not None else float("nan"),
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": float(curves["persistence_position"][-1]),
                    "floor": float(curves["floor_position"][-1]),
                },
                widest_se=0.0,
            )),
        },
        "noise": {"curve": None, "median": None},
        "displacement": {"curve": None, "median": None},
        "summary": summary,
    }


def sweep_cell(args, cell: Cell, device, train, val, taus) -> tuple[int, dict | None]:
    """One cell: 12/14 from `prepare_cell`, the reference temperature and its
    self-check (30) BEFORE any other temperature, then the rest of the grid.

    The reference is scored first because the caller's order says so, not
    because the pre-registered grid happens to list it first: a cell whose
    pass at the shipped temperature does not reproduce its diagnostic must
    cost one pass, not five, and on the real nine cells five passes is an
    hour. A grid without the reference temperature is refused before any of
    it runs, for the same reason -- every contrast is against it.
    """
    taus = [float(t) for t in taus]
    if REFERENCE_TAU not in taus:
        raise ValueError(
            f"the grid {taus} does not contain the reference temperature {REFERENCE_TAU}; "
            "every contrast is against it and the self-check is taken on it"
        )
    keys = [tau_key(t) for t in taus]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"the grid {taus} has two temperatures that spell the same record key "
            f"({keys}); one entry would overwrite the other and `taus` would list a "
            "temperature no entry was scored at"
        )
    status, prepared = prepare_cell(_cell_args(args, args.reference), cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    h, _ = decision_horizon(prepared.horizon)
    entries: dict[str, dict] = {}
    check = None
    for tau in [REFERENCE_TAU] + [t for t in taus if t != REFERENCE_TAU]:
        traj = reference_trajectories(
            prepared.model, val, prepared.embedding_probe,
            rollout_temperature=float(tau), **prepared.common,
        )
        if float(tau) == REFERENCE_TAU:
            check = self_check(traj, cell.diagnostic)
            if not check.ok:
                print(
                    f"\nSELF-CHECK FAILED for {cell.arm} seed {cell.seed} at tau="
                    f"{REFERENCE_TAU}: " + "; ".join(check.failures())
                    + ". The pass at the shipped temperature is not the one the gate scored, "
                    "so no sharper temperature of this cell is evaluated. No record written."
                )
                return EXIT_SELF_CHECK_FAILED, None
        entry = tau_entry(traj, cell.record, prepared.horizon)
        noise = np.asarray(traj.noise_embedding, dtype=float)
        entry["tau"] = float(tau)
        entry["noise"] = {
            "curve": noise.mean(axis=0),
            "median": float(np.median(noise)),
        }
        # The statistic the study turns on, beside the noise it is compared
        # with: how far the imagination moved in embedding space,
        # ||e_hat(h) - e_hat(0)||, per horizon step, median over windows.
        entry["displacement"] = {
            "curve": np.median(np.asarray(traj.embedding_displacement, dtype=float), axis=0),
            "median": float(np.median(traj.embedding_displacement)),
        }
        entries[tau_key(tau)] = entry
    record = {
        "arm": cell.arm,
        "seed": int(cell.seed),
        "source": str(args.reference),
        "step": int(cell.record["steps"]),
        "record_git_sha": cell.record.get("git_sha", "unknown"),
        "context": int(prepared.context),
        "horizon": int(prepared.horizon),
        "decision_h": int(h),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "taus": [float(t) for t in taus],
        "episodes": {"val": list(cell.record["episodes"]["val"])},
        "windows": {
            "total": int(entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["total"]),
            "episode": [
                int(e) for e in entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["episode"]
            ],
            "clusters": int(
                entries[tau_key(REFERENCE_TAU)]["summary"]["windows"]["clusters"]
            ),
        },
        "self_check": check.record(),
        "entries": entries,
    }
    path = sweep_record_path(args.sweep_out, cell.arm, cell.seed)
    write_record(path, record)
    gaps = " / ".join(
        f"tau {entries[tau_key(t)]['tau']:.1f}: gap {entries[tau_key(t)]['gate']['gap_final']:+.3f}"
        for t in taus
    )
    print(f"{cell.arm} seed {cell.seed}: {gaps}; wrote {path}")
    return EXIT_OK, record


def sweep_phase(args, cells, device, train, val, taus) -> int:
    try:
        loaded = [load_cell(args.reference, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.sweep_out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, _ = sweep_cell(args, cell, device, train, val, taus)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# The sweep's records -> Reading N's inputs.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s. What is decided here is only WHICH series
# goes in: the survival indicator over the windows that moved within the
# horizon, at one temperature against the same cell at REFERENCE_TAU. The two
# temperature groups of one cell are labelled as two "arms" so
# `paired_contrast` -- which refuses an arm against itself -- reads them as
# two groups on the same windows, exactly as M3f paired a rung against 20000.


def _entry(record: dict, tau: float) -> dict:
    return record["entries"][tau_key(tau)]


def _survival_series(summary: dict, h: int, channel: str, *, label: str, seed: int, val,
                      horizon: int, context: int, device: str, torch_version: str) -> pooling.CellSeries:
    """One per-window survival series at one channel, from anything shaped
    like a stratum summary or a trust record -- both carry `crossing` and
    `windows` the same way (see `retrain_inputs`'s docstring note). The one
    place `survival_indicator` + `cell_series` are wired together, so
    Reading N's free-channel series and Reading M's free/probe ones are the
    same call at a different channel, not two copies of it."""
    values, changed = survival_indicator(summary, channel, h)
    return cell_series(
        summary, values, changed, arm=label, seed=int(seed), rung="val", channel=f"S/{channel}",
        val=val, horizon=int(horizon), context=int(context), device=str(device),
        torch_version=str(torch_version),
    )


def _series(record: dict, tau: float, h: int, label: str, channel: str = "free") -> pooling.CellSeries:
    return _survival_series(
        _entry(record, tau)["summary"], h, channel, label=label, seed=record["seed"],
        val=record["episodes"]["val"], horizon=record["horizon"], context=record["context"],
        device=record["device"], torch_version=record["torch_version"],
    )


_NO_CONTRAST = StratumContrast(estimate=float("nan"), se=float("nan"), z=float("nan"), clusters=0)


def _paired(treatment, control) -> StratumContrast:
    """`pooling.paired_contrast`, or the NaN contrast when there is nothing to
    pair -- no cell, or no window every cell of both groups changed."""
    treatment, control = list(treatment), list(control)
    if not treatment or not control:
        return _NO_CONTRAST
    if not np.logical_and.reduce([c.changed for c in treatment + control]).any():
        return _NO_CONTRAST
    c = pooling.paired_contrast(treatment, control)
    return StratumContrast(estimate=c.mean, se=c.se, z=c.z, clusters=c.clusters)


def _tau_inputs(records: dict, arm: str, seeds, tau: float, h: int, channel: str = "free") -> TauInputs:
    def contrast(seed_list):
        kept = [records[(arm, int(s))] for s in seed_list]
        return _paired(
            [_series(r, tau, h, f"{arm}@{tau_key(tau)}", channel) for r in kept],
            [_series(r, REFERENCE_TAU, h, f"{arm}@{tau_key(REFERENCE_TAU)}", channel) for r in kept],
        )

    return TauInputs(
        free=contrast(seeds),
        per_seed={int(s): contrast([s]) for s in seeds},
    )


def _ordered(arms) -> list[str]:
    return sorted(arms, key=lambda a: ARMS_ORDER.index(a) if a in ARMS_ORDER else len(ARMS_ORDER))


def _clusters(records: dict) -> int:
    return int(next(iter(records.values()))["windows"]["clusters"])


def sweep_inputs(records: dict, *, arms, seeds, h: int, taus) -> SweepInputs:
    """Every arm's inputs at `h`, one entry per non-reference temperature, and
    `z_fam` over the val stratum's cluster count (spec 3.1)."""
    candidates = [float(t) for t in taus if float(t) != REFERENCE_TAU]
    return SweepInputs(
        arms={
            arm: SweepArm(taus={t: _tau_inputs(records, arm, seeds, t, h) for t in candidates})
            for arm in _ordered(arms)
        },
        z_fam=pooling.cluster_threshold(SWEEP_FAMILY, _clusters(records)),
        h=h,
    )


def load_sweep(args, cells) -> tuple[int, dict]:
    """Every requested sweep record, keyed by `(arm, seed)`; 11 names the
    first missing, 30 the first whose reference self-check does not reproduce
    the diagnostic within `mbfps.eval.reproduction`'s bound (spec 2.4).

    Only the prose moved here: the identity check below still demands an
    EXACT retrain, because training does not reproduce across a platform
    change at all -- that refusal is the right answer, not a bound to widen."""
    records: dict[tuple[str, int], dict] = {}
    for arm, seed in cells:
        path = sweep_record_path(args.sweep_out, arm, seed)
        if not path.exists():
            print(f"NO CELL: {arm} seed {seed}: no sweep record at {path}; run --phase sweep first")
            return EXIT_NO_CHECKPOINTS, {}
        records[(arm, int(seed))] = load_record(path)
    for (arm, seed), record in records.items():
        if not _trust.trustworthy(record["self_check"]):
            print(
                f"\nSELF-CHECK FAILED for {arm} seed {seed}: the sweep record's self-check is "
                f"{record['self_check']!r}; it was not taken against a pass that reproduced."
            )
            return EXIT_SELF_CHECK_FAILED, {}
    return EXIT_OK, records


# ---------------------------------------------------------------------------
# train: the gate, the identity check, the retrains.
# ---------------------------------------------------------------------------


def retrain_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    return Path(out_dir) / f"retrain_{arm}_seed{seed}.json"


def identity_check(args, buffer, cell: Cell) -> tuple[int, dict | None]:
    """A short retrain at REFERENCE_TAU whose per-step losses must equal the
    M3c record's prefix exactly. M3f measured this box's MPS training to be
    bitwise deterministic, which is what makes the shipped cells the control
    arm without a second fifteen-hour run; this is that premise verified
    rather than assumed, and it is also the only check that the temperature
    edit is a no-op where it must be.

    Run for EVERY cell that retrains, not once for the first: the result is
    written into that cell's record, and a 0.0 measured on another cell would
    be a number that means nothing sitting where one that means something
    goes. `prepare_cell`'s own reproduction message notes the delta is
    arm-dependent across devices, so one cell's result does not stand for
    another's."""
    steps = min(IDENTITY_STEPS, int(args.steps))
    if steps < 1:
        raise ValueError(f"--steps {args.steps} leaves no identity check to run")
    cfg = get_config(
        cell.arm, steps=steps, seq_len=int(cell.record["seq_len"]), seed=cell.seed,
        device=args.device, sample_temperature=REFERENCE_TAU,
    )
    history = train_world_model(cfg, buffer, out_dir=None)
    delta, first = anchor_delta(history["loss"], cell.record["history"]["loss"], steps)
    if delta != 0.0:
        print(
            f"\nIDENTITY CHECK FAILED for {cell.arm} seed {cell.seed}: a retrain at tau="
            f"{REFERENCE_TAU} does not reproduce the reference run's per-step loss over "
            f"{steps} steps (max abs {delta:.3e}, first at step {first}). The shipped cells "
            "are the control arm only if a retrain at the shipped temperature IS the shipped "
            "run. No cell is retrained."
        )
        return EXIT_IDENTITY_CHECK_FAILED, None
    print(
        f"identity check: {cell.arm} seed {cell.seed} at tau={REFERENCE_TAU} reproduces "
        f"{steps} steps exactly (max abs 0.0e+00)"
    )
    # Self-describing: the result names the cell it was measured on, so no
    # record can carry a 0.0 that was measured somewhere else.
    return EXIT_OK, {
        "arm": cell.arm, "seed": int(cell.seed), "tau": float(REFERENCE_TAU),
        "steps": int(steps), "max_delta": float(delta), "first_step": first,
    }


def train_cell(args, buffer, cell: Cell, tau: float, identity: dict) -> tuple[int, dict]:
    """One cell retrained at `tau` from the study's own seed and sequence
    length, into `--out` in the study's layout."""
    cfg = get_config(
        cell.arm, steps=int(args.steps), seq_len=int(cell.record["seq_len"]), seed=cell.seed,
        device=args.device, sample_temperature=float(tau),
    )
    history = train_world_model(cfg, buffer, out_dir=args.out)
    record = {
        "arm": cell.arm,
        "seed": int(cell.seed),
        "tau": float(tau),
        "steps": int(args.steps),
        "seq_len": int(cell.record["seq_len"]),
        "history": history_record(history),
        "seconds": float(history["seconds"]),
        "identity": dict(identity),
        "device": str(args.device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "reference_git_sha": cell.record["git_sha"],
    }
    path = retrain_record_path(args.out, cell.arm, cell.seed)
    write_record(path, record)
    print(
        f"{cell.arm} seed {cell.seed}: retrained {args.steps} steps at tau={tau} in "
        f"{history['seconds']:.0f}s; wrote {path}"
    )
    return EXIT_OK, record


def train_phase(args, cells, device, buffer, taus) -> int:
    """The sweep's records (11, 30), Reading N, the gate (35), the identity
    check (37), then every requested cell at tau*."""
    status, records = load_sweep(args, cells)
    if status != EXIT_OK:
        return status
    h = int(next(iter(records.values()))["decision_h"])
    inputs = sweep_inputs(
        records, arms=list(args.arms), seeds=[int(s) for s in args.seeds], h=h, taus=taus,
    )
    reading = reading_noise(inputs)
    print(format_reading_noise(reading, inputs), end="")
    if reading.status is not SweepStatus.NOISE_LIMITED or reading.tau_star is None:
        print(
            f"\n{reading.status.name.replace('_', ' ')}: {reading.rule}. The retrain is gated on "
            "the sweep, so nothing is trained: a sharper sampler does not help these models at "
            "this grid, and fifteen hours would be spent on a refuted premise."
        )
        return EXIT_NOT_NOISE_LIMITED
    tau = float(reading.tau_star)
    try:
        loaded = [load_cell(args.reference, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, identity = identity_check(args, buffer, cell)
        if status != EXIT_OK:
            return status
        status, _ = train_cell(args, buffer, cell, tau, identity)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# evaluate: the study's own evaluation half, then the trust pass, rung-style.
# ---------------------------------------------------------------------------


def history_from_record(record: dict) -> dict:
    """The trainer's history dict, rebuilt from THIS SCRIPT's own retrain
    record, in the shape `evaluate_job(history=...)` wants: `steps`,
    `seconds`, `kl_dyn_max`, `kl_rate_above_free_bits`, `loss`, `parts`.

    NOT `checkpoint_ladder.history_from_train_record`. That function reads
    `kl_dyn_max`, `kl_rate_above_free_bits` and `checkpoint_seconds` off the
    record's own top level -- which is where THAT script's `train_record`
    keeps them, because it is built straight from a live `train_world_model`
    history that still carries them. `train_cell`'s retrain record above does
    not carry any of the three (see `RETRAIN_KEYS` in the tests: `history`
    holds only `loss` and `parts`, `history_record`'s own policy) -- so
    calling `history_from_train_record` on it raises `KeyError('kl_dyn_max')`
    before a single rollout runs. The two KL summaries are instead recomputed
    here from the stored per-step `parts`, exactly as `train_world_model`
    computes `kl_dyn_max` and `kl_rate_above_free_bits` from the live history
    it returns.
    """
    parts = record["history"]["parts"]
    kl_values = [float(p["kl_dyn"]) for p in parts]
    return {
        "arm": record["arm"],
        "steps": int(record["steps"]),
        "loss": [float(v) for v in record["history"]["loss"]],
        "parts": [dict(p) for p in parts],
        "seconds": float(record["seconds"]),
        "kl_dyn_max": max(kl_values, default=0.0),
        "kl_rate_above_free_bits": (
            sum(1 for v in kl_values if v > KL_FREE_BITS) / len(kl_values) if kl_values else 0.0
        ),
        "checkpoint_seconds": {},
    }


def evaluate_cell(args, buffer, arm: str, seed: int, device, train, val) -> tuple[int, dict | None]:
    """One retrained cell: its retrain record (11), `evaluate_job` at the
    cell's own temperature, then `prepare_cell` on a rung-style cell (36, 12,
    14) and the trust pass. A retrained cell has no prior diagnostic to
    self-check against -- exactly as M3f's rungs had none -- so the check
    that stands here is the temperature the payload carries."""
    path = retrain_record_path(args.out, arm, seed)
    if not path.exists():
        print(f"NO CELL: {arm} seed {seed}: no retrain record at {path}; run --phase train first")
        return EXIT_NO_CHECKPOINTS, None
    record = load_record(path)
    tau = float(record["tau"])
    # The retrain record carries no protocol of its own (`RETRAIN_KEYS`): a
    # retrained cell is evaluated at the M3c reference's own context/horizon,
    # read off its study record, unless a flag overrides it -- the same
    # None-means-the-cell's-own-value rule `_cell_args`/`protocol_mismatch`
    # apply everywhere else in this file.
    reference_record = load_record(job_record_path(args.reference, StudyJob(arm=arm, seed=seed)))
    context = int(args.context) if args.context is not None else int(reference_record["context"])
    horizon = int(args.horizon) if args.horizon is not None else int(reference_record["horizon"])
    # The payload's temperature is checked BEFORE `evaluate_job` runs: it
    # writes `result_<arm>_seed<n>.json` unconditionally, so refusing only
    # afterwards would persist -- and overwrite any earlier valid record
    # with -- a record describing a model that never existed.
    trained_at = float(
        torch.load(
            _trust.checkpoint_path(args.out, arm, seed), map_location="cpu", weights_only=True
        ).get("sample_temperature", REFERENCE_TAU)
    )
    if trained_at != tau:
        print(
            f"\nTEMPERATURE MISMATCH for {arm} seed {seed}: the checkpoint was trained at "
            f"sample_temperature={trained_at}, but its retrain record says {tau}. Nothing is "
            "evaluated: a record written from the wrong sampler describes a model that never "
            "existed."
        )
        return EXIT_TEMPERATURE_MISMATCH, None
    try:
        evaluate_job(
            StudyJob(arm=arm, seed=seed), buffer, args.out,
            history=history_from_record(record), steps=int(record["steps"]),
            seq_len=int(record["seq_len"]), context=context,
            horizon=horizon, device=args.device, sample_temperature=tau,
        )
        cell = rung_cell(args.out, arm, seed)
        status, prepared = prepare_cell(_cell_args(args, args.out, tau), cell, device, train, val)
    # `_trust.TemperatureMismatch`, NOT this script's own `_diagnose` copy:
    # `_sibling` executes the file afresh per importer, so the class
    # `prepare_cell` raises is trust_horizon's, and catching any other copy
    # lets it escape as a traceback instead of exit 36.
    except _trust.TemperatureMismatch as error:
        print(f"\nTEMPERATURE MISMATCH for {arm} seed {seed}: {error}")
        return EXIT_TEMPERATURE_MISMATCH, None
    if status != EXIT_OK:
        return status, None
    traj = reference_trajectories(prepared.model, val, prepared.embedding_probe, **prepared.common)
    # `cell.diagnostic` (from `rung_cell`) is only `{context, horizon}` --
    # exactly what `checkpoint_ladder.py` reads there itself, because a rung
    # has no diagnostic either (`rung_entry` there passes its own trust pass
    # `check=None`). `_trust.trust_record` has no such escape hatch: it always
    # calls `self_check`, which needs a diagnostic's `curves.reference_position`,
    # `curves.persistence_position` and `windows` to compare the pass against.
    # A retrained cell has no earlier pass to compare to, so the comparison is
    # built from THIS SAME pass -- exact against itself by construction -- and
    # `curves.floor_position` / `probe.embedding_selection_r2`, which self_check
    # never compares, come from the study record `evaluate_job` just wrote.
    # The meaningful check for a retrained cell is the temperature the payload
    # carries (already refused above, 36), not a bitwise reproduction with
    # nothing earlier to reproduce.
    diagnostic = {
        "curves": {
            "reference_position": _trust._distance(
                traj.positions, traj.true_positions
            ).mean(axis=0).tolist(),
            "persistence_position": _trust._distance(
                traj.positions_at_context[:, None, :], traj.true_positions
            ).mean(axis=0).tolist(),
            "floor_position": list(cell.record["curves"]["floor_position"]),
        },
        "probe": {"embedding_selection_r2": cell.record["probe"]["embedding_selection_r2"]},
        "windows": {"total": int(traj.windows_total), "episode": [int(e) for e in traj.window_episode]},
        "episodes": {"val": list(cell.record["episodes"]["val"])},
    }
    trust_record = _trust.trust_record(
        arm, seed, traj, diagnostic,
        context=prepared.context, horizon=prepared.horizon, device=device,
    )
    # The self-check that `trust_record` just computed compared this pass with
    # itself, so it reads 0.0 by construction. A sweep record's 0.0 means the
    # pass reproduced the shipped diagnostic bitwise; this one would mean
    # nothing, and the two are indistinguishable once written. So the record
    # says there was nothing to reproduce, and the check that DOES bind a
    # retrained cell -- the temperature its payload carries -- has already
    # refused above (36).
    trust_record["self_check"] = None
    written = _trust.write_trust_record(args.out, trust_record)
    print(f"{arm} seed {seed}: evaluated at tau={tau}; wrote {written}")
    return EXIT_OK, trust_record


def evaluate_phase(args, cells, device, buffer, train, val) -> int:
    for arm, seed in cells:
        status, _ = evaluate_cell(args, buffer, arm, seed, device, train, val)
        if status != EXIT_OK:
            return status
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: the tables, the two readings, sweep.txt and sharper.txt.
# ---------------------------------------------------------------------------


def _num(value, spec: str = ".3f") -> str:
    return fmt_z(float(value), spec)


def _cells_in_order(records: dict) -> list[tuple[str, int]]:
    return sorted(records, key=lambda c: (_ordered([c[0]] + list(ARMS_ORDER)).index(c[0]), c[1]))


def _self_check_table(records: dict) -> str:
    lines = [
        "--- self-check per cell: the tau=1.0 pass against the cell's own diagnostic (exact) ---",
        f"  {'arm':<12}{'seed':>5}{'step':>7}{'ref max|d|':>12}{'pers max|d|':>13}{'windows':>9}{'clusters':>10}",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        check = r["self_check"]
        lines.append(
            f"  {arm:<12}{seed:>5}{int(r['step']):>7}"
            f"{_num(check['reference_position_max_delta'], '.1e'):>12}"
            f"{_num(check['persistence_position_max_delta'], '.1e'):>13}"
            f"{int(r['windows']['total']):>9}{int(r['windows']['clusters']):>10}"
        )
    return "\n".join(lines) + "\n"


def _gate_table(records: dict, taus) -> str:
    """`gap_closed(45)` per seed at every temperature, with the unanimity flag.
    Reported, never decided on (spec 3.2)."""
    # The seed columns are the records' own, not a hardcoded (0, 1, 2): a run
    # over a subset would otherwise print seed 1's gap under an `s0` caption.
    seed_columns = sorted({int(seed) for _, seed in records})
    lines = [
        "--- the gate at every temperature: gap_closed at the horizon on position per seed "
        "(NaN = non-positive band); GATE PASSES = > 0 in every seed. Reported, not decided on. ---",
        f"  {'arm':<12}{'tau':>5}" + "".join(f"{f's{s}':>10}" for s in seed_columns)
        + f"{'passes':>9}{'degen(max)':>12}",
    ]
    arms = sorted({arm for arm, _ in records}, key=lambda a: _ordered([a] + list(ARMS_ORDER)).index(a))
    for arm in arms:
        seeds = sorted(seed for a, seed in records if a == arm)
        for tau in taus:
            gaps = [
                float(_entry(records[(arm, s)], tau)["gate"]["gap_final"])
                if (arm, s) in records else float("nan")
                for s in seed_columns
            ]
            degen = max(int(_entry(records[(arm, s)], tau)["gate"]["degenerate"]) for s in seeds)
            passes = bool(gaps) and all(np.isfinite(g) and g > 0 for g in gaps)
            cells = "".join(f"{_num(g, '+.4f'):>10}" for g in gaps)
            lines.append(
                f"  {arm:<12}{float(tau):>5.1f}{cells}{'YES' if passes else 'no':>9}{degen:>12}"
            )
    return "\n".join(lines) + "\n"


def _noise_table(records: dict, taus, horizon: int) -> str:
    """The statistic that motivated the study: two draws of the same model
    against that model's own imagined displacement, at every temperature."""
    hs = [h for h in (1, 5, 15, 45) if h <= horizon]
    lines = [
        "--- the noise reference against the imagined displacement (medians over windows, "
        "embedding units through the same head): two draws of one model, then how far that "
        "model imagined it moved ---",
        f"  {'arm':<12}{'tau':>5}" + "".join(f"{f'noise({h})':>12}{f'moved({h})':>12}" for h in hs),
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        for tau in taus:
            entry = _entry(r, tau)
            noise = np.asarray(entry["noise"]["curve"], dtype=float)
            moved = np.asarray(entry["displacement"]["curve"], dtype=float)
            cells = "".join(
                f"{_num(noise[h - 1]):>12}{_num(moved[h - 1]):>12}" for h in hs
            )
            lines.append(f"  {arm + ' s' + str(seed):<12}{float(tau):>5.1f}{cells}")
    return "\n".join(lines) + "\n"


def _survival_table(records: dict, taus, horizon: int) -> str:
    hs = [h for h in (1, 2, 3, 5, 10, 15, 30, 45) if h <= horizon]
    lines = [
        "--- survival by temperature: S(h) = fraction of moved draws with h_x > h, free channel, "
        "seeds stacked ---",
        f"  {'arm':<12}{'tau':>5}" + "".join(f"{f'S({h})':>8}" for h in hs),
    ]
    arms = sorted({arm for arm, _ in records}, key=lambda a: _ordered([a] + list(ARMS_ORDER)).index(a))
    for arm in arms:
        seeds = sorted(seed for a, seed in records if a == arm)
        for tau in taus:
            stacked = np.concatenate([
                np.asarray(_entry(records[(arm, s)], tau)["summary"]["survival"]["free"], dtype=float)[None, :]
                for s in seeds
            ])
            lines.append(
                f"  {arm:<12}{float(tau):>5.1f}"
                + "".join(f"{_num(stacked[:, h - 1].mean()):>8}" for h in hs)
            )
    return "\n".join(lines) + "\n"


def write_curves(records: dict, taus, figure: Path) -> str:
    """Two panels against the temperature: S(15) on the free channel and the
    gate's own metric, arms coloured, seeds thin. A missing or broken
    matplotlib, or an unwritable path, costs the FIGURE and nothing else."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    colours = {"pixel_ae": "tab:orange", "frozen_ssl": "tab:blue", "random_vit": "tab:gray"}
    order = [float(t) for t in taus]
    x = np.arange(len(order))
    h = int(next(iter(records.values()))["decision_h"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    try:
        survival, gate = axes.flat
        for (arm, seed), record in records.items():
            colour = colours.get(arm, "black")
            survival.plot(
                x, [float(np.asarray(_entry(record, t)["summary"]["survival"]["free"])[h - 1]) for t in order],
                marker="o", markersize=3, linewidth=0.9, color=colour, label=f"{arm} s{seed}",
            )
            gate.plot(
                x, [float(_entry(record, t)["gate"]["gap_final"]) for t in order],
                marker="o", markersize=3, linewidth=0.9, color=colour,
            )
        gate.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        survival.set_title(f"S({h}), free channel")
        gate.set_title("gap_closed at the horizon (position)")
        for ax in (survival, gate):
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t:.1f}" for t in order])
            ax.set_xlabel("rollout sampling temperature")
        handles, labels = survival.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def sweep_text(records: dict, *, arms, seeds, taus, figure_line: str) -> str:
    """Everything the sweep half of `read` prints, written byte-identical."""
    horizon = int(next(iter(records.values()))["horizon"])
    h = int(next(iter(records.values()))["decision_h"])
    _, clamped = decision_horizon(horizon)
    inputs = sweep_inputs(records, arms=arms, seeds=seeds, h=h, taus=taus)
    reading = reading_noise(inputs)
    note = (f"  NOTE: decision horizon clamped to the run's horizon h={h} "
            f"(pre-registered DECISION_H = {DECISION_H}).\n" if clamped else "")
    return "".join([
        _self_check_table(records),
        _gate_table(records, taus),
        _noise_table(records, taus, horizon),
        _survival_table(records, taus, horizon),
        note,
        f"  pooling: z_fam = cluster_threshold({SWEEP_FAMILY}, {_clusters(records)}) = "
        f"{inputs.z_fam:.2f}; every contrast is this cell at this temperature against the same "
        f"cell at tau={REFERENCE_TAU} on the same val windows; a temperature counts for an arm "
        f"when it clears pooled and in >= {SEEDS_REQUIRED} of its seeds, and the status needs "
        f"{ARMS_REQUIRED} arms.\n",
        format_reading_noise(reading, inputs),
        f"  {figure_line}\n",
    ])


def retrain_inputs(new_trust: dict, old_trust: dict, *, arms, seeds, h: int, tau: float) -> RetrainInputs:
    """Reading M's inputs: each retrained cell paired against the M3c cell of
    the same arm and seed on the same windows, both channels.

    `survival_indicator` reads `summary["crossing"][channel]`, and a trust
    record stores its crossings under the same key (`crossing: {probe,
    free}`) with `windows` shaped the same way too, so a trust record can be
    passed where a stratum summary is expected -- `_survival_series` is the
    one place that wiring happens, shared with Reading N's free-channel
    series rather than duplicated here at a second channel."""
    def contrast(arm, channel, seed_list):
        def series(records, label):
            return [
                _survival_series(
                    records[(arm, int(s))], h, channel, label=label, seed=int(s),
                    val=records[(arm, int(s))]["episodes"]["val"],
                    horizon=records[(arm, int(s))]["horizon"], context=records[(arm, int(s))]["context"],
                    device=records[(arm, int(s))]["device"],
                    torch_version=records[(arm, int(s))]["torch_version"],
                )
                for s in seed_list
            ]

        return _paired(series(new_trust, f"{arm}@sharper"), series(old_trust, f"{arm}@m3c"))

    def arm_inputs(arm, seed_list, per_seed):
        return RetrainArm(
            free=contrast(arm, "free", seed_list),
            probe=contrast(arm, "probe", seed_list),
            per_seed=per_seed,
        )

    return RetrainInputs(
        arms={
            arm: arm_inputs(
                arm, seeds,
                {int(s): arm_inputs(arm, [s], None) for s in seeds},
            )
            for arm in _ordered(arms)
        },
        z_fam=pooling.cluster_threshold(RETRAIN_FAMILY, _clusters_trust(new_trust)),
        h=h, tau=float(tau),
    )


def _clusters_trust(records: dict) -> int:
    first = next(iter(records.values()))
    return int(np.unique(np.asarray(first["windows"]["episode"])).size)


def _load_trust(directory: Path, cells) -> dict:
    """The trust records of a study directory, or an empty dict when any is
    missing -- the retrain half of the read is simply not printed then."""
    records = {}
    for arm, seed in cells:
        path = Path(directory) / f"trust_{arm}_seed{seed}.json"
        if not path.exists():
            return {}
        records[(arm, int(seed))] = load_record(path)
    return records


def _reference_trust(records: dict, cells) -> dict:
    """The M3c side of Reading M, taken from the sweep's own REFERENCE_TAU
    entry.

    ALWAYS this, never a `trust_<arm>_seed<n>.json` that happens to sit in
    `--reference`. The shipped M3c directory does carry M3d's trust records,
    so preferring them would have meant the real run reading a file and every
    test reading this -- two paths, and the untested one shipping. Beyond
    that, this entry is the better control: `load_sweep` has already proven
    it bitwise identical to the M3c diagnostic in THIS run (exit 30
    otherwise), it scores the same windows at the same protocol as the
    retrain it is compared against, and it carries `crossing`/`windows` in
    the shape a trust record does, for both channels."""
    out = {}
    for arm, seed in cells:
        record = records[(arm, int(seed))]
        summary = _entry(record, REFERENCE_TAU)["summary"]
        out[(arm, int(seed))] = {
            "crossing": summary["crossing"],
            "windows": summary["windows"],
            "episodes": record["episodes"],
            "horizon": record["horizon"],
            "context": record["context"],
            "device": record["device"],
            "torch_version": record["torch_version"],
        }
    return out


def sharper_text(new_trust: dict, old_trust: dict, *, arms, seeds, h: int, tau: float) -> str:
    inputs = retrain_inputs(new_trust, old_trust, arms=arms, seeds=seeds, h=h, tau=tau)
    reading = reading_sharper(inputs)
    return "".join([
        f"  pooling: z_fam = cluster_threshold({RETRAIN_FAMILY}, {_clusters_trust(new_trust)}) = "
        f"{inputs.z_fam:.2f}; each retrained cell is paired against the M3c cell of the same arm "
        "and seed on the same val windows.\n",
        format_reading_sharper(reading, inputs),
    ])


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def read_phase(args, cells) -> int:
    """The sweep's records (11, 30) and its reading always; the retrain's
    reading too when the retrained cells' own trust records are on disk.

    `old_trust`, Reading M's control arm, is ALWAYS `_reference_trust` -- the
    sweep's own REFERENCE_TAU entry, which `load_sweep` has already proven
    bitwise identical to the M3c diagnostic in this run. It is never a
    `trust_<arm>_seed<n>.json` that happens to sit under `--reference`: the
    shipped M3c directory carries M3d's, so preferring those would mean the
    real run reading files and every test reading the derived path."""
    status, records = load_sweep(args, cells)
    if status != EXIT_OK:
        return status
    taus = [float(t) for t in next(iter(records.values()))["taus"]]
    arms, seeds = list(args.arms), [int(s) for s in args.seeds]
    figure = args.figure if args.figure is not None else args.sweep_out / "sweep_curves.png"
    figure_line = write_curves(records, taus, figure)
    text = sweep_text(records, arms=arms, seeds=seeds, taus=taus, figure_line=figure_line)
    print(text, end="")
    write_text(args.sweep_out / "sweep.txt", text)

    new_trust = _load_trust(args.out, cells)
    old_trust = _reference_trust(records, cells)
    if new_trust and old_trust:
        retrain = load_record(retrain_record_path(args.out, *cells[0]))
        h = int(next(iter(records.values()))["decision_h"])
        retrain_text = sharper_text(
            new_trust, old_trust, arms=arms, seeds=seeds, h=h, tau=float(retrain["tau"]),
        )
        print(retrain_text, end="")
        write_text(args.out / "sharper.txt", retrain_text)
    return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3h_sharper"))
    parser.add_argument("--sweep-out", type=Path, default=Path("runs/m3h_sweep"))
    parser.add_argument("--reference", type=Path, default=Path("runs/m3_study_v2"),
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
    parser.add_argument("--steps", type=int, default=STEPS, help="the retrain's length")
    parser.add_argument("--figure", type=Path, default=None,
                        help="the curves figure; default <sweep-out>/sweep_curves.png")
    return parser


def _check_usage(parser: argparse.ArgumentParser, args) -> None:
    for name, other in (("--reference", args.reference), ("--sweep-out", args.sweep_out)):
        if args.out.resolve() == Path(other).resolve():
            parser.error(
                f"--out and {name} are the same directory; the retrain would write its "
                "checkpoints over the ones it is being compared against"
            )


def main(argv: list[str] | None = None, *, taus=TAU_GRID) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    if args.phase in ("sweep", "train", "evaluate", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
        if args.phase in ("sweep", "all"):
            status = sweep_phase(args, cells, device, train, val, taus)
            if status != EXIT_OK:
                return status
        if args.phase in ("train", "all"):
            status = train_phase(args, cells, device, buffer, taus)
            if status != EXIT_OK:
                return status
        if args.phase in ("evaluate", "all"):
            status = evaluate_phase(args, cells, device, buffer, train, val)
            if status != EXIT_OK:
                return status
    if args.phase in ("read", "all"):
        return read_phase(args, cells)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
