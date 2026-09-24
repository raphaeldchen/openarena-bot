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


def _series(record: dict, tau: float, h: int, label: str) -> pooling.CellSeries:
    values, changed = survival_indicator(_entry(record, tau)["summary"], "free", h)
    return cell_series(
        _entry(record, tau)["summary"], values, changed,
        arm=label, seed=record["seed"], rung="val", channel="S/free",
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


def _tau_inputs(records: dict, arm: str, seeds, tau: float, h: int) -> TauInputs:
    def contrast(seed_list):
        kept = [records[(arm, int(s))] for s in seed_list]
        return _paired(
            [_series(r, tau, h, f"{arm}@{tau_key(tau)}") for r in kept],
            [_series(r, REFERENCE_TAU, h, f"{arm}@{tau_key(REFERENCE_TAU)}") for r in kept],
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
    first missing, 30 the first whose reference self-check is not exact."""
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


def read_phase(args, cells) -> int:
    raise NotImplementedError("read is Task 8")


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
