"""M3j: where on the `enc(t) -> z -> h` path is observed motion lost -- the
measure phase.

M3i read NO_MOTION at the decision horizon and retired the prior-side levers.
What is left is a choice between two levers that cost the same 13.5-hour
retrain: the ENCODER, if the frozen features do not resolve motion, or the
OBJECTIVE, if they do and the RSSM discards it. `mbfps.eval.retention` is the
pure half of the answer -- the ladder of blocks, the backward targets, the
row bookkeeping -- and owns none of torch, the model, or the record schema.
This script is the impure half: per shipped cell, it prepares the checkpoint
through `trust_horizon.prepare_cell`, takes the three DISJOINT gathers
`gain_from_blocks` needs, runs the ladder over both targets and every reported
k, fits the base control, and returns one record. `scripts/latent_motion.py`
mirrors it in age but not in shape -- see `measure_cell` for what carries over
and what does not.

  measure   per shipped cell: `enc(t) (+) B` -> backward displacement, for
            `B` in `RUNGS` and displacement in `TARGETS`, at every reported k;
            plus the base control `enc(t)` -> absolute position. One record
            per cell.
  read      NOT YET BUILT -- Task 8 pools these records into Reading E and
            decides the lever. `EXIT_BASE_UNRESOLVED` and
            `EXIT_MOTION_UNRESOLVED` are that phase's; they are defined here
            because this script owns the exit-code range, not because
            `measure_phase` raises them.

LOADING IS `trust_horizon.py`'S, exactly as `scripts/latent_motion.py` loads
it: `Cell`, `CellMissing`, `load_cell`, `self_check` and `prepare_cell` are
imported by path through `_sibling`, so a cell is refused here for the same
reasons and in the same words `trust_horizon.py` and `scripts/split_gap.py`
refuse it, and so this module holds its own copy of every class `_sibling`
defines -- never catch another importer's copy of `CellMissing`.

THE CHECKS, BY PHASE:

  measure:  EXIT_NO_CHECKPOINTS (11)   a requested cell lacks its checkpoint,
                                        record or diagnostic.
            EXIT_SPLIT_MISMATCH (12)   the split by name is not the record's.
            EXIT_RECORD_MISMATCH (14)  --context/--horizon disagree with the
                                        protocol, or `evaluate_rollout` no
                                        longer reproduces the record's curve.
            EXIT_SELF_CHECK_FAILED (30) a fresh `reference_trajectories` pass
                                        does not reproduce the cell's
                                        diagnostic within the bound.
  read:     EXIT_NO_CHECKPOINTS (11)   a requested cell has no retention
                                        record.
            EXIT_BASE_UNRESOLVED (39)  `enc(t)` -> absolute position did not
                                        clear `retention.BASE_R2_FLOOR`.
            EXIT_MOTION_UNRESOLVED (40) no rung read either target at any
                                        horizon.

0 / 11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose. 39 and 40
are in no other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10,
diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages 34,
sharper_latent 35-37, latent_motion 38, argparse 2, a traceback 1).

TWO CORRECTIONS TO THE M3J TASK-7 BRIEF, found by tracing the types the brief's
own sketch of `measure_cell` passes around:

  * THE BRIEF'S `measure_cell` SKETCH CALLED `self_check(cell, prepared.reference)`.
    `self_check`'s signature is `(traj: Trajectories, diagnostic: dict)`;
    `cell` is a `trust_horizon.Cell` (no `.positions`, no `.true_positions`)
    and `prepared.reference` is a `rollout.RolloutResult` (mean curves only,
    not the per-window rows `self_check` reduces) -- neither reading of the
    call type-checks, so as written it cannot run. `EXIT_SELF_CHECK_FAILED` is
    in this script's own exit-code table above and is explicitly re-exported
    from `trust_horizon.py` for exactly this use, and the task brief's own
    prose says to "keep its self_check call" from `scripts/latent_motion.py`
    -- so `measure_cell` here takes a second, fresh
    `reference_trajectories(prepared.model, val, prepared.embedding_probe,
    **prepared.common)` pass, the same one `scripts/latent_motion.py` and
    `trust_horizon.py`'s own `_run_cell` take, and gates on `self_check(traj,
    cell.diagnostic)` exactly as they do. `prepare_cell`'s own reproduction
    check (14) compares `evaluate_rollout`'s MEAN curves against the study
    record and is free; this second pass compares the PER-WINDOW rows against
    the diagnostic and is not -- a regression that kept the mean curve right
    and the rows wrong would still clear 14 and must not write a ladder built
    on those rows.
  * THE BRIEF'S SKETCH READ `cell.record["step"]`. Every other script that
    reads a study record's step count (`scripts/latent_motion.py`,
    `scripts/sharper_latent.py`, `scripts/stage_decomposition.py`,
    `scripts/checkpoint_ladder.py`) reads `cell.record["steps"]` -- the
    key `scripts/run_study.py` actually writes. The singular key does not
    exist on a real record; a live run would `KeyError` on the first cell.
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
from mbfps.eval.probe import (
    GainSplit, apply_probe, fit_probe, gain_from_blocks, gather_probe_data, probe_r2,
)
from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, CONFIDENCE, DECISION_K, K_REPORTED, RESAMPLES, RUNGS,
    SEEDS_REQUIRED, TARGETS, BaseControl, RetentionInputs, RetentionStatus,
    backward_rotation, backward_translation, format_ladder, format_reading_retention,
    reading_retention, rung_arm, rung_block, shifted_rows,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_latent_retention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED

EXIT_BASE_UNRESOLVED = 39
"""`enc(t)` -> absolute position did not read. The base arm is the thing every
gain is measured against, so if the current frame cannot linearly say where it
is, a null on displacement says nothing about the representation and no reading
is taken."""

EXIT_MOTION_UNRESOLVED = 40
"""No rung read either target at any horizon. The measurement is empty: it
detected no motion anywhere on the path, so it cannot localise a loss and no
lever is chosen."""

PHASES: tuple[str, ...] = ("measure", "read", "all")

FIT_EPISODES: int = 20
"""Training episodes the probe weights are fit on -- `probe.PROBE_EPISODE_LIMIT`,
the same set gate criterion 4 fits on, so the levels stay comparable."""

SELECT_EPISODES: int = 20
"""Training episodes AFTER the first `FIT_EPISODES` that select the ridge.

`probe.filtering_gain`'s default, and NOT `scripts/latent_motion.py`'s 4. That
docstring records the measurement: a 4-episode selection split picks 1e5 for
both arms and reports +0.0325 where a 20-episode split picks 1e3 and recovers
-0.0208 -- THE SIGN FLIPS on a one-step selection error, because a gain is a
difference of levels and the grid moves each level by ~0.10 per decade.
Shrinking this to save a gather does not make the number noisier, it makes it
wrong."""


def retention_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs, and M3i's own
    task reports for what it cost there."""
    return Path(out_dir) / f"retention_{arm}_seed{seed}.json"


def k_key(k: int) -> str:
    """`15 -> "k15"`. A record key may not contain '.', since `write_record`
    addresses non-finite fields by dotted path."""
    return f"k{int(k)}"


TARGET_BUILDERS = {
    "translation": backward_translation,
    "rotation": backward_rotation,
}
"""`TARGETS` -> the builder that makes it. A dict rather than an if-chain so a
target named in `retention.TARGETS` and missing here fails at import-time
lookup rather than silently producing a ladder with a hole in it."""

assert set(TARGET_BUILDERS) == set(TARGETS), (
    f"TARGET_BUILDERS {sorted(TARGET_BUILDERS)} must name exactly retention.TARGETS "
    f"{sorted(TARGETS)} -- otherwise the docstring above's 'fails at import-time lookup' "
    "claim is false: a target missing from this dict would instead raise a plain "
    "KeyError the first time cell_ladder looked it up, at runtime, with a hole already "
    "in the ladder."
)


def gather_three_splits(prepared, train, val, *, seed: int):
    """The three DISJOINT gathers `probe.gain_from_blocks` needs.

    The same structure `probe.filtering_gain` uses, and for the reason its
    docstring records: the weights come from the first `FIT_EPISODES` training
    episodes, the ridge is selected on the NEXT `SELECT_EPISODES` (from beyond
    the fit set, so the weights train on the same episodes criterion 4 uses),
    and the reported R^2 and interval come from the validation windows that
    neither the weights nor the selection ever saw. Seeds are `seed`, `seed + 2`
    and `seed + 1` in that order, matching `filtering_gain` exactly so the two
    diagnostics describe the same rows.

    THE SCORED SPLIT TAKES EVERY VALIDATION EPISODE, WHICH IS WHERE THIS
    DEPARTS FROM `filtering_gain`. That function's `limit` caps the fit split
    AND the scored split at the same number -- its docstring says so
    ("episodes for the fit split, and for the scored split") -- so mirroring it
    literally would have scored 20 of the 24 validation episodes and silently
    thrown away four. The smoke run caught it: 20 clusters and 9,261 rows at
    k = 1 against the 24 and 11,221 spec sections 3.4 and 6 state and accept on.
    The protocol every milestone since M3d reads is 229 windows over 24
    validation episodes, and that is what the reading must be taken on.
    """
    used = list(train)
    fit_paths = used[:FIT_EPISODES]
    if not fit_paths:
        raise ValueError("no training episodes to fit the retention probes on")
    select_paths = used[FIT_EPISODES:FIT_EPISODES + SELECT_EPISODES]

    def gather(paths, draw: int, limit: int | None = None):
        return gather_probe_data(
            prepared.model, paths,
            prepared.common["feature_backbone"], prepared.common["device"],
            context=prepared.context, horizon=prepared.horizon,
            limit=len(paths) if limit is None else limit, seed=draw,
        )

    return (
        gather(fit_paths, seed),
        gather(select_paths, seed + 2) if select_paths else None,
        gather(val, seed + 1),
    )


def _split_for(data: dict, target: str, k: int, rung: str, h_dim: int):
    """One rung's `GainSplit` at one target and horizon, plus the rows it uses.

    `base` is `enc(t)` on exactly the rows the target is defined on, so every
    rung at this `(target, k)` is handed a BYTE-IDENTICAL base array. The base
    control is not built here: it is a level on every row, not a gain on a
    shifted subset, so `base_control` fits it directly.
    """
    values, rows = TARGET_BUILDERS[target](
        data["targets"], data["window"], data["step"], k,
    )
    if rows.size == 0:
        raise ValueError(
            f"k={k}: no window is long enough to carry a backward displacement; "
            "a gain computed on zero rows would describe nothing"
        )
    _, source = shifted_rows(data["window"], data["step"], k)
    return GainSplit(
        base=np.asarray(data["encoder_embedding"], dtype=np.float64)[rows],
        block=rung_block(data, rung, rows=rows, source=source, h_dim=h_dim),
        target=values,
    ), rows


def cell_ladder(fit: dict, select: dict | None, score: dict, *, h_dim: int,
                seed: int = 0, ks=K_REPORTED) -> dict:
    """Every `(target, k, rung)` gain for one cell, plus the row count per k.

EVERY RUNG AT ONE (target, k) IS HANDED A BYTE-IDENTICAL BASE ARRAY, which is
    what makes the four gains differences against the same base level and so
    comparable to each other rather than each only to its own fit (spec 2.1).
    `fit_probe` is deterministic, so identical inputs give an identical base
    level and the redundant re-fit inside each `gain_from_blocks` call costs
    accuracy nothing -- the base is ~1/8 of a joint solve, so sharing the fitted
    probe would save ~11% of solve time and is not worth the extra interface.
    Four INDEPENDENT row selections would break the property while still
    producing four plausible gains, which is why a test pins the arrays.

    The bootstrap groups on `episode`, not `window` (spec 3.1).

    `seed` IS THE CELL'S SEED, threaded into every `gain_from_blocks` call's own
    bootstrap draw -- `filtering_gain`'s convention exactly (it passes its own
    `seed` straight through to `_gain_from_splits` and on into
    `gain_from_blocks`). A caller that left this at its default would hand every
    cell, rung, target and horizon the SAME bootstrap draw (seed 0), which
    correlates the interval noise across cells that the seeds x arms agreement
    rule in `retention.py` treats as independent -- and that rule carries this
    milestone's entire multiple-comparison burden. Common random numbers across
    rungs WITHIN one cell are fine and intended (they are what makes the four
    gains comparable); it is only across cells that the draws must differ.
    """
    ladder: dict = {target: {} for target in TARGETS}
    rows_by_k: dict[str, int] = {}
    for k in ks:
        rows_by_target: dict[str, int] = {}
        for target in TARGETS:
            splits, row_sets = {}, {}
            for rung in RUNGS:
                splits[rung] = {}
                for name, data in (("fit", fit), ("select", select), ("score", score)):
                    if data is None:
                        splits[rung][name] = None
                        continue
                    split, rows = _split_for(data, target, k, rung, h_dim)
                    splits[rung][name] = split
                    row_sets[name] = rows
            groups = np.asarray(score["episode"])[row_sets["score"]]
            rows_by_target[target] = int(row_sets["score"].size)
            ladder[target][k_key(k)] = {
                rung: gain_from_blocks(
                    splits[rung]["fit"], splits[rung]["select"], splits[rung]["score"],
                    groups=groups, resamples=RESAMPLES, confidence=CONFIDENCE, seed=seed,
                )
                for rung in RUNGS
            }
        # Both targets are built from the same `shifted_rows(window, step, k)`
        # (Task 4 pins that), so they MUST agree on the scored row count at
        # this k; assigning inside the target loop above (as this used to)
        # silently kept only the last target's count under one shared key. A
        # future target with a different row rule would then relabel an
        # earlier target's count without either loop noticing.
        counts = set(rows_by_target.values())
        if len(counts) != 1:
            raise ValueError(
                f"k={k}: targets disagree on scored row count {rows_by_target}, but "
                "every target at one k is supposed to share one row set"
            )
        rows_by_k[k_key(k)] = counts.pop()
    return {"ladder": ladder, "rows": rows_by_k}


def _per_column_r2(predicted: np.ndarray, targets: np.ndarray) -> list[float]:
    """`probe._mean_r2`'s four addends, UNAVERAGED -- pos_x, pos_y, sin(angle),
    cos(angle) in that order. Duplicates that function's per-column formula
    rather than importing its private name, so a zero-variance column reports
    NaN here (a fact about that column) instead of silently vanishing from an
    average the way it does inside `_mean_r2` itself."""
    scores = []
    for c in range(targets.shape[1]):
        truth = targets[:, c]
        denom = float(((truth - truth.mean()) ** 2).sum())
        if denom == 0.0:
            scores.append(float("nan"))
            continue
        scores.append(1.0 - float(((truth - predicted[:, c]) ** 2).sum()) / denom)
    return scores


def base_control(fit: dict, select: dict | None, score: dict) -> dict:
    """`enc(t)` -> absolute position, as an r2 LEVEL on the validation windows.

    The positive control M3i measured only during its final review and recorded
    nowhere; its section 9 provenance note asked a successor to build it into the
    measure phase, and this is that. Every row is used -- there is no backward
    shift, because the target is the frame's own state.

    A level, not a gain: this is the arm every gain in `cell_ladder` is measured
    against, so the only question is whether it reads at all.

    `r2` IS THE GATED NUMBER and it stays a 4-column mean over pos_x, pos_y,
    sin(angle), cos(angle) -- the same shape as `latent_selection_r2`, on
    purpose: `retention.BASE_R2_FLOOR` (0.10) was calibrated against that
    number's scale, and re-pointing the gate at position alone would silently
    change what a pre-registered threshold means.

    `position_r2` is a COMPANION, not a second gate: the same fitted probe's
    r2 against the first two target columns only. It exists because
    `EXIT_BASE_UNRESOLVED`'s docstring and this reading's rule text both say
    the gate is about whether the current frame can say "absolute position",
    while the gated `r2` mixes heading into that claim. A cell whose `r2`
    clears 0.10 while `position_r2` does not (plausible for a frozen
    single-frame backbone, where heading can be far easier to read than map
    position) is a fact the results section must REPORT, not one this record
    is allowed to hide by averaging it away.

    `per_column_r2` is the same probe's four individual column scores, so the
    split between position and heading is checkable without refitting anything.
    """
    def probe_for(data):
        return (
            np.asarray(data["encoder_embedding"], dtype=np.float64),
            np.asarray(data["targets"], dtype=np.float64),
        )

    fit_x, fit_y = probe_for(fit)
    score_x, score_y = probe_for(score)
    if select is None:
        probe = fit_probe(fit_x, fit_y)
    else:
        select_x, select_y = probe_for(select)
        probe = fit_probe(fit_x, fit_y, select_x, select_y)
    predicted = apply_probe(probe, score_x)
    return {
        "r2": probe_r2(probe, score_x, score_y),
        "position_r2": probe_r2(probe, score_x, score_y[:, :2]),
        "per_column_r2": _per_column_r2(predicted, score_y),
        "ridge": probe["ridge"],
        "ridge_selected": select is not None,
        "rows": int(score_y.shape[0]),
    }


# ---------------------------------------------------------------------------
# measure: one cell, then every cell.
# ---------------------------------------------------------------------------


def _cell_args(args, source: Path) -> types.SimpleNamespace:
    """`prepare_cell` reads `out`, `device`, `context` and `horizon` off its
    args, and its `out` is the STUDY directory the checkpoint is loaded from
    (`trust_horizon.load_checkpoint_model(args.out, ...)`). This script's own
    `--out` holds the retention records and nothing else, so handing `args`
    through unchanged would look for the nine M3c checkpoints in the output
    directory and refuse every cell. Same shim, same reason, as
    `scripts/latent_motion.py`.
    """
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context, horizon=args.horizon,
    )


def measure_cell(args, cell: Cell, device, train, val, ks=K_REPORTED) -> tuple[int, dict | None]:
    """One cell: the checks, the three gathers, the ladder, the base control.

    12 and 14 are `prepare_cell`'s, so this script, `scripts/split_gap.py` and
    `scripts/latent_motion.py` all refuse a cell for the same reasons in the
    same words -- and the reproduction bound of M3h spec 2.4 is checked there
    for free, because it compares `evaluate_rollout`'s mean curves against the
    study record.

    30 is NOT free. It needs a SECOND pass -- `reference_trajectories`, the
    same one `scripts/latent_motion.py`'s own `measure_cell` and
    `trust_horizon.py`'s own `_run_cell` take -- because `self_check` reduces
    the PER-WINDOW rows, not the mean curves `prepare_cell` already checked. A
    regression that kept the mean curve right and the rows wrong would still
    clear 14 and must not write a ladder built on those rows.

    `(EXIT_OK, record)` once the record is built; `(status, None)` on any
    refusal, with the refusal printed. The record is not yet WRITTEN -- that
    is `measure_phase`'s job, so every cell's record picks up the same
    `git_sha` provenance and the same non-finite scan through one call to
    `write_record`.
    """
    arm, seed = cell.arm, cell.seed
    status, prepared = prepare_cell(
        _cell_args(args, args.source), cell, device, train, val
    )
    if prepared is None:
        return status, None

    traj = reference_trajectories(
        prepared.model, val, prepared.embedding_probe, **prepared.common,
    )
    check = self_check(traj, cell.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {arm} seed {seed}: " + "; ".join(check.failures())
            + ". Same windows, same rollout, same refit probe -- or this is not measuring "
            "what the gate measured. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None

    fit, select, score = gather_three_splits(prepared, train, val, seed=cell.seed)
    h_dim = int(prepared.model.rssm.cfg.h_dim)
    # `seed=cell.seed`, matching `filtering_gain`'s convention exactly -- see
    # `cell_ladder`'s own docstring for why leaving this at cell_ladder's
    # default (seed 0) would correlate the bootstrap noise across cells.
    ladder = cell_ladder(fit, select, score, h_dim=h_dim, seed=cell.seed, ks=ks)
    record = {
        "arm": cell.arm, "seed": cell.seed, "step": int(cell.record["steps"]),
        "record_git_sha": cell.record.get("git_sha", "unknown"),
        "device": str(device), "context": prepared.context, "horizon": prepared.horizon,
        "h_dim": h_dim,
        "ks": [int(k) for k in ks],
        "split_seed": SPLIT_SEED,
        "torch_version": torch.__version__,
        "ladder": ladder["ladder"],
        "rows": ladder["rows"],
        "base_control": base_control(fit, select, score),
        "clusters": int(np.unique(score["episode"]).size),
        "windows": {
            "episode": np.asarray(score["episode"]).tolist(),
            "window": np.asarray(score["window"]).tolist(),
        },
        "self_check": check.record(),
        "episodes": {"fit": [p.name for p in train[:FIT_EPISODES]],
                     "select": [p.name for p in
                                train[FIT_EPISODES:FIT_EPISODES + SELECT_EPISODES]],
                     "val": [p.name for p in val]},
    }
    return EXIT_OK, record


def measure_phase(args, cells, device, train, val, ks=K_REPORTED) -> int:
    """Every requested cell, refusing the whole run before any pass when a
    cell is missing (11).

    `measure_cell` builds each record; this function writes it through the
    same `write_record` `scripts/latent_motion.py` uses -- so the non-finite
    scan and the `git_sha` provenance are shared -- prints one line per cell,
    and returns the first non-`EXIT_OK` status or `EXIT_OK`.

    CELLS ARE LOADED FROM `args.source`, NOT `args.out`. `--out` is this
    script's own record directory -- where retention records are WRITTEN,
    below -- and the nine M3c checkpoints, study records and diagnostics live
    in the study directory instead, exactly as `scripts/latent_motion.py`
    splits `--out` (its motion records) from `--reference` (the same M3c
    study). Loading from `args.out` here would look for those three files in
    a directory that starts out empty and fail before a single cell measured.
    """
    try:
        loaded = [load_cell(args.source, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.out.mkdir(parents=True, exist_ok=True)
    decision_key = k_key(DECISION_K)
    for cell in loaded:
        status, record = measure_cell(args, cell, device, train, val, ks=ks)
        if status != EXIT_OK:
            return status
        record["git_sha"] = git_sha()
        path = retention_record_path(args.out, cell.arm, cell.seed)
        write_record(path, record)
        translation = record["ladder"].get(TARGETS[0], {})
        if decision_key in translation:
            det = translation[decision_key]["deterministic"]
            headline = f"{TARGETS[0]}/deterministic gain at k={DECISION_K} {det['gain']:+.3f}"
        else:
            headline = f"k={DECISION_K} not reported"
        print(
            f"{cell.arm} seed {cell.seed}: base r2 {record['base_control']['r2']:.3f}; "
            f"{headline}; wrote {path}"
        )
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: the nine records pooled into Reading E.
# ---------------------------------------------------------------------------


def require_readable_plan(arms, seeds) -> None:
    """Refuse a plan too narrow for the rule to reach any status.

    The rule needs `ARMS_REQUIRED` arms, each clearing in `SEEDS_REQUIRED`
    seeds. A plan with fewer cannot reach MOTION_RETAINED, BOTTLENECK_LOSS,
    MOTION_DISCARDED or TRANSLATION_UNRESOLVED at all -- it can only ever print
    UNRESOLVED_MOTION, which would look like a finding. M3i shipped this defect
    and caught it in review: a narrowed plan printed `z +996.13, 3/3 up` beside
    a NO DIFFERENCE verdict.
    """
    if len(arms) < ARMS_REQUIRED or len(seeds) < SEEDS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(arms)} arm(s) x {len(seeds)} seed(s) cannot reach any "
            f"status: the rule needs {ARMS_REQUIRED} arms clearing in "
            f"{SEEDS_REQUIRED} seeds each, so this plan can only ever print "
            f"UNRESOLVED_MOTION -- which would read as a finding"
        )


def load_retention(out_dir: Path, arms, seeds) -> dict:
    """Every planned cell's record, keyed by `(arm, seed)` -- or `CellMissing`
    naming the first that is not on disk.

    Named, never skipped: a pool over the cells that happen to be present,
    printed under the nine cells' names, is exactly the failure
    `pooling.MissingCell` exists for. A record whose own `arm`/`seed` disagree
    with the file it was read under is refused too -- a swapped pair of files
    would pool one cell under another's name with every count still right.
    """
    records: dict[tuple[str, int], dict] = {}
    for arm in arms:
        for seed in seeds:
            path = retention_record_path(out_dir, arm, int(seed))
            if not path.exists():
                raise CellMissing(
                    f"{arm} seed {int(seed)}: no retention record at {path}; "
                    "run --phase measure first"
                )
            record = load_record(path)
            if record.get("arm") != arm or int(record.get("seed", -1)) != int(seed):
                raise ValueError(
                    f"{path.name} was read for {arm} seed {int(seed)} but its record says "
                    f"arm={record.get('arm')!r} seed={record.get('seed')!r}"
                )
            records[(arm, int(seed))] = record
    return records


def require_one_protocol(records: dict) -> None:
    """Every record reports the same protocol on the same windows, or the two
    that disagree are named with the field.

    Nine cells pooled into one reading must describe the same rows: two
    protocols pooled as one would be a reading over a union nothing measured.
    Mirrors `scripts/latent_motion.py`'s namesake, which compares `ks` and
    `torch_version` in addition to the windows/protocol fields below -- a
    retention record carries both keys too, so without them here nine cells
    measured on different torch builds, or against different reported k
    grids, would pool with no refusal at all. `git_sha` is compared for the
    same reason: Task 9's acceptance step requires nine records at one
    `git_sha`, and pooling across two code versions is exactly what this
    refusal exists to prevent. All three are read with `.get(...)` so a
    record written before they existed is handled -- defaulted identically
    across every record -- rather than raising a bare `KeyError`.

    `clusters` and `rows` are NOT compared here: they are print-only
    (`retention_inputs` takes both from whichever record sorts first, by its
    own docstring) and gate nothing downstream, so no lever can flip on
    them -- this function guards only the fields a wrong pooling could act
    on.
    """
    items = sorted(records.items())
    if not items:
        raise SystemExit("no retention record to read")
    (first_cell, first) = items[0]
    for cell, record in items[1:]:
        for field, pick in (
            ("windows.episode", lambda r: list(r["windows"]["episode"])),
            ("episodes.val", lambda r: list(r["episodes"]["val"])),
            ("context", lambda r: int(r["context"])),
            ("horizon", lambda r: int(r["horizon"])),
            ("device", lambda r: str(r["device"])),
            ("ks", lambda r: list(r.get("ks", []))),
            ("torch_version", lambda r: str(r.get("torch_version", ""))),
            ("git_sha", lambda r: str(r.get("git_sha", ""))),
        ):
            mine, theirs = pick(record), pick(first)
            if mine != theirs:
                shown = (
                    f" ({len(mine)} vs {len(theirs)} windows)" if field == "windows.episode"
                    else f": {mine!r} vs {theirs!r}"
                )
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]} and {first_cell[0]} seed {first_cell[1]} "
                    f"disagree on {field}{shown}; they are not one measurement and their "
                    "arms cannot be read against one bar"
                )


def retention_inputs(records: dict) -> RetentionInputs:
    """Pool the nine records into Reading E's input.

    The seed tally is built from the per-seed intervals in the records, not
    from a pooled estimate: r2 is not a per-window quantity, so there is
    nothing to pool the way `pooling.paired_contrast` pools a contrast. The
    agreement requirement IS the rule here (spec 3.1), which is also what
    carries the multiple-comparison burden across 4 rungs x 3 horizons.

    `retention.rung_arm` raises on a non-finite gain or bound -- M3i's ledger
    note, closed there rather than here: a NaN must stop the read rather than
    evaluate False in every predicate at once.

    THE BASE CONTROL READS `["r2"]`, THE GATED 4-COLUMN MEAN -- not
    `embedding_r2` (the task brief's stale *Interfaces* prose named a key
    `base_control` has never returned; its own Step 3 sketch, and the shipped
    `base_control` in this file, both agree on `"r2"`). `position_r2`, the
    companion this milestone's read phase also reports, is read separately by
    `retention_text` from the raw records -- it plays no part in the gate and
    so has no place on `RetentionInputs`, which carries only what Reading E is
    decided on.
    """
    arms = sorted({arm for arm, _ in records})
    ladder: dict = {}
    for target in TARGETS:
        ladder[target] = {}
        for k in K_REPORTED:
            ladder[target][k] = {}
            for rung in RUNGS:
                ladder[target][k][rung] = {
                    arm: rung_arm([
                        records[(arm, seed)]["ladder"][target][k_key(k)][rung]
                        for _, seed in sorted(cell for cell in records if cell[0] == arm)
                    ])
                    for arm in arms
                }
    base = {}
    for arm in arms:
        levels = [
            float(records[(arm, seed)]["base_control"]["r2"])
            for _, seed in sorted(cell for cell in records if cell[0] == arm)
        ]
        if not all(np.isfinite(levels)):
            raise ValueError(f"non-finite base control r2 in arm {arm}: {levels}")
        base[arm] = BaseControl(
            r2=float(np.mean(levels)),
            seeds_clear=sum(1 for r in levels if r > BASE_R2_FLOOR),
            seeds_total=len(levels),
        )
    first = records[next(iter(sorted(records)))]
    return RetentionInputs(
        ladder=ladder, base=base,
        clusters=int(first["clusters"]),
        rows={k: int(first["rows"][k_key(k)]) for k in K_REPORTED},
    )


READ_EXITS = {
    "UNRESOLVED_BASE": EXIT_BASE_UNRESOLVED,
    "UNRESOLVED_MOTION": EXIT_MOTION_UNRESOLVED,
}
"""Status -> exit code, for the two statuses that are refusals. Every other
status is a reading and exits 0: a milestone that exited non-zero on a finding
would make "the run worked" and "the news was good" the same signal."""


# The self-check table's columns, restricted to what every retention record
# carries -- `arm`, `seed`, `step`, the windows and clusters it was scored
# over, and whether this script's own measure pass reproduced the cell's
# diagnostic. A real record's `self_check` also carries the two curve deltas
# `trust_horizon.SelfCheck.record()` writes, but nothing downstream of `read`
# needs them re-printed here, and pinning this table to fields every record
# actually has (rather than to the richer schema `measure_cell` happens to
# produce) is what keeps a minimal fixture a genuine drive through this
# script rather than a fixture shaped to fit a wider table.
SELF_CHECK_COLUMNS: tuple[str, ...] = ("arm", "seed", "step", "windows", "clusters", "ok")
SELF_CHECK_WIDTHS: tuple[str, ...] = ("<12", ">5", ">8", ">9", ">10", ">5")


def _table_line(values, widths) -> str:
    return "  " + "".join(
        f"{value!s:{spec}}" for value, spec in zip(values, widths, strict=True)
    )


def _self_check_table(records: dict) -> str:
    """What this script's own measure pass reproduced, per cell -- printed
    first, so a reader meets the instrument before the ladder built on it."""
    lines = [
        "--- self-check per record: this script's measure pass against the cell's "
        "diagnostic, and the windows it was scored on ---",
        _table_line(SELF_CHECK_COLUMNS, SELF_CHECK_WIDTHS),
    ]
    for (arm, seed), record in sorted(records.items()):
        lines.append(_table_line((
            arm,
            int(seed),
            int(record["step"]),
            len(record["windows"]["episode"]),
            int(record["clusters"]),
            "yes" if record["self_check"]["ok"] else "NO",
        ), SELF_CHECK_WIDTHS))
    return "\n".join(lines)


def _position_r2(record: dict, cell: tuple[str, int]) -> float:
    """`base_control["position_r2"]`, named -- not a bare `KeyError` -- for a
    record written before `position_r2` shipped (Correction 3), unlike this
    function's named-refusal neighbours (`load_retention`,
    `require_one_protocol`)."""
    base = record["base_control"]
    if "position_r2" not in base:
        raise ValueError(
            f"{cell[0]} seed {cell[1]}: this record's base_control has no "
            "'position_r2' -- it predates Correction 3 and must be re-measured "
            "before this reading can report the position companion"
        )
    return float(base["position_r2"])


def _position_companion(records: dict) -> str:
    """This script's own companion to the base control line printed above it
    -- Correction 3.

    `base_control`'s `r2` stays the GATED 4-column mean (position and heading
    mixed), calibrated against `latent_selection_r2`'s scale; re-pointing the
    gate at `position_r2` alone would silently change what `BASE_R2_FLOOR`
    means. But `EXIT_BASE_UNRESOLVED`'s own docstring and this reading's rule
    text both describe the gate as being about whether the current frame can
    say "absolute position", while the gated `r2` mixes heading into that
    claim -- so a cell whose `r2` clears the floor while `position_r2` does
    not (plausible for a frozen single-frame backbone, where heading can be
    far easier to read than map position) is a fact the results must REPORT,
    not one the mean is allowed to hide.

    Read PER CELL (arm, seed) here, not off the aggregated `BaseControl` on
    `RetentionInputs`: the aggregate carries only a seed TALLY against the
    gate, and averaging `position_r2` first would wash out exactly the
    divergence this function exists to catch, the same way the 4-column mean
    already washes out position against heading.
    """
    arms = sorted({arm for arm, _ in records})

    def cells_for(arm: str):
        return [(a, s) for a, s in sorted(records) if a == arm]

    line = "  position control (enc(t) -> absolute position, x/y only; the same fitted " \
        "probe's r2 against position alone, reported beside the gated 4-column r2 above): " \
        + ", ".join(
            f"{arm} position_r2="
            f"{np.mean([_position_r2(records[cell], cell) for cell in cells_for(arm)]):+.3f}"
            for arm in arms
        )
    warnings = []
    for (arm, seed), record in sorted(records.items()):
        r2 = float(record["base_control"]["r2"])
        position_r2 = _position_r2(record, (arm, seed))
        if r2 > BASE_R2_FLOOR and not position_r2 > BASE_R2_FLOOR:
            warnings.append(
                f"  WARNING: {arm} seed {seed} clears BASE_R2_FLOOR ({BASE_R2_FLOOR:.2f}) on "
                f"the gated r2 ({r2:+.3f}) but its position_r2 ({position_r2:+.3f}) does not "
                "-- the 4-column mean here is carried by heading, not by absolute position; "
                "see base_control's docstring"
            )
    return "\n".join([line, *warnings])


def retention_text(records: dict, inputs: RetentionInputs, reading: RetentionStatus) -> str:
    """Everything `read` prints, in the order a reader should meet it: the
    self-check, the ladder in full, Reading E's verdict, and this script's own
    position-control companion (Correction 3) -- written to `retention.txt`
    and printed to stdout byte for byte the same string.

    THE LADDER PRINTS BEFORE THE VERDICT, on purpose: a reader meets the
    evidence before the conclusion, `motion.txt`'s own order.
    """
    return "\n\n".join([
        _self_check_table(records),
        format_ladder(inputs),
        format_reading_retention(reading, inputs),
        _position_companion(records),
    ]) + "\n"


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def read_phase(args) -> int:
    """Refuse a plan that cannot reach the verdict, pool every requested cell
    (11 names the first missing), decide Reading E, write `retention.txt`,
    print the same text, and return `READ_EXITS.get(reading.status, EXIT_OK)`
    -- 0 for every reading, 39 or 40 for the two refusals.
    """
    require_readable_plan(args.arms, args.seeds)
    try:
        records = load_retention(args.out, args.arms, [int(s) for s in args.seeds])
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    require_one_protocol(records)
    inputs = retention_inputs(records)
    reading = reading_retention(inputs)
    text = retention_text(records, inputs, reading)
    write_text(args.out / "retention.txt", text)
    print(text, end="")
    return READ_EXITS.get(reading.status, EXIT_OK)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3j_retention"))
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


def main(argv: list[str] | None = None, *, ks=K_REPORTED) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    device = get_device(prefer=args.device)
    cells = [(arm, int(seed)) for arm in args.arms for seed in args.seeds]
    if args.phase in ("measure", "all"):
        buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
        train, val = episode_split(
            buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
        )
        status = measure_phase(args, cells, device, train, val, ks=ks)
        if status != EXIT_OK:
            return status
    if args.phase in ("read", "all"):
        status = read_phase(args)
        if status != EXIT_OK:
            return status
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
