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

import importlib.util
from pathlib import Path

import numpy as np

from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.probe import (
    GainSplit, fit_probe, gain_from_blocks, gather_probe_data, probe_r2,
)
from mbfps.eval.retention import (
    CONFIDENCE, DECISION_K, K_REPORTED, RESAMPLES, RUNGS, TARGETS,
    backward_rotation, backward_translation, rung_block, shifted_rows,
)
from mbfps.eval.study import git_sha, write_record


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
        gather(val, seed + 1, limit=FIT_EPISODES),
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
                ks=K_REPORTED) -> dict:
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
    """
    ladder: dict = {target: {} for target in TARGETS}
    rows_by_k: dict[str, int] = {}
    for k in ks:
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
            rows_by_k[k_key(k)] = int(row_sets["score"].size)
            ladder[target][k_key(k)] = {
                rung: gain_from_blocks(
                    splits[rung]["fit"], splits[rung]["select"], splits[rung]["score"],
                    groups=groups, resamples=RESAMPLES, confidence=CONFIDENCE,
                )
                for rung in RUNGS
            }
    return {"ladder": ladder, "rows": rows_by_k}


def base_control(fit: dict, select: dict | None, score: dict) -> dict:
    """`enc(t)` -> absolute position, as an r2 LEVEL on the validation windows.

    The positive control M3i measured only during its final review and recorded
    nowhere; its section 9 provenance note asked a successor to build it into the
    measure phase, and this is that. Every row is used -- there is no backward
    shift, because the target is the frame's own state.

    A level, not a gain: this is the arm every gain in `cell_ladder` is measured
    against, so the only question is whether it reads at all.
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
    return {
        "r2": probe_r2(probe, score_x, score_y),
        "ridge": probe["ridge"],
        "ridge_selected": select is not None,
        "rows": int(score_y.shape[0]),
    }


# ---------------------------------------------------------------------------
# measure: one cell, then every cell.
# ---------------------------------------------------------------------------


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
    status, prepared = prepare_cell(args, cell, device, train, val)
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
    ladder = cell_ladder(fit, select, score, h_dim=h_dim, ks=ks)
    record = {
        "arm": cell.arm, "seed": cell.seed, "step": int(cell.record["steps"]),
        "device": str(device), "context": prepared.context, "horizon": prepared.horizon,
        "h_dim": h_dim,
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
    """
    try:
        loaded = [load_cell(args.out, arm, seed) for arm, seed in cells]
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
            headline = f"translation/deterministic gain at k={DECISION_K} {det['gain']:+.3f}"
        else:
            headline = f"k={DECISION_K} not reported"
        print(
            f"{cell.arm} seed {cell.seed}: base r2 {record['base_control']['r2']:.3f}; "
            f"{headline}; wrote {path}"
        )
    return EXIT_OK
