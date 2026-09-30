"""M3k: is the past frame's advantage over the recurrent state information, or
feature count? -- the measure phase.

M3j found `two_frame` (the encoder's embedding of a past frame, 2048 columns)
beating `deterministic` (the recurrent state `h`, 512 columns) at k = 15. That
result picks between two 13.5-hour retrains, and it is confounded: a wider block
can win on feature count alone. `mbfps.eval.width` is the pure half of the
answer -- the pass constants, the projections, Reading F and its table -- and
owns none of torch, the model, or the record schema. This script is the impure
half.

Three passes of M3j's ladder over ONE gather per cell:

  shipped   every block at its native width. Reproduces M3j; the baseline.
  down      every block projected to 512, matching count AND rank. DECIDES.
  up        every block lifted to 2048, matching count only. CALIBRATES.

Only the probe fits differ between passes, which is what keeps this from
costing three times M3j: the encoder pass, the posterior and the windows are
gathered once per cell and every pass reads the same three arrays.

`projection(native, target)` returns `None` when the widths already match, so
each projecting pass carries a free known-answer anchor -- `deterministic` is
untouched in `down`, `two_frame` is untouched in `up` -- and each anchor's gain
must be BIT-IDENTICAL to the shipped pass, not merely close. `anchor_check`
reads that back and the record carries the result; `read` decides what
a broken anchor means.

  measure   per shipped cell: the three passes over both targets and every
            reported k, the contrast Reading F is taken on, the position
            control, the anchors. One record per cell.
  read      pools the nine records into Reading F -- the verdict -- and prints
            two companions that decide nothing: the width-bias table (what
            count alone is worth, `up` minus `shipped`) and Reading E re-run on
            the `down` pass under the corrected z-bearing set. Writes
            `width.txt`, byte for byte what it prints.

LOADING IS `trust_horizon.py`'S, exactly as `scripts/latent_retention.py` loads
it: `Cell`, `CellMissing`, `load_cell`, `self_check` and `prepare_cell` are
imported by path through `_sibling`, so a cell is refused here for the same
reasons and in the same words every other diagnostic refuses it, and this
module holds its own copy of every class `_sibling` defines -- never catch
another importer's copy of `CellMissing`.

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
  read:     EXIT_NO_CHECKPOINTS (11)   a requested cell has no width record.
            EXIT_BASE_UNRESOLVED (41)  `enc(t)` -> absolute position did not
                                        clear `BASE_R2_FLOOR`.
            EXIT_ANCHOR_BROKEN (42)    the `down` anchor did not reproduce.

0 / 11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose. 41 and 42
are in no other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10,
diagnose 11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages 34,
sharper_latent 35-37, latent_motion 38, latent_retention 39-40, argparse 2, a
traceback 1).

THE PLAN IS CHECKED BEFORE THE PROBE, not after it. `reading_contrast` raises
when it has fewer than `ARMS_REQUIRED` arms or `SEEDS_REQUIRED` seeds -- inside
the reading, i.e. after every gather and every fit. `scripts/latent_retention.py`
refuses a plan too narrow to read only in its `read` phase, so its `--phase all`
does the whole measure and then refuses. Here `measure_phase` refuses first
whenever a read will follow. A `--phase measure` run is NOT refused: Task 7's
smoke is one cell, and a narrow plan is a legitimate thing to MEASURE.
"""

import argparse
import importlib.util
import json
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
    GainSplit, apply_probe, contrast_from_blocks, fit_probe, gain_from_blocks,
    gather_probe_data, probe_r2,
)
from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, CONFIDENCE, K_REPORTED, RESAMPLES, RETENTION_FAMILY,
    RUNGS, SEEDS_REQUIRED, TARGETS, BaseControl, RetentionInputs, backward_rotation,
    backward_translation, format_ladder, format_reading_retention, reading_retention,
    rung_arm, rung_block, shifted_rows,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.eval.width import (
    ANCHOR, CONTRAST_K, DOWN_WIDTH, PASSES, PROJECTION_SEED, RUNG_WIDTH, UP_WIDTH,
    ContrastInputs, contrast_arm, format_reading_contrast, pass_block, reading_contrast,
)
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_latent_width", path)
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

EXIT_BASE_UNRESOLVED = 41
"""`enc(t)` -> absolute POSITION did not read. Every gain and every contrast is
measured against that base, so if the current frame cannot linearly say where it
is, no comparison between blocks means anything."""

EXIT_ANCHOR_BROKEN = 42
"""The `down` pass's anchor rung did not reproduce its shipped gain. `projection`
returns the identity when the native width already equals the target, so that
rung is untouched and its gain MUST be bit-identical. Anything else means the
projection machinery ran where it should not, and the pass Reading F is taken on
is unreadable."""

PHASES: tuple[str, ...] = ("measure", "read", "all")

FIT_EPISODES: int = 20
"""Training episodes the probe weights are fit on -- `probe.PROBE_EPISODE_LIMIT`,
the same set gate criterion 4 fits on, so the levels stay comparable."""

SELECT_EPISODES: int = 20
"""Training episodes AFTER the first `FIT_EPISODES` that select the ridge.

`probe.filtering_gain`'s default, and NOT a smaller number: that docstring
records the measurement -- a 4-episode selection split picks 1e5 for both arms
and reports +0.0325 where a 20-episode split picks 1e3 and recovers -0.0208, so
THE SIGN FLIPS on a one-step selection error. Shrinking this to save a gather
does not make the number noisier, it makes it wrong."""

CONTRAST_PASS: str = "down"
CONTRAST_TARGET: str = "translation"
"""Reading F is taken on the `down` pass, where the widths are matched, and on
the `translation` target, which is where M3j's objective-lever claim lives. A
contrast on the shipped pass would reproduce exactly the confound this milestone
exists to remove."""

CONTRAST_RUNGS: tuple[str, str] = ("two_frame", "deterministic")
"""`(a, b)` of `contrast_from_blocks(a, b)`, which is `a - b`, so a positive
contrast is `two_frame` ahead -- PAST_FRAME_AHEAD. These are the only two rungs
whose splits must still exist when the contrast is taken; `cell_passes` keeps
just these two alive past their own gain, and lets every other rung's go."""

_SPLITS: tuple[str, ...] = ("fit", "select", "score")


def width_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"width_{arm}_seed{seed}.json"


def k_key(k: int) -> str:
    """`15 -> "k15"`. A record key may not contain '.', since `write_record`
    addresses non-finite fields by dotted path."""
    return f"k{int(k)}"


TARGET_BUILDERS = {
    "translation": backward_translation,
    "rotation": backward_rotation,
}
"""`TARGETS` -> the builder that makes it. A dict rather than an if-chain so a
target named in `retention.TARGETS` and missing here fails at import time
rather than silently producing a ladder with a hole in it."""

assert set(TARGET_BUILDERS) == set(TARGETS), (
    f"TARGET_BUILDERS {sorted(TARGET_BUILDERS)} must name exactly retention.TARGETS "
    f"{sorted(TARGETS)}"
)
assert CONTRAST_TARGET in TARGETS and CONTRAST_PASS in PASSES
assert set(CONTRAST_RUNGS) <= set(RUNGS) and len(set(CONTRAST_RUNGS)) == 2


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
    thrown away four. M3j's smoke run caught it: 20 clusters and 9,261 rows at
    k = 1 against the 24 and 11,221 the spec accepts on. The protocol every
    milestone since M3d reads is 229 windows over 24 validation episodes, and
    that is what the reading must be taken on.
    """
    used = list(train)
    fit_paths = used[:FIT_EPISODES]
    if not fit_paths:
        raise ValueError("no training episodes to fit the width probes on")
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


def _target_rows(data: dict, target: str, k: int):
    """One split's row selection for one `(target, k)`: `(values, rows, source)`.

    `values` is the target on `rows`, and `source` is each row's partner `k`
    steps earlier -- what `two_frame` reads. Nothing here depends on the pass or
    on the rung, which is what lets `cell_passes` make this selection ONCE per
    `(target, k, split)` and hand it to every pass and every rung: they all
    describe the same rows because there is one selection, not because four
    separate ones happened to agree.
    """
    values, rows = TARGET_BUILDERS[target](
        data["targets"], data["window"], data["step"], k,
    )
    if rows.size == 0:
        raise ValueError(
            f"k={k}: no window is long enough to carry a backward displacement; "
            "a gain computed on zero rows would describe nothing"
        )
    shifted, source = shifted_rows(data["window"], data["step"], k)
    if rows.size == shifted.size and not np.array_equal(rows, shifted):
        # ONLY the equal-count case belongs here, DELIBERATELY. Dropping the
        # condition would subsume the count guard outright -- both targets
        # derive from the same `shifted_rows`, so a disagreement between them
        # implies at least one differs from `shifted` -- and would leave that
        # guard's test unable to reach it, which is the defect this file has
        # already been fixed for twice. So the unequal-count case is covered in
        # two places instead: `cell_passes`' count guard when the two targets
        # DISAGREE (named, per target, before any fit), and `GainSplit.rows()`
        # when they agree with each other but both differ from `shifted` -- that
        # last one only because `rung_block` reads `two_frame` from `source`, a
        # coupling this guard does not rely on for the equal-count half it owns.
        # This is the finer half: same count, different ORDER, which no count
        # can see and no length check can reach.
        raise ValueError(
            f"k={k}, target {target!r}: the builder's rows are not "
            "`shifted_rows`' rows, so row i of the target and row i of `source` "
            "describe different windows. `source` is derived here INDEPENDENTLY "
            "of the builder, and it is what `two_frame` reads -- pairing the two "
            "is only valid while they are the same selection in the same order. "
            "Such a builder passes the row-count agreement check, leaves "
            "`rows` in the record untouched, and silently misaligns "
            "`two_frame`'s block against its own target -- a wrong number in "
            "the deciding arm with no refusal anywhere"
        )
    return values, rows, source


def _base_for(data: dict, rows) -> np.ndarray:
    """`enc(t)` on `rows`: the base every rung, in every pass, is a gain against.

    It is NOT projected: matching applies to the rung's block only, so every
    gain stays on M3j's scale and the base R^2 is identical across passes by
    construction. It is read from `encoder_embedding`, the raw encoder output --
    never `embedding`, the model's PREDICTED embedding, which would ask whether
    the latent beats its own head's reconstruction.

    ONE ARRAY, SHARED BY EVERY GainSplit AT THIS `(target, k, split)` -- twelve
    fits use it (three passes x four rungs). Re-materialising it per rung as this
    used to did cost a fresh float64 copy of ~500 MB per rung per pass at the
    production sizes, for an array `pass_block` never touches. It is marked
    read-only for the reason `width.projection` marks its matrices: something
    shared by twelve consumers must not be mutable by any one of them.
    """
    base = np.asarray(data["encoder_embedding"], dtype=np.float64)[rows]
    base.flags.writeable = False
    return base


def _split_for(data: dict, selection, base: np.ndarray, rung: str, pass_name: str,
               h_dim: int) -> GainSplit:
    """One rung's `GainSplit` for one pass, on a selection and base built once."""
    values, rows, source = selection
    block = rung_block(data, rung, rows=rows, source=source, h_dim=h_dim)
    return GainSplit(
        base=base, block=pass_block(block, pass_name, seed=PROJECTION_SEED),
        target=values,
    )


def _measure_target(gathers, selection, groups, *, k: int, target: str, h_dim: int,
                    seed: int, resamples: int):
    """Every pass's four gains for one `(k, target)`, and the contrast when it
    is taken here. Returns `({pass: {rung: gain_dict}}, contrast_dict | None)`.

    A function of its own so that everything it holds -- the shared base, every
    block -- is released when it returns, before the next target's is built.

    RUNG BY RUNG: a rung's splits are built, its gain is taken, and they are let
    go, so at most one rung's blocks exist at a time -- except that the two
    contrast rungs are kept until `contrast_from_blocks` has run. The probe is
    called in exactly the order it always was (four gains in `RUNGS` order, then
    the contrast), and every call draws its bootstrap from its own `seed`, so
    nothing about the numbers depends on when a split was built.
    """
    bases = {name: _base_for(data, selection[name][1]) for name, data in gathers}
    by_pass: dict = {}
    contrast = None
    for pass_name in PASSES:
        takes_contrast = (
            k == CONTRAST_K and target == CONTRAST_TARGET and pass_name == CONTRAST_PASS
        )
        gains, kept = {}, {}
        for rung in RUNGS:
            splits = {name: None for name in _SPLITS}    # only ever the select split
            for name, data in gathers:
                splits[name] = _split_for(
                    data, selection[name], bases[name], rung, pass_name, h_dim,
                )
            gains[rung] = gain_from_blocks(
                splits["fit"], splits["select"], splits["score"],
                groups=groups, resamples=resamples, confidence=CONFIDENCE, seed=seed,
            )
            if takes_contrast and rung in CONTRAST_RUNGS:
                kept[rung] = splits
        by_pass[pass_name] = gains
        if takes_contrast:
            a, b = (tuple(kept[rung][name] for name in _SPLITS) for rung in CONTRAST_RUNGS)
            contrast = contrast_from_blocks(
                a, b, groups=groups, resamples=resamples, confidence=CONFIDENCE,
                seed=seed,
            )
    return by_pass, contrast


def cell_passes(fit: dict, select: dict | None, score: dict, *, h_dim: int,
                seed: int = 0, ks=K_REPORTED, resamples: int = RESAMPLES) -> dict:
    """Every `(pass, target, k, rung)` gain for one cell, plus the contrast.

    Returns `{"passes": {pass: {target: {k_key: {rung: gain_dict}}}},
    "contrast": {k_key: contrast_dict}, "rows": {k_key: int}}`.

    THREE PASSES OVER ONE GATHER: the arguments are the three arrays a single
    `gather_three_splits` produced, and only the probe fits differ between
    passes. Every rung at one `(target, k)` is handed the SAME base array -- one
    object, read-only, in every pass -- which is what makes all twelve gains
    differences against the same base level.

    THE CONTRAST is `two_frame - deterministic` on the `down` pass, the
    `translation` target and `CONTRAST_K` only. It is built from the very
    splits the `down` gains were fit on, so `a_r2` and `b_r2` ARE those two
    rungs' `joint_r2`. `contrast_from_blocks(a, b)` is `a - b`, so a positive
    contrast is `two_frame` ahead -- PAST_FRAME_AHEAD. A horizon list that does
    not contain `CONTRAST_K` produces no contrast rather than a contrast at
    some other horizon.

    `seed` IS THE CELL'S SEED, threaded into every `gain_from_blocks` and
    `contrast_from_blocks` call's own bootstrap draw -- `filtering_gain`'s
    convention exactly. A caller that left this at its default would hand every
    cell the SAME bootstrap draw, which correlates the interval noise across
    cells that the seeds x arms agreement rule treats as independent -- and that
    rule carries this milestone's entire multiple-comparison burden. Common
    random numbers across passes and rungs WITHIN one cell are intended (they
    are what makes the anchors' intervals reproduce as well as their gains); it
    is only across cells that the draws must differ. The seed is for the
    bootstrap ONLY: the projection is one matrix per `(native, target)` drawn
    from `PROJECTION_SEED` and shared by all nine cells.

    `resamples` is the bootstrap's draw count and defaults to the protocol's
    `RESAMPLES`; `measure_cell` never passes it, so a production run cannot
    depart from the pre-registered figure. It exists so a test that asserts on
    no interval can pay for fewer draws. That saves little -- the bootstrap is
    ~3% of a cell's time and the ridge solve is nearly all the rest -- so it is
    not a way to make a real run cheaper.

    The bootstrap groups on `episode`, never `window`.

    EVERY TARGET AT ONE k MUST AGREE ON THE SCORED ROW COUNT, and this function
    refuses a disagreement rather than choose one. `rows` is the figure the
    record reports and Task 7 cites, one per k, and each `TARGET_BUILDERS` entry
    is free to have a row rule of its own. Both current targets are built from
    the same `shifted_rows(window, step, k)`, so they agree today; a third with
    another rule would otherwise leave `rows` reporting whichever target was
    written last under that one key.

    `GainSplit.rows()` would incidentally refuse a row-DROPPING builder today --
    `two_frame`'s block is read from `shifted_rows`' `source`, so it no longer
    matches the shortened base -- but that refusal names no target, arrives at
    whichever fit reaches it first, and holds only while `rung_block` keeps
    reading `source`. This check leans on none of that: it is made from the row
    selections alone, before a single fit is paid for.

    The pass is not part of that check because it cannot be: `pass_block` is
    handed the rung's block and nothing else, and the selection is made once
    per `(target, k, split)` above the pass loop, so every pass reads the same
    rows by construction.
    """
    gathers = [
        (name, data) for name, data in zip(_SPLITS, (fit, select, score))
        if data is not None
    ]
    passes: dict = {p: {t: {} for t in TARGETS} for p in PASSES}
    contrast: dict = {}
    rows_by_k: dict[str, int] = {}
    for k in ks:
        selections = {
            target: {name: _target_rows(data, target, k) for name, data in gathers}
            for target in TARGETS
        }
        rows_by_target = {
            target: int(selections[target]["score"][1].size) for target in TARGETS
        }
        counts = set(rows_by_target.values())
        if len(counts) != 1:
            raise ValueError(
                f"k={k}: targets disagree on scored row count {rows_by_target}, but "
                "every target at one k is supposed to share one row set"
            )
        rows_by_k[k_key(k)] = counts.pop()
        for target in TARGETS:
            selection = selections[target]
            groups = np.asarray(score["episode"])[selection["score"][1]]
            by_pass, taken = _measure_target(
                gathers, selection, groups, k=k, target=target, h_dim=h_dim,
                seed=seed, resamples=resamples,
            )
            for pass_name, gains in by_pass.items():
                passes[pass_name][target][k_key(k)] = gains
            if taken is not None:
                contrast[k_key(k)] = taken
    return {"passes": passes, "contrast": contrast, "rows": rows_by_k}


def anchor_check(passes: dict) -> dict[str, bool]:
    """Did each projecting pass leave its anchor rung bit-identical?

    `ANCHOR[pass]` is the rung whose native width already equals that pass's
    target, so `projection` returns the identity and the rung is not touched.
    Exact equality, not `isclose`: an approximate match would mean a matmul ran.

    Checked at EVERY measured horizon and both targets rather than only at
    `CONTRAST_K` on translation, so a smoke run whose `ks` omit the contrast
    horizon still checks its anchors instead of raising a KeyError -- and so a
    projection that leaks at one horizon or one target and not another cannot
    hide.
    """
    shipped = passes["shipped"]
    horizons = sorted({key for target in TARGETS for key in shipped[target]})
    if not horizons:
        raise ValueError("no horizon was measured; there is no anchor to check")
    return {
        pass_name: all(
            passes[pass_name][target][key][rung]["gain"]
            == shipped[target][key][rung]["gain"]
            for target in TARGETS
            for key in horizons
        )
        for pass_name, rung in ANCHOR.items()
    }


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

    Every row is used -- there is no backward shift, because the target is the
    frame's own state. A level, not a gain: this is the arm every gain in
    `cell_passes` is measured against, so the only question is whether it reads
    at all. It is fit once per cell and is the same in every pass, because the
    base is never projected.

    `position_r2` IS THE GATED NUMBER (spec 2.5): the same fitted probe's r2
    against the first two target columns only, which is what
    `EXIT_BASE_UNRESOLVED` says it is about -- whether the current frame can
    say where it is. `BASE_R2_FLOOR` is deliberately NOT re-chosen; gated on
    position alone it is loose, and the spec records that rather than correcting
    it.

    `r2` is the 4-column mean over pos_x, pos_y, sin(angle), cos(angle) that
    M3j gated on. It is kept, not gated on, so this record bridges to M3j's.

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
    `--out` holds the width records and nothing else, so handing `args`
    through unchanged would look for the nine M3c checkpoints in the output
    directory and refuse every cell. Same shim, same reason, as
    `scripts/latent_motion.py` and `scripts/latent_retention.py`.
    """
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context, horizon=args.horizon,
    )


def measure_cell(args, cell: Cell, device, train, val, ks=K_REPORTED) -> tuple[int, dict | None]:
    """One cell: the checks, the three gathers, the three passes, the base control.

    12 and 14 are `prepare_cell`'s, so this script, `scripts/split_gap.py` and
    `scripts/latent_retention.py` all refuse a cell for the same reasons in the
    same words -- and the reproduction bound of M3h spec 2.4 is checked there
    for free, because it compares `evaluate_rollout`'s mean curves against the
    study record.

    30 is NOT free. It needs a SECOND pass -- `reference_trajectories`, the
    same one `scripts/latent_retention.py`'s own `measure_cell` takes -- because
    `self_check` reduces the PER-WINDOW rows, not the mean curves
    `prepare_cell` already checked. It refuses BEFORE the gathers, the
    expensive part.

    ONE `gather_three_splits` per cell serves all three passes.

    `(EXIT_OK, record)` once the record is built; `(status, None)` on any
    refusal, with the refusal printed. The record is not yet WRITTEN -- that
    is `measure_phase`'s job, so every cell's record picks up the same
    `git_sha` provenance and the same non-finite scan through one call to
    `write_record`. A broken anchor is RECORDED, not raised: `read` decides
    what it means, and a `measure` that raised would discard the evidence of
    which pass broke.
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
    # `seed=cell.seed`: the bootstrap draw is the CELL's -- see `cell_passes`.
    result = cell_passes(fit, select, score, h_dim=h_dim, seed=cell.seed, ks=ks)
    record = {
        "arm": cell.arm, "seed": cell.seed, "step": int(cell.record["steps"]),
        "record_git_sha": cell.record.get("git_sha", "unknown"),
        "device": str(device), "context": prepared.context, "horizon": prepared.horizon,
        "h_dim": h_dim,
        "ks": [int(k) for k in ks],
        "split_seed": SPLIT_SEED,
        "torch_version": torch.__version__,
        "passes": result["passes"],
        "contrast": result["contrast"],
        "rows": result["rows"],
        "anchors": anchor_check(result["passes"]),
        "base_control": base_control(fit, select, score),
        # What makes each pass reproducible: the one seed every matrix is drawn
        # from, and the native width each rung was projected FROM.
        "projection_seed": PROJECTION_SEED,
        "rung_width": {rung: RUNG_WIDTH[rung] for rung in RUNGS},
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


def require_readable_plan(arms, seeds) -> None:
    """Refuse a plan too narrow for Reading F to be read at all.

    The reading needs `ARMS_REQUIRED` arms, each summarised over at least
    `SEEDS_REQUIRED` seeds, and `reading_contrast` RAISES when it has fewer --
    inside the reading, after every gather and every fit. Asked here it costs
    nothing; asked there, `--arms frozen_ssl` is ~25 minutes of GPU work and a
    traceback. The same failure mode that got an earlier over-strict guard
    removed from `width.py`, and the check `scripts/latent_retention.py` makes
    before ITS probe -- except that this one also runs before the MEASURE
    whenever a read will follow.
    """
    if len(arms) < ARMS_REQUIRED or len(seeds) < SEEDS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(arms)} arm(s) x {len(seeds)} seed(s) cannot be read: "
            f"Reading F needs {ARMS_REQUIRED} arms, each over at least "
            f"{SEEDS_REQUIRED} seeds, and reading_contrast would raise only AFTER "
            "every gather and fit had been paid for"
        )


def _cell_line(record: dict, path: Path) -> str:
    """One line per measured cell: the position control, the contrast Reading F
    is taken on, and both anchors -- `up=BROKEN` in the spelling `width.txt`
    will use, so a broken anchor is unmissable in a long log."""
    contrast = record["contrast"].get(k_key(CONTRAST_K))
    if contrast is None:
        headline = f"k={CONTRAST_K} not reported"
    else:
        headline = (
            f"two_frame - deterministic on {CONTRAST_PASS} at k={CONTRAST_K} "
            f"{contrast['contrast']:+.4f} "
            f"[{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}]"
        )
    anchors = " ".join(
        f"{name}={'ok' if ok else 'BROKEN'}" for name, ok in record["anchors"].items()
    )
    return (
        f"{record['arm']} seed {record['seed']}: position r2 "
        f"{record['base_control']['position_r2']:.3f}; {headline}; "
        f"anchors {anchors}; wrote {path}"
    )


def measure_phase(args, cells, device, train, val, ks=K_REPORTED) -> int:
    """Every requested cell, refusing the whole run before any pass when the
    plan cannot be read (a plan check, only when a read follows) or a cell is
    missing (11).

    `measure_cell` builds each record; this function writes it through the
    same `write_record` `scripts/latent_retention.py` uses -- so the non-finite
    scan and the `git_sha` provenance are shared -- prints one line per cell,
    and returns the first non-`EXIT_OK` status or `EXIT_OK`.

    THE PLAN CHECK FIRES ONLY WHEN `args.phase == "all"`. `read` refuses its own
    plan, and `--phase measure` is allowed a plan `read` could not
    read: the smoke is one cell. Both are read off `cells` -- the distinct arms
    and the distinct seeds actually about to be measured.

    THAT MAKES `args.phase` LOAD-BEARING, so a phase that is not one of `PHASES`
    -- absent, `None`, or a typo -- is REFUSED here. Read with a default, a
    missing `phase` is indistinguishable from `measure`: the plan check is
    skipped without a word and `--arms frozen_ssl` becomes ~25 minutes of GPU
    work ending in a traceback, which is exactly what the check exists to
    prevent.

    CELLS ARE LOADED FROM `args.source`, NOT `args.out`. `--out` is this
    script's own record directory -- where width records are WRITTEN, below --
    and the nine M3c checkpoints, study records and diagnostics live in the
    study directory instead. Loading from `args.out` would look for those three
    files in a directory that starts out empty and fail before a single cell
    measured.
    """
    phase = getattr(args, "phase", None)
    if phase not in PHASES:
        raise SystemExit(
            f"unknown phase {phase!r}: expected one of {PHASES}. The plan check is "
            "made only for --phase all, so a phase that is missing or misspelled "
            "would skip it in silence"
        )
    if phase == "all":
        require_readable_plan(
            sorted({arm for arm, _ in cells}), sorted({seed for _, seed in cells}),
        )
    try:
        loaded = [load_cell(args.source, arm, seed) for arm, seed in cells]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in loaded:
        status, record = measure_cell(args, cell, device, train, val, ks=ks)
        if status != EXIT_OK:
            return status
        record["git_sha"] = git_sha()
        path = width_record_path(args.out, cell.arm, cell.seed)
        write_record(path, record)
        print(_cell_line(record, path))
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: what pooling these records requires.
# ---------------------------------------------------------------------------


_PROTOCOL_FIELDS = (
    ("windows.episode", lambda r: list(r["windows"]["episode"])),
    ("episodes.val", lambda r: list(r["episodes"]["val"])),
    ("context", lambda r: int(r["context"])),
    ("horizon", lambda r: int(r["horizon"])),
    ("device", lambda r: str(r["device"])),
    ("ks", lambda r: list(r.get("ks", []))),
    ("torch_version", lambda r: str(r.get("torch_version", ""))),
    ("git_sha", lambda r: str(r.get("git_sha", ""))),
    ("projection_seed", lambda r: int(r["projection_seed"])),
    ("rung_width", lambda r: dict(r["rung_width"])),
    ("h_dim", lambda r: int(r["h_dim"])),
)
"""Every field `require_one_protocol` compares, as `(name, pick)`, in one
module-level table that the function ITERATES and a test reads -- so a field
cannot be added to the comparison without the test noticing it has no
disagreement case. While the tuple lived inside the function, no test could see
it, and a check on the test's own literal compared that literal to itself.

`h_dim` is here because it is the one width the record MEASURES -- read off the
checkpoint -- while `projection_seed` and `rung_width` are written from module
constants and so cannot vary between two records of one code version. Nine
cells whose checkpoints disagreed on `h_dim` would go through DIFFERENT
projection matrices while carrying identical `rung_width`, and would pool
without a word: the only thing that would notice is the `down` anchor breaking,
which the read phase reports as "the projection machinery ran where it should
not" -- the wrong diagnosis, reached after the whole measure. An `embed_dim`
disagreement breaks only the `up` anchor, which deliberately gates nothing, and
a `z_dim` disagreement is invisible to every other check."""


def require_one_protocol(records: dict) -> None:
    """Every record reports the same protocol on the same windows, through the
    same matrices, or the two that disagree are named with the field.

    Nine cells pooled into one reading must describe the same rows: two
    protocols pooled as one would be a reading over a union nothing measured.
    Mirrors `scripts/latent_retention.py`'s namesake, and its eight fields are
    ALL here -- `ks` and `torch_version` as well as the windows/protocol fields,
    and `git_sha`, whose absence let an earlier version of that function pool
    records from different torch builds and code versions into one finding with
    no refusal at all. The three that older records may lack are read with
    `.get(...)` so a record without them is defaulted identically across every
    record rather than raising a bare `KeyError`.

    TWO MORE ARE THIS MILESTONE'S. `projection_seed` and `rung_width`: the
    passes are only comparable across cells because they went through the SAME
    random matrices from the SAME native widths, which is the whole reason the
    projection is one fixed draw shared by all nine cells. Records that differ
    on either are not one measurement. Both are indexed directly -- a width
    record always carries them, and a record without them cannot say which
    matrices it used.

    `clusters` and `rows` are NOT compared HERE, and they are not unguarded: they
    are print-only (no reading can flip on them), but `contrast_inputs` takes ONE
    value of each for a caption that speaks for nine cells, so it refuses a
    disagreement itself, at the point where the pick is made, rather than
    depending on this function having run first.
    """
    items = sorted(records.items())
    if not items:
        raise SystemExit("no width record to read")
    (first_cell, first) = items[0]
    for cell, record in items[1:]:
        for field, pick in _PROTOCOL_FIELDS:
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


# ---------------------------------------------------------------------------
# read: the nine records pooled into Reading F.
# ---------------------------------------------------------------------------

READ_EXITS = {
    "UNRESOLVED_BASE": EXIT_BASE_UNRESOLVED,
    "UNRESOLVED_ANCHOR": EXIT_ANCHOR_BROKEN,
}
"""Status -> exit code, for the two statuses that are refusals. Every other
status is a reading and exits 0: a milestone that exited non-zero on a finding
would make "the run worked" and "the news was good" the same signal."""

RETENTION_PASS: str = "down"
CORRECTED_Z_BEARING: tuple[str, ...] = ("stochastic",)
"""The corrected Reading E: re-run on the pass where widths are matched, with the
z-bearing set M3j's own results recorded as the right one. `full` is `h (+) z`, so
a clearing `full` cannot attribute anything to `z`. Passed EXPLICITLY, and
`retention.Z_BEARING_RUNGS` keeps its value: M3j's records read under the rule
they were taken under."""

SELF_CHECK_COLUMNS: tuple[str, ...] = (
    "arm", "seed", "step", "windows", "gathered", "clusters", "ok",
)
SELF_CHECK_WIDTHS: tuple[str, ...] = ("<12", ">5", ">8", ">9", ">10", ">10", ">5")

BIAS_COLUMNS: tuple[str, ...] = ("target", "rung", "width")
BIAS_WIDTHS: tuple[int, ...] = (13, 15, 7)
BIAS_VALUE_WIDTH: int = 13
"""The width-bias table's leading columns and the width of every value column
after them -- one per arm, then `all`. The arms are the run's, so the columns
are built from the records rather than declared."""


def _table_line(values, widths) -> str:
    return "  " + "".join(
        f"{value!s:{spec}}" for value, spec in zip(values, widths, strict=True)
    )


def _cell_name(cell: tuple[str, int]) -> str:
    return f"{cell[0]} seed {cell[1]}"


def _get(record: dict, cell: tuple[str, int], *path):
    """`record[path[0]][path[1]]...`, or a refusal naming the cell and the path.

    Every field the reading needs is read through here, so a record that lacks
    one -- a smoke run that omitted `k15`, a record from before a field existed --
    stops the read with the cell and the dotted path, rather than surfacing as a
    bare `KeyError` after the reader has paid for the measure."""
    value = record
    for key in path:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            raise SystemExit(
                f"{_cell_name(cell)}: the record has no {'.'.join(str(p) for p in path)}; "
                "it cannot be read as Reading F's input and must be re-measured "
                "(--phase measure)"
            ) from None
    return value


def _finite(record: dict, cell: tuple[str, int], *path) -> float:
    """`_get`, coerced to float and refused if it is not finite. A NaN compares
    False against every threshold, so it would read as "did not clear" AND "did
    not fail" at once -- an error about the measurement, never a finding."""
    raw = _get(record, cell, *path)
    try:
        number = float(raw)
    except (TypeError, ValueError):
        number = float("nan")
    if not np.isfinite(number):
        raise SystemExit(
            f"{_cell_name(cell)}: {'.'.join(str(p) for p in path)} is {raw!r}, not a "
            "finite number; that is an error about the measurement, not a reading"
        )
    return number


def _cells_by_arm(records: dict) -> dict[str, list[tuple[str, int]]]:
    """Each arm's cells, seeds ascending, arms in sorted order."""
    grouped: dict[str, list[tuple[str, int]]] = {}
    for cell in sorted(records):
        grouped.setdefault(cell[0], []).append(cell)
    return grouped


def _one_value(records: dict, pick, what: str):
    """The one value every record reports for `what`, or a refusal naming the
    cells that differ.

    For the fields a caption prints ONCE on behalf of nine cells. Taking the
    first record's would be one member standing for the set: a caption saying
    `8015 rows over 24 clusters` over records that do not agree on either."""
    by_value: dict[str, list[tuple[str, int]]] = {}
    values: dict[str, object] = {}
    for cell, record in sorted(records.items()):
        value = pick(record, cell)
        # Canonical JSON, so two dicts with the same items in another insertion
        # order are ONE value rather than a disagreement.
        label = json.dumps(value, sort_keys=True)
        by_value.setdefault(label, []).append(cell)
        values[label] = value
    if len(by_value) != 1:
        detail = "; ".join(
            f"{values[label]!r} in {', '.join(_cell_name(c) for c in cells)}"
            for label, cells in by_value.items()
        )
        raise SystemExit(
            f"the records disagree on {what}: {detail}. A caption prints one "
            f"{what} for the whole set, so records that differ are not one "
            "measurement"
        )
    return next(iter(values.values()))


def pooled_anchors(records: dict) -> dict[str, bool]:
    """Each projecting pass's anchor across ALL the cells: it held only if it held
    in every one. `all`, not `any` and not one record standing for the set -- a
    single cell whose anchor broke is a cell in which the projection machinery ran
    where it should not.

    A flag that is not a bool is refused: `"false"` is truthy, and a damaged
    record would otherwise read as an anchor that HELD."""
    pooled = {}
    for name in ANCHOR:
        flags = []
        for cell, record in sorted(records.items()):
            flag = _get(record, cell, "anchors", name)
            if not isinstance(flag, bool):
                raise SystemExit(
                    f"{_cell_name(cell)}: anchors.{name} is {flag!r}, not a boolean; "
                    "a non-boolean would be read as truthy and pass a broken anchor"
                )
            flags.append(flag)
        pooled[name] = all(flags)
    return pooled


def base_controls(records: dict) -> dict[str, BaseControl]:
    """Each arm's base control, GATED ON `position_r2`.

    `BaseControl` carries one number called `r2` beside the seed tally, and
    `format_reading_contrast` prints them together as `r2=+0.650 3/3`. This is
    where they are made to be the same quantity: `r2` is the arm's mean POSITION
    r2 and `seeds_clear` counts seeds whose POSITION r2 exceeds `BASE_R2_FLOOR`.
    The record's other number, `base_control.r2` -- the 4-column mean over
    position and heading that M3j gated on -- is read nowhere here and is printed
    nowhere: a tally printed beside a figure it was not read from is how M3h
    shipped an `up` tally beside a verdict read from `down`.

    The tally is per SEED, then per arm (`BaseControl.clears`): it is not a
    threshold on the arm's mean. `BASE_R2_FLOOR` stays 0.10 and is not re-chosen;
    gated on position alone it is loose, and the spec records that rather than
    correcting it. Strict `>`, as `retention` applies it."""
    controls = {}
    for arm, cells in _cells_by_arm(records).items():
        levels = [_finite(records[c], c, "base_control", "position_r2") for c in cells]
        controls[arm] = BaseControl(
            r2=float(np.mean(levels)),
            seeds_clear=sum(1 for level in levels if level > BASE_R2_FLOOR),
            seeds_total=len(levels),
        )
    return controls


def _pooled_shape(records: dict) -> tuple[int, dict[str, int]]:
    """`(clusters, rows by k_key)` -- ONE of each, refused unless every cell agrees.

    `clusters` and `rows` gate nothing: no verdict can flip on them. They are
    still one caption's worth of numbers standing for nine cells, and a caption
    that says `8015 rows over 24 clusters` must be true of every one."""
    clusters = _one_value(
        records, lambda r, c: int(_get(r, c, "clusters")), "clusters",
    )
    rows = _one_value(
        records,
        lambda r, c: {str(k): int(v) for k, v in _get(r, c, "rows").items()},
        "rows",
    )
    return clusters, rows


def _require_readable_records(by_arm: dict) -> None:
    """Refuse a record set `reading_contrast` would raise on, by name.

    Fewer than `ARMS_REQUIRED` arms, or arms with different seed counts (no true
    "N of M seeds" rule) -- each asked of the RECORDS, so a direct caller of
    `contrast_inputs` is refused the same way `read_phase` is.

    An arm with fewer than `SEEDS_REQUIRED` seeds is NOT checked here:
    `contrast_arm` refuses it, and `contrast_inputs` names the arm when it does.
    A second check for it would be unreachable -- nothing could fail with it
    removed -- which is the guard-that-subsumes-a-guard defect this file has
    already been fixed for twice."""
    counts = {arm: len(cells) for arm, cells in by_arm.items()}
    if len(by_arm) < ARMS_REQUIRED:
        raise SystemExit(
            f"the records cover {len(by_arm)} arm(s) ({', '.join(sorted(by_arm))}); "
            f"Reading F needs at least {ARMS_REQUIRED}"
        )
    if len(set(counts.values())) != 1:
        raise SystemExit(
            f"the arms do not share one seed count {counts}; the '{SEEDS_REQUIRED} of N "
            "seeds' rule would be true of some rows and false of others"
        )


def contrast_inputs(records: dict) -> ContrastInputs:
    """Pool the nine records into Reading F's input.

    The seed tally comes from the per-seed intervals in the records, not from a
    pooled estimate: R^2 is not a per-window quantity, so there is nothing to
    pool the way a paired contrast pools. The agreement requirement IS the rule.

    EVERY PRECONDITION `reading_contrast` ENFORCES IS ESTABLISHED HERE, as a
    named refusal rather than a traceback out of the reading: at least
    `ARMS_REQUIRED` arms; each arm with the same seed count, at least
    `SEEDS_REQUIRED`; an `anchors` whose keys are exactly `ANCHOR`'s (built by
    iterating it, so `shipped` and a stray key cannot appear, and `down` cannot
    be missing); a `base` naming exactly the arms (built from the same records).
    The two ambiguities that remain -- an arm clearing both ways, and both
    directions at the bar -- are refused by the reading itself, and only past its
    two gates; `read_phase` turns those into a named refusal too. They are NOT
    checked here, because refusing them ahead of the gates would turn a
    legitimate UNRESOLVED_BASE or UNRESOLVED_ANCHOR record into a crash.

    `clusters` and `rows` are print-only, and each is ONE number for a caption
    that speaks for nine cells, so a disagreement between records is refused
    (`_pooled_shape`) rather than sampled from whichever record sorts first."""
    key = k_key(CONTRAST_K)
    by_arm = _cells_by_arm(records)
    _require_readable_records(by_arm)
    arms = {}
    for arm, cells in by_arm.items():
        seeds = [c[1] for c in cells]
        try:
            arms[arm] = contrast_arm([_get(records[c], c, "contrast", key) for c in cells])
        except ValueError as error:
            raise SystemExit(f"{arm} (seeds {seeds}): {error}") from error
    clusters, rows = _pooled_shape(records)
    try:
        at_contrast_k = rows[key]
    except KeyError:
        raise SystemExit(
            f"the records carry no rows.{key}; Reading F is taken at k = {CONTRAST_K}"
        ) from None
    return ContrastInputs(
        arms=arms, base=base_controls(records), anchors=pooled_anchors(records),
        clusters=clusters, rows=at_contrast_k,
    )


def _reading_e_unreadable(records: dict) -> str | None:
    """Why Reading E cannot be taken from these records, or `None`.

    Both are limits of the PLAN, not defects in a record: `reading_retention` is
    defined over exactly `RETENTION_FAMILY` arms and over every `K_REPORTED`
    horizon. Reading F needs neither -- two arms clear its bar, and it is taken at
    k = 15 alone -- so the companion says it cannot be read instead of vetoing a
    verdict that can."""
    arms = sorted({arm for arm, _ in records})
    if len(arms) != RETENTION_FAMILY:
        return (
            f"Reading E is defined over exactly {RETENTION_FAMILY} arms and these "
            f"records carry {len(arms)} ({', '.join(arms)})"
        )
    missing = sorted({
        k for record in records.values() for k in K_REPORTED
        if any(
            k_key(k) not in record.get("passes", {}).get(RETENTION_PASS, {}).get(target, {})
            for target in TARGETS
        )
    })
    if missing:
        return (
            f"Reading E is a disjunction over k = {', '.join(str(k) for k in K_REPORTED)} "
            f"and these records lack k = {', '.join(str(k) for k in missing)}"
        )
    return None


def retention_inputs(records: dict, *, pass_name: str = RETENTION_PASS) -> RetentionInputs:
    """M3j's `RetentionInputs`, built from one pass of the width records.

    `ladder[target][k][rung][arm]` is `rung_arm` over that arm's own seeds' gains
    on `pass_name` -- the `down` pass by default, where every block is 512 wide.
    The base is `base_controls` -- the SAME position-gated objects Reading F
    reads, because both corrections M3j recorded are adopted (spec 2.5) and a
    second gate on the 4-column mean beside the first would let the two readings
    disagree about whether the instrument works.

    `rung_arm` raises on a non-finite gain or bound, a missing key, an inverted
    interval; that is an error about the measurement, and it is turned into a
    refusal naming the cell of the ladder. Assumes `_reading_e_unreadable` is
    `None`."""
    by_arm = _cells_by_arm(records)
    ladder: dict = {}
    for target in TARGETS:
        ladder[target] = {}
        for k in K_REPORTED:
            ladder[target][k] = {}
            for rung in RUNGS:
                ladder[target][k][rung] = {}
                for arm, cells in by_arm.items():
                    gains = [
                        _get(records[c], c, "passes", pass_name, target, k_key(k), rung)
                        for c in cells
                    ]
                    try:
                        ladder[target][k][rung][arm] = rung_arm(gains)
                    except ValueError as error:
                        raise SystemExit(
                            f"{arm} on the {pass_name} pass, {target} at k = {k}, rung "
                            f"{rung} (seeds {[c[1] for c in cells]}): {error}"
                        ) from error
    clusters, rows = _pooled_shape(records)
    return RetentionInputs(
        ladder=ladder, base=base_controls(records), clusters=clusters,
        rows={k: rows[k_key(k)] for k in K_REPORTED},
    )


def corrected_reading_e(records: dict) -> str:
    """Reading E re-run on the `down` pass under `CORRECTED_Z_BEARING`, as the text
    `width.txt` carries: its heading, the whole ladder, then Reading E.

    A COMPANION THAT DECIDES NOTHING, and one owed to M3j: remove `full` from the
    z-bearing set and M3j's MOTION_RETAINED hangs entirely on `stochastic`, so the
    corrected, width-matched reading may differ from M3j's status. It would not
    overturn it -- M3j's verdict was taken under its own rule -- it would mean the
    two statuses differ for reasons documented before the numbers were seen.

    The heading says which pass and which corrections, because `retention`'s own
    formatter does not: its captions name neither."""
    heading = (
        f"--- Companion, decides nothing: Reading E re-run on the `{RETENTION_PASS}` pass "
        f"(every block {DOWN_WIDTH} columns wide), with z-bearing rungs "
        f"{CORRECTED_Z_BEARING} and the base control gated on position alone -- the two "
        "corrections M3j's results recorded. M3j's own status was taken under its own "
        "rule and stands as recorded; the two may differ, for reasons pre-registered "
        "before the numbers were seen ---"
    )
    why = _reading_e_unreadable(records)
    if why is not None:
        return f"{heading}\n  unreadable: {why}"
    inputs = retention_inputs(records)
    reading = reading_retention(inputs, z_bearing=CORRECTED_Z_BEARING)
    return "\n\n".join([heading, format_ladder(inputs), format_reading_retention(reading, inputs)])


def width_bias_table(records: dict) -> str:
    """What feature COUNT alone is worth: `up` minus `shipped`, per rung and target,
    at `CONTRAST_K`.

    A companion that decides nothing. `up` lifts every block to `UP_WIDTH` by a
    fixed random matrix -- count matched, rank and information unchanged -- so
    the difference is what a wider block buys with nothing added to it. Spec 1.1
    measured about +0.0137 for `h` on two cells; this puts it on every cell and
    every rung. Per arm (mean over its seeds) and over all cells.

    THE `up` ANCHOR GATES IT. `two_frame` is already `UP_WIDTH` wide, so its `up`
    gain must equal its shipped one exactly; if it did not, in any cell, the lift
    is not what it claims and calibrates nothing, so NO number prints. The
    decision is read off `pooled_anchors` -- the same object that prints `up=BROKEN`
    beside Reading F -- so the table and the anchors line cannot disagree. The
    `down` anchor does not gate this table: it gates Reading F.

    `width` is the native width each rung was lifted FROM, read off the records
    (which must agree) and not off `RUNG_WIDTH`: it is what was measured."""
    key = k_key(CONTRAST_K)
    arms = sorted({arm for arm, _ in records})
    heading = (
        f"--- The width bias: what feature count alone is worth. `up` (every block lifted "
        f"to {UP_WIDTH} columns by a fixed random matrix: count matched, rank and "
        f"information unchanged) minus `shipped` (native widths), gain over enc(t) at "
        f"k = {CONTRAST_K}, mean over each arm's seeds and over all {len(records)} cells; "
        f"decides nothing, and the {ANCHOR['up']} row is `up`'s anchor and reads exactly "
        "zero while the lift is sound ---"
    )
    if not pooled_anchors(records)["up"]:
        broken = sum(1 for cell, record in records.items() if not record["anchors"]["up"])
        return (
            f"{heading}\n  unreadable: the up pass's anchor ({ANCHOR['up']}) did not "
            f"reproduce its shipped gain in {broken} of {len(records)} cells, so the lift "
            "is not doing what it claims and calibrates nothing; Reading F is taken on "
            "the down pass and is unaffected"
        )
    widths = _one_value(records, lambda r, c: dict(_get(r, c, "rung_width")), "rung_width")
    columns = (*BIAS_WIDTHS, *([BIAS_VALUE_WIDTH] * (len(arms) + 1)))

    def gain(cell, pass_name, target, rung) -> float:
        return _finite(records[cell], cell, "passes", pass_name, target, key, rung, "gain")

    lines = [heading, _table_line((*BIAS_COLUMNS, *arms, "all"), [f">{w}" for w in columns])]
    for target in TARGETS:
        for rung in RUNGS:
            bias = {
                cell: gain(cell, "up", target, rung) - gain(cell, "shipped", target, rung)
                for cell in sorted(records)
            }
            per_arm = [
                float(np.mean([bias[c] for c in cells]))
                for cells in _cells_by_arm(records).values()
            ]
            lines.append(_table_line(
                (target, rung, widths[rung],
                 *(f"{v:+.4f}" for v in per_arm), f"{np.mean(list(bias.values())):+.4f}"),
                [f">{w}" for w in columns],
            ))
    return "\n".join(lines)


def _window_count(record: dict, cell: tuple[str, int]) -> int:
    """Distinct `(episode, window)` pairs the record was scored on.

    `windows.episode` and `windows.window` are PER GATHERED ROW, not per window --
    a real record's lists are 11,450 long over 229 windows -- so their length is
    a count of rows. `latent_retention`'s table prints that length under the
    header `windows`, which is a caption disagreeing with its column; this table
    counts what the header says, and prints the gathered rows under their own."""
    episode = _get(record, cell, "windows", "episode")
    window = _get(record, cell, "windows", "window")
    return len(set(zip(episode, window, strict=True)))


def _self_check_table(records: dict) -> str:
    """What this script's own measure pass reproduced, per cell -- printed first,
    so a reader meets the instrument before anything built on it."""
    lines = [
        "--- self-check per record: this script's measure pass against the cell's "
        "diagnostic, and what it was scored on: distinct windows, gathered rows "
        "(before any horizon's backward shift) and episode clusters ---",
        _table_line(SELF_CHECK_COLUMNS, SELF_CHECK_WIDTHS),
    ]
    for (arm, seed), record in sorted(records.items()):
        cell = (arm, seed)
        lines.append(_table_line((
            arm, int(seed), int(_get(record, cell, "step")),
            _window_count(record, cell),
            len(_get(record, cell, "windows", "episode")),
            int(_get(record, cell, "clusters")),
            "yes" if _get(record, cell, "self_check", "ok") else "NO",
        ), SELF_CHECK_WIDTHS))
    return "\n".join(lines)


def width_text(records: dict, inputs: ContrastInputs, reading) -> str:
    """Everything `read` prints, in the order a reader should meet it: the
    self-check, the width bias, the corrected Reading E, then Reading F -- the
    evidence before the conclusion -- written to `width.txt` and printed as the
    same string."""
    return "\n\n".join([
        _self_check_table(records),
        width_bias_table(records),
        corrected_reading_e(records),
        format_reading_contrast(reading, inputs),
    ]) + "\n"


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def load_width(out_dir: Path, arms, seeds) -> dict:
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
            path = width_record_path(out_dir, arm, int(seed))
            if not path.exists():
                raise CellMissing(
                    f"{arm} seed {int(seed)}: no width record at {path}; "
                    "run --phase measure first"
                )
            record = load_record(path)
            if record.get("arm") != arm or int(record.get("seed", -1)) != int(seed):
                raise SystemExit(
                    f"{path.name} was read for {arm} seed {int(seed)} but its record says "
                    f"arm={record.get('arm')!r} seed={record.get('seed')!r}"
                )
            records[(arm, int(seed))] = record
    return records


def _plan(args) -> tuple[list[str], list[int]]:
    """The DISTINCT arms and seeds, in the order given. `--arms a a` is one arm and
    `--seeds 0 0 0` one seed: counting the lists would let a plan through that the
    records dict, keyed by cell, cannot honour -- and would measure a cell twice."""
    return (
        list(dict.fromkeys(args.arms)),
        list(dict.fromkeys(int(s) for s in args.seeds)),
    )


def read_phase(args) -> int:
    """Refuse a plan that cannot be read, pool every requested cell (11 names the
    first missing), decide Reading F, write `width.txt`, print the same text, and
    return `READ_EXITS.get(reading.status, EXIT_OK)` -- 0 for every reading, 41 or
    42 for the two refusals.

    `reading_contrast` raises two ValueErrors that no builder can pre-empt without
    turning a legitimate UNRESOLVED_* record into a crash -- an arm clearing both
    ways, and both directions at the bar -- and both arise only PAST its gates.
    They become a SystemExit here, with the reading's own message, so the reader
    gets a named refusal instead of a traceback, and NOTHING is written: a refused
    read has no artefact.

    THE ARTEFACT IS WRITTEN BEFORE ANYTHING IS PRINTED, so `width.txt` gates the
    log."""
    arms, seeds = _plan(args)
    require_readable_plan(arms, seeds)
    try:
        records = load_width(args.out, arms, seeds)
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS
    require_one_protocol(records)
    inputs = contrast_inputs(records)
    try:
        reading = reading_contrast(inputs)
    except ValueError as error:
        raise SystemExit(f"Reading F refuses these records: {error}") from error
    text = width_text(records, inputs, reading)
    write_text(args.out / "width.txt", text)
    print(text, end="")
    return READ_EXITS.get(reading.status, EXIT_OK)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3k: width-matched ladder -- measure the three passes, read Reading F",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/m3k_retention"))
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
    args = _parser().parse_args(argv)
    args.arms, args.seeds = _plan(args)
    if args.phase in ("measure", "all"):
        device = get_device(prefer=args.device)
        cells = [(arm, seed) for arm in args.arms for seed in args.seeds]
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
