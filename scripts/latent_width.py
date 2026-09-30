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
reads that back and the record carries the result; `read` (Task 6) decides what
a broken anchor means.

  measure   per shipped cell: the three passes over both targets and every
            reported k, the contrast Reading F is taken on, the position
            control, the anchors. One record per cell.
  read      Task 6: pools these records into Reading F.

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
  read:     EXIT_BASE_UNRESOLVED (41)  `enc(t)` -> absolute position did not
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

import importlib.util
import types
from pathlib import Path

import numpy as np
import torch

from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.probe import (
    GainSplit, apply_probe, contrast_from_blocks, fit_probe, gain_from_blocks,
    gather_probe_data, probe_r2,
)
from mbfps.eval.retention import (
    ARMS_REQUIRED, CONFIDENCE, K_REPORTED, RESAMPLES, RUNGS, SEEDS_REQUIRED, TARGETS,
    backward_rotation, backward_translation, rung_block, shifted_rows,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, write_record
from mbfps.eval.width import ANCHOR, CONTRAST_K, PASSES, PROJECTION_SEED, RUNG_WIDTH, pass_block


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
    _, source = shifted_rows(data["window"], data["step"], k)
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
    plan (Task 6), and `--phase measure` is allowed a plan `read` could not
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
)
"""Every field `require_one_protocol` compares, as `(name, pick)`, in one
module-level table that the function ITERATES and a test reads -- so a field
cannot be added to the comparison without the test noticing it has no
disagreement case. While the tuple lived inside the function, no test could see
it, and a check on the test's own literal compared that literal to itself."""


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

    `clusters` and `rows` are NOT compared: they are print-only and gate nothing
    downstream, so no reading can flip on them.
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
