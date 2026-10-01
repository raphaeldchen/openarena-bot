"""M3l: is `z` out of room, or was it never asked? -- the measure phase.

M3j found the study's largest, most unanimous effect: `h` carries the agent's
observed motion and `z` does not, 9 of 9 cells in both targets. The rollout runs
on `z`, so motion never reaches what predicts forward. But that observation is
COMMON TO BOTH EXPLANATIONS -- "`z` lacks the capacity" and "`z` was never
asked" (the loss targets the current frame) predict it equally -- and M3k, built
to separate them by a DIFFERENCE of two fitted quantities, returned
INDISTINGUISHABLE at effect/noise 0.22.

So M3l asks the question that separates them directly, and asks it as a LEVEL
AGAINST AN EXACT CEILING: how many of the code's `z_cats * log2(z_classes)`
available bits does the posterior carry, and are those bits just the current
frame re-encoded? `mbfps.eval.capacity` is the pure half -- the estimator, the
derived ceiling, the controls, Reading G and its table -- and owns none of
torch, the model, or the record schema. This script is the impure half.

PER CELL, FROM ONE GATHER:

  bits            `Sum_j I(z_j ; h, enc(t))` with an episode-clustered
                  interval, against the 160-bit derived ceiling.
  frame_share     `R2(enc(t) -> post_probs)` over the 1024 flattened columns
                  under the three-split discipline, zero-variance columns
                  excluded, with an interval of its own. `live` is how many
                  columns survived that exclusion.
  prior_bits      the same estimator on `prior_probs`. A COMPANION THAT DECIDES
                  NOTHING, free from the same forward pass, and spec 3.3
                  requires it per cell: the prior does not see `enc(t)`, so it
                  is the information the prior's distribution carries through
                  `h` alone, which is what M3g's "the prior is the failing
                  stage" finding is about.
  redundancy      `redundancy_bits` and `redundancy_floor`, the second
                  COMPANION THAT DECIDES NOTHING -- and the one that makes a
                  high `bits` readable at all. `bits_carried` sums the
                  PER-CATEGORICAL informations, so it UPPER-BOUNDS the joint:
                  32 categoricals all copying one 5-bit variable read the full
                  160 while carrying 5 bits jointly. A reading BELOW a cut is
                  conservative; a reading ABOVE one does not establish "the
                  capacity is in use" unless the redundancy sits near its
                  floor. Both ship as plain floats; the ratio is formed at read
                  time by `redundancy_ratio`, which is None for a collapsed
                  code. Measured cost for both: 0.154 s per cell.
  checks          the three estimator checks, each reported separately with its
                  four numbers, so a failure names itself and can be audited
                  from the record.
  base_control    `enc(t) -> position` as a level, the arm every claim about
                  the code is made under.

ONE GATHER PER CELL, so every number describes ONE row set. Deriving
`frame_share` from a second gather would reintroduce the row-alignment hazard
M3k needed two layered guards for: a builder returning the same count of rows
in a different order silently misaligns a block against its own target, with no
refusal anywhere.

LOADING IS `trust_horizon.py`'S, exactly as `scripts/latent_retention.py` and
`scripts/latent_width.py` load it: `Cell`, `CellMissing`, `load_cell`,
`self_check` and `prepare_cell` are imported by path through `_sibling`, so a
cell is refused here for the same reasons and in the same words every other
diagnostic refuses it, and this module holds its own copy of every class
`_sibling` defines -- never catch another importer's copy of `CellMissing`.

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
  read:     EXIT_NO_CHECKPOINTS (11)   a requested cell has no capacity record.
            EXIT_ESTIMATOR_BROKEN (43) the estimator missed a known answer.
            EXIT_BASE_UNRESOLVED (44)  `enc(t)` -> position did not read.

43 AND 44 ARE RAISED IN THE READ PHASE, NEVER HERE. `measure_phase` RECORDS a
failed check and carries on, because the nine records are the artefact and a
measure that raised would discard the evidence of which cell broke. 0 / 11 / 12
/ 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 43 and 44 are in no
other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10, diagnose
11-17, pool 18-22, trust 30, split_gap 31, ladder 32-33, stages 34,
sharper_latent 35-37, latent_motion 38, latent_retention 39-40, latent_width
41-42, argparse 2, a traceback 1).

THE PLAN IS CHECKED BEFORE THE PROBE, not after it. `reading_capacity` raises
on fewer than `ARMS_REQUIRED` arms, on arms that disagree on `seeds_total`, and
on a shared `seeds_total` below `SEEDS_REQUIRED` -- INSIDE the reading, i.e.
after every gather and every fit. Here `measure_phase` refuses first whenever a
read will follow, as `scripts/latent_width.py` does. A `--phase measure` run is
NOT refused: the milestone's smoke is one cell, and a narrow plan is a
legitimate thing to MEASURE.
"""

import importlib.util
import types
from pathlib import Path

import numpy as np
import torch

from mbfps.eval.capacity import (
    CEILING_BITS, argmax_marginal_bits, bits_carried, bits_interval, ceiling_bits,
    episode_stats, floor_bits, live_classes, redundancy_bits, redundancy_floor,
)
from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.probe import apply_probe, fit_probe, gather_probe_data, probe_r2
from mbfps.eval.retention import (
    ARMS_REQUIRED, CONFIDENCE, RESAMPLES, SEEDS_REQUIRED,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_latent_capacity", path)
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

EXIT_ESTIMATOR_BROKEN: int = 43
"""The estimator missed a known answer. 38 is M3i, 39-40 M3j, 41-42 M3k.

A reading taken from an estimator that failed its own floor, two-route or
inequality check is not a weaker reading, it is not a reading -- which is why
`reading_capacity` gives `UNRESOLVED_ESTIMATOR` precedence over everything. It
is RAISED IN THE READ PHASE, from the pooled records, never here."""

EXIT_BASE_UNRESOLVED: int = 44
"""`enc(t) -> position` did not clear `BASE_R2_FLOOR` in `ARMS_REQUIRED` arms.

A claim about what the code carries means nothing where the current frame cannot
linearly say where it is. Also raised in the read phase."""

PHASES: tuple[str, ...] = ("all", "measure", "read")

FIT_EPISODES: int = 20
"""Training episodes the probe weights are fit on -- `probe.PROBE_EPISODE_LIMIT`,
the same set gate criterion 4 fits on, so the levels stay comparable."""

SELECT_EPISODES: int = 20
"""Training episodes AFTER the first `FIT_EPISODES` that select the ridge.

`probe.filtering_gain`'s default, and NOT a smaller number: that docstring
records the measurement -- a 4-episode selection split picks 1e5 for both arms
and reports +0.0325 where a 20-episode split picks 1e3 and recovers -0.0208, so
THE SIGN FLIPS on a one-step selection error."""

TOLERANCE: float = 1e-9
"""Every tolerance in the estimator checks.

Not equality. `floor_bits` is zero in exact arithmetic but drifts by float
SUMMATION ORDER: measured +5.68e-14 at 500 rows, -1.42e-13 at 2,000 and
+5.40e-13 at the ~11,000 rows a real cell carries. 1e-9 is four orders above
that, and the reason it is stated rather than tightened is that an M3h sweep
once refused on an 8.527e-14 mismatch that was summation order and not a
defect."""

CHECK_FLAGS: tuple[str, ...] = ("floor_ok", "routes_ok", "bracket_ok")
"""The three booleans `estimator_checks` reports, as a table every reader
ITERATES rather than re-spells.

A fourth check cannot be added without a reader noticing it has no consumer, and
a boolean no reader reads cannot accumulate on the record. `_cell_line` and
Task 5's `control_flags` both read this."""


def capacity_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed -- see
    `study.job_record_path` for what a colliding name costs."""
    return Path(out_dir) / f"capacity_{arm}_seed{seed}.json"


# ---------------------------------------------------------------------------
# One gather per cell.
# ---------------------------------------------------------------------------


def gather_once(prepared, train, val, *, seed: int):
    """ONE gather per cell, scoring EVERY validation episode.

    Returns the `(fit, select, score)` triple the three-split discipline needs:
    the weights come from the first `FIT_EPISODES` training episodes, the ridge
    is selected on the NEXT `SELECT_EPISODES` (from beyond the fit set, so the
    weights train on the same episodes criterion 4 uses), and the reported
    numbers come from the validation windows that neither the weights nor the
    selection ever saw. Seeds are `seed`, `seed + 2` and `seed + 1` in that
    order, matching `probe.filtering_gain` exactly so the two diagnostics
    describe the same rows.

    ONE GATHER means ONE row set, so `bits_carried` and `frame_share` describe
    the same rows. A second gather would reintroduce the row-alignment hazard
    M3k needed two layered guards for.

    THE SCORED SPLIT TAKES EVERY VALIDATION EPISODE, `limit=len(val)` and never
    `FIT_EPISODES`. `filtering_gain`'s `limit` caps the fit split AND the scored
    split at the same number -- its docstring says so ("episodes for the fit
    split, and for the scored split") -- so mirroring it literally scored 20 of
    the 24 validation episodes on M3j and silently threw four away, 17% of the
    evaluation data. The test that was meant to pin it could not, because
    `len(val)` was ALSO 20 in its fixture. The protocol every milestone since
    M3d reads is 229 windows over 24 validation episodes, and that is what the
    reading must be taken on.
    """
    used = list(train)
    scored = list(val)
    fit_paths = used[:FIT_EPISODES]
    if not fit_paths:
        raise ValueError("no training episodes to fit the capacity probe on")
    select_paths = used[FIT_EPISODES:FIT_EPISODES + SELECT_EPISODES]

    def gather(paths, draw: int):
        # `limit=len(paths)`: THE SLICES ABOVE ARE THE CAP, and this argument
        # never is. The `used[...]` slices bind because they also PARTITION the
        # pool -- `select_paths` starts exactly where `fit_paths` ends -- and
        # because `measure_cell` records the same slices as `episodes.fit` and
        # `episodes.select`. A second cap here (`limit=FIT_EPISODES`) could only
        # ever agree with the slice, and the day it did not it would silently
        # drop episodes the record says were used. `probe.filtering_gain`
        # gathers its fit and select splits the same way. The scored split has
        # no slice, so `len(scored)` is what takes every validation episode.
        return gather_probe_data(
            prepared.model, paths,
            prepared.common["feature_backbone"], prepared.common["device"],
            context=prepared.context, horizon=prepared.horizon,
            limit=len(paths), seed=draw,
        )

    return (
        gather(fit_paths, seed),
        gather(select_paths, seed + 2) if select_paths else None,
        gather(scored, seed + 1),
    )


def require_temporal_order(data: dict) -> None:
    """The gather's rows must arrive in TEMPORAL ORDER, and nothing else says so.

    `redundancy_floor` rolls each categorical's rows CIRCULARLY by its own
    random offset. That destroys dependence between categoricals while
    PRESERVING each series' own autocorrelation, which is exactly the null the
    redundancy comparison needs -- and it only works if consecutive rows are
    consecutive frames. Shuffled rows have no autocorrelation to preserve and
    the roll degenerates into the plain permutation it replaces. Measured, on
    independent categoricals persisting within 50-row windows, a permutation
    floor reads them at 1.61x / 3.90x / 10.74x their true ratio against the
    circular-shift floor's 1.01x / 1.04x / 1.12x -- and the review's own fixture
    read 27.00x against 1.34x. So the ratio would stop measuring redundancy and
    start measuring autocorrelation, silently, with every row count and cluster
    count still correct.

    `gather_probe_data` DOES produce rows in window order -- it appends
    `np.full(rows, window_index)` and `np.arange(rows)` per window, with
    `window_index` ascending, and
    `test_gather_probe_data_labels_every_row_with_its_window_and_step` pins
    that. Nothing enforces it for a CALLER, though, and the three things a
    caller does that would break it -- a sort on another key, a shuffle, a row
    subsample -- all leave the shapes, the labels and the counts intact. So it
    is asserted here rather than assumed.

    The check: `window` non-decreasing (so each window's rows are one
    contiguous run, labels being integers), and `step` exactly `0 .. n - 1`
    inside each run (so no row was dropped, reordered or duplicated).
    """
    window = np.asarray(data["window"])
    step = np.asarray(data["step"])
    if window.ndim != 1 or window.shape != step.shape:
        raise ValueError(
            f"`window` {window.shape} and `step` {step.shape} must be one "
            "label per row; the rows cannot be checked for temporal order"
        )
    if window.size == 0:
        raise ValueError("the gather produced no rows, so there is no temporal order")
    if np.any(np.diff(window) < 0):
        raise ValueError(
            "the scored rows are not in temporal order: `window` is not "
            "non-decreasing, so one window's rows are split across the array. "
            "`redundancy_floor`'s circular shift preserves each categorical's "
            "autocorrelation only while consecutive rows are consecutive "
            "frames; out of order it degenerates to the permutation floor it "
            "replaces, which reads independent-but-persistent categoricals at "
            "up to 27x their true ratio"
        )
    boundaries = np.flatnonzero(np.diff(window) != 0) + 1
    for block in np.split(np.arange(window.size), boundaries):
        if not np.array_equal(step[block], np.arange(block.size)):
            raise ValueError(
                f"the scored rows are not in temporal order: window "
                f"{window[block[0]]!r} carries steps "
                f"{step[block].tolist()[:8]}... where `0 .. {block.size - 1}` is "
                "required. A sort, a shuffle or a row subsample leaves every "
                "shape, label and count intact and still breaks the "
                "autocorrelation `redundancy_floor`'s circular shift exists to "
                "preserve, turning it into the permutation floor it replaces"
            )


# ---------------------------------------------------------------------------
# The three estimator checks.
# ---------------------------------------------------------------------------


def estimator_checks(probs) -> dict:
    """The three checks, each reported separately so a failure names itself.

    All three are substitutions into the REAL estimator on the REAL data, not
    fixtures -- the same shape of known-answer control as M3k's anchors.

    `floor_ok`   `floor_bits` within `TOLERANCE` of 0. Every row replaced by the
                 marginal gives `H(m) - mean_n H(m)`, zero by algebra, so a
                 reading away from zero beyond summation drift is an arithmetic
                 defect: a missing normalisation, or a mean over the wrong axis.
    `routes_ok`  `ceiling_bits` against `argmax_marginal_bits` within
                 `TOLERANCE`. Two independent routes to one answer pin the
                 ROUTING: a mean over the wrong axis, or a substitution that
                 leaks a row's own distribution, breaks the agreement.
    `bracket_ok` THE TWO THEOREMS, and only two:
                   `-TOLERANCE <= bits <= CEILING_BITS + TOLERANCE`
                   `-TOLERANCE <= ceiling <= CEILING_BITS + TOLERANCE`
                 `bits = H(marginal) - E_n H(row) <= H(marginal) <= z_cats *
                 log2(z_classes)`, so the upper bounds follow.

    `bits_carried <= ceiling_bits` IS NOT A THEOREM AND IS NOT CHECKED. An
    earlier draft of the design asserted it. `ceiling_bits` is the information
    content of the ARGMAX PATTERN while `bits_carried` is the distribution's, so
    a code whose argmax never moves while its tail varies reads 7.5207 bits
    against a `ceiling_bits` of 0.0000. Refusing on it would reject valid
    readings precisely in the diffuse, low-information regime this milestone
    exists to investigate, which is the worst possible place for a spurious
    refusal, because it is where the answer lives.
    `tests/eval/test_capacity.py::test_ceiling_bits_is_not_an_upper_bound_on_
    bits_carried` carries the counterexample and this file drives the same
    fixture through here to assert it is NOT refused.

    THE LOWER BOUND ON `ceiling` CARRIES THE TOLERANCE TOO, where spec 2.3
    writes a bare `0 <=`. `ceiling_bits` is `bits_carried` of the one-hot
    substitution, so it is the same summation and drifts negative by the same
    ~5e-13 at run scale; a strict `0 <=` would refuse a code whose argmax never
    moves -- the collapsed case -- for a float artefact. Spec 2.3 states every
    tolerance here is 1e-9, and this is that tolerance applied to the end the
    inequality was written without.

    Returns the three booleans AND the four numbers, because a check that
    reports only a boolean cannot be audited from the record: a reader has to
    see HOW far the floor drifted and WHERE the two routes landed.
    """
    bits = float(bits_carried(probs))
    floor = float(floor_bits(probs))
    ceiling = float(ceiling_bits(probs))
    marginal = float(argmax_marginal_bits(probs))
    return {
        "bits": bits,
        "floor": floor,
        "ceiling": ceiling,
        "argmax_marginal": marginal,
        "floor_ok": bool(abs(floor) <= TOLERANCE),
        "routes_ok": bool(abs(ceiling - marginal) <= TOLERANCE),
        "bracket_ok": bool(
            -TOLERANCE <= bits <= CEILING_BITS + TOLERANCE
            and -TOLERANCE <= ceiling <= CEILING_BITS + TOLERANCE
        ),
    }


# ---------------------------------------------------------------------------
# frame_share: R2(enc(t) -> post_probs) on the live columns.
# ---------------------------------------------------------------------------


def _flat_probs(data: dict) -> np.ndarray:
    """One split's `post_probs` as `(N, z_cats * z_classes)` float64.

    `post_probs`, never `prior_probs` and never `latent`: `frame_share` asks how
    much of the POSTERIOR code's variance the current frame explains, and the
    posterior is the distribution `z` was actually drawn from."""
    probs = np.asarray(data["post_probs"], dtype=np.float64)
    return probs.reshape(probs.shape[0], -1)


def _varying_columns(probs: np.ndarray) -> np.ndarray:
    """Which of the flattened columns vary across the rows: `max > min`.

    `probs` is `(N, z_cats, z_classes)`, as every caller passes it, and the
    result is one boolean per FLATTENED column, `z_cats * z_classes` of them.

    `capacity.live_classes`' rule exactly, and for its reason: the standard
    deviation of identical float64 values is rounding noise rather than zero
    (measured, a fully collapsed code read 1013 of 1024 columns live at 300
    rows under `std > 0.0`), so `std > 0` would call a column that never varies
    live in exactly the posterior-collapse case this count exists to report.

    `cell_capacity` cross-checks the count against `live_classes` so the printed
    saturation signal cannot describe a different set of columns from the
    `frame_share` beside it."""
    flat = probs.reshape(probs.shape[0], -1)
    return flat.max(axis=0) > flat.min(axis=0)


def _r2_stats(predicted: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> dict:
    """Per-episode sufficient statistics for the per-column-mean R^2.

    `probe._mean_r2` is `mean_c [1 - SSE_c / SST_c]`, and `SSE` decomposes over
    episodes: `SSE = sum_e sse_e`. So the bootstrap resamples an
    `(n_episodes, columns)` array instead of re-slicing and re-reducing an
    `(11000, 1024)` one per draw -- the same reasoning, and the same order of
    speedup, that `capacity.EpisodeStats` records for `bits_carried`.

    `variance` IS THE COLUMN'S FULL-SAMPLE VARIANCE AND IS NOT RESAMPLED. That
    is the one decision in this file that departs from `probe._block_bootstrap_
    ci`, which recomputes `_mean_r2` -- numerator AND denominator -- on each
    draw's rows. It has to, and here is the measurement that says so.

    `post_probs` spans 32 orders of magnitude: a softmax column whose
    probability sits at 7e-33 is `max > min` live, and its full-sample `SST` is
    a healthy 1e-3 or more (the smallest of the 1024 is 7.9e-4) only because the
    column is large in SOME episodes. Draw a bootstrap sample that happens to
    miss those episodes and the column is effectively constant inside the draw.
    Measured on this file's own fixture -- 6 clusters, `frame_probe`'s own
    draw sequence at `seed=0`, 200 draws, the denominator recomputed per draw
    the way `probe._block_bootstrap_ci` does it: the worst column-draw has `SST`
    4.1e-32 against an `SSE` of 1.9e-4, so its R^2 reads -4.6e27; 77 of the 200
    draws read a mean R^2 below -1000 and the worst reads -4.7e24; the worst
    draw has 790 of the 1024 columns past -1000. And there is NO clean
    threshold that separates them: the draw-to-full `SST` ratio of the columns
    past -1000 runs from 4.9e-30 up to 1.0e-3, and that of the columns NOT past
    -1000 from 8.2e-6 up to 4.5, so the two ranges overlap by two orders of
    magnitude and excluding the degenerate columns by tolerance is a guess
    rather than a rule. A lower bound of -4.7e24 is not a measurement, and
    `capacity_arm` would accept it as one.

    THE FIGURES DEPEND ON THE DRAW SEQUENCE and the conclusion does not: at
    seeds 1, 2 and 3 of the same 200 draws, 66, 73 and 52 draws fall below
    -1000 with worst means of -1.3e24, -1.7e37 and -6.0e24. (`rng.choice` and the
    precedent's `integers` draw the same indices, so `seed=0` reads identically
    through either.)

    So each draw's R^2 compares the draw's MEAN squared residual against the
    column's full-sample variance:

        R2_draw_c = 1 - (SSE_draw_c / n_draw) / (SST_full_c / n_full)

    The denominator is a property of the EVALUATION SET -- how much each column
    varies over the 24 validation episodes -- not of the model, and holding it
    fixed is the same move `_block_bootstrap_ci` already makes for the PROBES
    ("the probes are held FIXED across resamples. This is an interval on the
    scored sample"). What the interval then measures is how much the residual
    moves on another draw of evaluation episodes, which is what the reading is
    about.

    TWO CONSEQUENCES, BOTH STATED BEFORE THE NUMBERS. The interval is slightly
    NARROWER than a fully resampled one, because one nuisance component of the
    sampling variability is removed; a narrower frame interval makes
    `FRAME_REENCODING` easier to clear, which pushes AGAINST the asymmetry spec
    3.3 records (the fall-through `CAPACITY_BOUND` is already the easiest status
    to reach) rather than with it. And at `picked = every episode once` the
    formula collapses to `1 - SSE_full / SST_full` exactly, so the point
    estimate IS `probe._mean_r2` on the live columns -- pinned by a test against
    that function -- rather than a second formula that agrees to 1e-12 and can
    still straddle a bound `capacity_arm` refuses.

    `variance` is computed with the TWO-PASS formula on the full sample, never
    `sy2 - sy^2/n`: on a column whose values sit near 1 with small variance that
    difference loses most of its significant digits.
    """
    residual = targets - predicted
    labels = np.unique(groups)
    masks = [groups == label for label in labels]
    rows = np.array([int(m.sum()) for m in masks], dtype=np.float64)
    return {
        "sse": np.stack([(residual[m] ** 2).sum(axis=0) for m in masks]),
        "rows": rows,
        # Strictly positive for every column `_varying_columns` kept: `max >
        # min` means at least one squared deviation is positive.
        "variance": ((targets - targets.mean(axis=0)) ** 2).sum(axis=0) / rows.sum(),
        "labels": labels,
    }


def _r2_from_stats(stats: dict, picked: np.ndarray) -> float:
    """The per-column-mean R^2 over the episodes `picked` names.

    `1 - mean_squared_residual / full_sample_variance`, averaged over the
    columns -- see `_r2_stats` for why the denominator is the full sample's."""
    mse = stats["sse"][picked].sum(axis=0) / stats["rows"][picked].sum()
    return float(np.mean(1.0 - mse / stats["variance"]))


def frame_probe(fit: dict, select: dict | None, score: dict, *, columns: np.ndarray,
                groups: np.ndarray, resamples: int = RESAMPLES,
                confidence: float = CONFIDENCE, seed: int = 0) -> dict:
    """`R2(enc(t) -> post_probs)` on `columns`, with an episode-clustered interval.

    The three-split discipline, unchanged: weights from `fit`, ridge selected on
    `select`, the reported R^2 and interval from `score`. A GENERALISATION
    number, not a fit statistic -- which is the whole reason this is not one
    `fit_probe` on the scored rows.

    THE FEATURES ARE `encoder_embedding`, THE RAW ENCODER OUTPUT, never
    `embedding`, the model's PREDICTED embedding. `frame_share` asks what the
    CURRENT FRAME explains about the code; against the predicted embedding it
    would instead ask whether the code beats its own head's reconstruction,
    which a code can win while carrying nothing of the frame. That is the
    confound `probe.gather_probe_data`'s docstring records at length, and the
    one a fixture whose two embeddings were byte-identical hid for 13 tests.

    `columns` is the subset of the flattened `z_cats * z_classes` columns that
    VARY on the scored rows. They are the only ones scored, so they are the only
    ones fitted: R^2 is undefined for a constant column, and a probe fitted to
    predict constants would spend its ridge selection on them. A code in which
    NO column varies is refused rather than scored, which is what
    `probe._mean_r2` does with the same wording -- there is no R^2 at all there,
    and a NaN on the record would be refused by `capacity_arm` one phase later
    with a message about non-finite values rather than about a collapsed code.

    The bootstrap groups on EPISODES, never windows: several non-overlapping
    windows cut from one trajectory are not independent observations, and every
    reading from M3e onward clusters on episodes.

    THIS INTERVAL IS NOT THE PROJECT'S STANDARD BOOTSTRAP, and the return value
    says what it IS. `_r2_stats` holds each column's R^2 DENOMINATOR at the full
    sample's variance rather than resampling it, as `probe._block_bootstrap_ci`
    does, because resampling it puts the mean R^2 below -1000 in 77 of 200
    draws on the file's own fixture (worst -4.7e24). `frame_confidence` and
    `frame_resamples` are the level and the draw count the interval was
    ACTUALLY TAKEN AT -- the arguments this function used, not a claim made by
    its caller -- so a reader of the record does not take `bits`' `confidence`
    to describe it.

    Returns `frame_share`, `frame_low`, `frame_high`, `frame_confidence`,
    `frame_resamples`, `frame_ridge` and `frame_ridge_selected`. Only
    `frame_low` is compared to a cut downstream.
    """
    if not np.any(columns):
        raise ValueError(
            "every one of the code's columns has zero variance across the "
            "scored rows; there is no R^2 to score, so frame_share cannot be "
            "measured on this cell. A fully collapsed posterior is a real "
            "outcome of this milestone -- `bits` reads it at the bottom of its "
            "scale -- but it is not a share of anything"
        )
    fit_x = np.asarray(fit["encoder_embedding"], dtype=np.float64)
    fit_y = _flat_probs(fit)[:, columns]
    score_x = np.asarray(score["encoder_embedding"], dtype=np.float64)
    score_y = _flat_probs(score)[:, columns]
    if select is None:
        probe = fit_probe(fit_x, fit_y)
    else:
        probe = fit_probe(
            fit_x, fit_y,
            np.asarray(select["encoder_embedding"], dtype=np.float64),
            _flat_probs(select)[:, columns],
        )
    predicted = apply_probe(probe, score_x)

    stats = _r2_stats(predicted, score_y, np.asarray(groups))
    if not np.all(stats["variance"] > 0.0):
        dead = int((stats["variance"] <= 0.0).sum())
        raise ValueError(
            f"{dead} of the {columns.sum()} scored columns have zero full-sample "
            "variance, so R^2 is undefined for them and the mean would divide by "
            "zero. `columns` must name only the columns that vary on the scored "
            "rows -- `_varying_columns` of the SCORED split, never of the fit "
            "split and never every column"
        )
    n_episodes = stats["labels"].size
    if n_episodes < 2:
        raise ValueError(
            "a bootstrap interval needs at least two resampling units (distinct "
            f"episode labels); got {n_episodes}"
        )
    index = np.arange(n_episodes)
    rng = np.random.default_rng(seed)
    draws = np.array([
        _r2_from_stats(stats, rng.choice(index, size=n_episodes, replace=True))
        for _ in range(resamples)
    ])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "frame_share": _r2_from_stats(stats, index),
        "frame_low": float(low),
        "frame_high": float(high),
        "frame_confidence": float(confidence),
        "frame_resamples": int(resamples),
        "frame_ridge": probe["ridge"],
        "frame_ridge_selected": select is not None,
    }


# ---------------------------------------------------------------------------
# One cell's measurement.
# ---------------------------------------------------------------------------


def cell_capacity(gathered, *, seed: int, resamples: int = RESAMPLES) -> dict:
    """One cell's `bits`, interval, `frame_share`, `live`, `prior_bits`,
    `redundancy_bits`, `redundancy_floor` and the three checks.

    `gathered` is the `(fit, select, score)` triple `gather_once` returns -- ONE
    gather, so every number below describes the same rows.

    `seed` IS THE CELL'S SEED INDEX (`cell.seed`: 0, 1 or 2) and it is threaded
    into EVERY estimator call: `bits_interval`, `frame_probe` and
    `redundancy_floor`. `redundancy_floor(probs, seed=seed)` takes it as a
    REQUIRED argument, never a default, because a defaulted seed is how M3j
    shipped every cell sharing one bootstrap seed -- which correlates the
    interval noise the seeds x arms agreement rule treats as independent.
    Common random numbers WITHIN one cell are intended, as in
    `latent_width.cell_passes`: the three draws share `seed` on purpose.

    `resamples` defaults to the protocol's `RESAMPLES` and `measure_cell` never
    passes it, so a production run cannot depart from the pre-registered figure.
    It exists so a test that asserts on no interval width can pay for fewer.

    TWO INTERVALS, TWO METHODS, AND THE RECORD SAYS WHICH IS WHICH. `confidence`
    and `n_episodes` come from `bits_interval` and describe the `bits` interval
    ONLY. The frame interval carries its own `frame_confidence` and
    `frame_resamples`, because a reader taking the one `confidence` to describe
    both would be assuming the frame interval is the project's standard
    bootstrap -- and it deliberately is not: `frame_probe` HOLDS THE R^2
    DENOMINATOR AT THE FULL SAMPLE instead of resampling it, as
    `probe._block_bootstrap_ci` does (see `_r2_stats` for the measurement that
    forces it). That makes the frame interval slightly NARROWER than a
    fully-resampled one, and `frame_low` -- the only frame number Reading G
    compares to a cut -- inherits it. `bits_interval` does not record its draw
    count; `cell_capacity` hands both the same `resamples`.

    THE SCORED ROWS MUST BE IN TEMPORAL ORDER (`require_temporal_order`),
    checked first, before anything is paid for.

    `frame_share` is `R2(enc(t) -> post_probs)` on the flattened
    `z_cats * z_classes` columns, ZERO-VARIANCE COLUMNS EXCLUDED -- R^2 is
    undefined for them and this project has already refused a fixture for
    exactly that. Their count ships as `live`, which is also a saturation
    signal: a `CAPACITY_BOUND` verdict printed beside a low `live` contradicts
    itself. The count and the averaged set are tied to ONE rule by the
    cross-check below, so the printed signal cannot describe different columns
    from the number beside it.

    `prior_bits` is the same estimator on `prior_probs`, which `observe` already
    returned, so it costs nothing. NOT a known-answer control: the prior is a
    function of `h`, `h` encodes past frames, so its distribution genuinely
    varies across rows and the reading is positive. It decides nothing, and spec
    3.3 requires it per cell.

    Both redundancy numbers ship as PLAIN FLOATS. The ratio is derived at read
    time by `redundancy_ratio`, which returns None for a collapsed code -- a
    value JSON cannot carry and no cut can be applied to. They GATE NOTHING: no
    check, status or refusal here depends on either.
    """
    fit, select, score = gathered
    require_temporal_order(score)
    probs = np.asarray(score["post_probs"], dtype=np.float64)
    groups = np.asarray(score["episode"])

    columns = _varying_columns(probs)
    live = int(columns.sum())
    reported = int(live_classes(probs))
    if live != reported:
        raise ValueError(
            f"the columns `frame_share` averages ({live}) are not the ones "
            f"`live_classes` counts ({reported}); `live` prints beside the "
            "verdict as a saturation signal and must describe the same set of "
            "columns the share was taken over"
        )

    interval = bits_interval(
        episode_stats(probs, groups),
        resamples=resamples, confidence=CONFIDENCE, seed=seed,
    )
    frame = frame_probe(
        fit, select, score, columns=columns, groups=groups,
        resamples=resamples, confidence=CONFIDENCE, seed=seed,
    )
    return {
        **interval,
        **frame,
        "live": live,
        "prior_bits": float(
            bits_carried(np.asarray(score["prior_probs"], dtype=np.float64))
        ),
        "redundancy_bits": float(redundancy_bits(probs)),
        "redundancy_floor": float(redundancy_floor(probs, seed=seed)),
        "checks": estimator_checks(probs),
        # The ONE place the scored row count is computed. `measure_cell` reads
        # it back onto the record's top level rather than counting again: two
        # counts of one thing are two numbers that can disagree.
        "rows": int(probs.shape[0]),
    }


def _per_column_r2(predicted: np.ndarray, targets: np.ndarray) -> list[float]:
    """`probe._mean_r2`'s four addends, UNAVERAGED -- pos_x, pos_y, sin(angle),
    cos(angle) in that order. Duplicates that function's per-column formula
    rather than importing its private name, so a zero-variance column reports
    NaN here (a fact about that column) instead of silently vanishing from an
    average the way it does inside `_mean_r2` itself."""
    scores = []
    for column in range(targets.shape[1]):
        truth = targets[:, column]
        denom = float(((truth - truth.mean()) ** 2).sum())
        if denom == 0.0:
            scores.append(float("nan"))
            continue
        scores.append(1.0 - float(((truth - predicted[:, column]) ** 2).sum()) / denom)
    return scores


def base_control(fit: dict, select: dict | None, score: dict) -> dict:
    """`enc(t)` -> absolute position, as an r2 LEVEL on the validation windows.

    Every row is used -- there is no backward shift, because the target is the
    frame's own state. A level, not a gain: this is the arm every claim about
    the code is made under, so the only question is whether it reads at all.

    `position_r2` IS THE GATED NUMBER (spec 5, M3k's correction): the same
    fitted probe's r2 against the first two target columns only, which is what
    `EXIT_BASE_UNRESOLVED` says it is about -- whether the current frame can say
    where it is. `BASE_R2_FLOOR` stays 0.10 and is deliberately NOT re-chosen;
    gated on position alone it is loose, and the spec records that rather than
    correcting it.

    `r2` is the 4-column mean over pos_x, pos_y, sin(angle), cos(angle) that M3j
    gated on. It is KEPT AND NOT GATED ON, so this record bridges to M3j's --
    and the read phase must not print it in a column a tally is read from,
    which is how M3h shipped a tally beside a figure it was not read from.

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
    `--out` holds the capacity records and nothing else, so handing `args`
    through unchanged would look for the nine M3c checkpoints in the output
    directory and refuse EVERY cell -- a real run blocker on M3j, fixed with
    this same shim. Same shim, same reason, as `scripts/latent_motion.py`,
    `scripts/latent_retention.py` and `scripts/latent_width.py`.
    """
    return types.SimpleNamespace(
        out=Path(source), device=args.device, context=args.context,
        horizon=args.horizon,
    )


def measure_cell(args, cell: Cell, device, train, val) -> tuple[int, dict | None]:
    """One cell: the checks, the one gather, the measurement, the base control.

    12 and 14 are `prepare_cell`'s, so this script, `scripts/split_gap.py`,
    `scripts/latent_retention.py` and `scripts/latent_width.py` all refuse a
    cell for the same reasons in the same words -- and the reproduction bound of
    M3h spec 2.4 is checked there for free, because it compares
    `evaluate_rollout`'s mean curves against the study record.

    30 is NOT free. It needs a SECOND pass -- `reference_trajectories`, the same
    one the two sibling scripts take -- because `self_check` reduces the
    PER-WINDOW rows, not the mean curves `prepare_cell` already compared. It
    refuses BEFORE the gather, the expensive part.

    `(EXIT_OK, record)` once the record is built; `(status, None)` on any
    refusal, with the refusal printed. The record is not yet WRITTEN -- that is
    `measure_phase`'s job, so every cell's record picks up the same `git_sha`
    provenance and the same non-finite scan through one call to `write_record`.

    A FAILED ESTIMATOR CHECK IS RECORDED, NOT RAISED. The read phase decides
    what it means (43), and a measure that raised would discard the evidence of
    which cell broke -- the nine records are the artefact.

    `z_cats` and `z_classes` ARE READ OFF THE CHECKPOINT'S OWN `rssm.cfg`, not
    off `RSSMConfig`'s class defaults: they set the ceiling every cut in Reading
    G is a fraction of, and `require_one_protocol` compares them across records
    for the same reason `latent_width` compares the measured `h_dim`. Written
    from the module constant they would be a constant that cannot vary between
    two records of one code version, which is exactly the trap M3k's review
    found in `projection_seed` and `rung_width`. (`capacity._require_
    distributions` independently refuses probs whose shape disagrees with
    `RSSMConfig`, so within one code version the two agree; the comparison is
    what catches records pooled ACROSS versions, which `git_sha` alone cannot,
    since a reader may legitimately pool records and the shape is what the
    ceiling is derived from.)

    `ceiling_bits` on the record is `CEILING_BITS` -- the derived figure the
    cuts are fractions of, written down so a reader of the JSON need not
    re-derive it. It decides nothing and is not compared.
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
            + ". Same windows, same rollout, same refit probe -- or this is not "
            "measuring what the gate measured. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None

    gathered = gather_once(prepared, train, val, seed=cell.seed)
    fit, select, score = gathered
    cfg = prepared.model.rssm.cfg
    # `seed=cell.seed`: every bootstrap draw is the CELL's -- see
    # `cell_capacity`. `resamples` is deliberately not passed, so a production
    # run cannot depart from the pre-registered `RESAMPLES`.
    capacity = cell_capacity(gathered, seed=cell.seed)
    record = {
        "arm": cell.arm, "seed": cell.seed, "step": int(cell.record["steps"]),
        "record_git_sha": cell.record.get("git_sha", "unknown"),
        "device": str(device), "context": prepared.context, "horizon": prepared.horizon,
        "z_cats": int(cfg.z_cats), "z_classes": int(cfg.z_classes),
        "ceiling_bits": CEILING_BITS,
        "split_seed": SPLIT_SEED,
        "torch_version": torch.__version__,
        "capacity": capacity,
        "base_control": base_control(fit, select, score),
        "clusters": int(np.unique(score["episode"]).size),
        # Read BACK off the measurement, never counted a second time: two
        # counts of one thing are two numbers that can disagree.
        "rows": capacity["rows"],
        "windows": {
            "episode": np.asarray(score["episode"]).tolist(),
            "window": np.asarray(score["window"]).tolist(),
        },
        "self_check": check.record(),
        "episodes": {
            "fit": [p.name for p in train[:FIT_EPISODES]],
            "select": [p.name for p in
                       train[FIT_EPISODES:FIT_EPISODES + SELECT_EPISODES]],
            "val": [p.name for p in val],
        },
    }
    return EXIT_OK, record


def require_readable_plan(arms, seeds) -> None:
    """Refuse a plan too narrow for Reading G to be read at all.

    The reading needs `ARMS_REQUIRED` arms, each summarised over at least
    `SEEDS_REQUIRED` seeds, and `reading_capacity` RAISES when it has fewer --
    inside the reading, after every gather and every fit. Asked here it costs
    nothing; asked there, `--arms frozen_ssl` is the whole measure followed by a
    traceback.

    TWO REFUSALS, EACH NAMING ITS OWN CONSTANT AND NOT THE OTHER. A single
    message mentioning both would satisfy both of the tests' `match=` patterns
    no matter which branch fired, which is how a pair of assertions like that
    stops being able to fail.
    """
    if len(arms) < ARMS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(arms)} arm(s) cannot be read: Reading G needs "
            f"ARMS_REQUIRED={ARMS_REQUIRED} arms, and reading_capacity would "
            "raise only AFTER every gather and fit had been paid for"
        )
    if len(seeds) < SEEDS_REQUIRED:
        raise SystemExit(
            f"a plan of {len(seeds)} seed(s) cannot be read: `--seeds` is "
            f"global, so each arm would have {len(seeds)} and an arm needs "
            f"SEEDS_REQUIRED={SEEDS_REQUIRED} to clear a cut -- and "
            "capacity_arm would raise only AFTER every gather and fit had been "
            "paid for"
        )


def _cell_line(record: dict, path: Path) -> str:
    """One line per measured cell: the position control, the bits level with its
    interval, the frame share, the live count and the three checks.

    `checks BROKEN(floor)` names WHICH check failed, in the spelling a long log
    can be grepped for -- a failed check is recorded rather than raised, so the
    line is the only thing that makes it visible before the read phase."""
    capacity = record["capacity"]
    broken = [flag for flag in CHECK_FLAGS if not capacity["checks"][flag]]
    checks = (
        "checks ok" if not broken
        else "checks BROKEN(" + ",".join(f.removesuffix("_ok") for f in broken) + ")"
    )
    return (
        f"{record['arm']} seed {record['seed']}: position r2 "
        f"{record['base_control']['position_r2']:.3f}; bits "
        f"{capacity['bits']:.4f} [{capacity['ci_low']:.4f}, "
        f"{capacity['ci_high']:.4f}] of {CEILING_BITS:.0f}; frame "
        f"{capacity['frame_share']:.4f}; live {capacity['live']}; "
        f"{checks}; wrote {path}"
    )


def measure_phase(args, cells, device, train, val) -> int:
    """Every requested cell, refusing the whole run before any gather when the
    plan cannot be read (a plan check, only when a read follows) or a cell is
    missing (11).

    `measure_cell` builds each record; this function writes it through the same
    `write_record` every other diagnostic uses -- so the non-finite scan and the
    `git_sha` provenance are shared -- prints one line per cell, and returns the
    first non-`EXIT_OK` status or `EXIT_OK`.

    A FAILED ESTIMATOR CHECK OR BASE CONTROL DOES NOT STOP THE RUN. 43 and 44
    belong to the read phase, which decides them from the pooled records; here
    the failure is written onto the record and the next cell is measured. The
    line says `checks BROKEN(...)` so it cannot pass unnoticed in a long log.

    THE PLAN CHECK FIRES ONLY WHEN `args.phase == "all"`. `read` refuses its own
    plan, and `--phase measure` is allowed a plan `read` could not read: the
    milestone's smoke is one cell. Both are read off `cells` -- the distinct
    arms and the distinct seeds actually about to be measured.

    THAT MAKES `args.phase` LOAD-BEARING, so a phase that is not one of `PHASES`
    -- absent, `None`, or a typo -- is REFUSED here. Read with a default, a
    missing `phase` is indistinguishable from `measure`: the plan check is
    skipped without a word and a one-arm `--phase all` becomes the whole measure
    ending in a traceback, which is exactly what the check exists to prevent.

    CELLS ARE LOADED FROM `args.source`, NOT `args.out`. `--out` is this
    script's own record directory -- where the capacity records are WRITTEN,
    below -- and the nine M3c checkpoints, study records and diagnostics live in
    the study directory instead.
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
        status, record = measure_cell(args, cell, device, train, val)
        if status != EXIT_OK:
            return status
        record["git_sha"] = git_sha()
        path = capacity_record_path(args.out, cell.arm, cell.seed)
        write_record(path, record)
        print(_cell_line(record, path))
    return EXIT_OK


# ---------------------------------------------------------------------------
# read: what pooling these records requires.
# ---------------------------------------------------------------------------


_PROTOCOL_FIELDS = (
    ("windows.episode", lambda r: list(r["windows"]["episode"])),
    ("windows.window", lambda r: list(r["windows"]["window"])),
    ("episodes.val", lambda r: list(r["episodes"]["val"])),
    ("context", lambda r: int(r["context"])),
    ("horizon", lambda r: int(r["horizon"])),
    ("device", lambda r: str(r["device"])),
    ("torch_version", lambda r: str(r.get("torch_version", ""))),
    ("git_sha", lambda r: str(r.get("git_sha", ""))),
    ("z_cats", lambda r: int(r["z_cats"])),
    ("z_classes", lambda r: int(r["z_classes"])),
)
"""Every field `require_one_protocol` compares, as `(name, pick)`, in one
module-level table that the function ITERATES and a test reads -- so a field
cannot be added to the comparison without the test noticing it has no
disagreement case. While such a tuple lived inside `latent_width`'s namesake no
test could see it, and a check on the test's own literal compared that literal
to itself.

`z_cats` AND `z_classes` ARE THE MEASURED SHAPE, read off each checkpoint's own
`rssm.cfg`, and they are here for the reason M3k's review put `h_dim` in the
width table: `projection_seed` and `rung_width` were CONSTANTS that cannot vary
between two records of one code version, while the one width the record MEASURED
went unchecked. These two set `CEILING_BITS`, which both of Reading G's cuts are
a fraction of, so two records that disagree on them were measured against
different ceilings and would pool into one reading with no refusal anywhere --
`SPARE_CUT` is half of 160 for one and half of something else for the other.
Nothing else would notice: the floor, the two ceiling routes and both
inequalities are all satisfied on either shape.

`windows.window` is here because `windows.episode` alone does not identify a
row: two records that agree on every episode label but differ on the window
indices within them were scored on different rows and would pool with no
refusal.

`ceiling_bits`, `split_seed` and the three `CHECK_FLAGS` are deliberately NOT
compared. The first two are written from module constants and so cannot vary
between two records of one code version -- comparing them is the tautology this
table exists to avoid -- and a failed check is a FINDING the read phase turns
into 43, not a protocol disagreement."""


def require_one_protocol(records: dict) -> None:
    """Every record reports the same protocol on the same windows, at the same
    latent shape, or the two that disagree are named with the field.

    Nine cells pooled into one reading must describe the same rows: two
    protocols pooled as one would be a reading over a union nothing measured.
    Mirrors `scripts/latent_width.py`'s namesake. The two fields older records
    may lack are read with `.get(...)` so a record without them is defaulted
    identically across every record rather than raising a bare `KeyError`;
    `git_sha` is among them, and its absence from an earlier version of that
    function let records from different torch builds and code versions pool into
    one finding with no refusal at all.

    `clusters` and `rows` are NOT compared here, and they are not unguarded:
    they are print-only (no reading can flip on them), but Task 5's
    `capacity_inputs` takes ONE value of each for a caption that speaks for nine
    cells, so it refuses a disagreement itself, at the point where the pick is
    made, rather than depending on this function having run first.
    """
    items = sorted(records.items())
    if not items:
        raise SystemExit("no capacity record to read")
    (first_cell, first) = items[0]
    for cell, record in items[1:]:
        for field, pick in _PROTOCOL_FIELDS:
            mine, theirs = pick(record), pick(first)
            if mine != theirs:
                shown = (
                    f" ({len(mine)} vs {len(theirs)} rows)"
                    if field in ("windows.episode", "windows.window")
                    else f": {mine!r} vs {theirs!r}"
                )
                raise SystemExit(
                    f"{cell[0]} seed {cell[1]} and {first_cell[0]} seed "
                    f"{first_cell[1]} disagree on {field}{shown}; they are not "
                    "one measurement and their arms cannot be read against one "
                    "bar"
                )


def load_capacity(out_dir: Path, arms, seeds) -> dict:
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
            path = capacity_record_path(out_dir, arm, int(seed))
            if not path.exists():
                raise CellMissing(
                    f"{arm} seed {int(seed)}: no capacity record at {path}; "
                    "run --phase measure first"
                )
            record = load_record(path)
            if record.get("arm") != arm or int(record.get("seed", -1)) != int(seed):
                raise SystemExit(
                    f"{path.name} was read for {arm} seed {int(seed)} but its "
                    f"record says arm={record.get('arm')!r} "
                    f"seed={record.get('seed')!r}"
                )
            records[(arm, int(seed))] = record
    return records


if __name__ == "__main__":
    # `_parser` and `main` belong with the read phase and arrive with it. Until
    # then this refuses rather than doing nothing: a script that exits 0 having
    # measured nothing is the one outcome a driver cannot tell from success.
    raise SystemExit(
        "scripts/latent_capacity.py has no command line yet: the measure phase "
        "is importable (`measure_phase`), and `--phase`/`main` arrive with the "
        "read phase"
    )
