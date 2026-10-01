"""M3l: how many bits does the posterior code carry, and about what?

M3j established the study's largest, most unanimous effect -- `h` carries
observed motion and `z` does not, 9/9 in both targets. But that observation is
COMMON TO BOTH LEVERS: "`z` lacks the capacity" and "`z` was never asked"
predict it equally. M3k was built to separate them and returned
INDISTINGUISHABLE at effect/noise 0.22, and more seeds make that worse rather
than better -- a majority bar recedes as n grows whenever the per-seed rate is
below one half.

So this module asks the question that separates them directly, and asks it as a
LEVEL AGAINST AN EXACT CEILING rather than as a difference of two fitted
quantities. That is the property M3k lacked. Every win this project has had has
been a level (the anchors, compared with `==` against a known value); every
stall has been a difference.

Pure in its own code: numpy only, plus the shape constants it reads off
`RSSMConfig` (which loads torch transitively -- deriving the ceiling rather than
re-spelling it is the stronger constraint). No `Path`, no I/O, no record schema
-- those live in
`scripts/latent_capacity.py`.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, CONFIDENCE, RESAMPLES, SEEDS_REQUIRED, BaseControl,
)
from mbfps.models.rssm import RSSMConfig

CEILING_BITS: float = RSSMConfig.z_cats * math.log2(RSSMConfig.z_classes)
"""The most the posterior code can carry: one `log2(z_classes)` per categorical.

DERIVED from the model's own config rather than re-spelled. A hardcoded 160.0
would silently disagree with the model if the latent shape changed, and both of
Reading G's cuts are fractions of this."""

SPARE_CUT: float = CEILING_BITS / 2
"""Below this, most of the capacity is idle.

HALF THE DERIVED CEILING, not half the measured `ceiling_bits` -- otherwise a
code that collapsed its own ceiling could satisfy the threshold by degenerating,
which is the opposite of what the status detects. The half is one principle's
third instance: M3k's seed bar is the weakest MAJORITY of seeds, its derivation
at five seeds gives three, and here a majority of the capacity. The claim
"capacity is the binding constraint" requires that most of the capacity is in
use."""

FRAME_CUT: float = 0.5
"""Above this, most of the code's variance is explained by the current frame,
which is what the embedding loss asks for. The same half, the same principle."""


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in BITS over the last axis, with `0 log 0 = 0`.

    `np.where` guards the log twice on purpose: once to select, once to keep the
    argument positive, because `np.where` evaluates both branches and
    `log2(0)` would warn before being discarded. Test output must be pristine.
    """
    safe = np.where(p > 0.0, p, 1.0)
    return -np.where(p > 0.0, p * np.log2(safe), 0.0).sum(axis=-1)


def _require_distributions(probs: np.ndarray) -> np.ndarray:
    """`probs` as `(N, z_cats, z_classes)` of proper distributions, or a refusal.

    A caller handing logits instead of probabilities would otherwise get a
    number, and it would look plausible against a 160-bit ceiling rather than
    obviously wrong.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 3 or probs.shape[1:] != (RSSMConfig.z_cats, RSSMConfig.z_classes):
        raise ValueError(
            f"probs must be (N, z_cats, z_classes) = "
            f"(N, {RSSMConfig.z_cats}, {RSSMConfig.z_classes}), got {probs.shape}: "
            "a flattened array cannot tell the categorical groups apart, and the "
            "estimator sums one entropy per group"
        )
    if (probs < 0.0).any():
        raise ValueError(
            "probs carries a negative value, so it is not a distribution; "
            "logits were probably handed in place of probabilities"
        )
    if probs.shape[0] == 0:
        raise ValueError(
            "probs has no rows: `bits_carried` would divide by zero and return "
            "NaN with a warning, and NaN compares False against every cut, so a "
            "gather that produced nothing would read as a clean null rather than "
            "as the refusal it is"
        )
    sums = probs.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError(
            f"each categorical must sum to 1, got {sums.min():.6f}..{sums.max():.6f}; "
            "logits were probably handed in place of probabilities"
        )
    return probs


def bits_carried(probs: np.ndarray) -> float:
    """`Sum_j I(z_j ; h, enc(t))` in bits: the per-categorical informations summed.

    AN UPPER BOUND ON THE CODE'S JOINT INFORMATION, not the joint information
    itself, because redundancy ACROSS categoricals is counted once per
    categorical. Measured: 32 categoricals that all copy one 5-bit variable read
    160.0000 -- the full ceiling -- while carrying 5 bits jointly, a 32x
    overstatement. An independent code of the same reading carries all 160.
    Nothing here can tell those two apart.

    What that does and does not license:
      - A reading BELOW a cut is sound and conservative: if the upper bound is
        below the cut, the joint information is too.
      - A reading ABOVE a cut does NOT establish "the capacity is in use", since
        a fully redundant code reads the ceiling while being almost entirely
        idle. Any status resting on a high reading must say so.

    `H(marginal) - E_n H(row)`, summed over the categoricals. Exact from the
    distributions -- no sampling.

    Nor is it "information about the frame": the posterior conditions on `h` AND
    `enc(t)`, and this measures how much the code varies across rows for any
    reason. `frame_share` is what separates those two sources.
    """
    probs = _require_distributions(probs)
    marginal = probs.mean(axis=0)
    return float(entropy_bits(marginal).sum() - entropy_bits(probs).sum(axis=1).mean())


def floor_bits(probs: np.ndarray) -> float:
    """Every row replaced by the marginal. ZERO by construction, to summation
    order.

    A substitution into the real estimator on the real data, not a fixture --
    the same shape of control as M3k's anchors, whose answer is known in
    advance. `H(m) - mean_n H(m)` is zero in exact arithmetic, so any reading
    away from zero beyond float summation drift is an arithmetic defect. NOT
    bit-exact -- compare with a tolerance, never `==`. It does NOT pin the log
    base: measured, nats also reads 0.000000.
    """
    probs = _require_distributions(probs)
    marginal = probs.mean(axis=0)
    return bits_carried(np.broadcast_to(marginal, probs.shape).copy())


def ceiling_bits(probs: np.ndarray) -> float:
    """Every row replaced by a one-hot at its argmax: what this code's argmax
    PATTERN carries. `argmax_marginal_bits` is the second route.

    NOT an upper bound on `bits_carried`, and not to be refused on as one. A code
    whose argmax never moves while its tail varies reads 7.5207 bits carried
    against 0.0000 here. Its only job is the two-route routing check."""
    probs = _require_distributions(probs)
    onehot = np.zeros_like(probs)
    np.put_along_axis(onehot, probs.argmax(axis=-1)[..., None], 1.0, axis=-1)
    return bits_carried(onehot)


def argmax_marginal_bits(probs: np.ndarray) -> float:
    """`ceiling_bits` again, from the argmax INDICES via a histogram.

    Two independent routes to one answer pins the routing -- a mean over the
    wrong axis, or a substitution that leaks a row's own distribution, breaks
    the agreement. It does not pin the log base; both routes agree in nats too.
    """
    probs = _require_distributions(probs)
    idx = probs.argmax(axis=-1)
    n = idx.shape[0]
    total = 0.0
    for j in range(idx.shape[1]):
        share = np.bincount(idx[:, j], minlength=probs.shape[-1]) / n
        total += float(entropy_bits(share))
    return total


def live_classes(probs: np.ndarray) -> int:
    """How many of the `z_cats * z_classes` columns vary across the rows.

    A class that never varies is capacity the code is not using, and `R^2` is
    undefined for it, so `frame_share` excludes it. Reported beside the verdict:
    a CAPACITY_BOUND read printed next to a low count would be
    self-contradicting.

    `max > min`, not `std > 0`: the standard deviation of identical float64
    values is rounding noise rather than zero (measured: a fully collapsed code
    read 1013 of 1024 columns live at 300 rows), so `std > 0.0` would call a
    column that never varies live -- in exactly the posterior-collapse case this
    count exists to report.
    """
    probs = _require_distributions(probs)
    flat = probs.reshape(probs.shape[0], -1)
    return int((flat.max(axis=0) > flat.min(axis=0)).sum())


PAIR_CEILING_BITS: float = math.log2(RSSMConfig.z_classes)
"""The most one pair of categoricals can share: `log2(z_classes)`, at full
redundancy. Derived, like `CEILING_BITS`, rather than re-spelled."""

RATIO_MIN_FLOOR: float = 1e-9
"""Below this a floor is float noise, and a ratio against it measures noise.

The same 1e-9 every other tolerance in this module uses: summation order moves
a zero by ~1e-13, and a floor that is real is orders of magnitude above that."""


def _pair_mutual_information(probs: np.ndarray) -> float:
    """Mean over the `j < k` pairs of `I(z_j ; z_k)`, in bits, from the
    DISTRIBUTIONS.

    The pair joint is `joint[j,k,a,b] = (1/n) * sum_n p_j(a|n) * p_k(b|n)`: the
    joint of two categorical values SAMPLED from one row's distributions,
    averaged over rows. That is the dependence between what the model actually
    draws, and it is the same object `bits_carried` measures one categorical at a
    time -- which is the point. An argmax basis cannot see redundancy living in
    the tails. Measured at 11,008 rows, independent argmaxes with four per-row
    values shared across all 32 categoricals: an argmax basis reads 1.004x its
    floor while `bits_carried` is 98.32, "capacity in use", and the
    distribution-level redundancy is 12.55x its floor. (The review's fixture read
    89.73 and 4x.)

    One `einsum`, then the upper triangle. The diagonal (`I(z_j ; z_j)`, a
    categorical's own entropy) and the lower triangle (the same pairs again) are
    never read.
    """
    n, cats, _ = probs.shape
    series = probs.transpose(1, 0, 2)  # (cats, n, classes)
    joints = np.einsum("ina,jnb->ijab", series, series, optimize=True) / n
    first, second = np.triu_indices(cats, k=1)
    joint = joints[first, second]  # (pairs, classes, classes)
    outer = joint.sum(axis=2)[:, :, None] * joint.sum(axis=1)[:, None, :]
    nz = joint > 0.0
    ratio = np.where(nz, joint / np.where(nz, outer, 1.0), 1.0)
    return float(np.where(nz, joint * np.log2(ratio), 0.0).sum(axis=(1, 2)).mean())


def redundancy_bits(probs: np.ndarray) -> float:
    """How much the categoricals duplicate each other, in bits per pair.

    A COMPANION THAT DECIDES NOTHING, and the one that makes a high
    `bits_carried` interpretable. `bits_carried` sums the per-categorical
    informations, so redundancy across categoricals is counted once per
    categorical: 32 categoricals that all copy one 5-bit variable read the full
    160-bit ceiling while carrying 5 bits jointly. Nothing in `bits_carried`
    itself can tell that code from an independent one of the same reading.

    The mean pairwise mutual information between the categoricals' DISTRIBUTIONS
    (see `_pair_mutual_information`), against a `PAIR_CEILING_BITS` per-pair
    ceiling. Read it against `redundancy_floor`, never against zero: finite-sample
    bias puts an independent code at ~0.064 bits per pair at 11,008 rows. Near
    the floor, the categoricals are pairwise independent and the summed reading
    approximates the joint one -- so a high `bits_carried` does mean the capacity
    is in use. Far above it, the reading is inflated and a high value establishes
    nothing about capacity.

    PAIRWISE ONLY. Dependence that exists only among three or more categoricals
    (`z_3 = z_1 XOR z_2` is pairwise independent) is invisible here, so a ratio
    near 1 is evidence against redundancy and not a proof of its absence.

    A fully collapsed code reads ~0.0 here and ~0.0 on the floor, so the ratio is
    0/0: form it with `redundancy_ratio`, which returns None where it is undefined.

    Cost, measured at 11,008 rows for all 496 pairs: 0.040 s and 0.029 GB of
    extra peak memory (median of 7; `redundancy_floor` adds a roll, 0.055 s and
    0.18 GB, because it copies the shifted rows). The argmax loop this replaced
    measured 0.030 s on the same machine, so what the argmax basis saved was ~0.01
    s per call -- and what it cost was blindness to the tails. A cell pays ~0.1 s
    for both.
    """
    return _pair_mutual_information(_require_distributions(probs))


def redundancy_floor(probs: np.ndarray, *, seed: int) -> float:
    """`redundancy_bits`' finite-sample null, calibrated on THIS data.

    Mutual information is biased upward at finite sample: an independent code
    reads 0.0643 bits per pair at 11,008 rows (0.0641-0.0644 across seeds), not 0.
    That sits 2% above Miller-Madow's `(K-1)^2 / (2 n ln 2)` = 0.0630 for
    `K = z_classes`, the independent known answer the tests pin it to. So "near
    zero" is the wrong comparison and a fixed tolerance would be a guess about the
    sample size.

    A NULL CALIBRATION, not a known answer. `floor_bits` is zero by algebra; this
    is an empirical null -- what the estimator reads when the cross-categorical
    dependence has been destroyed -- and it carries sampling noise of its own, so
    it varies with `seed`.

    Each categorical's rows are rolled CIRCULARLY by its own random offset. That
    destroys dependence BETWEEN categoricals while preserving each series' OWN
    autocorrelation, which is exactly the null the comparison needs. A full
    permutation destroys the autocorrelation too, and this gather's rows are
    clustered -- windows of consecutive frames inside episodes -- so against a
    permutation floor independent categoricals that merely persist look massively
    redundant, and the ratio would measure autocorrelation, not redundancy.
    Measured on independent categoricals persisting within 50-row windows
    (persistence 0.5 / 0.8 / 0.95, 5,000 rows, soft rows): redundancy over a
    permutation floor 1.61x / 3.90x / 10.74x, over this circular-shift floor
    1.01x / 1.04x / 1.12x. The review's fixture read 3.28x / 14.01x / 27.00x
    against 1.28x / 1.36x / 1.34x.

    ROW ORDER MATTERS. `probs` must be in the gather's temporal order. Shuffled
    rows have no autocorrelation to preserve, and the roll degenerates to the
    permutation it replaces.

    Offsets are drawn independently, so by arithmetic about 2 x window / n of the
    pairs (0.9% at 11,000 rows and 50-row windows) land within a persistence
    length of each other and keep a little of their dependence. Measured, it does
    not show: 32 identical persistent categoricals read a floor of 0.372 at
    11,000 rows against 0.371 for 32 independent persistent ones.

    `seed` is REQUIRED: a defaulted seed is how a previous milestone shipped
    every cell sharing one bootstrap seed. Pass the cell's own.
    """
    probs = _require_distributions(probs)
    n, cats, _ = probs.shape
    offsets = np.random.default_rng(seed).integers(0, n, size=cats)
    shifted = np.stack(
        [np.roll(probs[:, j], int(offset), axis=0) for j, offset in enumerate(offsets)],
        axis=1,
    )
    return _pair_mutual_information(shifted)


def redundancy_ratio(redundancy: float, floor: float) -> float | None:
    """`redundancy_bits / redundancy_floor`, or None where it is undefined.

    UNDEFINED when the floor is float noise (below `RATIO_MIN_FLOOR`): no pair of
    categoricals varies, which is a fully collapsed code and also a code with a
    single live categorical. Both numbers then read ~0.0 -- and not merely close:
    every row identical makes the circular shift the identity, so the floor
    equals the redundancy exactly (measured: 3.4e-17 for both at 300 rows,
    -4.5e-16 for both at 11,000, since MI can round negative). The naive ratio is
    therefore exactly 1.0, the single reading that says "independent, so the
    capacity is in use", for a code with nothing in it.

    None rather than NaN, on purpose: NaN compares False against every cut and
    would read as a clean answer in both senses at once, while `None > 3` raises.
    None rather than a refusal: posterior collapse is a legitimate outcome of
    this milestone, and the ratio gates nothing, so a collapsed cell must not
    abort the measure phase. A reader should print it as "not defined", and it
    means nothing for a collapsed code anyway -- `bits_carried` is at the bottom
    of the scale there and no status rests on a high reading.
    """
    if not (math.isfinite(redundancy) and math.isfinite(floor)):
        raise ValueError(
            f"redundancy={redundancy} and floor={floor} must be finite: a "
            "non-finite input is an error about the measurement, never a ratio"
        )
    if floor < RATIO_MIN_FLOOR:
        return None
    return redundancy / floor


@dataclasses.dataclass(frozen=True)
class EpisodeStats:
    """Per-episode sufficient statistics for `bits_carried`.

    `bits_carried` is a function of exactly three per-episode quantities: the
    summed probabilities, the summed row entropies and the row count. So the
    bootstrap resamples these small arrays instead of re-slicing an
    (11000, 32, 32) one -- measured, 0.02 ms against 54.89 ms per resample, a
    2601x speedup, and the reason the whole run is ~30 minutes rather than ~9.

    Exact, not an approximation -- but exact up to summation ORDER, so a test
    compares with `approx` rather than `==`. An M3h sweep once refused on an
    8.527e-14 mismatch that was summation order and not a defect.
    """

    prob_sum: np.ndarray   # (n_episodes, z_cats, z_classes)
    entropy_sum: np.ndarray  # (n_episodes,)
    rows: np.ndarray       # (n_episodes,)
    labels: np.ndarray     # (n_episodes,)


def episode_stats(probs: np.ndarray, groups: np.ndarray) -> EpisodeStats:
    """`EpisodeStats` for `probs` clustered by `groups`, one entry per label."""
    probs = _require_distributions(probs)
    groups = np.asarray(groups)
    if groups.shape != (probs.shape[0],):
        raise ValueError(
            f"groups must be one label per row; got {groups.shape} for "
            f"{probs.shape[0]} rows"
        )
    row_entropy = entropy_bits(probs).sum(axis=1)
    labels = np.unique(groups)
    masks = [groups == g for g in labels]
    return EpisodeStats(
        prob_sum=np.stack([probs[m].sum(axis=0) for m in masks]),
        entropy_sum=np.array([row_entropy[m].sum() for m in masks]),
        rows=np.array([int(m.sum()) for m in masks], dtype=np.float64),
        labels=labels,
    )


def _bits_from_stats(stats: EpisodeStats, picked: np.ndarray) -> float:
    """`bits_carried` over the episodes `picked` names, from the statistics."""
    total = stats.rows[picked].sum()
    marginal = stats.prob_sum[picked].sum(axis=0) / total
    return float(entropy_bits(marginal).sum() - stats.entropy_sum[picked].sum() / total)


def bits_interval(
    stats: EpisodeStats, *, resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE, seed: int = 0,
) -> dict:
    """`bits_carried` with an episode-clustered percentile interval.

    Episodes, not windows: windows are cut non-overlapping but several windows
    from one trajectory are not independent observations, and every reading from
    M3e onward clusters on episodes.
    """
    n_episodes = stats.labels.size
    if n_episodes < 2:
        raise ValueError(
            "a bootstrap interval needs at least two resampling units (distinct "
            f"episode labels); got {n_episodes}"
        )
    index = np.arange(n_episodes)
    rng = np.random.default_rng(seed)
    draws = np.array([
        _bits_from_stats(stats, rng.choice(index, size=n_episodes, replace=True))
        for _ in range(resamples)
    ])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "bits": _bits_from_stats(stats, index),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "n_episodes": int(n_episodes),
    }


READING_COLUMNS: tuple[str, ...] = (
    "arm", "bits", "ci_low", "ci_high", "frame", "live", "red_ratio", "clears",
)
"""The columns of Reading G's table, in order.

`live` and `red_ratio` are the two COMPANIONS, and they are in the table rather
than only in the records because `bits_carried` sums per-categorical
informations: it upper-bounds the code's joint information, so a high reading
does not establish "the capacity is in use". A `CAPACITY_BOUND` verdict printed
beside a `red_ratio` far above 1, or beside a low `live`, contradicts itself, and
a reader can only see that if both are on the page beside the verdict."""

READING_WIDTHS: tuple[int, ...] = (13, 10, 10, 10, 8, 7, 11, 13)
"""One width per column, each wider than the widest value the column can carry.

The cells are right-aligned and unseparated, so a value as wide as its column
butts against its neighbour and a wider one shifts the whole row. The widest
cells are `undefined` (9) in `red_ratio` and `spare+frame` (11) in `clears`."""


@dataclasses.dataclass(frozen=True)
class CapacityArm:
    """One arm's reading, summarised across its seeds.

    The intervals are the CONSERVATIVE hull -- the lowest low and the highest
    high across the arm's seeds -- so an arm is never credited with a bound only
    its luckiest seed reached. `bits`, `frame_share` and both redundancy figures
    are the seed means; `live` is the lowest count.

    Neither point estimate decides anything: the two `seeds_*` tallies do, and
    they are counted per seed against the cuts before ever being averaged. The
    two redundancy figures decide nothing either. They are carried so the table
    can print `redundancy_ratio` beside the verdict, which is the only thing that
    makes a high `bits` readable.
    """

    bits: float
    bits_low: float
    bits_high: float
    frame_share: float
    frame_low: float
    frame_high: float
    live: int
    redundancy_bits: float   # seed mean; a companion that decides nothing
    redundancy_floor: float  # seed mean of the circular-shift null
    seeds_spare: int
    seeds_frame: int
    seeds_total: int

    def clears_spare(self) -> bool:
        """`SEEDS_REQUIRED` seeds whose whole bits interval sits below the cut."""
        return self.seeds_spare >= SEEDS_REQUIRED

    def clears_frame(self) -> bool:
        """`SEEDS_REQUIRED` seeds whose whole frame interval sits above the cut."""
        return self.seeds_frame >= SEEDS_REQUIRED


@dataclasses.dataclass(frozen=True)
class CapacityInputs:
    """Reading G's inputs: the arms, their base controls, their estimator checks.

    `controls` is per arm and True when every one of that arm's seeds passed the
    floor, two-route and inequality checks. `clusters` and `rows` are caption
    figures and decide nothing.
    """

    arms: dict[str, CapacityArm]
    base: dict[str, BaseControl]
    controls: dict[str, bool]
    clusters: int
    rows: int


@dataclasses.dataclass(frozen=True)
class CapacityStatus:
    status: str
    rule: str
    arms_spare: tuple[str, ...]
    arms_frame: tuple[str, ...]
    base_failed: tuple[str, ...]
    controls_failed: tuple[str, ...]


def capacity_arm(seeds: list[dict]) -> CapacityArm:
    """One arm's `CapacityArm` from its per-seed measurement dicts.

    THE REFUSALS LIVE HERE, and they raise. A non-finite bound makes every
    comparison False, so one bad cell would read "does not clear" in both
    senses at once -- a verdict moved toward the wrong status while looking like
    a clean null. An inverted interval, a point estimate outside its own
    interval, and fewer than `SEEDS_REQUIRED` seeds are refused for the same
    reason: each would otherwise pass silently and read as a null.
    `retention.rung_arm` refuses the same three.

    The tallies are made HERE, per seed, against the interval's own end: a seed
    is spare when ITS `ci_high` is strictly below `SPARE_CUT` and framey when ITS
    `frame_low` is strictly above `FRAME_CUT`. Never the point estimate, which
    can sit on the clearing side of a cut while its interval straddles it, and
    never the hull, which one unlucky seed widens past a cut the others clear.
    """
    if len(seeds) < SEEDS_REQUIRED:
        raise ValueError(
            f"an arm needs at least SEEDS_REQUIRED={SEEDS_REQUIRED} seeds to "
            f"read, got {len(seeds)}: an arm with fewer can never satisfy the "
            "bar, so it would read as a null rather than as the refusal it is"
        )
    fields = ("bits", "ci_low", "ci_high", "frame_share", "frame_low", "frame_high",
              "redundancy_bits", "redundancy_floor")
    for i, seed in enumerate(seeds):
        for key in (*fields, "live"):
            if key not in seed:
                raise ValueError(f"seed index {i} is missing {key}")
        values = {k: float(seed[k]) for k in fields}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(
                f"seed index {i} carries a non-finite value {values}: that is an "
                "error about the measurement, never a statement about the code"
            )
        for lo, mid, hi in (("ci_low", "bits", "ci_high"),
                            ("frame_low", "frame_share", "frame_high")):
            if values[lo] > values[hi]:
                raise ValueError(
                    f"seed index {i} has an inverted interval "
                    f"[{values[lo]}, {values[hi]}]"
                )
            if not values[lo] <= values[mid] <= values[hi]:
                raise ValueError(
                    f"seed index {i} has {mid}={values[mid]} outside its own "
                    f"interval [{values[lo]}, {values[hi]}]"
                )
    return CapacityArm(
        bits=float(np.mean([s["bits"] for s in seeds])),
        bits_low=float(min(s["ci_low"] for s in seeds)),
        bits_high=float(max(s["ci_high"] for s in seeds)),
        frame_share=float(np.mean([s["frame_share"] for s in seeds])),
        frame_low=float(min(s["frame_low"] for s in seeds)),
        frame_high=float(max(s["frame_high"] for s in seeds)),
        live=int(min(s["live"] for s in seeds)),
        redundancy_bits=float(np.mean([s["redundancy_bits"] for s in seeds])),
        redundancy_floor=float(np.mean([s["redundancy_floor"] for s in seeds])),
        seeds_spare=sum(1 for s in seeds if float(s["ci_high"]) < SPARE_CUT),
        seeds_frame=sum(1 for s in seeds if float(s["frame_low"]) > FRAME_CUT),
        seeds_total=len(seeds),
    )


def reading_capacity(inputs: CapacityInputs) -> CapacityStatus:
    """Reading G: is `z` out of room, or was it never asked?

    Precedence. `UNRESOLVED_ESTIMATOR` outranks everything: a reading taken from
    an estimator that missed a known answer is not a weaker reading, it is not a
    reading. Then `UNRESOLVED_BASE`, as in M3j and M3k -- a claim about the code
    means nothing if the current frame cannot say where it is. Then
    `SPARE_CAPACITY`, then `FRAME_REENCODING`, then the fall-through.

    THE TWO READINGS CAN BOTH CLEAR, and that is not an ambiguity. They are
    tallies on two different quantities -- how many bits, and how much of the
    code's variance enc(t) explains -- so one arm can sit in both lists: a code
    that carries little, and of that little mostly the frame. The cuts do not
    make them exclusive. What makes the status unambiguous is that precedence
    says which label wins (`SPARE_CAPACITY`, "not spare" being the other's
    condition) and that BOTH labels send the next milestone to the same lever,
    so the order decides a label and not a direction -- which is why M3k's
    refusal of an arm clearing opposite ways has no counterpart here. Both tuples
    are reported, so the record shows the overlap and the table labels it.

    `CAPACITY_BOUND` IS THE FALL-THROUGH. Both objective-lever statuses must
    clear an interval; this one is what remains. So the rule makes the lever
    M3j's h-vs-z evidence already favours the EASIEST status to reach, which is
    backwards, and the rule text says so in the words M3k used for
    INDISTINGUISHABLE: by default rather than by evidence.

    It also does not establish that the capacity is in use, which is the one
    positive claim the bottleneck lever rests on. `bits_carried` sums the
    per-categorical informations and so upper-bounds the joint, so a reading
    BELOW a cut is sound and a reading above one is not evidence of use. The
    rule sends the reader to the two columns that make it readable, `red_ratio`
    and `live`. It cannot carry their VALUES: they decide nothing, and
    `CapacityStatus` is compared whole to prove it.

    All four tuples are reported in every branch where a reading is taken, and
    the two UNRESOLVED branches report the failures that decided them while
    leaving the arms empty, because no arm votes there. `ARMS_REQUIRED` arms
    holding is enough to proceed, so one arm can fail its base control while its
    measurement still votes -- the record must say so rather than read "nothing
    failed". M3k shipped exactly that hole and it survived 82 tests.
    """
    if len(inputs.arms) < ARMS_REQUIRED:
        raise ValueError(
            f"Reading G needs at least ARMS_REQUIRED={ARMS_REQUIRED} arms to "
            f"read, got {len(inputs.arms)}"
        )
    if set(inputs.base) != set(inputs.arms) or set(inputs.controls) != set(inputs.arms):
        raise ValueError(
            "inputs.base and inputs.controls must name exactly inputs.arms; got "
            f"arms {sorted(inputs.arms)}, base {sorted(inputs.base)}, "
            f"controls {sorted(inputs.controls)}: an arm with no base control or "
            "no estimator check would vote with nothing gating it"
        )
    controls_failed = tuple(sorted(a for a, ok in inputs.controls.items() if not ok))
    base_failed = tuple(
        sorted(a for a, control in inputs.base.items() if not control.clears())
    )
    spare = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_spare()))
    frame = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_frame()))
    n_arms = len(inputs.arms)

    if controls_failed:
        return CapacityStatus(
            status="UNRESOLVED_ESTIMATOR",
            rule=(
                f"the estimator missed a known answer in {', '.join(controls_failed)}; "
                "a reading taken from an estimator that failed its own floor, "
                "two-route or inequality check is not a weaker reading, it is not "
                "a reading"
            ),
            arms_spare=(), arms_frame=(), base_failed=base_failed,
            controls_failed=controls_failed,
        )
    if n_arms - len(base_failed) < ARMS_REQUIRED:
        return CapacityStatus(
            status="UNRESOLVED_BASE",
            rule=(
                f"only {n_arms - len(base_failed)} of {n_arms} arms clear "
                f"enc(t) -> position at r2 {BASE_R2_FLOOR:.2f} "
                f"({', '.join(base_failed)} failed); a claim about what the code "
                "carries means nothing where the frame cannot say where it is"
            ),
            arms_spare=(), arms_frame=(), base_failed=base_failed,
            controls_failed=controls_failed,
        )
    if len(spare) >= ARMS_REQUIRED:
        return CapacityStatus(
            status="SPARE_CAPACITY",
            rule=(
                f"the whole bits interval sits below {SPARE_CUT:.0f} of the "
                f"{CEILING_BITS:.0f}-bit ceiling in {len(spare)} of {n_arms} arms "
                f"({', '.join(spare)}), each in at least {SEEDS_REQUIRED} seeds; "
                "bits is an upper bound on the code's joint information, so that "
                "is below the cut too: most of the capacity is idle, adding "
                "capacity cannot be what limits the model, and the OBJECTIVE "
                "lever is where the next milestone goes"
            ),
            arms_spare=spare, arms_frame=frame, base_failed=base_failed,
            controls_failed=controls_failed,
        )
    if len(frame) >= ARMS_REQUIRED:
        return CapacityStatus(
            status="FRAME_REENCODING",
            rule=(
                f"the whole interval on the share of the code's variance that "
                f"enc(t) alone explains sits above {FRAME_CUT:.2f} in {len(frame)} "
                f"of {n_arms} arms ({', '.join(frame)}), each in at least "
                f"{SEEDS_REQUIRED} seeds, and spare capacity did not clear first; "
                "most of the code is the current frame re-encoded, which is "
                "exactly what the embedding loss asks for -- the OBJECTIVE lever "
                "is where the next milestone goes"
            ),
            arms_spare=spare, arms_frame=frame, base_failed=base_failed,
            controls_failed=controls_failed,
        )
    return CapacityStatus(
        status="CAPACITY_BOUND",
        rule=(
            f"neither objective-lever status clears in {ARMS_REQUIRED} arms: the "
            "code could not be shown to be mostly idle and could not be shown to "
            "be mostly a re-encoding of the frame, so what limits it is "
            "consistent with capacity -- but this status is the FALL-THROUGH, not "
            "a bar that was cleared, so the bottleneck lever stands BY DEFAULT "
            "rather than by evidence. Nor does it establish that the capacity is "
            "in use: bits is an upper bound on the joint information, so a high "
            "reading means that only where red_ratio sits near 1 (the "
            "categoricals independent) and live is high, and a verdict beside a "
            "red_ratio well above 1 or a low live count contradicts itself. How "
            "much capacity would be enough is a magnitude and belongs in the "
            "run's own report, not here"
        ),
        arms_spare=spare, arms_frame=frame, base_failed=base_failed,
        controls_failed=controls_failed,
    )


def _row(values, widths) -> str:
    return "  " + "".join(f"{v:>{w}}" for v, w in zip(values, widths, strict=True))


def _clears_label(arm: CapacityArm) -> str:
    """Which cut(s) an arm's own seeds cleared, as the table's `clears` column.

    BOTH is a label of its own. The two senses are different quantities, so an
    arm can clear both, and a column printing only `spare` would read, beside the
    reading's `arms_frame`, as an arm that did not clear the frame cut."""
    spare, frame = arm.clears_spare(), arm.clears_frame()
    if spare and frame:
        return "spare+frame"
    return "spare" if spare else ("frame" if frame else "no")


def format_reading_capacity(reading: CapacityStatus, inputs: CapacityInputs) -> str:
    """Reading G as `capacity.txt` carries it, byte for byte.

    Every number in the caption and legend is interpolated, never a literal: the
    ceiling, both cuts and the ratio's tolerance from the module, the arm count
    from `inputs.arms`, the seed count from the arms' common `seeds_total`, the
    shape from `inputs` and `RSSMConfig`. This project has shipped a caption
    disagreeing with its own columns three times.

    `red_ratio` is `redundancy_ratio` of the arm's two seed means, and prints
    `undefined` where that is None. Never a bare 1.0: a collapsed code reads its
    redundancy equal to its floor, and 1.0 is the one value that says
    "independent, so the capacity is in use".

    `clears` is printed for every arm under every verdict, as the width table
    does, including the UNRESOLVED ones where no arm votes; the legend says the
    gates and the arm bar are the verdict's, not the column's.
    """
    counts = {arm.seeds_total for arm in inputs.arms.values()}
    if len(counts) != 1:
        raise ValueError(
            "the arms must share one seeds_total for the caption's "
            f"'{SEEDS_REQUIRED} of N seeds' to be true of every row, got "
            f"{ {a: arm.seeds_total for a, arm in sorted(inputs.arms.items())} }"
        )
    (seeds_total,) = counts
    lines = [
        f"--- Reading G: how many of the {CEILING_BITS:.0f} bits does the posterior "
        f"code carry, and is it a re-encoding of enc(t)? (spare when the whole bits "
        f"interval is below {SPARE_CUT:.0f}; frame when the whole share interval is "
        f"above {FRAME_CUT:.2f}; both in {SEEDS_REQUIRED} of {seeds_total} seeds and "
        f"{ARMS_REQUIRED} of {len(inputs.arms)} arms); "
        f"{inputs.rows} rows over {inputs.clusters} clusters ---",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for name, arm in sorted(inputs.arms.items()):
        ratio = redundancy_ratio(arm.redundancy_bits, arm.redundancy_floor)
        lines.append(_row(
            (name, f"{arm.bits:.4f}", f"{arm.bits_low:.4f}", f"{arm.bits_high:.4f}",
             f"{arm.frame_share:.4f}", str(arm.live),
             "undefined" if ratio is None else f"{ratio:.3f}", _clears_label(arm)),
            READING_WIDTHS,
        ))
    lines.append(
        "  red_ratio = redundancy / its circular-shift floor, a companion that "
        "gates nothing: near 1 the categoricals are independent and bits is close "
        "to the code's joint information; well above 1 bits overstates it "
        f"('undefined' where the floor is below {RATIO_MIN_FLOOR:g}: no pair of "
        "categoricals varies)"
    )
    lines.append(
        f"  live = columns of {RSSMConfig.z_cats * RSSMConfig.z_classes} that vary "
        "across rows; clears = the cut an arm's own seeds cleared, before the "
        f"gates and the {ARMS_REQUIRED}-arm bar that the verdict applies"
    )
    lines.append(
        "  base control (enc(t) -> position, must clear r2 "
        f"{BASE_R2_FLOOR:.2f}): " + ", ".join(
            f"{a} r2={inputs.base[a].r2:+.3f} "
            f"{inputs.base[a].seeds_clear}/{inputs.base[a].seeds_total}"
            for a in sorted(inputs.base)
        )
    )
    lines.append(
        "  estimator checks (floor exactly 0, two ceiling routes agreeing, the "
        "two theorem inequalities): " + ", ".join(
            f"{a}={'ok' if inputs.controls[a] else 'BROKEN'}"
            for a in sorted(inputs.controls)
        )
    )
    lines.append(
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}"
    )
    return "\n".join(lines)
