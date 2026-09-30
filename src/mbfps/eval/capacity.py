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

from mbfps.eval.retention import CONFIDENCE, RESAMPLES
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
