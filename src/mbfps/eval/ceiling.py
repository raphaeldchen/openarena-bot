"""M3o: what can the READOUT reach, independently of the model?

M3n returned NO_MAJORITY. Its gate failed in 4 of 9 cells because a perfect
one-step predictor beats a copying one by 0.576-11.824 map units against
readout errors of 131.88-259.06. Thirteen milestones have measured the world
model through that readout; this module measures the readout.

THREE EXPLANATIONS, and they call for different responses: the linear probe
family is too weak, the representations do not carry position, or position is
not recoverable from a single frame in this maze. The third is untested and
would explain why three different information sets -- embedding alone, latent
alone, both jointly -- all land within 0.03 of each other at r2 ~0.33.

WHY AN RBF AND NOT AN MLP. Kernel ridge is the SAME estimator as the shipped
probe, lifted into a nonlinear feature space, so the linear-versus-nonlinear
comparison is a capacity comparison within one family rather than across two
fitting procedures. It is closed form: no SGD, no seeds, no training-loop
variance entering the one measurement whose job is to bound variance. It is
also literally `probe._solve`'s dual branch with a kernel in place of the
Gram, so the memory fix and the nonlinear probe are one piece of machinery.

AND WHY "THE KERNEL WINS" IS NOT A FINDING. The kernel readout searches
{the linear kernel} u {an RBF at each bandwidth}, each over
`CEILING_RIDGES`, so the linear reference `fit_linear_probe` is a RUNG of its
search and the held-out selection cannot do worse than it on the split it
selects on. The finding is the MAGNITUDE of the difference and whether it is
resolvable -- the trap M3n walked into from the other side, where `skill` had
a positive point estimate in 8 of 9 cells and meant nothing until the interval
was drawn.

Nesting was first argued asymptotically (an RBF approaches the linear kernel as
its bandwidth grows) and a fixed diagonal jitter made that false in practice:
the widest rung is capped at an effective linear ridge far below the ~1e3 the
linear probe selects, and the RBF read 0.05-0.06 BELOW linear at linear r2
above ~0.37. See `fit_kernel_probe`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mbfps.eval.burden import strict_majority
from mbfps.eval.probe import (
    _block_bootstrap_ci, _mean_r2, _validate_groups, apply_probe, fit_probe,
    position_error,
)

BANDWIDTHS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""RBF length scales, as multiples of the median pairwise distance.

Expressed as multiples rather than absolutes so the ladder means the same
thing at D = 2048 and D = 37632, where raw distances differ by orders of
magnitude. The largest rung is where the kernel is closest to linear."""

NEIGHBOURS: tuple[int, ...] = (1, 4, 16, 64)
"""k for the k-NN readout. 1 is the aliasing question at its sharpest: if the
single nearest frame in feature space is at a distant true position, the
feature space aliases."""

CEILING_RIDGES: tuple[float, ...] = tuple(
    10.0 ** (-1.0 + 8.0 * i / 32) for i in range(33)
)
"""The ridge grid BOTH sides of `D - C` search: 33 rungs at a 10^0.25 step over
`probe.RIDGES`'s span, [1e-1, 1e7].

`RIDGES` steps by 100x, and that is too coarse to resolve an interior optimum.
Measured on a purely linear task (`null_floor`'s), the kernel ladder beat the
linear reference on the shipped grid by

    D=1024   +0.0109   (range +0.0087 .. +0.0128)
    D=2048   +0.0201   (range +0.0171 .. +0.0228)

-- and the gain an ORACLE ridge would have bought the linear reference over the
shipped grid at the same shapes is

    D=1024   +0.0110   (range +0.0089 .. +0.0125)
    D=2048   +0.0208   (range +0.0184 .. +0.0233)

The same number. Every figure here is a MEAN OVER 6 DRAWS (seeds 0-5, three
splits each as `3 * seed + i`), at n=2000, scored on the third split, with the
range of the 6 beside it; no single draw reads the mean. The RBF rungs were not
finding nonlinear structure, they were interpolating the ridge grid, and
`PROBE_LIMITED` would have fired on it.

On this grid, over the same 6 draws, the floor is -0.0001 at D=1024 (range
-0.0005 .. +0.0000) and -0.0000 at D=2048 (-0.0000 .. +0.0000), and the linear
reference rises to the oracle's value to within 0.0001: C is 0.4855 on the
shipped grid and 0.4964 here at D=1024 (oracle 0.4964), 0.4957 and 0.5164 at
D=2048 (oracle 0.5165). The SPAN is unchanged -- the optimum was interior at
every shape measured, 0 of 6 draws at an endpoint -- so only the step was
wrong. Where the ridge response has no sharp interior optimum, the raw-pixel
row at D >> n, there was no floor on either grid: at D=37632 (2 draws, not 6)
the floor is +0.0000 and -0.0000 on both, and C is 0.4643 and 0.4746 on the
shipped grid against 0.4643 and 0.4754 here.

IT CONTAINS `RIDGES` BITWISE, as every 8th rung, and `probe._solve_ladder`
writes each rung's diagonal from a pristine copy rather than from the previous
rung's. So this grid ADDS rungs and moves no number any earlier milestone
recorded. `probe.RIDGES` itself is untouched."""

CONFIDENCE: float = 0.95
"""The level of `difference_interval`'s percentile interval.

A PARAMETER here, unlike `headroom.CONFIDENCE`, which only labels percentiles
hard-coded elsewhere: this is passed to `probe._block_bootstrap_ci`, which
derives the percentile pair from it, so changing it moves both bounds."""

RESAMPLES: int = 2000
"""Episode-clustered bootstrap resamples behind `difference_interval`."""

_R2_AGREEMENT: float = 1e-6
"""How far a readout's stored `r2` may sit from its recomputation in
`difference_interval`. Honest inputs agree to the last bit (the same function on
the same float64 arrays); this leaves room for a caller that scored on a
narrower dtype, and is far below anything a gain is read to, so a disagreement
this size is a different number, not rounding."""


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of `a` and of `b`.

    Through the Gram identity |a|^2 + |b|^2 - 2 a.b, so the working set is
    n x m rather than the n x m x D of `a[:, None] - b[None]`. Rounding can
    leave a tiny negative where two rows coincide; it is clipped to 0.
    """
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(sq, 0.0)


def _median_scale(x: np.ndarray) -> float:
    """The median pairwise distance, the length unit `BANDWIDTHS` multiplies.

    Computed on a capped subsample: it is a scale, not a statistic, and the
    full pairwise matrix is n^2 for no gain.

    The distances come from `_sqdist`, not from the difference tensor. That
    tensor is n x n x D: at the 512-row cap and D = 2048 it is 4.3 GB, and
    `norm` allocates a second one, so this one line would have taken a
    latent-scale cell's fit to ~9 GB and the raw-pixel cell's (D = 37632) to
    ~160 GB. A test on the value cannot see the difference between the forms.

    A scale that is no larger than rounding error is no scale. For rows that
    coincide the Gram identity can return up to ~1e-8 x the row norm instead of
    0, depending on the BLAS, so "greater than 0" would accept it. The floor is
    1e-6 x the RMS row norm.
    """
    rng = np.random.default_rng(0)
    take = x if x.shape[0] <= 512 else x[rng.choice(x.shape[0], 512, replace=False)]
    off = np.sqrt(_sqdist(take, take)[~np.eye(len(take), dtype=bool)])
    scale = float(np.median(off))
    floor = 1e-6 * float(np.sqrt((take * take).sum(1).mean()))
    if not scale > floor:
        raise ValueError("every pair of feature rows is identical; no scale to set")
    return scale


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * _sqdist(a, b))


def _kernel_matrix(kernel: str, a: np.ndarray, b: np.ndarray, gamma: float | None) -> np.ndarray:
    """The `kernel` rung's matrix between the rows of `a` and of `b`.

    One definition shared by the fit and by `apply_readout`, so a readout is
    applied with exactly the kernel it was selected with.
    """
    if kernel == "linear":
        return a @ b.T
    if kernel == "rbf":
        return _rbf(a, b, gamma)
    raise KeyError(f"unknown kernel {kernel!r}; expected linear or rbf")


def _kernel_rungs(xs: np.ndarray, xv: np.ndarray, unit: float):
    """Each kernel in the ladder as `(kernel, gamma, bandwidth, fit gram, query gram)`.

    A generator, so one rung's n x n matrices are alive at a time: the whole
    ladder held at once is seven of them, ~300 MB at n = 2000.

    The linear rung comes first and its bandwidth is infinite, which is what a
    linear kernel is as a limit of the RBF family, and keeps `float(bandwidth)`
    meaningful for a caller that records it. Ties go to the earlier rung, so a
    nonlinear kernel has to BEAT linear on the selection split rather than match
    it.
    """
    yield "linear", None, float("inf"), xs @ xs.T, xv @ xs.T
    for multiple in BANDWIDTHS:
        gamma = 1.0 / (2.0 * (multiple * unit) ** 2)
        yield "rbf", gamma, multiple, _rbf(xs, xs, gamma), _rbf(xv, xs, gamma)


def fit_kernel_probe(x, y, val_x, val_y) -> dict:
    """Kernel ridge over {linear kernel} u {RBF at each bandwidth}, each over
    `CEILING_RIDGES`, selected TOGETHER on held-out data.

    The same three-split discipline as `fit_probe`: fit here, select the kernel,
    bandwidth and diagonal jitter on `val_*`, and the caller scores on a third
    split. A capacity selected on the fit split selects itself in, which is
    exactly what makes a nonlinear probe untrustworthy as evidence.

    THE LINEAR KERNEL IS A RUNG, which is what makes this nest `fit_probe`
    rather than approach it. Kernel ridge with the linear kernel `xs xs'` of the
    standardised fit rows, a jitter `lambda` on the diagonal and centred
    targets IS the dual form of `fit_probe`'s ridge regression at the same
    `lambda` (`probe._solve_ladder` computes exactly this), and the jitters are
    `CEILING_RIDGES`, the grid `fit_linear_probe` searches. So the search space
    contains `fit_linear_probe`'s and, selecting on the same split, the selected
    score on that split cannot be lower: that is exact. On the caller's third
    split it can be lower by selection noise, when an RBF rung edges linear on
    the selection split by ~1e-4 and loses ~1e-3 on the score split (worst
    -0.0009 over 12 draws of the embedding-shaped test fixture, against -0.038
    to -0.064 before).

    WHY NOT JUST AN RBF LADDER. The first version fixed the jitter at 1e-3 and
    leaned on the widest RBF being nearly linear. It is linear only to first
    order, with an effective linear ridge of ~1e-3 x (bandwidth x median
    distance)^2: ~130 at D = 1024 against the ~1e3 the linear probe selects. On
    a noisy low-rank task (D = 1024, n = 400, linear r2 0.45) that left the
    readout 0.046-0.064 BELOW the linear probe. Selecting the jitter over
    `RIDGES` with NO linear rung closes most of that (-0.001 to -0.003 on the
    same three fixtures) but not all of it, and not by construction: an RBF
    kernel's eigenvalues are bounded by n, so the jitters of 1e3 and up that the
    linear probe selects only flatten an RBF rung to the mean, and the rungs
    that can act are the small ones. The linear rung is what makes the nesting
    exact rather than close.

    THE LINEAR RUNG AND THE LINEAR REFERENCE SEARCH THE SAME GRID. Nesting is a
    claim about search spaces: with the ladder on `RIDGES` and `fit_linear_probe`
    on `CEILING_RIDGES` the ladder's linear rung would have 5 jitters against the
    reference's 33, and the ladder would no longer contain the reference. Its
    linear rung would read 0.32-0.35 where the reference reads 0.42-0.45 (n=300,
    D=150), and the RBF rungs would be left to make up the difference at the
    coarse jitters -- which is the defect `CEILING_RIDGES` exists to remove, with
    the roles swapped. They do not make it all up: `D - C` reads -0.0004 to
    -0.0021 on three draws, and on 4 of 6 the ladder's SELECTION r2 falls below
    the reference's by up to 0.0035. Both are `CEILING_RIDGES`. The jitter
    ladder's length is nearly free here: each kernel is eigendecomposed once and
    every jitter is then O(n^2), where the decomposition is O(n^3).

    EACH KERNEL IS EIGENDECOMPOSED ONCE, `K = V diag(vals) V'`, and every
    jitter is then `V ((V'(y - centre)) / (vals + jitter))`, O(n^2) per rung
    where a solve is O(n^3). The decomposition is itself the expensive step --
    at n = 2000 it is ~0.4 s, more than five LU solves of the same kernel
    (~0.15 s) -- so what this buys is a jitter ladder whose length is free, not
    a cheaper five-rung one. The jitters are all >= 0.1, so rounding's slightly
    negative eigenvalues (~1e-12 x the largest) never reach the denominator.

    The readout records which kernel was selected, as `kernel`, because the
    linear and RBF rungs apply differently. `kind` stays "rbf", the name of
    this estimator in the M3o readouts, whichever rung won. `bandwidth` is
    `inf` for a linear rung and `gamma` is None.
    """
    if len(CEILING_RIDGES) == 0:
        raise ValueError(
            "fit_kernel_probe: `CEILING_RIDGES` is empty, so there is no "
            "jitter to select among"
        )
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / scale
    xv = (np.asarray(val_x, dtype=np.float64) - mean) / scale
    unit = _median_scale(xs)
    centre = y.mean(0)
    best = None
    for kernel, gamma, bandwidth, gram, query in _kernel_rungs(xs, xv, unit):
        vals, vecs = np.linalg.eigh(gram)
        coords = vecs.T @ (y - centre)
        for jitter in CEILING_RIDGES:
            alpha = vecs @ (coords / (vals + jitter)[:, None])
            score = _mean_r2(query @ alpha + centre, val_y)
            if best is None or score > best[0]:
                best = (score, kernel, gamma, bandwidth, jitter, alpha)
    return {
        "kind": "rbf", "kernel": best[1], "mean": mean, "scale": scale,
        "anchors": xs, "alpha": best[5], "centre": centre, "gamma": best[2],
        "bandwidth": best[3], "jitter": best[4], "r2": best[0],
    }


def fit_knn_probe(x, y, val_x, val_y) -> dict:
    """k-nearest-neighbour regression, k selected on held-out data.

    Assumption-free, and the most direct test of aliasing: it asks whether
    near-in-feature-space implies near-in-position, which is the question.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / scale
    xv = (np.asarray(val_x, dtype=np.float64) - mean) / scale
    best = None
    for k in NEIGHBOURS:
        if k > xs.shape[0]:
            continue
        score = _mean_r2(_knn_predict(xs, y, xv, k), val_y)
        if best is None or score > best[0]:
            best = (score, k)
    if best is None:
        raise ValueError(
            f"no rung of NEIGHBOURS={NEIGHBOURS} fits {xs.shape[0]} fit rows"
        )
    return {
        "kind": "knn", "mean": mean, "scale": scale, "anchors": xs,
        "targets": y, "k": best[1], "r2": best[0],
    }


def _knn_predict(anchors, targets, query, k) -> np.ndarray:
    near = np.argpartition(_sqdist(query, anchors), kth=k - 1, axis=1)[:, :k]
    return targets[near].mean(axis=1)


def apply_readout(readout: dict, x: np.ndarray) -> np.ndarray:
    """Apply any of the three readouts, dispatching on `kind`.

    An "rbf" readout is further dispatched on its `kernel`, "linear" or "rbf",
    because the kernel ladder's selected rung may be either.

    A readout with no `kind` is the shipped linear probe and goes to
    `probe.apply_probe` UNCHANGED. Reimplementing that path would silently
    change thirteen milestones' worth of recorded numbers, so it is delegated
    rather than duplicated.
    """
    kind = readout.get("kind", "linear")
    if kind == "linear":
        return apply_probe(readout, x)
    if kind not in ("rbf", "knn"):
        # Before touching `mean`: an unknown readout has none, and the lookup
        # would raise `KeyError: 'mean'` and hide what was actually wrong.
        raise KeyError(f"unknown readout kind {kind!r}; expected linear, rbf or knn")
    xs = (np.asarray(x, dtype=np.float64) - readout["mean"]) / readout["scale"]
    if kind == "rbf":
        query = _kernel_matrix(readout["kernel"], xs, readout["anchors"], readout["gamma"])
        return query @ readout["alpha"] + readout["centre"]
    return _knn_predict(readout["anchors"], readout["targets"], xs, readout["k"])


def fit_linear_probe(x, y, val_x, val_y) -> dict:
    """The linear reference `C`: the shipped probe over `CEILING_RIDGES`.

    NOT the shipped probe's own number -- that is `fit_probe` at its default,
    which M3o records separately as `shipped_grid_r2` and reconciles against the
    study. `C` is the best LINEAR readout, so that `D - C` isolates
    nonlinearity rather than ridge-grid resolution.
    """
    return fit_probe(x, y, val_x, val_y, ridges=CEILING_RIDGES)


_NULL_RANK: int = 40
_NULL_NOISE: float = 120.0


def _null_split(seed: int, n: int, d: int):
    """One split of a SYNTHETIC purely-linear task: a low-rank signal pushed
    through a fixed random map into `d` dimensions, plus isotropic nuisance.

    The map and the target weights come from a generator that does NOT depend on
    `seed`, so the three splits sample ONE task. Drawn from `seed`, each split
    would get its own map and the readouts would be fit on one task and scored
    on another.
    """
    g = np.random.default_rng(777)
    a = g.normal(size=(_NULL_RANK, d)); w = g.normal(size=(_NULL_RANK, 4))
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, _NULL_RANK))
    x = z @ a + rng.normal(size=(n, d)) * 3.0
    return x, (z @ w) * 20.0 + rng.normal(size=(n, 4)) * _NULL_NOISE


def null_floor(n: int, d: int, *, seed: int = 0) -> float:
    """`D - C` at this shape when the target is EXACTLY linear in the features.

    The honest answer is 0: nothing nonlinear is there to find. Anything
    materially above it is the ladder beating the linear reference for
    instrument reasons, which would read as `PROBE_LIMITED` and mean nothing
    about the representation.

    MEASURED PER CELL rather than taken from a figure, because the floor proved
    sharply shape-dependent: on the shipped grid it was +0.0201 at D=2048 (mean
    over 6 draws, n=2000) and +0.0000 at D=37632 (2 draws). Three shapes do not
    license nine cells.
    """
    train, select, score = (_null_split(3 * seed + i, n, d) for i in range(3))
    linear = fit_linear_probe(*train, *select)
    kernel = fit_kernel_probe(*train, *select)
    return (_mean_r2(apply_readout(kernel, score[0]), score[1])
            - _mean_r2(apply_readout(linear, score[0]), score[1]))


FRAME_BUDGET: int = 12000
"""Frames drawn from the validation episodes, by a fixed stride.

STATED RATHER THAN DISCOVERED. `_solve`'s dual branch costs n^2 float64s:
12,000 frames is 1.15 GB (the whole 12,073-frame validation set is 1.17 GB),
20,000 is 3.2 GB. M3o's predecessor found an 18 GB allocation at run time,
which is the failure this constant exists to prevent."""

ZERO_INFORMATION_MEDIAN: float = 283.05
"""Median error of predicting the training mean, on the shipped split.

Measured on 12,073 validation frames from 24 episodes. The scale every
map-unit figure in this milestone is read against. Recomputed as a control
rather than trusted.

READ THE STATISTICS BESIDE IT. The shipped rollout probe's floor error of
131.88-259.06 is a MEAN over validation windows (`rollout.evaluate_rollout`
stacks the per-window errors and takes `.mean(axis=0)`), so its like-for-like
baseline is the zero-information MEAN, which the spec records as 292.82: 45% to
88% of it, which is what says that probe carries real information and is not
broken. Against this MEDIAN the same two figures are 47% and 92%, a mean over a
median, which is not a ratio. `Readout.mean_error` is read against the mean and
`Readout.median_error` against this constant, and any ratio says which
statistic sits on each side.

ONLY THE MEDIAN IS PINNED. No verified value for the mean exists in this
module, so there is no constant for it: `zero_information_error` recomputes it
beside the median, from the cell's own splits, and it is recorded rather than
trusted."""


@dataclass(frozen=True, eq=False)
class Readout:
    """One estimator scored on one feature set, in both units.

    `eq=False`: the generated `__eq__` compares fields as a tuple, and the
    `ndarray` in `predicted` makes that comparison ambiguous (`a == b` raised
    `ValueError`) while the generated `__hash__` of a frozen dataclass hashes the
    same tuple (`hash(a)` raised `TypeError`). A readout is therefore compared
    and hashed by IDENTITY, as any plain object is: `a == a` is true, and two
    separately scored readouts are never equal, whatever they hold.

    `predicted` is stored BY REFERENCE, not copied and not made read-only,
    because it is the caller's array. A caller that mutates it afterwards
    invalidates the `r2`, `mean_error` and `median_error` stored beside it, which
    were computed from the values it held when this was built;
    `difference_interval` recomputes `r2` from it and refuses a readout whose two
    no longer agree.
    """

    name: str
    r2: float
    mean_error: float
    """MEAN Euclidean position error in map units, over the scored rows.

    Carried beside r2 because r2 is not the unit the decision is made in. M3n
    was unreadable because a 0.576-11.824 signal sat under a 131.88-259.06
    readout error, and a result in r2 alone cannot be set against that.

    THE MEAN IS THE ONE THAT COMPARES TO 131.88-259.06. `rollout.py:237`
    averages the per-window position error across windows, so that figure is a
    mean; a median reported against it would be a different statistic in the
    same unit."""
    median_error: float
    """MEDIAN of the same errors, carried beside the mean.

    Robust to the tail a readout produces when it extrapolates off its support,
    and it is the statistic `ZERO_INFORMATION_MEDIAN` is expressed in. Every
    ratio must state which statistic is on each side: 131.88-259.06 against
    283.05 is a mean over a median and is not one."""
    capacity: float
    """The selected ridge, bandwidth or k, by `_CAPACITY_KEY`.

    For the RBF this is the BANDWIDTH ONLY, and it is `inf` when the ladder's
    linear rung won. The fit is `(capacity, jitter)`; see `jitter`."""
    jitter: float | None
    """The selected diagonal jitter, the second half of an RBF cell's fit.

    `fit_kernel_probe` selects the kernel, its bandwidth and the jitter JOINTLY,
    and `capacity` keeps only the bandwidth, so two RBF cells with one bandwidth
    and different jitters are different fits that `capacity` alone cannot tell
    apart. This is what makes a record able to name the fit it scored.

    Per readout: the RBF's selected jitter; for both linear readouts their
    selected ridge, which is the same quantity (a jitter on the Gram's diagonal
    IS the dual form of the ridge at that lambda, so the ladder's linear rung
    at jitter L is `fit_linear_probe` at ridge L, and a record can set the two
    side by side); and `None` for the k-NN, which has no ridge to select."""
    predicted: np.ndarray
    """The scored-row predictions, kept so a difference between two readouts
    can be bootstrapped with both held fixed. Held by reference; see the class
    docstring."""


_FITTERS = {
    "linear": fit_linear_probe,      # C: the best linear readout, CEILING_RIDGES
    "linear_shipped": fit_probe,     # C_shipped: the study's own probe, RIDGES
    "rbf": fit_kernel_probe,
    "knn": fit_knn_probe,
}
"""Two linear entries, and the difference is the whole point of Task 2b.

`"linear"` searches `CEILING_RIDGES` and is what `D - C` is read against, so the
difference isolates nonlinearity rather than ridge-grid resolution.
`"linear_shipped"` is `fit_probe` at its default and is the ONLY one that
reconciles against the study's `embedding_selection_r2` -- it is what exit 49
reads. Gating on `"linear"` would exit 49 on a correct run, because the finer
grid is expected to read up to ~0.02 ABOVE the shipped band."""

_CAPACITY_KEY = {
    "linear": "ridge", "linear_shipped": "ridge",
    "rbf": "bandwidth", "knn": "k",
}

_JITTER_KEY = {
    "linear": "ridge", "linear_shipped": "ridge",
    "rbf": "jitter", "knn": None,
}
"""The readout-dict key behind `Readout.jitter`; `None` where there is none."""


def score_readout(name: str, fit, select, score) -> Readout:
    """Fit on `fit`, select capacity on `select`, score on `score`.

    Each of the three is a `(features, targets)` pair. Three splits, never two.
    A capacity selected on the fit split selects itself in, and that is the
    confound which makes a nonlinear probe untrustworthy as evidence rather
    than merely optimistic.
    """
    if name not in _FITTERS:
        raise KeyError(f"unknown readout {name!r}; expected {sorted(_FITTERS)}")
    readout = _FITTERS[name](fit[0], fit[1], select[0], select[1])
    predicted = apply_readout(readout, score[0])
    errors = position_error(predicted, score[1])
    jitter_key = _JITTER_KEY[name]
    return Readout(
        name=name,
        r2=_mean_r2(predicted, score[1]),
        mean_error=float(np.mean(errors)),
        median_error=float(np.median(errors)),
        capacity=float(readout[_CAPACITY_KEY[name]]),
        jitter=None if jitter_key is None else float(readout[jitter_key]),
        predicted=predicted,
    )


def zero_information_error(fit_targets, score_targets) -> dict:
    """`{"mean", "median"}` map-unit error of predicting the FIT split's mean
    position.

    The floor of informativeness. A readout that scores above this carries
    nothing; see `ZERO_INFORMATION_MEDIAN` for where the shipped rollout
    probe's floor error sits against it, and in which statistic.

    BOTH STATISTICS, from one set of errors, because the readout errors have
    both: `Readout.mean_error` is read against the mean and
    `Readout.median_error` against the median, and a mean over a median is not a
    ratio. Only the median has a pinned value (`ZERO_INFORMATION_MEDIAN`). The
    mean has none: it is recomputed here and recorded, never compared with a
    constant.
    """
    mu = np.asarray(fit_targets, dtype=np.float64)[:, :2].mean(0)
    truth = np.asarray(score_targets, dtype=np.float64)
    errors = position_error(np.broadcast_to(mu, (truth.shape[0], 2)), truth)
    return {"mean": float(np.mean(errors)), "median": float(np.median(errors))}


def difference_interval(a: Readout, b: Readout, score_targets, groups, *, seed) -> dict:
    """`a.r2 - b.r2` with an episode-clustered percentile interval.

    `probe._block_bootstrap_ci` already does exactly this: it takes two
    prediction arrays and the targets, resamples `groups` with both readouts
    held FIXED, and returns the percentile interval on the difference. Its
    parameter names say joint and embedding, but the computation is generic in
    the two arrays, so no second resampler is written here.

    THE THREE ARGUMENTS MUST DESCRIBE THE SAME ROWS, and nothing downstream can
    tell: `point` is `a.r2 - b.r2`, taken from whatever each readout was scored
    on, while the interval resamples `score_targets`. A longer `a.predicted`
    returned an interval over its first rows beside a point over all of them,
    and a same-length but wrong `score_targets` put the two on different
    quantities, both without an error. So, as `GainSplit.rows` does rather than
    trusting its caller, the row counts are compared, and each readout's `r2` is
    recomputed against `score_targets` and must match the one it carries. That
    is two r2 evaluations against the bootstrap's 2000 each.

    `groups` is then validated, by the check `gain_from_blocks` uses: with one
    distinct label every resample is the whole sample, and the interval would
    come back at width zero around the point -- a resolved-looking answer from
    a single resampling unit.
    """
    targets = np.asarray(score_targets, dtype=np.float64)
    rows = (a.predicted.shape[0], b.predicted.shape[0], targets.shape[0])
    if len(set(rows)) != 1:
        raise ValueError(
            "a.predicted, b.predicted and score_targets must describe the same "
            f"number of rows; got {rows[0]}, {rows[1]} and {rows[2]}"
        )
    for label, readout in (("a", a), ("b", b)):
        recomputed = _mean_r2(readout.predicted, targets)
        if not abs(recomputed - readout.r2) <= _R2_AGREEMENT:
            raise ValueError(
                f"{label}.r2 is {readout.r2:.6f} but {label}.predicted scores "
                f"{recomputed:.6f} against score_targets: the readout was not "
                "scored on these targets, so the point and the interval would "
                "describe different quantities"
            )
    labels, _ = _validate_groups(groups, targets.shape[0])
    low, high = _block_bootstrap_ci(
        a.predicted, b.predicted, targets,
        resamples=RESAMPLES, confidence=CONFIDENCE, seed=seed, groups=labels,
    )
    return {"point": a.r2 - b.r2, "ci_low": low, "ci_high": high}


VERDICTS: tuple[str, ...] = (
    "PROBE_LIMITED", "REPRESENTATION_LIMITED", "OBSERVATION_LIMITED",
)
NO_MAJORITY: str = "NO_MAJORITY"
SEEDS_MINIMUM: int = 3
ARMS_REQUIRED: int = 2

_INTERVAL_KEYS: tuple[str, ...] = ("point", "ci_low", "ci_high")


def _check_interval(where: str, interval: dict) -> None:
    """Refuse an interval that is not one: a non-finite figure, or bounds crossed.

    `nan > 0.0` is False, so a NaN lower bound would read as `does not resolve`
    and count toward OBSERVATION_LIMITED: a missing measurement answering as the
    verdict that says no readout can do better. The point is not required to sit
    inside the interval, since a percentile interval need not contain it.
    """
    for key in _INTERVAL_KEYS:
        if not math.isfinite(interval[key]):
            raise ValueError(
                f"{where}[{key!r}] is {interval[key]}; an interval needs finite figures"
            )
    if interval["ci_low"] > interval["ci_high"]:
        raise ValueError(
            f"{where} has ci_low {interval['ci_low']} above ci_high {interval['ci_high']}"
        )


def _resolvable(interval: dict) -> bool:
    """The whole interval lies above zero. `ci_low > 0.0`, strictly.

    An interval whose lower bound is exactly 0.0 does not exclude zero, and the
    reading is a question about exclusion.
    """
    return interval["ci_low"] > 0.0


def probe_gain_over_floor(
    d: Readout, c: Readout, floor: float, score_targets, groups, *, seed
) -> dict:
    """`D - C - floor`, with its interval: the one place `probe_gain` is formed.

    `D` is the `"rbf"` readout and `C` is the `"linear"` one, the best LINEAR
    readout over `CEILING_RIDGES`; `"linear_shipped"` is refused as `C`. The
    shipped-grid readout is the reconciliation anchor and reads up to ~0.02 below
    `C`, so a gain taken against it is inflated by the ridge-grid resolution that
    `CEILING_RIDGES` exists to remove, and `PROBE_LIMITED` would fire on it. `D`
    and `C` are both `Readout`s and either can be passed as the other, which
    would return a gain of the wrong sign; the names are what tell them apart.

    THE FLOOR IS SUBTRACTED FROM THE POINT AND BOTH BOUNDS, so the verdict's
    question stays `is the whole interval above zero`, now asked of the gain over
    what the instrument alone produces on a purely linear task of this cell's
    shape (`null_floor`). A floor read against zero is the defect Task 2b exists
    to have fixed.

    A NEGATIVE FLOOR IS CLAMPED TO ZERO. The floor can only raise the bar: a null
    that reads below zero means the ladder under-read a linear task by selection
    noise, which is no credit to hand the real difference. Subtracting it as given
    would add up to |floor| to the lower bound and let an interval that includes
    zero read as `PROBE_LIMITED`; the measured floors run -0.0016 to +0.0000, so
    that is the usual sign and not an edge.

    THE FLOOR IS ONE SYNTHETIC DRAW PER CELL and is subtracted as a constant, so
    its own sampling noise is not in the interval. Over six draws it ranged
    -0.0005 to +0.0000 at D=1024 (`CEILING_RIDGES`); how that compares with the
    real intervals' widths is not yet measured. It is a point estimate, and the
    record keeps it beside the gain.

    `score_targets` must be the exact targets both readouts were scored on, and
    `groups` the episode label of each of those rows: `difference_interval`
    refuses a readout whose stored `r2` is not what its predictions score against
    them.
    """
    if d.name != "rbf":
        raise ValueError(f"D must be the 'rbf' readout; got {d.name!r}")
    if c.name != "linear":
        raise ValueError(
            f"C must be the 'linear' readout, the best linear one over "
            f"CEILING_RIDGES; got {c.name!r}"
        )
    if not math.isfinite(floor):
        raise ValueError(f"floor must be a finite number; got {floor}")
    interval = difference_interval(d, c, score_targets, groups, seed=seed)
    bar = max(floor, 0.0)
    return {key: interval[key] - bar for key in _INTERVAL_KEYS}


@dataclass(frozen=True)
class CeilingCell:
    """One arm at one seed: the three differences the verdict reads."""

    arm: str
    seed: int
    probe_gain: dict
    """`D - C - floor`: the nonlinear readout over the best linear one on the
    same features, net of this cell's own measured null floor. Formed by
    `probe_gain_over_floor`, never by subtracting `D.r2 - C.r2` by hand."""
    pixel_gain: dict
    """`B - D`: the best readout of raw pixels over the best of the
    representation. Formed against THIS cell's `D` from the one study-wide
    `B`, so the nine differences share a term and are correlated. The record
    says so; it does not make them invalid, since each asks about its own
    cell's representation."""
    sequence_gain: dict
    """`F - B`: a window of frames over one frame. A RIDER, not a verdict."""

    def __post_init__(self):
        for field in ("probe_gain", "pixel_gain", "sequence_gain"):
            _check_interval(f"CeilingCell({self.arm!r}, {self.seed}).{field}", getattr(self, field))

    def verdict(self) -> str:
        """`PROBE_LIMITED` takes precedence when both differences resolve.

        `probe_gain` here is already `D - C - floor`: the cell's own measured
        floor is subtracted before the interval is read, so `_resolvable` still
        asks whether the whole interval is above zero. The floor is what keeps
        this verdict from firing for a reason that has nothing to do with
        nonlinear structure: with both sides on the shipped 100x ridge grid the
        ladder beat the linear reference by +0.011 (D=1024) and +0.020 (D=2048)
        on a purely linear task, and on `CEILING_RIDGES` it reads -0.0001 and
        -0.0000. It is measured per cell because shapes not measured may differ.

        If the linear family was the limit, that is the finding regardless of
        the pixel comparison, because that comparison was made with the weaker
        estimator on one side and is not yet interpretable.

        `sequence_gain` is never read: it is a rider on the verdict.
        """
        if _resolvable(self.probe_gain):
            return "PROBE_LIMITED"
        if _resolvable(self.pixel_gain):
            return "REPRESENTATION_LIMITED"
        return "OBSERVATION_LIMITED"


@dataclass(frozen=True)
class CeilingInputs:
    cells: dict[tuple[str, int], CeilingCell]


@dataclass(frozen=True)
class CeilingStatus:
    verdict: str
    rule: str
    tally: dict[str, tuple[tuple[str, int], ...]]


def reading_ceiling(inputs: CeilingInputs) -> CeilingStatus:
    """Strict majority of the cells, bar computed from the cells present.

    THE BAR IS COMPUTED. `strict_majority(len(cells))` returns 5 at nine cells
    and 6 at eleven, so storing 5 would be a majority at one cell count and a
    minority at another.

    REFUSES an arm short of `SEEDS_MINIMUM`, fewer than `ARMS_REQUIRED` arms, and
    a cell filed under a key other than its own, which the tally and the minimum
    would count in the wrong arm without a word.

    `ARMS_REQUIRED` IS ENFORCED TWICE, because neither check covers the other:
    the bar counts cells and not arms, so arms of 3 and 9 are twelve cells with a
    bar of 7 that the nine-seed arm reaches alone. A verdict whose winning cells
    span fewer than `ARMS_REQUIRED` arms is NO_MAJORITY, not a raise: the inputs
    were legal and the arms did not agree. It counts the arms the winners come
    from, not how many cells each contributes, as `headroom.reading_headroom`
    does. At three seeds in each of three arms it cannot fire, since no arm holds
    the bar of 5 alone.
    """
    if not inputs.cells:
        raise ValueError("reading_ceiling needs at least one cell")
    for (arm, seed), cell in inputs.cells.items():
        if (cell.arm, cell.seed) != (arm, seed):
            raise ValueError(
                f"the cell for {(cell.arm, cell.seed)!r} is filed under "
                f"{(arm, seed)!r}; the tally reads the key, so it would be counted "
                "as the wrong cell"
            )
    by_arm: dict[str, set[int]] = {}
    for arm, seed in inputs.cells:
        by_arm.setdefault(arm, set()).add(seed)
    short = {a: sorted(s) for a, s in by_arm.items() if len(s) < SEEDS_MINIMUM}
    if short:
        raise ValueError(
            f"every arm needs SEEDS_MINIMUM={SEEDS_MINIMUM} seeds before a verdict "
            f"is tallied; got {short}"
        )
    if len(by_arm) < ARMS_REQUIRED:
        raise ValueError(
            f"a verdict needs ARMS_REQUIRED={ARMS_REQUIRED} arms; got {sorted(by_arm)}"
        )
    tally: dict[str, list[tuple[str, int]]] = {name: [] for name in VERDICTS}
    for key, cell in sorted(inputs.cells.items()):
        tally[cell.verdict()].append(key)
    frozen = {name: tuple(v) for name, v in tally.items()}
    needed = strict_majority(len(inputs.cells))
    winners = [n for n in VERDICTS if len(frozen[n]) >= needed]
    assert len(winners) <= 1, (
        f"two verdicts cannot both hold a strict majority of {len(inputs.cells)}; "
        f"got {winners} -- the partition is broken"
    )
    counts = ", ".join(f"{n}={len(frozen[n])}" for n in VERDICTS)
    if not winners:
        return CeilingStatus(
            verdict=NO_MAJORITY,
            rule=f"no verdict reached {needed} of {len(inputs.cells)} cells ({counts})",
            tally=frozen,
        )
    verdict = winners[0]
    winning_arms = sorted({arm for arm, _ in frozen[verdict]})
    if len(winning_arms) < ARMS_REQUIRED:
        return CeilingStatus(
            verdict=NO_MAJORITY,
            rule=(
                f"{verdict} held a numeric majority ({len(frozen[verdict])} of "
                f"{len(inputs.cells)} cells, strict majority {needed}) but spanned "
                f"only {len(winning_arms)} {'arm' if len(winning_arms) == 1 else 'arms'} "
                f"({', '.join(winning_arms)}); ARMS_REQUIRED={ARMS_REQUIRED} arms "
                f"must agree ({counts})"
            ),
            tally=frozen,
        )
    return CeilingStatus(
        verdict=verdict,
        rule=(
            f"{verdict} in {len(frozen[verdict])} of {len(inputs.cells)} cells "
            f"(strict majority {needed}, computed from the cells present; {counts})"
        ),
        tally=frozen,
    )


READING_COLUMNS: tuple[str, ...] = (
    "cell", "D-C-floor (probe)", "B-D (pixels)", "F-B (sequence)", "verdict",
)
READING_WIDTHS: tuple[int, ...] = (22, 30, 30, 30, 24)


def _row(values, widths) -> str:
    return "".join(str(v).ljust(w) for v, w in zip(values, widths, strict=True))


def _gap(iv: dict) -> str:
    return f"{iv['point']:+7.4f} [{iv['ci_low']:+7.4f},{iv['ci_high']:+7.4f}]"


def format_reading_ceiling(reading: CeilingStatus, inputs: CeilingInputs) -> str:
    """The table, the verdict, and a legend conditioned on what each test shows.

    EVERY DIRECTIONAL SENTENCE IS CONDITIONAL. The predecessor shipped an
    unconditional one in three places its own results refuted, and the
    milestone before that licensed "the map genuinely does not predict motion"
    from a test that only failed to reject.

    The verdict and rule lines are `reading`'s, as handed over; only the table
    and the rider count are read from `inputs`. Every figure here is a
    difference in r2, so no mean or median map-unit error appears, and none is
    set against a baseline.
    """
    lines = [
        "M3o probe ceiling -- what the readout can reach, independently of the model",
        "",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for (arm, seed), cell in sorted(inputs.cells.items()):
        lines.append(_row(
            (f"{arm}_seed{seed}", _gap(cell.probe_gain), _gap(cell.pixel_gain),
             _gap(cell.sequence_gain), cell.verdict()),
            READING_WIDTHS,
        ))
    riders = sum(1 for c in inputs.cells.values() if _resolvable(c.sequence_gain))
    lines += [
        "",
        f"verdict: {reading.verdict}",
        f"rule:    {reading.rule}",
        f"rider:   history resolves what one frame cannot in {riders} of "
        f"{len(inputs.cells)} cells (F - B resolvably above zero)",
        "",
        "legend",
        "  All three are differences in r2 (a fraction of variance), not in map units.",
        "  D-C-floor  the nonlinear readout over the best linear one, on the same",
        "       features, MINUS the cell's own measured null floor (D - C on a purely",
        "       linear task of this cell's shape, clamped at zero from below). The",
        "       interval is read against zero AFTER that subtraction. An RBF nests",
        "       the linear kernel, so D >= C is near-guaranteed and only a",
        "       RESOLVABLE difference is evidence of anything.",
        "  B-D  the best readout of raw pixels over the best of the representation.",
        "       B is one study-wide number, so the nine differences share a term",
        "       and are correlated; each still asks about its own representation.",
        "  F-B  a window of frames over one frame. A RIDER on the verdict, not a",
        "       verdict: it answers gate criterion 4's question without criterion",
        "       4's 'A beats B' framing.",
        "  A difference that does not resolve is the ruler failing to find one,",
        "  not a measurement that there is none; each interval's upper bound is",
        "  how much is NOT excluded.",
        "",
    ]
    return "\n".join(lines)
