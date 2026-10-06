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

AND WHY "THE RBF WINS" IS NOT A FINDING. An RBF approaches the linear kernel
as its bandwidth grows, so the linear probe is nearly a special case and the
held-out selection will pick something like it if nothing better exists. The
finding is the MAGNITUDE of the difference and whether it is resolvable --
the trap M3n walked into from the other side, where `skill` had a positive
point estimate in 8 of 9 cells and meant nothing until the interval was drawn.
"""

from __future__ import annotations

import numpy as np

from mbfps.eval.probe import _mean_r2, apply_probe, fit_probe

BANDWIDTHS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""RBF length scales, as multiples of the median pairwise distance.

Expressed as multiples rather than absolutes so the ladder means the same
thing at D = 2048 and D = 37632, where raw distances differ by orders of
magnitude. The largest rung is where the kernel is closest to linear."""

NEIGHBOURS: tuple[int, ...] = (1, 4, 16, 64)
"""k for the k-NN readout. 1 is the aliasing question at its sharpest: if the
single nearest frame in feature space is at a distant true position, the
feature space aliases."""


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


def fit_kernel_probe(x, y, val_x, val_y) -> dict:
    """Kernel ridge with an RBF kernel, bandwidth selected on held-out data.

    The same three-split discipline as `fit_probe`: fit here, select the
    bandwidth on `val_*`, and the caller scores on a third split. A bandwidth
    selected on the fit split selects itself in, which is exactly what makes a
    nonlinear probe untrustworthy as evidence.

    THE DIAGONAL JITTER IS FIXED AT 1e-3 AND NOT SELECTED, which bounds how
    well this nests the linear probe. The widest rung's kernel is linear to
    first order, with an effective linear ridge of ~1e-3 x (bandwidth x median
    distance)^2: ~130 at D = 1024 against the shipped probe's optimum of 1e3.
    On a noisy low-rank synthetic task (D = 1024, linear r2 0.46-0.78) that left this
    readout 0.05-0.06 BELOW the linear probe, and selecting the jitter on the
    held-out split alongside the bandwidth closed the gap to within 0.003. At
    a high signal-to-noise ratio the fixed jitter does not show, which is the
    regime `test_the_rbf_matches_the_linear_probe_on_a_linear_task` samples.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / scale
    xv = (np.asarray(val_x, dtype=np.float64) - mean) / scale
    unit = _median_scale(xs)
    centre = y.mean(0)
    best = None
    for multiple in BANDWIDTHS:
        gamma = 1.0 / (2.0 * (multiple * unit) ** 2)
        gram = _rbf(xs, xs, gamma)
        gram.flat[:: gram.shape[0] + 1] += 1e-3
        alpha = np.linalg.solve(gram, y - centre)
        score = _mean_r2(_rbf(xv, xs, gamma) @ alpha + centre, val_y)
        if best is None or score > best[0]:
            best = (score, multiple, gamma, alpha)
    return {
        "kind": "rbf", "mean": mean, "scale": scale, "anchors": xs,
        "alpha": best[3], "centre": centre, "gamma": best[2],
        "bandwidth": best[1], "r2": best[0],
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
        return _rbf(xs, readout["anchors"], readout["gamma"]) @ readout["alpha"] \
            + readout["centre"]
    return _knn_predict(readout["anchors"], readout["targets"], xs, readout["k"])
