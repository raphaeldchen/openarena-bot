import tracemalloc

import numpy as np
import pytest

import mbfps.eval.ceiling as ceiling
import mbfps.eval.probe as probe_module
from mbfps.eval.ceiling import (
    BANDWIDTHS, CEILING_RIDGES, NEIGHBOURS, _median_scale, apply_readout,
    fit_kernel_probe, fit_knn_probe, fit_linear_probe, null_floor,
)
from mbfps.eval.probe import RIDGES, _mean_r2, apply_probe, fit_probe


def _linear_task(seed=0, n=240, d=12):
    """A task a LINEAR probe solves well. The RBF must match it, not beat it.

    The weights come from a generator that does NOT depend on `seed`, so the
    fit, selection and scoring splits all sample ONE map. Drawn from `seed`,
    each split got its own and the linear probe, fit on one map and scored on
    another, read -0.016.
    """
    w = np.random.default_rng(12345).normal(size=(d, 4)) * 30.0
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    return x, x @ w + rng.normal(size=(n, 4)) * 0.5


def _embedding_task(seed=0, n=400, d=1024, rank=40, noise=120.0):
    """A noisy, low-rank, embedding-shaped task: a linear target the probe only
    partly recovers.

    The features are a 40-dimensional signal pushed through a fixed random
    map into D = 1024 dimensions, plus isotropic nuisance; the labels are a
    linear function of the signal plus label noise. At `noise` 120 the shipped
    probe reads r2 ~0.42-0.46 on held-out splits, which is where the measured
    embedding sits (~0.33) and, more to the point, past the ~0.37 above which a
    fixed-jitter RBF read BELOW it. `_linear_task` cannot show that: its signal
    std is ~104 against noise 0.5, so no jitter is visible there.

    As in `_linear_task`, the maps come from a generator that does not depend on
    `seed`, so every split samples one task.
    """
    g = np.random.default_rng(777)
    a = g.normal(size=(rank, d)); w = g.normal(size=(rank, 4))
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, rank))
    x = z @ a + rng.normal(size=(n, d)) * 3.0
    return x, (z @ w) * 20.0 + rng.normal(size=(n, 4)) * noise


def _curved_task(seed=0, n=240, d=3, noise=0.0, spread=1.0):
    """A task a linear probe CANNOT solve: the target is a radial function.

    This is the fixture that makes "the RBF recovers more" a measurement
    rather than an artefact of capacity. Its discrimination is asserted in the
    test before anything is concluded from it.

    `spread` scales the inputs and the labels are recomputed from the scaled
    inputs, so a `spread` of 0.4 is a held-out split of the SAME function
    drawn from its dense core, not a mislabelled one. `noise` is the std of
    Gaussian label noise; the defaults reproduce the noise-free full-range
    task.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * spread
    r = np.linalg.norm(x, axis=1, keepdims=True)
    y = np.concatenate([np.sin(2.0 * r), np.cos(2.0 * r), r ** 2, -r], axis=1) * 40.0
    if noise:
        y = y + rng.normal(size=y.shape) * noise
    return x, y


def _score_on(readout, split):
    return _mean_r2(apply_readout(readout, split[0]), split[1])


def _score_at_one_rung(monkeypatch, ladder, rung, fit, fit_split, select_split, score_split):
    """Fit with `ladder` collapsed to the single `rung`, score on `score_split`.

    Scored here through `apply_readout`, not read from the readout's own
    recorded `r2`, so an expectation built from it does not come from the code
    that chose the rung.
    """
    with monkeypatch.context() as patch:
        patch.setattr(ceiling, ladder, (rung,))
        readout = fit(*fit_split, *select_split)
    return _score_on(readout, score_split)


def _brute_force_ladder(fit, select):
    """Every rung of the kernel ladder, as kernel ridge written from its definition.

    One entry per (kernel, bandwidth, jitter): the linear kernel and an RBF at
    each of `BANDWIDTHS`, each at every jitter in `CEILING_RIDGES`, fit by
    `np.linalg.solve` on `gram + jitter * I` with centred targets, inputs
    standardised by the fit split. The bandwidth is a multiple of the median
    pairwise distance of the standardised fit rows. Nothing here is shared with
    the estimator but the constants it is meant to use, so it cannot agree with
    it by sharing a mistake.

    Each entry carries its `select` r2 and a `predict(x)`. Small fixtures only:
    the pairwise differences are n x n x D.
    """
    fit_x, fit_y = fit
    mean, scale = fit_x.mean(axis=0), fit_x.std(axis=0) + 1e-8
    anchors = (fit_x - mean) / scale
    centre = fit_y.mean(axis=0)
    pairwise = np.sqrt(((anchors[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=-1))
    unit = np.median(pairwise[np.triu_indices(len(anchors), k=1)])

    def rbf(sigma):
        return lambda a, b: np.exp(
            -((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1) / (2.0 * sigma ** 2)
        )

    rungs = [("linear", np.inf, lambda a, b: a @ b.T)]
    rungs += [("rbf", m, rbf(m * unit)) for m in BANDWIDTHS]
    entries = []
    for kernel, bandwidth, gram in rungs:
        k = gram(anchors, anchors)
        for jitter in CEILING_RIDGES:
            alpha = np.linalg.solve(k + jitter * np.eye(len(anchors)), fit_y - centre)

            def predict(x, gram=gram, alpha=alpha):
                return gram((x - mean) / scale, anchors) @ alpha + centre

            entries.append({
                "kernel": kernel, "bandwidth": bandwidth, "jitter": jitter,
                "select": _mean_r2(predict(select[0]), select[1]), "predict": predict,
            })
    return entries


def test_the_rbf_recovers_a_curved_target_the_linear_probe_cannot(monkeypatch):
    """The fixture is shown to defeat the linear probe FIRST.

    Without that assertion the comparison could pass on a task both solve, and
    would then be measuring nothing.

    It is also shown to defeat the near-linear END of the ladder. An RBF at
    8x the median distance is not the linear probe: its second-order term is
    quadratic, so it recovers about half of this radial target (r^2 and -r
    are both functions of |x|). A bar of "beats linear by 0.2" therefore
    admits a readout stuck on the widest rung, and the bar is set between what
    that rung reaches and what selection reaches.
    """
    train, select, score = _curved_task(0), _curved_task(1), _curved_task(2)
    linear = _score_on(fit_linear_probe(*train, *select), score)
    assert linear < 0.5, f"fixture must defeat the linear probe; it scored {linear:.3f}"

    widest = _score_at_one_rung(
        monkeypatch, "BANDWIDTHS", BANDWIDTHS[-1], fit_kernel_probe, train, select, score
    )
    assert widest < 0.8, (
        f"fixture must defeat the nearly-linear end of the ladder; rung "
        f"{BANDWIDTHS[-1]} scored {widest:.3f}"
    )

    rbf = _score_on(fit_kernel_probe(*train, *select), score)
    assert rbf > linear + 0.2
    assert rbf > 0.8, f"selection should find a rung the widest one cannot; got {rbf:.3f}"


def test_the_rbf_matches_the_linear_probe_on_a_linear_task():
    """Nesting on a task the linear probe solves: the kernel readout must not be worse.

    This fixture is high signal-to-noise, so it cannot show the failure that
    needed the linear kernel in the ladder -- see
    `test_the_kernel_readout_is_never_worse_than_the_linear_probe` for that.
    What it does show is that the readout is not merely agreeing with the linear
    probe because both reach the ceiling: an RBF confined to the narrowest rung
    is shown to score far below the linear probe on the same split, so a pass
    here is the ladder finding the linear rung.

    The narrowest-rung control is the brute-force oracle's, not the estimator's
    with its ladder collapsed: the linear kernel is a rung of every ladder, so
    collapsing `BANDWIDTHS` can no longer produce an RBF-only readout.
    """
    train, select, score = _linear_task(0), _linear_task(1), _linear_task(2)
    linear = _score_on(fit_linear_probe(*train, *select), score)
    assert linear > 0.9, f"fixture must be solvable linearly; linear scored {linear:.3f}"

    narrow = [e for e in _brute_force_ladder(train, select)
              if e["kernel"] == "rbf" and e["bandwidth"] == BANDWIDTHS[0]]
    narrowest = _mean_r2(max(narrow, key=lambda e: e["select"])["predict"](score[0]), score[1])
    assert narrowest < linear - 0.5, (
        f"an RBF stuck on rung {BANDWIDTHS[0]} must read worse than linear here "
        f"for the nesting comparison to be able to fail; it scored {narrowest:.3f}"
    )

    rbf = _score_on(fit_kernel_probe(*train, *select), score)
    assert rbf > linear - 0.05, f"the kernel readout nests the linear probe; {rbf:.3f} vs {linear:.3f}"


def test_the_knn_recovers_a_curved_target_and_reports_its_k():
    train, select, score = _curved_task(0), _curved_task(1), _curved_task(2)
    readout = fit_knn_probe(*train, *select)
    assert readout["k"] in NEIGHBOURS
    assert _score_on(readout, score) > 0.5


def _brute_force_knn(anchors, targets, query, k):
    """The mean target of each query's k nearest anchors, one query at a time."""
    out = np.empty((len(query), targets.shape[1]))
    for i, q in enumerate(query):
        nearest = np.argsort(((anchors - q) ** 2).sum(axis=1))[:k]
        out[i] = targets[nearest].mean(axis=0)
    return out


@pytest.mark.parametrize("k", NEIGHBOURS)
def test_the_knn_prediction_is_the_mean_of_the_k_nearest_targets(k):
    """Compared with a brute-force oracle at EVERY rung, `k` imposed on the readout.

    Not through selection: on a noise-free fixture selection picks k = 1, where
    "the mean of the k nearest" and "the first of them" are the same thing, and
    a predictor that ignores `k` reads exactly as well. Setting `k` directly
    puts the averaging, the one thing that distinguishes the rungs, under test.
    """
    fit_x, fit_y = _curved_task(0, noise=20.0)
    query_x, _ = _curved_task(2)
    readout = dict(fit_knn_probe(fit_x, fit_y, *_curved_task(1)), k=k)

    mean, scale = fit_x.mean(axis=0), fit_x.std(axis=0) + 1e-8
    expected = _brute_force_knn(
        (fit_x - mean) / scale, fit_y, (query_x - mean) / scale, k
    )
    np.testing.assert_allclose(apply_readout(readout, query_x), expected, rtol=1e-12)


@pytest.mark.parametrize("name, expected_kernel, fit, select, query", [
    ("curved", "rbf", _curved_task(0, n=60), _curved_task(1, n=60), _curved_task(2, n=40)),
    ("linear", "linear", _linear_task(0), _linear_task(1), _linear_task(2)),
])
def test_the_kernel_readout_is_the_best_kernel_ridge_over_the_whole_ladder(
    name, expected_kernel, fit, select, query
):
    """The selected rung and its predictions, against kernel ridge written out.

    Pins what the value tests cannot see: that `bandwidth` is a MULTIPLE of the
    median pairwise distance of the standardised fit rows, that `gamma` is
    1 / (2 sigma^2), that the targets are centred, that the jitter is added to
    the diagonal and is one of `CEILING_RIDGES`, that the linear rung is the plain Gram
    of the standardised rows, and that the eigendecomposed solve is the same
    answer as `np.linalg.solve`.

    The readout must be the argmax of the oracle over ALL 231 rungs on the
    selection split, so a ladder that skips a rung, selects the worst, or
    selects on the fit split fails here. Both kernels are exercised through
    `apply_readout`: the expected kernel is asserted, so neither case can pass
    by having selected the other.
    """
    readout = fit_kernel_probe(*fit, *select)
    assert readout["kernel"] == expected_kernel, f"{name}: selected {readout['kernel']}"

    entries = _brute_force_ladder(fit, select)
    assert len(entries) == (1 + len(BANDWIDTHS)) * len(CEILING_RIDGES)
    best = max(entries, key=lambda e: e["select"])
    assert (readout["kernel"], readout["bandwidth"], readout["jitter"]) == (
        best["kernel"], best["bandwidth"], best["jitter"]
    )
    assert readout["r2"] == pytest.approx(best["select"], abs=1e-8)
    np.testing.assert_allclose(
        apply_readout(readout, query[0]), best["predict"](query[0]), rtol=1e-6, atol=1e-6
    )


def test_apply_readout_refuses_a_kernel_it_does_not_know():
    readout = dict(
        fit_kernel_probe(*_curved_task(0, n=40), *_curved_task(1, n=40)), kernel="wavelet"
    )
    with pytest.raises(KeyError, match="unknown kernel"):
        apply_readout(readout, _curved_task(2, n=5)[0])


def test_the_kernel_probe_selects_its_bandwidth_on_the_split_it_is_handed(monkeypatch):
    """A capacity chosen on the FIT split selects itself in.

    The expected rung is the one with the best score on the selection split
    among the declared ladder, found by fitting each rung alone and scoring it
    through `apply_readout`. Two selection splits are used, and they are first
    shown to disagree: both are held-out draws of the same function, one
    across its full range and one from its dense core, which wants a narrower
    kernel. Only then is it meaningful that the choice follows the split.
    """
    train = _curved_task(0)
    splits = {"full range": _curved_task(1), "dense core": _curved_task(1, spread=0.4)}
    expected = {
        name: max(
            BANDWIDTHS,
            key=lambda rung: _score_at_one_rung(
                monkeypatch, "BANDWIDTHS", rung, fit_kernel_probe, train, split, split
            ),
        )
        for name, split in splits.items()
    }
    assert expected["full range"] != expected["dense core"], (
        f"the two selection splits must want different rungs; both want {expected}"
    )

    for name, split in splits.items():
        chosen = fit_kernel_probe(*train, *split)["bandwidth"]
        assert chosen in BANDWIDTHS
        assert chosen == expected[name], f"{name}: chose {chosen}, the held-out best is {expected[name]}"


def test_the_knn_selects_its_k_on_the_split_it_is_handed(monkeypatch):
    """The same discipline for k, with the same two-step structure.

    The fit labels are noisy, so a wider neighbourhood is worth having; a
    noisy full-range selection split wants k = 4 and a clean dense-core one,
    where the target is smooth, wants a wider neighbourhood. The fit split
    itself would want k = 1, the one rung that reproduces its own rows.
    """
    train = _curved_task(0, noise=20.0)
    splits = {
        "noisy full range": _curved_task(1, noise=20.0),
        "clean dense core": _curved_task(1, spread=0.5),
    }
    expected = {
        name: max(
            NEIGHBOURS,
            key=lambda rung: _score_at_one_rung(
                monkeypatch, "NEIGHBOURS", rung, fit_knn_probe, train, split, split
            ),
        )
        for name, split in splits.items()
    }
    assert expected["noisy full range"] != expected["clean dense core"], (
        f"the two selection splits must want different rungs; both want {expected}"
    )
    assert 1 not in expected.values(), "k = 1 is what selecting on the fit split would pick"

    for name, split in splits.items():
        chosen = fit_knn_probe(*train, *split)["k"]
        assert chosen in NEIGHBOURS
        assert chosen == expected[name], f"{name}: chose k={chosen}, the held-out best is k={expected[name]}"


def test_the_knn_skips_rungs_larger_than_the_fit_set():
    fit_x, fit_y = _curved_task(0, n=3)
    readout = fit_knn_probe(fit_x, fit_y, *_curved_task(1, n=5))
    assert readout["k"] == 1


def test_apply_readout_sends_a_linear_probe_to_the_shipped_path_unchanged():
    """`apply_probe` is the shipped function and must not be reimplemented.

    Compared bitwise, because any divergence would silently change thirteen
    milestones' worth of recorded numbers.
    """
    xf, yf = _linear_task(0); xs, _ = _linear_task(2)
    probe = fit_probe(xf, yf, *_linear_task(1))
    np.testing.assert_array_equal(apply_readout(probe, xs), apply_probe(probe, xs))


def test_apply_readout_calls_apply_probe_rather_than_repeating_it(monkeypatch):
    """The bitwise test cannot tell delegation from a verbatim copy.

    A copy of `apply_probe`'s three lines returns the same bits, so equality
    holds whether the shipped function is called or reimplemented. What
    separates them is whether it is CALLED: here it is replaced by a spy that
    returns a value no reimplementation could produce.
    """
    calls = []
    sentinel = np.full((7, 4), 7.0)

    def spy(probe, latents):
        calls.append((probe, latents))
        return sentinel

    monkeypatch.setattr(probe_module, "apply_probe", spy)
    monkeypatch.setattr(ceiling, "apply_probe", spy, raising=False)
    xf, yf = _linear_task(0); xs, _ = _linear_task(2)
    probe = fit_probe(xf, yf, *_linear_task(1))
    named = dict(probe, kind="linear")

    assert apply_readout(probe, xs) is sentinel
    assert apply_readout(named, xs) is sentinel
    assert len(calls) == 2
    assert calls[0][0] is probe and calls[0][1] is xs
    assert calls[1][0] is named and calls[1][1] is xs


def test_apply_readout_refuses_a_readout_whose_kind_it_does_not_know():
    """The refusal must name the kind.

    An unknown readout has no `mean`, so standardising the input before the
    kind is checked would fail on that lookup with `KeyError: 'mean'` and
    never reach the message that says what was wrong.
    """
    with pytest.raises(KeyError, match="unknown readout kind"):
        apply_readout({"kind": "wavelet"}, np.zeros((3, 3)))


def test_the_median_scale_is_the_median_pairwise_distance():
    """Stable on the fixtures, and on the capped-subsample path.

    The first two are well under the 512-row cap and are exact; the third has
    1500 rows, so it takes the 512-row subsample, and must land on the full-set
    median to within 3%. It lands at 1.5%: the pairs share rows, so the
    effective sample is nearer 512 than the 260,000 pairs it contains.
    """
    def exact(z):
        rows = []
        for start in range(0, len(z), 100):
            diff = z[start:start + 100, None, :] - z[None, :, :]
            rows.append(np.sqrt((diff ** 2).sum(axis=-1)))
        pairwise = np.concatenate(rows)
        return float(np.median(pairwise[~np.eye(len(z), dtype=bool)]))

    for task in (_linear_task(0), _curved_task(0)):
        z = (task[0] - task[0].mean(axis=0)) / task[0].std(axis=0)
        assert _median_scale(z) == pytest.approx(exact(z), rel=1e-12)

    z = np.random.default_rng(3).normal(size=(1500, 8))
    assert 512 < len(z)
    assert _median_scale(z) == pytest.approx(exact(z), rel=0.03)


def test_the_median_scale_does_not_build_the_pairwise_difference_tensor():
    """The scale is a length, so it needs the n x n distances and no more.

    The difference `x[:, None] - x[None]` is n x n x D: at the 512-row cap and
    D = 2048 that is 4.3 GB before `norm` allocates a second one, and at the
    raw-pixel D = 37632 it is 79 GB. It is also the obvious one-line form, and
    a test on the value cannot see it. Measured through `tracemalloc`, which
    sees numpy's allocations, at 200 rows of 1000 features, where the tensor
    is 305 MiB:

        this code               1.8 MiB (the input is 1.5)
        the difference tensor   611 MiB (the tensor and the square `norm` takes)

    The budget is a sixteenth of the tensor. The peak must also be ABOVE the
    input's own size, which is what proves `tracemalloc` is seeing numpy.
    """
    x = np.random.default_rng(5).normal(size=(200, 1000))
    tensor_bytes = 200 * 200 * 1000 * 8

    tracemalloc.start()
    try:
        _median_scale(x)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak > x.nbytes, (
        f"peak {peak / 2**20:.2f} MiB is not above the input's own "
        f"{x.nbytes / 2**20:.2f} MiB: `tracemalloc` is not seeing numpy's allocations"
    )
    assert peak < tensor_bytes // 16, (
        f"peak {peak / 2**20:.1f} MiB; the difference tensor is {tensor_bytes / 2**20:.0f} MiB"
    )


def test_a_set_of_identical_feature_rows_has_no_scale_and_says_so():
    """Raised rather than papered over: a zero scale is a zero bandwidth.

    Constant columns standardise to the same row everywhere, and all-zero rows
    are the same row by construction. Identical rows with a large norm are the
    case a Gram-identity distance can get wrong -- see the next test -- and
    must still be refused whichever way its rounding falls.
    """
    rng = np.random.default_rng(11)
    targets = rng.normal(size=(30, 4))
    constant = np.full((30, 6), 3.7)
    with pytest.raises(ValueError, match="identical"):
        fit_kernel_probe(constant, targets, constant, targets)

    with pytest.raises(ValueError, match="identical"):
        _median_scale(np.zeros((20, 5)))

    with pytest.raises(ValueError, match="identical"):
        _median_scale(np.tile(rng.normal(size=(1, 50)) * 1e3, (40, 1)))


def test_a_scale_that_is_only_rounding_noise_is_not_accepted_as_one(monkeypatch):
    """The Gram identity cannot resolve a distance below ~1e-8 x the row norm.

    For rows that coincide it returns rounding noise rather than 0, and "> 0"
    would take that for a length scale. Whether the noise comes out positive
    depends on the BLAS in use, so it is injected rather than hoped for: the
    squared distances are replaced by 1e-6, which is a distance of 1e-3
    against rows of norm ~7e3, a ratio of 1.4e-7. The unpatched call on the
    same rows is the control that the rows themselves are fine.
    """
    x = np.random.default_rng(13).normal(size=(40, 50)) * 1e3
    assert _median_scale(x) > 1e3, "the rows must have a real scale before it is faked away"

    monkeypatch.setattr(ceiling, "_sqdist", lambda a, b: np.full((len(a), len(b)), 1e-6))
    with pytest.raises(ValueError, match="identical"):
        _median_scale(x)


@pytest.mark.parametrize("seed", (0, 1))
def test_the_kernel_readout_is_never_worse_than_the_linear_probe(seed):
    """Nesting, asserted directly, on a task where the linear probe is NOT at ceiling.

    The comparison is against `fit_linear_probe`, the linear readout over the
    SAME `CEILING_RIDGES` the ladder searches, not against `fit_probe` over the
    shipped grid: the ladder's search space contains the former's.

    The fixed-jitter RBF this replaced read below the linear probe wherever
    linear r2 passed ~0.37: -0.0635 and -0.0512 on these two fixtures (linear
    0.4552 / 0.4637, RBF 0.3918 / 0.4125). M3o's PROBE_LIMITED verdict fires on
    D - C resolvably above zero, so an instrument that reads D - C < 0 on a
    linear-ish task makes that verdict unreachable. With the linear kernel in
    the ladder both read +0.0000: nothing nonlinear helps, so selection lands on
    the linear rung and the readout IS the linear probe.

    The linear score is asserted to be in the regime where the old code failed
    first, so a pass is not the two readouts agreeing on an easy task.

    THE TOLERANCE IS 1e-6, AND IT HOLDS ONLY WHERE THE LINEAR RUNG IS SELECTED.
    On the score split `D - C >= 0` is not guaranteed: an RBF rung that edges
    linear on the SELECTION split by ~1e-4 can lose ~1e-3 on the score split.
    Over 12 draws of this fixture the linear rung was selected in 8 and D - C
    was exactly 0 in those; the worst of the rest was -0.0009, against -0.046 to
    -0.064 before. That residual is selection noise, which a test cannot
    separate from a fault and which M3o's interval is for. What IS exact is the
    selection split, and the next two tests pin that. These two seeds are
    where the linear rung wins, so this is the "ladder finds linear" case, not
    a claim about every draw.
    """
    train, select, score = (_embedding_task(3 * seed + i) for i in range(3))
    linear = _score_on(fit_linear_probe(*train, *select), score)
    assert linear > 0.37, f"fixture must be where the fixed jitter failed; linear scored {linear:.3f}"

    kernel = _score_on(fit_kernel_probe(*train, *select), score)
    assert kernel >= linear - 1e-6, f"D - C = {kernel - linear:+.4f} ({kernel:.4f} vs linear {linear:.4f})"


@pytest.mark.parametrize("seed", range(6))
def test_the_kernel_readout_cannot_select_below_the_linear_probe_on_its_selection_split(seed):
    """The exact form of nesting: the ladder's search space contains `fit_linear_probe`'s.

    Both readouts select on the same split over the same `CEILING_RIDGES`, so the
    kernel readout's recorded selection r2 is at least the linear reference's, with
    no allowance beyond rounding (1e-9). Unlike the score-split comparison this
    holds for EVERY draw, including those where an RBF rung wins.
    """
    train, select, _ = (_embedding_task(3 * seed + i) for i in range(3))
    linear = fit_linear_probe(*train, *select)
    kernel = fit_kernel_probe(*train, *select)
    assert kernel["r2"] >= linear["r2"] - 1e-9, (
        f"selected {kernel['kernel']} at {kernel['r2']:.6f}, linear is {linear['r2']:.6f}"
    )


def test_the_linear_rung_is_the_linear_reference_and_not_an_approximation_of_it(monkeypatch):
    """Collapse the RBF ladder to a rung that cannot win; what is left must be `fit_linear_probe`.

    The grid is `CEILING_RIDGES` on both sides, which is the grid the claim is
    about: against `fit_probe` over the shipped grid the ridge match below would
    hold here only by coincidence, because the optimum on this fixture lands on
    a shipped rung (1000), and would fail on `_gapped_task`, where it is 177.8.
    With only the narrowest bandwidth beside it the linear kernel is selected,
    at the ridge `fit_linear_probe` selects, and the predictions are that
    probe's. The two solve in different spaces (the dual, through an
    eigendecomposition, against `probe._solve_ladder`), so agreement to 1e-6
    is the claim that a linear kernel with jitter lambda IS ridge regression at
    lambda, which the nesting argument rests on.
    """
    train, select, score = (_embedding_task(i) for i in range(3))
    probe = fit_linear_probe(*train, *select)
    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "BANDWIDTHS", (BANDWIDTHS[0],))
        readout = fit_kernel_probe(*train, *select)

    assert readout["kernel"] == "linear"
    assert readout["bandwidth"] == float("inf") and readout["gamma"] is None
    assert readout["jitter"] == probe["ridge"]
    np.testing.assert_allclose(
        apply_readout(readout, score[0]), apply_readout(probe, score[0]), rtol=1e-6, atol=1e-6
    )


def _gapped_task(seed=0, n=300, d=150):
    """A linear task whose best ridge falls BETWEEN the shipped grid's rungs.

    `_embedding_task` at the shape where the grid's 100x step costs the most. The
    optimum is 177.8; `probe.RIDGES` offers 1e1 and 1e3 and selects 1e3 in every
    draw, for a scored r2 0.09-0.11 below what the fine grid reaches.
    `_embedding_task` at its own default shape cannot show this -- at n=400
    D=1024 the ridge response is flat enough that the shipped grid loses nothing.
    """
    return _embedding_task(seed, n=n, d=d)


def _null_floor_over(monkeypatch, ridges, n, d, *, seed):
    """`null_floor` with the grid both sides search replaced by `ridges`.

    The grid is swapped for the call only, so a second call outside this helper
    reads the real `CEILING_RIDGES`. `null_floor` reaches the grid through
    `fit_linear_probe` AND through `fit_kernel_probe`'s jitter loop, and both
    read the module global at call time, so one swap moves both sides together.
    Calling `null_floor` itself, rather than repeating its body, means this
    cannot drift from the function it is validating.
    """
    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "CEILING_RIDGES", ridges)
        return null_floor(n, d, seed=seed)


def test_the_ceiling_grid_contains_the_shipped_grid_and_is_finer():
    """Containment by VALUE, the rung count and the step, pinned separately.

    Containment alone does not pin the step: a 17-rung 10^0.5 grid contains
    `RIDGES` too, and it left a +0.0009 floor at n=2000, D=2048 (seed 0) where
    33 rungs left -0.0000. So the count is asserted as well.
    """
    assert len(CEILING_RIDGES) == 33
    assert set(RIDGES) <= set(CEILING_RIDGES)
    assert CEILING_RIDGES[::8] == RIDGES, "the shipped rungs are every 8th fine rung"
    assert CEILING_RIDGES[0] == RIDGES[0] and CEILING_RIDGES[-1] == RIDGES[-1]
    assert all(a < b for a, b in zip(CEILING_RIDGES, CEILING_RIDGES[1:]))
    np.testing.assert_allclose(np.diff(np.log10(CEILING_RIDGES)), 0.25, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_the_shipped_grid_straddles_the_best_ridge_and_the_fine_grid_resolves_it(seed):
    """The whole reason `C` does not use `probe.RIDGES`.

    The fixture is shown to discriminate first: the shipped grid must select an
    ENDPOINT of the interval the optimum lies in, so there is something to
    resolve. Then the fine grid must land strictly inside that interval and read
    materially higher on a third split.
    """
    train, select, score = (_gapped_task(3 * seed + i) for i in range(3))
    shipped = fit_probe(*train, *select)
    assert shipped["ridge"] == 1e3, f"fixture must straddle; shipped picked {shipped['ridge']:g}"

    fine = fit_linear_probe(*train, *select)
    assert 1e1 < fine["ridge"] < 1e3, f"fine grid picked {fine['ridge']:g}, not interior"
    gap = _score_on(fine, score) - _score_on(shipped, score)
    assert gap > 0.05, f"fine grid bought only {gap:+.4f}"


@pytest.mark.parametrize("seed", range(3))
def test_the_null_floor_is_near_zero_on_the_grid_the_ceiling_searches(monkeypatch, seed):
    """`D - C` where there is nothing nonlinear to find.

    The fixture is shown to discriminate by measuring the SAME quantity over the
    shipped grid, which must come back far above zero -- otherwise a test that
    only checks `null_floor` is near zero would pass on a coarse grid too.
    """
    coarse = _null_floor_over(monkeypatch, RIDGES, 300, 150, seed=seed)
    assert coarse > 0.05, f"fixture must show a coarse-grid floor; got {coarse:+.4f}"
    floor = null_floor(300, 150, seed=seed)
    assert abs(floor) < 0.005, f"floor is {floor:+.4f} on CEILING_RIDGES"
