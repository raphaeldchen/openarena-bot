import contextlib
import dataclasses
import math
import re
import tracemalloc

import numpy as np
import pytest

import mbfps.eval.ceiling as ceiling
import mbfps.eval.probe as probe_module
from mbfps.eval.ceiling import (
    ARMS_REQUIRED, BANDWIDTHS, CEILING_RIDGES, CONFIDENCE, FRAME_BUDGET, NEIGHBOURS,
    NO_MAJORITY, NULL_DRAWS, READING_COLUMNS, READING_WIDTHS, RESAMPLES, SEEDS_MINIMUM,
    VERDICTS, ZERO_INFORMATION_MEDIAN, CeilingCell, CeilingInputs, CeilingStatus,
    Readout, _gap, _median_scale, _row, apply_readout, difference_interval,
    fit_kernel_probe, fit_knn_probe, fit_linear_probe, format_reading_ceiling,
    null_floor, probe_gain_over_floor, reading_ceiling, score_readout,
    zero_information_error, capacity_for_record,
)
from mbfps.eval.probe import RIDGES, _block_bootstrap_ci, _mean_r2, apply_probe, fit_probe


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


def _gapped_task(seed=0, n=300, d=150):
    """A linear task whose best ridge falls BETWEEN the shipped grid's rungs.

    `_embedding_task` at the shape where the grid's 100x step costs the most. The
    optimum is 177.8; `probe.RIDGES` offers 1e1 and 1e3 and selects 1e3 in every
    draw, for a scored r2 0.09-0.11 below what the fine grid reaches.
    `_embedding_task` at its own default shape cannot show this -- at n=400
    D=1024 the ridge response is flat enough that the shipped grid loses nothing.
    """
    return _embedding_task(seed, n=n, d=d)


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


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("poison", ["an inf in the selection features", "a nan in the selection targets"])
def test_the_kernel_probe_refuses_a_non_finite_selection_score_rather_than_pinning_the_first_rung(
    poison,
):
    """`nan > x` is False, so a NaN score on the FIRST rung is never beaten by a later
    one: the ladder returned its first rung (the linear kernel at the smallest jitter)
    with `r2 = nan`, and `score_readout` recomputes r2 on the score split, so the
    pinned readout then scored a plausible number and every finite check downstream
    passed. The failure direction is an understated `D`, which reads as
    `OBSERVATION_LIMITED` where `PROBE_LIMITED` holds.

    The fixture is shown to discriminate before anything is asserted: on the clean
    selection split the ladder selects an RBF rung, so pinning the first rung (linear)
    would be a visibly different readout, and applying that readout to the poisoned
    split scores non-finite, so the poison does reach the selection score.

    THE MUTATION THIS EXISTS FOR, run against it: delete the guard in
    `fit_kernel_probe`'s jitter loop."""
    train, select = _curved_task(0), _curved_task(1)
    clean = fit_kernel_probe(*train, *select)
    assert math.isfinite(clean["r2"])
    assert clean["kernel"] == "rbf", "pinning the first rung (the linear kernel) must be a visible change"

    val_x, val_y = select[0].copy(), select[1].copy()
    if "features" in poison:
        val_x[3, 1] = np.inf
    else:
        val_y[3, 1] = np.nan
    assert not math.isfinite(_mean_r2(apply_readout(clean, val_x), val_y)), "the poison must reach the score"

    with pytest.raises(ValueError, match="non-finite selection score") as caught:
        fit_kernel_probe(*train, val_x, val_y)
    message = str(caught.value)
    assert "fit_kernel_probe" in message and "'linear'" in message
    assert f"{CEILING_RIDGES[0]:g}" in message, "the refusal names the rung it stopped at"


def test_the_knn_probe_refuses_a_non_finite_selection_score_rather_than_pinning_the_first_rung():
    """The same defect in the k ladder: a NaN score at k = 1 is never beaten, so the
    readout is pinned to k = 1, the one rung that reproduces the fit rows it is
    handed. An inf in the selection FEATURES does not reach this estimator's score
    (its NaN distances only reorder a poisoned row's neighbours, and the targets
    stay finite), so the poison is a NaN in the selection TARGETS, which poisons
    every rung's r2 alike.

    The fixture is shown to discriminate: on the clean noisy split the ladder selects
    a k other than 1, so a pinned readout would be a visibly different one.

    THE MUTATION THIS EXISTS FOR, run against it: delete the guard in
    `fit_knn_probe`'s loop."""
    train, select = _curved_task(0, noise=20.0), _curved_task(1, noise=20.0)
    clean = fit_knn_probe(*train, *select)
    assert math.isfinite(clean["r2"])
    assert clean["k"] != NEIGHBOURS[0], "pinning the first rung (k = 1) must be a visible change"

    val_y = select[1].copy()
    val_y[3, 1] = np.nan
    assert not math.isfinite(_mean_r2(apply_readout(clean, select[0]), val_y)), "the poison must reach the score"

    with pytest.raises(ValueError, match="non-finite selection score") as caught:
        fit_knn_probe(*train, select[0], val_y)
    message = str(caught.value)
    assert "fit_knn_probe" in message and f"k={NEIGHBOURS[0]}" in message


def test_a_capacity_is_recorded_as_a_float_or_the_string_inf_and_nothing_else():
    """The one spelling of a linear rung's infinite bandwidth, for every writer of a
    record. Shown on a genuine linear-rung readout first: the ladder on a purely
    linear task selects the linear kernel, whose bandwidth is `inf`, so the string
    is what a real readout reaches and not only a literal."""
    readout = fit_kernel_probe(*_linear_task(0), *_linear_task(1))
    assert readout["kernel"] == "linear" and math.isinf(readout["bandwidth"])
    assert capacity_for_record(readout["bandwidth"]) == "inf"
    assert capacity_for_record(np.float64(0.25)) == 0.25
    assert capacity_for_record(16) == 16.0 and isinstance(capacity_for_record(16), float)
    for bad in (float("nan"), float("-inf")):
        with pytest.raises(ValueError, match="capacity"):
            capacity_for_record(bad)


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
@pytest.mark.parametrize("task, off_grid", [(_embedding_task, False), (_gapped_task, True)],
                         ids=["embedding", "gapped"])
def test_the_kernel_readout_cannot_select_below_the_linear_probe_on_its_selection_split(
    task, off_grid, seed
):
    """The exact form of nesting: the ladder's search space contains `fit_linear_probe`'s.

    Both readouts select on the same split over the same `CEILING_RIDGES`, so the
    kernel readout's recorded selection r2 is at least the linear reference's, with
    no allowance beyond rounding (1e-9). Unlike the score-split comparison this
    holds for EVERY draw, including those where an RBF rung wins.

    TWO FIXTURES, because only one of them can see the grid. On `_embedding_task`
    the optimum lands on a shipped rung (1000 in all 6 draws), so a ladder whose
    jitters are the shipped `RIDGES` passes there: it contains the reference's
    selection anyway. On `_gapped_task` the optimum is 177.8, between shipped
    rungs, and that ladder's linear rung stops at 1e3. The RBF rungs then edge
    it up toward the reference but not past it: measured, the ladder selected
    0.0006-0.0035 below the reference on 4 of 6 draws. The `gapped` case is
    asserted to be off the shipped grid, and the `embedding` case on it, before
    anything is concluded from either: the two parametrisations cannot collapse
    into one.
    """
    train, select, _ = (task(3 * seed + i) for i in range(3))
    linear = fit_linear_probe(*train, *select)
    assert (linear["ridge"] not in RIDGES) == off_grid, (
        f"fixture must put the optimum {'off' if off_grid else 'on'} the "
        f"shipped grid; got {linear['ridge']:g}"
    )
    kernel = fit_kernel_probe(*train, *select)
    assert kernel["r2"] >= linear["r2"] - 1e-9, (
        f"selected {kernel['kernel']} at {kernel['r2']:.6f}, linear is {linear['r2']:.6f}"
    )


@pytest.mark.parametrize("task, off_grid", [(_embedding_task, False), (_gapped_task, True)],
                         ids=["embedding", "gapped"])
def test_the_linear_rung_is_the_linear_reference_and_not_an_approximation_of_it(
    monkeypatch, task, off_grid
):
    """Collapse the RBF ladder to a rung that cannot win; what is left must
    be `fit_linear_probe`.

    The grid is `CEILING_RIDGES` on both sides, which is the grid the claim is
    about. On `_embedding_task` the optimum lands on a shipped rung (1000), so
    the ridge match below would hold there against `fit_probe` over the shipped
    grid too, and against a ladder whose jitters are `RIDGES`, by coincidence.
    On `_gapped_task` it is 177.8: a ladder whose linear rung stops at 1e3 is
    selected at 1e3 there, 0.10 below the reference, and fails the ridge match.
    The `off_grid` case is asserted to be off the shipped grid first.
    With only the narrowest bandwidth beside it the linear kernel is selected,
    at the ridge `fit_linear_probe` selects, and the predictions are that
    probe's. The two solve in different spaces (the dual, through an
    eigendecomposition, against `probe._solve_ladder`), so agreement to 1e-6
    is the claim that a linear kernel with jitter lambda IS ridge regression at
    lambda, which the nesting argument rests on.
    """
    train, select, score = (task(i) for i in range(3))
    probe = fit_linear_probe(*train, *select)
    assert (probe["ridge"] not in RIDGES) == off_grid, (
        f"fixture must put the optimum {'off' if off_grid else 'on'} the shipped grid; "
        f"got {probe['ridge']:g}"
    )
    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "BANDWIDTHS", (BANDWIDTHS[0],))
        readout = fit_kernel_probe(*train, *select)

    assert readout["kernel"] == "linear"
    assert readout["bandwidth"] == float("inf") and readout["gamma"] is None
    assert readout["jitter"] == probe["ridge"]
    np.testing.assert_allclose(
        apply_readout(readout, score[0]), apply_readout(probe, score[0]), rtol=1e-6, atol=1e-6
    )


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


@contextlib.contextmanager
def _spied_null_splits(monkeypatch, score_scales=None):
    """Record every split `null_floor` draws, and hand it exactly what was recorded.

    Yields a list of `(split_seed, x, y)` in call order, three splits to a draw
    (fit, select, score), so an expectation built from the record is built from
    the splits `null_floor` actually used. `score_scales[j]`, when given,
    multiplies the features of draw `j`'s third split. The spy is removed on
    exit, so two calls in one test do not wrap each other.
    """
    real = ceiling._null_split
    drawn = []

    def spy(split_seed, rows, dims):
        x, y = real(split_seed, rows, dims)
        k = len(drawn)
        if score_scales is not None and k % 3 == 2:
            x = x * score_scales[k // 3]
        drawn.append((split_seed, x, y))
        return x, y

    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "_null_split", spy)
        yield drawn


@pytest.mark.parametrize("seed", [4, 14])
def test_the_null_floor_is_the_mean_of_d_minus_c_scored_on_the_third_split_of_each_draw(
    monkeypatch, seed
):
    """Fit on a draw's first split, select on its second, SCORE on its third; the
    floor is the mean of the draws' scores.

    On the selection split `D >= C` is exact, because the ladder contains the
    reference and both selected on that split, so a floor read there is
    structurally non-negative and silently optimistic. Read on the fit split
    it is in-sample. `null_floor`'s result is only a floor if it is read on the
    split neither readout saw, and on a null task the selection-split and
    score-split readings differ by ~2e-3, which a threshold on the result
    cannot separate from the right one.

    So the splits are made distinguishable by construction. Each draw's third
    split is moved out of the readouts' support (its features scaled, by a
    DIFFERENT factor per draw), where a kernel readout reverts to the mean and
    the linear one extrapolates, so `D - C` reads in the units there instead of
    at 1e-3, and the three draws read far apart. That is no longer a null task
    and the number is not claimed to be a floor: the claim is only which split
    each draw is read on, and that the draws are averaged. The spy returns what
    `_null_split` draws, so the expectation is built from the splits used.

    The fixture is shown to discriminate before the result is compared. In every
    draw the selected rung must be an RBF one (a linear rung makes `D - C`
    exactly 0 on every split and nothing could be told apart; at this shape that
    holds for one draw in three on average, and seeds 4 and 14 are the two of the
    first sixty where it holds for all three), the selection-split reading must
    be non-negative as nesting
    requires, and the score-split reading must be separated from the fit-split
    and selection-split ones by far more than the tolerance. The three draws'
    readings must differ from one another and the mean from every one of them
    and from their median, so a floor taken from one draw, or the median of
    three, or a sum, cannot match. Reached: scoring on the selection split,
    scoring on the fit split, the third split taking the selection role, a
    single draw returned, a sum in place of a mean.
    """
    n, d = 200, 60
    with _spied_null_splits(monkeypatch, score_scales=(4.0, 6.0, 5.0)) as drawn:
        floor = null_floor(n, d, seed=seed)
    assert len(drawn) == 3 * NULL_DRAWS, f"expected {NULL_DRAWS} draws of three splits, got {len(drawn)}"

    on_scores = []
    for draw in range(NULL_DRAWS):
        train, select, score = [(x, y) for _, x, y in drawn[3 * draw:3 * draw + 3]]
        linear = fit_linear_probe(*train, *select)
        kernel = fit_kernel_probe(*train, *select)
        assert kernel["kernel"] == "rbf", (
            f"fixture must select an RBF rung in draw {draw}; got {kernel['kernel']}"
        )

        def gap_on(split):
            return _score_on(kernel, split) - _score_on(linear, split)

        on_train, on_select, on_score = gap_on(train), gap_on(select), gap_on(score)
        assert on_select >= -1e-9, (
            f"draw {draw}: nesting, D - C on the selection split is {on_select:+.6f}"
        )
        assert on_score - on_select > 1.0, (
            f"draw {draw}: score-split reading {on_score:+.4f} is not apart from the "
            f"selection-split one {on_select:+.4f}"
        )
        assert on_score - on_train > 1.0, (
            f"draw {draw}: score-split reading {on_score:+.4f} is not apart from the "
            f"fit-split one {on_train:+.4f}"
        )
        on_scores.append(on_score)

    mean = sum(on_scores) / NULL_DRAWS
    assert all(abs(a - b) > 0.5 for a, b in zip(on_scores, on_scores[1:] + on_scores[:1])), (
        f"fixture: the draws read {on_scores}, and they must differ from one another"
    )
    assert all(abs(mean - v) > 0.1 for v in on_scores), "fixture: the mean must match no draw"
    assert abs(mean - sorted(on_scores)[NULL_DRAWS // 2]) > 0.1, "fixture: mean is not the median"

    assert floor == pytest.approx(mean, abs=1e-9), (
        f"null_floor returned {floor:+.6f}; the mean of D - C on each draw's third "
        f"split is {mean:+.6f} (draws {on_scores})"
    )


def test_no_two_null_draws_share_a_split_and_no_two_seeds_share_one(monkeypatch):
    """Each draw is a task of its own: its three splits are drawn from seeds no
    other draw uses, and a different `seed` argument gets a block of its own.

    Averaging draws that share splits is not averaging: the old derivation,
    `_null_split(3 * seed + i, ...)` for each draw, would give three copies of
    one draw and a mean no more precise than a single one. Overlap by one split
    (`3 * seed + draw + i`) is the quieter version of the same error, and
    this checks that too, since it makes draw `j`'s select split draw `j + 1`'s
    fit split.

    Reached: every draw on the same three splits, draws offset by one, and
    neighbouring seeds whose blocks of splits overlap.
    """
    with _spied_null_splits(monkeypatch) as at_zero:
        null_floor(60, 20, seed=0)
    with _spied_null_splits(monkeypatch) as at_one:
        null_floor(60, 20, seed=1)
    zero = [s for s, _, _ in at_zero]
    one = [s for s, _, _ in at_one]
    assert len(zero) == len(one) == 3 * NULL_DRAWS, "fixture: three splits to a draw"
    assert len(set(zero)) == len(zero), f"a split is reused within one call: {zero}"
    assert len(set(one)) == len(one), f"a split is reused within one call: {one}"
    assert not set(zero) & set(one), f"seeds 0 and 1 share splits: {sorted(set(zero) & set(one))}"
    draws = [zero[3 * j:3 * j + 3] for j in range(NULL_DRAWS)]
    for a in range(NULL_DRAWS):
        for b in range(a + 1, NULL_DRAWS):
            assert not set(draws[a]) & set(draws[b]), f"draws {a} and {b} share a split: {draws}"
    # and the data differ, not only the labels
    xs = [x for _, x, _ in at_zero]
    assert all(
        not np.array_equal(xs[a], xs[b]) for a in range(len(xs)) for b in range(a + 1, len(xs))
    )


@pytest.mark.parametrize("gaps, mean, bar", [
    pytest.param((-0.004, -0.002, 0.003), -0.001, 0.0, id="negative-mean"),
    pytest.param((-0.002, -0.001, 0.003), 0.0, 0.0, id="the-spec-example"),
    pytest.param((0.004, -0.001, -0.001), 0.002 / 3, 0.002 / 3, id="positive-mean"),
])
def test_the_floor_averages_its_draws_before_it_clamps_and_returns_the_mean_unclamped(
    monkeypatch, gaps, mean, bar
):
    """`max(mean(draws), 0)` is not `mean(max(draw, 0))`, and the first is the bar.

    The draws are SCRIPTED, so the orders can be told apart by value: `null_floor`
    is run with each readout's score replaced by a number the test chooses, the
    linear one at 0 and the kernel one at the draw's gap. Draws of -0.004, -0.002
    and +0.003 average to -0.001, which clamps to a bar of 0.0; clamped first they
    average to +0.001. The second row is the spec's own example, and the third has
    a positive mean the per-draw clamp would inflate.

    Two things are pinned, each by a value written out here and not computed by
    the code under test: `null_floor` returns the UNCLAMPED mean (so it is
    negative where the mean is, and clamps nothing), and the gain built from it
    carries the bar `max(mean, 0)` and not the clamp-first one.

    Reached: each draw clamped before the mean, the mean clamped inside
    `null_floor`, one draw returned, the sum not divided.
    """
    clamp_first = sum(max(g, 0.0) for g in gaps) / len(gaps)
    assert abs(clamp_first - bar) > 5e-4, "fixture: clamping first must give a different bar"
    assert len(gaps) == NULL_DRAWS
    assert sum(gaps) / len(gaps) == pytest.approx(mean, abs=1e-12), "fixture: the stated mean"

    seen = {"kernel": 0, "linear": 0}

    def scripted_r2(tag, targets):
        draw = seen[tag]
        seen[tag] += 1
        return gaps[draw] if tag == "kernel" else 0.0

    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "fit_kernel_probe", lambda *a: {"tag": "kernel"})
        patch.setattr(ceiling, "fit_linear_probe", lambda *a: {"tag": "linear"})
        patch.setattr(ceiling, "apply_readout", lambda readout, x: readout["tag"])
        patch.setattr(ceiling, "_mean_r2", scripted_r2)
        floor = null_floor(10, 4)
    assert seen == {"kernel": len(gaps), "linear": len(gaps)}, "one scored pair per draw"
    assert floor == pytest.approx(mean, abs=1e-12), (
        f"null_floor returned {floor:+.6f}; the unclamped mean of {gaps} is {mean:+.6f}"
    )

    d, c, truth, episode = _d_and_c()
    raw = difference_interval(d, c, truth, episode, seed=0)
    got = probe_gain_over_floor(d, c, floor, truth, episode, seed=0)
    for key in ("point", "ci_low", "ci_high"):
        assert got[key] == pytest.approx(raw[key] - bar, abs=1e-12), key
        assert abs(got[key] - (raw[key] - clamp_first)) > 5e-4, f"{key}: clamp-first bar"


def test_an_empty_ridge_grid_is_refused_by_both_probes(monkeypatch):
    """`D` and `C` both read `CEILING_RIDGES`, so both must name it if empty.

    Unchecked, each ends in `TypeError: 'NoneType' object is not subscriptable`
    on its `best`, which says nothing about the grid. A one-rung grid on the
    same splits is the control: the data fit, and it is the emptiness that
    raises. `fit_linear_probe` reaches the check in `fit_probe` and
    `fit_kernel_probe` has its own.
    """
    train, select, _ = (_linear_task(i) for i in range(3))
    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "CEILING_RIDGES", (1e-1,))
        assert fit_linear_probe(*train, *select)["ridge"] == 1e-1
        assert fit_kernel_probe(*train, *select)["jitter"] == 1e-1

    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "CEILING_RIDGES", ())
        with pytest.raises(ValueError, match="ridges.*empty"):
            fit_linear_probe(*train, *select)
        with pytest.raises(ValueError, match="CEILING_RIDGES.*empty"):
            fit_kernel_probe(*train, *select)


def test_a_readout_reports_its_error_in_map_units_not_only_r2():
    """r2 is not the unit the decision is made in.

    M3n's readout error was 131.88-259.06 MAP UNITS against a signal of
    0.576-11.824, and that comparison is what made it unreadable. A result in
    r2 alone cannot be set against it.

    The expected figures are built here, from the shipped probe functions and
    numpy, never from the `Readout`. The three splits are first shown to give
    different answers, and the median to differ from the mean and from a norm
    over all four target columns, so that scoring on the wrong split, swapping
    the two statistics, and including the angle columns each move the number by
    far more than the tolerance. Reached by the oracle: r2, the mean and the
    median taken on the selection split instead of the score split, `np.mean`
    for `np.median` and the reverse, and a four-column norm.
    """
    train, select, score = (_gapped_task(i) for i in range(3))
    probe = fit_linear_probe(*train, *select)
    predicted = apply_probe(probe, score[0])
    errors = np.linalg.norm(predicted[:, :2] - score[1][:, :2], axis=1)
    expected_r2, expected_median = _mean_r2(predicted, score[1]), float(np.median(errors))
    expected_mean = float(errors.mean())

    on_select = apply_probe(probe, select[0])
    select_errors = np.linalg.norm(on_select[:, :2] - select[1][:, :2], axis=1)
    assert abs(_mean_r2(on_select, select[1]) - expected_r2) > 1e-4, "r2 must differ by split"
    assert abs(np.median(select_errors) - expected_median) > 1.0, "median must differ by split"
    assert abs(select_errors.mean() - expected_mean) > 1.0, "mean must differ by split"
    assert abs(expected_mean - expected_median) > 1.0, "fixture must separate mean from median"
    four_column = np.linalg.norm(predicted - score[1], axis=1)
    assert abs(np.median(four_column) - expected_median) > 1.0, "must separate position from all four"
    assert abs(four_column.mean() - expected_mean) > 1.0, "must separate position from all four"

    out = score_readout("linear", train, select, score)
    assert out.name == "linear" and 0.0 < out.r2 <= 1.0
    assert out.r2 == pytest.approx(expected_r2, abs=1e-9)
    assert out.mean_error == pytest.approx(expected_mean, abs=1e-6)
    assert out.median_error == pytest.approx(expected_median, abs=1e-6)
    assert out.capacity == probe["ridge"]
    assert out.predicted.shape == score[1].shape
    np.testing.assert_allclose(out.predicted, predicted, rtol=1e-9, atol=1e-9)


def test_the_median_error_is_a_median_of_position_errors_known_by_construction(monkeypatch):
    """The scoring arithmetic with the fit taken out of it.

    The fitter and `apply_readout` are replaced, so the predictions are whatever
    this test writes down and the expected median is read off them by hand: the
    position errors are the 3-4-5, 6-8-10, 0, 5-12-13 and 8-15-17 right triangles,
    in that unsorted order, so the answers are exact: median 10, mean 9, max 17,
    and the unsorted middle row 0. The two angle columns carry an error of 1000,
    so a norm over more than the first two columns lands nowhere near.

    Reached: `np.mean` for `np.median` and the reverse, a norm over every
    column, and indexing the middle row without sorting. Exact equality, not
    approx: every value here is exactly representable and a tolerance would only
    widen what survives.
    """
    truth = np.arange(20.0).reshape(5, 4) ** 2
    miss = np.array([[3.0, 4.0], [6.0, 8.0], [0.0, 0.0], [5.0, 12.0], [8.0, 15.0]])
    known = truth + np.concatenate([miss, np.full((5, 2), 1000.0)], axis=1)
    assert np.linalg.norm(miss, axis=1).tolist() == [5.0, 10.0, 0.0, 13.0, 17.0]

    monkeypatch.setitem(ceiling._FITTERS, "linear", lambda fx, fy, vx, vy: {"ridge": 7.0})
    monkeypatch.setattr(ceiling, "apply_readout", lambda readout, x: known)
    out = score_readout("linear", (truth, truth), (truth, truth), (truth, truth))

    assert out.mean_error == 9.0
    assert out.median_error == 10.0
    assert out.capacity == 7.0
    assert out.predicted is known
    assert out.r2 == _mean_r2(known, truth)


def test_the_map_unit_metric_is_probes_one_definition_and_is_called_once(monkeypatch):
    """`position_error` is the codebase's single map-unit metric, and both
    functions here take their errors from it.

    `rollout`, `diagnostics` and `burden` import it, and `burden`'s docstring
    records that a second Euclidean distance is the drift that already destroyed
    a signal once. Equal numbers cannot tell an import from a verbatim copy, so
    `ceiling.position_error` is replaced by a spy that returns errors no norm of
    these arrays could produce (1, 2, 3, 4 and 90: mean 20, median 3), and the
    spy's figures must be the ones that come out.

    Called ONCE per call, so the mean and the median of a readout are statistics
    of one set of errors rather than of two computations that could drift apart.

    Reached: the metric spelled out locally in either function (the spy is never
    reached and the real distance, 1414.2, comes out), the import removed (the
    name is gone, and patching it raises), and one call per statistic.
    """
    calls = []
    errors = np.array([1.0, 2.0, 3.0, 4.0, 90.0])

    def spy(predicted, true):
        calls.append((predicted, true))
        return errors

    truth = np.arange(20.0).reshape(5, 4) ** 2
    known = truth + 1000.0
    monkeypatch.setattr(ceiling, "position_error", spy)
    monkeypatch.setitem(ceiling._FITTERS, "linear", lambda fx, fy, vx, vy: {"ridge": 7.0})
    monkeypatch.setattr(ceiling, "apply_readout", lambda readout, x: known)

    out = score_readout("linear", (truth, truth), (truth, truth), (truth, truth))
    assert (out.mean_error, out.median_error) == (20.0, 3.0)
    assert len(calls) == 1, f"one set of errors feeds both statistics; got {len(calls)} calls"
    assert calls[0][0] is known and calls[0][1] is truth

    calls.clear()
    fit_targets = truth[::-1] * 0.5
    assert zero_information_error(fit_targets, truth) == {"mean": 20.0, "median": 3.0}
    assert len(calls) == 1, f"one set of errors feeds both statistics; got {len(calls)} calls"
    np.testing.assert_array_equal(
        calls[0][0], np.tile(fit_targets[:, :2].mean(axis=0), (5, 1))
    )
    np.testing.assert_array_equal(calls[0][1], truth)


def _selection_case(name, monkeypatch):
    """One readout's three-split fixture, the rungs it chooses among, and a
    scorer that fits ONE rung alone on the fit split and scores it on a split.

    `scorer(split)(rung)` is that rung's score through `apply_readout` or
    `apply_probe`, which makes it an oracle that shares nothing with the
    estimator's own selection loop.
    """
    if name in ("linear", "linear_shipped"):
        train, select, score = (_gapped_task(i) for i in range(3))
        grid = CEILING_RIDGES if name == "linear" else RIDGES

        def scorer(split):
            return lambda ridge: _score_on(fit_probe(*train, *split, ridges=(ridge,)), split)

        return train, {"held out": select}, score, grid, scorer
    if name == "rbf":
        train, score = _curved_task(0), _curved_task(2)
        splits = {"full range": _curved_task(1), "dense core": _curved_task(1, spread=0.4)}
        ladder, fitter, rungs = "BANDWIDTHS", fit_kernel_probe, BANDWIDTHS
    else:
        train, score = _curved_task(0, noise=20.0), _curved_task(2)
        splits = {
            "noisy full range": _curved_task(1, noise=20.0),
            "clean dense core": _curved_task(1, spread=0.5),
        }
        ladder, fitter, rungs = "NEIGHBOURS", fit_knn_probe, NEIGHBOURS

    def scorer(split):
        return lambda rung: _score_at_one_rung(
            monkeypatch, ladder, rung, fitter, train, split, split
        )

    return train, splits, score, rungs, scorer


@pytest.mark.parametrize("name", ["linear", "linear_shipped", "rbf", "knn"])
def test_score_readout_selects_its_capacity_on_the_selection_split_not_the_fit_split(
    monkeypatch, name
):
    """The capacity `score_readout` records is the held-out best, per estimator.

    A capacity selected on the fit split selects itself in: the ridge goes to
    its smallest, the bandwidth to its narrowest, k to 1. The expected capacity
    is found by fitting each rung alone and scoring it on the selection split,
    and the capacity the FIT split would have picked is found the same way and
    shown to differ from it, so that a `score_readout` selecting on the fit split
    cannot pass. Reached: handing the fitter `fit` in place of `select`.

    Both linear entries run on `_gapped_task`, whose optimum is off the shipped
    grid, and the grid claim is asserted before the result: `"linear"` must pick
    a ridge `probe.RIDGES` does not offer and `"linear_shipped"` one it does.
    Reached: `_FITTERS["linear"]` bound to `fit_probe`, which would put `D - C`
    against the shipped grid again.

    The jitter beside the capacity: a linear readout's is its selected ridge,
    the same quantity (compared with the oracle's ridge, not with `capacity`),
    and the k-NN has none. The rbf's is `test_an_rbf_readout_records_...`.
    Reached: a linear readout's jitter read from the wrong key or left `None`,
    and a k-NN given one (k itself, here).
    """
    train, splits, score, rungs, scorer = _selection_case(name, monkeypatch)
    expected = {label: max(rungs, key=scorer(split)) for label, split in splits.items()}
    on_fit = max(rungs, key=scorer(train))
    assert len({*expected.values(), on_fit}) > 1, (
        f"fixture must make selection on the fit split pick another rung; "
        f"held-out best {expected}, fit-split best {on_fit}"
    )
    if name in ("linear", "linear_shipped"):
        assert all((c not in RIDGES) == (name == "linear") for c in expected.values()), (
            f"{name}: expected ridge {expected} must be {'off' if name == 'linear' else 'on'} "
            f"the shipped grid"
        )

    for label, split in splits.items():
        out = score_readout(name, train, split, score)
        assert out.name == name
        assert out.capacity == expected[label], (
            f"{label}: capacity {out.capacity}, the held-out best is {expected[label]}"
        )
        if name == "knn":
            assert out.jitter is None, "a k-NN has no ridge"
        elif name != "rbf":
            assert out.jitter == expected[label], (
                f"{label}: jitter {out.jitter}, the held-out best ridge is {expected[label]}"
            )


@pytest.mark.parametrize("name, fit, select, score, expected_kernel", [
    ("noisy curved", _curved_task(0, n=60, noise=60.0), _curved_task(1, n=60, noise=60.0),
     _curved_task(2, n=60), "rbf"),
    ("noisy linear", _embedding_task(0, n=150, d=60, rank=20),
     _embedding_task(1, n=150, d=60, rank=20), _embedding_task(2, n=100, d=60, rank=20),
     "linear"),
])
def test_an_rbf_readout_records_the_jitter_it_was_selected_with(
    name, fit, select, score, expected_kernel
):
    """`capacity` keeps only the bandwidth, and the fit is `(capacity, jitter)`.

    `fit_kernel_probe` selects the kernel, bandwidth and jitter jointly; the
    readout has to carry the jitter or a record cannot name the fit it scored.
    The expected pair is the argmax of `_brute_force_ladder` over all 231
    rungs on the selection split, kernel ridge written from its definition.

    The fixture is shown to need the assertion before it is made: the winning
    jitter is interior to `CEILING_RIDGES` (so a constant at either end is
    wrong), is worth 0.01 of selection r2 or more over both ends at its own
    rung (so it is selected, not incidental), and is not numerically the
    bandwidth (so the two keys cannot be confused). The noisy curved task wants
    a finite bandwidth and a jitter of ~5.6; the noisy linear one wants the
    linear rung, so `capacity` is `inf`, and a jitter of ~178.
    Reached: `jitter` read from the bandwidth's key, left `None`, or taken from
    either end of the grid.
    """
    entries = _brute_force_ladder(fit, select)
    best = max(entries, key=lambda e: e["select"])
    rung = [e for e in entries if (e["kernel"], e["bandwidth"]) == (best["kernel"], best["bandwidth"])]
    assert best["kernel"] == expected_kernel, f"{name}: the oracle selects {best['kernel']}"
    assert best["jitter"] not in (CEILING_RIDGES[0], CEILING_RIDGES[-1]), (
        f"{name}: the winning jitter {best['jitter']} must be interior to the grid"
    )
    assert best["select"] - rung[0]["select"] > 0.01, f"{name}: the smallest jitter must lose"
    assert best["select"] - rung[-1]["select"] > 0.01, f"{name}: the largest jitter must lose"
    assert best["jitter"] != best["bandwidth"], f"{name}: bandwidth and jitter must be told apart"

    out = score_readout("rbf", fit, select, score)
    assert (out.capacity, out.jitter) == (best["bandwidth"], best["jitter"])


def test_score_readout_refuses_a_readout_it_does_not_know():
    train = _linear_task(0)
    with pytest.raises(KeyError, match="unknown readout 'wavelet'"):
        score_readout("wavelet", train, _linear_task(1), _linear_task(2))


def test_the_zero_information_error_is_the_distance_to_the_fit_mean():
    """The scale every map-unit figure is read against, derived not asserted.

    Both statistics, because the readout errors have both: `mean_error` is read
    against the mean and `median_error` against the median, and a mean over a
    median is not a ratio. The expected values are computed here from the
    arrays, not taken from the function, so a change inside it cannot move both
    sides.

    The score split is shifted by (60, -40) so that its own mean sits far from
    the fit split's: unshifted, the two means differ by about 1 map unit in the
    median error and only a 1e-12 tolerance separates them. The alternatives
    are shown to differ from the expected values before they are compared with.
    Reached: the SCORE split's mean in place of the fit split's, either
    statistic taken for the other, and a norm over all four columns.
    """
    _, yf = _linear_task(0); _, ys = _linear_task(2)
    ys = ys + np.array([60.0, -40.0, 0.0, 0.0])
    errors = np.linalg.norm(ys[:, :2] - yf[:, :2].mean(0), axis=1)
    expected_mean, expected_median = float(errors.mean()), float(np.median(errors))

    own = np.linalg.norm(ys[:, :2] - ys[:, :2].mean(0), axis=1)
    assert abs(expected_median - np.median(own)) > 10.0, "must separate the two splits' means"
    assert abs(expected_mean - own.mean()) > 10.0, "must separate the two splits' means"
    assert abs(expected_mean - expected_median) > 10.0, "fixture must separate mean from median"
    four_column = np.linalg.norm(ys - yf.mean(0), axis=1)
    assert abs(expected_median - np.median(four_column)) > 10.0, "must separate position from all four"
    assert abs(expected_mean - four_column.mean()) > 10.0, "must separate position from all four"

    out = zero_information_error(yf, ys)
    assert set(out) == {"mean", "median"}
    assert out["median"] == pytest.approx(expected_median, abs=1e-12)
    assert out["mean"] == pytest.approx(expected_mean, abs=1e-12)


def test_the_stated_constants_are_the_ones_the_plan_records():
    """A change detector for five figures that are STATED, not derived.

    `FRAME_BUDGET` is the n^2 dual's memory ceiling, `ZERO_INFORMATION_MEDIAN`
    the measured scale the map-unit figures are read against, `CONFIDENCE` and
    `RESAMPLES` the interval the verdict's differences are drawn at, and
    `NULL_DRAWS` the number of draws the floor is averaged over (and so the
    multiple of one draw's cost that `null_floor` takes). None is computed by
    the code, so the only test of a value is that it has not moved.
    """
    assert NULL_DRAWS == 3
    assert FRAME_BUDGET == 12000
    assert ZERO_INFORMATION_MEDIAN == 283.05
    assert CONFIDENCE == 0.95
    assert RESAMPLES == 2000
    assert FRAME_BUDGET ** 2 * 8 < 1.2e9, "the dual's Gram at the budget is ~1.15 GB"


def _episode_pair(episodes=8, per_episode=30, bad=20.0, noise=3.0):
    """Two readouts of one target whose DISAGREEMENT lives in whole episodes.

    Both read the curved target with small independent noise. Readout `b` also
    carries one fixed offset per episode in the first half of the episodes, so
    its errors are shared by every row of an episode (a fixed `bad` per column,
    of random sign): the rows of an episode are one observation, not thirty.
    Built by hand so the r2 fields are the test's own `_mean_r2` and the
    structure is known by construction.
    """
    truth = _curved_task(2, n=episodes * per_episode)[1]
    rng = np.random.default_rng(0)
    episode = np.repeat(np.arange(episodes), per_episode)
    offset = rng.choice([-bad, bad], size=(episodes, truth.shape[1]))
    offset[episodes // 2:] = 0.0
    a = truth + rng.normal(size=truth.shape) * noise
    b = truth + rng.normal(size=truth.shape) * noise + offset[episode]

    def readout(name, predicted):
        return Readout(
            name=name, r2=_mean_r2(predicted, truth), mean_error=0.0,
            median_error=0.0, capacity=0.0, jitter=None, predicted=predicted,
        )

    return readout("a", a), readout("b", b), truth, episode


def test_a_difference_interval_is_clustered_on_the_groups_it_is_given():
    """Two cluster labellings on the same rows must give different widths.

    Episodes, never frames: adjacent Doom frames are near-duplicates, and this
    project measured a 1.26x-2.17x penalty for pretending otherwise.

    THE BRIEF'S FIXTURE COULD NOT SHOW THIS. It scored two real readouts on
    `_curved_task`, whose rows are independent draws, and labelled them in four
    blocks of 60: with no structure in the blocks, four-way resampling gave an
    interval NARROWER than row-level resampling on every seed (0.052 against
    0.082). Here the disagreement is an episode-level offset by construction, so
    resampling episodes must be far wider than resampling rows, and the fixture
    is shown to have that structure before the intervals are compared: the
    per-episode r2 of `a` over `b` is large in the offset episodes and
    negligible in the rest.

    Reached: `groups` ignored for row-level labels, and the point estimate
    taken as `b.r2 - a.r2`.
    """
    a, b, truth, episode = _episode_pair()
    per_episode = np.array([
        _mean_r2(a.predicted[episode == e], truth[episode == e])
        - _mean_r2(b.predicted[episode == e], truth[episode == e])
        for e in range(episode.max() + 1)
    ])
    assert per_episode[:4].min() > 0.3, f"offset episodes must favour a: {per_episode[:4]}"
    assert np.abs(per_episode[4:]).max() < 0.1, f"clean episodes must tie: {per_episode[4:]}"

    clustered = difference_interval(a, b, truth, episode, seed=0)
    by_row = difference_interval(a, b, truth, np.arange(len(truth)), seed=0)
    width = lambda ci: ci["ci_high"] - ci["ci_low"]
    assert 0.0 < width(by_row), "fixture must give a nonzero-width row-level interval"
    assert width(clustered) > 3.0 * width(by_row), (
        f"clustered {width(clustered):.3f} against row-level {width(by_row):.3f}"
    )
    assert clustered["point"] == a.r2 - b.r2 > 0.0
    assert clustered["ci_low"] < clustered["point"] < clustered["ci_high"]


def test_a_difference_of_two_scored_readouts_is_bracketed_by_its_interval():
    """`score_readout`'s output feeds `difference_interval` directly.

    On the curved task, where the RBF recovers what the linear probe cannot. The
    rows here are independent draws, so these labels carry no cluster structure
    and this does NOT test clustering -- the test above does. What it does show
    is that the `predicted` a `Readout` keeps is what the interval resamples:
    the interval brackets the point built from the same two readouts' r2.
    """
    train, select, score = _curved_task(0), _curved_task(1), _curved_task(2)
    rbf = score_readout("rbf", train, select, score)
    lin = score_readout("linear", train, select, score)
    assert rbf.r2 - lin.r2 > 0.5, "fixture must put the two readouts far apart"

    n = score[1].shape[0]
    interval = difference_interval(rbf, lin, score[1], np.arange(n) // 60, seed=0)
    assert interval["point"] == rbf.r2 - lin.r2
    assert interval["ci_low"] < interval["point"] < interval["ci_high"]


def test_the_difference_interval_is_the_shipped_bootstrap_at_the_recorded_level(monkeypatch):
    """Delegation pinned by value, not by reading the call.

    The expected interval comes from `probe._block_bootstrap_ci` called here with
    the numbers written out -- 2000 resamples at 0.95, `a` as the joint arm and
    `b` as the embedding arm -- and equality is exact, because the same generator
    seeded the same way draws the same blocks. A one-sided tolerance would let a
    changed resample count through.

    The constants are then moved and the interval must move with them: `CONFIDENCE`
    and `RESAMPLES` are parameters here, not labels, and a constant that no
    interval reads would be the worse defect. The two oracles are shown to differ
    first, or the second check could pass on a fixture the constants cannot move.
    Reached: a resample count other than 2000, the arms swapped (the interval
    would be `b - a`), and either constant defined but not passed.
    """
    a, b, truth, episode = _episode_pair()

    def oracle(resamples, confidence, seed):
        return _block_bootstrap_ci(
            a.predicted, b.predicted, truth,
            resamples=resamples, confidence=confidence, seed=seed, groups=episode,
        )

    interval = difference_interval(a, b, truth, episode, seed=3)
    assert (interval["ci_low"], interval["ci_high"]) == oracle(2000, 0.95, 3)

    assert oracle(500, 0.8, 3) != oracle(2000, 0.95, 3)
    with monkeypatch.context() as patch:
        patch.setattr(ceiling, "RESAMPLES", 500)
        patch.setattr(ceiling, "CONFIDENCE", 0.8)
        moved = difference_interval(a, b, truth, episode, seed=3)
    assert (moved["ci_low"], moved["ci_high"]) == oracle(500, 0.8, 3)


def test_the_difference_interval_follows_its_seed():
    """The seed is the caller's, so it must reach the resampler.

    Two seeds are first shown to give different intervals through the shipped
    bootstrap itself, so that a hard-coded seed could not hide behind two draws
    that happen to agree; then each seed's interval must be that bootstrap's at
    that seed. Reached: the seed fixed at 0 inside `difference_interval`.
    """
    a, b, truth, episode = _episode_pair()

    def oracle(seed):
        return _block_bootstrap_ci(
            a.predicted, b.predicted, truth,
            resamples=2000, confidence=0.95, seed=seed, groups=episode,
        )

    assert oracle(1) != oracle(2), "fixture must give a different interval per seed"
    for seed in (1, 2):
        interval = difference_interval(a, b, truth, episode, seed=seed)
        assert (interval["ci_low"], interval["ci_high"]) == oracle(seed)


def test_the_difference_interval_refuses_groups_it_cannot_resample():
    """One distinct label is one resampling unit, and every resample is the sample.

    The interval would come back at width zero around the point: a resolved-looking
    answer from nothing. `probe._validate_groups` refuses it, as it does for
    `gain_from_blocks`. A label count that does not match the rows is refused too.
    """
    a, b, truth, episode = _episode_pair()
    with pytest.raises(ValueError, match="at least two"):
        difference_interval(a, b, truth, np.zeros(len(truth), dtype=int), seed=0)
    with pytest.raises(ValueError, match="one label per scored row"):
        difference_interval(a, b, truth, episode[:-1], seed=0)
    assert difference_interval(a, b, truth, episode, seed=0)["ci_low"] < a.r2 - b.r2


@pytest.mark.parametrize("odd", ["a", "b"])
def test_the_difference_interval_refuses_predictions_over_another_number_of_rows(odd):
    """A longer `predicted` than the targets used to come back with an answer.

    The point is `a.r2 - b.r2` over every row each readout was scored on, and the
    interval resamples only the rows `score_targets` has, so a 300-row
    `a.predicted` against 240 targets gave a point over 300 rows beside an
    interval over the first 240, and no error.

    That is shown first, through the shipped bootstrap on the same arrays: it
    accepts the longer array and returns an interval, so it is the guard in
    `difference_interval` that refuses and not something downstream of it. The
    expected counts are the fixture's own, 300 and 240.
    Reached: the row-count check removed, for either readout.
    """
    a, b, truth, episode = _episode_pair()
    readouts = {"a": a, "b": b}
    longer = readouts[odd].predicted
    readouts[odd] = dataclasses.replace(
        readouts[odd], predicted=np.concatenate([longer, longer[:60]])
    )
    assert readouts[odd].predicted.shape[0] == 300 and truth.shape[0] == 240

    low, high = _block_bootstrap_ci(
        readouts["a"].predicted, readouts["b"].predicted, truth,
        resamples=20, confidence=0.95, seed=0, groups=episode,
    )
    assert low < high, "the bootstrap alone must accept the longer array"

    counts = "300, 240 and 240" if odd == "a" else "240, 300 and 240"
    with pytest.raises(ValueError, match=re.escape(f"same number of rows; got {counts}")):
        difference_interval(readouts["a"], readouts["b"], truth, episode, seed=0)


def test_the_difference_interval_refuses_targets_over_another_number_of_rows():
    """The same refusal when it is the targets that are longer.

    `groups` is given one label per TARGET row, so the groups check cannot be
    what fires. The shipped bootstrap on the same arrays does not say what is
    wrong: it resamples rows the predictions do not have and fails with a bare
    `IndexError`.
    """
    a, b, truth, episode = _episode_pair()
    targets = np.concatenate([truth, truth[:60]])
    groups = np.concatenate([episode, episode[:60]])
    with pytest.raises(IndexError):
        _block_bootstrap_ci(
            a.predicted, b.predicted, targets,
            resamples=20, confidence=0.95, seed=0, groups=groups,
        )
    with pytest.raises(ValueError, match=re.escape("same number of rows; got 240, 240 and 300")):
        difference_interval(a, b, targets, groups, seed=0)


@pytest.mark.parametrize("case, culprit", [
    ("the targets are not the ones both were scored on", "a"),
    ("a was scored on other targets", "a"),
    ("b was scored on other targets", "b"),
])
def test_the_difference_interval_refuses_a_readout_not_scored_on_these_targets(case, culprit):
    """The same number of rows is not the same rows.

    A same-length `score_targets` that is not the one the readouts were scored
    on gave a point over one set of targets beside an interval over another, and
    no error. The readout's `r2` is recomputed from its own `predicted` against
    the targets handed in, and a readout that disagrees with it is refused,
    naming which. All three ways to get there are here: the targets are wrong
    for both readouts (`a` is reported, it is checked first), `a` was scored on
    other targets, and `b` was.

    Shown first, per case: the stored r2 is far (> 0.5) from the recomputation,
    so the refusal is not rounding, and the shipped bootstrap accepts exactly
    these arguments and returns an interval.
    Reached: the r2 recomputation removed, for either readout.
    """
    a, b, truth, episode = _episode_pair()
    other = _curved_task(5, n=truth.shape[0])[1]
    assert other.shape == truth.shape and not np.array_equal(other, truth)
    if case.startswith("the targets"):
        first, second, targets = a, b, other
    else:
        stray = a if culprit == "a" else b
        moved = dataclasses.replace(stray, r2=_mean_r2(stray.predicted, other))
        first, second, targets = (moved, b, truth) if culprit == "a" else (a, moved, truth)

    claimed = {"a": first, "b": second}[culprit]
    assert abs(claimed.r2 - _mean_r2(claimed.predicted, targets)) > 0.5, "r2 must disagree widely"
    low, high = _block_bootstrap_ci(
        first.predicted, second.predicted, targets,
        resamples=20, confidence=0.95, seed=0, groups=episode,
    )
    assert low < high, "the bootstrap alone must accept these arguments"

    with pytest.raises(ValueError, match=re.escape(f"{culprit}.r2 is")) as caught:
        difference_interval(first, second, targets, episode, seed=0)
    assert "was not scored on these targets" in str(caught.value)


def test_the_difference_interval_tolerates_rounding_and_refuses_a_real_disagreement():
    """The r2 agreement is a tolerance, and both sides of it are pinned.

    A stored r2 off by 1e-9 is rounding (a caller that scored on a narrower
    dtype) and is accepted; off by 1e-3 it is a different number and is refused.
    Honest readouts come first, to show the call succeeds before it is nudged.
    Reached: a tolerance of exactly zero, and one wide enough to wave a 1e-3
    disagreement through.
    """
    a, b, truth, episode = _episode_pair()
    honest = difference_interval(a, b, truth, episode, seed=0)
    assert honest["point"] == a.r2 - b.r2

    rounded = dataclasses.replace(a, r2=a.r2 + 1e-9)
    assert difference_interval(rounded, b, truth, episode, seed=0)["point"] == rounded.r2 - b.r2
    with pytest.raises(ValueError, match=re.escape("a.r2 is")):
        difference_interval(dataclasses.replace(a, r2=a.r2 + 1e-3), b, truth, episode, seed=0)


def test_a_readouts_mean_error_comes_directly_before_its_median_error():
    """The order is part of the record type, not decoration.

    `mean_error` was added beside `median_error`, before it, so that a caller
    still constructing a `Readout` positionally against the old field order gets
    a missing-argument error rather than a median silently landing in the mean's
    place. Every construction in this file is by keyword; this is the pin on
    the order that those no longer exercise.
    Reached by the names: the two fields swapped, and the mean moved away from
    the median. The last assertion shows the stale five-field call is refused;
    arity alone refuses it, so it does not discriminate the order.
    """
    names = [field.name for field in dataclasses.fields(Readout)]
    assert names[:2] == ["name", "r2"]
    assert names.index("mean_error") + 1 == names.index("median_error")
    with pytest.raises(TypeError, match="missing"):
        Readout("a", 0.5, 1.0, 2.0, np.zeros((3, 4)))  # the old five-field order


def test_a_readout_is_compared_and_hashed_by_identity_despite_its_array_field():
    """`eq=False`: the `ndarray` field made the generated `__eq__` and `__hash__`
    raise, so neither was usable on a `Readout`.

    First shown to be the array: a field-by-field comparison of a readout with a
    copy of itself (equal in every field, with the array copied so it is a
    different object) raises the ambiguity error. Then the readout itself: `==`
    answers instead of raising, `hash` is defined and stable, and a readout keys
    a dict and fills a set. Identity is the semantics, so the copy is a different
    readout.
    Reached: `eq=True` restored (`a == twin` raises `ValueError`, `hash(a)`
    raises `TypeError`).
    """
    a, b, truth, episode = _episode_pair()
    twin = dataclasses.replace(a, predicted=a.predicted.copy())

    def fields(r):
        return (r.name, r.r2, r.mean_error, r.median_error, r.capacity, r.jitter, r.predicted)

    with pytest.raises(ValueError, match="ambiguous"):
        fields(a) == fields(twin)

    assert a == a
    assert (a == twin) is False and (a != twin) is True
    assert (a == b) is False
    assert isinstance(hash(a), int) and hash(a) == hash(a)
    assert {a: "a", b: "b"}[a] == "a"
    assert len({a, b, twin}) == 3


# ---------------------------------------------------------------------------
# The reading: nine measured cells in, one of three verdicts out.
# ---------------------------------------------------------------------------

_ARMS = ("frozen_ssl", "pixel_ae", "random_vit")
# `_UP`: the whole interval is above zero. `_FLAT`: a positive point and a
# positive upper bound over a NEGATIVE lower bound, so it does not resolve and
# it is what separates `ci_low` from `point` and from `ci_high`.
_UP = (0.20, 0.10, 0.30)
_FLAT = (0.01, -0.05, 0.07)


def _iv(point, low, high):
    return {"point": point, "ci_low": low, "ci_high": high}


def _cell(arm, seed, probe, pixel, seq=_FLAT):
    return CeilingCell(
        arm=arm, seed=seed, probe_gain=_iv(*probe), pixel_gain=_iv(*pixel),
        sequence_gain=_iv(*seq),
    )


def _layout(*seeds_per_arm):
    """The (arm, seed) keys: `seeds_per_arm[i]` seeds for the i-th arm."""
    return [
        (arm, seed)
        for arm, n in zip(_ARMS, seeds_per_arm, strict=False)
        for seed in range(n)
    ]


_GAINS_FOR = {
    "PROBE_LIMITED": (_UP, _FLAT),
    "REPRESENTATION_LIMITED": (_FLAT, _UP),
    "OBSERVATION_LIMITED": (_FLAT, _FLAT),
}


def _cells_for(layout, counts, *, reverse=False):
    """`counts[verdict]` cells per verdict, assigned in layout order.

    The fixture is shown to hold what the assertions need BEFORE they run, and
    from the raw interval bounds rather than from `verdict()`, so a wrong
    `verdict()` cannot vouch for its own inputs.
    """
    plan = [verdict for verdict, n in counts.items() for _ in range(n)]
    assert len(plan) == len(layout), "the counts must cover every cell"
    cells = {}
    for (arm, seed), verdict in zip(layout, plan, strict=True):
        probe, pixel = _GAINS_FOR[verdict]
        cells[(arm, seed)] = _cell(arm, seed, probe, pixel)
    probe_up = [k for k, c in cells.items() if c.probe_gain["ci_low"] > 0.0]
    pixel_only = [
        k for k, c in cells.items()
        if c.probe_gain["ci_low"] <= 0.0 < c.pixel_gain["ci_low"]
    ]
    assert len(probe_up) == counts.get("PROBE_LIMITED", 0)
    assert len(pixel_only) == counts.get("REPRESENTATION_LIMITED", 0)
    items = list(cells.items())
    return CeilingInputs(cells=dict(reversed(items) if reverse else items))


@pytest.mark.parametrize("probe_up, pixel_up, seq_up, expected", [
    (False, False, False, "OBSERVATION_LIMITED"),
    (True, False, False, "PROBE_LIMITED"),
    (False, True, False, "REPRESENTATION_LIMITED"),
    (False, False, True, "OBSERVATION_LIMITED"),
    (True, True, False, "PROBE_LIMITED"),
    (True, False, True, "PROBE_LIMITED"),
    (False, True, True, "REPRESENTATION_LIMITED"),
    (True, True, True, "PROBE_LIMITED"),
])
def test_the_verdict_is_a_function_of_which_differences_resolve_and_the_sequence_is_never_one(
    probe_up, pixel_up, seq_up, expected
):
    """All eight combinations, each stated as a value.

    Three same-typed fields feed one verdict, and a read of the wrong one
    returns a plausible verdict: with `sequence_gain` read in place of
    `pixel_gain`, a cell whose ONLY resolved difference is the sequence would
    be called REPRESENTATION_LIMITED. The three rows with a single resolved
    field are what tell the fields apart; the rest pin the precedence.

    Reached: `pixel_gain` and `probe_gain` exchanged, `sequence_gain` read in
    place of `pixel_gain`, and the pixel comparison tested before the probe's.
    """
    cell = _cell(
        "a", 0, _UP if probe_up else _FLAT, _UP if pixel_up else _FLAT,
        _UP if seq_up else _FLAT,
    )
    assert (
        cell.probe_gain["ci_low"] > 0.0,
        cell.pixel_gain["ci_low"] > 0.0,
        cell.sequence_gain["ci_low"] > 0.0,
    ) == (probe_up, pixel_up, seq_up), "fixture must resolve exactly the fields it names"
    assert cell.verdict() == expected


def test_each_verdict_is_reachable_and_the_three_are_distinct():
    """Built as a table so no two rows share a discriminating field by accident."""
    table = {
        "PROBE_LIMITED": ((0.20, 0.10, 0.30), (0.01, -0.05, 0.07)),
        "REPRESENTATION_LIMITED": ((0.01, -0.05, 0.07), (0.20, 0.10, 0.30)),
        "OBSERVATION_LIMITED": ((0.01, -0.05, 0.07), (0.01, -0.05, 0.07)),
    }
    seen = {k: _cell("a", 0, *v).verdict() for k, v in table.items()}
    assert seen == {k: k for k in table}
    assert len(set(seen.values())) == 3
    assert set(seen) == set(VERDICTS)


def test_probe_limited_wins_when_both_differences_resolve():
    """Precedence, stated as a value rather than left to branch order.

    If the linear family was the limit, that is the finding regardless of what
    the pixel comparison says, because the representation comparison was made
    with the weaker estimator on one side.

    Reached: the pixel comparison tested before the probe's.
    """
    assert _cell("a", 0, _UP, _UP).verdict() == "PROBE_LIMITED"


def test_a_lower_bound_of_exactly_zero_does_not_resolve_and_one_above_it_does():
    """`ci_low > 0.0`, strictly. An interval that touches zero does not exclude it.

    No fixture above has a lower bound of exactly 0.0, so a `>=` survived the
    brief's tests. Each comparison is checked at the boundary and just above
    it, so neither can be loosened alone.

    Reached: `ci_low >= 0.0`, in the probe comparison or in the pixel one.
    """
    on_zero, above = (0.05, 0.0, 0.10), (0.05, 1e-9, 0.10)
    assert on_zero[1] == 0.0 < above[1]
    assert _cell("a", 0, on_zero, _FLAT).verdict() == "OBSERVATION_LIMITED"
    assert _cell("a", 0, above, _FLAT).verdict() == "PROBE_LIMITED"
    assert _cell("a", 0, _FLAT, on_zero).verdict() == "OBSERVATION_LIMITED"
    assert _cell("a", 0, _FLAT, above).verdict() == "REPRESENTATION_LIMITED"


@pytest.mark.parametrize("field", ["probe_gain", "pixel_gain", "sequence_gain"])
@pytest.mark.parametrize("bad", [
    _iv(float("nan"), 0.1, 0.3), _iv(0.2, float("nan"), 0.3),
    _iv(0.2, 0.1, float("nan")), _iv(0.2, 0.1, float("inf")),
    _iv(0.2, 0.4, 0.3),
], ids=["nan-point", "nan-low", "nan-high", "inf-high", "inverted"])
def test_a_cell_whose_interval_is_not_one_cannot_be_built(field, bad):
    """`nan > 0.0` is False, so a NaN lower bound read as `no difference` and
    counted toward OBSERVATION_LIMITED, the verdict that says no readout can do
    better, reached by a missing measurement. The cell refuses it, by field name.

    Reached: the validation removed.
    """
    fields = {"probe_gain": _iv(*_UP), "pixel_gain": _iv(*_UP), "sequence_gain": _iv(*_UP)}
    CeilingCell(arm="a", seed=0, **fields)  # the same cell with a sound interval builds
    with pytest.raises(ValueError, match=field):
        CeilingCell(arm="a", seed=0, **{**fields, field: bad})


@pytest.mark.parametrize("field", ["probe_gain", "pixel_gain", "sequence_gain"])
@pytest.mark.parametrize("bad, needle", [
    ({"point": 0.2, "low": 0.1, "high": 0.3}, "'ci_low'"),
    ({"point": 0.2, "ci_low": 0.1}, "'ci_high'"),
    ({"ci_low": 0.1, "ci_high": 0.3}, "'point'"),
    ({}, "'point'"),
    ((0.2, 0.1, 0.3), "tuple"),
    (None, "NoneType"),
], ids=["renamed", "no-high", "no-point", "empty", "tuple", "none"])
def test_a_cell_whose_interval_lacks_a_key_is_refused_by_name_and_is_not_a_keyerror(
    field, bad, needle
):
    """Every other refusal here is a `ValueError` that names the field, and a
    gain dict whose keys were renamed (`low` and `high` for `ci_low` and
    `ci_high`) escaped as `KeyError: 'ci_low'` from inside `__post_init__`,
    which says neither which field nor which cell. The message now names the
    field it belongs to and the key (or the type) that is wrong. `KeyError` is
    not a `ValueError`, so `pytest.raises(ValueError)` fails on the old code.

    Reached: the missing-key and not-a-dict checks removed.
    """
    fields = {"probe_gain": _iv(*_UP), "pixel_gain": _iv(*_UP), "sequence_gain": _iv(*_UP)}
    CeilingCell(arm="a", seed=0, **fields)  # the same cell with a sound interval builds
    with pytest.raises(ValueError) as caught:
        CeilingCell(arm="a", seed=0, **{**fields, field: bad})
    message = str(caught.value)
    assert f".{field}" in message, message
    assert needle in message, message


def test_a_cell_built_positionally_is_refused_because_its_three_gains_look_alike():
    """The three gains are dicts of one shape, so `CeilingCell(arm, seed, pixel,
    probe, sequence)` would build without complaint and read
    REPRESENTATION_LIMITED where PROBE_LIMITED holds, with every figure in the
    table still plausible. `probe_gain_over_floor` guards which READOUTS form a
    gain, not which FIELD receives it, so only the signature can refuse this.

    The fixture is shown to discriminate first: the swap, spelled with keywords,
    does build, and does read a different verdict. Then the positional forms are
    refused, the swapped one and the one in the right order alike.

    Reached: `kw_only=True` removed from `CeilingCell`.
    """
    probe, pixel, sequence = _iv(*_UP), _iv(*_FLAT), _iv(*_FLAT)
    right = CeilingCell(arm="a", seed=0, probe_gain=probe, pixel_gain=pixel, sequence_gain=sequence)
    swapped = CeilingCell(arm="a", seed=0, probe_gain=pixel, pixel_gain=probe, sequence_gain=sequence)
    assert (right.verdict(), swapped.verdict()) == ("PROBE_LIMITED", "REPRESENTATION_LIMITED")

    with pytest.raises(TypeError, match="positional"):
        CeilingCell("a", 0, pixel, probe, sequence)
    with pytest.raises(TypeError, match="positional"):
        CeilingCell("a", 0, probe, pixel, sequence)
    with pytest.raises(TypeError, match="positional"):
        CeilingCell("a", seed=0, probe_gain=probe, pixel_gain=pixel, sequence_gain=sequence)


@pytest.mark.parametrize("sizes, counts, verdict, bar", [
    ((3, 3, 3), {"PROBE_LIMITED": 5, "OBSERVATION_LIMITED": 4}, "PROBE_LIMITED", 5),
    ((3, 3, 3), {"PROBE_LIMITED": 4, "REPRESENTATION_LIMITED": 5}, "REPRESENTATION_LIMITED", 5),
    ((3, 3, 3), {"PROBE_LIMITED": 4, "OBSERVATION_LIMITED": 5}, "OBSERVATION_LIMITED", 5),
    ((3, 3, 3),
     {"PROBE_LIMITED": 4, "REPRESENTATION_LIMITED": 3, "OBSERVATION_LIMITED": 2},
     NO_MAJORITY, 5),
    ((5, 5), {"PROBE_LIMITED": 5, "OBSERVATION_LIMITED": 5}, NO_MAJORITY, 6),
    ((5, 5), {"PROBE_LIMITED": 6, "OBSERVATION_LIMITED": 4}, "PROBE_LIMITED", 6),
    ((3, 3, 5),
     {"PROBE_LIMITED": 5, "REPRESENTATION_LIMITED": 3, "OBSERVATION_LIMITED": 3},
     NO_MAJORITY, 6),
    ((3, 3, 5),
     {"PROBE_LIMITED": 6, "REPRESENTATION_LIMITED": 3, "OBSERVATION_LIMITED": 2},
     "PROBE_LIMITED", 6),
], ids=[
    "nine-five-carries", "nine-representation-five", "nine-four-of-probe-leaves-five-flat",
    "nine-four-three-two", "ten-a-tie", "ten-six", "eleven-five-is-not-six", "eleven-six",
])
def test_a_strict_majority_is_computed_from_the_cells_present(sizes, counts, verdict, bar):
    """Five of nine carries, four does not; six of eleven carries, FIVE DOES NOT;
    half of ten is a tie and a tie is not a majority.

    Nine cells cannot tell a computed bar from a stored 5, since both are 5
    there: the eleven-cell rows can, and read 5 of 11 as a verdict under a
    stored 5. The ten-cell tie separates a strict majority from half.

    THE BRIEF'S "FOUR DOES NOT CARRY" FIXTURE DID NOT SHOW IT. It put four
    cells on PROBE_LIMITED and left the other five flat, and five flat cells ARE
    a strict majority of nine: the reading was OBSERVATION_LIMITED in 5 of 9,
    and its assertion accepted that or NO_MAJORITY, so it could not fail. The
    third row keeps that fixture and says what it reads; the fourth is the one
    in which nothing reaches the bar.

    Reached: the bar stored as 5, a bar of half the cells (`n // 2`), and
    `>= needed` weakened to `> needed`.
    """
    n = sum(sizes)
    got = reading_ceiling(_cells_for(_layout(*sizes), counts))
    assert got.verdict == verdict
    if verdict == NO_MAJORITY:
        assert f"no verdict reached {bar} of {n} cells" in got.rule
    else:
        assert f"{verdict} in {counts[verdict]} of {n} cells" in got.rule
        assert f"strict majority {bar}" in got.rule
    assert {k: len(v) for k, v in got.tally.items()} == {
        name: counts.get(name, 0) for name in VERDICTS
    }


def test_the_tally_names_the_cells_in_sorted_order_whatever_order_they_arrive_in():
    """Every verdict is a key, each holding its (arm, seed) cells sorted.

    The same nine cells inserted in reverse must tally identically, and the
    expected tally is written out rather than taken from either reading.

    Reached: `sorted(...)` removed from the tally loop.
    """
    counts = {"PROBE_LIMITED": 4, "REPRESENTATION_LIMITED": 3, "OBSERVATION_LIMITED": 2}
    expected = {
        "PROBE_LIMITED": (
            ("frozen_ssl", 0), ("frozen_ssl", 1), ("frozen_ssl", 2), ("pixel_ae", 0),
        ),
        "REPRESENTATION_LIMITED": (("pixel_ae", 1), ("pixel_ae", 2), ("random_vit", 0)),
        "OBSERVATION_LIMITED": (("random_vit", 1), ("random_vit", 2)),
    }
    forward = reading_ceiling(_cells_for(_layout(3, 3, 3), counts))
    backward = reading_ceiling(_cells_for(_layout(3, 3, 3), counts, reverse=True))
    assert forward.tally == expected
    assert backward.tally == expected
    assert isinstance(forward, CeilingStatus) and NO_MAJORITY not in forward.tally


def test_two_verdicts_both_holding_the_bar_is_an_error_not_the_first_one(monkeypatch):
    """The partition guard fires rather than picking a winner.

    Two verdicts cannot both hold a strict majority of the same cells, so a bar
    that lets both through is a broken bar, and the answer would otherwise be
    whichever verdict is listed first. The bar is patched to 1 to build the
    state, and the fixture is shown to put two verdicts above it.

    Reached: the `assert len(winners) <= 1` removed.
    """
    cells = _cells_for(_layout(3, 3, 3), {"PROBE_LIMITED": 5, "OBSERVATION_LIMITED": 4})
    monkeypatch.setattr(ceiling, "strict_majority", lambda n: 1)
    with pytest.raises(AssertionError, match="partition is broken"):
        reading_ceiling(cells)


def test_a_majority_that_one_arm_alone_supplies_is_not_a_verdict():
    """`ARMS_REQUIRED` applies to the cells that WIN, not only to the cells present.

    The bar counts cells and not arms: arms of 3 and 9 are twelve cells, the bar
    is 7, and the nine-seed arm reaches it alone. That is NO_MAJORITY and not a
    raise, because the inputs were legal and the arms simply did not agree. One
    cell of the other arm joining the nine makes it a verdict.

    Reached: the check on the winning cells' arms removed.
    """
    one_arm = _cells_for(_layout(3, 9), {"OBSERVATION_LIMITED": 3, "PROBE_LIMITED": 9})
    assert {arm for (arm, _), c in one_arm.cells.items() if c.probe_gain["ci_low"] > 0.0} == {
        "pixel_ae"
    }, "fixture: every winning cell must come from one arm"
    got = reading_ceiling(one_arm)
    assert got.verdict == NO_MAJORITY
    assert "held a numeric majority" in got.rule and "spanned only 1 arm" in got.rule
    assert "pixel_ae" in got.rule
    assert len(got.tally["PROBE_LIMITED"]) == 9, "the tally still carries every vote"

    two_arms = _cells_for(_layout(3, 9), {"OBSERVATION_LIMITED": 2, "PROBE_LIMITED": 10})
    assert reading_ceiling(two_arms).verdict == "PROBE_LIMITED"


@pytest.mark.parametrize("seeds", [1, 2])
def test_an_arm_short_of_seeds_minimum_is_refused_by_name(seeds):
    """`strict_majority(1) == 1`, so without the refusal one lucky cell would
    establish an arm. The short arm is named, with the seeds it does have, and
    the arm that is fine is not.

    One seed AND two: the minimum is three, so an arm of two is short too.

    Reached: the refusal removed, and `< SEEDS_MINIMUM` loosened to `<= 1`.
    """
    cells = {("frozen_ssl", s): _cell("frozen_ssl", s, _FLAT, _FLAT) for s in (0, 1, 2)}
    cells.update({("pixel_ae", s): _cell("pixel_ae", s, _FLAT, _FLAT) for s in range(seeds)})
    with pytest.raises(ValueError, match="SEEDS_MINIMUM") as raised:
        reading_ceiling(CeilingInputs(cells=cells))
    assert "pixel_ae" in str(raised.value) and "frozen_ssl" not in str(raised.value)


def test_one_arm_is_not_a_comparison_and_an_empty_study_is_not_a_reading():
    """Three seeds of ONE arm clear SEEDS_MINIMUM and are still not a verdict.

    Reached: the `ARMS_REQUIRED` refusal removed; the empty-cells refusal removed.
    """
    one = {("frozen_ssl", s): _cell("frozen_ssl", s, _UP, _FLAT) for s in range(3)}
    with pytest.raises(ValueError, match="ARMS_REQUIRED"):
        reading_ceiling(CeilingInputs(cells=one))
    with pytest.raises(ValueError, match="at least one cell"):
        reading_ceiling(CeilingInputs(cells={}))


def test_a_cell_filed_under_another_arm_or_seed_than_its_own_is_refused():
    """The dict key and the cell each say which cell it is, and the tally,
    the seed minimum and the formatter all read the KEY. A cell filed under the
    wrong one is silently counted in the wrong arm; here a seed-0 cell sits at
    seed 1 and an arm-pixel cell at an arm-frozen key.

    Reached: the key/cell comparison removed.
    """
    good = {k: _cell(*k, _FLAT, _FLAT) for k in _layout(3, 3)}
    reading_ceiling(CeilingInputs(cells=good))
    wrong_seed = {**good, ("frozen_ssl", 1): _cell("frozen_ssl", 0, _FLAT, _FLAT)}
    wrong_arm = {**good, ("frozen_ssl", 1): _cell("pixel_ae", 1, _FLAT, _FLAT)}
    for cells in (wrong_seed, wrong_arm):
        with pytest.raises(ValueError, match="filed under"):
            reading_ceiling(CeilingInputs(cells=cells))


def test_the_reading_constants_are_the_ones_the_plan_records():
    """Change detector for the figures that are STATED, not derived."""
    assert SEEDS_MINIMUM == 3
    assert ARMS_REQUIRED == 2
    assert VERDICTS == ("PROBE_LIMITED", "REPRESENTATION_LIMITED", "OBSERVATION_LIMITED")
    assert NO_MAJORITY == "NO_MAJORITY"
    assert NO_MAJORITY not in VERDICTS


def _distinct_nine(*, reverse=False):
    """Nine cells whose three gains all differ, per cell and per column.

    So a column printed under the wrong heading, or a row printed for the wrong
    cell, changes a figure. The probe resolves in 6 cells, the pixel difference
    in 2 others, the sequence in 4 (the cell at index 1 sits exactly ON zero and
    must not be counted), so the rider count of 4 equals neither of the other
    two nor any verdict's tally.
    """
    cells = {}
    for i, (arm, seed) in enumerate(_layout(3, 3, 3)):
        probe = ((0.10 + 0.01 * i, 0.05 + 0.01 * i, 0.15 + 0.01 * i)
                 if i < 6 else (0.01, -0.05 - 0.01 * i, 0.07))
        pixel = ((0.30 + 0.01 * i, 0.20 + 0.01 * i, 0.40 + 0.01 * i)
                 if i in (6, 7) else (0.01, -0.06 - 0.01 * i, 0.08))
        if i in (0, 2, 4, 8):
            seq = (0.50 + 0.01 * i, 0.40 + 0.01 * i, 0.60 + 0.01 * i)
        elif i == 1:
            seq = (0.02, 0.0, 0.09)
        else:
            seq = (0.02, -0.07 - 0.01 * i, 0.09)
        cells[(arm, seed)] = _cell(arm, seed, probe, pixel, seq)
    items = list(cells.items())
    return CeilingInputs(cells=dict(reversed(items) if reverse else items))


def _fields(line):
    """A table line cut at READING_WIDTHS, each field stripped."""
    out, at = [], 0
    for width in READING_WIDTHS:
        out.append(line[at:at + width].strip())
        at += width
    return tuple(out)


def _table(text):
    body = [ln for ln in text.splitlines() if ln.startswith(_ARMS)]
    header = next(ln for ln in text.splitlines() if ln.startswith(READING_COLUMNS[0]))
    return header, body


def test_the_formatter_aligns_and_names_the_sequence_rider_separately():
    """The sequence result is a rider, not a verdict, and must read as one.

    Every figure is checked against a string written out by hand, per column and
    per cell, from cells whose three gains all differ -- the brief's version
    built nine identical cells and checked lengths, so a column under the wrong
    heading, a row for the wrong cell and a rider counted from the wrong gain
    all passed it. Reached: probe and pixel columns exchanged, the rider counted
    from `probe_gain` or from `pixel_gain`, the rider's bar loosened to `>= 0.0`,
    `sorted(...)` removed (the cells arrive reversed), a figure printed at three
    decimals, and a column narrowed until it fills its width.
    """
    inputs = _distinct_nine(reverse=True)
    resolved = [
        sum(getattr(c, gain)["ci_low"] > 0.0 for c in inputs.cells.values())
        for gain in ("probe_gain", "pixel_gain", "sequence_gain")
    ]
    assert resolved == [6, 2, 4], "fixture: the three gains resolve in different numbers of cells"
    assert inputs.cells[("frozen_ssl", 1)].sequence_gain["ci_low"] == 0.0
    text = format_reading_ceiling(reading_ceiling(inputs), inputs)
    header, body = _table(text)

    assert len(header) == sum(READING_WIDTHS)
    assert _fields(header) == READING_COLUMNS
    assert len(body) == 9 and all(len(ln) == len(header) for ln in body)
    # a column that fills its width runs into the next one: every field but the
    # last ends in at least one space of its own
    edges = [sum(READING_WIDTHS[:i]) - 1 for i in range(1, len(READING_WIDTHS))]
    assert all(ln[edge] == " " for ln in [header, *body] for edge in edges)
    assert [ln.split()[0] for ln in body] == [
        f"{arm}_seed{seed}" for arm, seed in _layout(3, 3, 3)
    ]

    rows = {_fields(ln)[0]: _fields(ln) for ln in body}
    assert rows["frozen_ssl_seed0"] == (
        "frozen_ssl_seed0", "+0.1000 [+0.0500,+0.1500]", "+0.0100 [-0.0600,+0.0800]",
        "+0.5000 [+0.4000,+0.6000]", "PROBE_LIMITED",
    )
    assert rows["frozen_ssl_seed1"] == (
        "frozen_ssl_seed1", "+0.1100 [+0.0600,+0.1600]", "+0.0100 [-0.0700,+0.0800]",
        "+0.0200 [+0.0000,+0.0900]", "PROBE_LIMITED",
    )
    assert rows["random_vit_seed1"] == (
        "random_vit_seed1", "+0.0100 [-0.1200,+0.0700]", "+0.3700 [+0.2700,+0.4700]",
        "+0.0200 [-0.1400,+0.0900]", "REPRESENTATION_LIMITED",
    )
    assert rows["random_vit_seed2"] == (
        "random_vit_seed2", "+0.0100 [-0.1300,+0.0700]", "+0.0100 [-0.1400,+0.0800]",
        "+0.5800 [+0.4800,+0.6800]", "OBSERVATION_LIMITED",
    )

    rider = re.search(
        r"rider:\s+history resolves what one frame cannot in (\d+) of (\d+) cells", text
    )
    assert rider is not None and rider.groups() == ("4", "9")
    assert "history resolves" in text.lower()
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_the_formatter_prints_the_status_it_is_handed_and_does_not_recompute_it():
    """The verdict and rule lines are the caller's, verbatim.

    A status is built by hand that no tally of these cells would produce: the
    probe resolves in 6 of the 9, which `reading_ceiling` reads as PROBE_LIMITED
    (the fixture is shown to hold six from the raw bounds), and the hand-made
    status says NO_MAJORITY. Reached: the verdict line recomputed from the cells.
    """
    inputs = _distinct_nine()
    assert sum(c.probe_gain["ci_low"] > 0.0 for c in inputs.cells.values()) == 6
    handed = CeilingStatus(
        verdict=NO_MAJORITY, rule="a rule written for this test",
        tally={name: () for name in VERDICTS},
    )
    lines = format_reading_ceiling(handed, inputs).splitlines()
    assert "verdict: NO_MAJORITY" in lines
    assert "rule:    a rule written for this test" in lines
    assert "verdict: PROBE_LIMITED" not in lines


def test_the_formatter_states_what_a_difference_does_and_does_not_show():
    """Every directional sentence is conditional, and the unit is named.

    The predecessor licensed `the map does not predict motion` from a test that
    only failed to reject. The legend says a difference that does not resolve is
    the ruler failing to find one, and that these are differences in r2 and not
    in map units: no mean or median error appears in this table, so none is
    compared with either baseline here.
    """
    inputs = _distinct_nine()
    text = format_reading_ceiling(reading_ceiling(inputs), inputs)
    assert "not a measurement that there is none" in text
    assert "differences in r2" in text and "not in map units" in text
    assert "MINUS the cell's own measured null floor" in text
    assert "D-C-floor" in READING_COLUMNS[1]
    # the floor is a mean of three draws, averaged BEFORE it is clamped, and the
    # legend is wrapped, so compare with the whitespace folded
    legend = " ".join(text.split())
    assert "the mean of 3 draws, THEN clamped at zero from below" in legend


def test_a_row_refuses_a_value_count_that_does_not_match_its_widths():
    """`zip` stops at the shorter, so a widths tuple one short would print a
    table with its last column silently missing -- and the header, cut the same
    way, would still be `sum(READING_WIDTHS)` long and look aligned.

    Reached: `strict=True` dropped from `_row`.
    """
    assert _row(("a", "b"), (3, 3)) == "a  b  "
    with pytest.raises(ValueError):
        _row(("a", "b", "c"), (3, 3))
    with pytest.raises(ValueError):
        _row(("a", "b"), (3, 3, 3))
    assert len(READING_COLUMNS) == len(READING_WIDTHS)


def test_a_value_that_fills_its_column_is_refused_and_one_a_space_short_is_not():
    """The boundary, at the real width of the probe column. A value that fills
    its column runs into the next, and a row built from it is no longer
    `sum(widths)` long, or is exactly that long and cut in the wrong place.

    Reached: the refusal removed, and `>=` relaxed to `>`.
    """
    width = READING_WIDTHS[1]
    fits = _row(("cell", "y" * (width - 1)), READING_WIDTHS[:2])
    assert len(fits) == READING_WIDTHS[0] + width and fits.endswith("y" + " ")
    with pytest.raises(ValueError, match="cannot hold"):
        _row(("cell", "y" * width), READING_WIDTHS[:2])
    with pytest.raises(ValueError, match="cannot hold"):
        _row(("cell", "y" * (width + 3)), READING_WIDTHS[:2])


def test_the_reading_widths_are_the_ones_the_plan_records():
    """A column is made to fit by refusing what does not, not by widening it."""
    assert READING_WIDTHS == (22, 30, 30, 30, 24)


@pytest.mark.parametrize("where", ["probe_gain", "pixel_gain", "sequence_gain"])
def test_a_figure_too_wide_for_its_column_is_refused_not_printed_misaligned(where):
    """A `probe_gain` point of -1000.0 used to give a row longer than the header,
    with no space between its first two columns, and a fixed-width reader then
    cut every field after it in the wrong place. It is now a `ValueError` naming
    the cell and the string that does not fit, from the formatter, and the
    verdict is still read: only the TABLE is refused, because it cannot show the
    figure faithfully. An r2 difference that large is not expected on the planned
    run.

    The fixture is shown to discriminate first: the figure's cell is at least a
    column wide, the same cell with sound figures is narrower and builds, and
    `reading_ceiling` reads the cells either way.

    Reached: the width check removed from `_row`, for each of the three gains.
    """
    wide, sound, culprit = (-1000.0, -1100.0, -900.0), (-1.0, -1.1, -0.9), "pixel_ae_seed1"

    def study(figures):
        cells = dict(_distinct_nine().cells)
        flat = (0.01, -0.05, 0.07)
        gains = {"probe_gain": flat, "pixel_gain": flat, "sequence_gain": flat}
        gains[where] = figures
        cells[("pixel_ae", 1)] = CeilingCell(
            arm="pixel_ae", seed=1, **{k: _iv(*v) for k, v in gains.items()}
        )
        return CeilingInputs(cells=cells)

    wide_study, sound_study = study(wide), study(sound)
    assert len(_gap(_iv(*wide))) >= READING_WIDTHS[1], "fixture: the figure must not fit"
    assert len(_gap(_iv(*sound))) < READING_WIDTHS[1]
    for inputs in (wide_study, sound_study):
        assert reading_ceiling(inputs).verdict in (*VERDICTS, NO_MAJORITY)
    format_reading_ceiling(reading_ceiling(sound_study), sound_study)

    with pytest.raises(ValueError, match=culprit) as caught:
        format_reading_ceiling(reading_ceiling(wide_study), wide_study)
    assert _gap(_iv(*wide)) in str(caught.value)


def test_an_arm_name_too_long_for_the_cell_column_is_refused_too():
    """The same defect in the first column: `f"{arm}_seed{seed}"` against
    `READING_WIDTHS[0]`, which a long arm name overflows into the probe column.

    Reached: the width check removed from `_row`.
    """
    def study(arm):
        flat = (0.01, -0.05, 0.07)
        return CeilingInputs(cells={
            (a, s): _cell(a, s, flat, flat) for a in (arm, "pixel_ae") for s in range(3)
        })

    sound, long_arm = study("x" * 10), study("x" * 20)
    assert len(f"{'x' * 20}_seed0") >= READING_WIDTHS[0], "fixture: the label must not fit"
    assert len(f"{'x' * 10}_seed0") < READING_WIDTHS[0]
    format_reading_ceiling(reading_ceiling(sound), sound)
    with pytest.raises(ValueError, match="cannot hold"):
        format_reading_ceiling(reading_ceiling(long_arm), long_arm)


# --- the probe gain, read against the cell's own measured floor -------------


def _d_and_c():
    """Two readouts named as the fits name them, `rbf` as D and `linear` as C.

    `_episode_pair` at a milder offset, so the gain of D over C is of the order
    of a floor (0.058, interval 0.017 to 0.100) rather than far above any.
    """
    a, b, truth, episode = _episode_pair(bad=10.0)
    return dataclasses.replace(a, name="rbf"), dataclasses.replace(b, name="linear"), truth, episode


def test_the_floor_comes_off_the_point_and_both_bounds_and_nothing_else():
    """`D - C - floor`: the cell's measured floor, subtracted from all three.

    Expected values are the unshifted interval from `difference_interval`, called
    here with the same arguments, less the floor; equality is exact, since the
    same subtraction on the same floats gives the same float.

    Reached: the floor not subtracted, and subtracted from one bound only.
    """
    d, c, truth, episode = _d_and_c()
    raw = difference_interval(d, c, truth, episode, seed=0)
    assert 0.0 < raw["ci_low"] < raw["point"] < raw["ci_high"]
    got = probe_gain_over_floor(d, c, 0.01, truth, episode, seed=0)
    assert got == {key: raw[key] - 0.01 for key in ("point", "ci_low", "ci_high")}


def test_a_gain_the_floor_swallows_is_not_probe_limited():
    """THE DEFECT TASK 2b EXISTS TO HAVE FIXED, END TO END: a floor read against
    zero. The same D and C give PROBE_LIMITED at a floor of zero and below the
    interval's lower bound, and do not once the floor reaches the lower bound.

    The floors come from the interval itself: half its lower bound (the gain
    survives), the lower bound exactly (the floored bound is exactly 0.0, which
    does not resolve), and halfway from the bound to the point (it does not
    survive). Reached: the floor ignored, and `ci_low >= 0.0`.
    """
    d, c, truth, episode = _d_and_c()
    raw = difference_interval(d, c, truth, episode, seed=0)
    low, point = raw["ci_low"], raw["point"]
    assert 0.0 < low < point

    def verdict_at(floor):
        gain = probe_gain_over_floor(d, c, floor, truth, episode, seed=0)
        return _cell("a", 0, (gain["point"], gain["ci_low"], gain["ci_high"]), _FLAT).verdict()

    assert verdict_at(0.0) == "PROBE_LIMITED"
    assert verdict_at(low / 2.0) == "PROBE_LIMITED"
    assert verdict_at(low) == "OBSERVATION_LIMITED"
    assert verdict_at((low + point) / 2.0) == "OBSERVATION_LIMITED"


def test_a_negative_floor_is_never_added_back():
    """The floor can only RAISE the bar. A null that reads below zero says the
    ladder under-read a linear task by selection noise, which is no credit to
    hand the real difference: `-0.05` subtracted literally would add 0.05 and
    let an interval that includes zero read as PROBE_LIMITED (the measured floors
    run -0.0016 to +0.0000, so this is the usual sign).

    Reached: the floor subtracted as given rather than clamped at zero.
    """
    d, c, truth, episode = _d_and_c()
    raw = difference_interval(d, c, truth, episode, seed=0)
    got = probe_gain_over_floor(d, c, -0.05, truth, episode, seed=0)
    assert got == {key: raw[key] for key in ("point", "ci_low", "ci_high")}
    assert raw["ci_low"] + 0.05 != raw["ci_low"], "fixture: the literal formula would move it"


@pytest.mark.parametrize("d_name, c_name, match", [
    ("rbf", "linear_shipped", "linear_shipped"),
    ("rbf", "rbf", "C must be"),
    ("linear", "rbf", "D must be"),
    ("knn", "linear", "D must be"),
])
def test_the_probe_gain_refuses_readouts_that_are_not_d_and_c(d_name, c_name, match):
    """`C` is the readout over `CEILING_RIDGES`; the shipped-grid readout is the
    reconciliation anchor and reads up to ~0.02 BELOW it, so a gain against it
    is inflated by exactly the ridge-grid resolution Task 2b removed. D and C are
    both `Readout`s and either can be passed as the other: a swap would return a
    gain of the wrong sign with nothing to say so but the name.

    Reached: `linear_shipped` accepted as C, D and C exchanged.
    """
    d, c, truth, episode = _d_and_c()
    d, c = dataclasses.replace(d, name=d_name), dataclasses.replace(c, name=c_name)
    with pytest.raises(ValueError, match=match):
        probe_gain_over_floor(d, c, 0.0, truth, episode, seed=0)


def test_the_probe_gain_refuses_a_floor_that_is_not_a_number():
    d, c, truth, episode = _d_and_c()
    for floor in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="floor"):
            probe_gain_over_floor(d, c, floor, truth, episode, seed=0)


def test_the_probe_gain_draws_its_interval_from_its_own_seed_groups_and_targets():
    """The arguments reach the bootstrap. Each oracle is `difference_interval` at
    a floor of zero (a no-op), and the alternatives it must not equal are shown
    to differ first, so a fixed seed or row-level groups could not hide.

    Reached: the seed fixed at 0, and the groups replaced by one label per row.
    """
    d, c, truth, episode = _d_and_c()

    def oracle(seed, groups):
        return difference_interval(d, c, truth, groups, seed=seed)

    rows = np.arange(len(truth))
    assert oracle(1, episode) != oracle(2, episode), "fixture: the seed must matter"
    assert oracle(1, episode) != oracle(1, rows), "fixture: the groups must matter"
    for seed in (1, 2):
        assert probe_gain_over_floor(d, c, 0.0, truth, episode, seed=seed) == oracle(seed, episode)


def test_a_probe_gain_built_from_scored_readouts_feeds_the_cell():
    """The names the fits give their readouts are the names the guard accepts.

    On the curved task, where the RBF recovers what the linear probe cannot, a
    real `score_readout("rbf")` over a real `score_readout("linear")` gives a gain
    that resolves and a cell that reads PROBE_LIMITED -- and the real
    `score_readout("linear_shipped")` is refused as C.
    """
    train, select, score = _curved_task(0), _curved_task(1), _curved_task(2)
    rbf = score_readout("rbf", train, select, score)
    linear = score_readout("linear", train, select, score)
    shipped = score_readout("linear_shipped", train, select, score)
    assert rbf.r2 - linear.r2 > 0.5, "fixture must put the two readouts far apart"
    groups = np.arange(score[1].shape[0]) // 60

    gain = probe_gain_over_floor(rbf, linear, 0.0, score[1], groups, seed=0)
    assert gain["ci_low"] > 0.0
    cell = CeilingCell(arm="a", seed=0, probe_gain=gain, pixel_gain=_iv(*_FLAT),
                       sequence_gain=_iv(*_FLAT))
    assert cell.verdict() == "PROBE_LIMITED"
    with pytest.raises(ValueError, match="linear_shipped"):
        probe_gain_over_floor(rbf, shipped, 0.0, score[1], groups, seed=0)
