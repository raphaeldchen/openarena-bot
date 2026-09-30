"""M3l: is `z` out of room, or was it never asked?"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mbfps.eval.capacity import (
    CEILING_BITS, FRAME_CUT, SPARE_CUT, argmax_marginal_bits, bits_carried,
    bits_interval, ceiling_bits, entropy_bits, episode_stats, floor_bits,
    live_classes,
)
from mbfps.eval.retention import CONFIDENCE, RESAMPLES
from mbfps.models.rssm import RSSMConfig

CATS, CLASSES = RSSMConfig.z_cats, RSSMConfig.z_classes


def _dirichlet(n: int, *, seed: int = 0, alpha: float = 1.0) -> np.ndarray:
    """A plausible posterior: `n` rows of `CATS` independent distributions."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(CLASSES, alpha), size=(n, CATS))


def _deterministic_uniform(n: int, *, seed: int = 0) -> np.ndarray:
    """A code that is a DETERMINISTIC function of the row, whose argmax is
    uniform over the classes. Its information content is exactly the ceiling."""
    rng = np.random.default_rng(seed)
    probs = np.zeros((n, CATS, CLASSES))
    for j in range(CATS):
        # Every class used exactly n/CLASSES times, so the marginal is uniform.
        classes = np.repeat(np.arange(CLASSES), n // CLASSES)
        rng.shuffle(classes)
        probs[np.arange(n), j, classes] = 1.0
    return probs


def test_the_ceiling_is_derived_from_the_model_config_not_re_spelled():
    """A hardcoded 160.0 would silently disagree with the model if the latent
    shape changed, and every cut in Reading G is a fraction of it."""
    assert CEILING_BITS == RSSMConfig.z_cats * math.log2(RSSMConfig.z_classes)
    assert CEILING_BITS == pytest.approx(160.0)
    assert SPARE_CUT == CEILING_BITS / 2, "the spare cut is HALF THE DERIVED ceiling"
    assert FRAME_CUT == 0.5


def test_the_ceiling_follows_the_config_when_the_config_changes(monkeypatch):
    """The test above cannot tell a derived ceiling from a hardcoded 160.0 --
    at the current latent shape the two are equal, so it passes either way.
    Re-execute the module under a DIFFERENT shape and the difference shows. A
    fresh copy under another name, so the real module is never reloaded."""
    import importlib.util
    import sys

    import mbfps.eval.capacity as real

    monkeypatch.setattr(RSSMConfig, "z_cats", 8)
    monkeypatch.setattr(RSSMConfig, "z_classes", 16)
    spec = importlib.util.spec_from_file_location("_capacity_reshaped", real.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_capacity_reshaped", fresh)
    spec.loader.exec_module(fresh)

    assert fresh.CEILING_BITS == 8 * math.log2(16) == 32.0
    assert fresh.SPARE_CUT == 16.0, "half the DERIVED ceiling, not a spelled 80.0"


def test_a_deterministic_uniform_code_reads_exactly_log2_of_the_classes():
    """THE LOG-BASE CHECK, and the only one that catches nats.

    Measured: in nats the floor control still reads 0.000000 and both ceiling
    routes still agree exactly, and nats reads BELOW the bits ceiling so no
    inequality catches it either. Only a case with a known ABSOLUTE value does.
    A deterministic code with a uniform marginal carries exactly
    `log2(z_classes)` bits per categorical -- 5.000, where nats reads 3.466."""
    probs = _deterministic_uniform(CLASSES * 20)
    per_categorical = bits_carried(probs) / CATS
    assert per_categorical == pytest.approx(math.log2(CLASSES), abs=1e-9), (
        f"read {per_categorical:.4f} bits per categorical; "
        f"log2({CLASSES}) = {math.log2(CLASSES):.4f}, "
        f"ln({CLASSES}) = {math.log(CLASSES):.4f}"
    )
    assert bits_carried(probs) == pytest.approx(CEILING_BITS, abs=1e-9)


def test_the_floor_control_reads_zero_to_summation_order():
    """Every row replaced by the marginal: `H(m) - mean_n H(m)` is identically
    zero in exact arithmetic, so any reading away from zero is an arithmetic
    defect -- a missing normalisation, or a mean over the wrong axis. Computed
    through the real estimator on real-shaped data, not a fixture.

    NOT bit-exact, and the tolerance is the point. Measured: +5.68e-14 at 500
    rows, +8.53e-14 at 2,000, -4.26e-13 at the ~11,000 a real cell carries --
    float summation order, not a defect. This project has been bitten by the
    other reading of this: an M3h sweep refused on an 8.527e-14 mismatch that
    was summation order, costing a debugging cycle. 1e-9 is four orders above
    the observed drift and eleven below anything a real defect would produce."""
    for n in (500, 2000, 11000):
        assert floor_bits(_dirichlet(n, seed=1)) == pytest.approx(0.0, abs=1e-9)


def test_the_two_ceiling_routes_agree():
    """`ceiling_bits` substitutes a one-hot at each row's argmax and runs the
    real estimator; `argmax_marginal_bits` histograms the argmax INDICES. Two
    routes, one answer. This pins the routing -- a mean over the wrong axis, or
    a substitution that leaks a row's own distribution, breaks the agreement --
    but NOT the log base, which the test above owns."""
    probs = _dirichlet(500, seed=2)
    assert ceiling_bits(probs) == pytest.approx(argmax_marginal_bits(probs), abs=1e-9)


def test_the_only_bound_on_bits_carried_is_the_derived_ceiling():
    """THE TWO inequalities a refusal may rest on, both theorems:

        -1e-9 <= bits_carried <= CEILING_BITS + 1e-9
         0    <= ceiling_bits <= CEILING_BITS + 1e-9

    `bits = H(marginal) - E_n H(row) <= H(marginal) <= cats * log2(classes)`,
    so the upper bound follows. The lower bound needs the TOLERANCE, not `0 <=`:
    a collapsed code reads +4.3e-13 at 11,000 rows and can read negative by the
    same summation order.

    Both are checked at several row counts, because a bound that holds at 500
    rows and not at 11,000 is the one that would bite on the real run."""
    for n in (500, 2000, 11000):
        probs = _dirichlet(n, seed=3)
        assert -1e-9 <= bits_carried(probs) <= CEILING_BITS + 1e-9
        assert 0.0 <= ceiling_bits(probs) <= CEILING_BITS + 1e-9


def test_ceiling_bits_is_not_an_upper_bound_on_bits_carried():
    """`bits_carried <= ceiling_bits` IS NOT A THEOREM, and an earlier draft of
    the design asserted it as a refusal.

    The two measure different things. `ceiling_bits` is the information content
    of the ARGMAX PATTERN; `bits_carried` is that of the DISTRIBUTION. A code
    whose argmax never moves while its tail varies carries real information with
    an argmax pattern carrying none.

    This test exists so nobody re-adds that refusal. It would have rejected
    valid readings precisely in the diffuse, low-information regime this
    milestone exists to investigate -- the most dangerous place for a spurious
    refusal, because it is where the answer lives."""
    rng = np.random.default_rng(0)
    tail = rng.dirichlet(np.ones(CLASSES - 1), size=(500, CATS)) * 0.4
    probs = np.concatenate([np.full((500, CATS, 1), 0.6), tail], axis=-1)
    assert (probs.argmax(axis=-1) == 0).all(), "the premise: the argmax never moves"

    carried, ceiling = bits_carried(probs), ceiling_bits(probs)
    assert ceiling == pytest.approx(0.0, abs=1e-9), "an argmax that never moves"
    assert carried > 1.0, "yet the distribution carries real information"
    assert carried > ceiling, (
        "the counterexample: bits_carried exceeds ceiling_bits, so a refusal on "
        "`bits <= ceiling` would reject this valid reading"
    )
    assert carried <= CEILING_BITS + 1e-9, "the real bound still holds"


@pytest.mark.parametrize("n", [300, 11000])
def test_a_code_that_ignores_its_input_carries_no_bits(n):
    """The other end of the scale, and the posterior-collapse case: every row
    the SAME distribution means the code says nothing about which row it is.

    1e-9, not tighter: this is the same summation-order arithmetic as the floor
    control. Measured on this fixture: -5.7e-14 at 300 rows and up to 8e-13 at
    11,000, so a 1e-12 tolerance leaves no headroom at a real cell's size."""
    one = _dirichlet(1, seed=4)[0]
    probs = np.broadcast_to(one, (n, CATS, CLASSES)).copy()
    assert bits_carried(probs) == pytest.approx(0.0, abs=1e-9)


def test_live_classes_counts_the_columns_that_vary():
    """A class that never varies across the scored rows is capacity the code is
    not using, and R^2 is undefined for it, so `frame_share` must exclude it.
    The count is reported beside the verdict: a CAPACITY_BOUND read printed
    next to a low `live_classes` would be self-contradicting."""
    probs = _dirichlet(300, seed=8)
    assert live_classes(probs) == CATS * CLASSES, "every class varies here"

    # A DIRICHLET base, not the one-hot one: zeroing a class in a one-hot code
    # empties every row whose one-hot WAS that class, and renormalising then
    # divides by zero. Measured on the Dirichlet base, the smallest row sum
    # after zeroing is 0.7242, so no row is emptied and no warning is raised.
    dead = probs.copy()
    dead[:, :, 7] = 0.0  # class 7 never varies anywhere: CATS dead columns
    dead /= dead.sum(axis=-1, keepdims=True)
    assert live_classes(dead) == CATS * CLASSES - CATS

    # The case the zeroed-class fixture cannot tell apart: a column that is
    # CONSTANT BUT NONZERO -- which is what posterior collapse looks like, every
    # row the same distribution. "Varies" is not "is nonzero", and it is not
    # `std > 0` either: float64 `std` of identical values is rounding noise, not
    # zero. Measured on this fixture: 1013 of 1024 columns read as live at 300
    # rows and all 1024 at 11,000, from a code in which NOTHING varies.
    collapsed = np.broadcast_to(_dirichlet(1, seed=4)[0], (300, CATS, CLASSES)).copy()
    assert live_classes(collapsed) == 0, "a collapsed code has no live classes"


def test_the_interval_from_sufficient_statistics_matches_a_direct_resample():
    """`bits_carried` is a function of per-episode sufficient statistics --
    the summed probabilities, the summed row entropies and the row count -- so
    the bootstrap resamples 24 small arrays instead of re-slicing an
    (11000, 32, 32) one. Measured: 0.02 ms against 54.89 ms per resample, a
    2601x speedup, and that is what makes the whole run ~30 minutes.

    It is EXACT, not an approximation -- but exact up to summation ORDER, which
    is why this compares with `approx` at 1e-9 rather than `==`. This project
    has been bitten by claiming bit-identity where only near-identity holds: an
    M3h sweep refused on a 8.527e-14 mismatch that was summation order, not a
    defect."""
    probs = _dirichlet(600, seed=5)
    rng = np.random.default_rng(11)
    groups = rng.integers(0, 12, size=600)
    stats = episode_stats(probs, groups)

    picked = rng.integers(0, 12, size=12)
    rows = np.concatenate([np.flatnonzero(groups == g) for g in picked])
    direct = bits_carried(probs[rows])
    from mbfps.eval.capacity import _bits_from_stats
    assert _bits_from_stats(stats, picked) == pytest.approx(direct, abs=1e-9)


def test_the_interval_is_the_percentile_bootstrap_it_claims_to_be():
    """Rebuilt from the RAW ROWS under the same seed, not merely bracketed.

    Bracketing the point estimate and asserting a nonzero width is far too weak:
    measured, ALL of these mutants pass such a test --
      - `np.quantile(draws, [1 - confidence, confidence])`, a 90% interval where
        95% was asked for;
      - `(draws.min(), draws.max())`, no percentile at all;
      - `size=n_episodes // 2`, a half-size resample.
    Only a 50% interval fails, and only incidentally, because the point estimate
    falls outside it. This interval is what `SPARE_CAPACITY` compares against the
    cut, so its LEVEL is load-bearing and has to be pinned, not its existence.

    The reference arm draws from raw rows through `bits_carried`, so it shares no
    code with `_bits_from_stats` -- a comparison whose two arms came from one
    path could only detect nondeterminism, which has bitten this branch four
    times."""
    probs = _dirichlet(400, seed=6)
    groups = np.random.default_rng(12).integers(0, 10, size=400)
    stats = episode_stats(probs, groups)
    resamples = 200
    out = bits_interval(stats, resamples=resamples, confidence=CONFIDENCE, seed=0)

    n_episodes = int(np.unique(groups).size)
    rng = np.random.default_rng(0)
    index = np.arange(n_episodes)
    draws = []
    for _ in range(resamples):
        picked = rng.choice(index, size=n_episodes, replace=True)
        rows = np.concatenate([np.flatnonzero(groups == stats.labels[g]) for g in picked])
        draws.append(bits_carried(probs[rows]))
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])

    assert out["bits"] == pytest.approx(bits_carried(probs), abs=1e-9)
    assert out["ci_low"] == pytest.approx(float(low), abs=1e-9), (
        "the lower bound is not the percentile of a same-seed raw-row bootstrap"
    )
    assert out["ci_high"] == pytest.approx(float(high), abs=1e-9), (
        "the upper bound is not the percentile of a same-seed raw-row bootstrap"
    )
    assert out["ci_low"] < out["ci_high"]
    assert out["n_episodes"] == n_episodes == 10
    assert out["confidence"] == CONFIDENCE


def test_the_interval_needs_two_episodes():
    """One resampling unit cannot produce an interval, and returning a
    zero-width one would read as a precise measurement."""
    probs = _dirichlet(400, seed=6)
    with pytest.raises(ValueError, match="at least two"):
        bits_interval(episode_stats(probs, np.zeros(400, dtype=int)),
                      resamples=10, confidence=CONFIDENCE, seed=0)


def test_the_estimator_refuses_rows_that_are_not_distributions():
    """A caller handing logits instead of probabilities would get a number, not
    an error, and it would look plausible. Refused at the point the array first
    becomes a reading."""
    probs = _dirichlet(50, seed=7)
    bad = probs.copy()
    bad[3, 5, :] *= 2.0
    with pytest.raises(ValueError, match="sum to 1"):
        bits_carried(bad)

    negative = probs.copy()
    negative[1, 2, 0] = -0.1
    negative[1, 2, 1] += 0.1
    with pytest.raises(ValueError, match="negative"):
        bits_carried(negative)

    with pytest.raises(ValueError, match="z_cats"):
        bits_carried(probs.reshape(50, CATS * CLASSES))


def test_the_estimator_refuses_an_empty_gather():
    """Zero rows made `bits_carried` divide by zero and return NaN with a
    RuntimeWarning. NaN compares False against every cut, so an empty gather
    would have read as a clean null in both senses at once -- the exact failure
    mode the finiteness guards elsewhere in this project exist for."""
    with pytest.raises(ValueError, match="no rows"):
        bits_carried(np.zeros((0, CATS, CLASSES)))
