"""M3l: is `z` out of room, or was it never asked?"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mbfps.eval.capacity import (
    CEILING_BITS, FRAME_CUT, PAIR_CEILING_BITS, RATIO_MIN_FLOOR, SPARE_CUT,
    _pair_mutual_information, argmax_marginal_bits, bits_carried, bits_interval,
    ceiling_bits, entropy_bits, episode_stats, floor_bits, live_classes,
    redundancy_bits, redundancy_floor, redundancy_ratio,
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
    assert fresh.PAIR_CEILING_BITS == math.log2(16) == 4.0, (
        "a hardcoded 5.0 equals log2(32) at the shipped shape and passes the "
        "test that checks it against CLASSES; only another shape tells them apart"
    )

    # The estimator itself, at the other shape: a `32` spelled anywhere in the
    # pair enumeration or the joint's reshape raises or misreads here, and at the
    # shipped shape it would pass. Every categorical copies one 16-class variable,
    # so each pair shares all of it -- exactly the derived per-pair ceiling.
    rng = np.random.default_rng(0)
    shared = np.repeat(np.arange(16), 20)
    rng.shuffle(shared)
    reshaped = np.zeros((shared.size, 8, 16))
    reshaped[np.arange(shared.size), :, shared] = 1.0
    assert fresh.redundancy_bits(reshaped) == pytest.approx(
        fresh.PAIR_CEILING_BITS, abs=1e-9
    )
    assert fresh.redundancy_floor(reshaped, seed=0) > 0.0


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




# ---------------------------------------------------------------------------
# redundancy: the companion that makes a high `bits_carried` interpretable
#
# THE FIXTURES HERE ARE DELIBERATELY NOT ALL ONE-HOT AND NOT ALL IID. The first
# set were both, with exactly uniform marginals and every pair behaving
# identically, and that is why they could see neither of the two failures the
# review found -- an argmax basis blind to the tails, and a permutation floor
# blind to autocorrelation -- nor three estimator bugs (a wrong `outer`, an
# adjacent-only pair enumeration, a hardcoded ceiling). Each fixture below names
# the wrong implementation it exists to catch.
# ---------------------------------------------------------------------------

N_PAIRS = CATS * (CATS - 1) // 2


def _onehot_from_indices(idx: np.ndarray) -> np.ndarray:
    probs = np.zeros((idx.shape[0], CATS, CLASSES))
    np.put_along_axis(probs, idx[..., None], 1.0, axis=-1)
    return probs


def _exactly_uniform_column(n: int, rng) -> np.ndarray:
    """One categorical using every class EXACTLY `n // CLASSES` times.

    `rng.integers` gives an approximately uniform marginal, and approximate is
    not good enough: these fixtures' premise is that both codes read EXACTLY the
    ceiling, and `integers` reads 159.4645 at n = 1280. A shuffled `repeat` makes
    the marginal exactly uniform, so the reading is exactly 160.0000."""
    classes = np.repeat(np.arange(CLASSES), n // CLASSES)
    rng.shuffle(classes)
    return classes


def _independent_and_duplicated(n: int, *, seed: int):
    """Two codes that `bits_carried` CANNOT tell apart: both read exactly the
    ceiling, one carrying all 160 bits jointly and one carrying 5."""
    rng = np.random.default_rng(seed)
    independent = _onehot_from_indices(
        np.stack([_exactly_uniform_column(n, rng) for _ in range(CATS)], axis=1)
    )
    shared = _exactly_uniform_column(n, rng)
    return independent, _onehot_from_indices(np.stack([shared] * CATS, axis=1))


def _reference_pair_mi(pj: np.ndarray, pk: np.ndarray) -> float:
    """`I(z_j ; z_k)` in bits for two `(n, classes)` arrays of row distributions.

    The textbook definition, ONE PAIR AT A TIME through a matmul and with the
    marginals taken from the data. It shares no code with the estimator under
    test, which does every pair in one `einsum` and reads the marginals off the
    joint -- so a bug in either construction shows as a disagreement."""
    joint = pj.T @ pk / pj.shape[0]
    outer = np.outer(pj.mean(axis=0), pk.mean(axis=0))
    nz = joint > 0.0
    return float((joint[nz] * np.log2(joint[nz] / outer[nz])).sum())


def _reference_pair_matrix(probs: np.ndarray) -> np.ndarray:
    """`(CATS, CATS)` upper-triangular matrix of `_reference_pair_mi`."""
    out = np.zeros((CATS, CATS))
    for j in range(CATS):
        for k in range(j + 1, CATS):
            out[j, k] = _reference_pair_mi(probs[:, j], probs[:, k])
    return out


def _permutation_floor(probs: np.ndarray, *, seed: int) -> float:
    """The null this module USED to use: every categorical's rows permuted
    independently. Kept here, as a reference, to show what it does to clustered
    data -- the module no longer ships it."""
    rng = np.random.default_rng(seed)
    permuted = probs.copy()
    for j in range(CATS):
        permuted[:, j] = probs[rng.permutation(probs.shape[0]), j]
    return _pair_mutual_information(permuted)


def test_redundancy_separates_an_independent_code_from_a_duplicated_one():
    """The companion exists because `bits_carried` CANNOT tell these two apart:
    both read the full 160-bit ceiling, one carrying 160 bits jointly and one
    carrying 5. That is the whole reason a high `bits_carried` does not
    establish "the capacity is in use" on its own.

    So this test asserts the thing `bits_carried` cannot: that the two codes are
    distinguishable, and in which direction."""
    independent, duplicated = _independent_and_duplicated(CLASSES * 80, seed=0)

    # The premise: bits_carried is blind to the difference. Both read EXACTLY
    # the ceiling, so nothing about the level distinguishes them.
    assert bits_carried(independent) == pytest.approx(CEILING_BITS, abs=1e-9)
    assert bits_carried(duplicated) == pytest.approx(CEILING_BITS, abs=1e-9)

    low, high = redundancy_bits(independent), redundancy_bits(duplicated)
    assert high == pytest.approx(PAIR_CEILING_BITS, abs=1e-9), (
        "every categorical is the same variable, so each pair shares all of it"
    )
    assert low < high / 10.0, f"independent {low:.4f} against duplicated {high:.4f}"
    assert PAIR_CEILING_BITS == pytest.approx(math.log2(CLASSES))


def test_an_independent_code_sits_at_its_floor_and_a_duplicated_one_far_above():
    """Mutual information is biased UPWARD at finite sample, so an independent
    code does not read 0 -- it reads ~0.064 bits per pair at 11,008 rows. A fixed
    tolerance would be a guess about the sample size.

    `redundancy_floor` rolls each categorical circularly by its own offset, which
    destroys the dependence between categoricals while keeping each one's own
    structure, so what it reads is the bias for this data at this size. It is a
    NULL CALIBRATION, not a known answer: the known answer is pinned separately,
    by Miller-Madow, in the test below. Two things must hold here: an
    independent code sits AT its own floor, and a duplicated one sits far above."""
    independent, duplicated = _independent_and_duplicated(CLASSES * 80, seed=1)

    floor = redundancy_floor(independent, seed=0)
    assert floor > 0.0, "the bias is real; a floor of exactly 0 would be a bug"
    # Measured: the independent code sits at ratio 0.997-1.001 of its own floor
    # at n = 320, 1280, 2560 and 11,008, so `rel=0.05` is the tight assertion the
    # data supports rather than a loose one chosen to pass.
    assert redundancy_bits(independent) == pytest.approx(floor, rel=0.05), (
        "an independent code must sit at its own finite-sample floor"
    )
    # Measured ratios for the duplicated code: 2.6x at n=320, 7.6x at 1280, 16.0x
    # at 2560, 78.0x at the ~11,000 a real cell carries. The factor below holds
    # from 2560 up, and the fixture is 2560 -- chosen so the assertion is not
    # tuned to the one size where it barely passes.
    assert redundancy_bits(duplicated) > 10.0 * redundancy_floor(duplicated, seed=0), (
        "a duplicated code must sit far above the floor, or the floor is "
        "absorbing the signal it exists to exclude"
    )


def test_the_floor_matches_miller_madow_for_independent_categoricals():
    """THE FLOOR'S INDEPENDENT KNOWN ANSWER, which it otherwise lacks.

    `redundancy_floor` is measured through the code path under test, so on its
    own it could be wrong by a constant factor and still agree with
    `redundancy_bits` on every independent code -- both would share the error.
    The plug-in bias of the mutual information between two independent discrete
    variables of `K` classes is `(K-1)^2 / (2 n ln 2)` bits (Miller-Madow),
    derived without running any of this code: 0.0630 at n = 11,008.

    Measured on this fixture: floor 0.0641-0.0644 across five seeds, 1.8-2.3%
    above the first-order formula (the next term is positive). So `rel=0.05` is
    the tight bound the data supports. Only at the size a real cell carries: at
    n = 1280 the same formula is 15% off, because it is first-order in 1/n, which
    is why this is not parametrised over small sizes."""
    n = CLASSES * 344
    assert n == 11008
    rng = np.random.default_rng(0)
    independent = _onehot_from_indices(
        np.stack([_exactly_uniform_column(n, rng) for _ in range(CATS)], axis=1)
    )
    bias = (CLASSES - 1) ** 2 / (2 * n * math.log(2))
    assert bias == pytest.approx(0.0630, abs=5e-5), "the formula itself, not the module"

    assert redundancy_floor(independent, seed=0) == pytest.approx(bias, rel=0.05)
    assert redundancy_bits(independent) == pytest.approx(bias, rel=0.05)


def test_the_floor_shrinks_as_the_sample_grows():
    """It is a finite-SAMPLE bias, so it must fall with n. If it did not, it
    would be measuring something other than the bias -- and a floor that stayed
    put would silently mis-scale on the real run's ~11,000 rows."""
    floors = []
    for n in (CLASSES * 10, CLASSES * 80):
        independent, _ = _independent_and_duplicated(n, seed=2)
        floors.append(redundancy_floor(independent, seed=0))
    assert floors[0] > floors[1] * 1.5, f"floors did not shrink with n: {floors}"


# --- Important 1: the argmax basis cannot see redundancy living in the tails ---


def _tail_shared_code(n: int, *, seed: int, own_mass: float = 0.55) -> np.ndarray:
    """Independent argmaxes; a tail shared across ALL 32 categoricals.

    Each categorical's argmax is its own exactly-uniform column, so the argmax
    pattern is independent by construction. Every row's remaining `1 - own_mass`
    is spread by ONE distribution that is a function of four per-row numbers
    common to every categorical -- the redundancy lives entirely in the tails.
    `own_mass > 0.5` keeps each argmax on its own column whatever the tail does.

    SOFT rows and a SHARED structure: neither one-hot nor independent across
    categoricals, which is exactly what the first fixtures were not."""
    rng = np.random.default_rng(seed)
    own = np.stack([_exactly_uniform_column(n, rng) for _ in range(CATS)], axis=1)
    shared = rng.standard_normal((n, 4))
    logits = 2.0 * shared @ rng.standard_normal((4, CLASSES))
    tail = np.exp(logits - logits.max(axis=1, keepdims=True))
    tail /= tail.sum(axis=1, keepdims=True)
    probs = np.zeros((n, CATS, CLASSES))
    np.put_along_axis(probs, own[..., None], own_mass, axis=-1)
    probs += ((1.0 - own_mass) * tail)[:, None, :]
    assert (probs.argmax(axis=-1) == own).all(), "the premise: argmaxes stay put"
    return probs


def test_redundancy_sees_dependence_that_lives_only_in_the_tails():
    """AN ARGMAX BASIS CALLS THIS CODE INDEPENDENT, and `bits_carried` reads it
    as in use. Both are wrong.

    The argmaxes are independent, so a redundancy measured on them sits at its
    floor (measured ratio 1.004). But `bits_carried` is a function of the
    DISTRIBUTIONS, and four per-row values shared across all 32 categoricals make
    the distributions strongly redundant. Measured at 11,008 rows: `bits_carried`
    98.32 -- above the 80-bit cut, so it reads "capacity in use" -- with the
    distribution-level redundancy at 0.1353 bits per pair against a floor of
    0.0108, a ratio of 12.55.

    An estimator on the argmaxes reads that ratio as 1.004 and reports everything
    fine; this test fails on it. The review's own counterexample was sharper:
    argmax redundancy at 1.006x its floor while the true joint information was
    below the cut."""
    probs = _tail_shared_code(CLASSES * 344, seed=0)

    # The premises. High reading: the status this companion exists to qualify.
    assert bits_carried(probs) > SPARE_CUT, "must read 'capacity in use'"
    # And the argmax pattern really is independent, judged by the module's own
    # estimator on the one-hot of the argmaxes -- what an argmax basis would see.
    argmax_code = _onehot_from_indices(probs.argmax(axis=-1))
    argmax_ratio = redundancy_bits(argmax_code) / redundancy_floor(argmax_code, seed=0)
    assert argmax_ratio == pytest.approx(1.0, abs=0.05), (
        "the premise: an argmax basis finds nothing here"
    )

    ratio = redundancy_ratio(redundancy_bits(probs), redundancy_floor(probs, seed=0))
    assert ratio is not None and ratio > 4.0, (
        f"distribution-level redundancy is {ratio}x its floor; the tails are "
        "shared across all 32 categoricals, so this must be far above 1"
    )


# --- Important 2: the floor assumes exchangeable rows; this gather's are clustered ---


def _persistent_independent_code(
    windows: int, *, persistence: float, seed: int, window: int = 50, noise: float = 0.15
) -> np.ndarray:
    """Categoricals INDEPENDENT of each other that persist within a window.

    Each categorical's class is kept from the previous row with probability
    `persistence` and redrawn otherwise, restarting at every `window`-row
    boundary -- the shape of this gather, whose rows are windows of consecutive
    frames inside episodes. Rows are soft (`noise` of fresh Dirichlet mass on a
    row's own categorical) and strongly autocorrelated, neither of which the
    first fixtures were. Independence BETWEEN categoricals holds by construction:
    each draws from its own stream."""
    rng = np.random.default_rng(seed)
    classes = np.empty((windows, window, CATS), dtype=int)
    classes[:, 0] = rng.integers(0, CLASSES, size=(windows, CATS))
    for t in range(1, window):
        keep = rng.random((windows, CATS)) < persistence
        fresh = rng.integers(0, CLASSES, size=(windows, CATS))
        classes[:, t] = np.where(keep, classes[:, t - 1], fresh)
    probs = _onehot_from_indices(classes.reshape(windows * window, CATS))
    soft = rng.dirichlet(np.ones(CLASSES), size=probs.shape[:2])
    return (1.0 - noise) * probs + noise * soft


def test_the_floor_keeps_autocorrelation_so_persistence_is_not_read_as_redundancy():
    """INDEPENDENT CATEGORICALS THAT MERELY PERSIST MUST SIT NEAR THEIR FLOOR.

    This gather's ~11,450 rows are windows of consecutive frames, so every
    categorical is strongly autocorrelated, and a plug-in mutual information is
    biased UP further by exactly that (fewer effective samples). A floor that
    permutes the rows destroys the autocorrelation and so underestimates the
    bias, and independent categoricals then read as massively redundant: the
    ratio measures the autocorrelation and "near the floor" becomes unreachable.

    Measured at 5,000 rows, persistence 0.95: against the circular-shift floor the
    independent code reads 1.12x, against a full-permutation floor 10.74x.
    `bits_carried` is 118, so the high reading this companion exists to qualify
    is exactly what is at stake. A permutation floor fails this test."""
    probs = _persistent_independent_code(100, persistence=0.95, seed=0)
    assert bits_carried(probs) > SPARE_CUT, "must read 'capacity in use'"

    redundancy = redundancy_bits(probs)
    # The premise, shown rather than assumed: the OLD null mistakes this code for
    # a redundant one. If the fixture were not autocorrelated enough for that,
    # the assertion below could not tell the two floors apart.
    permuted = redundancy / _permutation_floor(probs, seed=0)
    assert permuted > 5.0, f"the fixture is not persistent enough ({permuted:.2f}x)"

    ratio = redundancy_ratio(redundancy, redundancy_floor(probs, seed=0))
    assert ratio is not None and ratio < 1.5, (
        f"independent-but-persistent categoricals read {ratio}x their floor; the "
        "floor is not preserving each series' own autocorrelation"
    )


# --- Important 3: estimator bugs that every one-hot iid fixture is blind to ---


def _enumerated_state_code(counts, *, seed: int):
    """Rows that are an EXACT enumeration of a few discrete states.

    Row `n` is in state `s`, and categorical `j`'s distribution is `table[s, j]`,
    a SOFT SKEWED draw (Dirichlet 0.3). Categoricals are dependent only through
    the shared state -- partial dependence, at most `H(state)` bits -- and every
    pair has its own marginals and its own value. Nothing is sampled, so the
    empirical joint equals the population joint exactly and an analytic answer
    can be computed from the table alone."""
    rng = np.random.default_rng(seed)
    table = rng.dirichlet(np.full(CLASSES, 0.3), size=(len(counts), CATS))
    states = np.repeat(np.arange(len(counts)), counts)
    return table[states], table, np.asarray(counts, dtype=float) / sum(counts)


def _analytic_pair_mi(table, weights, j: int, k: int) -> float:
    """`I(z_j ; z_k)` from the generating table alone -- no fixture row is read.

    `joint = sum_s w_s * outer(table[s, j], table[s, k])`, the marginals are the
    same mixture of the per-state rows."""
    joint = sum(w * np.outer(t[j], t[k]) for w, t in zip(weights, table))
    mj = sum(w * t[j] for w, t in zip(weights, table))
    mk = sum(w * t[k] for w, t in zip(weights, table))
    return float((joint * np.log2(joint / np.outer(mj, mk))).sum())


def test_redundancy_matches_an_analytic_value_with_skewed_marginals():
    """SKEWED MARGINALS AND PARTIAL DEPENDENCE, against a value computed
    independently. Uniform marginals are what let these wrong `outer` products
    through -- with them every ordering of the axes gives the same number:
      - `outer` built from `sum(axis=1)` twice (both factors the row marginal);
      - `outer` with swapped axes, or `joint` transposed alone;
      - a mean over the diagonal as well as the `j < k` pairs;
      - adjacent-only pairs, since every pair here has its own value.
    The expected value is the mean over all 496 pairs of the analytic pair MI,
    computed from the table that generated the rows and nothing else. Measured:
    0.37311, agreeing with the module to 1.1e-16, the pairs spanning 0.204 to
    0.615 -- so a mean over the wrong pairs cannot hide in a flat distribution."""
    probs, table, weights = _enumerated_state_code([1000, 500, 300, 200], seed=0)
    pair_values = [
        _analytic_pair_mi(table, weights, j, k)
        for j in range(CATS) for k in range(j + 1, CATS)
    ]
    expected = float(np.mean(pair_values))

    # The premises -- that this fixture can tell the mutants apart at all.
    marginals = probs.mean(axis=0)
    assert marginals.max(axis=1).mean() > 3.0 / CLASSES, "marginals are skewed"
    assert not np.allclose(marginals[0], marginals[1]), "j and k differ in marginal"
    assert np.ptp(pair_values) > 0.1, "every pair must have its own value"
    assert 0.0 < expected < 1.0, "partial dependence, below H(state) = 1.74 bits"

    assert redundancy_bits(probs) == pytest.approx(expected, abs=1e-9)


def test_redundancy_matches_a_table_computed_by_hand():
    """The smallest case with an answer that needs no code to derive.

    Twenty rows; categoricals 3 and 20 (NOT adjacent) take two classes each with
    joint `[[.55, .05], [.20, .20]]`, so marginals (.60, .40) and (.75, .25) --
    different from each other and from uniform. The other thirty categoricals are
    constant. `I = sum p log2(p / (pj pk))`, written out term by term below, and
    the module's figure is that over 496 pairs, the other 495 reading 0.

    Pins the `/ 496` and the pair set as well as the formula."""
    cells = [(0, 0)] * 11 + [(0, 1)] * 1 + [(1, 0)] * 4 + [(1, 1)] * 4
    idx = np.zeros((20, CATS), dtype=int)
    idx[:, 3] = [a for a, _ in cells]
    idx[:, 20] = [b for _, b in cells]

    pair = (
        0.55 * math.log2(0.55 / (0.60 * 0.75))
        + 0.05 * math.log2(0.05 / (0.60 * 0.25))
        + 0.20 * math.log2(0.20 / (0.40 * 0.75))
        + 0.20 * math.log2(0.20 / (0.40 * 0.25))
    )
    assert pair == pytest.approx(0.16299, abs=5e-6), "the hand derivation itself"
    assert N_PAIRS == 496
    assert redundancy_bits(_onehot_from_indices(idx)) == pytest.approx(
        pair / 496, abs=1e-9
    )


def test_redundancy_finds_dependence_in_one_non_adjacent_pair():
    """CATEGORICALS 0 AND 17 DUPLICATED, ALL OTHERS INDEPENDENT.

    An enumeration of only the adjacent pairs `(j, j+1)` never sees this pair, and
    on a fixture where every pair behaves alike it cannot tell. Soft rows
    (Dirichlet 0.3), not one-hot, and iid across rows.

    The whole 496-pair mean is compared with a reference built one pair at a time
    through a matmul, which shares no code with the module's single `einsum`."""
    n = CLASSES * 160
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.full(CLASSES, 0.3), size=(n, CATS))
    probs[:, 17] = probs[:, 0]

    reference = _reference_pair_matrix(probs)
    pairs = reference[np.triu_indices(CATS, k=1)]
    adjacent = np.array([reference[j, j + 1] for j in range(CATS - 1)])

    # The premises. The planted pair is the largest by far, it is not adjacent,
    # and leaving it out moves the mean by far more than the 1e-9 asserted below.
    # Measured: the planted pair reads 0.1198 bits against a median 0.0012, and the
    # adjacent-only mean sits 2.4e-4 below the all-pair mean -- about 24 times the
    # spread of the unplanted pairs' mean (1.0e-5), and 5 orders above 1e-9.
    assert reference[0, 17] == pytest.approx(pairs.max())
    assert reference[0, 17] > 10.0 * np.median(pairs)
    assert abs(pairs.mean() - adjacent.mean()) > 1e-4

    assert redundancy_bits(probs) == pytest.approx(pairs.mean(), abs=1e-9)


# --- Minor 1: the seed ---


def test_the_floor_requires_its_seed_and_the_seed_matters():
    """A defaulted seed is how a previous milestone shipped every cell sharing one
    bootstrap seed, which correlates noise the reading treats as independent. So
    `seed` has no default, a different seed gives a different floor, and one seed
    is deterministic."""
    independent, _ = _independent_and_duplicated(CLASSES * 80, seed=3)

    with pytest.raises(TypeError, match="seed"):
        redundancy_floor(independent)

    floors = [redundancy_floor(independent, seed=s) for s in (0, 1, 2)]
    assert len(set(floors)) == 3, f"three seeds gave {floors}: the seed is ignored"
    assert redundancy_floor(independent, seed=1) == floors[1], "one seed, one answer"


# --- Minor 2: the collapsed case ---


def test_a_collapsed_code_has_no_ratio_rather_than_a_garbage_one():
    """A fully collapsed code reads redundancy ~0 and floor ~0, so the ratio is
    0/0 -- and the undefined thing must not become a number.

    Measured, because the failure is quieter than a NaN: every row identical means
    the circular shift is the identity, so the floor equals the redundancy BIT
    FOR BIT, to within rounding that can be negative (-4.5e-16 at 11,000 rows). The
    naive ratio is therefore exactly 1.0 -- the single reading that says
    'independent, so the capacity is in use' -- for a code with nothing in it.
    `redundancy_ratio` returns None instead, and None is not NaN: `None > 3`
    raises, where NaN would compare False in both senses at once."""
    for n in (300, 2000):
        collapsed = np.broadcast_to(_dirichlet(1, seed=4)[0], (n, CATS, CLASSES)).copy()
        red, flo = redundancy_bits(collapsed), redundancy_floor(collapsed, seed=0)
        assert abs(red) < RATIO_MIN_FLOOR and abs(flo) < RATIO_MIN_FLOOR
        assert red == flo, "the roll of identical rows is the identity"
        assert redundancy_ratio(red, flo) is None

    # One live categorical is the same case: no pair of categoricals varies.
    one_live = np.broadcast_to(_dirichlet(1, seed=4)[0], (2000, CATS, CLASSES)).copy()
    one_live[:, 5] = _dirichlet(2000, seed=5)[:, 5]
    assert bits_carried(one_live) > 0.1, "it carries something; it has no PAIR"
    assert redundancy_ratio(
        redundancy_bits(one_live), redundancy_floor(one_live, seed=0)
    ) is None


def test_the_ratio_is_a_plain_quotient_where_it_is_defined():
    """And only where it is defined: the boundary is the derived noise tolerance,
    and a non-finite input is a refusal rather than a ratio."""
    independent, _ = _independent_and_duplicated(CLASSES * 80, seed=1)
    red, flo = redundancy_bits(independent), redundancy_floor(independent, seed=0)
    assert redundancy_ratio(red, flo) == red / flo
    assert redundancy_ratio(1.0, 2.0 * RATIO_MIN_FLOOR) == pytest.approx(
        0.5 / RATIO_MIN_FLOOR
    )
    assert redundancy_ratio(1.0, 0.5 * RATIO_MIN_FLOOR) is None
    assert redundancy_ratio(1.0, -1e-16) is None, "a floor can round negative"

    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            redundancy_ratio(bad, 0.1)
        with pytest.raises(ValueError, match="finite"):
            redundancy_ratio(0.1, bad)
