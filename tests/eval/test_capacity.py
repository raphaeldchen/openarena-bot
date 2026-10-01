"""M3l: is `z` out of room, or was it never asked?"""
from __future__ import annotations

import dataclasses
import itertools
import math
import re
from unittest.mock import patch

import numpy as np
import pytest

from mbfps.eval.capacity import (
    CEILING_BITS, FRAME_CUT, PAIR_CEILING_BITS, RATIO_MIN_FLOOR, READING_COLUMNS,
    READING_WIDTHS, SPARE_CUT, CapacityArm, CapacityInputs, _pair_mutual_information,
    argmax_marginal_bits, bits_carried, bits_interval, capacity_arm, ceiling_bits,
    entropy_bits, episode_stats, floor_bits, format_reading_capacity, live_classes,
    reading_capacity, redundancy_bits, redundancy_floor, redundancy_ratio,
)
from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, CONFIDENCE, RESAMPLES, SEEDS_REQUIRED, BaseControl,
)
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


# ---------------------------------------------------------------------------
# Reading G
# ---------------------------------------------------------------------------

ARMS = ("frozen_ssl", "pixel_ae", "random_vit")
SEEDS_TOTAL = 3
BASE_HOLDS, BASE_FAILS = 0.65, 0.05

SPARE = {"bits": 40.0, "ci_low": 30.0, "ci_high": 50.0}
FULL = {"bits": 150.0, "ci_low": 140.0, "ci_high": 158.0}
FRAMEY = (0.80, 0.70, 0.90)
UNFRAMEY = (0.20, 0.10, 0.30)


def test_the_fixtures_sit_where_their_names_say():
    """Every status test below rests on these four intervals being on the side
    of the cut their names claim. A drifted constant would turn a precedence
    test into one that exercises a different, easier case."""
    assert SPARE["ci_high"] < SPARE_CUT < FULL["ci_low"]
    assert FRAMEY[1] > FRAME_CUT > UNFRAMEY[2]


def _arm(bits: dict, frame: tuple[float, float, float], *, live: int = 1024,
         redundancy: tuple[float, float] = (0.065, 0.064),
         seeds_spare: int | None = None, seeds_frame: int | None = None,
         seeds_total: int = SEEDS_TOTAL) -> CapacityArm:
    """An arm whose bits interval is `bits` and frame interval is `frame`.

    `seeds_*` default to ALL seeds when the corresponding interval clears and
    NONE when it does not -- the 0-or-3 shape most fixtures want. Pass them to
    sit a fixture ON a threshold, which is the only way a bar gets pinned:
    M3k shipped four off-by-one mutations because every fixture was 0 or 3.

    This helper makes the tally itself, so it is for tests of Reading G's
    PRECEDENCE and BARS. The tally's own rule is `capacity_arm`'s, and is tested
    through `capacity_arm` below -- a test that built its arm here and then
    asserted the tally could not fail against a wrong `capacity_arm`.
    """
    share, low, high = frame
    spare = seeds_total if bits["ci_high"] < SPARE_CUT else 0
    framey = seeds_total if low > FRAME_CUT else 0
    return CapacityArm(
        bits=bits["bits"], bits_low=bits["ci_low"], bits_high=bits["ci_high"],
        frame_share=share, frame_low=low, frame_high=high, live=live,
        redundancy_bits=redundancy[0], redundancy_floor=redundancy[1],
        seeds_spare=spare if seeds_spare is None else seeds_spare,
        seeds_frame=framey if seeds_frame is None else seeds_frame,
        seeds_total=seeds_total,
    )


def _base(r2: float, seeds_total: int = SEEDS_TOTAL) -> BaseControl:
    return BaseControl(r2=r2, seeds_clear=seeds_total if r2 > BASE_R2_FLOOR else 0,
                       seeds_total=seeds_total)


def _inputs(arms=None, *, base_r2=BASE_HOLDS, controls=None) -> CapacityInputs:
    arms = arms if arms is not None else {
        a: _arm(FULL, (0.20, 0.10, 0.30)) for a in ARMS
    }
    per_arm = base_r2 if isinstance(base_r2, dict) else dict.fromkeys(arms, base_r2)
    return CapacityInputs(
        arms=arms,
        base={a: _base(per_arm.get(a, BASE_HOLDS)) for a in arms},
        controls=controls if controls is not None else dict.fromkeys(arms, True),
        clusters=24, rows=11221,
    )


def _seed(**overrides) -> dict:
    """One seed's measurement as `capacity_arm` takes it: a spare-looking,
    frame-poor cell with a healthy companion. Override what a test varies."""
    return {"bits": 40.0, "ci_low": 30.0, "ci_high": 50.0, "frame_share": 0.2,
            "frame_low": 0.1, "frame_high": 0.3, "live": 1024,
            "redundancy_bits": 0.065, "redundancy_floor": 0.064, **overrides}


def _table_line(text: str, first_token: str) -> str:
    """The one table row whose first token is `first_token`.

    Selected by token and asserted unique, NOT by indentation: the base-control
    and verdict lines are also indented and also contain every arm name, so an
    indentation filter would find the wrong line whenever the row order changed.
    """
    hits = [ln for ln in text.splitlines() if ln.split()[:1] == [first_token]]
    assert len(hits) == 1, f"expected exactly one row for {first_token}, got {len(hits)}"
    return hits[0]


def _parse_row(line: str) -> dict[str, str]:
    """A table row split back into `READING_COLUMNS` by the shipped widths."""
    out, at = {}, 2
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        out[name] = line[at:at + width].strip()
        at += width
    return out


def test_most_of_the_capacity_idle_reads_spare_and_names_the_objective_lever():
    reading = reading_capacity(_inputs({a: _arm(SPARE, (0.20, 0.10, 0.30)) for a in ARMS}))
    assert reading.status == "SPARE_CAPACITY"
    assert reading.arms_spare == ARMS
    # The lever is NAMED, in the rule's own casing, and the other lever is not:
    # a rule pointing at the wrong lever is the verdict disagreeing with itself.
    assert "OBJECTIVE lever" in reading.rule and "bottleneck" not in reading.rule


def test_a_code_that_re_encodes_the_frame_reads_frame_reencoding():
    reading = reading_capacity(_inputs({a: _arm(FULL, (0.80, 0.70, 0.90)) for a in ARMS}))
    assert reading.status == "FRAME_REENCODING"
    assert reading.arms_frame == ARMS
    assert "OBJECTIVE lever" in reading.rule and "bottleneck" not in reading.rule
    assert "in use" not in reading.rule, (
        "FRAME_REENCODING rests on the frame share, not on bits, so it must not "
        "claim the capacity is in use: not clearing SPARE_CAPACITY is not that, "
        "and `bits_carried` over-reads a redundant code"
    )


def test_full_capacity_carrying_something_else_falls_through_to_capacity_bound():
    """CAPACITY_BOUND is the FALL-THROUGH and its rule must say so. The two
    objective-lever statuses each have to clear an interval; this one is what
    remains, so it is the EASIEST status to reach -- which is backwards, since
    it is the lever M3j's h-vs-z evidence already favours."""
    reading = reading_capacity(_inputs({a: _arm(FULL, (0.20, 0.10, 0.30)) for a in ARMS}))
    assert reading.status == "CAPACITY_BOUND"
    assert reading.arms_spare == () and reading.arms_frame == ()
    assert "BY DEFAULT rather than by evidence" in reading.rule
    assert "bottleneck lever" in reading.rule
    assert "OBJECTIVE lever" not in reading.rule


def test_the_capacity_bound_rule_sends_the_reader_to_both_companions():
    """`bits_carried` upper-bounds the joint information, so a high reading does
    not establish "the capacity is in use" -- the one positive claim this status
    rests on. A CAPACITY_BOUND verdict beside a redundancy ratio far above 1, or
    beside a low `live` count, contradicts itself, so the rule has to name the
    two columns that make a reading interpretable, and they have to be columns
    the table actually prints."""
    reading = reading_capacity(_inputs())
    assert reading.status == "CAPACITY_BOUND"
    for column in ("red_ratio", "live"):
        assert column in reading.rule, f"the rule never names {column}"
        assert column in READING_COLUMNS, f"{column} is not a column of the table"
    assert "upper bound" in reading.rule


def test_a_broken_control_outranks_every_reading():
    """The estimator's own checks gate everything: a reading taken from an
    estimator that missed a known answer is not a weaker reading, it is not a
    reading. Both-true fixture -- the base fails at the same time -- so
    swapping the two checks in the source is caught."""
    reading = reading_capacity(_inputs(
        {a: _arm(SPARE, (0.20, 0.10, 0.30)) for a in ARMS},
        base_r2=BASE_FAILS, controls={**dict.fromkeys(ARMS, True), "pixel_ae": False},
    ))
    assert reading.status == "UNRESOLVED_ESTIMATOR"
    assert reading.controls_failed == ("pixel_ae",)
    assert reading.base_failed == ARMS, "the base failure is REPORTED, not hidden"
    assert reading.arms_spare == () and reading.arms_frame == ()


def test_a_failed_base_outranks_a_reading_but_not_a_broken_control():
    reading = reading_capacity(_inputs(
        {a: _arm(SPARE, (0.20, 0.10, 0.30)) for a in ARMS}, base_r2=BASE_FAILS,
    ))
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.base_failed == ARMS
    assert reading.controls_failed == ()


def _conditions(*, estimator: bool, base: bool, spare: bool, frame: bool) -> CapacityInputs:
    """Inputs in which exactly the named conditions hold, over three arms.

    `spare` / `frame` make EVERY arm clear that sense, so both can be true at
    once -- an arm may clear both, they are different quantities. `base` fails
    TWO arms (holding = ARMS_REQUIRED - 1, one short of the gate's bar); when it
    is off one arm still fails (holding = ARMS_REQUIRED, exactly at the bar), so
    no row is free of a partial base failure.
    """
    arms = {a: _arm(SPARE if spare else FULL, FRAMEY if frame else UNFRAMEY)
            for a in ARMS}
    failed = ("frozen_ssl", "pixel_ae") if base else ("pixel_ae",)
    return _inputs(
        arms, base_r2=dict.fromkeys(failed, BASE_FAILS),
        controls={**dict.fromkeys(ARMS, True), "random_vit": False} if estimator else None,
    )


def _combo_id(combo: tuple[bool, ...]) -> str:
    return "".join(letter if on else "-" for letter, on in zip("EBSF", combo, strict=True))


@pytest.mark.parametrize(
    "estimator, base, spare, frame",
    [pytest.param(*combo, id=_combo_id(combo))
     for combo in itertools.product((False, True), repeat=4)],
)
def test_precedence_over_every_combination_of_the_four_conditions(
    estimator, base, spare, frame,
):
    """All sixteen combinations of {estimator broken, base failed, spare, frame}.

    M3k shipped two inverted checks that 36 tests missed, because every fixture
    made exactly ONE status's condition true and so the order the rule considered
    them never mattered. Here every pair of adjacent statuses has a row where
    both conditions hold:

        ESTIMATOR / BASE   E B . .   (and E B S F)
        BASE / SPARE       . B S .   (and . B S F)
        SPARE / FRAME      . . S F
        FRAME / FALL-THROUGH  . . . F   against   . . . .

    The expected status is spelled out as the chain, not computed by the module.
    """
    reading = reading_capacity(
        _conditions(estimator=estimator, base=base, spare=spare, frame=frame)
    )
    if estimator:
        expected = "UNRESOLVED_ESTIMATOR"
    elif base:
        expected = "UNRESOLVED_BASE"
    elif spare:
        expected = "SPARE_CAPACITY"
    elif frame:
        expected = "FRAME_REENCODING"
    else:
        expected = "CAPACITY_BOUND"
    assert reading.status == expected

    # Every branch reports everything it computed. The two UNRESOLVED statuses
    # take no reading, so no arm votes there.
    voting = not (estimator or base)
    assert reading.controls_failed == (("random_vit",) if estimator else ())
    assert reading.base_failed == (("frozen_ssl", "pixel_ae") if base else ("pixel_ae",))
    assert reading.arms_spare == (ARMS if spare and voting else ())
    assert reading.arms_frame == (ARMS if frame and voting else ())


@pytest.mark.parametrize("status, arms, controls, base_r2, spare, framey", [
    ("UNRESOLVED_ESTIMATOR", {a: _arm(SPARE, (0.8, 0.7, 0.9)) for a in ARMS},
     {**dict.fromkeys(ARMS, True), "pixel_ae": False}, {"pixel_ae": BASE_FAILS}, (), ()),
    ("UNRESOLVED_BASE", {a: _arm(SPARE, (0.8, 0.7, 0.9)) for a in ARMS},
     None, dict.fromkeys(ARMS, BASE_FAILS), (), ()),
    ("SPARE_CAPACITY", {a: _arm(SPARE, (0.8, 0.7, 0.9)) for a in ARMS},
     None, {"pixel_ae": BASE_FAILS}, ARMS, ARMS),
    ("FRAME_REENCODING", {a: _arm(FULL, (0.8, 0.7, 0.9)) for a in ARMS},
     None, {"pixel_ae": BASE_FAILS}, (), ARMS),
    ("CAPACITY_BOUND", {a: _arm(FULL, (0.2, 0.1, 0.3)) for a in ARMS},
     None, {"pixel_ae": BASE_FAILS}, (), ()),
])
def test_every_branch_reports_all_four_of_its_tuples(
    status, arms, controls, base_r2, spare, framey,
):
    """M3k's final review found `arms_down=()` hardcoded in the deciding branch
    surviving 82 tests, because the parametrize that pinned `base_failed` used
    one-directional arms. So every case here carries a partial base failure AND
    arms that clear in both senses where the status allows, and all four
    reported tuples are asserted in all five branches."""
    reading = reading_capacity(_inputs(arms, base_r2=base_r2, controls=controls))
    assert reading.status == status
    assert reading.arms_spare == spare
    assert reading.arms_frame == framey
    expected_base = tuple(sorted(a for a, r in base_r2.items() if r <= BASE_R2_FLOOR))
    assert reading.base_failed == expected_base
    assert reading.controls_failed == (
        () if controls is None else tuple(sorted(a for a, ok in controls.items() if not ok))
    )


@pytest.mark.parametrize("status, build, spare, framey", [
    # Every case: the two tuples DIFFER, so reporting one in place of the other
    # shows. In the parametrize above spare == frame == ARMS, which cannot tell
    # `arms_frame=spare` from `arms_frame=frame`.
    ("SPARE_CAPACITY",
     {"frozen_ssl": (SPARE, UNFRAMEY), "pixel_ae": (SPARE, FRAMEY),
      "random_vit": (FULL, FRAMEY)},
     ("frozen_ssl", "pixel_ae"), ("pixel_ae", "random_vit")),
    # Spare in ONE arm is below the bar: it falls past SPARE_CAPACITY to FRAME,
    # and the sub-bar arm is still reported.
    ("FRAME_REENCODING",
     {"frozen_ssl": (SPARE, UNFRAMEY), "pixel_ae": (FULL, FRAMEY),
      "random_vit": (FULL, FRAMEY)},
     ("frozen_ssl",), ("pixel_ae", "random_vit")),
    # Neither sense reaches the bar, each in a different arm. The fall-through
    # reports what it saw: a hardcoded `()` here survived 82 tests in M3k.
    ("CAPACITY_BOUND",
     {"frozen_ssl": (SPARE, UNFRAMEY), "pixel_ae": (FULL, FRAMEY),
      "random_vit": (FULL, UNFRAMEY)},
     ("frozen_ssl",), ("pixel_ae",)),
])
def test_each_branch_reports_the_tuple_it_computed_for_each_sense(
    status, build, spare, framey,
):
    """Insertion order is the REVERSE of sorted, so a dropped `sorted` shows, and
    one arm fails its base control so the arm that votes is not the arm that
    holds."""
    arms = {a: _arm(*build[a]) for a in reversed(ARMS)}
    reading = reading_capacity(_inputs(arms, base_r2={"random_vit": BASE_FAILS}))
    assert reading.status == status
    assert reading.arms_spare == spare
    assert reading.arms_frame == framey
    assert reading.base_failed == ("random_vit",)
    assert reading.controls_failed == ()


def test_the_failures_are_reported_sorted_whatever_order_the_arms_came_in():
    """Two base failures and two broken controls, each pair given in the REVERSE
    of sorted order, so a dropped `sorted` is a different tuple -- and the rules
    that name them list them in the same order the tuples carry."""
    arms = {a: _arm(FULL, UNFRAMEY) for a in reversed(ARMS)}
    assert list(arms) != sorted(arms)
    base_r2 = {"pixel_ae": BASE_FAILS, "frozen_ssl": BASE_FAILS}

    broken = reading_capacity(_inputs(
        arms, base_r2=base_r2,
        controls={"random_vit": False, "pixel_ae": False, "frozen_ssl": True},
    ))
    assert broken.status == "UNRESOLVED_ESTIMATOR"
    assert broken.controls_failed == ("pixel_ae", "random_vit")
    assert broken.base_failed == ("frozen_ssl", "pixel_ae")
    assert "pixel_ae, random_vit" in broken.rule

    unresolved = reading_capacity(_inputs(arms, base_r2=base_r2))
    assert unresolved.status == "UNRESOLVED_BASE"
    assert unresolved.base_failed == ("frozen_ssl", "pixel_ae")
    assert "(frozen_ssl, pixel_ae failed)" in unresolved.rule


def test_exactly_arms_required_arms_read_and_one_short_does_not():
    """The bar IS Reading G. M3k shipped four off-by-one mutations on exactly
    this because no fixture sat on a threshold."""
    at_bar = {"frozen_ssl": _arm(SPARE, (0.2, 0.1, 0.3)),
              "pixel_ae": _arm(SPARE, (0.2, 0.1, 0.3)),
              "random_vit": _arm(FULL, (0.2, 0.1, 0.3))}
    reading = reading_capacity(_inputs(at_bar))
    assert len(reading.arms_spare) == ARMS_REQUIRED
    assert reading.status == "SPARE_CAPACITY"

    one_short = {**at_bar, "pixel_ae": _arm(FULL, (0.2, 0.1, 0.3))}
    assert len(reading_capacity(_inputs(one_short)).arms_spare) == ARMS_REQUIRED - 1
    assert reading_capacity(_inputs(one_short)).status == "CAPACITY_BOUND"


def test_the_frame_bar_is_arms_required_arms_too():
    """The same bar on the other status. The test above pins only SPARE's, so a
    `len(frame) >= 1` in the FRAME branch would survive it."""
    at_bar = {"frozen_ssl": _arm(FULL, FRAMEY), "pixel_ae": _arm(FULL, FRAMEY),
              "random_vit": _arm(FULL, UNFRAMEY)}
    reading = reading_capacity(_inputs(at_bar))
    assert len(reading.arms_frame) == ARMS_REQUIRED
    assert reading.status == "FRAME_REENCODING"

    one_short = {**at_bar, "pixel_ae": _arm(FULL, UNFRAMEY)}
    reading = reading_capacity(_inputs(one_short))
    assert len(reading.arms_frame) == ARMS_REQUIRED - 1
    assert reading.status == "CAPACITY_BOUND"


def test_an_arm_at_exactly_seeds_required_clears_and_one_short_does_not():
    assert _arm(SPARE, (0.2, 0.1, 0.3), seeds_spare=SEEDS_REQUIRED).clears_spare()
    assert not _arm(SPARE, (0.2, 0.1, 0.3),
                    seeds_spare=SEEDS_REQUIRED - 1).clears_spare()
    assert _arm(FULL, FRAMEY, seeds_frame=SEEDS_REQUIRED).clears_frame()
    assert not _arm(FULL, FRAMEY, seeds_frame=SEEDS_REQUIRED - 1).clears_frame()


@pytest.mark.parametrize("seeds, votes", [
    (SEEDS_REQUIRED - 1, False), (SEEDS_REQUIRED, True), (SEEDS_TOTAL, True),
])
def test_the_reading_counts_an_arm_only_at_seeds_required_seeds(seeds, votes):
    """The method test above pins `clears_*`. This pins that the READING asks
    them: an `if arm.seeds_spare > 0` in `reading_capacity` passes the method
    test untouched. Each status, on the threshold and one either side."""
    spare = reading_capacity(_inputs(
        {a: _arm(SPARE, UNFRAMEY, seeds_spare=seeds) for a in ARMS}
    ))
    assert (spare.status == "SPARE_CAPACITY") is votes
    assert spare.arms_spare == (ARMS if votes else ())

    frame = reading_capacity(_inputs(
        {a: _arm(FULL, FRAMEY, seeds_frame=seeds) for a in ARMS}
    ))
    assert (frame.status == "FRAME_REENCODING") is votes
    assert frame.arms_frame == (ARMS if votes else ())


def test_the_base_gate_needs_arms_required_arms_holding_and_reads_the_tally():
    """The gate sits at `ARMS_REQUIRED` arms holding: one arm above it proceeds,
    one below refuses. And it reads each arm's SEED TALLY, never its r2 -- the
    floor is applied by whoever builds `BaseControl`, and nothing here compares
    `r2` to `BASE_R2_FLOOR`."""
    spare_arms = {a: _arm(SPARE, UNFRAMEY) for a in ARMS}
    at_bar = len(ARMS) - ARMS_REQUIRED
    failing = ARMS[:at_bar]
    reading = reading_capacity(_inputs(spare_arms, base_r2=dict.fromkeys(failing, BASE_FAILS)))
    assert reading.status == "SPARE_CAPACITY", "exactly ARMS_REQUIRED arms hold"
    assert reading.base_failed == failing

    failing = ARMS[:at_bar + 1]
    reading = reading_capacity(_inputs(spare_arms, base_r2=dict.fromkeys(failing, BASE_FAILS)))
    assert reading.status == "UNRESOLVED_BASE", "one arm short of ARMS_REQUIRED hold"
    assert reading.base_failed == failing

    def inputs_with(controls: dict[str, BaseControl]) -> CapacityInputs:
        return CapacityInputs(arms=spare_arms, base=controls,
                              controls=dict.fromkeys(ARMS, True), clusters=24, rows=11221)

    # A high mean r2 with the tally one seed short does not hold; a low r2 with
    # the tally at the bar does. The tally decides.
    tally_short = {a: BaseControl(r2=0.9, seeds_clear=SEEDS_REQUIRED - 1,
                                  seeds_total=SEEDS_TOTAL) for a in ARMS}
    reading = reading_capacity(inputs_with(tally_short))
    assert reading.status == "UNRESOLVED_BASE" and reading.base_failed == ARMS

    tally_met = {a: BaseControl(r2=0.01, seeds_clear=SEEDS_REQUIRED,
                                seeds_total=SEEDS_TOTAL) for a in ARMS}
    reading = reading_capacity(inputs_with(tally_met))
    assert reading.status == "SPARE_CAPACITY" and reading.base_failed == ()


def test_reading_g_refuses_inputs_that_cannot_support_a_reading():
    """Each refusal is a gate that would otherwise fail OPEN: too few arms would
    read CAPACITY_BOUND from nothing, and an arm with no base control or no
    estimator check would vote with nothing gating it."""
    arms = {a: _arm(FULL, UNFRAMEY) for a in ARMS}
    whole = _inputs(arms)

    with pytest.raises(ValueError, match="ARMS_REQUIRED"):
        reading_capacity(_inputs({a: arms[a] for a in ARMS[:ARMS_REQUIRED - 1]}))

    gone = ARMS[0]
    with pytest.raises(ValueError, match="must name exactly"):
        reading_capacity(dataclasses.replace(
            whole, base={a: c for a, c in whole.base.items() if a != gone}))
    with pytest.raises(ValueError, match="must name exactly"):
        reading_capacity(dataclasses.replace(
            whole, controls={a: ok for a, ok in whole.controls.items() if a != gone}))
    with pytest.raises(ValueError, match="must name exactly"):
        reading_capacity(dataclasses.replace(
            whole, base={**whole.base, "stray": _base(BASE_HOLDS)}))
    with pytest.raises(ValueError, match="must name exactly"):
        reading_capacity(dataclasses.replace(
            whole, controls={**whole.controls, "stray": True}))


def test_the_cut_is_on_the_interval_not_the_point_estimate():
    """A point estimate on the clearing side of the cut whose interval straddles
    it must NOT clear. Half the capacity idle is a claim, and a claim needs an
    interval.

    Through `capacity_arm`, because that is where the tally is made. This test
    once built its arm with a helper that applied the cut itself and then
    asserted the helper's answer, so replacing `ci_high < SPARE_CUT` with
    `bits < SPARE_CUT` in the module left it green."""
    straddling = {"bits": 70.0, "ci_low": 60.0, "ci_high": 95.0}
    assert straddling["bits"] < SPARE_CUT < straddling["ci_high"]
    arm = capacity_arm([_seed(**straddling)] * SEEDS_TOTAL)
    assert arm.seeds_spare == 0, "the interval straddles the cut, so nothing clears"
    assert not arm.clears_spare()

    straddling_frame = {"frame_share": 0.6, "frame_low": 0.4, "frame_high": 0.8}
    assert straddling_frame["frame_low"] < FRAME_CUT < straddling_frame["frame_share"]
    arm = capacity_arm([_seed(**straddling_frame)] * SEEDS_TOTAL)
    assert arm.seeds_frame == 0, "the share's interval straddles the cut too"
    assert not arm.clears_frame()


def test_a_seed_clears_only_strictly_past_the_cut():
    """A threshold is pinned only by a fixture sitting ON it. 'Below this' and
    'above this' are strict: an interval that touches the cut has not cleared."""
    on_bits = capacity_arm([_seed(bits=60.0, ci_low=50.0, ci_high=SPARE_CUT)] * SEEDS_TOTAL)
    assert on_bits.seeds_spare == 0
    under = math.nextafter(SPARE_CUT, 0.0)
    assert capacity_arm(
        [_seed(bits=60.0, ci_low=50.0, ci_high=under)] * SEEDS_TOTAL
    ).seeds_spare == SEEDS_TOTAL

    on_frame = capacity_arm(
        [_seed(frame_share=0.7, frame_low=FRAME_CUT, frame_high=0.9)] * SEEDS_TOTAL
    )
    assert on_frame.seeds_frame == 0
    over = math.nextafter(FRAME_CUT, 1.0)
    assert capacity_arm(
        [_seed(frame_share=0.7, frame_low=over, frame_high=0.9)] * SEEDS_TOTAL
    ).seeds_frame == SEEDS_TOTAL


def test_the_tally_is_per_seed_so_an_arm_can_clear_while_its_hull_straddles():
    """The decision is `SEEDS_REQUIRED` seeds each clearing the cut with their
    OWN interval -- counted before anything is combined. The displayed hull is
    the lowest low and highest high, so a seed that straddles widens it past the
    cut; the arm still clears if enough OTHER seeds did. Replacing the tally with
    a test on the hull would read 0 here, and an `any` would read 1 below."""
    clear = _seed()
    straddle = _seed(bits=70.0, ci_low=60.0, ci_high=95.0,
                     frame_share=0.6, frame_low=0.4, frame_high=0.8)
    framey = _seed(frame_share=0.8, frame_low=0.7, frame_high=0.9)

    arm = capacity_arm([clear, clear, straddle])
    assert arm.bits_high == 95.0 > SPARE_CUT, "the hull straddles the cut"
    assert (arm.seeds_spare, arm.seeds_total) == (SEEDS_REQUIRED, 3)
    assert arm.clears_spare()
    assert capacity_arm([clear, straddle, straddle]).seeds_spare == SEEDS_REQUIRED - 1
    assert not capacity_arm([clear, straddle, straddle]).clears_spare()

    arm = capacity_arm([framey, framey, straddle])
    assert arm.frame_low == 0.4 < FRAME_CUT, "the hull straddles the cut"
    assert arm.seeds_frame == SEEDS_REQUIRED and arm.clears_frame()
    assert capacity_arm([framey, straddle, straddle]).seeds_frame == SEEDS_REQUIRED - 1
    assert not capacity_arm([framey, straddle, straddle]).clears_frame()


def test_capacity_arm_summarises_each_field_from_its_own_seeds():
    """Mean for the levels, the conservative hull for the intervals, the lowest
    count for `live` -- on three seeds whose every number differs, so a field
    read from its neighbour (or `min` for `max`) is a different value."""
    arm = capacity_arm([
        _seed(bits=40.0, ci_low=30.0, ci_high=50.0, frame_share=0.2, frame_low=0.1,
              frame_high=0.3, live=1000, redundancy_bits=0.10, redundancy_floor=0.05),
        _seed(bits=60.0, ci_low=55.0, ci_high=70.0, frame_share=0.6, frame_low=0.52,
              frame_high=0.7, live=900, redundancy_bits=0.30, redundancy_floor=0.07),
        _seed(bits=100.0, ci_low=90.0, ci_high=110.0, frame_share=0.4, frame_low=0.35,
              frame_high=0.45, live=1024, redundancy_bits=0.20, redundancy_floor=0.06),
    ])
    assert arm.bits == pytest.approx(200.0 / 3)
    assert (arm.bits_low, arm.bits_high) == (30.0, 110.0)
    assert arm.frame_share == pytest.approx(0.4)
    assert (arm.frame_low, arm.frame_high) == (0.1, 0.7)
    assert arm.live == 900 and isinstance(arm.live, int)
    assert arm.redundancy_bits == pytest.approx(0.20)
    assert arm.redundancy_floor == pytest.approx(0.06)
    assert (arm.seeds_spare, arm.seeds_frame, arm.seeds_total) == (2, 1, 3)


def test_capacity_arm_refuses_a_measurement_it_cannot_read():
    """Non-finite, inverted, and a point estimate outside its own interval --
    each would otherwise pass silently and read as a null, because every
    comparison against NaN is False. `retention.rung_arm` refuses the same
    three for the same reason. Both intervals, both companions, both sides of
    'outside', and every missing key."""
    ok = _seed()
    assert capacity_arm([ok] * SEEDS_REQUIRED).seeds_total == SEEDS_REQUIRED

    for field, value, match in (
        ("bits", float("nan"), "non-finite"),
        ("ci_low", 60.0, "inverted"),
        ("bits", 90.0, "outside its own interval"),
        ("bits", 10.0, "outside its own interval"),
        ("frame_share", float("nan"), "non-finite"),
        ("frame_high", float("inf"), "non-finite"),
        ("frame_low", 0.5, "inverted"),
        ("frame_share", 0.9, "outside its own interval"),
        ("frame_share", 0.05, "outside its own interval"),
        ("redundancy_floor", float("nan"), "non-finite"),
        ("redundancy_bits", float("inf"), "non-finite"),
    ):
        with pytest.raises(ValueError, match=rf"seed index 1\b.*{match}"):
            capacity_arm([ok, {**ok, field: value}, ok])

    with pytest.raises(ValueError, match="at least"):
        capacity_arm([ok] * (SEEDS_REQUIRED - 1))

    for key in ok:
        missing = {k: v for k, v in ok.items() if k != key}
        with pytest.raises(ValueError, match=f"missing {key}"):
            capacity_arm([ok, missing, ok])


def test_the_arm_carries_the_redundancy_companion_and_it_gates_nothing():
    """Task 6 must report the ratio beside `CAPACITY_BOUND`, derived from the
    records, so the arm has to carry both numbers -- as the seed MEANS, like
    `bits`. And they are a companion: wild values change no status."""
    ok = {"bits": 40.0, "ci_low": 30.0, "ci_high": 50.0, "frame_share": 0.2,
          "frame_low": 0.1, "frame_high": 0.3, "live": 1024}
    arm = capacity_arm([
        {**ok, "redundancy_bits": 0.10, "redundancy_floor": 0.05},
        {**ok, "redundancy_bits": 0.30, "redundancy_floor": 0.07},
        {**ok, "redundancy_bits": 0.20, "redundancy_floor": 0.06},
    ])
    assert arm.redundancy_bits == pytest.approx(0.20)
    assert arm.redundancy_floor == pytest.approx(0.06)

    base = _inputs()
    wild = _inputs({a: dataclasses.replace(x, redundancy_bits=9.0, redundancy_floor=0.0)
                    for a, x in base.arms.items()})
    assert reading_capacity(wild) == reading_capacity(base)


@pytest.mark.parametrize("status, kwargs", [
    ("UNRESOLVED_ESTIMATOR", dict(estimator=True, base=False, spare=False, frame=False)),
    ("UNRESOLVED_BASE", dict(estimator=False, base=True, spare=False, frame=False)),
    ("SPARE_CAPACITY", dict(estimator=False, base=False, spare=True, frame=False)),
    ("FRAME_REENCODING", dict(estimator=False, base=False, spare=False, frame=True)),
    ("CAPACITY_BOUND", dict(estimator=False, base=False, spare=False, frame=False)),
])
def test_no_redundancy_value_changes_any_of_the_five_statuses(status, kwargs):
    """The companion is shown beside the verdict and decides nothing, in EVERY
    branch -- a rule that read it only on the fall-through would pass a test
    that only built CAPACITY_BOUND. Two wild shapes: an undefined ratio (a zero
    floor) and an enormous defined one. And the table must print both."""
    base = _conditions(**kwargs)
    assert reading_capacity(base).status == status
    for wild_bits, wild_floor, shown in ((9.0, 0.0, "undefined"), (9.0, 0.001, "9000.000")):
        wild = dataclasses.replace(base, arms={
            a: dataclasses.replace(x, redundancy_bits=wild_bits,
                                   redundancy_floor=wild_floor)
            for a, x in base.arms.items()
        })
        assert reading_capacity(wild) == reading_capacity(base)
        text = format_reading_capacity(reading_capacity(wild), wild)
        assert [_parse_row(_table_line(text, a))["red_ratio"] for a in ARMS] == [shown] * 3


def test_the_bars_follow_retention_when_retention_changes(monkeypatch):
    """`ARMS_REQUIRED`, `SEEDS_REQUIRED` and `BASE_R2_FLOOR` are imported, never
    re-spelled. At the shipped values (2, 2, 0.10) a hardcoded `2` equals the
    import and every other test passes either way, so re-execute the module
    under DIFFERENT values: a fresh copy under another name, as the ceiling's
    test does, so the real module is never reloaded."""
    import importlib.util
    import sys

    import mbfps.eval.capacity as real
    import mbfps.eval.retention as retention

    monkeypatch.setattr(retention, "ARMS_REQUIRED", 3)
    monkeypatch.setattr(retention, "SEEDS_REQUIRED", 3)
    monkeypatch.setattr(retention, "BASE_R2_FLOOR", 0.37)
    spec = importlib.util.spec_from_file_location("_capacity_rebarred", real.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_capacity_rebarred", fresh)
    spec.loader.exec_module(fresh)

    def arm(seeds_spare: int, seeds_frame: int = 0):
        return fresh.CapacityArm(
            bits=40.0, bits_low=30.0, bits_high=50.0, frame_share=0.2, frame_low=0.1,
            frame_high=0.3, live=1024, redundancy_bits=0.065, redundancy_floor=0.064,
            seeds_spare=seeds_spare, seeds_frame=seeds_frame, seeds_total=3,
        )

    # One seed short of the NEW per-arm bar (3), though not of the shipped one (2).
    assert arm(3).clears_spare() and not arm(2).clears_spare()
    assert arm(0, 3).clears_frame() and not arm(0, 2).clears_frame()

    # ...and the arm builder's own refusal of an under-seeded arm.
    with pytest.raises(ValueError, match="SEEDS_REQUIRED=3"):
        fresh.capacity_arm([_seed(), _seed()])
    assert fresh.capacity_arm([_seed()] * 3).seeds_total == 3

    def inputs(arms: dict, base_failed: tuple[str, ...] = ()):
        return fresh.CapacityInputs(
            arms=arms,
            base={a: BaseControl(r2=0.5, seeds_clear=0 if a in base_failed else 3,
                                 seeds_total=3) for a in arms},
            controls=dict.fromkeys(arms, True), clusters=1, rows=1,
        )

    # Two spare arms of three read at the shipped arm bar and not at the new one.
    two_of_three = {"a": arm(3), "b": arm(3), "c": arm(0)}
    assert fresh.reading_capacity(inputs(two_of_three)).status == "CAPACITY_BOUND"
    three_of_three = {"a": arm(3), "b": arm(3), "c": arm(3)}
    assert fresh.reading_capacity(inputs(three_of_three)).status == "SPARE_CAPACITY"

    # The frame bar is the other `ARMS_REQUIRED` use, and the other per-arm bar.
    two_framey = {"a": arm(0, 3), "b": arm(0, 3), "c": arm(0, 0)}
    assert fresh.reading_capacity(inputs(two_framey)).status == "CAPACITY_BOUND"
    three_framey = {"a": arm(0, 3), "b": arm(0, 3), "c": arm(0, 3)}
    assert fresh.reading_capacity(inputs(three_framey)).status == "FRAME_REENCODING"

    # The base gate: two of three holding is enough at the shipped bar, not the new one.
    reading = fresh.reading_capacity(inputs(three_of_three, base_failed=("c",)))
    assert reading.status == "UNRESOLVED_BASE"
    assert "0.37" in reading.rule, "the floor is the imported one"

    with pytest.raises(ValueError, match="ARMS_REQUIRED=3"):
        fresh.reading_capacity(inputs({"a": arm(3), "b": arm(3)}))


def test_the_table_pins_every_column_to_the_arm_it_came_from():
    """This project has shipped a table whose caption disagreed with its
    columns THREE times; the worst printed an `up` tally beside a verdict read
    from the `down` tally, so the natural misreading was the exact opposite of
    the truth. Every column's VALUE is pinned to its field, on rows whose
    numbers all differ -- including the redundancy ratio, whose neighbours
    (`live`, `frame`) are the natural swap -- so a neighbour swap is caught."""
    arms = {"frozen_ssl": _arm(SPARE, (0.21, 0.11, 0.31), live=1000,
                               redundancy=(0.065, 0.060)),
            "pixel_ae": _arm(FULL, (0.82, 0.72, 0.92), live=1024,
                             redundancy=(0.900, 0.072)),
            "random_vit": _arm(FULL, (0.43, 0.33, 0.53), live=900,
                               redundancy=(0.240, 0.075))}
    inputs = _inputs(arms)
    text = format_reading_capacity(reading_capacity(inputs), inputs)

    assert _parse_row(_table_line(text, "arm")) == dict(zip(
        READING_COLUMNS, READING_COLUMNS, strict=True
    )), "the header must sit over the same columns the rows are parsed by"

    for name, clears in (("frozen_ssl", "spare"), ("pixel_ae", "frame"),
                         ("random_vit", "no")):
        arm = arms[name]
        row = _parse_row(_table_line(text, name))
        assert row["arm"] == name
        assert float(row["bits"]) == pytest.approx(arm.bits, abs=5e-5)
        assert float(row["ci_low"]) == pytest.approx(arm.bits_low, abs=5e-5)
        assert float(row["ci_high"]) == pytest.approx(arm.bits_high, abs=5e-5)
        assert float(row["frame"]) == pytest.approx(arm.frame_share, abs=5e-5)
        assert int(row["live"]) == arm.live
        # A plain quotient, written out here rather than through the module's
        # `redundancy_ratio`, so the column cannot agree with itself by sharing
        # a wrong formula. 1.083 / 12.5 / 3.2: none is 1, so a numerator and
        # denominator swapped in the table would read 0.923 / 0.08 / 0.3125.
        assert float(row["red_ratio"]) == pytest.approx(
            arm.redundancy_bits / arm.redundancy_floor, abs=5e-4
        )
        assert row["clears"] == clears

    # The legend DEFINES the column, so it must define the direction the values
    # above were pinned in: `redundancy / floor`, not the reverse.
    (legend,) = [ln for ln in text.splitlines() if ln.split()[:1] == ["red_ratio"]]
    assert legend.strip().startswith("red_ratio = redundancy / its circular-shift floor")


def test_the_table_rows_are_sorted_by_arm_whatever_order_the_arms_came_in():
    arms = {a: _arm(FULL, UNFRAMEY) for a in reversed(ARMS)}
    assert list(arms) != sorted(arms)
    inputs = _inputs(arms)
    text = format_reading_capacity(reading_capacity(inputs), inputs)
    rows = [ln.split()[0] for ln in text.splitlines() if ln.split()[:1] and ln.split()[0] in ARMS]
    assert rows == sorted(ARMS)


def test_an_undefined_ratio_prints_undefined_and_never_a_bare_one():
    """A collapsed code reads redundancy == floor, so the NAIVE ratio is exactly
    1.0 -- the one reading that says 'independent, so the capacity is in use' --
    for a code with nothing in it. `redundancy_ratio` returns None there; the
    table must print something nobody can read as a ratio, and must not crash.
    Beside it, an arm whose ratio really is 1.000 prints it, so the two are
    distinguishable on the page."""
    arms = {
        "frozen_ssl": _arm(FULL, UNFRAMEY, live=3, redundancy=(0.0, 0.0)),
        "pixel_ae": _arm(FULL, UNFRAMEY, redundancy=(0.064, 0.064)),
        "random_vit": _arm(FULL, UNFRAMEY, live=0, redundancy=(-4.5e-16, -4.5e-16)),
    }
    inputs = _inputs(arms)
    text = format_reading_capacity(reading_capacity(inputs), inputs)
    ratios = {a: _parse_row(_table_line(text, a))["red_ratio"] for a in ARMS}
    assert ratios == {"frozen_ssl": "undefined", "pixel_ae": "1.000",
                      "random_vit": "undefined"}
    lives = {a: int(_parse_row(_table_line(text, a))["live"]) for a in ARMS}
    assert lives == {"frozen_ssl": 3, "pixel_ae": 1024, "random_vit": 0}
    assert "1.000" not in _table_line(text, "frozen_ssl")


def test_an_arm_that_clears_both_senses_is_labelled_so_in_the_table():
    """The two senses are different quantities, so an arm can clear both -- the
    SPARE_CAPACITY case in the parametrize above has every arm doing it. The
    reading reports both tuples; a column that printed only `spare` would read,
    beside `arms_frame`, as an arm that did not clear the frame cut."""
    arms = {"frozen_ssl": _arm(SPARE, FRAMEY), "pixel_ae": _arm(SPARE, UNFRAMEY),
            "random_vit": _arm(FULL, FRAMEY)}
    inputs = _inputs(arms)
    reading = reading_capacity(inputs)
    text = format_reading_capacity(reading, inputs)
    clears = {a: _parse_row(_table_line(text, a))["clears"] for a in ARMS}
    assert clears == {"frozen_ssl": "spare+frame", "pixel_ae": "spare",
                      "random_vit": "frame"}
    # ...and each label agrees with the tuple the reading reports for that sense.
    assert {a for a, c in clears.items() if "spare" in c} == set(reading.arms_spare)
    assert {a for a, c in clears.items() if "frame" in c} == set(reading.arms_frame)


def test_the_base_and_estimator_lines_report_each_arm_as_it_was_given():
    """The two lines under the table are read by name, so a line zipped against
    the wrong arm order would put one arm's control beside another's name."""
    base = {"frozen_ssl": BaseControl(r2=0.65, seeds_clear=3, seeds_total=3),
            "pixel_ae": BaseControl(r2=0.05, seeds_clear=1, seeds_total=3),
            "random_vit": BaseControl(r2=0.31, seeds_clear=2, seeds_total=3)}
    inputs = CapacityInputs(
        arms={a: _arm(FULL, UNFRAMEY) for a in ARMS}, base=base,
        controls={"frozen_ssl": True, "pixel_ae": True, "random_vit": False},
        clusters=24, rows=11221,
    )
    text = format_reading_capacity(reading_capacity(inputs), inputs)

    (base_line,) = [ln for ln in text.splitlines() if ln.split()[:2] == ["base", "control"]]
    parsed = {m[1]: (float(m[2]), int(m[3]), int(m[4])) for m in re.finditer(
        r"(\w+) r2=([+-][\d.]+) (\d+)/(\d+)", base_line)}
    assert parsed == {"frozen_ssl": (0.65, 3, 3), "pixel_ae": (0.05, 1, 3),
                      "random_vit": (0.31, 2, 3)}

    (check_line,) = [ln for ln in text.splitlines() if ln.split()[:2] == ["estimator", "checks"]]
    assert dict(re.findall(r"(\w+)=(ok|BROKEN)", check_line)) == {
        "frozen_ssl": "ok", "pixel_ae": "ok", "random_vit": "BROKEN"}


def test_the_verdict_line_prints_the_status_and_the_rule_it_was_given():
    for kwargs in (dict(estimator=True, base=False, spare=False, frame=False),
                   dict(estimator=False, base=False, spare=True, frame=False),
                   dict(estimator=False, base=False, spare=False, frame=False)):
        inputs = _conditions(**kwargs)
        reading = reading_capacity(inputs)
        verdict = format_reading_capacity(reading, inputs).splitlines()[-1]
        assert verdict == (
            f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}"
        )


def test_each_rule_states_the_numbers_it_was_decided_on():
    """The rule is the sentence the verdict is read by, so every number in it comes
    from the inputs or from a constant -- the same discipline the caption has.
    Four arms, so a count differs from the arm total, and every constant patched
    to a value nothing else prints: a literal 80 or a literal 'of 3' in a rule
    reads right at the shipped shape and wrong everywhere else."""
    import mbfps.eval.capacity as capacity

    def read(arms, **kwargs):
        inputs = _inputs(arms, **kwargs)
        with patch.object(capacity, "CEILING_BITS", 246.0), \
             patch.object(capacity, "SPARE_CUT", 123.0), \
             patch.object(capacity, "FRAME_CUT", 0.75), \
             patch.object(capacity, "BASE_R2_FLOOR", 0.33), \
             patch.object(capacity, "ARMS_REQUIRED", 3), \
             patch.object(capacity, "SEEDS_REQUIRED", 3):
            return reading_capacity(inputs)

    names = ("a_arm", "b_arm", "c_arm", "d_arm")
    three = ("a_arm", "b_arm", "c_arm")

    spare = read({a: _arm(SPARE if a in three else FULL, UNFRAMEY) for a in names})
    assert spare.status == "SPARE_CAPACITY"
    assert ("below 123 of the 246-bit ceiling in 3 of 4 arms (a_arm, b_arm, c_arm), "
            "each in at least 3 seeds") in spare.rule

    frame = read({a: _arm(FULL, FRAMEY if a in three else UNFRAMEY) for a in names})
    assert frame.status == "FRAME_REENCODING"
    assert ("sits above 0.75 in 3 of 4 arms (a_arm, b_arm, c_arm), each in at "
            "least 3 seeds") in frame.rule

    unresolved = read({a: _arm(FULL, UNFRAMEY) for a in names},
                      base_r2=dict.fromkeys(three, BASE_FAILS))
    assert unresolved.status == "UNRESOLVED_BASE"
    assert ("only 1 of 4 arms clear enc(t) -> position at r2 0.33 "
            "(a_arm, b_arm, c_arm failed)") in unresolved.rule

    broken = read({a: _arm(FULL, UNFRAMEY) for a in names},
                  controls={"a_arm": True, "b_arm": False, "c_arm": False, "d_arm": True})
    assert broken.status == "UNRESOLVED_ESTIMATOR"
    assert "missed a known answer in b_arm, c_arm;" in broken.rule

    bound = read({a: _arm(SPARE if a in ("a_arm", "b_arm") else FULL, UNFRAMEY)
                  for a in names})
    assert bound.status == "CAPACITY_BOUND"
    assert "clears in 3 arms" in bound.rule, "ARMS_REQUIRED, as imported"


def test_the_caption_uses_the_inputs_and_the_derived_ceiling_not_literals():
    """`_inputs()` always builds clusters=24, rows=11221, three arms, three
    seeds -- so a caption printing those as literals would pass every other
    test. Built directly, with every number different and every constant the
    caption or legend prints patched to something else."""
    import mbfps.eval.capacity as capacity

    names = ("frozen_ssl", "pixel_ae", "random_vit", "clip")
    arms = {a: _arm(FULL, (0.2, 0.1, 0.3), live=77, seeds_total=5) for a in names}
    inputs = CapacityInputs(
        arms=arms, base={a: _base(BASE_HOLDS, 5) for a in names},
        controls=dict.fromkeys(names, True), clusters=7, rows=123,
    )
    with patch.object(capacity, "CEILING_BITS", 320.0), \
         patch.object(capacity, "SPARE_CUT", 160.0), \
         patch.object(capacity, "FRAME_CUT", 0.75), \
         patch.object(capacity, "BASE_R2_FLOOR", 0.33), \
         patch.object(capacity, "RATIO_MIN_FLOOR", 1e-3), \
         patch.object(capacity, "ARMS_REQUIRED", 3), \
         patch.object(capacity, "SEEDS_REQUIRED", 4), \
         patch.object(RSSMConfig, "z_cats", 4), patch.object(RSSMConfig, "z_classes", 8):
        text = format_reading_capacity(reading_capacity(inputs), inputs)
    caption = text.splitlines()[0]
    assert "7 clusters" in caption and "123 rows" in caption
    assert "24 clusters" not in caption and "11221 rows" not in caption
    assert "320" in caption and "160" in caption
    assert "above 0.75" in caption and "0.50" not in caption
    assert "of 4 arms" in caption and "of 5 seeds" in caption
    assert "of 3 arms" not in caption and "of 3 seeds" not in caption
    assert "both in 4 of 5 seeds and 3 of 4 arms" in caption, (
        "the bars are the imported SEEDS_REQUIRED and ARMS_REQUIRED, patched here "
        "to 4 and 3 because a literal 2 reads right at the shipped values"
    )
    assert "must clear r2 0.33" in text and "0.10" not in text

    (legend,) = [ln for ln in text.splitlines() if ln.split()[:1] == ["red_ratio"]]
    assert "0.001" in legend, "the undefined-ratio tolerance is the module's"
    assert "of 32 " in text and "1024" not in text, "live is out of z_cats * z_classes"
    assert "the 3-arm bar" in text, "the legend's bar is ARMS_REQUIRED, not a literal"


def test_arms_that_disagree_on_the_seed_count_have_no_true_caption():
    """'2 of N seeds' would be true of some rows and false of others."""
    arms = {"frozen_ssl": _arm(FULL, UNFRAMEY, seeds_total=3),
            "pixel_ae": _arm(FULL, UNFRAMEY, seeds_total=3),
            "random_vit": _arm(FULL, UNFRAMEY, seeds_total=5)}
    inputs = CapacityInputs(
        arms=arms, base={a: _base(BASE_HOLDS, arms[a].seeds_total) for a in arms},
        controls=dict.fromkeys(arms, True), clusters=24, rows=11221,
    )
    with pytest.raises(ValueError, match="share one seeds_total"):
        format_reading_capacity(reading_capacity(inputs), inputs)


def test_reading_columns_and_widths_stay_the_same_length():
    assert len(READING_COLUMNS) == len(READING_WIDTHS)
    assert READING_COLUMNS == ("arm", "bits", "ci_low", "ci_high", "frame", "live",
                               "red_ratio", "clears")


def test_every_column_is_wider_than_the_widest_thing_it_prints():
    """The cells are right-aligned and UNSEPARATED, so a value as wide as its
    column butts against its neighbour. The widest value each column can carry:
    the full ceiling in `bits`, 1024 live columns, `undefined`, `spare+frame`."""
    widest = {"arm": "random_vit", "bits": f"{160.0:.4f}", "ci_low": f"{160.0:.4f}",
              "ci_high": f"{160.0:.4f}", "frame": f"{1.0:.4f}", "live": "1024",
              "red_ratio": "undefined", "clears": "spare+frame"}
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        assert len(widest[name]) < width, f"{name} needs more than {width}"
        assert len(name) < width
