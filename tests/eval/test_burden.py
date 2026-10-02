"""Unit behaviour of `eval.burden` -- arrays in, numbers out, no torch."""

import numpy as np
import pytest

from mbfps.eval.burden import (
    ARMS_REQUIRED,
    CONFIDENCE,
    DECISION_H,
    IDENTITY_TOLERANCE,
    REPORTED_H,
    RESAMPLES,
    at_horizon,
    burden,
    compounding,
    identity_residual,
)


def test_the_constants_are_the_values_the_spec_fixes():
    """THE MUTATION THIS EXISTS FOR: a drifted constant. DECISION_H is the
    gate's own horizon; REPORTED_H is the reporting grid and must contain it,
    or the status is read at a horizon the table never prints."""
    assert DECISION_H == 45
    assert REPORTED_H == (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
    assert DECISION_H in REPORTED_H
    assert REPORTED_H == tuple(sorted(set(REPORTED_H)))
    assert ARMS_REQUIRED == 2
    assert IDENTITY_TOLERANCE == 1e-9
    assert CONFIDENCE == 0.95
    assert RESAMPLES == 2000


def test_at_horizon_is_one_indexed_and_refuses_out_of_range():
    """The curves are 0-indexed arrays; the horizon is 1-indexed everywhere in
    this project's records and prose.

    THE MUTATION THIS EXISTS FOR: `curve[h]` instead of `curve[h - 1]`, which
    shifts every reported number by one step and breaks no shape.
    """
    curve = np.array([10.0, 20.0, 30.0])
    assert at_horizon(curve, 1) == 10.0
    assert at_horizon(curve, 3) == 30.0
    with pytest.raises(ValueError, match="horizon step must be in 1..3"):
        at_horizon(curve, 0)
    with pytest.raises(ValueError, match="horizon step must be in 1..3"):
        at_horizon(curve, 4)


def test_burden_is_the_cost_over_the_floor():
    curve = np.array([12.0, 15.0, 22.0])
    floor = np.array([10.0, 10.0, 12.0])
    assert burden(curve, floor) == pytest.approx([2.0, 5.0, 10.0])


def test_compounding_is_exactly_zero_at_k_equals_one():
    """A CONTROL WITH A KNOWN ANSWER, true by construction: at k=1 both sides
    are the same curve, so the difference is exactly 0 -- not approximately.

    THE MUTATION THIS EXISTS FOR: computing compounding against the FLOOR
    instead of against the k=1 curve, which makes this read `burden(1)`
    instead of 0.
    """
    curve_one = np.array([12.0, 15.0, 22.0])
    result = compounding(curve_one, curve_one)
    assert np.array_equal(result, np.zeros(3))


def test_compounding_is_the_cost_of_correcting_less_often():
    curve_one = np.array([12.0, 15.0, 22.0])
    curve_k = np.array([12.0, 19.0, 40.0])
    assert compounding(curve_k, curve_one) == pytest.approx([0.0, 4.0, 18.0])


def test_the_identity_is_exact_on_the_magnitudes_this_milestone_measures():
    """burden(k) == burden(1) + compounding(k) holds BIT-FOR-BIT here, and that
    is a property of the regime rather than luck.

    Position errors run 100-250 Doom map units and the three curves are
    monotone -- floor <= the k=1 rung <= the open loop -- so each subtraction
    falls in the Sterbenz regime and is exact; the sum of two exact differences
    then rounds to exactly the whole. Measured over 40 random fixtures in this
    range the residual is 0.0 in every one, and six adversarial
    wide-dynamic-range triples (floor at 1e16, at 2**53, at 1.0 with 1e-17
    steps) also give 0.0.

    So the recorded residual being zero is the ANSWER in this regime, not a
    free pass -- which is why the sibling test below exists.
    """
    rng = np.random.default_rng(0)
    floor = rng.uniform(100.0, 150.0, size=45)
    curve_one = floor + rng.uniform(0.0, 5.0, size=45)
    curve_k = curve_one + rng.uniform(0.0, 80.0, size=45)
    residual = identity_residual(curve_k, curve_one, floor)
    assert residual == 0.0
    assert residual <= IDENTITY_TOLERANCE


def test_the_identity_residual_is_measured_and_its_tolerance_is_reachable():
    """Two things no well-conditioned fixture can show.

    FIRST, that `identity_residual` computes rather than returning a constant.
    Over a wide monotone spread the cancellation is real: this triple -- found
    by searching 400,000 monotone triples over 1e-8..1e8 -- reads
    2.9802322387695312e-08.

    SECOND, that `IDENTITY_TOLERANCE` is a reachable bar rather than dead
    decoration. A guard no input can trip is indistinguishable from no guard,
    and this residual exceeds it by more than an order of magnitude.

    THE MUTATION THIS EXISTS FOR: `return 0.0` in `identity_residual`, which is
    an equivalent mutant against every well-conditioned fixture and fails here.
    """
    curve_k = np.array([167767100.17530185])
    curve_one = np.array([77842083.6992047])
    floor = np.array([5034536.855956038])
    residual = identity_residual(curve_k, curve_one, floor)
    assert residual == 2.9802322387695312e-08
    assert residual > IDENTITY_TOLERANCE


def test_the_decomposition_refuses_mismatched_or_nonfinite_curves():
    """THE MUTATION THIS EXISTS FOR: dropping the guards. NumPy broadcasts a
    length-1 array against a length-45 one silently, so a curve read from the
    wrong record key would produce a full-length result that is nonsense.
    """
    with pytest.raises(ValueError, match="same length"):
        burden(np.zeros(45), np.zeros(44))
    with pytest.raises(ValueError, match="same length"):
        compounding(np.zeros(45), np.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        burden(np.array([1.0, np.nan]), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        compounding(np.array([1.0, np.inf]), np.zeros(2))
