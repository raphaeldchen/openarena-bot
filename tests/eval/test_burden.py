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


def test_the_identity_holds_and_is_reported_as_a_measured_residual():
    """burden(k) == burden(1) + compounding(k) is ALGEBRAIC: the floor cancels.
    In floating point it is a cancellation, so the residual is a few ULPs and
    is RECORDED rather than asserted to be zero -- M3l's floor_bits read
    5.7e-14, not 0.0, and the legend that called it "exactly 0" is still an
    open follow-up.

    THE MUTATION THIS EXISTS FOR: returning a hardcoded 0.0, which would hide
    a genuinely broken decomposition.
    """
    rng = np.random.default_rng(0)
    floor = rng.uniform(100.0, 150.0, size=45)
    curve_one = floor + rng.uniform(0.0, 5.0, size=45)
    curve_k = curve_one + rng.uniform(0.0, 80.0, size=45)
    residual = identity_residual(curve_k, curve_one, floor)
    assert residual < IDENTITY_TOLERANCE
    assert residual == pytest.approx(0.0, abs=1e-12)
    # And it is derived, not hardcoded: scaling the inputs by 1e6 scales the
    # cancellation error with them, so the value must move.
    scaled = identity_residual(curve_k * 1e6, curve_one * 1e6, floor * 1e6)
    assert scaled > residual


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
