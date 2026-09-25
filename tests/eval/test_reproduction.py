"""The reproduction rule: the bound, its edges, and the calibration it claims."""
import math

import pytest

from mbfps.eval.reproduction import REPRODUCTION_ULPS, reproduces, reproduction_bound


def test_the_bound_is_sixty_four_ulps_of_the_stored_magnitude():
    assert REPRODUCTION_ULPS == 64
    assert reproduction_bound(200.0) == 1.8189894035458565e-12
    assert reproduction_bound(1.0) == 1.4210854715202004e-14
    assert reproduction_bound(0.5) == 7.105427357601002e-15


def test_the_bound_scales_with_the_magnitude_not_with_the_delta():
    """Doubling the stored value doubles the bound: one absolute tolerance
    cannot serve a position curve near 200 and a loss near 0.4."""
    assert reproduction_bound(512.0) == 2 * reproduction_bound(256.0)


def test_the_measured_worst_case_reproduces_and_leaves_headroom():
    """1.705303e-13 at magnitude 231.9665544559 is frozen_ssl/s1, the worst of
    the nine M3c cells on macOS 27.0: 6.000002 ULPs against a bound of 64."""
    magnitude = 231.9665544559
    assert 1.705303e-13 / math.ulp(magnitude) == pytest.approx(6.0, abs=1e-5)
    assert reproduces(1.705303e-13, magnitude)


def test_a_wrong_device_does_not_reproduce():
    """The discrepancy the gate exists to refuse: 6-12 map units."""
    assert not reproduces(6.0, 231.9665544559)
    assert not reproduces(12.4, 214.9230630703)


def test_exactly_zero_always_reproduces():
    assert reproduces(0.0, 200.0)
    assert reproduces(0.0, 0.0)


def test_a_magnitude_with_no_scale_demands_the_exact_rule():
    """Nothing is granted a tolerance around zero or a non-finite magnitude:
    the bound is 0.0, so only an exact delta passes."""
    assert reproduction_bound(0.0) == 0.0
    assert reproduction_bound(float("nan")) == 0.0
    assert reproduction_bound(float("inf")) == 0.0
    assert not reproduces(1e-300, 0.0)


def test_ulps_zero_is_how_a_caller_asks_for_the_exact_rule():
    assert reproduction_bound(200.0, ulps=0) == 0.0
    assert not reproduces(1e-13, 200.0, ulps=0)
    assert reproduces(0.0, 200.0, ulps=0)


def test_a_non_finite_delta_never_reproduces():
    """`_max_delta` returns inf for a shape mismatch and `anchor_delta` for a
    NaN; both must stay refusals rather than unorderable comparisons.

    A NEGATIVE delta is the case that needs the explicit guard: `inf <= bound`
    and `nan <= bound` are already False under IEEE comparison, but `-inf <=
    bound` and `-5.0 <= bound` are both True. No caller can produce one --
    both helpers return a max of absolute values -- so this pins the
    predicate's contract for its future callers, sign and all.
    """
    assert not reproduces(float("inf"), 200.0)
    assert not reproduces(float("nan"), 200.0)
    assert not reproduces(float("-inf"), 200.0)
    assert not reproduces(-5.0, 200.0), "a finite negative distance is no more a delta than -inf"
    assert not reproduces(-5.0, 0.0)
