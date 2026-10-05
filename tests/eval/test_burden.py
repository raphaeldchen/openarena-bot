"""Unit behaviour of `eval.burden` -- arrays in, numbers out, no checkpoint and
no device.

`burden` itself is NOT torch-free in its import graph (it imports
`probe.position_error`, and `probe` imports torch at module level), so these
tests import torch too. What they do not need is a GPU, a checkpoint or a file:
every number under test is a function of the arrays handed in.
"""

import numpy as np
import pytest

from mbfps.eval.burden import (
    IDENTITY_TOLERANCE,
    REPORTED_H,
    SEEDS_MINIMUM,
    at_horizon,
    burden,
    compounding,
    identity_residual,
    one_step_persistence,
    scored_targets,
    strict_majority,
)


def test_the_constants_are_the_values_the_spec_fixes():
    """THE MUTATION THIS EXISTS FOR: a drifted constant. REPORTED_H is the
    reporting grid, sorted and without repeats, and it ends at the M3 gate's own
    horizon, 45 -- the horizon the script's base control and rulers are read at."""
    assert REPORTED_H == (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
    assert REPORTED_H == tuple(sorted(set(REPORTED_H)))
    assert IDENTITY_TOLERANCE == 1e-9


def test_at_horizon_is_one_indexed_and_refuses_out_of_range():
    """The curves are 0-indexed arrays; the horizon is 1-indexed everywhere in
    this project's records and prose.

    THE MUTATIONS THIS EXISTS FOR: `curve[h]` instead of `curve[h - 1]`, which
    shifts every reported number by one step and breaks no shape; and
    `int(curve[h - 1])` or `round(curve[h - 1])`, which an integer-valued curve
    cannot tell from the identity -- hence the fractions.
    """
    curve = np.array([10.5, 20.25, 30.75])
    assert at_horizon(curve, 1) == 10.5
    assert at_horizon(curve, 3) == 30.75
    with pytest.raises(ValueError, match="horizon step must be in 1..3"):
        at_horizon(curve, 0)
    with pytest.raises(ValueError, match="horizon step must be in 1..3"):
        at_horizon(curve, 4)


def test_burden_is_the_cost_over_the_floor():
    curve = np.array([12.0, 15.0, 22.0])
    floor = np.array([10.0, 10.0, 12.0])
    assert burden(curve, floor) == pytest.approx([2.0, 5.0, 10.0])


def test_burden_keeps_its_sign_when_the_curve_is_below_the_floor():
    """A negative burden is a result, not an error: in the nine cells of
    `runs/m3m_burden` the k=1 curve sits below the floor at 3 to 22 of the 45
    steps (16, 4, 22, 3, 20, 5, 7, 5, 15; the rungs are in no other record). A
    caller that wants a magnitude takes `abs` itself, so `burden` must hand back
    the sign.

    THE MUTATIONS THIS EXISTS FOR: `np.abs(curve_k - floor)` and
    `np.maximum(curve_k - floor, 0.0)`, both of which survive any fixture whose
    entries are all non-negative.
    """
    curve = np.array([12.0, 8.0, 22.0])
    floor = np.array([10.0, 10.0, 25.5])
    assert np.array_equal(burden(curve, floor), np.array([2.0, -2.0, -3.5]))


def test_compounding_is_exactly_zero_at_k_equals_one():
    """`compounding(x, x)` is exactly 0 -- not approximately. A CONTROL WITH A
    KNOWN ANSWER, true by construction: at k=1 both arguments are the same
    curve.

    THE MUTATIONS THIS EXISTS FOR: anything that makes `compounding(x, x)`
    nonzero -- `curve_k + curve_one`, `curve_k.copy()`, `curve_k - 0.5 *
    curve_one`, or an `np.where(curve_k == curve_one, 1, ...)` special case.

    WHAT IT CANNOT PIN is which curve the second argument is. "Compounding
    against the FLOOR instead of the k=1 curve" is inexpressible at this scope:
    there is no floor here, and renaming `compounding`'s second parameter to
    `floor` is the same function. That guard lives at the CALL SITE, where
    `identity_residual` must pass `curve_one`; passing `floor` there is caught
    by the identity tests below, which compare `burden(k)` against
    `burden(1) + compounding(k)`.
    """
    curve_one = np.array([12.0, 15.0, 22.0])
    result = compounding(curve_one, curve_one)
    assert np.array_equal(result, np.zeros(3))


def test_compounding_is_the_cost_of_correcting_less_often():
    curve_one = np.array([12.0, 15.0, 22.0])
    curve_k = np.array([12.0, 19.0, 40.0])
    assert compounding(curve_k, curve_one) == pytest.approx([0.0, 4.0, 18.0])


def test_compounding_keeps_its_sign_when_the_rung_is_below_the_k1_curve():
    """A negative compounding is a result, not an error: in the nine cells of
    `runs/m3m_burden` the k=3 curve sits below the k=1 curve at 7 to 25 of the 45
    steps (25, 13, 9, 11, 11, 14, 11, 15, 7; the rungs are in no other record). A
    caller that wants a magnitude takes `abs` itself, so `compounding` must hand
    back the sign.

    THE MUTATIONS THIS EXISTS FOR: `np.abs(curve_k - curve_one)` and
    `np.maximum(curve_k - curve_one, 0.0)`, both of which survive any fixture
    whose entries are all non-negative.
    """
    curve_one = np.array([12.0, 15.0, 22.5])
    curve_k = np.array([12.0, 19.0, 21.0])
    assert np.array_equal(
        compounding(curve_k, curve_one), np.array([0.0, 4.0, -1.5])
    )


def test_the_identity_is_exact_on_the_magnitudes_this_milestone_measures():
    """burden(k) == burden(1) + compounding(k) holds BIT-FOR-BIT on this
    fixture, and that is a property of its construction rather than luck.

    The floor is 100-150, the k=1 rung sits at most 5 above it and the open
    loop at most 80 above that. Every operand is therefore at least 64, so a
    multiple of 2**-46, and every difference the identity forms is below 128
    (the largest, `curve_k - floor`, is below 85). A multiple of 2**-46 below
    128 is exactly representable, so both summand differences are exact, so is
    their sum, and it equals the whole. That holds for every draw the
    construction can make, not only this seed's, and it does not depend on the
    ratio `curve_k / floor`.

    So a residual of 0.0 is the ANSWER here, not a free pass -- which is why
    the sibling tests below exist: they show `identity_residual` can report
    something other than zero.
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


def test_the_identity_residual_is_the_worst_step_of_the_horizon():
    """`identity_residual` is the LARGEST absolute violation over the horizon,
    and one corrupted step out of 45 is precisely what it exists to surface.
    Every fixture above is a single element or residual-free (all zeros), where
    any reduction agrees.

    Three steps, with the worst in the MIDDLE and a different value at each:
        step 0  2.842170943040401e-14  just past the exact regime -- the
                                       triple `IDENTITY_TOLERANCE` cites
        step 1  2.9802322387695312e-08 wide-spread; the worst
        step 2  0.0                    exact

    THE MUTATIONS THIS EXISTS FOR, each in place of `np.max`: `np.min` (reads
    0.0), `np.median` (2.8e-14), `np.mean` (9.9e-09), `np.sum`, first-element
    only (2.8e-14), last-element only (0.0) and a `[:1]` slice (2.8e-14).

    The last assertion is `==`, not `approx`, and has to be: `np.sum` reads
    2.980235080940474e-08, a relative 9.5e-07 above the max -- inside
    `pytest.approx`'s default 1e-06, so an approximate comparison would pass it.
    """
    curve_k = np.array([238.65922676171652, 167767100.17530185, 120.0])
    curve_one = np.array([110.35124079639952, 77842083.6992047, 110.0])
    floor = np.array([106.90575077107698, 5034536.855956038, 100.0])
    per_step = [
        identity_residual(curve_k[i : i + 1], curve_one[i : i + 1], floor[i : i + 1])
        for i in range(3)
    ]
    assert per_step == [2.842170943040401e-14, 2.9802322387695312e-08, 0.0]
    assert identity_residual(curve_k, curve_one, floor) == 2.9802322387695312e-08


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


@pytest.mark.parametrize(("left_len", "right_len"), [(44, 45), (1, 45)])
def test_the_decomposition_refuses_mismatched_lengths_whichever_curve_is_longer(
    left_len, right_len
):
    """The guard test above puts the LONGER array on the left both times, so a
    guard that only notices that case passes it. NumPy broadcasts a length-1
    array against a length-45 one in either order -- the (1, 45) case is that
    silent broadcast, and (44, 45) is the near miss.

    THE MUTATION THIS EXISTS FOR: `left.size > right.size` in place of
    `left.shape != right.shape`.
    """
    for function in (burden, compounding):
        with pytest.raises(ValueError, match="same length"):
            function(np.zeros(left_len), np.zeros(right_len))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_the_decomposition_refuses_a_nonfinite_value_in_the_right_operand(bad):
    """The guard test above puts every NaN and inf in the LEFT operand,
    `curve_k`. A nonfinite `floor` or `curve_one` is the right operand -- and
    the realistic failure, a curve read from a record that holds NaNs.

    THE MUTATION THIS EXISTS FOR: narrowing the guard to
    `np.isfinite(left).all()`. The bad value is tried at the first, middle and
    last entry.
    """
    curve = np.array([1.0, 2.0, 3.0])
    for position in range(3):
        right = np.array([5.0, 5.0, 5.0])
        right[position] = bad
        with pytest.raises(ValueError, match="finite"):
            burden(curve, right)
        with pytest.raises(ValueError, match="finite"):
            compounding(curve, right)


def test_the_curves_are_coerced_to_float64_before_they_are_subtracted():
    """`checked_pair` promises finite float64. An unsigned-integer curve left
    uncoerced WRAPS instead of going negative -- uint8 1 - 2 is 255 -- and a
    plain list has no `.shape` for the length guard to read.

    THE MUTATIONS THIS EXISTS FOR: dropping `dtype=np.float64` on both sides,
    and dropping the `np.asarray` coercion on either side. Dropping `dtype` on
    ONE side only is an equivalent mutant through the public functions -- NumPy
    upcasts that side against the other's float64 -- so it is not pinned here.
    """
    left = np.array([1, 5], dtype=np.uint8)
    right = np.array([2, 2], dtype=np.uint8)
    for result in (burden(left, right), compounding(left, right)):
        assert result.dtype == np.float64
        assert np.array_equal(result, np.array([-1.0, 3.0]))

    from_list_left = burden([3, 4], np.array([1.0, 1.0]))
    from_list_right = compounding(np.array([3.0, 4.0]), [1, 1])
    for result in (from_list_left, from_list_right):
        assert result.dtype == np.float64
        assert np.array_equal(result, np.array([2.0, 3.0]))


def test_scored_targets_are_the_rows_the_model_is_scored_on():
    """The baseline and the model MUST be scored on the same frames.

    `evaluate_rollout` scores horizon step j+1 against
    privileged[start + context + 1 + j]. The caller hands us
    privileged[start + context : start + need + 1], so row 0 is the last
    CONTEXT frame and rows 1.. are the model's truth. This test reproduces both
    slices from one array and asserts they agree.

    THE MUTATION THIS EXISTS FOR: `window_targets[:-1]` instead of
    `window_targets[1:]`, which scores the baseline one frame early, so the baseline
    and the model are scored on two different frames.
    """
    context, horizon, start = 5, 4, 7
    need = context + horizon
    privileged = np.arange(40.0).reshape(20, 2)
    privileged = np.hstack([privileged, np.zeros((20, 2))])  # (N, 4) targets

    window_targets = privileged[start + context : start + need + 1]
    model_truth = privileged[start + context + 1 : start + need + 1]

    assert np.array_equal(scored_targets(window_targets), model_truth)
    assert scored_targets(window_targets).shape[0] == horizon


def _targets(xy: np.ndarray, headings_degrees) -> np.ndarray:
    """`(N, 4)` targets in `probe_targets`' layout: x, y, sin(angle), cos(angle).

    The angle columns are NOT zero and they VARY from row to row, because the
    position metric must ignore them and a fixture that zeroes them cannot tell
    a Euclidean distance over `x, y` from one over all four columns.
    """
    radians = np.deg2rad(np.asarray(headings_degrees, dtype=np.float64))
    return np.column_stack([xy, np.sin(radians), np.cos(radians)])


def test_one_step_persistence_is_the_true_one_step_displacement():
    """Its error IS the displacement, known from ground truth with no model.

    THE MUTATION THIS EXISTS FOR: comparing each row to row 0 (the t-anchored
    persistence the record already carries) instead of to the row before it.
    That baseline is RIGGED for this comparison -- it has seen h-1 fewer frames
    than the k=1 rung -- and spec 2.2 forbids it by name.

    The second mutation: taking the norm over all four target columns rather than
    the position pair. MEASURED on this fixture, that reads 5.10, 1.41 and 10.10
    where the true displacements are 5, 0 and 10: the heading turns at every
    step, including the one where the agent does not move, so the angle columns
    add 1.00, 1.41 and 1.41. A fixture whose angle columns are zero, or constant,
    reads 5, 0 and 10 either way and cannot tell the two apart.
    """
    # Moves (3, 4) then (0, 0) then (6, 8): displacements 5, 0, 10.
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [9.0, 12.0]])
    window_targets = _targets(xy, [10.0, 70.0, 160.0, 250.0])
    assert one_step_persistence(window_targets) == pytest.approx([5.0, 0.0, 10.0])

    # The rigged baseline would give cumulative distance from row 0 instead.
    rigged = np.linalg.norm(xy[1:] - xy[0], axis=1)
    assert rigged == pytest.approx([5.0, 5.0, 15.0])
    assert one_step_persistence(window_targets) != pytest.approx(rigged)


def test_scored_targets_refuses_what_is_not_a_window_and_returns_float64():
    """A window is (horizon + 1, K) with at least two rows. The two operands of
    the guard are exercised separately: a 1-D array has plenty of elements but
    the wrong rank, a single row has the right rank but nothing to score, and a
    rank-3 array has plenty of rows but is not a window.

    THE MUTATIONS THIS EXISTS FOR: dropping either operand of the guard
    (`< 2` -> `< 1` lets a single row through, returning an empty array that
    every downstream mean turns into NaN), loosening the rank test from
    `!= 2` to `< 2` (which lets a rank-3 array through), and dropping the
    float64 coercion. That last one is pinned here by the RETURN DTYPE and
    nothing else: `scored_targets` only slices, it does not subtract, so the
    unsigned wrap is not expressible at this scope --
    `test_one_step_persistence_coerces_before_it_subtracts` pins it where it can
    occur.
    """
    with pytest.raises(ValueError, match="at least two rows"):
        scored_targets(np.zeros(5))  # rank 1, five elements
    with pytest.raises(ValueError, match="at least two rows"):
        scored_targets(np.zeros((1, 4)))  # rank 2, one row
    with pytest.raises(ValueError, match="at least two rows"):
        scored_targets(np.zeros((3, 4, 2)))  # rank 3, three rows
    with pytest.raises(ValueError, match="at least two rows"):
        scored_targets(np.zeros((0, 4)))

    integers = np.array([[0, 0], [3, 4], [9, 12]], dtype=np.uint8)
    scored = scored_targets(integers)
    assert scored.dtype == np.float64
    assert np.array_equal(scored, np.array([[3.0, 4.0], [9.0, 12.0]]))

    from_list = scored_targets([[0, 0], [3, 4], [9, 12]])
    assert from_list.dtype == np.float64
    assert np.array_equal(from_list, np.array([[3.0, 4.0], [9.0, 12.0]]))


def test_one_step_persistence_coerces_before_it_subtracts():
    """The window arrives from `probe_targets`, but a plain list has no `.shape`
    and an unsigned-integer window WRAPS: uint8 0 - 3 is 253 and 0 - 4 is 252, so
    the step that moves (3, 4) -- a displacement of 5 -- would read
    sqrt(253**2 + 252**2) = 357.09.

    THE MUTATIONS THIS EXISTS FOR: dropping the `np.asarray` coercion (the list
    case), and dropping `dtype=np.float64` from BOTH coercions (the uint8 case).
    Dropping `dtype` from ONE of them is an equivalent mutant: NumPy upcasts the
    uint8 operand against the other's float64, so nothing wraps -- not pinned.
    """
    xy = np.array([[0, 0], [3, 4], [3, 4], [9, 12]], dtype=np.uint8)
    window_targets = np.hstack([xy, np.zeros((4, 2), dtype=np.uint8)])
    # The coordinates INCREASE, so predictor - truth is negative at every step
    # that moves -- exactly where unsigned arithmetic wraps.
    result = one_step_persistence(window_targets)
    assert result.dtype == np.float64
    assert result == pytest.approx([5.0, 0.0, 10.0])

    as_lists = window_targets.astype(float).tolist()
    assert one_step_persistence(as_lists) == pytest.approx([5.0, 0.0, 10.0])


# --- The seed bar and the majority it is read against ------------------------


def test_the_seed_minimum_is_the_value_the_spec_fixes():
    assert SEEDS_MINIMUM == 3


def test_strict_majority_is_computed_not_stored():
    """A fixed SEEDS_REQUIRED=2 is a majority at 3 seeds and a MINORITY at 5.
    M3l's design was reworked for exactly this.

    THE MUTATIONS THIS EXISTS FOR: `return 2`, which passes at 3 seeds and is
    wrong at 1, 5 and 7; and `(n + 1) // 2`, which agrees with `n // 2 + 1` at
    every odd n and returns exactly HALF at every even one -- so only the even
    counts in the loop can see it.
    """
    assert strict_majority(1) == 1
    assert strict_majority(3) == 2
    assert strict_majority(4) == 3
    assert strict_majority(5) == 3
    assert strict_majority(7) == 4
    for n in range(1, 12):
        assert 2 * strict_majority(n) > n


def test_strict_majority_refuses_a_count_with_no_seeds():
    """THE MUTATION THIS EXISTS FOR: dropping the `n < 1` guard, after which
    `strict_majority(0)` is 1 -- a "majority" of nothing."""
    for n in (0, -1, -7):
        with pytest.raises(ValueError, match="at least one seed"):
            strict_majority(n)


# --- M3n removed the motion statistic; the ladder is what is left -----------


def test_the_contaminated_statistic_is_gone_from_the_module():
    """`motion_margin` and Reading H are removed, not deprecated in place.

    A live function whose legend says a reading of -116 means COPIES is the
    same hazard as the "both directions are sound" sentence, which shipped in
    three places M3m's own results refuted. Conditioning was the right fix
    while the records had to stay readable; leaving the function live now that
    a correct replacement exists converts a corrected error into a standing
    trap.

    EVERY NAME THE REMOVAL LISTS, not only the headline ones: the constants
    (`DECISION_H`, `ARMS_REQUIRED`, `CONFIDENCE`, `RESAMPLES`) are what a
    half-finished removal leaves behind, because nothing fails when a constant
    outlives its only reader. `margin_interval` went to `pooling` in M3n's first
    task, so it is listed to keep it from coming back.
    """
    from mbfps.eval import burden as module

    for gone in (
        "motion_margin", "margin_interval", "reading_burden",
        "format_reading_burden", "BurdenInputs", "BurdenStatus", "BurdenArm",
        "_fall_through_rule", "_row", "READING_COLUMNS", "READING_WIDTHS",
        "DECISION_H", "ARMS_REQUIRED", "CONFIDENCE", "RESAMPLES",
    ):
        assert not hasattr(module, gone), f"{gone} must be removed"


def test_the_ladder_survives_the_removal():
    """`burden`, `compounding` and `identity_residual` were right and stay.

    M3n's `deficit` IS `burden`, and `headroom.py` imports `checked_pair` and
    `strict_majority` from here, so a removal that took them would break the
    replacement.
    """
    from mbfps.eval import burden as module

    for kept in (
        "at_horizon", "checked_pair", "burden", "compounding",
        "identity_residual", "scored_targets", "one_step_persistence",
        "strict_majority", "SEEDS_MINIMUM", "REPORTED_H",
        "IDENTITY_TOLERANCE",
    ):
        assert hasattr(module, kept), f"{kept} must survive"
