"""Unit behaviour of `eval.burden` -- arrays in, numbers out, no checkpoint and
no device.

`burden` itself is NOT torch-free in its import graph (it imports
`probe.position_error`, and `probe` imports torch at module level), so these
tests import torch too. What they do not need is a GPU, a checkpoint or a file:
every number under test is a function of the arrays handed in.
"""

import dataclasses
import itertools

import numpy as np
import pytest

from mbfps.eval.burden import (
    ARMS_REQUIRED,
    CONFIDENCE,
    DECISION_H,
    IDENTITY_TOLERANCE,
    READING_COLUMNS,
    READING_WIDTHS,
    REPORTED_H,
    RESAMPLES,
    SEEDS_MINIMUM,
    BurdenArm,
    BurdenInputs,
    BurdenStatus,
    at_horizon,
    burden,
    compounding,
    format_reading_burden,
    identity_residual,
    margin_interval,
    motion_margin,
    one_step_persistence,
    reading_burden,
    scored_targets,
    strict_majority,
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
    `runs/m3_study_v2` the k=1 curve sits below the floor at 3 to 22 of the 45
    steps. A caller that wants a magnitude takes `abs` itself, so `burden` must
    hand back the sign.

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
    `runs/m3_study_v2` the k=3 curve sits below the k=1 curve at 7 to 25 of the
    45 steps. A caller that wants a magnitude takes `abs` itself, so
    `compounding` must hand back the sign.

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
    """`_checked_pair` promises finite float64. An unsigned-integer curve left
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
    `window_targets[1:]`, which scores the baseline one frame early and makes
    `motion_margin` a comparison of two different frames.
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


def test_motion_margin_is_positive_when_the_prior_beats_stillness():
    """Positive means one prior step beats assuming no motion."""
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [9.0, 12.0]])
    window_targets = _targets(xy, [10.0, 70.0, 160.0, 250.0])
    # Displacements are 5, 0, 10. A prior that errs by 2, 1, 3 beats stillness
    # at steps 1 and 3 and loses at step 2, where the agent did not move.
    curve_one = np.array([2.0, 1.0, 3.0])
    assert motion_margin(window_targets, curve_one) == pytest.approx([3.0, -1.0, 7.0])


def test_motion_margin_refuses_a_curve_that_is_not_the_scored_length():
    """THE MUTATION THIS EXISTS FOR: dropping the guard. A length-5 curve against
    a length-4 baseline never broadcasts: NumPy rejects it with its OWN
    ValueError, so a dropped guard is caught only because the message differs.
    The case that really does broadcast silently is a length-1 curve, which
    `test_motion_margin_refuses_a_broadcastable_curve_and_a_nonfinite_value`
    pins.
    """
    window_targets = np.zeros((5, 4))
    with pytest.raises(ValueError, match="same length"):
        motion_margin(window_targets, np.zeros(5))


def test_the_interval_clusters_on_episodes_and_requires_its_seed():
    """The resampling unit is the EPISODE, not the window: 229 windows over 24
    episodes, and consecutive Doom frames are near-duplicates, so a
    window-level bootstrap returns an interval several times too narrow.

    THE MUTATION THIS EXISTS FOR, TWICE OVER: resampling windows instead of
    episodes narrows the interval; and `seed: int = 0` lets every cell share
    one seed, which is how a previous milestone shipped nine identical draws.
    """
    # SIX episodes, not two. MEASURED: with two episodes the resample admits
    # only THREE distinct replicate values, so all 20 seeds I tried return
    # byte-identical bounds and the seed assertion below is dead. Six episodes
    # give 216 distinct replicates and no other seed reproduces seed 0's bounds.
    rng = np.random.default_rng(0)
    window_margin = np.vstack([
        rng.normal(mean, 0.05, size=(20, 3))
        for mean in (6.0, 4.0, 2.0, -2.0, -4.0, -6.0)
    ])
    groups = np.repeat(np.arange(6), 20)

    point, low, high = margin_interval(
        window_margin, groups, h=1, resamples=400, seed=0
    )
    assert point == pytest.approx(window_margin[:, 0].mean(), abs=1e-12)
    # Clustered on episodes the interval spans the spread of episode means; a
    # window-level bootstrap would collapse it near the mean.
    assert high - low > 5.0

    # The fixture must discriminate, and this test asserts that about itself:
    # a resample with too few clusters yields too few distinct replicates for
    # any percentile to choose between, and every assertion below then holds
    # for the wrong reason.
    replicates = {
        margin_interval(window_margin, groups, h=1, resamples=400, seed=s)[1:]
        for s in range(8)
    }
    assert len(replicates) > 1, "the bounds do not move with the seed at all"

    # Different seeds move the BOUNDS but never the point estimate.
    other = margin_interval(window_margin, groups, h=1, resamples=400, seed=1)
    assert other[0] == pytest.approx(point, abs=1e-12)
    assert (other[1], other[2]) != (low, high)
    # The same seed twice is identical.
    again = margin_interval(window_margin, groups, h=1, resamples=400, seed=0)
    assert (again[1], again[2]) == (low, high)

    # `seed` has no default.
    import inspect
    sig = inspect.signature(margin_interval)
    assert sig.parameters["seed"].default is inspect.Parameter.empty
    assert sig.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    # `resamples` DOES have a default, and it is the module constant.
    assert sig.parameters["resamples"].default == RESAMPLES


def test_the_interval_refuses_what_it_cannot_cluster():
    """`windows.episode` is null on a record whose ladder carried no
    clustering, and the record's own comment says a reader must REFUSE rather
    than treat every window as its own episode.

    THE MUTATION THIS EXISTS FOR: falling back to `np.arange(n)` for missing
    labels, which silently converts an episode bootstrap into a window one.
    """
    window_margin = np.zeros((6, 3))
    with pytest.raises(ValueError, match="one label per window"):
        margin_interval(window_margin, np.array([0, 0, 1]), h=1, resamples=10, seed=0)
    with pytest.raises(ValueError, match="at least two episodes"):
        margin_interval(
            window_margin, np.zeros(6, dtype=int), h=1, resamples=10, seed=0
        )

    # The record's own null. `windows.episode` is None when the ladder carried
    # no clustering, and a fallback to arange(n) here would silently convert
    # this into the window-level bootstrap the docstring rules out.
    with pytest.raises(ValueError, match="one label per window"):
        margin_interval(window_margin, None, h=1, resamples=10, seed=0)


def test_the_recorded_confidence_matches_what_the_estimator_takes():
    """CONFIDENCE describes `percentile_interval`; it does not configure it.

    THE MUTATION THIS EXISTS FOR: changing `percentile_interval`'s percentiles
    without changing CONFIDENCE, which would leave every record naming a level
    the estimator did not take -- the defect M3l shipped and then fixed.
    """
    from mbfps.eval.pooling import percentile_interval

    replicates = np.arange(10001, dtype=np.float64)
    low, high, _ = percentile_interval(replicates)
    tail = (1.0 - CONFIDENCE) / 2.0
    assert low == pytest.approx(np.percentile(replicates, 100 * tail))
    assert high == pytest.approx(np.percentile(replicates, 100 * (1 - tail)))


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


def test_motion_margin_refuses_a_broadcastable_curve_and_a_nonfinite_value():
    """The brief's length test uses a length-5 curve against a length-4
    baseline, which NumPy rejects with its OWN ValueError -- so a dropped guard
    is caught only because the message differs. A length-1 curve is the real
    hazard: it broadcasts against any baseline without complaint, and a k=1
    curve read from the wrong record key looks exactly like that.

    The other half is a non-finite value. A NaN in the k=1 curve (the right
    operand) or in the ground-truth window (which makes the baseline NaN) would
    otherwise flow into the per-window mean and surface as a NaN interval with
    no indication of which input was bad.

    THE MUTATION THIS EXISTS FOR: replacing the shared `_checked_pair` call
    with a bare length comparison, which keeps the length test green and loses
    the finiteness guard.
    """
    window_targets = np.zeros((5, 4))  # horizon 4, so the baseline has length 4
    for length in (1, 3):
        with pytest.raises(ValueError, match="same length"):
            motion_margin(window_targets, np.zeros(length))

    for bad in (np.nan, np.inf, -np.inf):
        for position in range(4):
            curve_one = np.ones(4)
            curve_one[position] = bad
            with pytest.raises(ValueError, match="finite"):
                motion_margin(window_targets, curve_one)

    corrupt = np.zeros((5, 4))
    corrupt[2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        motion_margin(corrupt, np.ones(4))

    # A plain list is a legitimate curve, and the result is float64.
    margin = motion_margin(window_targets, [1, 2, 3, 4])
    assert margin.dtype == np.float64
    assert np.array_equal(margin, np.array([-1.0, -2.0, -3.0, -4.0]))


def test_the_interval_is_the_percentile_interval_of_the_clustered_replicate_means():
    """The reduction, the column, the seed and the resample count, pinned
    against an oracle assembled from the two Task-1 pieces the interval is
    defined in terms of.

    The brief's interval test reads only `h=1`, so a hard-coded column 0 passes
    it, and its fixture never distinguishes a mean from a median. This one reads
    `h=3` of four columns, uses episodes of UNEQUAL size (3, 7, 4, 9, 5 windows)
    so that the window-pooled mean the replicates take differs from the mean of
    episode means, and then asserts about itself that each alternative reading
    lands on different bounds. MEASURED: the closest alternative (400 -> 2000
    resamples) moves a bound by 0.404, a different seed by 0.615, and every other
    alternative by more than 2.6; the point estimate differs from the mean of
    episode means by 2.26. The assertion floor is 0.1, a quarter of the closest.

    THE MUTATIONS THIS EXISTS FOR: `h - 1` -> `h` or a hard-coded column;
    `.mean()` -> `.median()` or `.sum()` in the replicate; `seed` or `resamples`
    not reaching the bootstrap; and a point estimate taken as the mean of episode
    means rather than of the windows.
    """
    from mbfps.eval.pooling import episode_bootstrap, percentile_interval

    rng = np.random.default_rng(3)
    sizes = (3, 7, 4, 9, 5)
    episode_mean = np.array([5.0, -1.0, 2.0, -4.0, 1.0])
    column_scale = np.array([1.0, -0.5, 2.0, 0.25])  # columns are not interchangeable
    window_margin = np.vstack([
        episode_mean[episode] * column_scale + rng.normal(0.0, 0.3, size=(size, 4))
        for episode, size in enumerate(sizes)
    ])
    groups = np.repeat(np.arange(5), sizes)
    h, resamples, seed = 3, 400, 5

    def oracle(column, labels, draws, draw_seed, reduce=np.mean):
        replicates = np.array([
            reduce(column[index]) for index in episode_bootstrap(labels, draws, draw_seed)
        ])
        low, high, _se = percentile_interval(replicates)
        return low, high

    column = window_margin[:, h - 1]
    expected = oracle(column, groups, resamples, seed)
    point, low, high = margin_interval(
        window_margin, groups, h=h, resamples=resamples, seed=seed
    )
    assert point == pytest.approx(column.mean(), abs=1e-12)
    assert (low, high) == pytest.approx(expected, abs=1e-12)

    # The fixture must discriminate: each alternative lands on other bounds.
    alternatives = {
        "h=1": oracle(window_margin[:, 0], groups, resamples, seed),
        "h=2": oracle(window_margin[:, 1], groups, resamples, seed),
        "h=4": oracle(window_margin[:, 3], groups, resamples, seed),
        "median": oracle(column, groups, resamples, seed, reduce=np.median),
        "sum": oracle(column, groups, resamples, seed, reduce=np.sum),
        "seed 0": oracle(column, groups, resamples, 0),
        "resamples 2000": oracle(column, groups, 2000, seed),
        "windows as episodes": oracle(column, np.arange(column.size), resamples, seed),
    }
    for name, bounds in alternatives.items():
        assert max(abs(bounds[0] - expected[0]), abs(bounds[1] - expected[1])) > 0.1, name

    episode_means = [column[groups == episode].mean() for episode in range(5)]
    assert abs(point - np.mean(episode_means)) > 2.0


def test_the_interval_refuses_a_malformed_margin_a_bad_resample_count_and_a_bad_horizon():
    """Each remaining guard, with the operand that fires it exercised alone.

    A groups array of shape (6, 1) has the right SIZE and the wrong RANK, so only
    the rank operand of the label guard rejects it. A horizon of 0 or -1 is not
    refused by NumPy at all -- `h - 1` is then -1 or -2, which index the LAST and
    the SECOND-TO-LAST column -- so only the lower bound stands between it and a
    silently wrong interval; a horizon past the end is caught by the upper bound
    (a missing one surfaces as an IndexError, which is not the ValueError
    asserted here). The two edges that must WORK are exercised too, so a bound
    moved one step inward also fails.

    THE MUTATIONS THIS EXISTS FOR: dropping the rank check on `window_margin`;
    dropping the rank operand of the label guard; `resamples < 1` -> `< 0`; and
    either bound of `1 <= h <= horizon` dropped or moved inward by one.
    """
    window_margin = np.arange(18.0).reshape(6, 3)
    groups = np.repeat([0, 1], 3)
    run = {"resamples": 10, "seed": 0}

    margin_interval(window_margin, groups, h=1, **run)  # first column
    margin_interval(window_margin, groups, h=3, **run)  # last column

    with pytest.raises(ValueError, match="windows, horizon"):
        margin_interval(window_margin[:, 0], groups, h=1, **run)
    with pytest.raises(ValueError, match="one label per window"):
        margin_interval(window_margin, groups.reshape(6, 1), h=1, **run)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="resamples must be"):
            margin_interval(window_margin, groups, h=1, resamples=bad, seed=0)
    for bad in (0, -1, 4):
        with pytest.raises(ValueError, match="horizon step must be"):
            margin_interval(window_margin, groups, h=bad, **run)


def test_the_interval_refuses_a_nonfinite_margin_wherever_it_sits():
    """A NaN in the column read at `h` used to return `(nan, nan, nan)` with no
    refusal. That triple is what Reading H's verdict is read from, so a silent
    one is a silent wrong verdict. MEASURED on this fixture before the guard
    existed: a NaN at (0, 0) gave `(nan, nan, nan)` at `h=1`, while one at (3, 1)
    or (5, 2) gave a finite interval -- the corrupt cell was silently ignored.

    The positions are the first, a middle and the last cell of the matrix. At
    `h=1` only the first lies in the column that is read; the middle and the last
    lie in columns that are NOT, so they pin that the refusal covers the whole
    array rather than the one column the mean is taken over.

    THE MUTATIONS THIS EXISTS FOR: dropping the finiteness guard, and narrowing
    it to the column read at `h` (or to the first row).
    """
    window_margin = np.arange(18.0).reshape(6, 3)
    groups = np.repeat([0, 1], 3)
    run = {"h": 1, "resamples": 10, "seed": 0}

    # The control: the same call on the unpoisoned array is finite, so each
    # refusal below is attributable to the one cell that was changed.
    assert np.isfinite(margin_interval(window_margin, groups, **run)).all()

    for bad in (np.nan, np.inf, -np.inf):
        for row, column in ((0, 0), (3, 1), (5, 2)):
            poisoned = window_margin.copy()
            poisoned[row, column] = bad
            with pytest.raises(ValueError, match="finite"):
                margin_interval(poisoned, groups, **run)


def test_the_interval_accepts_the_plain_lists_a_json_record_hands_over():
    """`windows.episode` and the per-window margins come out of a JSON record as
    Python lists, which have no `.ndim` for the guards to read.

    THE MUTATIONS THIS EXISTS FOR: dropping either `np.asarray` coercion at the
    top of `margin_interval`. Dropping `dtype=np.float64` from the margin one is
    an equivalent mutant -- an integer column's mean is the same float -- so it
    is not pinned.
    """
    rng = np.random.default_rng(1)
    window_margin = rng.normal(0.0, 1.0, size=(12, 3))
    groups = np.repeat(np.arange(4), 3)

    from_arrays = margin_interval(window_margin, groups, h=2, resamples=50, seed=7)
    from_lists = margin_interval(
        window_margin.tolist(), groups.tolist(), h=2, resamples=50, seed=7
    )
    assert from_lists == from_arrays


# --- Reading H: the records and the seed bar -------------------------------


def _arm(arm="pixel_ae", seed=0, margin=3.0, low=1.0, high=5.0, **over):
    """A cell that passes every control, so each test perturbs one thing."""
    fields = dict(
        arm=arm, seed=seed, margin=margin, margin_low=low, margin_high=high,
        burden_by_k={1: 2.0, 3: 6.0, 5: 11.0, 15: 30.0, 45: 90.0},
        compounding_by_k={1: 0.0, 3: 4.0, 5: 9.0, 15: 28.0, 45: 88.0},
        identity_residual=1.4e-13, open_loop_divergence=0.0,
        k_one_is_floor=False, displacement_median=40.0, floor_median=12.0,
        clusters=24, rows=229,
    )
    fields.update(over)
    return BurdenArm(**fields)


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


@pytest.mark.parametrize(
    "over, ok",
    [
        ({}, True),
        ({"identity_residual": IDENTITY_TOLERANCE}, True),
        ({"identity_residual": float(np.nextafter(IDENTITY_TOLERANCE, 1.0))}, False),
        ({"identity_residual": 1e-6}, False),
        ({"identity_residual": -1e-6}, False),
        ({"identity_residual": float("nan")}, False),
        ({"open_loop_divergence": 3.5e-4}, False),
        ({"open_loop_divergence": 1e-12}, False),
        ({"open_loop_divergence": -1e-12}, False),
        ({"open_loop_divergence": float("nan")}, False),
        ({"k_one_is_floor": True}, False),
    ],
    ids=[
        "all-controls-pass", "residual-at-tolerance", "residual-one-ulp-over",
        "residual-large", "residual-negative", "residual-nan",
        "divergence-large", "divergence-1e-12", "divergence-negative",
        "divergence-nan", "k1-is-floor",
    ],
)
def test_controls_ok_is_every_known_answer_hit(over, ok):
    """Each case perturbs ONE control on a cell that passes the rest, so a
    failure is attributable to the field that was changed.

    THE MUTATIONS THIS EXISTS FOR: dropping any one of the three conjuncts of
    `controls_ok`; `<=` -> `<` on the tolerance (the residual-at-tolerance case
    sits exactly on the bar); turning the divergence test into a tolerance
    (1e-12 is a nonzero divergence, and the k=45 rung must reproduce the record
    BITWISE); and writing a control as "not failed" instead of "passed", which
    lets NaN through.

    `identity_residual()` returns a max of absolute values and is never
    negative, so the negative-residual case does not model a real producer: it
    pins that the property itself reads a magnitude, for a hand-built cell.
    """
    assert _arm(**over).controls_ok == ok


@pytest.mark.parametrize(
    "displacement, floor, ok",
    [
        (40.0, 12.0, True),
        (12.0000001, 12.0, True),
        (12.0, 12.0, False),
        (9.0, 12.0, False),
        (float("nan"), 12.0, False),
    ],
    ids=["moves-far", "just-above", "equal", "below", "nan"],
)
def test_base_ok_needs_true_motion_strictly_above_the_readout_error(
    displacement, floor, ok
):
    """THE MUTATIONS THIS EXISTS FOR: `>` -> `>=` (the equal case: a median
    displacement exactly at the floor cannot be told from noise) and a
    constant `return True`. NaN must fail, so a "not below" rewrite is caught
    too."""
    cell = _arm(displacement_median=displacement, floor_median=floor)
    assert cell.base_ok == ok


def test_a_cell_refuses_an_interval_whose_low_end_is_above_its_high_end():
    """The two decisive statuses are exclusive per cell because
    `margin_low <= margin_high`, so the record refuses to be built without it
    rather than leaving it to the producer.

    THE MUTATIONS THIS EXISTS FOR: dropping the `__post_init__` guard, and
    `>` -> `>=`, which would also refuse a ZERO-width interval -- legal, since
    a bootstrap whose replicates all agree produces one.
    """
    with pytest.raises(ValueError, match="pixel_ae seed 4.*ci_low"):
        _arm(arm="pixel_ae", seed=4, low=2.0, high=1.0)
    assert _arm(low=1.5, high=1.5).margin_high == 1.5


def test_the_reading_records_are_frozen():
    """THE MUTATION THIS EXISTS FOR: dropping `frozen=True` from any of the
    three, after which a status could be edited after it was decided."""
    cell = _arm()
    inputs = BurdenInputs(cells={("pixel_ae", 0): cell}, decision_h=DECISION_H, ks=(1, 45))
    status = BurdenStatus(
        status="INDETERMINATE", rule="r", arms_motion=(), arms_copies=(),
        seeds_total={"pixel_ae": 1},
    )
    for record, field in ((cell, "margin"), (inputs, "decision_h"), (status, "status")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, field, 0)


# --- Reading H: the statuses and their precedence --------------------------

ARMS = ("frozen_ssl", "pixel_ae", "random_vit")

# The three places an interval can sit against the two bars. Margins are given
# so the point estimate lies inside its own interval.
CLEARS_MOTION = dict(margin=3.0, low=1.0, high=5.0)
CLEARS_COPIES = dict(margin=-3.0, low=-5.0, high=-1.0)
STRADDLES = dict(margin=1.0, low=-2.0, high=4.0)


def _inputs(cells, **over):
    fields = dict(
        cells={(c.arm, c.seed): c for c in cells}, decision_h=DECISION_H,
        ks=(1, 3, 5, 15, 45),
    )
    fields.update(over)
    return BurdenInputs(**fields)


def _nine(**over):
    return [_arm(arm=a, seed=s, **over) for a in ARMS for s in (0, 1, 2)]


def _with(cells, arm, seed, **over):
    """`cells` with the one cell `(arm, seed)` rebuilt with `over` applied."""
    return [
        dataclasses.replace(c, **over) if (c.arm, c.seed) == (arm, seed) else c
        for c in cells
    ]


def _arm_cells(arm, *shapes):
    """One arm, one seed per shape in `shapes`, seeds counted from 0."""
    return [_arm(arm=arm, seed=seed, **shape) for seed, shape in enumerate(shapes)]


def _by_arm(**shape_per_arm):
    """Three seeds per arm, every seed of an arm carrying the arm's one shape."""
    return [
        cell for arm, shape in shape_per_arm.items()
        for cell in _arm_cells(arm, shape, shape, shape)
    ]


def test_predicts_motion_when_the_whole_interval_clears_zero():
    """Cells go in in reverse, so the tally comes out sorted only if the
    reading sorts it: the rule prints the arms, and the table is byte for
    byte."""
    reading = reading_burden(_inputs(list(reversed(_nine(**CLEARS_MOTION)))))
    assert reading.status == "PREDICTS_MOTION"
    assert reading.arms_motion == ARMS
    assert f"({', '.join(ARMS)})" in reading.rule
    assert reading.arms_copies == ()
    assert reading.seeds_total == {arm: 3 for arm in ARMS}


def test_copies_when_the_whole_interval_is_at_or_below_zero():
    reading = reading_burden(_inputs(_nine(**CLEARS_COPIES)))
    assert reading.status == "COPIES"
    assert reading.arms_copies == ARMS
    assert reading.arms_motion == ()


def test_an_interval_straddling_zero_falls_through():
    """And the sentence must SAY it is a fall-through. M3k's INDISTINGUISHABLE
    overclaimed through three wordings before the review caught it.

    THE MUTATIONS THIS EXISTS FOR: dropping either phrase from the
    `INDETERMINATE` rule. Case matters: the test reads the lowercase words, so
    a capitalised `FALL-THROUGH` does not satisfy it.
    """
    reading = reading_burden(_inputs(_nine(**STRADDLES)))
    assert reading.status == "INDETERMINATE"
    assert reading.arms_motion == ()
    assert reading.arms_copies == ()
    assert "fall-through" in reading.rule
    assert "by default rather than by evidence" in reading.rule


def test_a_margin_of_exactly_zero_at_the_high_end_reads_copies():
    """The bar is `margin_high <= 0`, so a high end of exactly 0 clears COPIES.

    THE MUTATION THIS EXISTS FOR: `< 0` instead of `<= 0`, which would send an
    exactly-zero cell to INDETERMINATE.
    """
    reading = reading_burden(_inputs(_nine(margin=-1.0, low=-2.0, high=0.0)))
    assert reading.status == "COPIES"
    assert reading.arms_copies == ARMS


def test_a_margin_of_exactly_zero_at_the_low_end_does_not_read_motion():
    """The other side of the same edge: `margin_low > 0` is strict, so a low
    end of exactly 0 clears nothing.

    THE MUTATION THIS EXISTS FOR: `>= 0` instead of `> 0`, which would let a
    low end of exactly zero establish PREDICTS_MOTION.
    """
    reading = reading_burden(_inputs(_nine(margin=2.0, low=0.0, high=4.0)))
    assert reading.status == "INDETERMINATE"
    assert reading.arms_motion == ()


def test_the_two_decisive_statuses_can_never_both_clear():
    """Mutually exclusive because every interval has `margin_low <= margin_high`
    and the cell refuses to be built otherwise -- asserted THROUGH
    `reading_burden` on 500 random configurations rather than claimed. M3l's
    spec claimed exclusivity for two tallies on DIFFERENT quantities, where a
    single arm could clear both.

    No arm may sit in both tallies, and the status must follow the tallies. The
    arm tallies are also recomputed here by the plain "more than half of the
    seeds" rule, so a threshold that drifts shows up as well.

    THE MUTATIONS THIS EXISTS FOR: a copies predicate satisfiable together
    with the motion one (`>= 0` for `<= 0` on `margin_high`), which puts an
    arm in both tallies; and any drift in the per-arm or per-reading bar,
    which the recomputation sees.

    What it does NOT cover: with THREE arms and `ARMS_REQUIRED == 2`, two
    decisive statuses cannot both reach the bar (2 + 2 > 3). With four arms
    they could, and PREDICTS_MOTION is checked first.
    """
    rng = np.random.default_rng(0)
    seen = {"PREDICTS_MOTION": 0, "COPIES": 0, "INDETERMINATE": 0}
    for _ in range(500):
        cells = []
        for arm in ARMS:
            centre = rng.uniform(-4.0, 4.0)
            for seed in range(int(rng.integers(3, 8))):
                mid = centre + rng.normal(0.0, 1.0)
                half = rng.uniform(0.0, 3.0)
                cells.append(_arm(
                    arm=arm, seed=seed, margin=mid, low=mid - half, high=mid + half,
                ))
        reading = reading_burden(_inputs(cells))
        seen[reading.status] += 1

        assert not set(reading.arms_motion) & set(reading.arms_copies)
        want_motion, want_copies = [], []
        for arm in ARMS:
            mine = [c for c in cells if c.arm == arm]
            if 2 * sum(c.margin_low > 0.0 for c in mine) > len(mine):
                want_motion.append(arm)
            if 2 * sum(c.margin_high <= 0.0 for c in mine) > len(mine):
                want_copies.append(arm)
        assert reading.arms_motion == tuple(want_motion)
        assert reading.arms_copies == tuple(want_copies)
        if len(want_motion) >= ARMS_REQUIRED:
            assert reading.status == "PREDICTS_MOTION"
        elif len(want_copies) >= ARMS_REQUIRED:
            assert reading.status == "COPIES"
        else:
            assert reading.status == "INDETERMINATE"

    # The property is vacuous if the draw never reaches a status.
    assert all(count >= 25 for count in seen.values()), seen


@pytest.mark.parametrize(
    "setting",
    ["would-read-motion", "would-read-copies", "also-stationary", "also-short-arm"],
)
@pytest.mark.parametrize(
    "broken",
    [
        {"identity_residual": 1e-6},
        {"open_loop_divergence": 3.5e-4},
        {"k_one_is_floor": True},
    ],
    ids=["identity", "open-loop", "k1-floor"],
)
def test_one_broken_control_outranks_every_other_status(setting, broken):
    """A reading from an estimator that missed a known answer is not a weaker
    reading; it is not a reading.

    Each setting is one the OTHER statuses would claim if the control were
    ignored: `would-read-motion` and `would-read-copies` are decisive on every
    cell; `also-stationary` makes the broken cell itself fail the base
    control, so UNREADABLE would claim it; `also-short-arm` leaves another arm
    with two seeds, so UNREADABLE would claim it for that reason. The last two
    are what pin the ORDER of the two guards: with a clean fixture, moving the
    control check below the base check changes nothing.

    THE MUTATIONS THIS EXISTS FOR: `controls_ok` returning True, and checking
    the controls after the base or seed guard or after the tallies.
    """
    cells = {
        "would-read-motion": lambda: _nine(**CLEARS_MOTION),
        "would-read-copies": lambda: _nine(**CLEARS_COPIES),
        "also-stationary": lambda: _nine(**CLEARS_MOTION),
        "also-short-arm": lambda: [
            c for c in _nine(**CLEARS_MOTION) if (c.arm, c.seed) != ("pixel_ae", 2)
        ],
    }[setting]()
    over = dict(broken)
    if setting == "also-stationary":
        over["displacement_median"] = 9.0
    cells = _with(cells, "frozen_ssl", 1, **over)

    reading = reading_burden(_inputs(cells))
    assert reading.status == "UNRESOLVED_CONTROL"
    assert "frozen_ssl seed 1" in reading.rule
    assert "frozen_ssl seed 0" not in reading.rule
    assert reading.arms_motion == ()
    assert reading.arms_copies == ()
    assert reading.seeds_total == {
        "frozen_ssl": 3, "pixel_ae": 2 if setting == "also-short-arm" else 3,
        "random_vit": 3,
    }


def test_every_broken_cell_is_named_not_only_the_first():
    """Cells go in in reverse, so the names come out sorted only if the reading
    sorts them."""
    cells = _with(_nine(**CLEARS_MOTION), "frozen_ssl", 0, identity_residual=1e-6)
    cells = _with(cells, "random_vit", 2, k_one_is_floor=True)
    reading = reading_burden(_inputs(list(reversed(cells))))
    assert reading.status == "UNRESOLVED_CONTROL"
    assert "in frozen_ssl seed 0, random_vit seed 2:" in reading.rule
    assert "pixel_ae" not in reading.rule


def test_the_control_check_and_its_rule_use_the_tolerance_the_module_holds(monkeypatch):
    """A residual of 1e-8 is past today's tolerance and inside the patched one,
    so the same cell is a missed control under one and a hit under the other,
    and the sentence must print the tolerance the check used.

    THE MUTATIONS THIS EXISTS FOR: `controls_ok` comparing against a literal
    instead of `IDENTITY_TOLERANCE`, and the `UNRESOLVED_CONTROL` rule printing
    a literal where it interpolates the constant.
    """
    cells = _with(_nine(**CLEARS_MOTION), "pixel_ae", 1, identity_residual=1e-8)
    assert reading_burden(_inputs(cells)).status == "UNRESOLVED_CONTROL"

    monkeypatch.setattr("mbfps.eval.burden.IDENTITY_TOLERANCE", 1e-7)
    assert reading_burden(_inputs(cells)).status == "PREDICTS_MOTION"

    cells = _with(cells, "pixel_ae", 1, identity_residual=1e-6)
    reading = reading_burden(_inputs(cells))
    assert reading.status == "UNRESOLVED_CONTROL"
    assert "within 1e-07" in reading.rule
    assert "1e-09" not in reading.rule


@pytest.mark.parametrize(
    "shape", [CLEARS_MOTION, CLEARS_COPIES], ids=["would-read-motion", "would-read-copies"]
)
def test_a_failed_base_control_is_unreadable_and_names_the_cell(shape):
    """If the agent barely moved, motion_margin is at best zero and COPIES
    would be read for a reason that has nothing to do with the objective. The
    guard is not about direction, so both decisive statuses are tried: on every
    other cell the interval is decisive, so ignoring the base control would
    read PREDICTS_MOTION or COPIES.

    THE MUTATIONS THIS EXISTS FOR: `base_ok` returning True, and tallying the
    decisive statuses before the base guard.
    """
    cells = _with(
        _nine(**shape), "random_vit", 1, displacement_median=9.0, floor_median=12.0
    )
    reading = reading_burden(_inputs(cells))
    assert reading.status == "UNREADABLE"
    assert "random_vit seed 1" in reading.rule
    assert "random_vit seed 0" not in reading.rule
    assert "displacement" in reading.rule
    assert reading.arms_motion == ()
    assert reading.arms_copies == ()


@pytest.mark.parametrize("kept", [1, 2])
def test_an_arm_short_of_three_seeds_is_refused_not_tallied(kept):
    """At one seed `strict_majority(1) == 1`, so a single lucky cell would
    establish an arm -- the M3j trap of printing a row that clears beside a
    verdict that cannot. At two, both seeds clearing is a majority of two.
    Every kept cell is decisive, so tallying whatever is present would read
    PREDICTS_MOTION.

    THE MUTATION THIS EXISTS FOR: deleting the `short` term from the
    `UNREADABLE` guard.
    """
    cells = [
        c for c in _nine(**CLEARS_MOTION)
        if not (c.arm == "pixel_ae" and c.seed >= kept)
    ]
    reading = reading_burden(_inputs(cells))
    assert reading.status == "UNREADABLE"
    assert f"pixel_ae carries {kept} seed(s), fewer than {SEEDS_MINIMUM}" in reading.rule
    assert "frozen_ssl carries" not in reading.rule
    assert reading.seeds_total == {"frozen_ssl": 3, "pixel_ae": kept, "random_vit": 3}


def test_unreadable_names_every_cause_it_has():
    """Two stationary cells and two short arms, in reverse insertion order: every
    one is named, and each list comes out sorted only if the reading sorts it."""
    cells = _with(_nine(**CLEARS_MOTION), "frozen_ssl", 2, displacement_median=5.0)
    cells = _with(cells, "pixel_ae", 0, displacement_median=5.0)
    cells = [
        c for c in cells
        if not (c.arm == "frozen_ssl" and c.seed == 1)
        and not (c.arm == "random_vit" and c.seed >= 1)
    ]
    reading = reading_burden(_inputs(list(reversed(cells))))
    assert reading.status == "UNREADABLE"
    assert "in frozen_ssl seed 2, pixel_ae seed 0," in reading.rule
    assert (
        "frozen_ssl carries 2 seed(s), fewer than 3; "
        "random_vit carries 1 seed(s), fewer than 3"
    ) in reading.rule
    assert "pixel_ae carries" not in reading.rule


@pytest.mark.parametrize(
    "shape, status, tally",
    [
        (CLEARS_MOTION, "PREDICTS_MOTION", "arms_motion"),
        (CLEARS_COPIES, "COPIES", "arms_copies"),
    ],
)
def test_two_arms_are_enough_and_one_is_not(shape, status, tally):
    """THE MUTATIONS THIS EXISTS FOR: the arm bar written as `>= 3` (the
    two-arm case then falls through) or `>= 1` (the one-arm case then
    establishes a status). The tally is reported even when the bar is missed,
    so a reader can see which arm cleared alone."""
    two = reading_burden(_inputs(_by_arm(
        frozen_ssl=shape, pixel_ae=shape, random_vit=STRADDLES,
    )))
    assert two.status == status
    assert getattr(two, tally) == ("frozen_ssl", "pixel_ae")

    one = reading_burden(_inputs(_by_arm(
        frozen_ssl=shape, pixel_ae=STRADDLES, random_vit=STRADDLES,
    )))
    assert one.status == "INDETERMINATE"
    assert getattr(one, tally) == ("frozen_ssl",)


def test_arms_that_clear_in_opposite_directions_do_not_make_a_status():
    reading = reading_burden(_inputs(_by_arm(
        frozen_ssl=CLEARS_MOTION, pixel_ae=CLEARS_COPIES, random_vit=STRADDLES,
    )))
    assert reading.status == "INDETERMINATE"
    assert reading.arms_motion == ("frozen_ssl",)
    assert reading.arms_copies == ("pixel_ae",)


# --- Reading H: what the INDETERMINATE sentence says ------------------------
#
# The sentence ships in `burden.txt`, so it is specification. Its first draft
# said "the motion_margin interval straddles 0", which is false whenever an arm
# cleared cleanly -- and INDETERMINATE arises in three different ways, only one
# of which can have every interval straddle. The true condition is that neither
# tally reached `ARMS_REQUIRED`, so the sentence reports the tallies themselves.

FALL_THROUGH_WORDS = (
    "This status is the fall-through, not a bar that was cleared, so it arrived "
    "by default rather than by evidence"
)


def _says_nothing_about_straddling(rule):
    """The first draft's claim and its consequent. Neither is true once an arm
    has cleared, and neither is needed when none has."""
    assert "straddl" not in rule, rule
    assert "could not be shown" not in rule, rule


def test_the_fall_through_sentence_says_so_when_no_arm_clears_either_way():
    """Nothing cleared: both tallies are empty and the sentence says so. The
    third arm is MIXED -- one seed each of motion, copies and straddling -- so
    its intervals do not all straddle either, and the sentence still may not say
    they do.

    THE MUTATIONS THIS EXISTS FOR: the situation clause taken from the wrong
    branch (an empty tally read as "short of the bar", or as a split), and the
    first draft's wording that the interval straddles 0.
    """
    cells = (
        _arm_cells("frozen_ssl", STRADDLES, STRADDLES, STRADDLES)
        + _arm_cells("pixel_ae", STRADDLES, STRADDLES, STRADDLES)
        + _arm_cells("random_vit", CLEARS_MOTION, CLEARS_COPIES, STRADDLES)
    )
    reading = reading_burden(_inputs(cells))
    assert reading.status == "INDETERMINATE"
    assert (reading.arms_motion, reading.arms_copies) == ((), ())
    rule = reading.rule
    assert f"neither decisive status reaches {ARMS_REQUIRED} arms at horizon {DECISION_H}" in rule
    assert "motion is cleared by 0 of 3 arms (none) and copies by 0 of 3 arms (none)" in rule
    assert "no arm has a strict majority of its seeds clearing 0 in either direction" in rule
    assert "short of the bar" not in rule
    assert "opposite directions" not in rule
    _says_nothing_about_straddling(rule)
    assert FALL_THROUGH_WORDS in rule


@pytest.mark.parametrize(
    "shape, tallies",
    [
        (CLEARS_MOTION, "motion is cleared by 1 of 3 arms (frozen_ssl) and copies by 0 of 3 arms (none)"),
        (CLEARS_COPIES, "motion is cleared by 0 of 3 arms (none) and copies by 1 of 3 arms (frozen_ssl)"),
    ],
    ids=["motion", "copies"],
)
def test_the_fall_through_sentence_names_the_one_arm_that_cleared_and_how_far_short(
    shape, tallies
):
    """One arm clears cleanly and the other two straddle. That arm's interval
    does NOT straddle 0, so a sentence saying the interval straddles is false
    here; what is true is that the reading is one arm short of the bar.

    THE MUTATIONS THIS EXISTS FOR: the first draft's wording, which names no
    arm; the two tallies printed against each other's labels; the shortfall
    written as `ARMS_REQUIRED` rather than `ARMS_REQUIRED - len(cleared)`; and a
    `cleared` that reads only `arms_motion`, which the copies case sees.
    """
    reading = reading_burden(_inputs(_by_arm(
        frozen_ssl=shape, pixel_ae=STRADDLES, random_vit=STRADDLES,
    )))
    assert reading.status == "INDETERMINATE"
    assert (reading.arms_motion, reading.arms_copies) == (
        (("frozen_ssl",), ()) if shape is CLEARS_MOTION else ((), ("frozen_ssl",))
    )
    rule = reading.rule
    assert tallies in rule
    assert (
        "every arm that clears does so in the same direction, and the reading "
        "falls short of the bar by 1 arm(s)"
    ) in rule
    assert "opposite directions" not in rule
    assert "no arm has a strict majority" not in rule
    _says_nothing_about_straddling(rule)
    assert FALL_THROUGH_WORDS in rule


def test_the_fall_through_sentence_says_the_arms_disagree_when_they_clear_opposite_ways():
    """`frozen_ssl` clears motion and `pixel_ae` clears copies; neither interval
    straddles 0. That is a different situation from one arm short of the bar --
    the evidence points both ways -- and the sentence has to tell them apart.

    THE MUTATIONS THIS EXISTS FOR: the first draft's wording; the situation
    clause selected on `arms_motion or arms_copies` where it must need BOTH,
    which prints "short of the bar" over a split reading; and the two tallies
    printed against each other's labels.
    """
    reading = reading_burden(_inputs(_by_arm(
        frozen_ssl=CLEARS_MOTION, pixel_ae=CLEARS_COPIES, random_vit=STRADDLES,
    )))
    assert reading.status == "INDETERMINATE"
    assert (reading.arms_motion, reading.arms_copies) == (("frozen_ssl",), ("pixel_ae",))
    rule = reading.rule
    assert (
        "motion is cleared by 1 of 3 arms (frozen_ssl) and copies by 1 of 3 arms (pixel_ae)"
    ) in rule
    assert (
        "the arms clear in opposite directions, so the reading is split, not merely short"
    ) in rule
    assert "short of the bar" not in rule
    assert "no arm has a strict majority" not in rule
    _says_nothing_about_straddling(rule)
    assert FALL_THROUGH_WORDS in rule


def test_the_fall_through_sentence_is_true_of_every_arrangement_of_three_arms():
    """All 27 ways three arms can each clear motion, clear copies or straddle:
    wherever the status is INDETERMINATE, the tallies the sentence prints and
    the situation it names are recomputed from the FIXTURE's shapes, not read
    back from the reading's own tuples.

    THE MUTATION THIS EXISTS FOR: a branch of the situation clause that is
    wrong for some arrangement the three named tests above do not build (a tally
    printed from the wrong tuple, or a split read as short).
    """
    shapes = {"motion": CLEARS_MOTION, "copies": CLEARS_COPIES, "straddle": STRADDLES}
    seen = set()
    for choice in itertools.product(shapes, repeat=3):
        reading = reading_burden(_inputs(_by_arm(**dict(zip(ARMS, (shapes[c] for c in choice))))))
        motion = [a for a, c in zip(ARMS, choice) if c == "motion"]
        copies = [a for a, c in zip(ARMS, choice) if c == "copies"]
        if len(motion) >= ARMS_REQUIRED or len(copies) >= ARMS_REQUIRED:
            assert reading.status != "INDETERMINATE", choice
            continue
        assert reading.status == "INDETERMINATE", choice

        def tally(arms):
            return f"{len(arms)} of 3 arms ({', '.join(arms) or 'none'})"

        rule = reading.rule
        assert f"motion is cleared by {tally(motion)} and copies by {tally(copies)}" in rule, choice
        if motion and copies:
            situation = "split"
            assert "the arms clear in opposite directions, so the reading is split" in rule
            assert "short of the bar" not in rule and "no arm has" not in rule
        elif motion or copies:
            situation = "short"
            assert "falls short of the bar by 1 arm(s)" in rule
            assert "opposite directions" not in rule and "no arm has" not in rule
        else:
            situation = "none"
            assert "no arm has a strict majority of its seeds" in rule
            assert "short of the bar" not in rule and "opposite directions" not in rule
        _says_nothing_about_straddling(rule)
        assert FALL_THROUGH_WORDS in rule
        seen.add(situation)

    assert seen == {"none", "short", "split"}


def test_the_fall_through_sentence_takes_its_numbers_from_the_module_and_the_inputs(
    monkeypatch,
):
    """Four arms, a bar of three, a horizon of 30: none of them the shipped
    value, so a number written as a literal cannot pass. One arm clears motion,
    which is 2 short of three -- and 1 short of the two the module ships with.

    THE MUTATIONS THIS EXISTS FOR: the horizon, the bar, the shortfall's bar or
    the arm count written as a literal inside `_fall_through_rule`, each equal
    to the shipped value and so invisible to every three-arm test above.
    """
    monkeypatch.setattr("mbfps.eval.burden.ARMS_REQUIRED", 3)
    cells = _by_arm(
        frozen_ssl=CLEARS_MOTION, pixel_ae=STRADDLES, random_vit=STRADDLES,
        scratch=STRADDLES,
    )
    reading = reading_burden(_inputs(cells, decision_h=30))
    assert reading.status == "INDETERMINATE"
    assert reading.arms_motion == ("frozen_ssl",)
    rule = reading.rule
    assert "neither decisive status reaches 3 arms at horizon 30." in rule
    assert "motion is cleared by 1 of 4 arms (frozen_ssl)" in rule
    assert "falls short of the bar by 2 arm(s)" in rule
    for stale in ("horizon 45", "reaches 2 arms", "of 3 arms", "by 1 arm(s)"):
        assert stale not in rule, stale


@pytest.mark.parametrize(
    "seeds, clearing, counts",
    [(3, 2, True), (3, 1, False), (4, 3, True), (4, 2, False), (5, 3, True), (5, 2, False)],
)
@pytest.mark.parametrize(
    "shape, status, tally",
    [
        (CLEARS_MOTION, "PREDICTS_MOTION", "arms_motion"),
        (CLEARS_COPIES, "COPIES", "arms_copies"),
    ],
)
def test_an_arm_counts_on_a_strict_majority_of_its_own_seeds(
    shape, status, tally, seeds, clearing, counts
):
    """The probe arm has `seeds` seeds of which `clearing` are decisive and the
    rest straddle; the second arm clears outright and the third never does, so
    the reading is decisive exactly when the probe arm counts. Four seeds with
    two clearing is exactly HALF, which is not a majority.

    THE MUTATIONS THIS EXISTS FOR: the per-arm bar stored as `2` (right at
    three seeds, wrong at four and five), as `1`, and as every seed; and
    `>` for `>=` against the computed bar.
    """
    probe = _arm_cells(
        "frozen_ssl", *([shape] * clearing + [STRADDLES] * (seeds - clearing))
    )
    cells = (
        probe
        + _arm_cells("pixel_ae", shape, shape, shape)
        + _arm_cells("random_vit", STRADDLES, STRADDLES, STRADDLES)
    )
    reading = reading_burden(_inputs(cells))
    assert reading.status == (status if counts else "INDETERMINATE")
    assert ("frozen_ssl" in getattr(reading, tally)) == counts
    assert reading.seeds_total == {"frozen_ssl": seeds, "pixel_ae": 3, "random_vit": 3}


# --- Reading H: the table ---------------------------------------------------


def _table(inputs):
    return format_reading_burden(reading_burden(inputs), inputs)


def _fields(row):
    """A table row cut at the column boundaries `READING_WIDTHS` fixes."""
    out, at = [], 0
    for width in READING_WIDTHS:
        out.append(row[at:at + width].strip())
        at += width
    return out


def _header_and_rows(text, n_rows):
    lines = text.splitlines()
    at = next(i for i, line in enumerate(lines) if line.split() == list(READING_COLUMNS))
    return lines[at], lines[at + 1:at + 1 + n_rows]


def test_the_table_prints_one_row_per_cell_at_the_header_width():
    inputs = _inputs(_nine(**CLEARS_MOTION))
    text = _table(inputs)
    lines = text.splitlines()
    header, rows = _header_and_rows(text, 9)
    assert len(READING_COLUMNS) == len(READING_WIDTHS)
    assert len(header) == sum(READING_WIDTHS)
    for a in ARMS:
        arm_rows = [line for line in lines if line.strip().startswith(a)]
        assert len(arm_rows) == 3, a
        for row in arm_rows:
            assert len(row) == len(header), (a, row)
    assert [_fields(row)[:2] for row in rows] == [
        [arm, str(seed)] for arm in ARMS for seed in (0, 1, 2)
    ]
    assert "PREDICTS MOTION" in text
    assert text.endswith("\n")


def test_every_column_has_room_for_the_widest_value_it_can_carry():
    """Columns are right-aligned and unseparated, so a value that fills its
    column runs into the one before it and `str.split` merges the two.

    `-999.9999` is a deliberately generous bound at nine characters (position
    errors run 100-250 here, so a burden or a margin sits well inside it);
    `random_vit` is the longest arm name; `copies` and `motion` the longest
    `clears` labels.

    THE MUTATIONS THIS EXISTS FOR: the first width shrunk below the longest
    arm name, which lengthens the row beyond the header; and any numeric width
    shrunk to the number's own length, which leaves no space between it and
    its neighbour.
    """
    wide = dict(
        burden_by_k={1: -999.9999, 45: -999.9999},
        compounding_by_k={1: -999.9999, 45: -999.9999},
    )
    cells = [
        _arm(arm="random_vit", seed=0, margin=-999.9999, low=-999.9999, high=-999.9999, **wide),
        _arm(arm="random_vit", seed=1, margin=999.9999, low=999.9999, high=999.9999, **wide),
    ]
    text = _table(_inputs(cells, ks=(1, 45)))
    header, rows = _header_and_rows(text, 2)
    assert header.split() == list(READING_COLUMNS)
    assert all(len(name) < width for name, width in zip(READING_COLUMNS, READING_WIDTHS))
    for row in rows:
        assert len(row) == len(header)
        assert len(row.split()) == len(READING_COLUMNS), row
    assert _fields(rows[0])[-1] == "copies"
    assert _fields(rows[1])[-1] == "motion"


def test_each_column_carries_the_quantity_it_is_named_for():
    """Cells go in out of order and every number is distinct within its row, so
    a swapped column, a wrong rung, an unsorted table or a `clears` label that
    disagrees with the bar all show. The last two cells sit exactly ON the bars:
    a low end of 0 clears nothing and a high end of 0 clears copies.

    THE MUTATIONS THIS EXISTS FOR: `ci_low` and `ci_high` swapped; the rung
    read from `ks[0]` instead of `ks[-1]`; dropping the `sorted`; `>=` for `>`
    or `<` for `<=` in the `clears` label.
    """
    cells = _nine(**CLEARS_MOTION)
    cells = _with(
        cells, "pixel_ae", 1,
        margin=2.0, margin_low=0.0, margin_high=4.0,
        burden_by_k={1: 2.0, 3: 6.0, 5: 11.0, 15: 30.0, 45: 101.5},
        compounding_by_k={1: 0.0, 3: 4.0, 5: 9.0, 15: 28.0, 45: 99.25},
    )
    cells = _with(cells, "pixel_ae", 2, margin=-1.0, margin_low=-2.0, margin_high=0.0)
    text = _table(_inputs(list(reversed(cells))))
    _, rows = _header_and_rows(text, 9)

    table = [dict(zip(READING_COLUMNS, _fields(row), strict=True)) for row in rows]
    assert [(r["arm"], r["seed"]) for r in table] == [
        (arm, str(seed)) for arm in ARMS for seed in (0, 1, 2)
    ]
    by_cell = {(r["arm"], r["seed"]): r for r in table}
    assert by_cell[("frozen_ssl", "0")] == {
        "arm": "frozen_ssl", "seed": "0", "margin": "+3.0000", "ci_low": "+1.0000",
        "ci_high": "+5.0000", "burden_k": "+90.0000", "comp_k": "+88.0000",
        "clears": "motion",
    }
    assert by_cell[("pixel_ae", "1")] == {
        "arm": "pixel_ae", "seed": "1", "margin": "+2.0000", "ci_low": "+0.0000",
        "ci_high": "+4.0000", "burden_k": "+101.5000", "comp_k": "+99.2500",
        "clears": "-",
    }
    assert by_cell[("pixel_ae", "2")]["clears"] == "copies"
    assert by_cell[("pixel_ae", "2")]["margin"] == "-1.0000"
    assert by_cell[("pixel_ae", "2")]["ci_high"] == "+0.0000"


@pytest.mark.parametrize(
    "cells, status",
    [
        (_nine(**CLEARS_MOTION), "PREDICTS_MOTION"),
        (_nine(**CLEARS_COPIES), "COPIES"),
        (_nine(**STRADDLES), "INDETERMINATE"),
        (_with(_nine(**CLEARS_MOTION), "pixel_ae", 0, k_one_is_floor=True), "UNRESOLVED_CONTROL"),
        (_with(_nine(**CLEARS_MOTION), "pixel_ae", 0, displacement_median=1.0), "UNREADABLE"),
    ],
    ids=["predicts-motion", "copies", "indeterminate", "unresolved-control", "unreadable"],
)
def test_the_verdict_line_closes_the_table_with_the_status_and_its_rule(cells, status):
    """The table has to print under every status, including the two where no
    arm votes, and its last line is the verdict and the sentence that decided
    it, verbatim."""
    inputs = _inputs(cells)
    reading = reading_burden(inputs)
    assert reading.status == status
    text = format_reading_burden(reading, inputs)
    assert text.endswith("\n")
    assert text.splitlines()[-1] == (
        f"  verdict: {status.replace('_', ' ')} -- decided by: {reading.rule}"
    )
    _header_and_rows(text, len(cells))


def test_the_table_interpolates_its_numbers_and_never_hardcodes_them():
    """THE MUTATION THIS EXISTS FOR: the legend's tolerance written as a
    literal that is wrong. M3l shipped a legend reading "floor exactly 0" where
    the check is abs(floor) <= 1e-9 and real floors are ~1e-13; it is still an
    open follow-up.

    This test sees only a wrong literal or a dropped number: a literal that
    HAPPENS to equal the module's value passes it, and
    `test_the_table_follows_the_module_and_the_inputs_it_is_given` is the one
    that moves the values.
    """
    inputs = _inputs(_nine(**CLEARS_MOTION))
    text = _table(inputs)
    assert f"at horizon {DECISION_H} (re-grounding periods k = 1, 3, 5, 15, 45)" in text
    assert f"held within {IDENTITY_TOLERANCE:g}" in text
    assert f"in at least {ARMS_REQUIRED} of 3 arms" in text
    assert "exactly 0" not in text


def test_the_legend_states_both_axes_and_the_header_does_not_depend_on_ks():
    """`burden_k` and `comp_k` are `burden(k = ks[-1], h = decision_h)` -- two
    axes, and the header names neither value. The first draft called the columns
    `burden45` and `comp45`: both axes are 45 in production, so the label did not
    say WHICH 45, and it would have lied the first time `ks[-1]` was not 45.

    The legend is where the axes are stated, interpolated, so it must follow the
    inputs: here `ks` ends in 15 and the horizon is 30, and the header is the
    same line it is under the shipped `ks`.

    THE MUTATIONS THIS EXISTS FOR: header names carrying the rung as a digit,
    by constant (`burden45`) or built from `ks[-1]` per call; the legend's `k`
    read from `ks[0]`, from the horizon or as a literal 45; its `h` read from
    `DECISION_H`; and the axis words dropped, which leaves two bare numbers.
    """
    shipped = _inputs(_nine(**CLEARS_MOTION))
    moved = _inputs(_nine(**CLEARS_MOTION), decision_h=30, ks=(1, 3, 5, 15))
    shipped_text, moved_text = _table(shipped), _table(moved)

    assert shipped_text.splitlines()[1] == moved_text.splitlines()[1]
    header = moved_text.splitlines()[1]
    assert header.split() == list(READING_COLUMNS)
    assert not any(ch.isdigit() for ch in header), header

    burden_col, comp_col = READING_COLUMNS[5], READING_COLUMNS[6]
    legend = next(l for l in moved_text.splitlines() if l.startswith(f"  {burden_col}/{comp_col} ="))
    assert "burden(k=15, h=30) and compounding(k=15, h=30)" in legend
    assert "k is the re-grounding period" in legend
    assert "h is the horizon step" in legend
    assert "45" not in legend
    shipped_legend = next(l for l in shipped_text.splitlines() if l.startswith(f"  {burden_col}/"))
    assert "burden(k=45, h=45) and compounding(k=45, h=45)" in shipped_legend


def test_the_table_follows_the_module_and_the_inputs_it_is_given(monkeypatch):
    """Every number the caption, legend and rule print is moved to a value
    that is nowhere else in the text, so a number written as a literal -- even
    one that equals today's value -- cannot pass.

    The horizon is 30 against re-grounding periods that end in 45, and the rung
    is read BY k (a dict lookup at `ks[-1]`), so a table that reads the rung at
    `decision_h` raises KeyError instead of printing it under the wrong name.
    Four arms against a bar of four, a tolerance of 1e-7: none of the shipped
    values.

    THE MUTATIONS THIS EXISTS FOR: a literal, a module constant or the
    `ks`-derived value standing in for any of `inputs.decision_h`, the `ks`
    list, `IDENTITY_TOLERANCE`, `ARMS_REQUIRED` or the arm count.
    """
    monkeypatch.setattr("mbfps.eval.burden.IDENTITY_TOLERANCE", 1e-7)
    monkeypatch.setattr("mbfps.eval.burden.ARMS_REQUIRED", 4)
    ladder = dict(
        burden_by_k={1: 2.0, 2: 3.0, 7: 5.0, 45: 90.0},
        compounding_by_k={1: 0.0, 2: 1.0, 7: 3.0, 45: 88.0},
    )
    cells = [
        _arm(arm=arm, seed=seed, **CLEARS_MOTION, **ladder)
        for arm in (*ARMS, "scratch") for seed in (0, 1, 2)
    ]
    inputs = _inputs(cells, decision_h=30, ks=(1, 2, 7, 45))
    reading = reading_burden(inputs)
    assert reading.status == "PREDICTS_MOTION"
    text = format_reading_burden(reading, inputs)

    assert "at horizon 30 (re-grounding periods k = 1, 2, 7, 45)" in text
    assert "burden(k=45, h=30)" in text
    assert "compounding(k=45, h=30)" in text
    assert "held within 1e-07" in text
    assert "in at least 4 of 4 arms" in text
    assert "in 4 of 4 arms" in reading.rule
    assert "clears 0 at horizon 30" in reading.rule
    for stale in ("horizon 45", "h=45", "1e-09", "exactly 0"):
        assert stale not in text, stale
