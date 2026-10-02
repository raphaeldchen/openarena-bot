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
    margin_interval,
    motion_margin,
    one_step_persistence,
    scored_targets,
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


def test_one_step_persistence_is_the_true_one_step_displacement():
    """Its error IS the displacement, known from ground truth with no model.

    THE MUTATION THIS EXISTS FOR: comparing each row to row 0 (the t-anchored
    persistence the record already carries) instead of to the row before it.
    That baseline is RIGGED for this comparison -- it has seen h-1 fewer frames
    than the k=1 rung -- and spec 2.2 forbids it by name.
    """
    # Moves (3, 4) then (0, 0) then (6, 8): displacements 5, 0, 10.
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [9.0, 12.0]])
    window_targets = np.hstack([xy, np.zeros((4, 2))])
    assert one_step_persistence(window_targets) == pytest.approx([5.0, 0.0, 10.0])

    # The rigged baseline would give cumulative distance from row 0 instead.
    rigged = np.linalg.norm(xy[1:] - xy[0], axis=1)
    assert rigged == pytest.approx([5.0, 5.0, 15.0])
    assert one_step_persistence(window_targets) != pytest.approx(rigged)


def test_motion_margin_is_positive_when_the_prior_beats_stillness():
    """Positive means one prior step beats assuming no motion."""
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [9.0, 12.0]])
    window_targets = np.hstack([xy, np.zeros((4, 2))])
    # Displacements are 5, 0, 10. A prior that errs by 2, 1, 3 beats stillness
    # at steps 1 and 3 and loses at step 2, where the agent did not move.
    curve_one = np.array([2.0, 1.0, 3.0])
    assert motion_margin(window_targets, curve_one) == pytest.approx([3.0, -1.0, 7.0])


def test_motion_margin_refuses_a_curve_that_is_not_the_scored_length():
    """THE MUTATION THIS EXISTS FOR: dropping the guard. A curve of length
    horizon+1 would broadcast against a baseline of length horizon only by
    accident of the numbers, and silently not at all otherwise.
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
    the wrong rank, and a single row has the right rank but nothing to score.

    THE MUTATIONS THIS EXISTS FOR: dropping either operand of the guard
    (`< 2` -> `< 1` lets a single row through, returning an empty array that
    every downstream mean turns into NaN), and dropping the float64 coercion.
    An unsigned-integer window left uncoerced would wrap on subtraction.
    """
    with pytest.raises(ValueError, match="at least two rows"):
        scored_targets(np.zeros(5))  # rank 1, five elements
    with pytest.raises(ValueError, match="at least two rows"):
        scored_targets(np.zeros((1, 4)))  # rank 2, one row
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
    and an unsigned-integer window WRAPS: uint8 0 - 3 is 253, so a baseline of
    displacement 5 would read 356.

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
    refused by NumPy at all -- it indexes the LAST column -- so only the lower
    bound stands between it and a silently wrong interval; a horizon past the
    end is caught by the upper bound (a missing one surfaces as an IndexError,
    which is not the ValueError asserted here). The two edges that must WORK are
    exercised too, so a bound moved one step inward also fails.

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
