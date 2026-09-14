"""The trust horizon's pure functions, pinned by synthetic priors with known answers.

`mbfps.eval.trust` has no model, no files and no randomness: every input here
is a small array typed by hand and every expectation is typed beside it,
never read back from the function under test. The priors are the ones the
M3d design names, and each is a mutation the others cannot catch:

  * THE PERSISTENCE CLONE (`model_err == persist_err` everywhere) never
    crosses -- `h x = horizon + 1` on every moved row -- and its margin is 0.
  * THE PERFECT PREDICTOR (`model_err == 0`) never crosses either; its
    crossing is IDENTICAL to the clone's and only the margin (persistence's
    own error, positive) tells them apart. A test that read the crossing
    alone could not see a model that sits still.
  * A PRIOR THAT CROSSES AT A KNOWN STEP on some rows and never on others,
    with one row that loses BEFORE it moves (ignored: the search starts at
    the first moved step) and one that returns near its start after moving
    (still crosses: the mask gates the start of the search, not each step).
  * ROWS THAT NEVER MOVE are NaN, and NaN rows leave the survival curve's
    denominator -- counted, not scored as failures.
  * TIES NEVER CROSS. `>` not `>=`: a clone is not worse than persistence.
  * AN EVEN-COUNT CROSSING MULTISET, whose median is a half-step, gives an
    integer `H*_q` because `H*_q` reads the survival curve, not a quantile.

Task 2 appends the decomposition, embedding-ratio and scale-correction tests
below this file's last test; the header is not repeated.
"""

import numpy as np
import pytest

from mbfps.eval.trust import (
    MIN_MOVE,
    crossing_step,
    moved_mask,
    persistence_margin,
    survival,
    trust_horizon,
)

import warnings

from mbfps.eval.trust import (  # Task 2: decomposition, embedding ratio, scale correction
    ALPHAS,
    Decomposition,
    ScaleCorrection,
    displacement_decomposition,
    embedding_ratio,
    scale_corrected_error,
)

# Deliberately TEST-LOCAL: the shipped horizon is 45, but every answer below
# is typed by hand at a horizon small enough to check by eye.
HORIZON = 6
NEVER = float(HORIZON + 1)
T, F = True, False


# --- moved_mask ---------------------------------------------------------------


def test_min_move_is_five_map_units():
    assert MIN_MOVE == 5.0


def test_moved_mask_is_the_euclidean_displacement_at_or_above_the_threshold():
    anchor = np.array([[10.0, 20.0]])
    positions = np.array([[
        [13.0, 24.0],   # (3, 4): norm 5.0 -> moved, on the threshold (>=)
        [12.5, 22.5],   # (2.5, 2.5): norm 3.54 -> not moved (L1 would say 5)
        [10.0, 24.0],   # (0, 4): norm 4 -> not moved
        [10.0, 15.0],   # (0, -5): norm 5 -> moved; sign is irrelevant
        [10.0, 20.0],   # did not move at all
    ]])
    mask = moved_mask(positions, anchor)
    assert mask.dtype == bool
    assert mask.shape == (1, 5)
    assert mask.tolist() == [[T, F, F, T, F]]


def test_moved_mask_honours_min_move():
    anchor = np.array([[10.0, 20.0]])
    positions = np.array([[[13.0, 24.0], [12.5, 22.5], [10.0, 24.0], [10.0, 15.0], [10.0, 20.0]]])
    assert moved_mask(positions, anchor, min_move=4.0).tolist() == [[T, F, T, T, F]]
    assert moved_mask(positions, anchor, min_move=5.5).tolist() == [[F, F, F, F, F]]


def test_moved_mask_measures_each_row_against_its_own_anchor():
    anchors = np.array([[0.0, 0.0], [100.0, 100.0]])
    positions = np.array([
        [[6.0, 0.0], [0.0, 0.0]],        # row 0: moved 6, then back home
        [[6.0, 0.0], [100.0, 106.0]],    # row 1: (6, 0) is 137 units from ITS anchor
    ])
    # A mask that read every row against anchors[0] would say [[T, F], [T, T]].
    assert moved_mask(positions, anchors).tolist() == [[T, F], [T, T]]
    positions[1, 0] = [100.0, 100.0]
    assert moved_mask(positions, anchors).tolist() == [[T, F], [F, T]]


def test_moved_mask_refuses_the_wrong_shapes():
    with pytest.raises(ValueError, match=r"p_true must be \(n, H, 2\)"):
        moved_mask(np.zeros((2, 6)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"p_true must be \(n, H, 2\)"):
        moved_mask(np.zeros((2, 6, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"p_true0 must be \(n, 2\) = \(2, 2\)"):
        moved_mask(np.zeros((2, 6, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match=r"p_true0 must be \(n, 2\)"):
        moved_mask(np.zeros((2, 6, 2)), np.zeros((2,)))


# --- crossing_step and persistence_margin ------------------------------------


def test_a_persistence_clone_never_crosses_and_its_margin_is_zero():
    persist = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    model = persist.copy()
    moved = np.ones((3, HORIZON), dtype=bool)
    crossing = crossing_step(model, persist, moved)
    assert crossing.dtype == np.float64
    assert crossing.tolist() == [NEVER, NEVER, NEVER]
    assert persistence_margin(model, persist).tolist() == [[0.0] * HORIZON] * 3


def test_a_perfect_predictor_never_crosses_and_its_margin_is_persistences_error():
    persist = np.array([[3.0, 8.0, 12.0, 20.0, 25.0, 31.0]])
    model = np.zeros((1, HORIZON))
    moved = np.ones((1, HORIZON), dtype=bool)
    assert crossing_step(model, persist, moved).tolist() == [NEVER]
    assert persistence_margin(model, persist).tolist() == [[3.0, 8.0, 12.0, 20.0, 25.0, 31.0]]


def test_the_clone_and_the_perfect_predictor_share_a_crossing_and_differ_in_margin():
    # The crossing cannot tell "sits still" from "predicts exactly"; the margin
    # can, and Task 2's ratio does too. Pinned so a reader of `h x` alone is
    # never mistaken for a reader of trust.
    persist = np.array([[3.0, 8.0, 12.0, 20.0, 25.0, 31.0]])
    moved = np.ones((1, HORIZON), dtype=bool)
    clone, perfect = persist.copy(), np.zeros((1, HORIZON))
    assert crossing_step(clone, persist, moved).tolist() == crossing_step(perfect, persist, moved).tolist() == [NEVER]
    assert (persistence_margin(clone, persist) == 0.0).all()
    assert (persistence_margin(perfect, persist) > 0.0).all()


def test_crossing_is_the_first_strict_exceedance_at_or_after_the_first_moved_step():
    persist = np.full((4, HORIZON), 10.0)
    model = np.array([
        [5.0, 5.0, 11.0, 11.0, 11.0, 11.0],   # moved from step 1; loses from step 3 -> 3
        [12.0, 12.0, 5.0, 5.0, 12.0, 5.0],    # loses at 1, 2 BEFORE moving; moves at 3; loses at 5 -> 5
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],       # never loses -> never (7)
        [11.0, 9.0, 9.0, 9.0, 9.0, 9.0],      # loses at the first moved step itself -> 1
    ])
    moved = np.array([
        [T, T, T, T, T, T],
        [F, F, T, T, T, T],
        [F, T, T, T, T, T],
        [T, T, T, T, T, T],
    ])
    assert crossing_step(model, persist, moved).tolist() == [3.0, 5.0, NEVER, 1.0]


def test_a_window_that_returns_near_its_start_after_moving_still_crosses():
    # The mask gates the START of the search (h0), not each step after it:
    # a loss at a step where the agent has come back within MIN_MOVE of its
    # start still counts once the window has moved at all.
    persist = np.full((1, HORIZON), 10.0)
    model = np.array([[5.0, 5.0, 11.0, 5.0, 5.0, 5.0]])
    moved = np.array([[F, T, F, T, T, T]])   # moved at 2, back home at 3, gone again
    assert crossing_step(model, persist, moved).tolist() == [3.0]


def test_ties_never_cross():
    persist = np.full((3, HORIZON), 10.0)
    model = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],   # equal everywhere -> never
        [10.0, 10.0, 10.5, 10.0, 10.0, 10.0],   # strictly worse at 3 only -> 3
        [9.9, 9.9, 9.9, 9.9, 9.9, 9.9],         # always slightly better -> never
    ])
    moved = np.ones((3, HORIZON), dtype=bool)
    assert crossing_step(model, persist, moved).tolist() == [NEVER, 3.0, NEVER]


def test_a_row_that_never_moves_is_nan_whatever_its_errors_say():
    persist = np.full((3, HORIZON), 10.0)
    model = np.full((3, HORIZON), 11.0)   # loses everywhere on every row
    moved = np.array([
        [F, F, F, F, F, F],   # never moved -> NaN, not 1
        [T, T, T, T, T, T],   # -> 1
        [F, F, F, F, F, T],   # moves only at the last step -> 6
    ])
    crossing = crossing_step(model, persist, moved)
    assert np.isnan(crossing[0])
    assert crossing[1:].tolist() == [1.0, 6.0]


def test_crossing_step_refuses_mismatched_shapes_and_nonfinite_errors():
    ok = np.ones((2, HORIZON))
    moved = np.ones((2, HORIZON), dtype=bool)
    with pytest.raises(ValueError, match=r"model_err must be \(n, H\)"):
        crossing_step(np.ones(HORIZON), ok, moved)
    with pytest.raises(ValueError, match="share one \\(n, H\\) shape"):
        crossing_step(ok, np.ones((2, HORIZON - 1)), moved)
    with pytest.raises(ValueError, match="share one \\(n, H\\) shape"):
        crossing_step(ok, ok, np.ones((3, HORIZON), dtype=bool))
    with pytest.raises(ValueError, match="must be finite"):
        crossing_step(np.where(np.eye(2, HORIZON) > 0, np.nan, 1.0), ok, moved)
    with pytest.raises(ValueError, match="must be finite"):
        crossing_step(ok, np.full((2, HORIZON), np.inf), moved)


def test_persistence_margin_is_persistence_minus_model():
    model = np.array([[1.0, 4.0], [10.0, 0.0]])
    persist = np.array([[3.0, 2.0], [10.0, 5.0]])
    assert persistence_margin(model, persist).tolist() == [[2.0, -2.0], [0.0, 5.0]]
    with pytest.raises(ValueError, match="must share a shape"):
        persistence_margin(model, persist[:1])


# --- survival and trust_horizon ----------------------------------------------


def test_the_persistence_clones_survival_is_one_everywhere_and_its_horizon_is_the_horizon():
    crossings = np.full(3, NEVER)   # what crossing_step gives the clone
    curve = survival(crossings, HORIZON)
    assert curve.shape == (HORIZON + 1,)
    assert curve.tolist() == [1.0] * (HORIZON + 1)
    for q in (0.5, 0.75, 0.9, 1.0):
        assert trust_horizon(curve, q) == HORIZON


def test_survival_is_the_fraction_of_finite_crossings_strictly_beyond_h():
    crossings = np.array([2.0, 4.0, 4.0, NEVER])
    # h:            0    1    2     3     4     5     6
    expected = [1.0, 1.0, 0.75, 0.75, 0.25, 0.25, 0.25]
    assert survival(crossings, HORIZON).tolist() == expected


def test_survival_excludes_never_moved_rows_from_the_denominator():
    crossings = np.array([2.0, np.nan, np.nan, 4.0])
    # Over the two finite rows {2, 4}; a denominator of four would halve it.
    assert survival(crossings, 4).tolist() == [1.0, 1.0, 0.5, 0.5, 0.0]


@pytest.mark.parametrize(
    "crossings",
    [[1.0], [1.0, np.nan], [NEVER], [1.0, 3.0, NEVER, np.nan, 2.0]],
    ids=["all-cross-at-1", "one-nan", "never", "mixed"],
)
def test_survival_starts_at_one_whenever_any_crossing_is_finite(crossings):
    assert survival(np.array(crossings), HORIZON)[0] == 1.0


def test_survival_is_all_nan_without_a_finite_crossing_and_the_horizon_is_minus_one():
    curve = survival(np.array([np.nan, np.nan]), HORIZON)
    assert curve.shape == (HORIZON + 1,)
    assert np.isnan(curve).all()
    for q in (0.5, 0.75, 0.9):
        assert trust_horizon(curve, q) == -1


def test_survival_refuses_crossings_outside_one_to_horizon_plus_one():
    with pytest.raises(ValueError, match="crossings must lie in 1..7"):
        survival(np.array([0.0, 3.0]), HORIZON)          # a 0-based step
    with pytest.raises(ValueError, match="crossings must lie in 1..7"):
        survival(np.array([3.0, NEVER + 1.0]), HORIZON)  # a longer horizon's crossing
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        survival(np.array([1.0]), 0)
    with pytest.raises(ValueError, match=r"crossings must be \(n,\)"):
        survival(np.array([[1.0, 2.0]]), HORIZON)


def test_an_even_count_crossing_multiset_gives_an_integer_horizon_without_rounding():
    crossings = np.array([2.0, 2.0, NEVER, NEVER])
    assert np.median(crossings) == 4.5   # the half-step a quantile reading would print
    # h:            0    1    2    3    4    5    6
    expected = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    curve = survival(crossings, HORIZON)
    assert curve.tolist() == expected
    horizons = {q: trust_horizon(curve, q) for q in (0.5, 0.75, 0.9)}
    assert horizons == {0.5: 6, 0.75: 1, 0.9: 1}
    assert all(type(h) is int for h in horizons.values())


def test_trust_horizon_is_the_largest_step_at_or_above_q():
    curve = np.array([1.0, 1.0, 0.75, 0.75, 0.5, 0.25, 0.0])
    assert trust_horizon(curve, 1.0) == 1
    assert trust_horizon(curve, 0.9) == 1
    assert trust_horizon(curve, 0.75) == 3    # >=, not >: S(3) == 0.75 qualifies
    assert trust_horizon(curve, 0.5) == 4
    assert trust_horizon(curve, 0.25) == 5
    assert trust_horizon(curve, 0.1) == 5     # S(6) == 0 < 0.1
    assert trust_horizon(curve, 0.0) == 6


def test_trust_horizon_is_zero_when_no_step_reaches_q():
    assert trust_horizon(np.array([0.5, 0.25]), 0.75) == 0
    assert trust_horizon(np.array([1.0, 1.0]), 1.5) == 0
    assert type(trust_horizon(np.array([0.5, 0.25]), 0.75)) is int


def test_trust_horizon_refuses_a_curve_that_is_not_one_dimensional():
    with pytest.raises(ValueError, match=r"surv must be \(H \+ 1,\)"):
        trust_horizon(np.ones((2, HORIZON + 1)), 0.75)


def test_crossings_survive_into_the_horizon_end_to_end():
    # The four-row prior from the crossing test plus one never-moved row:
    # crossings {3, 5, never, 1} and NaN. Hand-built S over the four finite:
    # h:            0    1     2     3    4    5     6
    expected = [1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25]
    persist = np.full((5, HORIZON), 10.0)
    model = np.array([
        [5.0, 5.0, 11.0, 11.0, 11.0, 11.0],
        [12.0, 12.0, 5.0, 5.0, 12.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        [11.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
    ])
    moved = np.array([
        [T, T, T, T, T, T],
        [F, F, T, T, T, T],
        [F, T, T, T, T, T],
        [T, T, T, T, T, T],
        [F, F, F, F, F, F],
    ])
    curve = survival(crossing_step(model, persist, moved), HORIZON)
    assert curve.tolist() == expected
    assert trust_horizon(curve, 0.75) == 2
    assert trust_horizon(curve, 0.5) == 4
    assert trust_horizon(curve, 0.9) == 0   # only S(0) reaches 0.9: no step is trusted


# ============================================================================
# Task 2: displacement_decomposition, embedding_ratio, scale_corrected_error
# ============================================================================
#
# One ground truth serves every prior below, so the priors differ ONLY in what
# the model does with it. Row i starts at (10 i, 0) and walks +x by 5 (i + 1)
# map units per step, so |d(i, h)| = 5 (i + 1) (h + 1) -- every cell is moved
# (>= MIN_MOVE = 5), every displacement is an exact integer along one axis, and
# the fold split (episode i, even -> A, odd -> B) puts rows 0, 2 in A and rows
# 1, 3 in B. The probe reads every position with a constant bias of 3 units in
# +y, orthogonal to every walk: the prior's anchor p_hat(0) is the truth plus
# that bias, and so is a perfect prediction. The bias makes the perfect
# predictor's raw error a non-zero 3.0 at every cell, so "held-out error equals
# the raw error" is a real number and not 0 == 0.

N, H = 4, 3
BIAS = np.array([0.0, 3.0])


def _truth():
    p_true0 = np.stack([10.0 * np.arange(N), np.zeros(N)], axis=1)  # (N, 2)
    steps = 5.0 * np.arange(1, N + 1)[:, None] * np.arange(1, H + 1)[None, :]  # (N, H)
    p_true = p_true0[:, None, :] + np.stack([steps, np.zeros_like(steps)], axis=-1)
    return p_true0, p_true


def _displacement():
    """|d(i, h)| = 5 (i + 1) (h + 1), by hand."""
    return 5.0 * np.arange(1, N + 1)[:, None] * np.arange(1, H + 1)[None, :]


def _prior(scale: float):
    """A prior that predicts the anchor plus `scale` times the true displacement.

    scale 0 = persistence clone, 1 = perfect predictor, 2 = exact 2x overshoot.
    p_hat_real is the same probe reading the real future frames: the truth plus
    the bias, i.e. the perfect predictor.
    """
    p_true0, p_true = _truth()
    p_hat0 = p_true0 + BIAS
    d = p_true - p_true0[:, None]
    p_hat = p_hat0[:, None] + scale * d
    p_hat_real = p_true + BIAS
    return p_hat, p_true, p_hat0, p_true0, p_hat_real


ALL_MOVED = np.ones((N, H), dtype=bool)
EPISODE = np.arange(N)  # rows 0, 2 -> fold A; rows 1, 3 -> fold B


def test_alphas_is_the_pinned_grid():
    assert ALPHAS.shape == (201,)
    assert ALPHAS[0] == 0.0 and ALPHAS[-1] == 2.0
    assert ALPHAS[1] == pytest.approx(0.01)
    # the two values the priors below land on are exact grid points
    assert ALPHAS[100] == 1.0 and ALPHAS[50] == 0.5


# --- displacement_decomposition ---------------------------------------------


def test_persistence_clone_reads_ratio_zero_cosine_nan_and_flags_zero_displacement():
    dec = displacement_decomposition(*_prior(0.0), ALL_MOVED)
    assert isinstance(dec, Decomposition)
    np.testing.assert_array_equal(dec.ratio_probe, np.zeros((N, H)))
    np.testing.assert_array_equal(dec.ratio_raw, np.zeros((N, H)))
    assert np.isnan(dec.cosine).all()
    np.testing.assert_array_equal(dec.zero_displacement, ALL_MOVED)
    assert dec.moved is ALL_MOVED or np.array_equal(dec.moved, ALL_MOVED)


def test_perfect_predictor_reads_ratio_one_cosine_one():
    dec = displacement_decomposition(*_prior(1.0), ALL_MOVED)
    np.testing.assert_array_equal(dec.ratio_probe, np.ones((N, H)))
    np.testing.assert_array_equal(dec.ratio_raw, np.ones((N, H)))
    np.testing.assert_array_equal(dec.cosine, np.ones((N, H)))
    assert not dec.zero_displacement.any()


def test_exact_overshoot_reads_ratio_two_cosine_one():
    dec = displacement_decomposition(*_prior(2.0), ALL_MOVED)
    np.testing.assert_array_equal(dec.ratio_probe, np.full((N, H), 2.0))
    np.testing.assert_array_equal(dec.ratio_raw, np.full((N, H), 2.0))
    np.testing.assert_array_equal(dec.cosine, np.ones((N, H)))


def test_probe_attenuation_cancels_in_ratio_probe_but_not_in_ratio_raw():
    # A flatter probe reads BOTH the imagined and the real displacement at 0.3
    # of their true length; the probe-normalised ratio is the imagined
    # displacement over the same probe's reading of the real one, so the
    # attenuation cancels and the prior is read as moving the right distance.
    p_hat, p_true, p_hat0, p_true0, _ = _prior(0.3)
    p_hat_real = p_hat0[:, None] + 0.3 * (p_true - p_true0[:, None])
    dec = displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, ALL_MOVED)
    np.testing.assert_allclose(dec.ratio_probe, np.ones((N, H)))
    np.testing.assert_allclose(dec.ratio_raw, np.full((N, H), 0.3))
    np.testing.assert_allclose(dec.cosine, np.ones((N, H)))


def test_decomposition_respects_the_moved_mask_and_a_dead_probe_reading():
    # Row 0 is perfect but marked not-moved at h = 1; row 1 is a persistence
    # clone (d_hat = 0) marked not-moved everywhere; row 2's real-frame probe
    # reading never leaves the anchor (|d_hat_real| = 0); row 3 is perfect.
    p_hat, p_true, p_hat0, p_true0, p_hat_real = (a.copy() for a in _prior(1.0))
    p_hat[1] = p_hat0[1]
    p_hat_real[2] = p_hat0[2]
    moved = np.ones((N, H), dtype=bool)
    moved[0, 1] = False
    moved[1, :] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero RuntimeWarning may escape
        dec = displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved)
    # not moved -> every channel NaN, whatever the prior did there
    assert np.isnan(dec.ratio_probe[0, 1]) and np.isnan(dec.ratio_raw[0, 1]) and np.isnan(dec.cosine[0, 1])
    assert np.isnan(dec.ratio_probe[1]).all() and np.isnan(dec.ratio_raw[1]).all() and np.isnan(dec.cosine[1]).all()
    # a zero d_hat on a not-moved row is NOT a zero-displacement exclusion
    assert not dec.zero_displacement[1].any() and not dec.zero_displacement[0, 1]
    assert not dec.zero_displacement.any()
    # the dead probe reading: ratio_probe NaN, ratio_raw and cosine still read
    assert np.isnan(dec.ratio_probe[2]).all()
    np.testing.assert_array_equal(dec.ratio_raw[2], np.ones(H))
    np.testing.assert_array_equal(dec.cosine[2], np.ones(H))
    # everything else untouched
    np.testing.assert_array_equal(dec.ratio_probe[3], np.ones(H))
    np.testing.assert_array_equal(dec.ratio_probe[0, [0, 2]], np.ones(2))
    np.testing.assert_array_equal(dec.moved, moved)
    assert dec.ratio_probe.shape == dec.ratio_raw.shape == dec.cosine.shape == (N, H)


# --- embedding_ratio -----------------------------------------------------------


def test_embedding_ratio_is_the_imagined_over_the_true_norm():
    e_hat_disp = np.array([[2.0, 4.0], [0.0, 1.0]])
    e_true_disp = np.array([[1.0, 4.0], [3.0, 2.0]])
    out = embedding_ratio(e_hat_disp, e_true_disp, np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(out, np.array([[2.0, 1.0], [0.0, 0.5]]))


def test_embedding_ratio_is_nan_where_not_moved_or_where_the_truth_did_not_move():
    e_hat_disp = np.array([[2.0, 4.0], [7.0, 1.0]])
    e_true_disp = np.array([[1.0, 4.0], [0.0, 2.0]])
    moved = np.array([[True, False], [True, True]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = embedding_ratio(e_hat_disp, e_true_disp, moved)
    assert out[0, 0] == 2.0
    assert np.isnan(out[0, 1])  # not moved
    assert np.isnan(out[1, 0])  # moved, but the true embedding did not move: 7 / 0 is not a ratio
    assert out[1, 1] == 0.5
    assert out.shape == (2, 2)


# --- scale_corrected_error ----------------------------------------------------


def test_perfect_predictor_fits_alpha_one_on_both_folds_and_held_out_equals_its_raw_error():
    p_hat, p_true, p_hat0, _, _ = _prior(1.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    assert isinstance(sc, ScaleCorrection)
    assert sc.folds_available is True
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.ones(H))
    raw_error = np.linalg.norm(p_hat - p_true, axis=-1)  # 3.0 everywhere: the probe's bias
    np.testing.assert_array_equal(raw_error, np.full((N, H), 3.0))
    np.testing.assert_array_equal(sc.held_out, raw_error)
    np.testing.assert_array_equal(sc.score_a, np.full(H, 3.0))
    np.testing.assert_array_equal(sc.score_b, np.full(H, 3.0))
    assert not sc.boundary.any()
    assert sc.alpha_a.shape == sc.alpha_b.shape == sc.score_a.shape == sc.score_b.shape == sc.boundary.shape == (H,)
    assert sc.held_out.shape == (N, H)


def test_exact_overshoot_fits_alpha_half_and_its_held_out_error_is_the_perfect_predictors():
    p_hat, p_true, p_hat0, _, _ = _prior(2.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.full(H, 0.5))
    np.testing.assert_array_equal(sc.alpha_b, np.full(H, 0.5))
    # halved, the overshoot IS the perfect predictor: 3.0 of probe bias, nothing else
    np.testing.assert_array_equal(sc.held_out, np.full((N, H), 3.0))
    # ... while its raw error is far larger, so the correction did something
    raw_error = np.linalg.norm(p_hat - p_true, axis=-1)
    np.testing.assert_array_equal(raw_error, np.sqrt(9.0 + _displacement() ** 2))
    assert not sc.boundary.any()


def test_pure_noise_displacement_is_a_boundary_on_both_folds():
    # d_hat orthogonal to the truth on every row, sign alternating, from an
    # anchor exactly on the truth (no probe bias here: a biased anchor lets the
    # rows whose noise points against the bias cancel it, and that is a
    # correction, not noise). Then |alpha d_hat - d| = |d| sqrt(1 + alpha^2) on
    # every row: no positive alpha reduces the median error, the argmin lands
    # on alpha = 0 -- the grid edge -- and the record must say so. The
    # corrected number is NOT read: a boundary is not a correction (spec 2.2).
    _, p_true, _, p_true0, _ = _prior(1.0)
    p_hat0 = p_true0
    d = p_true - p_true0[:, None]
    orthogonal = np.stack([-d[..., 1], d[..., 0]], axis=-1)  # rotate d by 90 degrees
    signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)[:, None, None]
    p_hat = p_hat0[:, None] + signs * orthogonal
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.zeros(H))
    np.testing.assert_array_equal(sc.alpha_b, np.zeros(H))
    np.testing.assert_array_equal(sc.boundary, np.ones(H, dtype=bool))
    assert sc.folds_available is True


@pytest.mark.parametrize("grid", [ALPHAS, ALPHAS[::-1]], ids=["ascending", "descending"])
def test_persistence_clone_ties_every_alpha_and_the_smallest_alpha_wins(grid):
    # d_hat = 0, so alpha changes nothing and every grid point ties. The tie
    # rule is the SMALLEST alpha, not the first index -- pinned by running the
    # grid backwards too. At alpha = 0 the "corrected" error is persistence
    # itself, and the flag says boundary.
    p_hat, p_true, p_hat0, _, _ = _prior(0.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE, alphas=grid)
    np.testing.assert_array_equal(sc.alpha_a, np.zeros(H))
    np.testing.assert_array_equal(sc.alpha_b, np.zeros(H))
    np.testing.assert_array_equal(sc.boundary, np.ones(H, dtype=bool))
    persistence_error = np.sqrt(9.0 + _displacement() ** 2)  # |p_hat0 - p(h)| by hand
    np.testing.assert_array_equal(sc.held_out, persistence_error)


def test_fewer_than_two_episode_labels_disables_both_folds():
    p_hat, p_true, p_hat0, _, _ = _prior(1.0)
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, np.zeros(N, dtype=int))
    assert sc.folds_available is False
    for arr in (sc.alpha_a, sc.alpha_b, sc.score_a, sc.score_b):
        assert arr.shape == (H,) and np.isnan(arr).all()
    assert sc.held_out.shape == (N, H) and np.isnan(sc.held_out).all()
    assert sc.boundary.shape == (H,) and sc.boundary.dtype == bool and not sc.boundary.any()


def test_folds_are_even_and_odd_episode_labels_and_each_scores_the_other():
    # Fold A (rows 0, 2) is perfect; fold B (rows 1, 3) overshoots 2x. Then
    # alpha_a = 1 and alpha_b = 0.5, B's rows are scored with alpha_a = 1
    # (the overshoot left as is: sqrt(9 + |d|^2)) and A's rows with
    # alpha_b = 0.5 (the perfect prediction halved: sqrt(9 + |d|^2 / 4)).
    perfect = _prior(1.0)
    over = _prior(2.0)
    p_hat = np.where((np.arange(N) % 2 == 0)[:, None, None], perfect[0], over[0])
    _, p_true, p_hat0, _, _ = perfect
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.full(H, 0.5))
    disp = _displacement()
    expected = np.empty((N, H))
    expected[[1, 3]] = np.sqrt(9.0 + disp[[1, 3]] ** 2)  # fold B scored with alpha_a = 1
    expected[[0, 2]] = np.sqrt(9.0 + (0.5 * disp[[0, 2]]) ** 2)  # fold A scored with alpha_b = 0.5
    np.testing.assert_allclose(sc.held_out, expected)
    # the scoring-fold medians: score_a is over B's rows, score_b over A's
    np.testing.assert_allclose(sc.score_a, np.median(expected[[1, 3]], axis=0))
    np.testing.assert_allclose(sc.score_b, np.median(expected[[0, 2]], axis=0))
    assert not sc.boundary.any()


def test_the_fit_is_a_median_so_a_minority_outlier_row_does_not_move_alpha():
    # Fold A gets a third row (episode 4): two perfect rows and one 2x
    # overshoot. The median error at alpha = 1 is 3.0 (two of three rows) and
    # larger at every other alpha; a mean would be pulled toward 0.5.
    perfect = _prior(1.0)
    over = _prior(2.0)
    p_hat = np.concatenate([perfect[0], over[0][:1]])  # row 4 = row 0's walk, overshot
    p_true = np.concatenate([perfect[1], perfect[1][:1]])
    p_hat0 = np.concatenate([perfect[2], perfect[2][:1]])
    episode = np.array([0, 1, 2, 3, 4])
    sc = scale_corrected_error(p_hat, p_true, p_hat0, np.ones((5, H), dtype=bool), episode)
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.ones(H))


def test_scale_correction_fits_and_scores_only_moved_rows():
    # Row 2 (fold A) is a 2x overshoot marked not-moved everywhere. Fit on the
    # moved rows only, fold A is row 0 alone -> alpha_a = 1; had row 2 been
    # counted the two-row median would leave 1. Row 2 is never scored.
    perfect = _prior(1.0)
    over = _prior(2.0)
    p_hat = perfect[0].copy()
    p_hat[2] = over[0][2]
    _, p_true, p_hat0, _, _ = perfect
    moved = np.ones((N, H), dtype=bool)
    moved[2] = False
    sc = scale_corrected_error(p_hat, p_true, p_hat0, moved, EPISODE)
    np.testing.assert_array_equal(sc.alpha_a, np.ones(H))
    np.testing.assert_array_equal(sc.alpha_b, np.ones(H))
    assert np.isnan(sc.held_out[2]).all()
    np.testing.assert_array_equal(sc.held_out[[0, 1, 3]], np.full((3, H), 3.0))
    np.testing.assert_array_equal(sc.score_a, np.full(H, 3.0))
    np.testing.assert_array_equal(sc.score_b, np.full(H, 3.0))


def test_a_fold_with_no_moved_rows_at_a_step_is_nan_at_that_step_only():
    # At h = 0 neither fold-B row moved: fold B cannot be fit there (alpha_b
    # NaN), fold A's rows have no held-out alpha there (NaN), and B's rows are
    # not scored there either; h = 1, 2 are untouched.
    p_hat, p_true, p_hat0, _, _ = _prior(1.0)
    moved = np.ones((N, H), dtype=bool)
    moved[[1, 3], 0] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sc = scale_corrected_error(p_hat, p_true, p_hat0, moved, EPISODE)
    assert sc.folds_available is True
    assert sc.alpha_a[0] == 1.0 and np.isnan(sc.alpha_b[0])
    assert np.isnan(sc.score_a[0]) and np.isnan(sc.score_b[0])
    assert np.isnan(sc.held_out[:, 0]).all()
    assert not sc.boundary[0]
    np.testing.assert_array_equal(sc.alpha_a[1:], np.ones(2))
    np.testing.assert_array_equal(sc.alpha_b[1:], np.ones(2))
    np.testing.assert_array_equal(sc.held_out[:, 1:], np.full((N, 2), 3.0))


@pytest.mark.parametrize("edge_fold", ["a", "b"])
def test_boundary_flags_the_top_of_the_grid_on_either_fold(edge_fold):
    # One fold predicts a quarter of the displacement: its best alpha is 4,
    # off the top of the grid, so the fit stops at 2.0 -- a boundary. The
    # other fold is perfect (alpha 1). Either fold on an edge flags the step.
    perfect = _prior(1.0)
    under = _prior(0.25)
    edge_rows = (np.arange(N) % 2 == 0) if edge_fold == "a" else (np.arange(N) % 2 == 1)
    p_hat = np.where(edge_rows[:, None, None], under[0], perfect[0])
    _, p_true, p_hat0, _, _ = perfect
    sc = scale_corrected_error(p_hat, p_true, p_hat0, ALL_MOVED, EPISODE)
    edge_alpha, other_alpha = (sc.alpha_a, sc.alpha_b) if edge_fold == "a" else (sc.alpha_b, sc.alpha_a)
    np.testing.assert_array_equal(edge_alpha, np.full(H, 2.0))
    np.testing.assert_array_equal(other_alpha, np.ones(H))
    np.testing.assert_array_equal(sc.boundary, np.ones(H, dtype=bool))
