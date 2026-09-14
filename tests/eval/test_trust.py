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
