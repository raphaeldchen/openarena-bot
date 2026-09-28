"""M3j: where on the enc(t) -> z -> h path is observed motion lost?"""

import numpy as np
import pytest

from mbfps.eval.retention import (
    ARMS_REQUIRED,
    CONFIDENCE,
    DECISION_K,
    K_REPORTED,
    LATENT_RUNGS,
    RESAMPLES,
    RETENTION_FAMILY,
    RUNGS,
    SEEDS_REQUIRED,
    TARGETS,
    Z_BEARING_RUNGS,
    backward_rotation,
    backward_translation,
    rung_block,
    shifted_rows,
)

# Two windows of four steps. Positions are hand-chosen so every displacement
# below can be read off by eye, and the two windows differ so a target that
# silently paired rows across windows would change the numbers.
WINDOW = np.array([0, 0, 0, 0, 1, 1, 1, 1])
STEP = np.array([0, 1, 2, 3, 0, 1, 2, 3])
POS = np.array([
    [0.0, 0.0], [3.0, -2.0], [6.0, -4.0], [9.0, -6.0],
    [10.0, 10.0], [12.0, 13.0], [14.0, 16.0], [16.0, 19.0],
])
# Degrees. Window 0 WRAPS through 360 (350 -> 20 is +30, not -330), which is the
# case a plain subtraction gets wrong; window 1 turns a uniform +45.
ANGLE_DEG = np.array([350.0, 20.0, 50.0, 80.0, 100.0, 145.0, 190.0, 235.0])


def _targets() -> np.ndarray:
    """The `(N, 4)` array `probe.probe_targets` builds: pos_x, pos_y, sin, cos."""
    radians = np.deg2rad(ANGLE_DEG)
    return np.column_stack([POS[:, 0], POS[:, 1], np.sin(radians), np.cos(radians)])


def test_constants_are_the_pre_registered_values():
    """Spec 2.2, 2.3 and 3.1. These are pre-registered, so they are pinned by
    exact equality rather than by range -- the whole point of writing them down
    before the run is that they cannot drift afterwards."""
    assert RETENTION_FAMILY == 3
    assert SEEDS_REQUIRED == 2 and ARMS_REQUIRED == 2
    assert K_REPORTED == (1, 4, 15)
    assert DECISION_K == 4
    assert RUNGS == ("two_frame", "deterministic", "stochastic", "full")
    assert LATENT_RUNGS == ("deterministic", "stochastic", "full")
    assert Z_BEARING_RUNGS == ("stochastic", "full")
    assert TARGETS == ("translation", "rotation")
    assert CONFIDENCE == 0.95 and RESAMPLES == 1000


def test_decision_k_is_one_of_the_reported_horizons():
    """A decision horizon outside `K_REPORTED` would be decided on numbers the
    table never prints."""
    assert DECISION_K in K_REPORTED


def test_latent_rungs_and_z_bearing_rungs_are_subsets_of_rungs():
    """The status logic reads these three tuples; a name in one and not in
    `RUNGS` would be a rung nothing ever measures."""
    assert set(LATENT_RUNGS) < set(RUNGS)
    assert set(Z_BEARING_RUNGS) < set(LATENT_RUNGS)
    assert set(RUNGS) - set(LATENT_RUNGS) == {"two_frame"}


def test_shifted_rows_pairs_each_row_with_the_one_k_steps_back_in_its_window():
    rows, source = shifted_rows(WINDOW, STEP, 2)
    assert rows.tolist() == [2, 3, 6, 7]
    assert source.tolist() == [0, 1, 4, 5]


def test_shifted_rows_never_pairs_across_windows():
    """At k = 3 only the last row of each window qualifies. A pairing that
    walked off the front of window 1 into window 0 would still be shape-valid
    and would silently score a displacement between two different episodes."""
    rows, source = shifted_rows(WINDOW, STEP, 3)
    assert rows.tolist() == [3, 7]
    assert source.tolist() == [0, 4]
    assert np.array_equal(WINDOW[rows], WINDOW[source]), "a pair spans two windows"


def test_shifted_rows_is_order_independent():
    """The pairing comes from the LABELS, not from the array order. Deriving it
    from contiguity would pass on every gather that happens to emit rows in step
    order and break silently on one that does not."""
    straight_rows, straight_source = shifted_rows(WINDOW, STEP, 1)
    order = np.array([5, 0, 7, 2, 4, 1, 6, 3])
    rows, source = shifted_rows(WINDOW[order], STEP[order], 1)
    pairs = {
        (int(WINDOW[order][r]), int(STEP[order][r]), int(STEP[order][s]))
        for r, s in zip(rows, source)
    }
    expected = {
        (int(WINDOW[r]), int(STEP[r]), int(STEP[s]))
        for r, s in zip(straight_rows, straight_source)
    }
    assert pairs == expected


def test_shifted_rows_rejects_a_k_below_one():
    with pytest.raises(ValueError, match="k must be >= 1"):
        shifted_rows(WINDOW, STEP, 0)


def test_shifted_rows_rejects_duplicate_labels():
    """A duplicate (window, step) pair makes the pairing ambiguous. The builder
    must refuse, not silently pick one arbitrarily."""
    window = [0, 0, 0, 0]
    step = [0, 1, 1, 2]
    with pytest.raises(ValueError, match="duplicate label"):
        shifted_rows(window, step, 1)


def test_shifted_rows_returns_nothing_when_no_window_is_long_enough():
    """Not an error: the caller reports the row count and a horizon with no rows
    is a fact about the protocol, not a bug. Returning empty keeps the caller's
    refusal in one place."""
    rows, source = shifted_rows(WINDOW, STEP, 4)
    assert rows.size == 0 and source.size == 0


def test_backward_translation_is_the_displacement_already_observed():
    """`p(t) - p(t-k)`, hand-computed. Window 0 moves (3, -2) per step and
    window 1 moves (2, 3), so at k = 2 the two windows differ -- which is what
    catches a builder that paired every row with row 0 of its own window."""
    values, rows = backward_translation(_targets(), WINDOW, STEP, 2)
    assert rows.tolist() == [2, 3, 6, 7]
    np.testing.assert_allclose(values, [[6.0, -4.0], [6.0, -4.0], [4.0, 6.0], [4.0, 6.0]])


def test_backward_translation_at_k_one_is_the_single_step_move():
    values, rows = backward_translation(_targets(), WINDOW, STEP, 1)
    assert rows.tolist() == [1, 2, 3, 5, 6, 7]
    np.testing.assert_allclose(values, [
        [3.0, -2.0], [3.0, -2.0], [3.0, -2.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0],
    ])


def test_backward_rotation_reads_the_change_in_the_right_direction():
    """Pins the direction of the subtraction: a reversal flips the sign of the
    sin column. The fixture spans the 360° boundary (350°→20° is +30, not −330),
    confirming that the arctan2 reconstruction in backward_rotation needs no
    special wrapping -- it handles the wrap implicitly. Note: no wrap mutation is
    observable through (sin, cos) because both are 2π-periodic; sin(−330°) and
    sin(+30°) are bit-identical, as are their cosines. This test must not be read
    as covering wrap correctness; it only pins the direction of the difference."""
    values, rows = backward_rotation(_targets(), WINDOW, STEP, 1)
    assert rows.tolist() == [1, 2, 3, 5, 6, 7]
    thirty = [np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))]
    forty_five = [np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0))]
    np.testing.assert_allclose(
        values, [thirty, thirty, thirty, forty_five, forty_five, forty_five], atol=1e-12,
    )


def test_backward_rotation_and_translation_agree_on_their_rows():
    """The two targets are scored against the same feature rows at each k. If
    they disagreed on which rows qualify, the rotation control would be measured
    on a different sample than the reading it controls."""
    for k in (1, 2, 3):
        _, translation_rows = backward_translation(_targets(), WINDOW, STEP, k)
        _, rotation_rows = backward_rotation(_targets(), WINDOW, STEP, k)
        assert translation_rows.tolist() == rotation_rows.tolist(), f"k={k}"


def _data() -> dict:
    """The three feature arrays `rung_block` reads, with h_dim = 2 so `h` and
    `z` are DIFFERENT widths -- slicing the wrong end is then a shape error in
    some cases and a wrong number in others, and both must be caught."""
    enc = np.arange(8 * 2, dtype=np.float64).reshape(8, 2) * 1.0
    latent = np.arange(8 * 5, dtype=np.float64).reshape(8, 5) * 10.0
    return {"encoder_embedding": enc, "latent": latent}


def test_rung_block_two_frame_is_the_frame_the_displacement_is_measured_FROM():
    """Spec 2.2: `enc(t-k)`, not `enc(t-1)`. At k = 2 the two differ, and only
    `enc(t-k)` can carry a 2-step displacement together with `enc(t)`."""
    rows, source = shifted_rows(WINDOW, STEP, 2)
    block = rung_block(_data(), "two_frame", rows=rows, source=source, h_dim=2)
    np.testing.assert_array_equal(block, _data()["encoder_embedding"][source])
    # And it is NOT enc(t-1): row 3's partner is row 1, not row 2.
    assert not np.array_equal(block, _data()["encoder_embedding"][rows - 1])


def test_rung_block_splits_the_latent_at_h_dim():
    """`latent` is `cat([h, z])`, h FIRST. The two halves are different widths
    here so a reversed slice is caught by the shape as well as the values."""
    rows, source = shifted_rows(WINDOW, STEP, 1)
    latent = _data()["latent"]
    np.testing.assert_array_equal(
        rung_block(_data(), "deterministic", rows=rows, source=source, h_dim=2),
        latent[rows, :2],
    )
    np.testing.assert_array_equal(
        rung_block(_data(), "stochastic", rows=rows, source=source, h_dim=2),
        latent[rows, 2:],
    )
    np.testing.assert_array_equal(
        rung_block(_data(), "full", rows=rows, source=source, h_dim=2), latent[rows],
    )


def test_rung_block_gives_every_rung_the_same_number_of_rows():
    """All four rungs are differences against ONE shared base fit on ONE row
    set (spec 2.1). A rung with a different row count would be a gain against a
    different base, and the four would stop being comparable."""
    rows, source = shifted_rows(WINDOW, STEP, 2)
    widths = {
        rung: rung_block(_data(), rung, rows=rows, source=source, h_dim=2).shape[0]
        for rung in RUNGS
    }
    assert set(widths.values()) == {rows.size}, widths


def test_rung_block_rejects_an_unknown_rung():
    rows, source = shifted_rows(WINDOW, STEP, 1)
    with pytest.raises(ValueError, match="unknown rung"):
        rung_block(_data(), "recurrent", rows=rows, source=source, h_dim=2)


def test_rung_block_rejects_an_h_dim_that_does_not_index_the_latent():
    """A silently out-of-range slice returns an empty or full block instead of
    raising, and the gain would then be measured on the wrong half."""
    rows, source = shifted_rows(WINDOW, STEP, 1)
    for bad in (0, 5, 9):
        with pytest.raises(ValueError, match="does not index"):
            rung_block(_data(), "deterministic", rows=rows, source=source, h_dim=bad)
