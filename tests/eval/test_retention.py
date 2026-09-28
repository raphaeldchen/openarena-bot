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
    """Pins the DIRECTION of the subtraction: reversing it to `then - now` flips
    the sign of the sin column, which these assertions catch.

    IT DOES NOT COVER WRAP CORRECTNESS, and must not be read as if it did. No
    wrap mutation is observable through this target at all: sin and cos are both
    2π-periodic, so an unwrapped 20° - 350° = -330° and the wrapped +30° agree to
    floating-point precision in BOTH columns (measured: a difference of at most
    1 ULP, ~8e-16, far inside this test's atol). An earlier version of this test
    claimed the opposite -- that the pair "only matches if the wrap is handled" --
    and that claim is false; it is recorded here so it does not get re-added.

    The 350°→20° fixture still earns its place: it spans the 360° boundary and
    shows the result needs no special-casing there. The reason is the periodicity
    above, NOT anything arctan2 does -- a naive raw-degree subtraction that skips
    the reconstruction entirely gives the same sin and cos."""
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


from mbfps.eval.retention import (  # noqa: E402
    BASE_R2_FLOOR,
    BaseControl,
    RetentionInputs,
    RungArm,
    reading_retention,
    rung_arm,
    rung_clears_at,
)

ARMS = ("frozen_ssl", "pixel_ae", "random_vit")


def _arm(ci_low: float, seeds_clear: int | None = None) -> RungArm:
    """A `RungArm` whose clearing is set by `ci_low` and the seed tally alone.

    `RungArm.clears()` reads only `seeds_clear`, never the aggregate `ci_low`
    (proved by `test_rung_arm_reports_the_seed_mean_and_the_least_lower_bound`,
    whose arm clears with a NEGATIVE aggregate `ci_low`) -- so this fixture
    derives the default tally FROM `ci_low`'s sign, unless a test overrides it
    to exercise `ARMS_REQUIRED`/`SEEDS_REQUIRED` directly.
    """
    if seeds_clear is None:
        seeds_clear = 3 if ci_low > 0 else 0
    return RungArm(gain=ci_low + 0.05, ci_low=ci_low, ci_high=ci_low + 0.10,
                   seeds_clear=seeds_clear, seeds_total=3)


def _ladder(clearing: dict) -> dict:
    """`{(target, k, rung): ci_low}` -> the full nested ladder, every unnamed
    cell at ci_low -0.01 (a rung that did not clear)."""
    out = {}
    for target in TARGETS:
        out[target] = {}
        for k in K_REPORTED:
            out[target][k] = {}
            for rung in RUNGS:
                ci_low = clearing.get((target, k, rung), -0.01)
                out[target][k][rung] = {a: _arm(ci_low) for a in ARMS}
    return out


def _inputs(
    clearing: dict, *, base_r2: float = 0.30, base_seeds: int | None = None,
) -> RetentionInputs:
    """Same tally-follows-the-value pattern as `_arm`, for the base control:
    `BaseControl.clears()` reads only `seeds_clear`, so a `base_r2` below
    `BASE_R2_FLOOR` must default to a tally that fails it, not the fixed `3`
    that made every arm hold regardless of `base_r2`."""
    if base_seeds is None:
        base_seeds = 3 if base_r2 > BASE_R2_FLOOR else 0
    return RetentionInputs(
        ladder=_ladder(clearing),
        base={a: BaseControl(r2=base_r2, seeds_clear=base_seeds, seeds_total=3)
              for a in ARMS},
        clusters=24,
        rows={1: 11221, 4: 10534, 15: 8015},
    )


# --- the finiteness guard: M3i's ledger left this as a note for its successor --


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["gain", "ci_low", "ci_high"])
def test_rung_arm_refuses_a_non_finite_value_in_any_field(field, bad):
    """M3i's `clears_up`, `clears_down` and `leaks` ALL evaluate False on NaN, so
    one NaN would read "no clear" and "no leak" at once -- moving a verdict
    toward the wrong status while looking like a clean null. A refusal here is
    the whole point: a non-finite value is an error, never a non-clear.

    3x3 over {gain, ci_low, ci_high} x {nan, inf, -inf}: the shipped guard is a
    uniform `np.isfinite` over all three fields, but a partial guard (e.g.
    `math.isnan(ci_low)` alone) would still pass a suite that only ever put the
    bad value in `gain`, as the two tests this replaces did."""
    seed = {"gain": 0.1, "ci_low": 0.05, "ci_high": 0.2}
    seed[field] = bad
    with pytest.raises(ValueError, match="non-finite"):
        rung_arm([seed])


def test_rung_arm_refuses_an_empty_seed_list():
    with pytest.raises(ValueError, match="at least one seed"):
        rung_arm([])


def test_rung_arm_refuses_fewer_than_seeds_required_seeds():
    """A one-seed arm can never satisfy `seeds_clear >= SEEDS_REQUIRED`, so
    without this guard it silently reads as a null instead of refusing --
    exactly the shape of error `UNRESOLVED_MOTION`/`BOTTLENECK_LOSS` would wear
    if the milestone's finding were actually "not enough data was collected"."""
    with pytest.raises(ValueError, match="SEEDS_REQUIRED"):
        rung_arm([{"gain": 0.1, "ci_low": 0.05, "ci_high": 0.2}])


def test_rung_arm_refuses_a_seed_missing_a_key():
    """A missing key must raise the module's `ValueError` idiom, not a bare
    `KeyError` that looks like a bug rather than a refusal."""
    with pytest.raises(ValueError, match="missing"):
        rung_arm([
            {"gain": 0.1, "ci_low": 0.05},
            {"gain": 0.1, "ci_low": 0.05, "ci_high": 0.2},
        ])


def test_rung_arm_refuses_an_inverted_interval():
    with pytest.raises(ValueError, match="ci_low"):
        rung_arm([
            {"gain": 0.1, "ci_low": 0.2, "ci_high": 0.05},
            {"gain": 0.1, "ci_low": 0.05, "ci_high": 0.2},
        ])


def test_rung_arm_refuses_a_gain_outside_its_own_interval():
    with pytest.raises(ValueError, match="outside"):
        rung_arm([
            {"gain": 0.9, "ci_low": 0.05, "ci_high": 0.2},
            {"gain": 0.1, "ci_low": 0.05, "ci_high": 0.2},
        ])


def test_rung_arm_reports_the_seed_mean_and_the_least_lower_bound():
    """The gain is a MEAN so the printed number describes the arm; `ci_low` is
    the LEAST of the seeds' bounds, which is the conservative summary -- a rung
    is not credited for an interval only its luckiest seed achieved. The tally
    is what decides, and it counts seeds whose OWN bound cleared zero.

    Gains are 0.45/0.20/0.10 rather than the earlier 0.30/0.20/0.10: that
    fixture's mean and median were both 0.20, so it was symmetric under a
    mean -> median substitution in the implementation. 0.45/0.20/0.10 has
    mean 0.25 and median 0.20, which differ."""
    arm = rung_arm([
        {"gain": 0.45, "ci_low": 0.10, "ci_high": 0.50},
        {"gain": 0.20, "ci_low": -0.05, "ci_high": 0.45},
        {"gain": 0.10, "ci_low": 0.02, "ci_high": 0.18},
    ])
    assert arm.gain == pytest.approx(0.25)
    assert arm.ci_low == pytest.approx(-0.05)
    assert arm.ci_high == pytest.approx(0.50), "the GREATEST upper bound, not the mean"
    assert arm.seeds_clear == 2 and arm.seeds_total == 3
    assert arm.clears() is True, "2 of 3 seeds is SEEDS_REQUIRED"


def test_rung_arm_clearing_is_one_sided():
    """A NEGATIVE gain means appending the block made held-out R^2 worse: noise
    or selection slack, not a finding. M3i's control was two-sided on an
    argument its own results refuted, and the lesson is applied here rather than
    re-derived. A seed whose whole interval sits below zero must not clear."""
    arm = rung_arm([
        {"gain": -0.40, "ci_low": -0.60, "ci_high": -0.20},
        {"gain": -0.35, "ci_low": -0.55, "ci_high": -0.15},
        {"gain": -0.30, "ci_low": -0.50, "ci_high": -0.10},
    ])
    assert arm.seeds_clear == 0 and arm.clears() is False


def test_rung_arm_does_not_clear_on_an_interval_that_only_touches_zero():
    """`ci_low > 0`, strictly. An interval whose lower bound is exactly zero has
    not excluded zero."""
    arm = rung_arm([{"gain": 0.1, "ci_low": 0.0, "ci_high": 0.2}] * 3)
    assert arm.seeds_clear == 0 and arm.clears() is False


# --- the ladder ------------------------------------------------------------


def test_rung_clears_at_reports_every_horizon_the_rung_cleared():
    """The rule is a DISJUNCTION over `K_REPORTED` (spec 2.3), so the reading
    needs the horizons, not just a boolean -- they go into the rule text."""
    inputs = _inputs({("translation", 1, "full"): 0.2,
                      ("translation", 15, "full"): 0.3})
    assert rung_clears_at(inputs, "translation", "full") == (1, 15)
    assert rung_clears_at(inputs, "translation", "two_frame") == ()


def test_rung_clears_at_needs_arms_required_arms():
    """One arm clearing is not the rule. Two of three is."""
    inputs = _inputs({})
    inputs.ladder["translation"][4]["full"]["pixel_ae"] = _arm(0.5)
    assert rung_clears_at(inputs, "translation", "full") == ()
    inputs.ladder["translation"][4]["full"]["frozen_ssl"] = _arm(0.5)
    assert rung_clears_at(inputs, "translation", "full") == (4,)


def test_rung_clears_at_needs_seeds_required_seeds_in_each_arm():
    """An arm whose pooled bound clears on ONE seed has not replicated."""
    inputs = _inputs({})
    for arm in ("pixel_ae", "frozen_ssl"):
        inputs.ladder["translation"][4]["full"][arm] = _arm(0.5, seeds_clear=1)
    assert rung_clears_at(inputs, "translation", "full") == ()


# --- the six statuses, one test each ---------------------------------------


def test_reading_is_unresolved_base_when_the_current_frame_cannot_locate_itself():
    """The one true control failure, and it outranks every result. If `enc(t)`
    cannot linearly say where it is, the instrument is broken and a null on
    displacement means nothing.

    `base` is rebuilt here in REVERSE of `ARMS` order: `ARMS` is already
    alphabetically sorted, so building it in `ARMS` order (as `_inputs` does)
    would leave `reading_retention`'s `sorted()` call unexercised -- the
    assertion below would pass even if that call were deleted."""
    inputs = _inputs({("translation", 4, "full"): 0.5}, base_r2=0.01)
    inputs = RetentionInputs(
        ladder=inputs.ladder,
        base={a: inputs.base[a] for a in reversed(ARMS)},
        clusters=inputs.clusters,
        rows=inputs.rows,
    )
    reading = reading_retention(inputs)
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.base_failed == tuple(sorted(ARMS))
    assert reading.surviving is None
    assert reading.translation_rungs == (), (
        "a suppressed reading must report no rungs at all, not report them "
        "beside a warning nobody reads"
    )


def test_reading_is_taken_when_only_one_base_arm_fails():
    """`ARMS_REQUIRED` = 2 of 3, so the gate must permit exactly one arm to
    fail. Verified survivors of the shipped threshold `holding <
    ARMS_REQUIRED`: replacing it with `bool(base_failed)` refuses whenever ANY
    arm fails (this test would then wrongly get `UNRESOLVED_BASE`), and with
    `holding == 0` refuses only when ALL arms fail (caught instead by the two-
    arm-failure test below). This test and that one together pin the
    threshold at exactly `ARMS_REQUIRED`."""
    inputs = _inputs({("translation", 4, "full"): 0.3})
    inputs.base["pixel_ae"] = BaseControl(r2=0.01, seeds_clear=0, seeds_total=3)
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_RETAINED", (
        "one failing arm out of three must still let a reading be taken"
    )


def test_reading_is_unresolved_base_when_two_of_three_arms_fail():
    """Two failures leave only one holding arm, below `ARMS_REQUIRED` = 2."""
    inputs = _inputs({("translation", 4, "full"): 0.3})
    inputs.base["pixel_ae"] = BaseControl(r2=0.01, seeds_clear=0, seeds_total=3)
    inputs.base["random_vit"] = BaseControl(r2=0.01, seeds_clear=0, seeds_total=3)
    reading = reading_retention(inputs)
    assert reading.status == "UNRESOLVED_BASE"


def test_reading_records_the_real_base_failed_on_a_normal_reading():
    """The gate permits one arm to fail while still taking a reading (spec:
    `ARMS_REQUIRED` of `RETENTION_FAMILY`). Every non-refusal branch used to
    hardcode `base_failed=()`, which would have this reading claim a clean
    positive control despite one arm having actually failed -- and Tasks 6/7
    write `RetentionStatus` into a record, so that false claim would persist.
    `UNRESOLVED_BASE`'s suppression of `translation_rungs`/`rotation_rungs` is
    a different, spec-required thing and is untouched by this."""
    inputs = _inputs({("translation", 4, "full"): 0.3})
    inputs.base["pixel_ae"] = BaseControl(r2=0.01, seeds_clear=0, seeds_total=3)
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_RETAINED"
    assert reading.base_failed == ("pixel_ae",)


def test_reading_is_motion_retained_when_a_z_bearing_rung_clears():
    inputs = _inputs({("translation", 15, "stochastic"): 0.2})
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_RETAINED"
    assert reading.surviving == "stochastic"
    assert "15" in reading.rule


def test_reading_is_motion_retained_even_when_two_frame_is_silent():
    """`h` integrates the ACTION sequence, which two frames do not contain, so a
    latent rung can clear where `two_frame` does not. An earlier draft of the
    spec refused in exactly this case by putting UNRESOLVED_MOTION second; this
    is the test that would have caught it."""
    inputs = _inputs({("translation", 4, "full"): 0.3})
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_RETAINED"
    assert reading.translation_rungs == ("full",)


def test_the_top_of_the_ladder_wins_when_every_rung_clears():
    """The one state that pins the ladder's ORDER rather than its membership.

    Both other MOTION_RETAINED fixtures clear exactly one rung, so they read the
    same under any reordering of the status checks. Measured: with
    `deterministic` checked before the z-bearing rungs, this state reads
    BOTTLENECK_LOSS/deterministic instead -- which would send the project at the
    32x32 bottleneck and `rep_scale` when the finding is that motion survived
    INTO z and M3i is partially overturned. That is the wrong-lever error this
    whole milestone exists to avoid, and until this test existed the suite was
    green under it.

    `surviving` is `stochastic` rather than `full` because `Z_BEARING_RUNGS` is
    traversed in order and `stochastic` is the stronger claim: motion survived
    into z alone, not merely into the concatenation that still carries h.
    """
    inputs = _inputs({("translation", 4, rung): 0.2 for rung in RUNGS})
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_RETAINED"
    assert reading.surviving == "stochastic"


def test_reading_is_bottleneck_loss_when_only_the_deterministic_rung_clears():
    """`h` carries it and `z` destroys it: the lever is the 32x32 bottleneck and
    `rep_scale`, not the objective."""
    inputs = _inputs({("translation", 4, "deterministic"): 0.2,
                      ("translation", 4, "two_frame"): 0.4})
    reading = reading_retention(inputs)
    assert reading.status == "BOTTLENECK_LOSS"
    assert reading.surviving == "deterministic"


def test_reading_is_motion_discarded_when_only_two_frame_clears():
    """The information is available and the RSSM throws it away: the lever is
    the objective, which never asks for motion."""
    inputs = _inputs({("translation", 4, "two_frame"): 0.4,
                      ("rotation", 4, "two_frame"): 0.5})
    reading = reading_retention(inputs)
    assert reading.status == "MOTION_DISCARDED"
    assert reading.surviving == "two_frame"


def test_reading_is_translation_unresolved_when_only_rotation_reads():
    """Nothing reads translation but something reads rotation: the lever is the
    encoder's INPUT -- resolution, stride, frame stacking -- not the loss."""
    inputs = _inputs({("rotation", 4, "two_frame"): 0.5,
                      ("rotation", 15, "full"): 0.3})
    reading = reading_retention(inputs)
    assert reading.status == "TRANSLATION_UNRESOLVED"
    assert reading.surviving is None
    assert reading.rotation_rungs == ("two_frame", "full")


def test_reading_is_unresolved_motion_when_nothing_clears_anything():
    """No rung reads either target: the measurement detects no motion anywhere,
    so no lever is chosen and the next milestone is about the instrument. This
    requires the WHOLE ladder to be silent on BOTH targets, not `two_frame`
    alone."""
    reading = reading_retention(_inputs({}))
    assert reading.status == "UNRESOLVED_MOTION"
    assert reading.surviving is None
    assert reading.translation_rungs == () and reading.rotation_rungs == ()


def test_every_status_is_reachable():
    """All six statuses must be reachable from SOME combination of (which rungs
    clear translation, which clear rotation, does the base hold). M3i shipped a
    status nothing exercised until the final review; this is the cheap version
    of that check.

    This collects only the SET of statuses seen across every combination, so it
    is invariant under any reordering of `reading_retention`'s status checks --
    it pins REACHABILITY, not which combination maps to which status. That
    mapping is pinned by the per-status tests above, including
    `test_the_top_of_the_ladder_wins_when_every_rung_clears`, which is the one
    combination (every rung clearing at once) none of the others exercise and
    that is therefore free to read as the wrong status under a reordering while
    this test still sees all six and stays green."""
    import itertools

    seen = set()
    for base_ok in (True, False):
        for t_rungs in itertools.chain.from_iterable(
            itertools.combinations(RUNGS, n) for n in range(len(RUNGS) + 1)
        ):
            for r_rungs in ((), ("two_frame",), ("full",)):
                clearing = {("translation", 4, r): 0.2 for r in t_rungs}
                clearing.update({("rotation", 4, r): 0.2 for r in r_rungs})
                reading = reading_retention(
                    _inputs(clearing, base_r2=0.30 if base_ok else 0.0)
                )
                seen.add(reading.status)
    expected = {
        "UNRESOLVED_BASE", "MOTION_RETAINED", "BOTTLENECK_LOSS",
        "MOTION_DISCARDED", "TRANSLATION_UNRESOLVED", "UNRESOLVED_MOTION",
    }
    assert seen == expected, (
        f"unreachable: {sorted(expected - seen)}; unexpected: {sorted(seen - expected)}"
    )


def test_base_control_floor_is_below_every_recorded_latent_selection_r2():
    """`BASE_R2_FLOOR` must fail only when the instrument is broken, not when a
    cell is merely weak. The M3c records' `latent_selection_r2` runs 0.18-0.34
    (one outlier at -0.008 on the LATENT, not on the 2048-d encoder output), and
    M3i measured `enc(t)` -> position at +0.291 and +0.165 from 132 rows. The
    floor sits below all of those and well above zero."""
    assert 0.0 < BASE_R2_FLOOR < 0.165


def test_base_r2_floor_is_the_pre_registered_value():
    """Pre-registered (spec), like every other constant in this module, so it
    is pinned by exact equality rather than only by the provenance range
    above -- that range alone would let the value drift anywhere inside it."""
    assert BASE_R2_FLOOR == 0.10


# --- shape refusals: a short or empty collection is a silent null, not data -


def test_reading_retention_refuses_a_ladder_cell_with_too_few_arms():
    """A cell with fewer than `RETENTION_FAMILY` arms must refuse rather than
    silently read as "did not clear", which would surface as a pre-registered
    finding manufactured from insufficient data. If left unguarded, an empty
    ladder cell reaches `UNRESOLVED_MOTION` -- whose rule text asserts "a
    linear read detects no motion anywhere on the path" -- from missing data
    rather than from an actual reading."""
    inputs = _inputs({("translation", 4, "full"): 0.3})
    del inputs.ladder["translation"][4]["full"]["pixel_ae"]
    with pytest.raises(ValueError, match="arm"):
        reading_retention(inputs)


def test_reading_retention_refuses_a_base_with_too_few_arms():
    """An empty (or short) `base` must refuse rather than take `holding = 0` as
    "0 of 0 arms failed" and report `UNRESOLVED_BASE` from no data at all."""
    inputs = _inputs({("translation", 4, "full"): 0.3})
    del inputs.base["pixel_ae"]
    with pytest.raises(ValueError, match="arm"):
        reading_retention(inputs)


# --- the printed tables: captions pinned to their columns -------------------


from mbfps.eval.retention import (  # noqa: E402
    LADDER_COLUMNS,
    LADDER_WIDTHS,
    READING_COLUMNS,
    READING_WIDTHS,
    format_ladder,
    format_reading_retention,
)


def test_reading_columns_and_widths_stay_the_same_length():
    """Header and rows are two separate f-strings built from these tuples. A
    length mismatch means one column's caption sits over another's values --
    the defect class this project has shipped three times."""
    assert len(READING_COLUMNS) == len(READING_WIDTHS)
    assert len(LADDER_COLUMNS) == len(LADDER_WIDTHS)


def test_every_reading_width_admits_its_widest_realistic_value():
    """A value as wide as its field glues onto the previous column with no
    separator. `seeds` holds "3/3" (3 chars), `clears` holds "yes"/"no", `arm`
    holds "frozen_ssl" (10) and "random_vit" (10), and a gain prints as
    "+0.1234" or "-12.3456" (8). Every width must EXCEED, not equal."""
    widest = {
        "rung": len("deterministic"), "arm": len("frozen_ssl"),
        "gain": len("-12.3456"), "ci_low": len("-12.3456"),
        "ci_high": len("-12.3456"), "seeds": len("3/3"), "clears": len("yes"),
    }
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        assert width > widest[name], (
            f"column {name!r} is {width} wide but holds up to {widest[name]} "
            "characters; a full-width value glues onto its left neighbour"
        )


def test_reading_header_captions_the_columns_it_prints():
    """The header and the rows must agree COLUMN BY COLUMN, sliced at the same
    offsets. Asserting the two strings look plausible is what let three wrong
    captions ship."""
    inputs = _inputs({("translation", DECISION_K, "full"): 0.2})
    text = format_reading_retention(reading_retention(inputs), inputs)
    lines = [line for line in text.splitlines() if line.startswith("  ")]
    header, first = lines[0], lines[1]
    offset = 2
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        assert header[offset:offset + width].strip() == name, (
            f"header column at offset {offset} is not {name!r}"
        )
        assert first[offset:offset + width].strip() != "", (
            f"the first row has nothing under the {name!r} caption"
        )
        offset += width


def test_reading_names_the_target_the_horizon_and_the_clusters_in_its_caption():
    """A table whose caption does not say what its numbers are is how this
    project shipped a wrong number three times. The caption has to name the
    statistic (a gain over enc(t), not a level), the target, the horizon, the
    one-sided rule and the cluster count."""
    inputs = _inputs({})
    caption = format_reading_retention(reading_retention(inputs), inputs).splitlines()[0]
    assert "gain over enc(t)" in caption
    assert "translation" in caption
    assert f"k = {DECISION_K}" in caption
    assert "ci_low > 0" in caption
    assert "24 clusters" in caption


def test_reading_prints_the_verdict_and_its_rule():
    inputs = _inputs({("translation", 4, "deterministic"): 0.2})
    text = format_reading_retention(reading_retention(inputs), inputs)
    assert "verdict: BOTTLENECK LOSS" in text
    assert "decided by:" in text


def test_reading_prints_the_base_control_beside_the_verdict():
    """The control is a gate, so its numbers belong next to the reading it
    licensed rather than in a companion nobody reads."""
    inputs = _inputs({})
    text = format_reading_retention(reading_retention(inputs), inputs)
    assert "base control (enc(t) -> absolute position" in text
    assert "frozen_ssl r2=+0.300" in text


def test_reading_prints_the_rotation_control_beside_the_verdict():
    """Reading rotation but not translation is a DIFFERENT lever from reading
    neither, so which rungs read rotation has to be on the face of the verdict."""
    inputs = _inputs({("rotation", 4, "two_frame"): 0.5})
    text = format_reading_retention(reading_retention(inputs), inputs)
    assert "rotation control" in text
    assert "two_frame" in text.split("rotation control")[1].splitlines()[0]


def test_ladder_prints_every_rung_at_every_horizon_for_both_targets():
    """`K_REPORTED` x `RUNGS` x `TARGETS` rows, all of them, because the rule is
    a disjunction over k and a reader has to be able to check it."""
    inputs = _inputs({("translation", 15, "full"): 0.3})
    rows = [
        line for line in format_ladder(inputs).splitlines()
        if line.startswith("  ") and "target" not in line
    ]
    assert len(rows) == len(TARGETS) * len(RUNGS) * len(K_REPORTED)


def test_ladder_header_captions_the_columns_it_prints():
    inputs = _inputs({})
    lines = [line for line in format_ladder(inputs).splitlines() if line.startswith("  ")]
    header, first = lines[0], lines[1]
    offset = 2
    for name, width in zip(LADDER_COLUMNS, LADDER_WIDTHS, strict=True):
        assert header[offset:offset + width].strip() == name
        assert first[offset:offset + width].strip() != ""
        offset += width


def test_ladder_says_it_decides_nothing_on_its_own():
    """Every horizon contributes to the disjunction, so no row "decides" alone.
    M3i printed `decides: no` beside every companion row for the opposite reason
    -- one horizon decided and the rest did not -- and a reader carrying that
    habit across would misread this table without the caption."""
    caption = format_ladder(_inputs({})).splitlines()[0]
    assert "any horizon counts" in caption
