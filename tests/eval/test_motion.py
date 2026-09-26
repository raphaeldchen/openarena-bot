"""src/mbfps/eval/motion.py: the pre-registered constants, the displacement a
window actually travelled, and the paired contrast against staying put."""

import numpy as np
import pytest

from mbfps.eval.motion import (
    ARMS_REQUIRED,
    CONTROL_SEED,
    K_REPORTED,
    MOTION_FAMILY,
    READING_COLUMNS,
    SEEDS_REQUIRED,
    MotionArm,
    MotionInputs,
    contrast_series,
    displacement,
    format_reading_displacement,
    latent_description,
    motion_threshold,
    reading_displacement,
)
from mbfps.eval.split_gap import DECISION_H


def test_the_pre_registered_constants():
    """Spec 3.1 and 3.2, fixed before the run and never tuned after it."""
    assert MOTION_FAMILY == 3
    assert SEEDS_REQUIRED == 2
    assert ARMS_REQUIRED == 2
    assert K_REPORTED == (1, 5, 15, 30, 45)
    assert CONTROL_SEED == 0
    assert DECISION_H == 15, "the decision horizon is split_gap's, not a second spelling"
    assert DECISION_H in K_REPORTED, "the decided horizon must also be reported"


def test_the_family_threshold_is_the_projects_cluster_rule():
    """Three arms over the shipped 24 episode clusters. Hand-typed: a bar that
    drifts with a refactor is a bar that decided nothing."""
    assert motion_threshold(24) == pytest.approx(2.5820, abs=5e-5)


def test_displacement_is_the_vector_a_window_travelled_over_k_steps():
    """`p(t+k) - p(t)` per window, in map units, from the first step onward.
    Not a speed and not a distance: the probe predicts a VECTOR."""
    positions = np.array([
        [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],     # window 0: +3,+4 a step
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],     # window 1: never moves
    ])
    np.testing.assert_allclose(displacement(positions, 1), [[3.0, 4.0], [0.0, 0.0]])
    np.testing.assert_allclose(displacement(positions, 2), [[6.0, 8.0], [0.0, 0.0]])


def test_displacement_refuses_a_k_the_window_is_too_short_for():
    positions = np.zeros((2, 3, 2))
    with pytest.raises(ValueError, match="k=5"):
        displacement(positions, 5)


def test_contrast_is_persistence_error_minus_model_error():
    """Positive means the latent beat staying put. `error_persist` is `||d||`
    because the persistence baseline predicts no displacement at all."""
    true = np.array([[3.0, 4.0], [0.0, 10.0]])        # norms 5 and 10
    predicted = np.array([[3.0, 0.0], [0.0, 10.0]])   # errors 4 and 0
    np.testing.assert_allclose(contrast_series(predicted, true), [1.0, 10.0])


def test_a_model_that_predicts_nothing_scores_exactly_zero():
    """The persistence baseline IS the zero prediction, so a probe that
    outputs zeros must contrast at exactly 0.0 -- not approximately. This is
    the fixed point the whole reading is read against."""
    true = np.array([[3.0, 4.0], [-6.0, 8.0], [0.0, 0.0]])
    np.testing.assert_array_equal(contrast_series(np.zeros_like(true), true), np.zeros(3))


def test_contrast_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        contrast_series(np.zeros((3, 2)), np.zeros((4, 2)))


Z = 2.5820


def _arm(z, up=3, down=0, total=3):
    """An arm whose estimate carries the sign of its z, so a test that flips a
    z does not leave an estimate contradicting it."""
    return MotionArm(estimate=0.1 * z, se=0.1, z=z, seeds_up=up, seeds_down=down, seeds_total=total)


def _inputs(arms, control=None, k=15):
    clean = {a: _arm(0.4, up=0, down=0) for a in arms}
    return MotionInputs(
        arms=arms, control=control if control is not None else clean,
        z_fam=Z, k=k, clusters=24,
    )


def test_two_arms_clearing_up_in_two_seeds_is_motion_encoded():
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0, up=1),
    }))
    assert reading.status == "MOTION_ENCODED"
    assert reading.arms_up == ("frozen_ssl", "pixel_ae")
    assert "2 of 3 arms" in reading.rule


def test_one_arm_clearing_up_is_not_enough():
    """ARMS_REQUIRED = 2. A single arm clearing is one cell's worth of
    evidence wearing a family-corrected bar."""
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(5.0, up=3), "frozen_ssl": _arm(0.2), "random_vit": _arm(0.1),
    }))
    assert reading.status == "NO_DIFFERENCE"


def test_an_arm_clearing_up_on_one_seed_does_not_count():
    """SEEDS_REQUIRED = 2, so a pooled clear carried by a single seed is not
    an arm that cleared."""
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(4.0, up=1), "frozen_ssl": _arm(4.0, up=1), "random_vit": _arm(0.1),
    }))
    assert reading.status == "NO_DIFFERENCE"


def test_no_arm_up_and_some_arm_down_is_no_motion():
    """The latent is measurably WORSE than staying put -- the result that
    retires M3h section 8's three prior-side levers."""
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(-3.3, up=0, down=2), "frozen_ssl": _arm(-4.0, up=0, down=3),
        "random_vit": _arm(-0.5, up=0, down=0),
    }))
    assert reading.status == "NO_MOTION"
    assert reading.arms_down == ("frozen_ssl", "pixel_ae")


def test_one_arm_up_and_two_down_is_not_NO_MOTION():
    """NO_MOTION requires that NO arm cleared positively -- not merely that
    enough arms cleared negatively.

    The distinction matters because NO_MOTION is the status that retires three
    training levers. An arm that beat staying put, sitting beside arms that
    lost to it, is a split result and must read as NO_DIFFERENCE; calling it
    NO_MOTION would retire those levers on evidence that contradicts itself.
    The guard is `not up`, and a later `len(up) < ARMS_REQUIRED` would pass
    every other test in this file.
    """
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(4.0, up=3), "frozen_ssl": _arm(-4.0, up=0, down=3),
        "random_vit": _arm(-3.5, up=0, down=2),
    }))
    assert reading.status == "NO_DIFFERENCE"
    assert reading.arms_up == ("pixel_ae",)
    assert reading.arms_down == ("frozen_ssl", "random_vit")


def test_nothing_clearing_either_way_is_no_difference():
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(1.0), "frozen_ssl": _arm(-1.2), "random_vit": _arm(0.3),
    }))
    assert reading.status == "NO_DIFFERENCE"


def test_a_leaking_control_suppresses_a_result_that_would_otherwise_pass():
    """Precedence: UNRESOLVED_CONTROL outranks every result. The permuted
    pairing cannot carry signal, so a control that clears means the instrument
    is reading structure that does not exist -- and the reading it would
    otherwise have printed is exactly the one not to trust."""
    arms = {"pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0)}
    control = {"pixel_ae": _arm(0.1, up=0), "frozen_ssl": _arm(3.9, up=0), "random_vit": _arm(0.2)}
    reading = reading_displacement(_inputs(arms, control=control))
    assert reading.status == "UNRESOLVED_CONTROL"
    assert reading.leaked == ("frozen_ssl",)
    assert reading.arms_up == (), "a suppressed reading reports no result"


def test_the_control_leaks_on_a_NEGATIVE_clear_too():
    """The control is two-sided: a permuted pairing that is reliably WORSE
    than chance is as much a broken instrument as one that is better."""
    arms = {"pixel_ae": _arm(0.2), "frozen_ssl": _arm(0.1), "random_vit": _arm(0.3)}
    control = {"pixel_ae": _arm(-4.5), "frozen_ssl": _arm(0.1), "random_vit": _arm(0.2)}
    assert reading_displacement(_inputs(arms, control=control)).status == "UNRESOLVED_CONTROL"


def test_the_rule_sentence_names_the_horizon_it_was_decided_at():
    reading = reading_displacement(_inputs({
        "pixel_ae": _arm(0.1), "frozen_ssl": _arm(0.1), "random_vit": _arm(0.1)}, k=15))
    assert "k = 15" in reading.rule


def test_every_column_header_is_printed_in_the_declared_order():
    """The caption-column pairing, pinned. Three defects of this exact class
    have shipped on this project and every one was caught by a person reading
    output rather than by a test."""
    arms = {"pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0)}
    text = format_reading_displacement(reading_displacement(_inputs(arms)), _inputs(arms))
    header = next(line for line in text.splitlines() if "estimate" in line)
    assert header.split() == list(READING_COLUMNS)


def test_each_arms_row_carries_its_own_numbers_under_those_headers():
    """A swapped column is the defect this pins: the seed tallies differ
    between the arms here, so up/down transposed or an arm's row taking its
    neighbour's numbers both fail.

    `estimate` and `se` are given distinct, non-uniform values per arm (the
    shared `_arm` helper fixes `se=0.1` for every arm, which would make an
    `estimate`/`se` swap invisible) so a swap between those two columns, or
    between two arms' rows, changes what a hand-typed assertion sees. The
    `clears` column is pinned for three arms so "up"/"down" transposed in
    that branch -- as opposed to merely omitted -- also fails.
    """
    arms = {
        "pixel_ae": MotionArm(
            estimate=0.3000, se=0.1000, z=3.0, seeds_up=2, seeds_down=0, seeds_total=3,
        ),
        "frozen_ssl": MotionArm(
            estimate=-0.4100, se=0.2500, z=-4.1, seeds_up=0, seeds_down=3, seeds_total=3,
        ),
        "random_vit": MotionArm(
            estimate=0.0700, se=0.0400, z=1.0, seeds_up=1, seeds_down=0, seeds_total=3,
        ),
    }
    inputs = _inputs(arms)
    text = format_reading_displacement(reading_displacement(inputs), inputs)
    rows = {line.split()[0]: line.split() for line in text.splitlines()
            if line.strip().startswith(("pixel_ae", "frozen_ssl", "random_vit"))}
    cols = list(READING_COLUMNS)
    assert rows["pixel_ae"][cols.index("estimate")] == "0.3000"
    assert rows["pixel_ae"][cols.index("se")] == "0.1000"
    assert rows["pixel_ae"][cols.index("z")] == "3.00"
    assert rows["pixel_ae"][cols.index("up")] == "2/3"
    assert rows["pixel_ae"][cols.index("dn")] == "0/3"
    assert rows["pixel_ae"][cols.index("clears")] == "up"
    assert rows["frozen_ssl"][cols.index("estimate")] == "-0.4100"
    assert rows["frozen_ssl"][cols.index("se")] == "0.2500"
    assert rows["frozen_ssl"][cols.index("z")] == "-4.10"
    assert rows["frozen_ssl"][cols.index("up")] == "0/3"
    assert rows["frozen_ssl"][cols.index("dn")] == "3/3"
    assert rows["frozen_ssl"][cols.index("clears")] == "down"
    assert rows["random_vit"][cols.index("estimate")] == "0.0700"
    assert rows["random_vit"][cols.index("se")] == "0.0400"
    assert rows["random_vit"][cols.index("clears")] == "no"


def test_extreme_values_do_not_glue_onto_the_previous_column():
    """A value that meets or exceeds its own field's width is printed with no
    separating space, so it runs into whatever the previous column printed --
    the arm-name-overflow defect this project already shipped once,
    relocated to a numeric column (e.g. a three-digit seed total's `120/120`
    is 7 characters, which used to overflow a 6-wide field).

    Deliberately generic: rather than re-checking only the specific fields
    someone hand-picked, this asserts every arm row splits into exactly
    `len(READING_COLUMNS)` whitespace-separated tokens -- which fails if ANY
    column glues onto its neighbour, including a column added later.
    """
    arms = {
        "pixel_ae": MotionArm(
            estimate=-123456.7890, se=0.1000, z=-100.00,
            seeds_up=0, seeds_down=120, seeds_total=120,
        ),
        "frozen_ssl": _arm(0.4),
        "random_vit": _arm(0.4),
    }
    inputs = _inputs(arms)
    text = format_reading_displacement(reading_displacement(inputs), inputs)
    rows = [line for line in text.splitlines()
            if line.strip().startswith(("pixel_ae", "frozen_ssl", "random_vit"))]
    assert len(rows) == 3
    for row in rows:
        assert len(row.split()) == len(READING_COLUMNS), row


def test_the_table_prints_the_status_and_its_rule():
    arms = {"pixel_ae": _arm(-3.3, up=0, down=2), "frozen_ssl": _arm(-4.0, up=0, down=3),
            "random_vit": _arm(-0.5)}
    inputs = _inputs(arms)
    text = format_reading_displacement(reading_displacement(inputs), inputs)
    assert "NO MOTION" in text
    assert "decided by:" in text
    assert text.endswith("\n")


def _logits(rows):
    """`(1, T, G, C)` from a list of per-step, per-group class indices, with a
    sharp one-hot at each. Sharp on purpose: a hand-typed entropy of 0 and a
    top-1 mass of 1 are values a reader can check without running anything."""
    rows = np.asarray(rows)
    t, g = rows.shape
    out = np.full((1, t, g, 4), -20.0)
    for i in range(t):
        for j in range(g):
            out[0, i, j, rows[i, j]] = 20.0
    return out


def test_a_sharp_latent_has_zero_entropy_and_full_top1_mass():
    post = _logits([[0, 1], [0, 1]])
    d = latent_description(post, post)
    assert d["entropy_mean"] == pytest.approx(0.0, abs=1e-6)
    assert d["top1_posterior"] == pytest.approx(1.0, abs=1e-6)
    assert d["top1_prior"] == pytest.approx(1.0, abs=1e-6)
    assert len(d["entropy_by_group"]) == 2


def test_a_uniform_latent_has_log_C_entropy():
    """Four classes, so the ceiling is log 4 = 1.386 nats -- the number a
    reader compares the real cells against."""
    post = np.zeros((1, 2, 3, 4))
    d = latent_description(post, post)
    assert d["entropy_mean"] == pytest.approx(np.log(4.0), abs=1e-6)
    assert d["entropy_max"] == pytest.approx(np.log(4.0), abs=1e-6)
    assert d["top1_posterior"] == pytest.approx(0.25, abs=1e-6)


def test_live_groups_counts_the_groups_whose_argmax_ever_changes():
    """Group 0 changes class between steps, group 1 never does. A latent whose
    groups are mostly constant is far smaller than G x C in effect, which is
    the whole reason this statistic is recorded."""
    post = _logits([[0, 1], [2, 1], [2, 1]])
    assert latent_description(post, post)["live_groups"] == pytest.approx(1.0)


def test_live_groups_counts_a_group_that_changes_and_reverts():
    """The case that separates "changed at ANY step" from "differs between the
    first and last step" -- and the only kind of input where the two diverge.

    Group 0 goes 0 -> 2 -> 0: it moved, and a latent that moved is live
    however it ended up. Comparing only the endpoints would call it dead and
    understate how much of the code is doing anything, which is the whole
    quantity this statistic exists to report. Without this case both
    semantics pass every other live-groups test in this file.
    """
    post = _logits([[0, 1], [2, 1], [0, 1]])
    assert latent_description(post, post)["live_groups"] == pytest.approx(1.0)


def test_live_groups_is_zero_for_a_latent_that_never_moves():
    post = _logits([[3, 3], [3, 3]])
    assert latent_description(post, post)["live_groups"] == pytest.approx(0.0)


def test_the_description_refuses_a_posterior_and_prior_of_different_shapes():
    """They describe the same windows, so a disagreement is a wiring bug in
    the pass that produced them -- and averaging over it anyway would report
    two different things under one cell's name."""
    post = _logits([[0, 1]])
    prior = np.zeros((1, 1, 3, 4))
    with pytest.raises(ValueError, match="same windows"):
        latent_description(post, prior)


def test_the_description_reads_the_prior_separately_from_the_posterior():
    """A swapped argument is the defect this catches: the two are different
    distributions here, so transposing them changes both top-1 masses."""
    post = _logits([[0, 0]])
    prior = np.zeros((1, 1, 2, 4))
    d = latent_description(post, prior)
    assert d["top1_posterior"] == pytest.approx(1.0, abs=1e-6)
    assert d["top1_prior"] == pytest.approx(0.25, abs=1e-6)


def test_a_suppressed_reading_prints_the_control_and_no_verdict_row():
    """UNRESOLVED_CONTROL must not print a table a reader could mistake for a
    result -- but suppression is a file-write decision owned by the script
    that later writes `motion.txt`, not a formatting one: an operator
    debugging a leaking control needs to see the per-arm rows on the console,
    so this formatter prints them unconditionally. Pinned here so a future
    change that wrongly hides the rows on this path fails: the constraint's
    own wording ("must not print a table a reader could mistake for a
    result") makes hiding them an easy, plausible-sounding mistake.
    """
    arms = {"pixel_ae": _arm(3.0, up=2), "frozen_ssl": _arm(4.1, up=3), "random_vit": _arm(1.0)}
    control = {"pixel_ae": _arm(0.1), "frozen_ssl": _arm(3.9), "random_vit": _arm(0.2)}
    inputs = _inputs(arms, control=control)
    text = format_reading_displacement(reading_displacement(inputs), inputs)
    assert "UNRESOLVED CONTROL" in text
    assert "MOTION ENCODED" not in text
    row_starts = [line.strip().split()[0] for line in text.splitlines() if line.strip()]
    for arm in ("pixel_ae", "frozen_ssl", "random_vit"):
        assert arm in row_starts, f"the {arm} row must still print under UNRESOLVED_CONTROL"
