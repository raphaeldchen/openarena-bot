"""src/mbfps/eval/motion.py: the pre-registered constants, the displacement a
window actually travelled, and the paired contrast against staying put."""

import numpy as np
import pytest

from mbfps.eval.motion import (
    ARMS_REQUIRED,
    CONTROL_SEED,
    K_REPORTED,
    MOTION_FAMILY,
    SEEDS_REQUIRED,
    contrast_series,
    displacement,
    motion_threshold,
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


from mbfps.eval.motion import (
    MotionArm,
    MotionInputs,
    reading_displacement,
)

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
