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
