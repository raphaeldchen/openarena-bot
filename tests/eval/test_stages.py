"""mbfps.eval.stages, the pure half of M3g: the latent-space statistics of
spec 2.3 on fabricated logits with hand-typed answers, and (Task 3) the
reading of spec 3.2 on fabricated contrasts.

The rig is deliberately small -- 2 windows, 3 window steps (1 context, 2
horizon), 2 groups of 3 classes -- so every expected number is typed by hand
in the test and every function is pinned against a value, not against
itself. Logits are built from class indices with `sharp` (1000 at the class:
after the softmax the other classes underflow to exactly 0, so entropies and
NLLs come out exact) or `uniform` (all zero).
"""

import numpy as np
import pytest

import mbfps.eval.stages as stages
from mbfps.eval.stages import (
    DECISION_H,
    FAMILY,
    KL_FREE_BITS,
    REPORTED_K,
    SEEDS_REQUIRED,
    STAGES,
    categorical_kl,
    decode_margin,
    entropy_by_group,
    information,
    log_softmax,
    marginal_accuracy,
    marginal_classes,
    mode,
    open_accuracy,
    open_marginal,
    open_persistence,
    persistence_accuracy,
    teacher_accuracy,
    teacher_nll,
)

GROUPS, CLASSES = 2, 3
CONTEXT = 1


def sharp(index) -> np.ndarray:
    """`(..., G)` class indices -> `(..., G, C)` logits with 1000 at the class."""
    index = np.asarray(index)
    return (np.eye(CLASSES)[index] * 1000.0).astype(np.float64)


def uniform(*leading) -> np.ndarray:
    return np.zeros((*leading, GROUPS, CLASSES), dtype=np.float64)


# The posterior over 3 window steps (step 0 is t0, the last context frame):
#   window 0: t0 [0, 1] -> h=1 [1, 1] -> h=2 [2, 0]
#   window 1: t0 [2, 2] -> h=1 [2, 2] -> h=2 [2, 2]
POST = sharp([[[0, 1], [1, 1], [2, 0]], [[2, 2], [2, 2], [2, 2]]])
# Teacher-forced prior over the 2 horizon steps: perfect on window 0, wrong
# everywhere on window 1.
TEACHER = sharp([[[1, 1], [2, 0]], [[0, 0], [0, 0]]])
# Open-loop prior: right at k=1 in both windows, wrong in both groups at k=2.
OPEN = sharp([[[1, 1], [1, 1]], [[2, 2], [1, 1]]])


def test_the_pre_registered_constants_are_the_specs():
    assert DECISION_H == 15 and FAMILY == 5 and SEEDS_REQUIRED == 2
    assert KL_FREE_BITS == 0.20
    assert REPORTED_K == (1, 2, 3, 5, 10, 15, 30, 45)
    assert STAGES == ("encode", "predict", "carry", "decode")


def test_log_softmax_normalises_and_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0]])
    log_p = log_softmax(logits)
    assert np.exp(log_p).sum() == pytest.approx(1.0)
    np.testing.assert_allclose(log_softmax(logits + 100.0), log_p)


def test_categorical_kl_is_zero_between_identical_distributions_and_sums_over_groups():
    """Identical logits give exactly 0.0 (the difference of two identical
    log-softmaxes is exactly zero), whatever the sharpness. Against a uniform
    prior the KL is `log C - H(q)` per group, summed over groups: a q of
    (1/2, 1/4, 1/4) in both groups gives 2 * (log 3 - 1.5 log 2)."""
    assert categorical_kl(POST, POST).shape == (2, 3)
    assert (categorical_kl(POST, POST) == 0.0).all()
    assert (categorical_kl(uniform(4), uniform(4)) == 0.0).all()
    q = np.log(np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]))[None]
    expected = 2 * (np.log(3) - 1.5 * np.log(2))
    assert categorical_kl(q, uniform(1)) == pytest.approx(np.array([expected]))
    with pytest.raises(ValueError, match="shape"):
        categorical_kl(POST, TEACHER)


def test_mode_is_the_argmax_over_classes_with_ties_to_the_smallest():
    np.testing.assert_array_equal(mode(POST), [[[0, 1], [1, 1], [2, 0]], [[2, 2], [2, 2], [2, 2]]])
    np.testing.assert_array_equal(mode(uniform(1)), [[0, 0]])
    assert mode(POST).dtype.kind == "i"


def test_entropy_by_group_is_zero_when_sharp_and_log_c_when_uniform():
    np.testing.assert_array_equal(entropy_by_group(POST), [0.0, 0.0])
    np.testing.assert_allclose(entropy_by_group(uniform(2, 3)), [np.log(3), np.log(3)])
    mixed = np.concatenate([sharp([[[0, 0]]]), uniform(1, 1)], axis=1)  # one sharp, one uniform step
    np.testing.assert_allclose(entropy_by_group(mixed), [np.log(3) / 2, np.log(3) / 2])


def test_information_is_the_horizon_mean_kl_of_posterior_against_the_teacher():
    """Window 0: the teacher equals the posterior at both horizon steps
    (KL 0). Window 1: the posterior is sharp at class 2, the teacher sharp
    at class 0, in both groups at both steps: KL per group is
    `1 * (0 - (-1000))` = 1000, two groups, so 2000 at each step."""
    np.testing.assert_allclose(information(POST, TEACHER, CONTEXT), [0.0, 2000.0])
    assert (information(POST, POST[:, CONTEXT:], CONTEXT) == 0.0).all()


def test_teacher_accuracy_persistence_and_marginal_are_the_hand_counts():
    """Window 0 teacher: both steps, both groups right -> 1.0; window 1 -> 0.0.
    Persistence (the class at h-1 predicts h): window 0 step 0->1 keeps g1
    only (1/2), step 1->2 keeps neither (0) -> 0.25; window 1 keeps all
    -> 1.0. Marginal classes over the four horizon (window, step) pairs:
    g0 sees [1, 2, 2, 2] -> 2, g1 sees [1, 0, 2, 2] -> 2; window 0 matches
    (0 + 1)/4 = 0.25, window 1 matches everything."""
    np.testing.assert_array_equal(teacher_accuracy(POST, TEACHER, CONTEXT), [1.0, 0.0])
    np.testing.assert_array_equal(persistence_accuracy(POST, CONTEXT), [0.25, 1.0])
    classes = marginal_classes(POST, CONTEXT)
    np.testing.assert_array_equal(classes, [2, 2])
    np.testing.assert_array_equal(marginal_accuracy(POST, classes, CONTEXT), [0.25, 1.0])


def test_marginal_classes_break_ties_toward_the_smallest_class():
    tied = sharp([[[0, 2], [1, 1]]])  # context 1: one horizon step, g0 sees [1], g1 sees [1]
    np.testing.assert_array_equal(marginal_classes(tied, CONTEXT), [1, 1])
    two_steps = sharp([[[0, 0], [2, 1], [1, 2]]])  # g0 sees [2, 1], g1 sees [1, 2]: ties -> 1
    np.testing.assert_array_equal(marginal_classes(two_steps, CONTEXT), [1, 1])


def test_open_loop_accuracy_and_its_two_baselines_per_step():
    """Open: right at k=1, wrong at k=2 in both windows. Persistence from t0:
    window 0 anchor [0, 1] against [1, 1] then [2, 0] -> 0.5, 0; window 1
    anchor [2, 2] against [2, 2] twice -> 1, 1. Marginal [2, 2]: window 0
    [1, 1] -> 0, [2, 0] -> 0.5; window 1 -> 1, 1."""
    np.testing.assert_array_equal(open_accuracy(POST, OPEN, CONTEXT), [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(open_persistence(POST, CONTEXT), [[0.5, 0.0], [1.0, 1.0]])
    classes = marginal_classes(POST, CONTEXT)
    np.testing.assert_array_equal(open_marginal(POST, classes, CONTEXT), [[0.0, 0.5], [1.0, 1.0]])


def test_open_accuracy_at_step_one_equals_the_teacher_accuracy_at_step_one_on_equal_priors():
    """When the open-loop prior at step 1 is the teacher-forced prior at step
    1 (the pass's identity), the two accuracies agree at h = 1 exactly."""
    opened = OPEN.copy()
    opened[:, 0] = TEACHER[:, 0]
    per_step_teacher = (mode(TEACHER) == mode(POST[:, CONTEXT:])).mean(axis=2)
    np.testing.assert_array_equal(open_accuracy(POST, opened, CONTEXT)[:, 0], per_step_teacher[:, 0])


def test_teacher_nll_is_zero_when_right_and_the_gap_in_logits_when_wrong():
    """A sharp prior right about the posterior's mode assigns it log p = 0;
    wrong by a 1000-logit margin it assigns -1000 -- window 0 averages 0,
    window 1 averages 1000."""
    np.testing.assert_allclose(teacher_nll(POST, TEACHER, CONTEXT), [0.0, 1000.0])
    assert teacher_nll(POST, uniform(2, 2), CONTEXT) == pytest.approx([np.log(3), np.log(3)])


def test_decode_margin_is_persistence_minus_model_at_h():
    persistence = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model = np.array([[0.5, 2.5, 1.0], [4.0, 1.0, 9.0]])
    np.testing.assert_array_equal(decode_margin(persistence, model, 1), [0.5, 0.0])
    np.testing.assert_array_equal(decode_margin(persistence, model, 3), [2.0, -3.0])
    with pytest.raises(ValueError, match="h"):
        decode_margin(persistence, model, 4)
    with pytest.raises(ValueError, match="shape"):
        decode_margin(persistence, model[:, :2], 1)


def test_the_shapes_and_the_context_are_checked_before_anything_is_reduced():
    with pytest.raises(ValueError, match="context"):
        information(POST, TEACHER, 0)
    with pytest.raises(ValueError, match="context"):
        persistence_accuracy(POST, 3)
    with pytest.raises(ValueError, match="horizon posterior"):
        teacher_accuracy(POST, TEACHER[:, :1], CONTEXT)
    with pytest.raises(ValueError, match="horizon posterior"):
        open_accuracy(POST, OPEN[:, :, :1], CONTEXT)
    with pytest.raises(ValueError, match="classes"):
        marginal_accuracy(POST, np.array([2]), CONTEXT)
    with pytest.raises(ValueError, match="groups, classes"):
        mode(np.zeros(3))
