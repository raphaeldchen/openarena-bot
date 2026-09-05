import pytest

from mbfps.envs.actions import build_action_set


def test_action_set_size_is_buttons_plus_noop():
    assert len(build_action_set(5)) == 6


def test_first_action_is_noop():
    assert build_action_set(3)[0] == [0, 0, 0]


def test_remaining_actions_are_one_hot():
    actions = build_action_set(3)
    assert actions[1] == [1, 0, 0]
    assert actions[2] == [0, 1, 0]
    assert actions[3] == [0, 0, 1]


def test_every_vector_has_length_n_buttons():
    assert all(len(a) == 4 for a in build_action_set(4))


def test_entries_are_plain_ints_for_vizdoom():
    # ViZDoom's make_action requires a list of ints, not numpy scalars.
    assert all(isinstance(v, int) for a in build_action_set(3) for v in a)


def test_zero_buttons_rejected():
    with pytest.raises(ValueError, match="at least one button"):
        build_action_set(0)
