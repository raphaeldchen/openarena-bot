import pytest
import torch

from mbfps.models.transforms import REWARD_TRANSFORMS, read_reward, reward_target, symexp, symlog


def test_round_trip_over_the_basic_reward_range():
    x = torch.linspace(-9.0, 101.0, 1101, dtype=torch.float64)
    torch.testing.assert_close(symexp(symlog(x)), x, rtol=0, atol=1e-9)


def test_both_are_odd():
    x = torch.tensor([0.5, 4.0, 97.0], dtype=torch.float64)
    torch.testing.assert_close(symlog(-x), -symlog(x))
    torch.testing.assert_close(symexp(-x), -symexp(x))


def test_known_values():
    """The spec's own figures: -4 -> -1.61, +97 -> +4.58."""
    out = symlog(torch.tensor([-4.0, 97.0]))
    torch.testing.assert_close(out, torch.tensor([-1.6094, 4.5850]), atol=1e-4, rtol=0)


def test_read_reward_identity_is_the_raw_output():
    y = torch.tensor([1.0, -2.0])
    assert torch.equal(read_reward(y, "identity"), y)


def test_read_reward_symlog_inverts():
    y = symlog(torch.tensor([97.0, -4.0]))
    torch.testing.assert_close(read_reward(y, "symlog"), torch.tensor([97.0, -4.0]))


def test_unknown_transform_is_refused():
    assert REWARD_TRANSFORMS == ("identity", "symlog")
    with pytest.raises(ValueError, match="reward_transform"):
        read_reward(torch.zeros(1), "log")


def test_reward_target_matches_read_reward():
    r = torch.tensor([-9.0, -4.0, 97.0])
    for transform in REWARD_TRANSFORMS:
        torch.testing.assert_close(read_reward(reward_target(r, transform), transform), r)


def test_read_reward_bounds_clip_in_target_space():
    """Bounds are reward units; the clip happens on the head's output, in the
    transform's space, so symexp never sees a value beyond symlog(r_max)."""
    head = torch.tensor([symlog(torch.tensor(500.0)).item(), 0.0, -50.0])
    out = read_reward(head, "symlog", bounds=(-9.0, 99.0))
    torch.testing.assert_close(out, torch.tensor([99.0, 0.0, -9.0]), rtol=1e-5, atol=1e-4)


def test_read_reward_bounds_are_identity_inside():
    head = symlog(torch.tensor([-4.0, 50.0]))
    torch.testing.assert_close(read_reward(head, "symlog", bounds=(-9.0, 99.0)),
                               torch.tensor([-4.0, 50.0]), rtol=1e-5, atol=1e-4)


def test_read_reward_without_bounds_is_unchanged():
    head = torch.tensor([7.0])
    assert torch.equal(read_reward(head, "symlog"), symexp(head))
