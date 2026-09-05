import pytest
import torch

from mbfps.models.heads import WorldModelHeads, continue_target
from mbfps.models.rssm import LATENT_DIM

B, T = 3, 5


def test_head_shapes():
    out = WorldModelHeads()(torch.randn(B, T, LATENT_DIM))
    assert out["embedding"].shape == (B, T, 2048)
    assert out["reward"].shape == (B, T)
    assert out["continue_logit"].shape == (B, T)


def test_continue_target_is_one_when_nothing_ended():
    t = torch.zeros(B, T, dtype=torch.bool)
    torch.testing.assert_close(continue_target(t, t), torch.ones(B, T))


def test_continue_target_is_zero_only_on_termination():
    terminated = torch.zeros(B, T, dtype=torch.bool)
    terminated[0, 2] = True
    expected = torch.ones(B, T)
    expected[0, 2] = 0.0
    torch.testing.assert_close(
        continue_target(terminated, torch.zeros_like(terminated)), expected
    )


def test_truncation_does_not_end_the_episode():
    """THE invariant. A time-limit cutoff is not a terminal state: the episode
    did not end, the clock ran out. Training the continue head to say otherwise
    corrupts bootstrapping, which M4 depends on directly."""
    truncated = torch.zeros(B, T, dtype=torch.bool)
    truncated[1, 3] = True
    torch.testing.assert_close(
        continue_target(torch.zeros_like(truncated), truncated), torch.ones(B, T)
    )


def test_terminated_and_truncated_together_still_terminate():
    flag = torch.zeros(B, T, dtype=torch.bool)
    flag[0, 0] = True
    expected = torch.ones(B, T)
    expected[0, 0] = 0.0
    torch.testing.assert_close(continue_target(flag, flag), expected)


def test_heads_init_is_independent_of_prior_rng_consumption():
    from mbfps.utils.seeding import seed_everything

    def build(waste: int) -> torch.Tensor:
        seed_everything(0)
        torch.randn(waste)
        return next(WorldModelHeads(seed=0).parameters()).detach().clone()

    assert torch.equal(build(0), build(250_000))


def test_forward_wires_each_head_to_its_own_named_output():
    """Guards against the dict keys being wired to the wrong submodule.

    `reward` and `continue_` are architecturally identical (same LATENT_DIM ->
    512 -> 512 -> 1 MLP), so a swap between them changes no shape and no test
    that only checks `.shape` can ever notice. This compares each dict entry
    against a direct call into the submodule it is supposed to come from.
    """
    heads = WorldModelHeads()
    latent = torch.randn(B, T, LATENT_DIM)
    out = heads(latent)
    torch.testing.assert_close(out["embedding"], heads.embedding(latent))
    torch.testing.assert_close(out["reward"], heads.reward(latent).squeeze(-1))
    torch.testing.assert_close(
        out["continue_logit"], heads.continue_(latent).squeeze(-1)
    )


@pytest.mark.parametrize("head_name", ["embedding", "reward", "continue_"])
def test_each_head_mlp_has_two_hidden_layers_of_512(head_name):
    """Pins '2 x 512' as an actual architecture check, not just an output shape.

    A head MLP collapsed to one hidden layer (or resized) still produces the
    right output shape, so only counting `nn.Linear` layers and reading their
    widths can catch it.
    """
    heads = WorldModelHeads()
    linears = [m for m in getattr(heads, head_name) if isinstance(m, torch.nn.Linear)]
    # in->hidden, hidden->hidden, hidden->out: exactly two hidden layers.
    assert len(linears) == 3
    assert linears[0].out_features == 512
    assert linears[1].in_features == 512
    assert linears[1].out_features == 512
