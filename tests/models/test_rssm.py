import pytest
import torch

from mbfps.models.rssm import LATENT_DIM, RSSM, RSSMConfig

B, T = 3, 7


@pytest.fixture
def rssm():
    return RSSM(RSSMConfig())


def _batch():
    return (
        torch.randn(B, T, 2048),
        torch.randint(0, 6, (B, T)),
    )


def test_observe_shapes(rssm):
    embeddings, actions = _batch()
    out = rssm.observe(embeddings, actions)
    assert out["h"].shape == (B, T, 512)
    assert out["z"].shape == (B, T, 1024)
    assert out["prior_logits"].shape == (B, T, 32, 32)
    assert out["post_logits"].shape == (B, T, 32, 32)
    assert out["latent"].shape == (B, T, LATENT_DIM)


def test_latent_is_h_concatenated_with_z(rssm):
    embeddings, actions = _batch()
    out = rssm.observe(embeddings, actions)
    torch.testing.assert_close(out["latent"], torch.cat([out["h"], out["z"]], dim=-1))


def test_z_is_a_valid_categorical_sample(rssm):
    """32 one-hot vectors of 32 classes: each group sums to 1 with a single 1."""
    embeddings, actions = _batch()
    z = rssm.observe(embeddings, actions)["z"].view(B, T, 32, 32)
    torch.testing.assert_close(z.sum(-1), torch.ones(B, T, 32))
    assert torch.equal(z.max(-1).values, torch.ones(B, T, 32))


def test_straight_through_gradient_reaches_the_posterior(rssm):
    """A hard one-hot sample has zero gradient; the straight-through estimator
    is what lets the encoder train at all. Without it this grad is None."""
    embeddings, actions = _batch()
    embeddings.requires_grad_(True)
    rssm.observe(embeddings, actions)["z"].sum().backward()
    assert embeddings.grad is not None
    assert embeddings.grad.abs().sum() > 0


def test_observe_posterior_depends_on_the_embedding(rssm):
    """The posterior sample must be informed by the CURRENT embedding, not just
    recurrent history -- i.e. observe() must not be secretly sampling the prior.

    This targets the same bug as the gradient test above
    (`test_straight_through_gradient_reaches_the_posterior`) but does not rely
    on that particular network's gradient topology to catch it: `prior_net`
    happens to never consume embeddings, so severing the posterior also
    happens to sever the gradient to `embeddings`. This test instead checks the
    forward VALUE directly -- with the RNG pinned to the same seed and the same
    actions, only two different embedding tensors can make the sampled `z`
    differ. A prior-only rollout would replay byte-identically regardless of
    the embedding.
    """
    actions = torch.randint(0, 6, (B, T))
    e1 = torch.randn(B, T, 2048)
    e2 = torch.randn(B, T, 2048)
    torch.manual_seed(0)
    z1 = rssm.observe(e1, actions)["z"]
    torch.manual_seed(0)
    z2 = rssm.observe(e2, actions)["z"]
    assert not torch.equal(z1, z2), "z is unaffected by the embedding -- observe() looks like it is sampling the prior"


def test_imagine_consumes_no_embeddings(rssm):
    """Imagination must run from actions alone -- this is the project's claim."""
    embeddings, actions = _batch()
    state = rssm.initial_state(B, embeddings.device)
    out = rssm.imagine(actions, state)
    assert out["latent"].shape == (B, T, LATENT_DIM)
    assert "post_logits" not in out


def _seeded_imagine(rssm, actions, state, seed=0):
    """Reproducible rollout WITHOUT taking the mode.

    Sampling is always stochastic -- taking the mode collapses the trajectory to
    3 distinct latents out of 45 and roughly quadruples error. Reproducibility
    therefore comes from fixing the generator, not from removing the sampling.
    """
    torch.manual_seed(seed)
    return rssm.imagine(actions, state)


def test_imagined_trajectory_depends_on_actions(rssm):
    """A model that ignores actions cannot be used for planning.

    The shared seed is essential. Two unseeded calls differ by RNG alone, so the
    assertion would hold even with the action input zeroed out, and the test
    could not fail.
    """
    state = rssm.initial_state(B, torch.device("cpu"))
    a = torch.zeros(B, T, dtype=torch.long)
    b = torch.full((B, T), 3, dtype=torch.long)
    ha = _seeded_imagine(rssm, a, state)["h"]
    hb = _seeded_imagine(rssm, b, state)["h"]
    assert not torch.allclose(ha, hb)


def test_same_seed_reproduces_an_imagined_rollout(rssm):
    """Guards the guard: if this fails, the test above passes on RNG noise.

    Also the property `evaluate_rollout` relies on -- an unseeded rollout made
    gap_closed@45 swing -0.121 / +0.319 / -0.428 across three runs of the same
    checkpoint, and the M3 gate is a test on that sign.
    """
    state = rssm.initial_state(B, torch.device("cpu"))
    a = torch.zeros(B, T, dtype=torch.long)
    torch.testing.assert_close(
        _seeded_imagine(rssm, a, state)["h"], _seeded_imagine(rssm, a, state)["h"]
    )


def test_rollouts_are_stochastic_not_collapsed(rssm):
    """Different seeds must give different trajectories.

    Pins that sampling was not quietly replaced by the mode. Taking the mode
    makes every rollout identical regardless of seed, and it is a tempting
    "fix" for reproducibility -- measured, it quadruples position error and
    gets worse as training sharpens the logits.
    """
    state = rssm.initial_state(B, torch.device("cpu"))
    a = torch.zeros(B, T, dtype=torch.long)
    first = _seeded_imagine(rssm, a, state, seed=0)["z"]
    second = _seeded_imagine(rssm, a, state, seed=1)["z"]
    assert not torch.equal(first, second), "sampling appears to be deterministic"
    distinct = len({tuple(r.argmax(-1).tolist()) for r in first[0].view(T, 32, 32)})
    assert distinct > T // 2, (
        f"only {distinct}/{T} distinct latents -- the trajectory has collapsed"
    )


def test_deterministic_state_carries_history(rssm):
    """h at step t must differ when the actions BEFORE t differed."""
    state = rssm.initial_state(B, torch.device("cpu"))
    a = torch.zeros(B, T, dtype=torch.long)
    b = a.clone()
    b[:, 0] = 5  # differ only at the first step
    ha = _seeded_imagine(rssm, a, state)["h"][:, -1]
    hb = _seeded_imagine(rssm, b, state)["h"][:, -1]
    assert not torch.allclose(ha, hb), "h does not propagate history"


def test_initial_state_shapes(rssm):
    h, z = rssm.initial_state(B, torch.device("cpu"))
    assert h.shape == (B, 512) and z.shape == (B, 1024)
    assert h.abs().sum() == 0 and z.abs().sum() == 0


def test_post_logits_depend_on_this_steps_action(rssm):
    """`post_logits[:, i]` must depend on `actions[:, i]` -- i.e. `h` has
    already been advanced with action `i` before the posterior at `i` is
    formed. Moving `h = self._step(...)` to AFTER the posterior is formed
    passes all the original 12 tests but shifts this dependency by one: the
    posterior would then depend on `actions[0..i-1]` instead of `actions[0..i]`.

    Two action sequences differing ONLY at step `i`, with the RNG pinned
    identically, must produce identical `post_logits` for every step BEFORE
    `i` (causality) and DIFFERENT `post_logits` at step `i` itself.
    """
    embeddings, actions = _batch()
    i = 3
    actions_a = actions.clone()
    actions_b = actions.clone()
    actions_b[:, i] = (actions_b[:, i] + 1) % 6  # differ ONLY at step i

    torch.manual_seed(0)
    post_a = rssm.observe(embeddings, actions_a)["post_logits"]
    torch.manual_seed(0)
    post_b = rssm.observe(embeddings, actions_b)["post_logits"]

    torch.testing.assert_close(post_a[:, :i], post_b[:, :i])
    assert not torch.allclose(post_a[:, i], post_b[:, i]), (
        "post_logits[:, i] is unaffected by actions[:, i] -- "
        "the posterior appears to see the PRE-action h"
    )


def test_post_logits_layout_matches_the_sampled_z(rssm):
    """`post_logits`' (cats, classes) group layout must match the groups `z`
    was actually sampled from. Replacing `_pack`'s
    `unflatten(-1, (cats, classes))` with
    `unflatten(-1, (classes, cats)).transpose(-1, -2)` keeps the same
    (B, T, 32, 32) shape and passes all 12 original tests, because
    cats == classes == 32 leaves no shape assertion able to catch it -- but
    the next task's KL loss consumes `post_logits` assuming it lines up with
    `z`, and a permuted layout would train silently on the wrong pairing.

    `post_net`'s last layer is scaled way up so its softmax is extremely
    peaked: sampling then almost always takes the mode, so a correctly
    laid-out `z`'s argmax must equal `post_logits`' argmax.
    """
    with torch.no_grad():
        last_layer = rssm.post_net[-1]
        last_layer.weight.mul_(1e5)
        last_layer.bias.mul_(1e5)

    embeddings, actions = _batch()
    out = rssm.observe(embeddings, actions)
    z_modes = out["z"].view(B, T, 32, 32).argmax(-1)
    logit_modes = out["post_logits"].argmax(-1)
    assert torch.equal(z_modes, logit_modes)


def test_observe_state_parameter_is_honoured(rssm):
    """Passing an explicit `state` must actually be used. Deleting the
    `state if state is not None else` fallback (always starting from zeros)
    passes all the original 12 tests, since none of them pass a non-None
    `state` into `observe`.
    """
    embeddings, actions = _batch()
    zero_state = rssm.initial_state(B, embeddings.device)
    nonzero_state = (torch.randn(B, 512), rssm._sample(torch.randn(B, 1024)))

    torch.manual_seed(0)
    out_none = rssm.observe(embeddings, actions, state=None)
    torch.manual_seed(0)
    out_zero = rssm.observe(embeddings, actions, state=zero_state)
    torch.manual_seed(0)
    out_nonzero = rssm.observe(embeddings, actions, state=nonzero_state)

    torch.testing.assert_close(out_none["h"], out_zero["h"])
    assert not torch.allclose(out_none["h"], out_nonzero["h"]), (
        "observe() output is unaffected by a non-None state -- "
        "the state= parameter appears to be ignored"
    )


def test_latent_dim_mismatch_is_rejected():
    """`LATENT_DIM` is hardcoded as `512 + 32*32`. A non-default `RSSMConfig`
    whose `h_dim + z_cats*z_classes` does not match it would silently diverge
    from what the heads and probes assume the latent width is, unless
    `RSSM.__init__` guards it."""
    with pytest.raises(ValueError, match="1535.*1536|1536.*1535"):
        RSSM(RSSMConfig(h_dim=511))


def test_init_is_independent_of_prior_rng_consumption():
    """Same guarantee as the shared decoder: construction order must not decide
    weights, or arms with different encoder sizes get different RSSMs."""
    from mbfps.utils.seeding import seed_everything

    def build(waste: int) -> torch.Tensor:
        seed_everything(0)
        torch.randn(waste)
        return next(RSSM(RSSMConfig(), seed=0).parameters()).detach().clone()

    assert torch.equal(build(0), build(500_000))
