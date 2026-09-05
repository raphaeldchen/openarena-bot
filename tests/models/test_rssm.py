import pytest
import torch

from mbfps.models.rssm import KL_FREE_BITS, LATENT_DIM, RSSM, RSSMConfig, _categorical_kl, kl_loss

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


def _logits(b=2, t=3, peak=0.0):
    x = torch.zeros(b, t, 32, 32)
    x[..., 0] = peak
    return x


def test_kl_is_zero_when_distributions_match_and_free_bits_off():
    same = _logits(peak=3.0)
    loss, parts = kl_loss(same, same.clone(), free_bits=0.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert parts["dyn"] == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_when_distributions_differ():
    loss, _ = kl_loss(_logits(peak=5.0), _logits(peak=0.0), free_bits=0.0)
    assert loss.item() > 0


def test_free_bits_clamps_small_divergences():
    """Below the floor the KL must not be optimised away -- that is the point:
    it stops posterior collapse to the prior."""
    near = _logits(peak=0.01)
    clamped, _ = kl_loss(near, _logits(peak=0.0), free_bits=1.0)
    unclamped, _ = kl_loss(near, _logits(peak=0.0), free_bits=0.0)
    assert unclamped.item() < clamped.item()


def test_dyn_and_rep_are_weighted_differently():
    """0.5 / 0.1 per the spec: the prior is pulled toward the posterior five
    times as hard as the reverse."""
    post, prior = _logits(peak=4.0), _logits(peak=0.0)
    only_dyn, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=1.0, rep_scale=0.0)
    only_rep, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=0.0, rep_scale=1.0)
    both, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=0.5, rep_scale=0.1)
    assert both.item() == pytest.approx(0.5 * only_dyn.item() + 0.1 * only_rep.item(), rel=1e-5)


def test_kl_balancing_stops_gradients_on_the_right_side():
    """dyn trains the PRIOR only; rep trains the POSTERIOR only. Getting this
    backwards trains the posterior to be uninformative."""
    post = _logits(peak=4.0).requires_grad_(True)
    prior = _logits(peak=0.0).requires_grad_(True)
    loss, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=1.0, rep_scale=0.0)
    loss.backward()
    assert post.grad.abs().sum() == 0, "dyn term must not train the posterior"
    assert prior.grad.abs().sum() > 0, "dyn term must train the prior"


def test_dyn_scale_and_rep_scale_control_independent_gradients():
    """`test_dyn_and_rep_are_weighted_differently` checks the LOSS VALUE, but
    `dyn` and `rep` are numerically identical for any `post`/`prior` pair --
    `.detach()` only cuts gradient flow, it never changes a forward value, so
    `KL(post.detach(), prior)` and `KL(post, prior.detach())` compute the same
    number. That makes the weighted-sum assertion true for ANY (dyn_scale,
    rep_scale) pair used consistently, including one where the two scales are
    swapped in the implementation -- it cannot tell `dyn_scale` and
    `rep_scale` apart.

    Gradients can, because `dyn_scale` only ever reaches `prior` (through the
    `dyn` term) and `rep_scale` only ever reaches `post` (through the `rep`
    term). Raising `dyn_scale` 5x must scale `prior`'s gradient by exactly 5x
    while leaving `post`'s gradient untouched, and vice versa; swapping the
    two scales in the formula would swap which parameter's gradient tracks
    which coefficient, which this test catches and the value-based test does
    not.
    """
    post = _logits(peak=4.0).requires_grad_(True)
    prior = _logits(peak=0.0).requires_grad_(True)

    loss1, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=1.0, rep_scale=1.0)
    grad_prior_1 = torch.autograd.grad(loss1, prior, retain_graph=True)[0].abs().sum()
    grad_post_1 = torch.autograd.grad(loss1, post, retain_graph=True)[0].abs().sum()

    loss2, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=5.0, rep_scale=1.0)
    grad_prior_2 = torch.autograd.grad(loss2, prior, retain_graph=True)[0].abs().sum()
    grad_post_2 = torch.autograd.grad(loss2, post, retain_graph=True)[0].abs().sum()

    assert grad_prior_2.item() == pytest.approx(5 * grad_prior_1.item(), rel=1e-4), (
        "raising dyn_scale 5x must scale prior's gradient by 5x"
    )
    assert grad_post_2.item() == pytest.approx(grad_post_1.item(), rel=1e-4), (
        "raising dyn_scale must not change post's gradient"
    )


def test_categorical_kl_matches_a_hand_computed_value():
    """Pins `_categorical_kl` against an arithmetic-by-hand answer for a small,
    deliberately ASYMMETRIC case: 1 batch, 1 time step, 2 categorical groups of
    3 classes. `q` = [1,2,3]/6 and [4,1,1]/6 against a uniform `p`.

    Every existing test in this file only checks relative properties (zero
    when equal, positive when different, ordered by free bits, weighted
    linearly, gradients on the right side) -- none of them pin the KL to an
    actual number. That leaves two defects invisible to the whole suite:

    1. SUM vs MEAN over the 32 (here 2) categorical groups.
       `_categorical_kl` sums `per_group` over groups before averaging over
       batch/time; replacing that `.sum(-1)` with `.mean(-1)` divides every
       KL by the group count. For this case that is 0.3182570841 / 2 =
       0.1591285, over 1e-5 away from the pinned value below -- caught.
       In the real model (32 groups) the same mutation divides the measured
       ~0.33 KL by 32 down to ~0.010, which never clears the 0.20 free-bits
       floor and silently zeroes `prior_net`'s gradient for the entire run.

    2. DIRECTION. `_categorical_kl(logits_q, logits_p)` must compute
       KL(q || p), not KL(p || q). Because `q` and `p` are NOT symmetric
       here, the two directions give different numbers -- KL(q||p) =
       0.3182570841 (0.3182570934 in fp32) vs the reversed KL(p||q) =
       0.3269430 -- a ~0.0087 gap, so a tolerance of 1e-6 tells them apart.
       A mutation that swaps which argument plays `q` and which plays `p`
       inside `_categorical_kl` produces the reversed number and fails the
       first assertion below; the second assertion checks the reversed call
       explicitly so the direction is pinned both ways, not just implied.

    Hand computation for the forward direction (KL(q||p), p uniform so
    KL = sum_i q_i * ln(3 * q_i)):
        group 1: 1/6*ln(1/2) + 2/6*ln(1) + 3/6*ln(3/2) = 0.0872083
        group 2: 4/6*ln(2)   + 1/6*ln(1/2)*2           = 0.2310491
        total  = 0.3182574  (0.3182570841 to higher precision)
    """
    q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 1.0]]) / 6.0
    logits_q = torch.log(q).unsqueeze(0).unsqueeze(0)  # (B=1, T=1, groups=2, classes=3)
    logits_p = torch.zeros_like(logits_q)  # uniform p

    forward = _categorical_kl(logits_q, logits_p)
    reverse = _categorical_kl(logits_p, logits_q)

    assert forward.item() == pytest.approx(0.3182570934, abs=1e-6), (
        "KL(q||p) does not match the hand-computed value -- check for a "
        "sum-vs-mean change over the categorical groups"
    )
    assert reverse.item() == pytest.approx(0.3269430, abs=1e-6), (
        "the reversed KL(p||q) does not match its own hand-computed value"
    )
    assert forward.item() != pytest.approx(reverse.item(), abs=1e-6), (
        "forward and reverse KL must differ for this asymmetric q, p -- "
        "if they match, _categorical_kl is not sensitive to argument order"
    )


def test_kl_free_bits_default_is_the_calibrated_0_20_not_the_rejected_1_0():
    """`KL_FREE_BITS` is the module's default `free_bits`, and the training
    loop calls `kl_loss(post, prior)` with NO override -- so this constant,
    not any value used in the tests above (which all pass `free_bits`
    explicitly), is what actually gates gradient to `prior_net` in training.
    1.0 is the DreamerV3 value this project measured and explicitly
    rejected (see `KL_FREE_BITS`'s docstring: at 1.0, `prior_net` gets
    gradient on only 1 of 9 sampled steps).

    Pins the constant directly, then exercises its live effect on the bare
    call path (only the two logit arguments, exactly as production calls
    it) against two literal, non-`KL_FREE_BITS`-derived comparisons -- so a
    mutation to the constant is caught even if it were left wired correctly
    into the `kl_loss` signature.
    """
    assert KL_FREE_BITS == pytest.approx(0.20, abs=1e-9)

    near = _logits(peak=0.01)
    other = _logits(peak=0.0)
    bare, _ = kl_loss(near, other)  # only the two logit arguments
    at_020, _ = kl_loss(near, other, free_bits=0.20)
    at_100, _ = kl_loss(near, other, free_bits=1.00)

    assert bare.item() == pytest.approx(at_020.item(), rel=1e-6), (
        "the bare call does not match an explicit free_bits=0.20 -- "
        "KL_FREE_BITS may not be 0.20"
    )
    assert bare.item() != pytest.approx(at_100.item(), rel=1e-6), (
        "the bare call matches free_bits=1.00 -- KL_FREE_BITS looks like "
        "the rejected DreamerV3 value"
    )


def test_kl_loss_default_scales_are_0_5_dyn_0_1_rep_not_swapped():
    """The training loop calls `kl_loss(post, prior)` with no `dyn_scale` or
    `rep_scale` override, so the two defaults (0.5, 0.1) are what actually
    weights the loss. `dyn` and `rep` are numerically identical for any
    fixed `post`/`prior` pair (`.detach()` only cuts gradient, never changes
    a forward value -- see `test_dyn_scale_and_rep_scale_control_independent_
    gradients`), so a loss-value comparison cannot tell 0.5/0.1 apart from a
    swapped 0.1/0.5 default. Gradients can: `dyn_scale` only ever reaches
    `prior` (through the `dyn` term) and `rep_scale` only ever reaches
    `post` (through the `rep` term).

    The bare call's gradients must match an explicit `dyn_scale=0.5,
    rep_scale=0.1` call and must NOT match an explicit swapped
    `dyn_scale=0.1, rep_scale=0.5` call.
    """
    post = _logits(peak=4.0).requires_grad_(True)
    prior = _logits(peak=0.0).requires_grad_(True)

    bare_loss, _ = kl_loss(post, prior, free_bits=0.0)  # only the two logit arguments (+ free_bits=0 to isolate scales)
    grad_prior_bare = torch.autograd.grad(bare_loss, prior, retain_graph=True)[0].abs().sum()
    grad_post_bare = torch.autograd.grad(bare_loss, post, retain_graph=True)[0].abs().sum()

    matched_loss, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=0.5, rep_scale=0.1)
    grad_prior_matched = torch.autograd.grad(matched_loss, prior, retain_graph=True)[0].abs().sum()
    grad_post_matched = torch.autograd.grad(matched_loss, post, retain_graph=True)[0].abs().sum()

    swapped_loss, _ = kl_loss(post, prior, free_bits=0.0, dyn_scale=0.1, rep_scale=0.5)
    grad_prior_swapped = torch.autograd.grad(swapped_loss, prior, retain_graph=True)[0].abs().sum()
    grad_post_swapped = torch.autograd.grad(swapped_loss, post, retain_graph=True)[0].abs().sum()

    assert grad_prior_bare.item() == pytest.approx(grad_prior_matched.item(), rel=1e-5), (
        "prior's gradient under the bare call does not match dyn_scale=0.5 -- "
        "the default dyn_scale may not be 0.5"
    )
    assert grad_post_bare.item() == pytest.approx(grad_post_matched.item(), rel=1e-5), (
        "post's gradient under the bare call does not match rep_scale=0.1 -- "
        "the default rep_scale may not be 0.1"
    )
    assert grad_prior_bare.item() != pytest.approx(grad_prior_swapped.item(), rel=1e-3), (
        "prior's gradient under the bare call matches the SWAPPED "
        "dyn_scale=0.1 -- the defaults look swapped"
    )
    assert grad_post_bare.item() != pytest.approx(grad_post_swapped.item(), rel=1e-3), (
        "post's gradient under the bare call matches the SWAPPED "
        "rep_scale=0.5 -- the defaults look swapped"
    )


def test_free_bits_floor_applies_to_rep_independently_of_dyn():
    """The floor must clamp `dyn` and `rep` INDEPENDENTLY. Replacing
    `rep_scale * torch.maximum(rep, floor)` with plain `rep_scale * rep`
    would let `rep` be driven all the way to zero -- collapsing the
    posterior into the prior on the representation side -- while `dyn`'s own
    floor still holds, so it is invisible to any test that leaves `dyn_scale`
    at its default (0.5 dominates the sum and is still correctly floored).

    Isolates `rep` with `dyn_scale=0.0, rep_scale=1.0`, mirroring how
    `test_kl_balancing_stops_gradients_on_the_right_side` isolates `dyn`
    with `dyn_scale=1.0, rep_scale=0.0`.
    """
    near = _logits(peak=0.01)
    other = _logits(peak=0.0)
    clamped, _ = kl_loss(near, other, free_bits=1.0, dyn_scale=0.0, rep_scale=1.0)
    unclamped, _ = kl_loss(near, other, free_bits=0.0, dyn_scale=0.0, rep_scale=1.0)
    assert unclamped.item() < clamped.item(), (
        "rep does not appear to be floored independently of dyn"
    )
