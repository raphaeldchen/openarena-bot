import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.training.world_model import WorldModel, train_world_model
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.seeding import seed_everything

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    steps = np.arange(t)
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(t + 1, dtype=np.uint8)
    obs[:, 0, 0, 1] = fill
    return Episode(
        obs=obs,
        actions=(steps % 6).astype(np.int32),
        rewards=(steps % 3).astype(np.float32),
        terminated=(steps == t - 1),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.stack(
            [
                np.full(t + 1, 100.0),
                np.arange(t + 1, dtype=np.float32) * 3.0,
                np.arange(t + 1, dtype=np.float32) * -2.0,
                np.zeros(t + 1),
                (np.arange(t + 1) * 7.0) % 360.0,
            ],
            axis=1,
        ).astype(np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    """Five episodes, not four.

    `train_world_model` holds out a validation split with
    `episode_split(..., val_fraction=0.2)`, and that function floors:
    `int(4 * 0.2) == 0`, which it rejects with "need at least one episode on
    each side". A four-episode fixture therefore makes every test that calls
    `train_world_model` die in the split rather than in the model. Five is the
    smallest count that yields a non-empty split (4 train / 1 val).
    """
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4, 5):
        buf.add(make_episode(t=40, fill=fill))
    return buf


def tiny(arm: str = "cnn", **kw):
    base = dict(steps=3, batch_size=2, seq_len=6, device="cpu")
    base.update(kw)
    return get_config(arm, **base)


def test_forward_returns_a_scalar_loss_and_named_parts(buffer):
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    cfg = tiny()
    model = WorldModel(cfg)
    batch = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample()
    loss, parts = model(to_device(batch, torch.device("cpu")))
    assert loss.ndim == 0 and torch.isfinite(loss)
    for key in ("embedding", "reward", "continue", "kl_dyn", "kl_rep"):
        assert key in parts


def test_every_trainable_parameter_receives_gradient(buffer):
    """The classic RSSM bug is a detached tensor silently freezing a submodule.

    `prior_net` is excluded and handled by the next two tests. The measured dyn
    KL at initialisation is 0.022-0.038 nat depending on arm, below the 0.20
    floor, so `torch.maximum(dyn, KL_FREE_BITS)` is constant at step 0 and the
    prior legitimately receives no gradient yet. That is the intended warm-up --
    the prior must not collapse the posterior before the posterior carries
    anything -- not a detached tensor. It ends quickly at this floor: the prior
    gets gradient on 8 of 9 sampled steps thereafter.
    """
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    model = WorldModel(tiny())
    batch = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample()
    loss, _ = model(to_device(batch, torch.device("cpu")))
    loss.backward()
    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad
        and not name.startswith("rssm.prior_net")
        and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    assert not dead, f"parameters receiving no gradient: {dead}"


def test_prior_net_trains_once_the_kl_exceeds_free_bits(buffer):
    """The other half: the warm-up must actually end.

    If the prior only ever sees a clamped constant it is never trained, and
    every `imagine()` rollout -- the whole M3 gate and all of M4 -- runs on
    random weights. This constructs the post-warm-up condition directly.
    """
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    model = WorldModel(tiny())
    # Drive the posterior far from the prior so the KL clears the floor.
    with torch.no_grad():
        for p in model.rssm.post_net[-1].parameters():
            p.mul_(50.0)
    batch = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample()
    loss, parts = model(to_device(batch, torch.device("cpu")))
    assert parts["kl_dyn"] > 1.0, (
        f"fixture failed to clear the free-bits floor (kl_dyn={parts['kl_dyn']:.3f}); "
        "this test cannot check what it claims to"
    )
    loss.backward()
    grad = sum(
        p.grad.abs().sum() for p in model.rssm.prior_net.parameters() if p.grad is not None
    )
    assert grad > 0, "prior_net receives no gradient even above the free-bits floor"


def test_history_reports_whether_the_kl_ever_cleared_the_floor(buffer):
    """Makes the warm-up an observable, not an assumption.

    Whether the KL ever clears the `KL_FREE_BITS` floor (0.20 nat) on THIS dataset at
    THIS scale is an empirical question. If it never does, the prior is frozen for the
    whole run and the result is meaningless -- so the history must carry the evidence
    either way.
    """
    history = train_world_model(tiny(), buffer, out_dir=None)
    assert "kl_dyn_max" in history
    assert "kl_rate_above_free_bits" in history
    assert history["kl_dyn_max"] == pytest.approx(
        max(p["kl_dyn"] for p in history["parts"])
    )
    expected = sum(
        1 for p in history["parts"] if p["kl_dyn"] > KL_FREE_BITS
    ) / len(history["parts"])
    assert history["kl_rate_above_free_bits"] == pytest.approx(expected)


def test_kl_rate_distinguishes_a_barely_trained_prior_from_a_trained_one():
    """A max cannot: one transient step above the floor sets it True forever.

    Measured, that hid two arms whose priors trained on 1.1% and 11.2% of steps
    behind an identical `True`.
    """
    from mbfps.training.world_model import _kl_rate

    barely = [0.01] * 99 + [5.0]
    trained = [0.5] * 100
    assert max(barely) > max(trained)          # a max ranks them backwards
    assert _kl_rate(barely) < _kl_rate(trained)


def test_overfits_a_single_fixed_batch(buffer):
    """A failure here is a bug, not a hyperparameter problem."""
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    torch.manual_seed(0)
    model = WorldModel(tiny(lr=3e-4))
    batch = to_device(
        SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample(),
        torch.device("cpu"),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
    first = float(model(batch)[0])
    for _ in range(120):
        loss, _ = model(batch)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    assert float(loss) < 0.5 * first, f"did not overfit: {first:.4f} -> {float(loss):.4f}"


def test_privileged_state_never_enters_the_loss(buffer):
    """privileged_state is evaluation-only. If it reaches a training tensor the
    whole experiment is silently invalid.

    Two independent barriers, both checked. `to_device` is a whitelist that
    omits `privileged`, so it never reaches the compute device at all -- that
    is the structural half. The behavioural half is that `forward` must not
    read it even when it IS present, which is what a future widening of the
    whitelist would produce; the tensor is therefore forced into the batch by
    hand before the backward pass.
    """
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    model = WorldModel(tiny())
    raw = SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample(
        include_privileged=True
    )
    assert "privileged" in raw
    batch = to_device(raw, torch.device("cpu"))
    assert "privileged" not in batch, (
        "to_device moved privileged onto the compute device; the whitelist that "
        "keeps invariant #2 structural has been widened or replaced"
    )

    batch["privileged"] = torch.from_numpy(raw["privileged"])
    batch["privileged"].requires_grad_(True)
    loss, _ = model(batch)
    loss.backward()
    assert batch["privileged"].grad is None or batch["privileged"].grad.abs().sum() == 0


def test_same_seed_reproduces_the_loss_curve(buffer):
    a = train_world_model(tiny(), buffer, out_dir=None)
    b = train_world_model(tiny(), buffer, out_dir=None)
    assert a["loss"] == b["loss"]


def test_different_seeds_diverge(buffer):
    a = train_world_model(tiny(seed=0), buffer, out_dir=None)
    b = train_world_model(tiny(seed=1), buffer, out_dir=None)
    assert a["loss"] != b["loss"]


def test_the_arms_this_file_trains_are_the_studys_three():
    """L7 guard for the parametrisation below: iterating `ARMS` would let a
    tuple that lost a member shrink the test instead of failing it, so the
    three names are written out here, once. `cnn` is not among them -- it is
    `tiny()`'s default above because the RSSM and loss tests are indifferent
    to which encoder feeds them and a pixel encoder needs no feature file,
    but it is not an arm the study trains."""
    assert ARMS == ("pixel_ae", "frozen_ssl", "random_vit")
    assert "cnn" not in ARMS


#: The cache each study arm reads, by suffix and row width. `pixel_ae` rows
#: are 32 wide (the M2 autoencoder's 2048-d projection cut into 64 rows);
#: the ViT arms' are 384 (DINOv2-small's hidden size). Written here rather
#: than derived from `BACKBONE_GEOMETRY`, so a registry that drifted would
#: fail this test rather than be copied into it.
_ARM_CACHE = {
    "pixel_ae": (".features_pixel_ae.npy", 32),
    "frozen_ssl": (".features.npy", 384),
    "random_vit": (".features_random_vit.npy", 384),
}


@pytest.mark.parametrize("arm", ARMS)
def test_all_three_arms_train(buffer, arm, tmp_path):
    """Every study arm trains end to end through `train_world_model`, each on
    a cache of ITS OWN geometry. Guarded against L7 by the test above."""
    assert set(_ARM_CACHE) == set(ARMS)
    suffix, width = _ARM_CACHE[arm]
    for path in buffer.episode_paths():
        np.save(path.with_suffix(suffix), np.zeros((41, 64, width), dtype=np.float16))
    history = train_world_model(tiny(arm), buffer, out_dir=None)
    assert history["steps"] == 3
    assert history["arm"] == arm
    assert all(np.isfinite(history["loss"]))


# --------------------------------------------------------------------------
# Guards added while mutation-testing Step 9. Everything above is the brief's
# own test list; everything below closes a mutation that survived it.
# --------------------------------------------------------------------------


def _cpu_batch(buffer, **kw):
    from mbfps.data.loader import SequenceLoader
    from mbfps.utils.device import to_device

    return to_device(
        SequenceLoader(buffer, batch_size=2, seq_len=6, seed=0).sample(**kw),
        torch.device("cpu"),
    )


def _seeded(model, batch, seed: int = 0):
    """One forward pass with the categorical sampler pinned.

    `RSSM._sample` draws from a `Categorical` on every step -- deliberately, at
    evaluation too -- so two `forward` calls on one batch differ by ~1e-3 in
    every reported part. Any test that compares two passes has to fix the RNG
    first or it is comparing sampling noise.
    """
    torch.manual_seed(seed)
    return model(batch)


def test_the_embedding_loss_target_carries_no_gradient(buffer):
    """`embeddings.detach()` -- the target is data, not a second gradient path.

    Without the detach the encoder can cut the embedding loss by dragging the
    TARGET toward the prediction rather than the prediction toward the target;
    the degenerate optimum of that objective is a collapsed, constant
    embedding. Nothing else in this file notices, because every parameter still
    receives gradient (through the prediction path) and the loss still falls.

    Note the trap this test had to avoid: the encoder receives embedding-loss
    gradient EITHER WAY, via embeddings -> post_net -> latent -> head. So "the
    encoder gets no gradient from this term" is false with the detach in place
    and cannot be the check. What is unambiguous is the target tensor itself,
    inspected at the call.
    """
    import torch.nn.functional as F

    model = WorldModel(tiny())
    batch = _cpu_batch(buffer)
    embed_dim = model.cfg.encoder.embed_dim
    seen: list[tuple[bool, bool]] = []
    real_mse = F.mse_loss

    def spy(prediction, target, *args, **kwargs):
        if target.shape[-1] == embed_dim and target.dim() == 3:
            seen.append((prediction.requires_grad, target.requires_grad))
        return real_mse(prediction, target, *args, **kwargs)

    monkeypatched = pytest.MonkeyPatch()
    try:
        monkeypatched.setattr(F, "mse_loss", spy)
        model(batch)
    finally:
        monkeypatched.undo()

    assert len(seen) == 1, (
        f"expected exactly one (B, T, {embed_dim}) mse_loss call in forward, saw "
        f"{len(seen)}; this test can no longer identify the embedding term"
    )
    prediction_requires_grad, target_requires_grad = seen[0]
    assert prediction_requires_grad, (
        "the embedding PREDICTION carries no gradient, so the head is not "
        "trained at all -- this test's premise is broken"
    )
    assert not target_requires_grad, (
        "the embedding loss target still requires grad: `embeddings` was passed "
        "where `embeddings.detach()` belongs, so the encoder is rewarded for "
        "collapsing the embedding toward the prediction"
    )


def test_encoder_gradient_matches_a_detached_target_and_not_an_attached_one(buffer):
    """The behavioural half, through the public `forward`.

    Reconstructs the objective twice -- once with the target detached, once
    without -- and requires `forward`'s own gradient into the encoder to equal
    the first and differ from the second. The two references are checked
    against each other first, so a run where they happened to coincide fails
    loudly instead of passing vacuously.
    """
    import torch.nn.functional as F

    from mbfps.models.heads import continue_target
    from mbfps.models.rssm import kl_loss

    model = WorldModel(tiny())
    batch = _cpu_batch(buffer)
    encoder_params = list(model.encoder.parameters())

    def encoder_grad(loss):
        grads = torch.autograd.grad(loss, encoder_params, allow_unused=True)
        return torch.cat([g.flatten() for g in grads if g is not None])

    def reference(detach_target: bool):
        torch.manual_seed(0)
        embeddings = model.embed(batch)
        out = model.rssm.observe(embeddings, batch["actions"])
        predictions = model.heads(out["latent"])
        target = embeddings.detach() if detach_target else embeddings
        loss = (
            F.mse_loss(predictions["embedding"], target)
            + F.mse_loss(predictions["reward"], batch["rewards"])
            + F.binary_cross_entropy_with_logits(
                predictions["continue_logit"],
                continue_target(batch["terminated"], batch["truncated"]),
            )
            + kl_loss(out["post_logits"], out["prior_logits"])[0]
        )
        return encoder_grad(loss)

    detached = reference(True)
    attached = reference(False)
    assert not torch.allclose(detached, attached), (
        "detaching the target makes no difference to the encoder gradient on "
        "this batch, so this test cannot detect the mutation"
    )

    actual = encoder_grad(_seeded(model, batch)[0])
    torch.testing.assert_close(actual, detached)
    assert not torch.allclose(actual, attached)


def test_continue_target_is_built_from_terminated_not_truncated(buffer):
    """The call site's argument ORDER, which `tests/models/test_heads.py` cannot see.

    `continue_target` deletes its second argument, so swapping the two at this
    call site -- `continue_target(batch["truncated"], batch["terminated"])` --
    silently trains the continue head on truncation instead of termination.
    Every shape is unchanged and every heads-level test still passes, because
    they exercise `continue_target` directly and never this call.

    Two batches identical except that one's flags are swapped must therefore
    produce a DIFFERENT continue loss.
    """
    model = WorldModel(tiny())
    batch = _cpu_batch(buffer)
    # Make the two flags genuinely different so a swap is observable.
    batch["terminated"] = torch.zeros_like(batch["terminated"])
    batch["terminated"][0, 2] = True
    batch["truncated"] = torch.zeros_like(batch["truncated"])
    batch["truncated"][1, 4] = True

    _, parts = _seeded(model, batch)
    swapped = dict(batch)
    swapped["terminated"], swapped["truncated"] = batch["truncated"], batch["terminated"]
    _, swapped_parts = _seeded(model, swapped)
    assert parts["continue"] != swapped_parts["continue"], (
        "the continue loss is unchanged when terminated and truncated are "
        "exchanged, so it is not reading terminated"
    )

    # And the direction: a batch with only truncation must score as fully
    # continuing, exactly as if nothing had happened.
    only_truncated = dict(batch)
    only_truncated["terminated"] = torch.zeros_like(batch["terminated"])
    nothing = dict(batch)
    nothing["terminated"] = torch.zeros_like(batch["terminated"])
    nothing["truncated"] = torch.zeros_like(batch["truncated"])
    assert _seeded(model, only_truncated)[1]["continue"] == pytest.approx(
        _seeded(model, nothing)[1]["continue"], rel=1e-9
    ), "truncation changes the continue target; it must not"


def test_embed_pairs_each_action_with_the_frame_it_produced(buffer):
    """`embeddings[:, 1:]`, not `[:, :-1]` -- the acausality guard.

    `actions[i]` is taken AT `obs[i]` and leads to `obs[i+1]`, and
    `RSSM.observe` conditions the posterior at step `i` on `embeddings[:, i]`
    AFTER advancing `h` with `actions[:, i]`. So `embed` must return the LAST T
    of the T+1 encoded frames. Returning the first T instead keeps every shape
    and every loss finite while making the model acausal, which no other test
    in this file can see.
    """
    model = WorldModel(tiny())
    batch = _cpu_batch(buffer)
    source = batch["obs"]
    b, t_plus_one = source.shape[0], source.shape[1]

    with torch.no_grad():
        all_frames = model.encoder(
            source.reshape(b * t_plus_one, *source.shape[2:])
        ).view(b, t_plus_one, -1)
        embedded = model.embed(batch)

    assert embedded.shape == (b, t_plus_one - 1, all_frames.shape[-1])
    torch.testing.assert_close(embedded, all_frames[:, 1:])
    assert not torch.allclose(embedded, all_frames[:, :-1]), (
        "the first and last T frames are indistinguishable in this fixture, so "
        "this test cannot detect the acausal pairing"
    )


def test_total_loss_is_exactly_its_four_parts_under_the_default_kl_weights(buffer):
    """Pins the VALUE of the objective, not just that gradient flows.

    Every gradient-routing test in this file passes just as happily when a term
    is scaled, dropped or double-counted. This reconstructs the scalar from the
    reported parts using the module's own free-bits floor and the pinned 0.5 /
    0.1 dyn/rep scales, so any reweighting -- including calling `kl_loss` with
    explicit arguments that bypass its calibrated defaults -- shows up here.
    """
    model = WorldModel(tiny())
    loss, parts = model(_cpu_batch(buffer))
    expected_kl = 0.5 * max(parts["kl_dyn"], KL_FREE_BITS) + 0.1 * max(
        parts["kl_rep"], KL_FREE_BITS
    )
    expected = parts["embedding"] + parts["reward"] + parts["continue"] + expected_kl
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_the_kl_reaches_the_prior_at_dyn_scale_not_rep_scale(buffer):
    """The 0.5 / 0.1 split, which no loss VALUE can see.

    Found by mutation: passing `dyn_scale=0.1, rep_scale=0.5` to `kl_loss` from
    `forward` survived every other test in this file. Two reasons it is
    invisible to values. `dyn` and `rep` differ only in which side is detached,
    so they are numerically EQUAL and `0.5*d + 0.1*d == 0.1*d + 0.5*d`. And at
    initialisation both sit under the 0.20 free-bits floor, so both are clamped
    to the same constant and the sum is identical either way.

    Only gradients separate them. `prior_logits` is detached inside `rep`, so
    everything `prior_net` receives arrives through `dyn` alone and carries
    `dyn_scale` as a factor -- swapping the two scales shrinks it fivefold. The
    posterior has to be driven clear of the floor first, or the clamp zeroes
    the gradient and there is nothing to measure.
    """
    from mbfps.models.rssm import kl_loss

    model = WorldModel(tiny())
    with torch.no_grad():  # clear the free-bits floor, as the warm-up test does
        for p in model.rssm.post_net[-1].parameters():
            p.mul_(50.0)
    batch = _cpu_batch(buffer)
    prior_params = list(model.rssm.prior_net.parameters())

    def prior_grad(loss):
        grads = torch.autograd.grad(loss, prior_params, allow_unused=True)
        return torch.cat([g.flatten() for g in grads if g is not None])

    def reference(dyn_scale, rep_scale):
        torch.manual_seed(0)
        out = model.rssm.observe(model.embed(batch), batch["actions"])
        kl, _ = kl_loss(
            out["post_logits"],
            out["prior_logits"],
            dyn_scale=dyn_scale,
            rep_scale=rep_scale,
        )
        return prior_grad(kl)

    loss, parts = _seeded(model, batch)
    assert parts["kl_dyn"] > KL_FREE_BITS, (
        f"kl_dyn={parts['kl_dyn']:.3f} is still under the free-bits floor, so "
        "prior_net gets no gradient and this test cannot check what it claims"
    )

    as_specified = reference(0.5, 0.1)
    swapped = reference(0.1, 0.5)
    assert not torch.allclose(as_specified, swapped), (
        "0.5/0.1 and 0.1/0.5 give prior_net the same gradient here, so this "
        "test cannot detect a swap"
    )
    torch.testing.assert_close(prior_grad(loss), as_specified)
    assert not torch.allclose(prior_grad(_seeded(model, batch)[0]), swapped)


def test_gradients_are_zeroed_between_optimiser_steps(buffer):
    """`optimiser.zero_grad()` must run once per step.

    Found by mutation: deleting it left all 29 tests green. PyTorch accumulates
    into `.grad`, so without the reset step `k` descends on the SUM of the
    first `k` gradients -- an effective learning rate that grows without bound
    over a 20k-step run, while a three-step history still looks perfectly
    healthy and perfectly reproducible.

    This counts the call rather than inferring accumulation from gradient
    magnitudes: the loop's batches and its categorical samples both differ
    step to step, so a "did the gradient grow?" assertion would be reading
    sampling noise, and a test that cannot fail reliably is worse than none.
    """
    seen = {}
    real_adam = torch.optim.Adam

    def spy(params, **kwargs):
        optimiser = real_adam(params, **kwargs)
        counts = {"zero_grad": 0, "step": 0}
        real_zero, real_step = optimiser.zero_grad, optimiser.step

        def zero_grad(*a, **kw):
            counts["zero_grad"] += 1
            return real_zero(*a, **kw)

        def step(*a, **kw):
            counts["step"] += 1
            return real_step(*a, **kw)

        optimiser.zero_grad = zero_grad
        optimiser.step = step
        seen["counts"] = counts
        return optimiser

    monkeypatched = pytest.MonkeyPatch()
    try:
        monkeypatched.setattr(torch.optim, "Adam", spy)
        train_world_model(tiny(steps=5), buffer, out_dir=None)
    finally:
        monkeypatched.undo()

    counts = seen["counts"]
    assert counts["step"] == 5, "the loop did not take one optimiser step per step"
    assert counts["zero_grad"] >= counts["step"], (
        f"{counts['zero_grad']} zero_grad() call(s) for {counts['step']} step(s): "
        "gradients accumulate across steps"
    )


def test_reward_and_continue_losses_read_their_own_targets(buffer):
    """Guards against the two scalar targets being crossed at the call site.

    `rewards` and the continue target are both `(B, T)` floats, so swapping
    which head is scored against which changes no shape. Perturbing one target
    must move exactly one reported part.
    """
    model = WorldModel(tiny())
    batch = _cpu_batch(buffer)
    batch["terminated"] = torch.zeros_like(batch["terminated"])
    base = _seeded(model, batch)[1]

    bumped_rewards = dict(batch)
    bumped_rewards["rewards"] = batch["rewards"] + 5.0
    moved = _seeded(model, bumped_rewards)[1]
    assert moved["reward"] != base["reward"]
    assert moved["continue"] == pytest.approx(base["continue"], rel=1e-9)

    bumped_terminated = dict(batch)
    flags = torch.zeros_like(batch["terminated"])
    flags[:, 3] = True
    bumped_terminated["terminated"] = flags
    moved = _seeded(model, bumped_terminated)[1]
    assert moved["continue"] != base["continue"]
    assert moved["reward"] == pytest.approx(base["reward"], rel=1e-9)


def test_training_holds_out_the_validation_split(buffer, monkeypatch):
    """The loop must train on `episode_split(...)[0]`, not the whole buffer.

    Nothing else here notices: training on all five episodes produces a
    perfectly healthy loss curve and every other assertion still holds. The
    held-out episode simply stops being held out, and M3's validation number
    silently becomes a training number.
    """
    import mbfps.training.world_model as wm
    from mbfps.data.split import episode_split

    captured = {}
    real = wm.SequenceLoader

    def spy(*args, **kwargs):
        captured["paths"] = kwargs.get("paths")
        return real(*args, **kwargs)

    monkeypatch.setattr(wm, "SequenceLoader", spy)
    train_world_model(tiny(), buffer, out_dir=None)

    train_paths, val_paths = episode_split(buffer.episode_paths(), 0.2, 0)
    assert val_paths, "the fixture yields no validation episodes to hold out"
    assert captured["paths"] == train_paths
    assert not set(captured["paths"]) & set(val_paths)


def test_the_split_seed_is_fixed_at_zero_not_the_training_seed(buffer, monkeypatch):
    """Every arm and every seed must hold out the SAME episodes.

    If the split followed `cfg.train.seed`, a cross-arm or cross-seed
    comparison would measure which episodes each run happened to get rather
    than which representation is better -- and no test that only looks at one
    run can see it.
    """
    import mbfps.training.world_model as wm
    from mbfps.data.split import episode_split

    captured = []
    real = wm.SequenceLoader

    def spy(*args, **kwargs):
        captured.append(kwargs.get("paths"))
        return real(*args, **kwargs)

    monkeypatch.setattr(wm, "SequenceLoader", spy)
    train_world_model(tiny(seed=0), buffer, out_dir=None)
    train_world_model(tiny(seed=1), buffer, out_dir=None)

    seeded_differently = episode_split(buffer.episode_paths(), 0.2, 1)[0]
    assert seeded_differently != episode_split(buffer.episode_paths(), 0.2, 0)[0], (
        "seeds 0 and 1 give the same split on this fixture, so this test cannot "
        "detect a split that follows cfg.train.seed"
    )
    assert captured[0] == captured[1]
    assert captured[0] == episode_split(buffer.episode_paths(), 0.2, 0)[0]


def test_optimiser_uses_the_configured_learning_rate(buffer, monkeypatch):
    """`cfg.train.lr` is 1e-4 by spec and must reach the optimiser.

    `test_overfits_a_single_fixed_batch` builds its own optimiser, so nothing
    else in this file observes the loop's learning rate at all: a hardcoded
    value would train every arm at a rate the config does not name.
    """
    captured = {}
    real_adam = torch.optim.Adam

    def spy(params, lr, **kwargs):
        captured["lr"] = lr
        return real_adam(params, lr=lr, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", spy)
    train_world_model(tiny(lr=7.5e-4), buffer, out_dir=None)
    assert captured["lr"] == pytest.approx(7.5e-4)


def test_gradients_are_clipped_at_the_specified_norm(buffer, monkeypatch):
    """The clip exists and its max_norm is 100.0.

    A clip at 0.0 zeroes every gradient and trains nothing while every loss
    curve stays finite and reproducible; a missing clip lets one exploding
    sequence wreck a 20k-step run. Neither is visible in a three-step history.
    """
    captured = {}
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        captured["max_norm"] = max_norm
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    train_world_model(tiny(), buffer, out_dir=None)
    assert captured["max_norm"] == pytest.approx(100.0)


def test_checkpoint_is_written_with_its_arm_and_seed(buffer, tmp_path):
    """`out_dir` is the only route by which M3's weights reach M4."""
    out = tmp_path / "ckpt"
    train_world_model(tiny(seed=3), buffer, out_dir=out)
    path = out / "world_model_cnn_seed3.pt"
    assert path.is_file()
    payload = torch.load(path, weights_only=True)
    assert payload["arm"] == "cnn"
    assert payload["seed"] == 3
    assert any(k.startswith("rssm.") for k in payload["state_dict"])
    assert any(k.startswith("encoder.") for k in payload["state_dict"])
    assert any(k.startswith("heads.") for k in payload["state_dict"])


def test_history_records_every_step(buffer):
    """`steps`, `loss` and `parts` must agree, and `seconds` must be real."""
    history = train_world_model(tiny(steps=4), buffer, out_dir=None)
    assert history["steps"] == 4
    assert len(history["loss"]) == 4
    assert len(history["parts"]) == 4
    assert history["seconds"] > 0
    assert history["arm"] == "cnn"


# --------------------------------------------------------------------------
# Behavioural guards added on top of the brief. Every test above this line
# counts calls or inspects gradients before `step()` -- none of them looks at
# what the loop actually DID. These six close exactly that gap: a training
# loop that is a complete, silent no-op passes every test above.
# --------------------------------------------------------------------------


def test_parameters_actually_move_during_training(buffer, tmp_path):
    """The loop must actually train something, not just run without crashing.

    Found by mutation: moving `optimiser.zero_grad()` from before `loss.backward()` to
    after it survives every test above, INCLUDING
    `test_gradients_are_zeroed_between_optimiser_steps` -- `zero_grad()` is still called
    exactly once per step, just too late to matter. Adam's update then divides a zeroed
    first moment by a zeroed second moment and takes an exact 0.0 step every time, so the
    loop runs to completion, logs a perfectly finite and perfectly reproducible loss
    curve, and trains nothing at all for the entire run. Measured: max parameter change
    over 3 steps is exactly 0.0 under this mutation.

    Uses the checkpoint -- the only route a trained model leaves `train_world_model` by --
    so this observes the real loop, not a hand-rolled one. `seed_everything` followed by
    `WorldModel(cfg)` with the same `cfg` reproduces the loop's own pre-training weights
    bit-for-bit (the same determinism `test_same_seed_reproduces_the_loss_curve` already
    relies on), so diffing the checkpoint against that reference isolates exactly what
    `step()` changed.
    """
    cfg = tiny()
    seed_everything(cfg.train.seed)
    init_model = WorldModel(cfg)
    init_state = init_model.state_dict()
    trainable = [name for name, p in init_model.named_parameters() if p.requires_grad]
    init_snapshot = {name: init_state[name].clone() for name in trainable}

    out = tmp_path / "ckpt"
    train_world_model(cfg, buffer, out_dir=out)
    payload = torch.load(
        out / f"world_model_{cfg.arm}_seed{cfg.train.seed}.pt", weights_only=True
    )
    trained_state = payload["state_dict"]

    total_change = 0.0
    changed = 0
    for name in trainable:
        diff = (
            (trained_state[name].cpu() - init_snapshot[name].cpu()).abs().sum().item()
        )
        total_change += diff
        if diff > 0:
            changed += 1

    assert total_change > 0, (
        "no trainable parameter moved at all across training -- the loop is a no-op"
    )
    assert changed > len(trainable) / 2, (
        f"only {changed}/{len(trainable)} trainable tensors changed; a healthy "
        "optimiser step moves most of them, not just one submodule"
    )


def test_grad_clip_reduces_a_large_gradient_norm_to_the_configured_max(buffer, monkeypatch):
    """The clip must run AFTER `backward()`, on real gradients -- not before, on zeros.

    Found by mutation: moving `clip_grad_norm_` before `loss.backward()` survives every
    test above, including `test_gradients_are_clipped_at_the_specified_norm` -- that
    test's spy only records the `max_norm` ARGUMENT, never whether the call had any
    effect. Moved early, the clip runs on gradients that were just zeroed by
    `zero_grad()` (`.grad` is `None` at that point, since `Adam.zero_grad()` defaults to
    `set_to_none=True`) and is a permanent no-op: the run trains completely unclipped,
    exactly the failure this code's own comment claims to prevent.

    This spy multiplies whatever gradients exist at the moment `clip_grad_norm_` is
    called by 1e6, then measures the total grad norm before and after the real clip. If
    gradients are real (populated by `backward()` already having run), the pre-clip norm
    is astronomically over `max_norm` and the post-clip norm must come back down to it.
    If the clip runs before `backward()`, every `.grad` is `None` at that point,
    multiplying by 1e6 does nothing, and the pre-clip norm is exactly zero -- which the
    first assertion below rules out.
    """
    captured = {}
    real_clip = torch.nn.utils.clip_grad_norm_

    def grad_norm(params):
        squared = torch.zeros(())
        for p in params:
            if p.grad is not None:
                squared = squared + (p.grad.detach() ** 2).sum()
        return torch.sqrt(squared)

    def spy(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(1e6)
        before = grad_norm(params)
        result = real_clip(params, max_norm, *args, **kwargs)
        after = grad_norm(params)
        captured["before"] = float(before)
        captured["after"] = float(after)
        captured["max_norm"] = max_norm
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    train_world_model(tiny(steps=1), buffer, out_dir=None)

    assert captured["before"] > captured["max_norm"], (
        "gradients were zero (or absent) when clip_grad_norm_ was called, so this run "
        "cannot tell whether the clip runs before or after backward()"
    )
    assert captured["after"] <= captured["max_norm"] + 1e-3, (
        f"post-clip grad norm {captured['after']:.3f} exceeds max_norm "
        f"{captured['max_norm']}; the clip had no effect"
    )


def test_rssm_and_head_seeds_each_follow_cfg_train_seed():
    """`RSSM(seed=cfg.train.seed)` and `WorldModelHeads(seed=cfg.train.seed)` are two
    independent wires, and each can be hardcoded without the other noticing.

    Found by mutation: hardcoding EITHER constructor's `seed=` kwarg to a constant `0`
    survives `test_different_seeds_diverge`, because that test only requires ONE source
    of variation anywhere in the loop -- the loader, the RSSM init, or the heads init --
    to produce a different loss curve. A multi-seed study needs every source pinned, not
    just one, so each submodule is checked on its own here.
    """
    model_a = WorldModel(tiny(seed=0))
    model_b = WorldModel(tiny(seed=1))

    rssm_a = torch.cat([p.detach().flatten() for p in model_a.rssm.parameters()])
    rssm_b = torch.cat([p.detach().flatten() for p in model_b.rssm.parameters()])
    assert not torch.allclose(rssm_a, rssm_b), (
        "RSSM parameters are identical across cfg.train.seed 0 and 1 -- the RSSM "
        "constructor's seed looks hardcoded"
    )

    heads_a = torch.cat([p.detach().flatten() for p in model_a.heads.parameters()])
    heads_b = torch.cat([p.detach().flatten() for p in model_b.heads.parameters()])
    assert not torch.allclose(heads_a, heads_b), (
        "WorldModelHeads parameters are identical across cfg.train.seed 0 and 1 -- the "
        "heads constructor's seed looks hardcoded"
    )


def test_loader_seed_follows_cfg_train_seed(buffer, monkeypatch):
    """The loop's `SequenceLoader(seed=cfg.train.seed, ...)` wire, checked directly.

    Found by mutation: hardcoding the loader's `seed=` kwarg to a constant `0` survives
    `test_different_seeds_diverge` -- comparing loss curves conflates the loader's
    sampling seed with `seed_everything`'s effect on the model's own forward-pass
    randomness (the categorical sampler) and weight init, any ONE of which is enough to
    move the loss. Two runs at different `cfg.train.seed` produce different losses even
    if BOTH loaders sample the identical windows, so a loss-only comparison cannot tell
    whether this wire is connected at all. This looks at the windows themselves.
    """
    import mbfps.training.world_model as wm

    captured_batches = []
    real_loader = wm.SequenceLoader

    def spy(*args, **kwargs):
        loader = real_loader(*args, **kwargs)
        real_sample = loader.sample

        def sample(*a, **kw):
            batch = real_sample(*a, **kw)
            captured_batches.append(
                (batch["window_start"].copy(), batch["episode_index"].copy())
            )
            return batch

        loader.sample = sample
        return loader

    monkeypatch.setattr(wm, "SequenceLoader", spy)
    train_world_model(tiny(seed=0, steps=1), buffer, out_dir=None)
    train_world_model(tiny(seed=1, steps=1), buffer, out_dir=None)

    assert len(captured_batches) == 2
    (starts_a, idx_a), (starts_b, idx_b) = captured_batches
    assert not (np.array_equal(starts_a, starts_b) and np.array_equal(idx_a, idx_b)), (
        "cfg.train.seed 0 and 1 sampled the identical window(s) from the training "
        "split, so this test cannot detect a loader seed hardcoded to a constant"
    )


def test_val_fraction_is_pinned_at_point_two(buffer, monkeypatch):
    """`episode_split(..., val_fraction=0.2, ...)` -- the literal, not just its effect.

    Found by mutation: `val_fraction=0.25` survives every test above, because on this
    five-episode fixture `int(5 * 0.25) == int(5 * 0.2) == 1` -- the two values produce
    an IDENTICAL split here and pass `test_training_holds_out_the_validation_split`
    unchanged. On the real 122-episode dataset the same drift moves the hold-out from 24
    to 30 episodes. This pins the literal value the call site passes, not its effect on
    one fixture size.
    """
    import mbfps.training.world_model as wm

    captured = {}
    real_split = wm.episode_split

    def spy(paths, val_fraction, seed):
        captured["val_fraction"] = val_fraction
        return real_split(paths, val_fraction, seed)

    monkeypatch.setattr(wm, "episode_split", spy)
    train_world_model(tiny(steps=1), buffer, out_dir=None)
    assert captured["val_fraction"] == pytest.approx(0.2)


def test_model_parameters_land_on_the_configured_device(buffer, monkeypatch):
    """`cfg.train.device` must actually place the model, not merely get threaded through.

    Found by mutation: calling `get_device()` with no argument (defaulting to `"mps"`)
    survives every test above that only inspects losses or history, and every test in
    this file that declares `device="cpu"` was, under that mutation, silently training on
    MPS instead. This spies on `get_device` to prove `cfg.train.device` reaches it, and
    then reads a live parameter's `.device` off the actual model the loop built and
    trained, to prove that value is what placed it.
    """
    import mbfps.training.world_model as wm

    captured = {}
    real_get_device = wm.get_device

    def spy(prefer="mps"):
        captured["prefer"] = prefer
        return real_get_device(prefer=prefer)

    real_world_model_cls = wm.WorldModel
    instances = []

    def spy_world_model(cfg):
        model = real_world_model_cls(cfg)
        instances.append(model)
        return model

    monkeypatch.setattr(wm, "get_device", spy)
    monkeypatch.setattr(wm, "WorldModel", spy_world_model)
    train_world_model(tiny(steps=1, device="cpu"), buffer, out_dir=None)

    assert captured["prefer"] == "cpu", (
        "cfg.train.device did not reach get_device() as its 'prefer' argument"
    )
    assert instances, "WorldModel was never constructed"
    device_seen = next(instances[0].parameters()).device
    assert device_seen.type == "cpu", (
        f"model parameters landed on {device_seen.type!r}, not the configured 'cpu'"
    )
