import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.training.autoencoder import AutoencoderModel, train_autoencoder
from mbfps.utils.config import get_config

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    return Episode(
        obs=np.full((t + 1, *OBS_SHAPE), fill, dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    rng = np.random.default_rng(0)
    for fill in (10, 60, 110):
        buf.add(make_episode(t=40, fill=fill))
    for path in buf.episode_paths():
        # Cache both backbones: frozen_ssl reads dinov2, random_vit reads its
        # own suffixed cache, and a fixture that caches only one would hide
        # exactly the bug this file now tests for (the two arms colliding).
        for suffix in (".features.npy", ".features_random_vit.npy"):
            feats = rng.random((41, 64, 384)).astype(np.float16)
            np.save(path.with_suffix(suffix), feats)
    return buf


def tiny(arm: str):
    return get_config(arm, steps=3, batch_size=2, seq_len=8, device="cpu")


def test_cnn_model_reconstructs_to_observation_shape(buffer):
    cfg = tiny("cnn")
    model = AutoencoderModel(cfg)
    batch = {"obs": torch.randint(0, 256, (2, 9, *OBS_SHAPE), dtype=torch.uint8)}
    recon, target = model(batch)
    assert recon.shape == target.shape
    assert recon.shape[-3:] == OBS_SHAPE


def test_ssl_model_consumes_features_and_reconstructs_pixels(buffer):
    cfg = tiny("frozen_ssl")
    model = AutoencoderModel(cfg)
    batch = {
        "obs": torch.randint(0, 256, (2, 9, *OBS_SHAPE), dtype=torch.uint8),
        "features": torch.randn(2, 9, 64, 384, dtype=torch.float16),
    }
    recon, target = model(batch)
    assert recon.shape == target.shape


def test_ssl_model_ignores_obs_as_input(buffer):
    """Feature arms must not sneak pixels into the encoder path."""
    cfg = tiny("frozen_ssl")
    model = AutoencoderModel(cfg)
    feats = torch.randn(2, 9, 64, 384, dtype=torch.float16)
    a = model({"obs": torch.zeros(2, 9, *OBS_SHAPE, dtype=torch.uint8),
               "features": feats})[0]
    b = model({"obs": torch.full((2, 9, *OBS_SHAPE), 255, dtype=torch.uint8),
               "features": feats})[0]
    assert torch.allclose(a, b), "reconstruction changed with obs; features arm leaked pixels"


@pytest.mark.parametrize("arm", ["cnn", "frozen_ssl", "random_vit"])
def test_training_runs_and_returns_history(buffer, arm):
    history = train_autoencoder(tiny(arm), buffer, out_dir=None)
    assert history["arm"] == arm
    assert history["steps"] == 3
    assert len(history["loss"]) == 3
    assert all(np.isfinite(history["loss"]))


def test_training_reduces_loss_on_a_trivial_dataset(buffer):
    """Every episode is a constant colour, so a working model must fit it fast."""
    history = train_autoencoder(
        get_config("cnn", steps=60, batch_size=2, seq_len=4, device="cpu", lr=1e-3),
        buffer,
        out_dir=None,
    )
    first, last = np.mean(history["loss"][:5]), np.mean(history["loss"][-5:])
    assert last < first * 0.6, f"loss barely moved: {first:.4f} -> {last:.4f}"


def test_ssl_arm_trains_only_its_bottleneck(buffer):
    """The frozen backbone must contribute no trainable parameters."""
    model = AutoencoderModel(tiny("frozen_ssl"))
    trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert trainable == 12_320


def test_checkpoint_written_when_out_dir_given(buffer, tmp_path):
    out = tmp_path / "run"
    train_autoencoder(tiny("cnn"), buffer, out_dir=out)
    assert (out / "autoencoder_cnn.pt").is_file()


def test_same_seed_reproduces_the_loss_curve(buffer):
    a = train_autoencoder(tiny("cnn"), buffer, out_dir=None)
    b = train_autoencoder(tiny("cnn"), buffer, out_dir=None)
    assert np.allclose(a["loss"], b["loss"]), "training is not reproducible from its seed"


def test_different_seeds_give_different_curves(buffer):
    a = train_autoencoder(get_config("cnn", steps=3, batch_size=2, seq_len=8,
                                     device="cpu", seed=0), buffer, out_dir=None)
    b = train_autoencoder(get_config("cnn", steps=3, batch_size=2, seq_len=8,
                                     device="cpu", seed=1), buffer, out_dir=None)
    assert not np.allclose(a["loss"], b["loss"])


def test_pixel_arm_trains_without_any_feature_cache(tmp_path):
    """The pixel arm must work on a dataset that was never feature-cached.

    The shared `buffer` fixture caches features for every arm, so a trainer
    that always requested them stays green there. This buffer has no cache at
    all, so requesting features raises FileNotFoundError in the loader.
    Verified by mutation: forcing load_features=True passed all other tests.
    """
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (10, 60, 110):
        buf.add(make_episode(t=40, fill=fill))
    assert not list(tmp_path.glob("*.features*.npy")), "fixture must have no cache"

    history = train_autoencoder(tiny("cnn"), buf, out_dir=None)
    assert history["arm"] == "cnn"
    assert history["steps"] == 3


@pytest.mark.parametrize("arm", ["frozen_ssl", "random_vit"])
def test_feature_arms_require_a_cache(tmp_path, arm):
    """The complement: proves the test above is not vacuous.

    If the trainer never requested features for any arm, the test above would
    pass for the wrong reason. These arms must fail loudly without a cache.
    """
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (10, 60, 110):
        buf.add(make_episode(t=40, fill=fill))

    with pytest.raises(FileNotFoundError, match="no cached features"):
        train_autoencoder(tiny(arm), buf, out_dir=None)


def test_random_vit_arm_will_not_silently_read_the_dinov2_cache(tmp_path):
    """The bug this guards: without an explicit backbone the loader defaults to
    dinov2, so the control arm trained on the treatment arm's features with no
    error at all. Here only a dinov2 cache exists, so the random_vit arm must
    fail rather than quietly use it.
    """
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    rng = np.random.default_rng(0)
    for fill in (10, 60, 110):
        buf.add(make_episode(t=40, fill=fill))
    for path in buf.episode_paths():
        np.save(path.with_suffix(".features.npy"),
                rng.random((41, 64, 384)).astype(np.float16))

    with pytest.raises(FileNotFoundError, match="features_random_vit"):
        train_autoencoder(tiny("random_vit"), buf, out_dir=None)


def test_random_vit_arm_reads_its_own_cache(tmp_path):
    """The complement: with the right cache present it must train normally."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    rng = np.random.default_rng(1)
    for fill in (10, 60, 110):
        buf.add(make_episode(t=40, fill=fill))
    for path in buf.episode_paths():
        np.save(path.with_suffix(".features_random_vit.npy"),
                rng.random((41, 64, 384)).astype(np.float16))

    history = train_autoencoder(tiny("random_vit"), buf, out_dir=None)
    assert history["arm"] == "random_vit"
    assert history["steps"] == 3


# --- Arm parity of the shared decoder ---------------------------------------


def test_decoder_initialises_identically_for_every_arm():
    """The decoder is shared, so it must not depend on which encoder preceded it.

    `seed_everything` runs once and `AutoencoderModel` builds the encoder first,
    so the decoder's weights were drawn from a global RNG that the encoder had
    already advanced -- by 26,382,304 draws for the CNN arm against 12,320 for
    the bottleneck arms. The "shared" decoder therefore started from different
    weights for the pixel arm than for the feature arms, an arm-parity violation
    in a module that is supposed to be identical across all three.
    """
    from mbfps.utils.seeding import seed_everything

    signatures = {}
    for arm in ("cnn", "frozen_ssl", "random_vit"):
        cfg = get_config(arm, steps=1, batch_size=2, seq_len=1, device="cpu")
        seed_everything(cfg.train.seed)
        model = AutoencoderModel(cfg)
        signatures[arm] = torch.cat(
            [p.detach().flatten() for p in model.decoder.parameters()]
        )

    reference = signatures["cnn"]
    # Guard against a vacuous comparison: an all-zero decoder would match too.
    assert reference.abs().sum() > 0
    for arm, weights in signatures.items():
        assert torch.equal(reference, weights), (
            f"decoder init for {arm!r} differs from 'cnn' -- the shared decoder "
            "is not arm-invariant"
        )


def test_decoder_init_tracks_the_train_seed():
    """Complement to the parity test: it must still depend on the seed itself,
    or a hardcoded constant would satisfy parity while destroying seed control."""
    from mbfps.utils.seeding import seed_everything

    weights = []
    for seed in (0, 1):
        cfg = get_config("cnn", steps=1, batch_size=2, seq_len=1, device="cpu", seed=seed)
        seed_everything(cfg.train.seed)
        model = AutoencoderModel(cfg)
        weights.append(torch.cat([p.detach().flatten() for p in model.decoder.parameters()]))
    assert not torch.equal(weights[0], weights[1])


# --- Learning tests the decoder bias alone cannot pass -----------------------
# `test_training_reduces_loss_on_a_trivial_dataset` builds every episode from a
# single flat colour, which PixelDecoder's output biases fit on their own with
# zero contribution from the encoder -- so severing the encoder left it green.
# It also covers only the `cnn` arm, leaving the 12,320-parameter bottleneck
# (the entire trainable model of arms 2 and 3) with no evidence it learns at
# all. Below, the target varies per frame and is predictable ONLY from the
# encoder's input, so a constant predictor is capped at the target's variance.


def make_learnable_episode(t: int, phase: int) -> Episode:
    """Frame `i` is a flat colour determined by `i`; features encode the same.

    A constant predictor can do no better than the variance of those colours,
    so beating that margin requires information to flow through the encoder.
    """
    levels = ((np.arange(t + 1) * 37 + phase * 11) % 200 + 28).astype(np.uint8)
    obs = np.repeat(levels, np.prod(OBS_SHAPE)).reshape(t + 1, *OBS_SHAPE)
    return Episode(
        obs=obs,
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=phase,
        scenario="my_way_home",
    )


@pytest.fixture
def learnable_buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for phase in (0, 1, 2):
        buf.add(make_learnable_episode(t=40, phase=phase))
    for path in buf.episode_paths():
        levels = np.load(path)["obs"][:, 0, 0, 0].astype(np.float32) / 255.0
        # Features must carry the information the target needs, or the feature
        # arms could not learn this even with a perfect bottleneck. The level
        # has to change each token's *direction*, not merely its scale: the
        # bottleneck standardises its input, so a token that is a constant, or
        # one that is a fixed pattern times the level, normalises to the same
        # vector for every frame and carries nothing. Real backbone features
        # put ~100% of their variance within tokens, so this matches them.
        rng_pat = np.random.default_rng(7)
        base = rng_pat.standard_normal(384).astype(np.float32)
        delta = rng_pat.standard_normal(384).astype(np.float32)
        feats = (base[None, None, :] + levels[:, None, None] * delta[None, None, :])
        feats = np.broadcast_to(feats, (len(levels), 64, 384)).astype(np.float16)
        for suffix in (".features.npy", ".features_random_vit.npy"):
            np.save(path.with_suffix(suffix), feats)
    return buf


@pytest.mark.parametrize("arm", ["cnn", "frozen_ssl", "random_vit"])
def test_every_arm_learns_something_its_decoder_bias_cannot(learnable_buffer, arm):
    """Severing the encoder must break this for all three arms."""
    history = train_autoencoder(
        get_config(arm, steps=120, batch_size=4, seq_len=2, device="cpu", lr=1e-3),
        learnable_buffer,
        out_dir=None,
    )
    last = float(np.mean(history["loss"][-10:]))
    # A constant predictor's floor is the variance of the frame levels.
    levels = (np.arange(41) * 37) % 200 + 28
    floor = float(np.var(levels / 255.0))
    assert last < 0.5 * floor, (
        f"{arm}: final loss {last:.5f} is not below half the constant-predictor "
        f"floor {floor:.5f} -- no information reached the decoder"
    )
