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
