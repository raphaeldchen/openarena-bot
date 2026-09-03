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
        feats = rng.random((41, 64, 384)).astype(np.float16)
        np.save(path.with_suffix(".features.npy"), feats)
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
