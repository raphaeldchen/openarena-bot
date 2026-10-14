import copy

import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.training.ac_trainer import (
    ACConfig, ReturnScale, continue_weights, freeze, lambda_returns, start_states,
    train_actor_critic,
)
from mbfps.training.world_model import WorldModel, load_world_model, train_world_model
from mbfps.utils.config import get_config
from mbfps.utils.device import to_device


def test_lambda_returns_hand_computed():
    """4 steps, a predicted end at step 2 (discount 0 there)."""
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    d = torch.tensor([[0.9, 0.9, 0.0, 0.9]])
    v = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
    lam = 0.5
    g4 = 50.0
    g3 = 4.0 + 0.9 * (0.5 * 50.0 + 0.5 * g4)
    g2 = 3.0 + 0.0 * (0.5 * 40.0 + 0.5 * g3)
    g1 = 2.0 + 0.9 * (0.5 * 30.0 + 0.5 * g2)
    g0 = 1.0 + 0.9 * (0.5 * 20.0 + 0.5 * g1)
    out = lambda_returns(r, d, v, lam)
    torch.testing.assert_close(out, torch.tensor([[g0, g1, g2, g3]]))


def test_continue_weights():
    c = torch.tensor([[0.5, 0.5, 0.0, 1.0]])
    torch.testing.assert_close(continue_weights(c), torch.tensor([[1.0, 0.5, 0.25, 0.0]]))


def test_return_scale_floor_and_ema():
    s = ReturnScale(decay=0.5)
    assert s.update(torch.zeros(100)) == 1.0          # P95 - P5 = 0 -> floor 1
    x = torch.linspace(0, 100, 101)
    s2 = ReturnScale(decay=0.5)
    first = s2.update(x)                               # first update sets directly
    assert first == pytest.approx(90.0)
    second = s2.update(x * 2)                          # 0.5*90 + 0.5*180
    assert second == pytest.approx(135.0)


KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def _ep(t, fill):
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 1] = fill
    term = np.zeros(t, dtype=bool); term[-1] = fill % 2 == 0
    return Episode(obs=obs, actions=(np.arange(t) % 4).astype(np.int32),
                   rewards=np.full(t, -4.0, np.float32), terminated=term,
                   truncated=np.zeros(t, bool),
                   privileged=np.zeros((t + 1, len(KEYS)), np.float32),
                   privileged_keys=KEYS, policy_name="random", seed=fill, scenario="basic")


@pytest.fixture
def world(tmp_path):
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 8):
        buf.add(_ep(30 if fill % 2 else 6, fill))
    cfg = get_config("cnn", steps=2, seq_len=16, batch_size=4, device="cpu",
                     n_actions=4, pad_short=True, reward_transform="symlog")
    train_world_model(cfg, buf, out_dir=tmp_path / "wm")
    return buf, load_world_model(tmp_path / "wm" / "world_model_cnn_seed0.pt", torch.device("cpu"))


def test_start_states_exclude_padding(world):
    buf, model = world
    batch = SequenceLoader(buf, batch_size=8, seq_len=16, seed=0, pad_short=True,
                           load_obs=True).sample()
    batch["mask"][0, 5:] = False       # a known pad, whatever the draw was
    h, z = start_states(model, to_device(batch, torch.device("cpu")))
    assert h.shape[0] == int(batch["mask"].sum())
    assert h.shape[0] <= 8 * 16 - 11
    assert z.shape == (h.shape[0], 1024)


def test_isolation_world_model_untouched(world, tmp_path):
    buf, model = world
    before = copy.deepcopy(model.state_dict())
    cfg = ACConfig(updates=3, checkpoint_every=3, batch_size=4, seq_len=16, device="cpu")
    history = train_actor_critic(model, buf, cfg, out_dir=tmp_path / "ac", log_every=0)
    assert history["isolation_ok"] is True
    for p in model.parameters():
        assert p.grad is None and not p.requires_grad
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert (tmp_path / "ac" / "actor_critic_cnn_seed0_step3.pt").is_file()


def test_training_is_reproducible(world, tmp_path):
    buf, model = world
    cfg = ACConfig(updates=2, checkpoint_every=2, batch_size=4, seq_len=16, device="cpu")
    a = train_actor_critic(model, buf, cfg, out_dir=None, log_every=0)
    b = train_actor_critic(model, buf, cfg, out_dir=None, log_every=0)
    assert a["actor_loss"] == b["actor_loss"]
    assert a["critic_loss"] == b["critic_loss"]


def test_checkpoint_every_must_divide_updates():
    with pytest.raises(ValueError, match="checkpoint_every"):
        ACConfig(updates=10, checkpoint_every=3)


def test_script_writes_record_and_checkpoints(world, tmp_path):
    import json, subprocess, sys
    from pathlib import Path
    buf, _ = world
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_actor_critic.py"
    out = tmp_path / "ac_script"
    proc = subprocess.run([sys.executable, str(script),
                           "--world-model", str(tmp_path / "wm" / "world_model_cnn_seed0.pt"),
                           "--data", str(tmp_path / "data"), "--out", str(out),
                           "--updates", "2", "--checkpoint-every", "1", "--device", "cpu"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads((out / "train_cnn_seed0.json").read_text())
    assert record["isolation_ok"] is True and record["updates"] == 2
    assert (out / "actor_critic_cnn_seed0_step1.pt").is_file()
    assert (out / "actor_critic_cnn_seed0_step2.pt").is_file()


def _count_empty_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(1))
    return calls


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_cache_released_every_update(world, tmp_path, monkeypatch):
    """N (masked-in start states) changes every update, so MPS never reuses blocks:
    driver memory grew ~60 MB/update to OOM near update 350. Release it each update."""
    buf, _ = world
    model = load_world_model(tmp_path / "wm" / "world_model_cnn_seed0.pt", torch.device("mps"))
    calls = _count_empty_cache(monkeypatch)
    cfg = ACConfig(updates=3, checkpoint_every=3, batch_size=4, seq_len=16, device="mps")
    train_actor_critic(model, buf, cfg, out_dir=None, log_every=0)
    assert len(calls) == 3


def test_cpu_never_touches_mps_cache(world, monkeypatch):
    buf, model = world
    calls = _count_empty_cache(monkeypatch)
    cfg = ACConfig(updates=2, checkpoint_every=2, batch_size=4, seq_len=16, device="cpu")
    train_actor_critic(model, buf, cfg, out_dir=None, log_every=0)
    assert calls == []
