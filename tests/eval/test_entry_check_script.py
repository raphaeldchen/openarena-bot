import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.eval.study import load_record
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import get_config

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "entry_check.py"
KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def _basic_like(t, fill, kill_at=None):
    rewards = np.full(t, -4.0, dtype=np.float32)
    terminated = np.zeros(t, dtype=bool)
    if kill_at is not None:
        rewards[kill_at] = 97.0
        terminated[kill_at] = True
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 1] = fill
    return Episode(
        obs=obs, actions=(np.arange(t) % 4).astype(np.int32), rewards=rewards,
        terminated=terminated, truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS, policy_name="random", seed=fill, scenario="basic",
    )


@pytest.fixture
def trained(tmp_path):
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 11):
        t = 30 if fill % 2 else 12
        buf.add(_basic_like(t, fill, kill_at=t - 1 if fill % 3 == 0 else None))
    cfg = get_config("cnn", steps=3, seq_len=16, batch_size=4, device="cpu",
                     n_actions=4, pad_short=True, reward_transform="symlog")
    train_world_model(cfg, buf, out_dir=tmp_path / "wm")
    return tmp_path, tmp_path / "wm" / "world_model_cnn_seed0.pt"


def _run(*args):
    proc = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_measure_writes_a_record_and_read_maps_the_verdict(trained):
    root, ckpt = trained
    out = root / "entry"
    code, log = _run("--phase", "measure", "--checkpoint", ckpt, "--data", root / "data",
                     "--out", out, "--resamples", 50, "--no-position", "--device", "cpu")
    assert code == 0, log
    record = load_record(out / "entry_cnn_seed0.json")
    for key in ("c_R", "c_C", "imagined", "posterior", "counts", "verdict", "git_sha"):
        assert key in record
    assert record["reward_transform"] == "symlog" and record["n_actions"] == 4
    assert record["counts"]["episodes"] >= 1

    # The read refuses without a control record.
    code, log = _run("--phase", "read", "--out", out, "--arm", "cnn", "--seed", 0)
    assert code == 11, log

    code, log = _run("--phase", "measure", "--checkpoint", ckpt, "--data", root / "data",
                     "--out", out, "--role", "control", "--resamples", 50,
                     "--no-position", "--device", "cpu")
    assert code == 0, log
    code, log = _run("--phase", "read", "--out", out, "--arm", "cnn", "--seed", 0)
    control = load_record(out / "entry_control.json")
    if control["imagined"]["return"]["ci_low"] > 0:   # NaN compares False, as the script does
        assert code == 62
    else:
        assert code in (0, 60, 61)
    assert (out / "entry.txt").is_file()
