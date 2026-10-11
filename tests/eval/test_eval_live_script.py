import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_live.py"
AC_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_actor_critic.py"


def _run(script, *args):
    p = subprocess.run([sys.executable, str(script), *map(str, args)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


@pytest.fixture
def agent(tmp_path):
    """A frozen_ssl world model on 4 actions over a tiny basic-like buffer with
    dinov2-geometry caches, and 2 actor-critic updates."""
    import numpy as np
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.episode import Episode
    from mbfps.envs.protocol import OBS_SHAPE
    from mbfps.training.world_model import train_world_model
    from mbfps.utils.config import get_config

    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 8):
        t = 30
        buf.add(Episode(obs=np.zeros((t + 1, *OBS_SHAPE), np.uint8),
                        actions=(np.arange(t) % 4).astype(np.int32),
                        rewards=np.full(t, -4.0, np.float32), terminated=np.zeros(t, bool),
                        truncated=np.zeros(t, bool),
                        privileged=np.zeros((t + 1, len(keys)), np.float32),
                        privileged_keys=keys, policy_name="random", seed=fill, scenario="basic"))
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        np.save(path.with_suffix(".features.npy"), rng.random((31, 64, 384)).astype(np.float16))
    cfg = get_config("frozen_ssl", steps=2, seq_len=16, batch_size=4, device="cpu",
                     n_actions=4, pad_short=True, reward_transform="symlog")
    train_world_model(cfg, buf, out_dir=tmp_path / "wm")
    wm = tmp_path / "wm" / "world_model_frozen_ssl_seed0.pt"
    code, log = _run(AC_SCRIPT, "--world-model", wm, "--data", tmp_path / "data",
                     "--out", tmp_path / "ac", "--updates", 2, "--checkpoint-every", 2,
                     "--device", "cpu")
    assert code == 0, log
    return tmp_path, wm


def test_measure_and_read(agent):
    root, wm = agent
    out = root / "live"
    code, log = _run(SCRIPT, "--phase", "measure", "--world-model", wm,
                     "--agent-dir", root / "ac", "--out", out, "--steps", "2",
                     "--episodes", 3, "--device", "cpu", "--encode-stub")
    assert code == 0, log
    record = json.loads((out / "live_frozen_ssl_seed0.json").read_text())
    assert len(record["random"]) == 3 and len(record["untrained"]) == 3
    assert set(record["rungs"]) == {"2"}
    assert [e["seed"] for e in record["random"]] == [e["seed"] for e in record["rungs"]["2"]["sample"]]

    code, log = _run(SCRIPT, "--phase", "read", "--out", out, "--arm", "frozen_ssl",
                     "--seed", 0, "--final-step", 2, "--agent-dir", root / "ac")
    assert code in (0, 70, 71), log
    assert (out / "live_frozen_ssl_seed0.txt").is_file()


def test_read_refuses_a_missing_record(tmp_path):
    code, _ = _run(SCRIPT, "--phase", "read", "--out", tmp_path, "--arm", "frozen_ssl",
                   "--seed", 0, "--final-step", 2, "--agent-dir", tmp_path)
    assert code == 11


def _write_live(out, untrained_return, agent_return, isolation=True):
    from mbfps.eval.study import write_record
    seeds = list(range(100))
    ep = lambda s, r: {"seed": s, "return": r + (s % 7), "length": 10, "killed": False,
                       "truncated": True, "imagined_return": r}
    write_record(out / "live_frozen_ssl_seed0.json", {
        "arm": "frozen_ssl", "seed": 0, "seeds": seeds,
        "random": [ep(s, -170.0) for s in seeds],
        "untrained": [ep(s, untrained_return) for s in seeds],
        "rungs": {"10000": {m: [ep(s, agent_return) for s in seeds]
                            for m in ("sample", "argmax")}}})
    write_record(out / "train_frozen_ssl_seed0.json", {"isolation_ok": isolation})


@pytest.mark.parametrize("untrained, agent, isolation, expected", [
    (-170.0, 80.0, True, 0),      # PASS
    (-170.0, -170.0, True, 70),   # AGENT_NOT_BETTER
    (80.0, 80.0, True, 71),       # CONTROL_BROKEN: the untrained actor passes
    (-170.0, 80.0, False, 72),    # ISOLATION_BROKEN
])
def test_read_maps_each_outcome(tmp_path, untrained, agent, isolation, expected):
    _write_live(tmp_path, untrained, agent, isolation)
    code, log = _run(SCRIPT, "--phase", "read", "--out", tmp_path, "--agent-dir", tmp_path,
                     "--arm", "frozen_ssl", "--seed", 0, "--final-step", 10000)
    assert code == expected, log


def test_pixel_ae_checkpoint_reaches_the_extractor(monkeypatch, tmp_path):
    """Live pixel_ae features must come from the autoencoder the cache was
    written with, not FeatureExtractor's my_way_home default."""
    import torch
    from mbfps.data import features
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    sys.path.insert(0, str(SCRIPT.parent))
    import eval_live

    seen = {}

    class Fake:
        def __init__(self, backbone, device, checkpoint=None):
            seen.update(backbone=backbone, checkpoint=checkpoint)
        def encode(self, frames):
            return frames

    monkeypatch.setattr(features, "FeatureExtractor", Fake)
    model = WorldModel(get_config("pixel_ae", device="cpu", n_actions=4))
    ckpt = tmp_path / "autoencoder_cnn.pt"
    eval_live._encoder_fn(model, torch.device("cpu"), stub=False, pixel_ae_checkpoint=ckpt)
    assert seen == {"backbone": "pixel_ae", "checkpoint": ckpt}


def test_pixel_ae_without_a_checkpoint_is_refused(monkeypatch):
    """No silent fallback to FeatureExtractor's my_way_home autoencoder."""
    import torch
    from mbfps.data import features
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    sys.path.insert(0, str(SCRIPT.parent))
    import eval_live

    class Fake:
        def __init__(self, backbone, device, checkpoint=None):
            raise AssertionError("must not construct an extractor")

    monkeypatch.setattr(features, "FeatureExtractor", Fake)
    model = WorldModel(get_config("pixel_ae", device="cpu", n_actions=4))
    with pytest.raises(ValueError, match="--pixel-ae-checkpoint"):
        eval_live._encoder_fn(model, torch.device("cpu"), stub=False, pixel_ae_checkpoint=None)


@pytest.mark.parametrize("which", ["random", "untrained", ("rungs", "sample"), ("rungs", "argmax")])
def test_read_refuses_unpaired_seeds(tmp_path, which):
    from mbfps.eval.study import load_record, write_record
    _write_live(tmp_path, -170.0, 80.0)
    path = tmp_path / "live_frozen_ssl_seed0.json"
    record = load_record(path)
    record.pop("nonfinite", None)  # write_record re-derives it
    lst = record[which] if isinstance(which, str) else record["rungs"]["10000"][which[1]]
    lst[0], lst[1] = lst[1], lst[0]
    write_record(path, record)
    code, log = _run(SCRIPT, "--phase", "read", "--out", tmp_path, "--agent-dir", tmp_path,
                     "--arm", "frozen_ssl", "--seed", 0, "--final-step", 10000)
    assert code == 73, log
    assert "SEED_MISMATCH" in log
