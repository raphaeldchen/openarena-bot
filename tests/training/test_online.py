import json

import numpy as np
import pytest
import torch

from mbfps.data.episode import load_episode
from mbfps.eval.study import load_record
from mbfps.training.dqn import DQNConfig
from mbfps.training.online import OnlineConfig, train_online
from tests.training.fake_env import FakeEnv, fake_encode

CPU = torch.device("cpu")
TINY = OnlineConfig(budget=60, prefill=10, act_steps=10, rounds_per_act=2, eval_every=20,
                    eval_seeds=(1_000_000, 1_000_001, 1_000_002))
DREAMER_KW = {"batch_size": 2, "seq_len": 8}


def _run(tmp_path, arm="dreamer", stop=None, kw=None):
    kw = DREAMER_KW if (arm == "dreamer" and kw is None) else (kw or {})
    return train_online(arm, seed=0, out_dir=tmp_path, env_factory=FakeEnv, encode=fake_encode,
                        device=CPU, cfg=TINY, arm_kwargs=kw, stop_after_step=stop)


@pytest.mark.parametrize("arm", ["dreamer", "dqn"])
def test_run_spends_exactly_the_budget(tmp_path, arm):
    kw = {"cfg": DQNConfig(batch_size=8)} if arm == "dqn" else None
    rec = _run(tmp_path, arm, kw=kw)
    assert rec["complete"] is True
    assert rec["budget_steps"] == 60
    assert [e["step"] for e in rec["evals"]] == [0, 20, 40, 60]
    assert rec["update_rounds"] == (60 - 10) // 10 * 2
    stored = sum(load_episode(p).length for p in (tmp_path / "replay").glob("ep_*.npz"))
    assert stored == 60
    for e in rec["evals"]:
        for mode, eps in e["modes"].items():
            assert [x["seed"] for x in eps] == list(TINY.eval_seeds)
    assert [x["seed"] for x in rec["random"]] == list(TINY.eval_seeds)


def test_prefill_cuts_at_exactly_the_prefill_step(tmp_path):
    _run(tmp_path)
    eps = sorted((tmp_path / "replay").glob("ep_*.npz"))
    lengths = np.cumsum([load_episode(p).length for p in eps])
    assert 10 in lengths.tolist()          # an episode boundary at the prefill step
    first_arm = eps[lengths.tolist().index(10) + 1]
    assert load_episode(first_arm).policy_name == "dreamer"


def test_resume_is_bit_identical(tmp_path):
    full = _run(tmp_path / "full")
    part = _run(tmp_path / "resumed", stop=20)
    assert part["complete"] is False
    resumed = _run(tmp_path / "resumed")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("seconds", "timing", "git_sha")}
    assert json.dumps(strip(full), sort_keys=True) == json.dumps(strip(resumed), sort_keys=True)
    a = torch.load(tmp_path / "full" / "state.pt", weights_only=False)["arm"]["world_model"]
    b = torch.load(tmp_path / "resumed" / "state.pt", weights_only=False)["arm"]["world_model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_evaluation_steps_never_enter_the_replay(tmp_path):
    rec = _run(tmp_path)
    seeds = {load_episode(p).seed for p in (tmp_path / "replay").glob("ep_*.npz")}
    assert not seeds & set(TINY.eval_seeds)


def test_isolation_recorded_for_dreamer(tmp_path):
    assert _run(tmp_path)["isolation_ok"] is True


def test_finished_run_without_state_refuses_to_restart(tmp_path):
    _run(tmp_path)
    assert load_record(tmp_path / "record.json")["complete"] is True
    n_files = len(list((tmp_path / "replay").iterdir()))
    (tmp_path / "state.pt").unlink()
    with pytest.raises(RuntimeError, match=str(tmp_path)):
        _run(tmp_path)
    assert len(list((tmp_path / "replay").iterdir())) == n_files
