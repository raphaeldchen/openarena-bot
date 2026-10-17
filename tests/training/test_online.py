import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.data.episode import load_episode
from mbfps.eval.study import load_record, write_record
from mbfps.training.dqn import DQNConfig
from mbfps.training import online
from mbfps.training.online import (OnlineConfig, delete_replay_features, train_online,
                                   verify_record)
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


def _strip(r):
    return json.dumps({k: v for k, v in r.items() if k not in ("seconds", "timing", "git_sha")},
                      sort_keys=True)


def _kw(arm):
    return {"cfg": DQNConfig(batch_size=8)} if arm == "dqn" else None


def _replay(d):
    return sorted(p.name for p in (d / "replay").iterdir())


def _stored(d):
    return sum(load_episode(p).length for p in (d / "replay").glob("ep_*.npz"))


@pytest.mark.parametrize("arm", ["dreamer", "dqn"])
def test_resume_is_bit_identical(tmp_path, arm):
    full = _run(tmp_path / "full", arm, kw=_kw(arm))
    part = _run(tmp_path / "resumed", arm, stop=20, kw=_kw(arm))
    assert part["complete"] is False
    resumed = _run(tmp_path / "resumed", arm, kw=_kw(arm))
    assert _strip(full) == _strip(resumed)
    a = torch.load(tmp_path / "full" / "state.pt", weights_only=False)["arm"]["world_model" if arm == "dreamer" else "q"]
    b = torch.load(tmp_path / "resumed" / "state.pt", weights_only=False)["arm"]["world_model" if arm == "dreamer" else "q"]
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("arm", ["dreamer", "dqn"])
def test_resume_after_a_crash_between_checkpoints(tmp_path, arm):
    full = _run(tmp_path / "full", arm, kw=_kw(arm))
    d = tmp_path / "crash"
    _run(d, arm, stop=20, kw=_kw(arm))
    shutil.copy(d / "state.pt", tmp_path / "state20.pt")
    shutil.copy(d / "record.json", tmp_path / "record20.json")
    _run(d, arm, stop=40, kw=_kw(arm))              # the process that then dies
    shutil.copy(tmp_path / "state20.pt", d / "state.pt")
    shutil.copy(tmp_path / "record20.json", d / "record.json")
    _run(d, arm, stop=40, kw=_kw(arm))
    resumed = _run(d, arm, kw=_kw(arm))
    assert _strip(full) == _strip(resumed)
    assert _stored(d) == 60
    assert _replay(d) == _replay(tmp_path / "full")


def test_crash_during_prefill_leaves_no_trace(tmp_path):
    full = _run(tmp_path / "full")
    d = tmp_path / "crash"
    shutil.copytree(tmp_path / "full" / "replay", d / "replay")   # partial replay, no state.pt
    (d / "replay" / "ep_000000_len00010.features_dummy.npy").write_bytes(b"x")
    resumed = _run(d)
    assert _strip(full) == _strip(resumed)
    assert _stored(d) == 60
    assert _replay(d) == _replay(tmp_path / "full")


def test_state_is_the_source_of_truth_over_record_json(tmp_path):
    _run(tmp_path, stop=20)
    (tmp_path / "record.json").write_text("{}")      # out of step with state.pt
    full = _run(tmp_path / "ref")
    assert _strip(_run(tmp_path)) == _strip(full)
    assert load_record(tmp_path / "record.json")["complete"] is True


@pytest.mark.parametrize("change", [{"seed": 1}, {"cfg": OnlineConfig(
    budget=60, prefill=10, act_steps=10, rounds_per_act=2, eval_every=30,
    eval_seeds=TINY.eval_seeds)}, {"arm_kwargs": {"batch_size": 4, "seq_len": 8}}])
def test_resume_refuses_a_different_run(tmp_path, change):
    _run(tmp_path, stop=20)
    call = dict(arm_name="dreamer", seed=0, out_dir=tmp_path, env_factory=FakeEnv,
                encode=fake_encode, device=CPU, cfg=TINY, arm_kwargs=DREAMER_KW)
    call.update(change)
    n = _replay(tmp_path)
    with pytest.raises(RuntimeError, match="differs"):
        train_online(**call)
    assert _replay(tmp_path) == n


def _call(tmp_path, **change):
    call = dict(arm_name="dreamer", seed=0, out_dir=tmp_path, env_factory=FakeEnv,
                encode=fake_encode, device=CPU, cfg=TINY, arm_kwargs=DREAMER_KW)
    call.update(change)
    return call


def test_resume_refuses_a_different_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(online, "git_sha", lambda: "commit-a")
    _run(tmp_path, stop=20)
    monkeypatch.setattr(online, "git_sha", lambda: "commit-b")
    n = _replay(tmp_path)
    with pytest.raises(RuntimeError, match="differs"):
        train_online(**_call(tmp_path))
    assert _replay(tmp_path) == n


def test_resume_at_the_same_commit_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(online, "git_sha", lambda: "commit-a")
    _run(tmp_path, stop=20)
    assert _run(tmp_path)["complete"] is True


def test_resume_refuses_a_different_device_type(tmp_path):
    _run(tmp_path, stop=20)
    n = _replay(tmp_path)
    with pytest.raises(RuntimeError, match="differs"):
        train_online(**_call(tmp_path, device=torch.device("mps")))
    assert _replay(tmp_path) == n


def test_meta_pins_the_commit_and_the_device_type(monkeypatch):
    monkeypatch.setattr(online, "git_sha", lambda: "commit-a")
    meta = online._meta("dreamer", 0, TINY, {}, CPU)
    assert meta["git_sha"] == "commit-a" and meta["device"] == "cpu"
    assert online._meta("dreamer", 0, TINY, {}, torch.device("cuda"))["device"] == "cuda"


def test_record_git_sha_is_set_once_and_never_overwritten(tmp_path, monkeypatch):
    shas = iter(f"sha{i}" for i in range(100))
    monkeypatch.setattr(online, "git_sha", lambda: next(shas))
    rec = _run(tmp_path)
    assert len(rec["evals"]) == 4                       # four checkpoints were written
    assert rec["git_sha"] == "sha0"                     # the one _meta read, not a later HEAD
    assert load_record(tmp_path / "record.json")["git_sha"] == "sha0"


def test_unknown_device_type_is_refused():
    meta = torch.device("meta")
    with pytest.raises(ValueError, match="meta"):
        online._rng_state(meta)
    with pytest.raises(ValueError, match="meta"):
        online._set_rng_state({"python": None}, meta)


def test_cuda_rng_is_saved_and_restored(monkeypatch):
    cuda = torch.device("cuda")
    seen = {}
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: ["gpu0", "gpu1"])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda s: seen.setdefault("set", s))
    state = online._rng_state(cuda)
    assert state["cuda"] == ["gpu0", "gpu1"]
    online._set_rng_state(state, cuda)
    assert seen["set"] == ["gpu0", "gpu1"]


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


def test_verify_accepts_a_complete_run(tmp_path):
    rec = _run(tmp_path)
    assert verify_record(rec, TINY) == []


def test_delete_refuses_incomplete_record(tmp_path):
    _run(tmp_path, stop=20)
    with pytest.raises(ValueError, match="complete"):
        delete_replay_features(tmp_path, TINY)
    assert list((tmp_path / "replay").glob("*.features*.npy"))
    assert (tmp_path / "state.pt").exists()


def test_delete_removes_features_and_state_only(tmp_path):
    _run(tmp_path)
    n_npz = len(list((tmp_path / "replay").glob("ep_*.npz")))
    deleted = delete_replay_features(tmp_path, TINY)
    assert deleted == n_npz + 1                       # every feature file, plus state.pt
    assert not list((tmp_path / "replay").glob("*.features*.npy"))
    assert not (tmp_path / "state.pt").exists()
    assert len(list((tmp_path / "replay").glob("ep_*.npz"))) == n_npz
    assert load_record(tmp_path / "record.json")["replay_features_deleted"] is True


def test_delete_flags_the_record_and_drops_state_before_any_feature(tmp_path, monkeypatch):
    _run(tmp_path)
    real_unlink = Path.unlink

    def failing_unlink(self, *a, **k):
        if ".features" in self.name:
            raise OSError("simulated crash")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(OSError, match="simulated"):
        delete_replay_features(tmp_path, TINY)
    monkeypatch.undo()
    assert load_record(tmp_path / "record.json")["replay_features_deleted"] is True
    assert not (tmp_path / "state.pt").exists()
    assert list((tmp_path / "replay").glob("*.features*.npy"))
    delete_replay_features(tmp_path, TINY)            # idempotent: finishes the job
    assert not list((tmp_path / "replay").glob("*.features*.npy"))


def test_verify_flags_a_reordered_eval(tmp_path):
    rec = _run(tmp_path)
    rec["evals"][1]["modes"]["sample"] = rec["evals"][1]["modes"]["sample"][::-1]
    assert any("seeds" in p for p in verify_record(rec, TINY))


def test_delete_keeps_a_non_finite_value_in_the_record(tmp_path):
    rec = _run(tmp_path)
    rec["evals"][0]["diagnostics"]["probe"] = float("nan")
    write_record(tmp_path / "record.json", rec)
    delete_replay_features(tmp_path, TINY)
    out = load_record(tmp_path / "record.json")
    assert np.isnan(out["evals"][0]["diagnostics"]["probe"])


@pytest.mark.parametrize("kw", [
    dict(budget=3000, prefill=1000, act_steps=100, eval_every=1000),
    dict(budget=1000, prefill=1000, act_steps=100, eval_every=1000),
])
def test_prefill_must_precede_the_first_checkpoint(kw):
    with pytest.raises(ValueError, match="step-0"):
        OnlineConfig(**kw)


def test_default_and_tiny_configs_construct():
    assert OnlineConfig().prefill < OnlineConfig().eval_every
    assert TINY.prefill < TINY.eval_every
