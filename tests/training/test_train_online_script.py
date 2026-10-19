"""scripts/train_online.py wires --scenario and --encoder through to the run."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_online.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_online_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("argv, scenario, encoder, backbone", [
    ([], "basic", "frozen_ssl", "dinov2"),
    (["--scenario", "defend_the_center", "--encoder", "random_vit"],
     "defend_the_center", "random_vit", "random_vit"),
])
def test_flags_reach_the_run(tmp_path, monkeypatch, argv, scenario, encoder, backbone):
    mod, seen = _load(), {}

    class _Extractor:
        def __init__(self, backbone, device):
            seen["backbone"] = backbone
            self.encode = lambda frames: frames

    def _train(arm, seed, out, env_factory, encode, device, cfg, arm_kwargs):
        env_factory()
        seen["arm_kwargs"] = arm_kwargs
        return {"complete": False}

    monkeypatch.setattr("mbfps.data.features.FeatureExtractor", _Extractor)
    monkeypatch.setattr(mod, "make_env", lambda name, **kw: seen.setdefault("env", (name, kw)))
    monkeypatch.setattr(mod, "train_online", _train)
    rc = mod.main(["--arm", "dqn", "--seed", "0", "--out", str(tmp_path), "--device", "cpu", *argv])
    assert rc == 1                                  # the stub record does not verify
    assert seen["env"] == ("vizdoom", {"scenario": scenario, "frame_skip": 4})
    assert seen["backbone"] == backbone
    assert seen["arm_kwargs"] == {"encoder": encoder}
