import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "m4_study.py"
sys.path.insert(0, str(SCRIPT.parent))
from m4_study import separation  # noqa: E402


def test_separation_requires_nine_of_nine():
    scores = {"a": [10, 11, 12], "b": [1, 2, 3], "c": [2, 11.5, 0]}
    out = separation(scores)
    assert out[("a", "b")] == "ABOVE"
    assert out[("a", "c")] == "NOT_SEPARATED"
    assert out[("b", "c")] == "NOT_SEPARATED"


def test_separation_below_and_ties():
    out = separation({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert out[("a", "b")] == "BELOW"
    tie = separation({"a": [1, 2, 4], "b": [4, 5, 6]})
    assert tie[("a", "b")] == "NOT_SEPARATED"     # 4 vs 4 is not a win


def _fake_cell(root: Path, arm: str, seed: int, ret: float):
    entry = root / "m4c_entry"; entry.mkdir(parents=True, exist_ok=True)
    iv = {"point": 0.1, "ci_low": 0.05, "ci_high": 0.2, "n_rows": 1, "n_episodes": 1}
    (entry / f"entry_{arm}_seed{seed}.json").write_text(json.dumps({
        "arm": arm, "seed": seed, "verdict": "ENTRY_PASS",
        "imagined": {"return": iv, "continue": iv}, "posterior": {"return": iv, "continue": iv}}))
    agent = root / "m4c_agent" / f"{arm}_seed{seed}"; agent.mkdir(parents=True, exist_ok=True)
    ep = lambda r: {"seed": 1, "return": r, "length": 10, "killed": True,
                    "truncated": False, "imagined_return": r}
    (agent / f"live_{arm}_seed{seed}.json").write_text(json.dumps({
        "arm": arm, "seed": seed, "random": [ep(-100.0)] * 4, "untrained": [ep(-100.0)] * 4,
        "rungs": {"10000": {"sample": [ep(ret)] * 4, "argmax": [ep(ret)] * 4}}}))
    (agent / f"train_{arm}_seed{seed}.json").write_text(json.dumps({"isolation_ok": True}))


def test_pixel_ae_live_command_names_the_basic_autoencoder(tmp_path):
    from m4_study import PIXEL_AE_BASIC, _commands
    live = _commands(tmp_path, tmp_path / "data", "pixel_ae", 0)[-1][1]
    assert live[-2:] == ["--pixel-ae-checkpoint", str(PIXEL_AE_BASIC)]
    vit = _commands(tmp_path, tmp_path / "data", "random_vit", 0)[-1][1]
    assert "--pixel-ae-checkpoint" not in vit


def test_read_reports_missing_cells(tmp_path):
    _fake_cell(tmp_path, "frozen_ssl", 0, 50.0)
    p = subprocess.run([sys.executable, str(SCRIPT), "--phase", "read", "--runs", str(tmp_path)],
                       capture_output=True, text=True)
    assert p.returncode == 80, p.stdout + p.stderr


def test_read_with_nine_cells(tmp_path):
    for arm, base in (("frozen_ssl", 50.0), ("pixel_ae", 0.0), ("random_vit", 0.5)):
        for seed in (0, 1, 2):
            _fake_cell(tmp_path, arm, seed, base + seed)
    p = subprocess.run([sys.executable, str(SCRIPT), "--phase", "read", "--runs", str(tmp_path)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    text = (tmp_path / "m4c_study" / "study.txt").read_text()
    assert "frozen_ssl vs pixel_ae: ABOVE" in text
    assert "pixel_ae vs random_vit: NOT_SEPARATED" in text
