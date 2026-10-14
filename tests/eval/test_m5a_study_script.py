import json
import subprocess
import sys
from pathlib import Path

import pytest

from mbfps.eval.study import write_record

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "m5a_study.py"
SEEDS = list(range(1_000_000, 1_000_100))
STEPS = list(range(0, 100_001, 10_000))


def _write(runs, arm, seed, level, step0=-160.0, isolation=True, budget=100_000, rec_seed=None):
    mode = "sample" if arm == "dreamer" else "greedy"
    eps = lambda r: [{"seed": s, "return": r + (s % 5)} for s in SEEDS]
    rec = {"arm": arm, "seed": seed if rec_seed is None else rec_seed, "gated_mode": mode, "complete": True,
           "budget_steps": budget, "seeds": SEEDS, "random": eps(-155.0),
           "isolation_ok": isolation if arm == "dreamer" else None,
           "update_rounds": 19_000, "seconds": 3600.0,
           "evals": [{"step": s, "modes": {mode: eps(step0 if s == 0 else level)}} for s in STEPS]}
    write_record(runs / f"{arm}_seed{seed}" / "record.json", rec)


def _run(*args):
    p = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_missing_runs_exit_90(tmp_path):
    _write(tmp_path, "dreamer", 0, 60.0)
    assert _run("--phase", "read", "--runs", tmp_path)[0] == 90


def test_ten_valid_runs_print_the_reading(tmp_path):
    for s in range(5):
        _write(tmp_path, "dreamer", s, 60.0 + s)
        _write(tmp_path, "dqn", s, 10.0 + s)
    code, out = _run("--phase", "read", "--runs", tmp_path)
    assert code == 0, out
    assert "DREAMER_MORE_EFFICIENT" in (tmp_path / "study.txt").read_text()


@pytest.mark.parametrize("bad", [{"step0": 95.0}, {"isolation": False}, {"budget": 99_900}])
def test_invalid_run_exits_91_without_the_reading(tmp_path, bad):
    for s in range(5):
        _write(tmp_path, "dreamer", s, 60.0 + s, **(bad if s == 2 else {}))
        _write(tmp_path, "dqn", s, 10.0 + s)
    code, out = _run("--phase", "read", "--runs", tmp_path)
    assert code == 91, out
    text = (tmp_path / "study.txt").read_text()
    assert "INVALID_RUNS" in text and "dreamer_seed2" in text
    assert "MORE_EFFICIENT" not in text and "NOT_SEPARATED" not in text


def test_a_record_that_is_not_its_directory_s_run_exits_91(tmp_path):
    for s in range(5):
        _write(tmp_path, "dreamer", s, 60.0 + s, rec_seed=0 if s == 2 else None)
        _write(tmp_path, "dqn", s, 10.0 + s)
    code, out = _run("--phase", "read", "--runs", tmp_path)
    assert code == 91, out
    assert "dreamer_seed2" in (tmp_path / "study.txt").read_text()


def test_run_skips_a_verified_run_whose_state_is_gone(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("m5a_study", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _write(tmp_path, "dreamer", 0, 60.0)               # verified, no state.pt, no flag
    launched = []
    monkeypatch.setattr(mod.subprocess, "call", lambda cmd: launched.append(cmd) or 0)
    assert mod.main(["--phase", "run", "--runs", str(tmp_path), "--arms", "dreamer",
                     "--seeds", "0"]) == 0
    assert launched == []
    (tmp_path / "dreamer_seed0" / "state.pt").write_bytes(b"x")   # state remains: resume
    assert mod.main(["--phase", "run", "--runs", str(tmp_path), "--arms", "dreamer",
                     "--seeds", "0"]) == 0
    assert len(launched) == 1
