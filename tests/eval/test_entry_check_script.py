import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.eval.study import load_record, write_record
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


def _train(tmp_path, length_of):
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 11):
        t = length_of(fill)
        buf.add(_basic_like(t, fill, kill_at=t - 1 if fill % 3 == 0 else None))
    cfg = get_config("cnn", steps=3, seq_len=16, batch_size=4, device="cpu",
                     n_actions=4, pad_short=True, reward_transform="symlog")
    train_world_model(cfg, buf, out_dir=tmp_path / "wm")
    return tmp_path, tmp_path / "wm" / "world_model_cnn_seed0.pt"


@pytest.fixture
def trained(tmp_path):
    return _train(tmp_path, lambda fill: 30 if fill % 2 else 12)


@pytest.fixture
def trained_too_short(tmp_path):
    # Every episode is below the 21 frames a context-5 + horizon-15 window needs.
    return _train(tmp_path, lambda fill: 12)


def _run(*args):
    proc = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_smoke_measure_writes_a_record_and_read_runs_end_to_end(trained):
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


def test_position_step_is_skipped_with_a_note_when_no_window_fits(trained_too_short):
    root, ckpt = trained_too_short
    out = root / "entry"
    # WITHOUT --no-position: the reported-only step must not crash the gate record.
    code, log = _run("--phase", "measure", "--checkpoint", ckpt, "--data", root / "data",
                     "--out", out, "--resamples", 50, "--device", "cpu")
    assert code == 0, log
    assert "position step skipped" in log
    record = load_record(out / "entry_cnn_seed0.json")
    assert record["position_gap_closed_h15"] is None
    assert isinstance(record["position_note"], str) and "21" in record["position_note"]
    assert "verdict" in record and record["counts"]["episodes"] >= 1


# ---- the read's exit-code mapping, with no model in the loop ---------------------------

def _iv(low):
    return {"point": low + 0.1, "ci_low": low, "ci_high": low + 0.2,
            "n_rows": 100, "n_episodes": 5}


def _readings(ret, cont):
    return {"return": _iv(ret), "continue": _iv(cont)}


def _record(imagined, posterior, role="cell"):
    record = {
        "arm": "cnn", "seed": 0, "role": role, "horizon": 15, "resamples": 50,
        "c_R": -1.0, "c_C": 0.1,
        "counts": {"episodes": 2, "starts": 20, "slots": 300, "kills": 1, "terminations": 1},
        "imagined": imagined, "posterior": posterior,
        "position_gap_closed_h15": None, "position_note": "skipped by --no-position",
    }
    if role == "cell":
        record["verdict"] = "STALE_AND_IGNORED"   # read must recompute, not trust this
    return record


PASS, FAIL = 0.2, -0.2


@pytest.mark.parametrize(
    "name,cell_imagined,cell_posterior,control_return_low,expected",
    [
        ("control_passes_so_broken_even_if_cell_passes",
         _readings(PASS, PASS), _readings(PASS, PASS), 0.05, 62),
        ("both_pass_control_fails",
         _readings(PASS, PASS), _readings(PASS, PASS), FAIL, 0),
        ("imagined_fails_posterior_fails_too",
         _readings(FAIL, PASS), _readings(FAIL, PASS), FAIL, 60),
        ("imagined_fails_posterior_passes",
         _readings(FAIL, PASS), _readings(PASS, PASS), FAIL, 61),
        ("continue_fails_posterior_continue_fails",
         _readings(PASS, FAIL), _readings(PASS, FAIL), FAIL, 60),
        ("continue_fails_posterior_continue_passes",
         _readings(PASS, FAIL), _readings(PASS, PASS), FAIL, 61),
        ("nan_control_is_not_broken",
         _readings(PASS, PASS), _readings(PASS, PASS), float("nan"), 0),
    ],
)
def test_read_maps_records_to_exit_codes(tmp_path, name, cell_imagined, cell_posterior,
                                         control_return_low, expected):
    out = tmp_path / "entry"
    write_record(out / "entry_cnn_seed0.json", _record(cell_imagined, cell_posterior))
    control_imagined = {"return": _iv(control_return_low), "continue": _iv(FAIL)}
    write_record(out / "entry_control.json",
                 _record(control_imagined, control_imagined, role="control"))
    code, log = _run("--phase", "read", "--out", out, "--arm", "cnn", "--seed", 0)
    assert code == expected, f"{name}: {log}"
    text = (out / "entry.txt").read_text()
    assert "skipped by --no-position" in text
    if expected == 62:
        assert "VERDICT: CONTROL_BROKEN" in text
