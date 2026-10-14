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


def _fake_cell(root: Path, arm: str, seed: int, ret: float, *, control_ret: float = -100.0,
               isolation: bool = True, swap_in: str | None = None, drop_final: bool = False):
    """One cell's three records. `live["seeds"]` and every episode's `seed` agree unless
    `swap_in` names a list ("random", "untrained", "sample", "argmax") whose first two
    episodes are swapped; `control_ret` is the untrained control's return (random scores
    -100, so -100 is a control that does nothing); `drop_final` omits rung "10000"."""
    entry = root / "m4c_entry"; entry.mkdir(parents=True, exist_ok=True)
    iv = {"point": 0.1, "ci_low": 0.05, "ci_high": 0.2, "n_rows": 1, "n_episodes": 1}
    (entry / f"entry_{arm}_seed{seed}.json").write_text(json.dumps({
        "arm": arm, "seed": seed, "verdict": "ENTRY_PASS",
        "imagined": {"return": iv, "continue": iv}, "posterior": {"return": iv, "continue": iv}}))
    agent = root / "m4c_agent" / f"{arm}_seed{seed}"; agent.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2, 3]

    def episodes(name: str, r: float) -> list[dict]:
        out = [{"seed": s, "return": r, "length": 10, "killed": True,
                "truncated": False, "imagined_return": r} for s in seeds]
        if swap_in == name:
            out[0], out[1] = out[1], out[0]
        return out

    rungs = {} if drop_final else {"10000": {"sample": episodes("sample", ret),
                                             "argmax": episodes("argmax", ret)}}
    (agent / f"live_{arm}_seed{seed}.json").write_text(json.dumps({
        "arm": arm, "seed": seed, "seeds": seeds, "random": episodes("random", -100.0),
        "untrained": episodes("untrained", control_ret), "rungs": rungs}))
    (agent / f"train_{arm}_seed{seed}.json").write_text(json.dumps({"isolation_ok": isolation}))


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


def _read(root: Path):
    return subprocess.run([sys.executable, str(SCRIPT), "--phase", "read", "--runs", str(root)],
                          capture_output=True, text=True)


@pytest.mark.parametrize("overrides, reason", [
    ({"control_ret": 0.0}, "control"),
    ({"isolation": False}, "isolation"),
    ({"swap_in": "random"}, "seed"),
    ({"swap_in": "untrained"}, "seed"),
    ({"swap_in": "sample"}, "seed"),
    ({"swap_in": "argmax"}, "seed"),
    ({"drop_final": True}, "10000"),
])
def test_read_refuses_an_invalid_cell(tmp_path, overrides, reason):
    for arm, base in (("frozen_ssl", 50.0), ("pixel_ae", 0.0), ("random_vit", 0.5)):
        for seed in (0, 1, 2):
            bad = overrides if (arm, seed) == ("pixel_ae", 1) else {}
            _fake_cell(tmp_path, arm, seed, base + seed, **bad)
    p = _read(tmp_path)
    assert p.returncode == 81, p.stdout + p.stderr
    invalid = [ln for ln in p.stdout.splitlines() if ln.startswith("INVALID_CELLS:")]
    assert len(invalid) == 1 and "pixel_ae/seed1" in invalid[0] and reason in invalid[0], p.stdout
    assert "frozen_ssl/seed0" not in invalid[0]
    assert "cross-arm reading" not in p.stdout and " vs " not in p.stdout
    assert "cross-arm reading" not in (tmp_path / "m4c_study" / "study.txt").read_text()


def test_missing_cells_take_precedence_over_invalid_ones(tmp_path):
    _fake_cell(tmp_path, "frozen_ssl", 0, 50.0, isolation=False)
    p = _read(tmp_path)
    assert p.returncode == 80, p.stdout + p.stderr
