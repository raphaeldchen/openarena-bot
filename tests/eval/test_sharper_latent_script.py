"""scripts/sharper_latent.py: the rollout-temperature sweep over shipped cells
(Task 6), then the gated retrain and its reading (Tasks 7-8).

The script is loaded by path. The REFERENCE is a real tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- with the diagnostic
`diagnose_dynamics.main` writes, exactly as the ladder and stage tests build
theirs. The sweep runs a two-value grid so a test is seconds rather than
minutes; `main` takes `taus` as a parameter with `TAU_GRID` as its default, so
a test never patches the pre-registration.

THE FIXTURE'S FACTS: six validation episodes, `window_starts(40, 2, 3)` cuts
eight windows per episode -- 48 windows over 6 clusters at every temperature;
the decision horizon clamps from 15 to the run's 3.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.sharper import REFERENCE_TAU, TAU_GRID
from mbfps.eval.study import StudyJob, load_record, run_job

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("sharper_latent")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
SWEEP_TAUS = (1.0, 0.0)
WINDOWS, CLUSTERS = 48, 6
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
SWEEP = "sweep_random_vit_seed1.json"

SWEEP_KEYS = {
    "arm", "seed", "source", "step", "record_git_sha", "context", "horizon", "decision_h",
    "split_seed", "device", "torch_version", "git_sha", "taus", "episodes", "windows",
    "self_check", "entries", "nonfinite",
}
ENTRY_KEYS = {"tau", "gate", "probe", "noise", "summary"}


@pytest.fixture
def reference(tmp_path, wide_buffer, capsys):
    """One real cell and the diagnostic the ladder writes for it."""
    out = tmp_path / "reference"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    capsys.readouterr()
    return types.SimpleNamespace(
        out=out, data=wide_buffer.root, sweep=tmp_path / "sweep", sharper=tmp_path / "sharper",
    )


def _argv(ref, *extra: str) -> list[str]:
    return [
        "--out", str(ref.sharper), "--sweep-out", str(ref.sweep), "--reference", str(ref.out),
        "--data", str(ref.data), "--device", "cpu", "--arms", JOB.arm, "--seeds", str(JOB.seed),
        *extra,
    ]


def _run(ref, *extra: str) -> int:
    return script.main(_argv(ref, *extra), taus=SWEEP_TAUS)


@pytest.fixture
def swept(reference, capsys):
    assert _run(reference, "--phase", "sweep") == script.EXIT_OK
    capsys.readouterr()
    return reference


def _doctor(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_pass(*args, **kwargs):
    raise AssertionError("reference_trajectories ran; this refusal must come before any pass")


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_are_the_specs():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3h_sharper") and args.sweep_out == Path("runs/m3h_sweep")
    assert args.reference == Path("runs/m3_study_v2") and args.data == Path("data/my_way_home")
    assert args.device == "mps" and args.context is None and args.horizon is None
    assert args.arms == ["pixel_ae", "frozen_ssl", "random_vit"] and args.seeds == [0, 1, 2]
    assert args.phase == script.PHASES[-1] and args.figure is None


def test_out_equal_to_reference_or_to_the_sweep_is_argparses_own_usage_error(tmp_path):
    same = str(tmp_path / "study")
    for flag in ("--reference", "--sweep-out"):
        with pytest.raises(SystemExit) as raised:
            script.main(["--out", same, flag, same], taus=SWEEP_TAUS)
        assert raised.value.code == 2


def test_the_reused_statuses_are_trust_horizons_and_35_36_37_are_new():
    assert script.EXIT_OK == trust.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == trust.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == trust.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == trust.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == trust.EXIT_SELF_CHECK_FAILED == 30
    assert script.EXIT_NOT_NOISE_LIMITED == 35
    assert script.EXIT_TEMPERATURE_MISMATCH == 36
    assert script.EXIT_IDENTITY_CHECK_FAILED == 37


# ---------------------------------------------------------------------------
# sweep.
# ---------------------------------------------------------------------------


def test_a_missing_cell_is_exit_11_before_any_pass_runs(reference, monkeypatch, capsys):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "sweep", "--seeds", "0") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert not reference.sweep.exists()


def test_the_sweep_writes_one_record_per_cell_holding_every_temperature(swept):
    record = load_record(swept.sweep / SWEEP)
    assert set(record) == SWEEP_KEYS
    assert record["taus"] == list(SWEEP_TAUS)
    assert record["context"] == CONTEXT and record["horizon"] == HORIZON
    assert record["decision_h"] == HORIZON  # 15 clamped to the run's horizon
    assert record["windows"] == {
        "total": WINDOWS, "episode": record["windows"]["episode"], "clusters": CLUSTERS,
    }
    assert len(record["windows"]["episode"]) == WINDOWS
    assert record["step"] == JOB_KW["steps"] and record["source"] == str(swept.out)
    assert set(record["entries"]) == {script.tau_key(t) for t in SWEEP_TAUS}
    for key, entry in record["entries"].items():
        assert set(entry) == ENTRY_KEYS
        assert entry["tau"] == script.tau_value(key)
        assert set(entry["gate"]) == {"gap_final", "degenerate"}
        assert set(entry["summary"]) >= {"windows", "curves", "band", "crossing", "margin", "survival", "counts"}
        assert set(entry["probe"]) == {"selection_r2", "measurable"}
        assert np.asarray(entry["noise"]["curve"]).shape == (HORIZON,)
        crossing = np.asarray(entry["summary"]["crossing"]["free"], dtype=float)
        assert crossing.shape == (WINDOWS,)


def test_a_temperature_key_carries_no_dot_and_round_trips():
    """`write_record` refuses a key containing '.', since the non-finite map
    addresses fields by dotted path -- `split_gap.q_key` exists for the same
    reason. The value must still come back exactly, or a record cannot say
    which temperature an entry was scored at."""
    for tau in TAU_GRID:
        key = script.tau_key(tau)
        assert "." not in key, key
        assert script.tau_value(key) == tau
    assert script.tau_key(0.3) == script.tau_key(0.30) == "tau30"
    assert script.tau_key(1.0) == "tau100" and script.tau_key(0.0) == "tau0"


def test_a_grid_without_the_reference_temperature_is_refused_before_any_pass(reference, monkeypatch):
    """Every contrast is against the reference temperature and the self-check
    is taken on it, so a grid without it can produce no reading -- and finding
    that out after five passes would cost an hour on the real nine cells."""
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    with pytest.raises(ValueError, match="reference temperature"):
        script.main(_argv(reference, "--phase", "sweep"), taus=(0.7, 0.3))


def test_a_grid_whose_temperatures_share_a_record_key_is_refused(reference, monkeypatch):
    """`tau_key` spells hundredths, so 0.300 and 0.305 are one key: one entry
    would overwrite the other while `taus` listed both, and a reader would be
    handed a temperature no entry was scored at."""
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    with pytest.raises(ValueError, match="same record key"):
        script.main(_argv(reference, "--phase", "sweep"), taus=(1.0, 0.3, 0.305))


def test_the_reference_temperature_is_scored_first_whatever_order_the_grid_gives(
    reference, monkeypatch, capsys
):
    """The guarantee is the code's, not the grid's: the pre-registered grid
    happens to list 1.0 first, and a cell must not stop being protected if a
    caller passes another order."""
    seen = []
    real = script.reference_trajectories

    def recording(*args, **kwargs):
        seen.append(kwargs.get("rollout_temperature"))
        return real(*args, **kwargs)

    monkeypatch.setattr(script, "reference_trajectories", recording)
    assert script.main(_argv(reference, "--phase", "sweep"), taus=(0.0, 1.0)) == script.EXIT_OK
    capsys.readouterr()
    assert seen[0] == REFERENCE_TAU, seen


def test_the_reference_temperature_reproduces_the_cells_diagnostic_bitwise(swept):
    """THE anchor: at tau = 1.0 the pass must be the one the gate scored, or
    no temperature below it means anything. Exact, not close."""
    record = load_record(swept.sweep / SWEEP)
    assert record["self_check"]["ok"] is True
    assert record["self_check"]["reference_position_max_delta"] == 0.0
    assert record["self_check"]["persistence_position_max_delta"] == 0.0


def test_a_sharper_temperature_moves_the_rollout_and_not_the_probe_or_the_windows(swept):
    """The sweep's whole premise, on real records: the crossings differ between
    temperatures while the probe, the window identity and the truth-derived
    masks are shared -- they are computed before the rollout or from the truth."""
    record = load_record(swept.sweep / SWEEP)
    warm = record["entries"][script.tau_key(REFERENCE_TAU)]
    sharp = record["entries"][script.tau_key(0.0)]
    # The R^2 is the cell's and cannot differ; what CAN differ is
    # measurability, which is judged on this temperature's own band.
    assert warm["probe"]["selection_r2"] == sharp["probe"]["selection_r2"]
    assert isinstance(warm["probe"]["measurable"], bool)
    assert isinstance(sharp["probe"]["measurable"], bool)
    assert warm["summary"]["counts"]["not_moved"] == sharp["summary"]["counts"]["not_moved"]
    assert warm["summary"]["crossing"]["free"] != sharp["summary"]["crossing"]["free"]
    assert sharp["noise"]["median"] == 0.0, "two draws at tau = 0 are one trajectory"
    assert warm["noise"]["median"] > 0.0, "at tau = 1 two draws of a sampling model differ"
    assert np.asarray(warm["noise"]["curve"]).shape == (HORIZON,)


def test_a_doctored_diagnostic_is_exit_30_and_writes_no_record(reference, capsys):
    _doctor(reference.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    assert _run(reference, "--phase", "sweep") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (reference.sweep / SWEEP).exists()


def test_the_self_check_is_judged_before_any_other_temperature_runs(reference, monkeypatch, capsys):
    """A cell whose reference pass does not reproduce must cost one pass, not
    five: the temperatures below 1.0 are never evaluated."""
    _doctor(reference.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    seen = []
    real = script.reference_trajectories

    def counting(*args, **kwargs):
        seen.append(kwargs.get("rollout_temperature"))
        return real(*args, **kwargs)

    monkeypatch.setattr(script, "reference_trajectories", counting)
    assert _run(reference, "--phase", "sweep") == script.EXIT_SELF_CHECK_FAILED
    capsys.readouterr()
    assert seen == [REFERENCE_TAU], seen


def test_a_split_that_is_not_the_records_is_exit_12_and_a_protocol_flag_is_exit_14(
    reference, monkeypatch, capsys
):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "sweep", "--context", "3") == script.EXIT_RECORD_MISMATCH
    assert "RECORD MISMATCH" in capsys.readouterr().out
    _doctor(reference.out / RECORD, lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]))
    assert _run(reference, "--phase", "sweep") == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH" in capsys.readouterr().out


def test_the_sweep_is_deterministic_on_cpu(swept, capsys):
    before = (swept.sweep / SWEEP).read_text()
    assert _run(swept, "--phase", "sweep") == script.EXIT_OK
    capsys.readouterr()
    assert (swept.sweep / SWEEP).read_text() == before
