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
from mbfps.eval.split_gap import survival_indicator
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


# ---------------------------------------------------------------------------
# The gate, the identity check and the retrain (Task 7). The sweep's own
# record decides whether fifteen hours are spent; these tests drive that
# decision by fabricating the reading rather than by finding a cell that
# happens to clear, so the gate is tested and not the fixture's luck.
# ---------------------------------------------------------------------------

from mbfps.eval.sharper import IDENTITY_STEPS, SweepStatus  # noqa: E402

RETRAIN = "retrain_random_vit_seed1.json"
CHECKPOINT = "world_model_random_vit_seed1.pt"
RETRAIN_TAU = 0.5
RETRAIN_KEYS = {
    "arm", "seed", "tau", "steps", "seq_len", "history", "seconds", "identity",
    "device", "torch_version", "git_sha", "reference_git_sha", "nonfinite",
}


def _reading(status: SweepStatus, tau=None):
    """A `reading_noise` stand-in with a chosen status and tau*.

    It carries a full cell table, as the real reading does: the gate is what
    these tests drive, and a fake missing the table would only prove the
    formatter tolerates one -- which it should not have to."""
    from mbfps.eval.sharper import ArmTau, SweepReading

    def fake(inputs):
        cells = {
            (arm, tau_value): ArmTau(
                arm=arm, tau=tau_value, clears_up=False, clears_down=False,
                seeds_up=0, seeds_down=0, seeds_total=3,
            )
            for arm, sweep_arm in inputs.arms.items()
            for tau_value in sweep_arm.taus
        }
        return SweepReading(
            cells=cells, arms_clearing={}, arms_against={}, tau_star=tau,
            status=status, rule="fabricated", h=inputs.h, z_fam=inputs.z_fam,
        )

    return fake


def _never_train(*args, **kwargs):
    raise AssertionError("train_world_model ran; this refusal must come before any training")


@pytest.fixture
def trained(swept, monkeypatch, capsys):
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    assert _run(swept, "--phase", "train", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    return swept


def test_train_without_a_sweep_record_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "train") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase sweep first" in capsys.readouterr().out


def test_the_real_reading_is_what_the_gate_consults(swept, monkeypatch, capsys):
    """Every other gate test fabricates the reading to drive the decision.
    This one does not: the fixture is one arm and one seed, so the real
    `reading_noise` cannot reach NOISE_LIMITED (it needs two arms and two
    seeds), and the refusal must come from it rather than from a stand-in.
    Without this, a swapped treatment/control in `_tau_inputs` would invert
    the gate and every fabricated test would still pass."""
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(swept, "--phase", "train") == script.EXIT_NOT_NOISE_LIMITED
    out = capsys.readouterr().out
    assert "NOT NOISE LIMITED" in out
    assert "fabricated" not in out
    assert "arms required" in out or "of 1 arms" in out or "of 3 arms" in out


def test_the_glue_pairs_a_temperature_against_the_reference_not_the_other_way(swept):
    """The contrast's orientation is the whole reading: `S_free(h)` at tau
    MINUS at the reference. Swapped, `SHARPER_WORSE` would read as
    `NOISE_LIMITED` and spend fifteen hours on it.

    Taken through `_tau_inputs` itself, not through a reconstruction of what
    it does: a first version of this test called `_series`/`_paired` directly
    and chose its own treatment and control, so a swap INSIDE `_tau_inputs`
    left the test and the bug each self-consistent and the suite green.
    """
    import types as _types

    status, records = script.load_sweep(
        _types.SimpleNamespace(sweep_out=swept.sweep), [(JOB.arm, JOB.seed)]
    )
    assert status == script.EXIT_OK
    record = records[(JOB.arm, JOB.seed)]
    h = int(record["decision_h"])
    inputs = script._tau_inputs(records, JOB.arm, [JOB.seed], 0.0, h)

    # The same difference, computed straight off the record without touching
    # the glue: the survival indicator at tau = 0 minus the one at tau = 1,
    # over the windows both temperatures changed.
    def survival(tau):
        return survival_indicator(record["entries"][script.tau_key(tau)]["summary"], "free", h)

    values, changed = survival(0.0)
    reference_values, reference_changed = survival(REFERENCE_TAU)
    keep = changed & reference_changed
    direct = float(np.mean(values[keep] - reference_values[keep]))

    assert inputs.free.estimate == pytest.approx(direct, abs=1e-12), (
        inputs.free.estimate, direct
    )
    assert inputs.per_seed[JOB.seed].estimate == pytest.approx(direct, abs=1e-12)


def test_a_sweep_that_is_not_noise_limited_refuses_to_retrain(swept, monkeypatch, capsys):
    """THE gate: fifteen hours are not spent on a refuted premise, and the
    refusal is the script's, not a person's reading of a table."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOT_NOISE_LIMITED))
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(swept, "--phase", "train") == script.EXIT_NOT_NOISE_LIMITED
    out = capsys.readouterr().out
    assert "NOT NOISE LIMITED" in out and "fabricated" in out
    assert not (swept.sharper / CHECKPOINT).exists()


def test_a_sharper_worse_sweep_also_refuses(swept, monkeypatch, capsys):
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.SHARPER_WORSE))
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(swept, "--phase", "train") == script.EXIT_NOT_NOISE_LIMITED
    assert "SHARPER WORSE" in capsys.readouterr().out


def test_a_broken_identity_check_refuses_before_any_cell_is_retrained(swept, monkeypatch, capsys):
    """The tau = 1.0 retrain must reproduce the M3c loss prefix exactly -- it
    is what makes the shipped cells the control arm without spending fifteen
    hours on one. Doctoring the reference history breaks it."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    _doctor(swept.out / RECORD, lambda r: r["history"]["loss"].__setitem__(0, r["history"]["loss"][0] + 1e-3))
    assert _run(swept, "--phase", "train", "--steps", "4") == script.EXIT_IDENTITY_CHECK_FAILED
    out = capsys.readouterr().out
    assert "IDENTITY CHECK FAILED" in out and "step 1" in out
    assert not (swept.sharper / CHECKPOINT).exists()


def test_the_identity_check_is_the_shipped_temperature_and_its_steps_are_pinned(swept, monkeypatch, capsys):
    """It is a tau = 1.0 retrain, not a tau* one: it asks whether the edit is a
    no-op where it must be."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    seen = []
    real = script.train_world_model

    def recording(cfg, *args, **kwargs):
        seen.append(cfg.train.sample_temperature)
        return real(cfg, *args, **kwargs)

    monkeypatch.setattr(script, "train_world_model", recording)
    assert _run(swept, "--phase", "train", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    assert seen[0] == REFERENCE_TAU, "the identity check runs first, at the shipped temperature"
    assert seen[1:] == [RETRAIN_TAU], "then every requested cell at tau*"
    assert IDENTITY_STEPS == 500


def test_the_retrain_writes_a_checkpoint_carrying_its_temperature_and_a_record(trained):
    import torch

    payload = torch.load(trained.sharper / CHECKPOINT, weights_only=True)
    assert payload["sample_temperature"] == RETRAIN_TAU
    assert (payload["arm"], payload["seed"]) == (JOB.arm, JOB.seed)
    record = load_record(trained.sharper / RETRAIN)
    assert set(record) == RETRAIN_KEYS
    assert record["tau"] == RETRAIN_TAU and record["steps"] == 4
    assert record["identity"]["max_delta"] == 0.0 and record["identity"]["first_step"] is None
    # Self-describing and measured on THIS cell: a 0.0 taken on another cell
    # would be a number that means nothing where one that means something goes.
    assert (record["identity"]["arm"], record["identity"]["seed"]) == (JOB.arm, JOB.seed)
    assert record["identity"]["tau"] == REFERENCE_TAU
    assert record["reference_git_sha"] == load_record(trained.out / RECORD)["git_sha"]
    assert len(record["history"]["loss"]) == 4


def test_evaluate_scores_the_retrained_cell_at_its_own_temperature(trained, capsys):
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    study = load_record(trained.sharper / RECORD)
    assert study["steps"] == 4
    trust_record = load_record(trained.sharper / "trust_random_vit_seed1.json")
    assert trust_record["windows"]["total"] == WINDOWS
    assert np.asarray(trust_record["crossing"]["free"], dtype=float).shape == (WINDOWS,)
    # A retrained cell has no earlier pass to reproduce, so its record must not
    # carry a self-check at all: a 0.0 computed against itself is
    # indistinguishable from a sweep record's 0.0, which means a real bitwise
    # reproduction of the pass the gate scored.
    assert trust_record["self_check"] is None


def test_evaluating_a_checkpoint_at_the_wrong_temperature_is_exit_36(trained, monkeypatch, capsys):
    """The payload says what sampler produced these weights; evaluating them
    with another one would report a model that never existed."""
    import torch

    path = trained.sharper / CHECKPOINT
    payload = torch.load(path, weights_only=True)
    payload["sample_temperature"] = 0.25
    torch.save(payload, path)
    before = (trained.sharper / RECORD).read_text() if (trained.sharper / RECORD).exists() else None
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_TEMPERATURE_MISMATCH
    assert "TEMPERATURE MISMATCH" in capsys.readouterr().out
    # Refused BEFORE `evaluate_job` writes anything: a study record from the
    # wrong sampler describes a model that never existed, and writing it would
    # also overwrite whatever valid record was there.
    after = (trained.sharper / RECORD).read_text() if (trained.sharper / RECORD).exists() else None
    assert after == before


def test_evaluate_without_a_retrain_record_is_exit_11(swept, capsys):
    assert _run(swept, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase train first" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# read (Task 8): the sweep's table and verdict always; the retrain's, when a
# retrain exists. stages.txt's discipline -- what is printed is what is
# written, byte for byte.
# ---------------------------------------------------------------------------

SWEEP_SECTIONS = (
    "--- self-check per cell",
    "--- the gate at every temperature",
    "--- the noise reference against the imagined displacement",
    "--- survival by temperature",
    "--- Reading N: is the rollout noise-limited at h=3",
)


@pytest.fixture
def swept_read(swept, capsys):
    assert _run(swept, "--phase", "read") == script.EXIT_OK
    text = capsys.readouterr().out
    return types.SimpleNamespace(ref=swept, text=text, path=swept.sweep / "sweep.txt")


def test_read_without_a_sweep_record_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "read") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase sweep first" in capsys.readouterr().out


def test_read_refuses_a_sweep_record_whose_self_check_is_not_exact(swept, capsys):
    _doctor(swept.sweep / SWEEP, lambda r: r["self_check"].update({"reference_position_max_delta": 1e-6}))
    assert _run(swept, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (swept.sweep / "sweep.txt").exists()


def test_read_prints_the_sweep_sections_and_writes_them_byte_identical(swept_read):
    for section in SWEEP_SECTIONS:
        assert section in swept_read.text, section
    assert swept_read.path.read_text() == swept_read.text
    assert "z_fam = cluster_threshold(9, 6)" in swept_read.text
    assert "verdict:" in swept_read.text
    assert f"{REFERENCE_TAU:.1f}" in swept_read.text
    assert "NOTE: decision horizon clamped to the run's horizon h=3" in swept_read.text


def test_the_sweep_read_alone_writes_no_retrain_text(swept_read):
    assert not (swept_read.ref.sharper / "sharper.txt").exists()
    assert "Reading M" not in swept_read.text


def test_read_after_a_retrain_adds_reading_m_against_the_reference_cells(trained, capsys, monkeypatch):
    """With both phases on disk the read pairs the retrained cells against the
    M3c ones on the same windows and prints Reading M beside Reading N."""
    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()
    assert _run(trained, "--phase", "read", "--steps", "4") == script.EXIT_OK
    text = capsys.readouterr().out
    assert "--- Reading M: the retrain at tau=0.5" in text
    assert (trained.sharper / "sharper.txt").read_text() in text
    assert "verdict: random_vit" in text


def test_reading_m_is_the_retrain_minus_the_m3c_cells_not_the_other_way(trained, capsys, monkeypatch):
    """Reading M's orientation is the study's central claim: `retrain - M3c`.
    Swapped, "sharper is worse" reads as "sharper is better" and the
    milestone reports the opposite of what it measured. Pinned through
    `retrain_inputs` itself against a difference computed straight off the
    two records, as Reading N's orientation is pinned through `_tau_inputs`.
    """
    import types as _types

    monkeypatch.setattr(script, "reading_noise", _reading(SweepStatus.NOISE_LIMITED, RETRAIN_TAU))
    assert _run(trained, "--phase", "evaluate", "--steps", "4") == script.EXIT_OK
    capsys.readouterr()

    status, records = script.load_sweep(
        _types.SimpleNamespace(sweep_out=trained.sweep), [(JOB.arm, JOB.seed)]
    )
    assert status == script.EXIT_OK
    h = int(records[(JOB.arm, JOB.seed)]["decision_h"])
    new_trust = script._load_trust(trained.sharper, [(JOB.arm, JOB.seed)])
    old_trust = script._reference_trust(records, [(JOB.arm, JOB.seed)])
    assert new_trust and old_trust

    inputs = script.retrain_inputs(
        new_trust, old_trust, arms=[JOB.arm], seeds=[JOB.seed], h=h, tau=RETRAIN_TAU,
    )

    def survival(source, channel):
        return survival_indicator(source[(JOB.arm, JOB.seed)], channel, h)

    for channel, contrast in (("free", inputs.arms[JOB.arm].free),
                              ("probe", inputs.arms[JOB.arm].probe)):
        values, changed = survival(new_trust, channel)
        control, control_changed = survival(old_trust, channel)
        keep = changed & control_changed
        direct = float(np.mean(values[keep] - control[keep]))
        assert contrast.estimate == pytest.approx(direct, abs=1e-12), (channel, contrast.estimate, direct)


def test_read_is_idempotent(swept_read, capsys):
    assert _run(swept_read.ref, "--phase", "read") == script.EXIT_OK
    assert capsys.readouterr().out == swept_read.text
