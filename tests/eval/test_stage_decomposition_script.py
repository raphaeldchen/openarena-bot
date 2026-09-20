"""scripts/stage_decomposition.py: per cell the trust pass with its latents,
the self-check, the identity, the statistics and one record (Task 4); then
(Task 5) the pooling, the control rule, Reading S, the tables and stages.txt.

The script is loaded by path. The REFERENCE is a real tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- with the diagnostic
`diagnose_dynamics.main` writes, exactly as the ladder tests build theirs.
The CONTROL is a second study directory holding the same arm at seeds 1 and
2 trained for 2 steps: a rung directory is a study directory, and
`rung_cell` reads a checkpoint and a record and nothing else. Every refusal
below is exercised by doctoring one file or monkeypatching one name.

THE FIXTURE'S FACTS: six validation episodes, `window_starts(40, 2, 3)` cuts
eight windows per episode -- 48 windows over 6 clusters on every record;
the decision horizon clamps from 15 to the run's 3.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.diagnostics import LatentIdentityError
from mbfps.eval.stages import KL_FREE_BITS, SEEDS_REQUIRED
from mbfps.eval.study import StudyJob, load_record, run_job

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("stage_decomposition")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
CONTROL_JOBS = (StudyJob("random_vit", 1), StudyJob("random_vit", 2))
CONTROL_KW = dict(steps=2, seq_len=4, context=2, horizon=3, device="cpu")
WINDOWS, CLUSTERS = 48, 6
GROUPS, CLASSES = 32, 32
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
STAGES_RECORD = "stages_random_vit_seed1.json"
CONTROL_RECORDS = ("stages_control_random_vit_seed1.json", "stages_control_random_vit_seed2.json")

RECORD_KEYS = {
    "arm", "seed", "kind", "label", "source", "step", "record_git_sha", "context", "horizon",
    "decision_h", "split_seed", "device", "torch_version", "git_sha", "episodes", "windows",
    "latent", "self_check", "identity", "information", "accuracy", "open", "decode",
    "companions", "nonfinite",
}


@pytest.fixture
def reference(tmp_path, wide_buffer, capsys):
    """One real cell with its diagnostic, and a control directory holding the
    same arm at two seeds trained for two steps."""
    out = tmp_path / "reference"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    control = tmp_path / "control"
    for job in CONTROL_JOBS:
        run_job(job, wide_buffer, control, **CONTROL_KW)
    capsys.readouterr()
    return types.SimpleNamespace(
        out=out, control=control, data=wide_buffer.root, stages=tmp_path / "stages",
    )


def _argv(ref, *extra: str) -> list[str]:
    return [
        "--out", str(ref.stages), "--reference", str(ref.out), "--control", str(ref.control),
        "--data", str(ref.data), "--device", "cpu", "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--control-arm", CONTROL_JOBS[0].arm,
        "--control-seeds", *[str(job.seed) for job in CONTROL_JOBS], *extra,
    ]


def _run(ref, *extra: str) -> int:
    return script.main(_argv(ref, *extra))


@pytest.fixture
def evaluated(reference, capsys):
    assert _run(reference, "--phase", "evaluate") == script.EXIT_OK
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


def test_the_parser_defaults_and_the_pinned_control_are_the_specs():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3g_stages") and args.reference == Path("runs/m3_study_v2")
    assert args.control == script.CONTROL_DIR == Path("runs/m3f_ladder/step4000")
    assert args.control_arm == script.CONTROL_ARM == "frozen_ssl"
    assert tuple(args.control_seeds) == script.CONTROL_SEEDS == (1, 2)
    assert args.data == Path("data/my_way_home") and args.device == "mps"
    assert args.context is None and args.horizon is None
    assert args.arms == ["pixel_ae", "frozen_ssl", "random_vit"] and args.seeds == [0, 1, 2]
    assert args.phase == script.PHASES[-1] and args.figure is None
    assert script.CONTROL_LABEL == "control"


def test_out_equal_to_reference_or_control_is_argparses_own_usage_error(tmp_path):
    """A record written into a study directory is a study directory changed:
    judged at the parser, before any directory is read."""
    same = str(tmp_path / "study")
    with pytest.raises(SystemExit) as raised:
        script.main(["--out", same, "--reference", same])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        script.main(["--out", same, "--control", same])
    assert raised.value.code == 2


def test_fewer_than_two_distinct_control_seeds_is_a_usage_error():
    """With one seed no stage can pass (SEEDS_REQUIRED = 2), so the control
    would read ENCODE_FAILS vacuously and validate nothing."""
    for seeds in (["1"], ["1", "1"]):
        with pytest.raises(SystemExit) as raised:
            script.main(["--control-seeds", *seeds])
        assert raised.value.code == 2
    assert SEEDS_REQUIRED == 2


def test_the_reused_statuses_are_trust_horizons_and_34_is_new():
    assert script.EXIT_OK == trust.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == trust.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == trust.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == trust.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == trust.EXIT_SELF_CHECK_FAILED == 30
    assert script.EXIT_CONTROL_MISREAD == 34


# ---------------------------------------------------------------------------
# evaluate.
# ---------------------------------------------------------------------------


def test_a_missing_cell_or_control_is_exit_11_before_any_pass_runs(reference, monkeypatch, capsys):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate", "--seeds", "0") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert _run(reference, "--phase", "evaluate", "--control-seeds", "1", "3") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert not reference.stages.exists()


def test_evaluate_writes_one_record_per_cell_and_per_control_with_the_spec_shapes(evaluated):
    cell = load_record(evaluated.stages / STAGES_RECORD)
    controls = [load_record(evaluated.stages / name) for name in CONTROL_RECORDS]
    for record in (cell, *controls):
        assert set(record) == RECORD_KEYS
        assert record["context"] == CONTEXT and record["horizon"] == HORIZON
        assert record["decision_h"] == HORIZON  # 15 clamped to the run's horizon
        assert record["windows"]["total"] == WINDOWS and record["windows"]["clusters"] == CLUSTERS
        assert len(record["windows"]["episode"]) == WINDOWS
        assert record["latent"] == {"groups": GROUPS, "classes": CLASSES}
        assert record["identity"] is True
        assert record["git_sha"] == cell["git_sha"]
        assert np.asarray(record["information"]).shape == (WINDOWS,)
        assert (np.asarray(record["information"]) >= 0).all()
        for name in ("teacher", "persistence", "marginal"):
            values = np.asarray(record["accuracy"][name], dtype=float)
            assert values.shape == (WINDOWS,) and (0 <= values).all() and (values <= 1).all()
        for name in ("accuracy", "persistence", "marginal"):
            values = np.asarray(record["open"][name], dtype=float)
            assert values.shape == (WINDOWS, HORIZON) and (0 <= values).all() and (values <= 1).all()
        for name in ("persistence_distance", "distance_to_truth", "probe_persistence", "probe_model"):
            values = np.asarray(record["decode"][name], dtype=float)
            assert values.shape == (WINDOWS, HORIZON) and (values >= 0).all()
        assert np.asarray(record["decode"]["moved"]).shape == (WINDOWS, HORIZON)
        assert np.asarray(record["decode"]["moved"]).dtype == bool
        assert np.asarray(record["companions"]["entropy"]).shape == (GROUPS,)
        assert np.asarray(record["companions"]["marginal_classes"]).shape == (GROUPS,)
        assert np.asarray(record["companions"]["nll"]).shape == (WINDOWS,)
        assert np.asarray(record["companions"]["rendering_median"]).shape == (HORIZON,)
        assert np.asarray(record["companions"]["jitter_median"]).shape == (HORIZON,)
    assert cell["kind"] == "cell" and cell["label"] == JOB.arm and cell["step"] == JOB_KW["steps"]
    assert cell["self_check"]["ok"] is True
    assert cell["self_check"]["reference_position_max_delta"] == 0.0
    assert cell["source"] == str(evaluated.out)
    for record, job in zip(controls, CONTROL_JOBS):
        assert record["kind"] == "control" and record["label"] == script.CONTROL_LABEL
        assert record["arm"] == job.arm and record["seed"] == job.seed
        assert record["step"] == CONTROL_KW["steps"] and record["self_check"] is None
        assert record["source"] == str(evaluated.control)


def test_evaluate_is_deterministic_on_cpu(evaluated, capsys):
    """The same pass twice writes the same bytes: the latents are the rollout
    the gate scored, seeded, and nothing in the statistics draws."""
    before = {name: (evaluated.stages / name).read_text() for name in (STAGES_RECORD, *CONTROL_RECORDS)}
    assert _run(evaluated, "--phase", "evaluate") == script.EXIT_OK
    capsys.readouterr()
    for name, text in before.items():
        assert (evaluated.stages / name).read_text() == text, name


def test_the_step_and_the_source_tell_a_control_from_its_cell(evaluated):
    """The control at seed 1 is the SAME arm and seed as the cell, two steps
    old: what tells them apart in the records is the kind, the label, the
    step and the source -- never the file name alone."""
    cell = load_record(evaluated.stages / STAGES_RECORD)
    control = load_record(evaluated.stages / CONTROL_RECORDS[0])
    assert (cell["arm"], cell["seed"]) == (control["arm"], control["seed"])
    assert cell["step"] != control["step"] and cell["source"] != control["source"]
    assert cell["label"] != control["label"]


def test_a_split_that_is_not_the_records_is_exit_12_and_a_protocol_flag_is_exit_14(reference, monkeypatch, capsys):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate", "--context", "3") == script.EXIT_RECORD_MISMATCH
    assert "RECORD MISMATCH" in capsys.readouterr().out
    _doctor(reference.out / RECORD, lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]))
    assert _run(reference, "--phase", "evaluate") == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH" in capsys.readouterr().out


def test_a_record_whose_curve_no_longer_reproduces_is_exit_14_before_the_trust_pass(reference, monkeypatch, capsys):
    _doctor(reference.out / RECORD, lambda r: r["curves"]["rssm_position"].__setitem__(
        0, r["curves"]["rssm_position"][0] + 1e-3))
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate") == script.EXIT_RECORD_MISMATCH
    assert "no longer reproduces" in capsys.readouterr().out


def test_a_doctored_diagnostic_is_exit_30_and_writes_no_record(reference, capsys):
    _doctor(reference.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    assert _run(reference, "--phase", "evaluate") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (reference.stages / STAGES_RECORD).exists()


def test_a_broken_step_one_identity_is_exit_30_by_name_and_writes_no_record(reference, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise LatentIdentityError("window at start 0: the open-loop prior at step 1 is not the teacher-forced prior")

    monkeypatch.setattr(script, "reference_trajectories", broken)
    assert _run(reference, "--phase", "evaluate") == script.EXIT_SELF_CHECK_FAILED
    out = capsys.readouterr().out
    assert "SELF-CHECK FAILED" in out and "window at start 0" in out
    assert not (reference.stages / STAGES_RECORD).exists()


def test_a_control_whose_checkpoint_is_missing_is_exit_11_before_any_pass(reference, monkeypatch, capsys):
    (reference.control / "world_model_random_vit_seed2.pt").unlink()
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "seed 2: no checkpoint" in capsys.readouterr().out


def test_the_decode_inputs_and_the_marginal_classes_are_what_read_will_consume(evaluated):
    """The logits are not written, so the record is pinned on what it does
    carry for `read`: the two distances of each decode channel are finite
    `(n, H)` arrays whose difference is the margin `read` pools, and the
    marginal classes are valid class indices."""
    record = load_record(evaluated.stages / STAGES_RECORD)
    decode = {k: np.asarray(v, dtype=float) for k, v in record["decode"].items() if k != "moved"}
    free = decode["persistence_distance"] - decode["distance_to_truth"]
    assert free.shape == (WINDOWS, HORIZON)
    probe = decode["probe_persistence"] - decode["probe_model"]
    assert np.isfinite(probe).all()
    assert record["companions"]["marginal_classes"] == [int(c) for c in record["companions"]["marginal_classes"]]
    assert all(0 <= c < CLASSES for c in record["companions"]["marginal_classes"])


# ---------------------------------------------------------------------------
# read: the pooling glue, the control rule, Reading S, stages.txt.
# ---------------------------------------------------------------------------

from mbfps.eval.pooling import cluster_threshold  # noqa: E402
from mbfps.eval.split_gap import StratumContrast  # noqa: E402
from mbfps.eval.stages import (  # noqa: E402
    FAMILY,
    STAGES,
    ArmReading,
    StageResult,
    StagesReading,
    Status,
)

SECTIONS = (
    "--- self-check per record",
    "--- stage components per arm",
    "--- carry: open-loop accuracy against k",
    "--- decode: the paired margin",
    "--- companions (change no verdict)",
    "--- the known-answer control",
    "--- Reading S: the first failing stage at h=3",
    "--- per seed",
)


@pytest.fixture
def stages_run(evaluated, capsys):
    assert _run(evaluated, "--phase", "read") == script.EXIT_OK
    text = capsys.readouterr().out
    return types.SimpleNamespace(ref=evaluated, text=text, path=evaluated.stages / "stages.txt")


def _fabricated_reading(status: Status):
    """A `reading_stages` stand-in giving every arm one status, with a full
    stage table so the control text can still be formatted."""
    def fake(inputs):
        arms = {}
        for arm in inputs.arms:
            stages = {
                stage: StageResult(stage=stage, passes=True, wording="passes",
                                   rule=f"{stage} passes: fabricated", seeds_holding=2, seeds_total=2)
                for stage in STAGES
            }
            arms[arm] = ArmReading(arm=arm, status=status, rule="fabricated", stages=stages)
        return StagesReading(arms=arms, h=inputs.h, z_fam=inputs.z_fam)
    return fake


def _record(n=4, horizon=3, information=(0.5, 0.1, 0.3, 0.9)):
    """A fabricated stages record: four windows, three steps, hand-typed series."""
    return {
        "label": "arm", "seed": 0, "arm": "arm", "kind": "cell",
        "episodes": {"val": ["a", "b"]}, "horizon": horizon, "context": 2, "decision_h": horizon,
        "device": "cpu", "torch_version": "x",
        "windows": {"total": n, "episode": [0, 0, 1, 1], "clusters": 2},
        "information": list(information),
        "accuracy": {"teacher": [0.9, 0.8, 0.7, 0.6], "persistence": [0.5, 0.5, 0.5, 0.5],
                     "marginal": [0.4, 0.3, 0.2, 0.1]},
        "open": {"accuracy": [[1.0, 0.8, 0.6]] * n, "persistence": [[0.5, 0.4, 0.3]] * n,
                 "marginal": [[0.2, 0.2, 0.2]] * n},
        "decode": {"persistence_distance": [[3.0, 4.0, 5.0]] * n, "distance_to_truth": [[1.0, 2.0, 6.0]] * n,
                   "probe_persistence": [[30.0, 40.0, 50.0]] * n, "probe_model": [[10.0, 20.0, 60.0]] * n,
                   "moved": [[True, True, True], [True, True, False], [True, True, True], [False, False, False]]},
    }


def test_contrast_values_are_the_pre_registered_differences_under_their_masks():
    r = _record()
    values, changed = script.contrast_values(r, "encode", 3)
    np.testing.assert_allclose(values, np.array([0.5, 0.1, 0.3, 0.9]) - KL_FREE_BITS)
    assert changed.all() and changed.shape == (4,)
    np.testing.assert_allclose(script.contrast_values(r, "predict_persistence", 3)[0], [0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(script.contrast_values(r, "predict_marginal", 3)[0], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(script.contrast_values(r, "carry", 2)[0], [0.4] * 4)
    np.testing.assert_allclose(script.contrast_values(r, "carry_marginal", 3)[0], [0.4] * 4)
    values, changed = script.contrast_values(r, "decode", 3)
    np.testing.assert_allclose(values, [-1.0] * 4)
    np.testing.assert_array_equal(changed, [True, False, True, False])
    values, changed = script.contrast_values(r, "decode_probe", 1)
    np.testing.assert_allclose(values, [20.0] * 4)
    np.testing.assert_array_equal(changed, [True, True, True, False])
    assert script.CONTRASTS == ("encode", "predict_persistence", "predict_marginal", "carry", "decode")
    with pytest.raises(KeyError):
        script.contrast_values(r, "render", 1)


def test_pooled_is_pool_arm_reduced_to_a_contrast_and_nan_with_nothing_to_pool():
    r = _record()
    series = script._series(r, *script.contrast_values(r, "encode", 3), channel="encode")
    p = script.pooled([series])
    assert isinstance(p, StratumContrast)
    assert p.estimate == pytest.approx(np.mean([0.5, 0.1, 0.3, 0.9]) - KL_FREE_BITS)
    assert p.clusters == 2 and np.isfinite(p.se)
    unmoved = dict(r, decode=dict(r["decode"], moved=[[False] * 3] * 4))
    empty = script.pooled([script._series(unmoved, *script.contrast_values(unmoved, "decode", 3), channel="decode")])
    assert np.isnan(empty.z) and empty.clusters == 0
    assert np.isnan(script.pooled([]).z)


def test_stages_inputs_pool_each_arm_and_the_control_under_the_family_bar(stages_run):
    status, records = script.load_records(
        types.SimpleNamespace(out=stages_run.ref.stages),
        [(JOB.arm, JOB.seed)], [(job.arm, job.seed) for job in CONTROL_JOBS],
    )
    assert status == script.EXIT_OK
    assert set(records) == {(JOB.arm, JOB.seed), (script.CONTROL_LABEL, 1), (script.CONTROL_LABEL, 2)}
    inputs, control = script.stages_inputs(
        records, arms=[JOB.arm], seeds=[JOB.seed], control_seeds=[1, 2], h=HORIZON,
    )
    assert inputs.z_fam == control.z_fam == cluster_threshold(FAMILY, CLUSTERS)
    assert inputs.h == control.h == HORIZON
    assert list(inputs.arms) == [JOB.arm] and list(control.arms) == [script.CONTROL_LABEL]
    a = inputs.arms[JOB.arm]
    assert set(a.per_seed) == {JOB.seed} and a.per_seed[JOB.seed].per_seed is None
    assert set(control.arms[script.CONTROL_LABEL].per_seed) == {1, 2}
    for stage in STAGES:
        for _, contrast in a.contrasts(stage):
            assert contrast.clusters == CLUSTERS and np.isfinite(contrast.estimate)


def test_read_without_records_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "read") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase evaluate first" in capsys.readouterr().out


def test_read_refuses_a_record_whose_self_check_is_not_ok_or_whose_identity_is_unset(evaluated, capsys):
    path = evaluated.stages / STAGES_RECORD
    original = path.read_text()
    _doctor(path, lambda r: r["self_check"].update({"ok": False}))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    path.write_text(original)
    _doctor(path, lambda r: r["self_check"].update({"reference_position_max_delta": 1e-6}))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    capsys.readouterr()
    path.write_text(original)
    _doctor(evaluated.stages / CONTROL_RECORDS[1], lambda r: r.__setitem__("identity", False))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "identity" in capsys.readouterr().out
    assert not (evaluated.stages / "stages.txt").exists()


def test_read_raises_on_records_at_different_protocols(evaluated):
    _doctor(evaluated.stages / CONTROL_RECORDS[0], lambda r: r.__setitem__("horizon", 2))
    with pytest.raises(ValueError, match="horizon"):
        _run(evaluated, "--phase", "read")


def test_a_control_that_does_not_read_encode_fails_is_exit_34_with_no_reading_and_no_file(evaluated, monkeypatch, capsys):
    """The known-blind control is the instrument's validation: read as
    anything but ENCODE_FAILS, the control tables are printed with the
    sentence, no arm's reading is, and stages.txt is not written."""
    monkeypatch.setattr(script, "reading_stages", _fabricated_reading(Status.NO_STAGE_FAILS))
    assert _run(evaluated, "--phase", "read") == script.EXIT_CONTROL_MISREAD
    out = capsys.readouterr().out
    assert "CONTROL MISREAD" in out and "NO STAGE FAILS" in out and "fabricated" in out
    assert "--- the known-answer control" in out
    assert "--- Reading S" not in out.split("CONTROL MISREAD")[1]
    assert f"verdict: {JOB.arm}" not in out
    assert not (evaluated.stages / "stages.txt").exists()


def test_a_control_reading_encode_fails_lets_the_reading_through(evaluated, monkeypatch, capsys):
    monkeypatch.setattr(script, "reading_stages", _fabricated_reading(Status.ENCODE_FAILS))
    assert _run(evaluated, "--phase", "read") == script.EXIT_OK
    out = capsys.readouterr().out
    assert "CONTROL MISREAD" not in out and f"verdict: {JOB.arm}" in out
    assert (evaluated.stages / "stages.txt").exists()


def test_read_prints_every_section_writes_it_byte_identical_and_draws_the_figure(stages_run):
    text = stages_run.text
    for section in SECTIONS:
        assert section in text, section
    assert stages_run.path.read_text() == text
    assert "z_fam = cluster_threshold(5, 6)" in text
    assert f"verdict: {JOB.arm:<12}ENCODE FAILS" in text  # one seed: nothing can replicate
    assert "seeds holding" in text and "this seed alone" in text
    assert "control" in text and "must read ENCODE FAILS" in text
    assert "NOTE: decision horizon clamped to the run's horizon h=3" in text
    figure = stages_run.ref.stages / "stages_curves.png"
    assert (f"figure={figure}" in text) == figure.exists()
    assert "figure NOT written" in text or figure.stat().st_size > 0


def test_read_is_idempotent_and_all_runs_both_phases(stages_run, capsys):
    assert _run(stages_run.ref, "--phase", "read") == script.EXIT_OK
    assert capsys.readouterr().out == stages_run.text
    assert _run(stages_run.ref, "--phase", "all") == script.EXIT_OK
    out = capsys.readouterr().out
    assert out.endswith(stages_run.text) and "wrote" in out


def test_the_groups_order_the_arms_as_the_readings_do_and_put_the_control_last():
    records = {("random_vit", 0): {}, ("pixel_ae", 0): {}, ("control", 1): {}, ("control", 2): {}}
    groups = script._groups(records, ["random_vit", "pixel_ae"], [0], [1, 2])
    assert [label for label, _ in groups] == ["pixel_ae", "random_vit", "control"]
    assert groups[-1][1] == [("control", 1), ("control", 2)]
