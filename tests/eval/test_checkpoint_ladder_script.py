"""scripts/checkpoint_ladder.py: retrain a cell saving rungs, anchor it to the
reference run, evaluate every rung and the reference through the study's own
evaluation half and the trust pass, one ladder record per cell; then (Task 6)
the pooling, Reading T, the tables and ladder.txt.

The script is loaded by path. The REFERENCE is a real tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- with the diagnostic
`diagnose_dynamics.main` writes, exactly as the split-gap tests build theirs.
The LADDER retrains the same cell to `--steps 4` saving rungs (2, 4): on CPU
the first four losses reproduce the reference's first four bitwise, so
`--anchor hard` passes and every refusal below is exercised by doctoring one
file.

THE FIXTURE'S FACTS: six validation episodes, `window_starts(40, 2, 3)` cuts
eight windows per episode -- 48 windows over 6 clusters on every rung.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.eval.ladder import CURVE_WINDOW, OBJECTIVE_BATCHES, REFERENCE_RUNG, STEPS
from mbfps.eval.split_gap import CURVE_NAMES
from mbfps.eval.study import StudyJob, job_record_path, load_record, run_job
from mbfps.training.world_model import history_at

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("checkpoint_ladder")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
LADDER_RUNGS = (2, 4)
LADDER_STEPS = 4
WINDOWS, CLUSTERS = 48, 6
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
TRAIN = "train_random_vit_seed1.json"
LADDER = "ladder_random_vit_seed1.json"
CHECKPOINT = "world_model_random_vit_seed1.pt"

LADDER_KEYS = {
    "arm", "seed", "context", "horizon", "split_seed", "device", "torch_version", "git_sha",
    "reference_git_sha", "train_git_sha", "anchor", "rungs", "reference_rung", "primary_rung",
    "embedding_min_step", "curve_window", "objective_batches", "episodes", "entries", "nonfinite",
}
ENTRY_KEYS = {
    "step", "record_git_sha", "gate", "probe", "objective", "train_embedding", "self_check", "summary",
}
OBJECTIVE_KEYS = {"loss", "embedding", "reward", "continue", "kl_dyn", "kl_rep"}


@pytest.fixture
def reference(tmp_path, wide_buffer, capsys):
    """One real cell on thirty episodes and the diagnostic the ladder writes for it."""
    out = tmp_path / "reference"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=wide_buffer.root, ladder=tmp_path / "ladder")


def _argv(ref, *extra: str) -> list[str]:
    return [
        "--out", str(ref.ladder), "--reference", str(ref.out), "--data", str(ref.data),
        "--device", "cpu", "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--steps", str(LADDER_STEPS), "--objective-batches", "2", "--window", "2", *extra,
    ]


def _run(ref, *extra: str) -> int:
    return script.main(_argv(ref, *extra), rungs=LADDER_RUNGS)


@pytest.fixture
def trained(reference, capsys):
    assert _run(reference, "--phase", "train", "--anchor", "hard") == script.EXIT_OK
    capsys.readouterr()
    return reference


@pytest.fixture
def evaluated(trained, capsys):
    assert _run(trained, "--phase", "evaluate", "--anchor", "hard") == script.EXIT_OK
    capsys.readouterr()
    return trained


def _doctor(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_train(*args, **kwargs):
    raise AssertionError("train_world_model ran; this refusal must come before any training")


def _never_evaluate(*args, **kwargs):
    raise AssertionError("evaluate_job ran; this refusal must come before any rung is evaluated")


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_are_the_specs():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3f_ladder") and args.reference == Path("runs/m3_study_v2")
    assert args.data == Path("data/my_way_home") and args.device == "mps"
    assert args.context is None and args.horizon is None
    assert args.steps == STEPS and args.phase == "all"
    assert args.anchor in script.ANCHOR_POLICIES == ("hard", "report")
    assert args.objective_batches == OBJECTIVE_BATCHES and args.window == CURVE_WINDOW
    assert args.figure is None


def test_steps_below_the_last_rung_is_argparses_own_usage_error():
    """Judged at the parser, before any directory is read."""
    with pytest.raises(SystemExit) as raised:
        script.main(["--steps", "1"], rungs=LADDER_RUNGS)
    assert raised.value.code == 2


def test_the_reused_statuses_are_trust_horizons_and_32_and_33_are_new():
    assert (script.EXIT_OK, script.EXIT_NO_CHECKPOINTS, script.EXIT_SPLIT_MISMATCH,
            script.EXIT_RECORD_MISMATCH, script.EXIT_SELF_CHECK_FAILED) == (0, 11, 12, 14, 30)
    assert script.EXIT_NO_CHECKPOINTS is trust.EXIT_NO_CHECKPOINTS
    assert script.EXIT_SELF_CHECK_FAILED is trust.EXIT_SELF_CHECK_FAILED
    assert (script.EXIT_ANCHOR_MISMATCH, script.EXIT_RUNG_MISLABELLED) == (32, 33)
    own = {v for k, v in vars(script).items() if k.startswith("EXIT_")}
    assert own == {0, 11, 12, 14, 30, 32, 33}


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_writes_the_rungs_the_final_checkpoint_and_a_train_record_anchored_at_zero(trained):
    for step in LADDER_RUNGS:
        payload = torch.load(trained.ladder / f"step{step}" / CHECKPOINT, weights_only=True)
        assert (payload["arm"], payload["seed"], payload["step"]) == (JOB.arm, JOB.seed, step)
    assert (trained.ladder / CHECKPOINT).is_file()

    record = load_record(trained.ladder / TRAIN)
    assert record["arm"] == JOB.arm and record["seed"] == JOB.seed
    assert record["steps"] == LADDER_STEPS and record["rungs"] == list(LADDER_RUNGS)
    assert record["seq_len"] == JOB_KW["seq_len"]
    assert len(record["history"]["loss"]) == LADDER_STEPS == len(record["history"]["parts"])
    assert sorted(record["checkpoint_seconds"]) == ["2", "4"]
    assert record["anchor"] == {
        "policy": "hard", "steps": LADDER_STEPS, "max_delta": 0.0, "first_step": None,
        "reference_git_sha": load_record(trained.out / RECORD)["git_sha"],
    }
    # THE ANCHOR'S CONTENT: the retrain's losses ARE the reference run's first four.
    reference = load_record(trained.out / RECORD)
    assert record["history"]["loss"] == reference["history"]["loss"][:LADDER_STEPS]
    assert record["git_sha"] and record["device"] == "cpu" and record["torch_version"] == torch.__version__


def test_a_missing_reference_cell_is_exit_11_before_any_training(reference, monkeypatch, capsys):
    (reference.out / DIAGNOSTIC).unlink()
    monkeypatch.setattr(script, "train_world_model", _never_train)
    assert _run(reference, "--phase", "train") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert not reference.ladder.exists()


def test_a_hard_anchor_that_does_not_match_is_exit_32_naming_the_first_differing_step(reference, capsys):
    _doctor(reference.out / RECORD, lambda r: r["history"]["loss"].__setitem__(1, r["history"]["loss"][1] + 1.0))
    assert _run(reference, "--phase", "train", "--anchor", "hard") == script.EXIT_ANCHOR_MISMATCH
    out = capsys.readouterr().out
    assert "ANCHOR MISMATCH" in out and "first at step 2" in out
    # The train record IS written -- it is the evidence -- and says what it measured.
    record = load_record(reference.ladder / TRAIN)
    assert record["anchor"]["max_delta"] == pytest.approx(1.0) and record["anchor"]["first_step"] == 2
    assert record["anchor"]["policy"] == "hard"


def test_a_report_anchor_prints_the_delta_and_continues(reference, capsys):
    _doctor(reference.out / RECORD, lambda r: r["history"]["loss"].__setitem__(1, r["history"]["loss"][1] + 1.0))
    assert _run(reference, "--phase", "train", "--anchor", "report") == script.EXIT_OK
    out = capsys.readouterr().out
    assert "anchor max|delta| 1.0e+00 (first at step 2)" in out and "ANCHOR MISMATCH" not in out
    assert load_record(reference.ladder / TRAIN)["anchor"]["policy"] == "report"


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_writes_a_study_record_per_rung_and_one_ladder_record(evaluated):
    reference = load_record(evaluated.out / RECORD)
    for step in LADDER_RUNGS:
        rung = load_record(job_record_path(evaluated.ladder / f"step{step}", JOB))
        assert rung["steps"] == step and len(rung["history"]["loss"]) == step
        assert rung["history"]["loss"] == reference["history"]["loss"][:step]
        assert (rung["context"], rung["horizon"], rung["seq_len"]) == (CONTEXT, HORIZON, JOB_KW["seq_len"])
        assert rung["episodes"] == reference["episodes"]

    ladder = load_record(evaluated.ladder / LADDER)
    assert set(ladder) == LADDER_KEYS
    assert ladder["rungs"] == list(LADDER_RUNGS) and ladder["reference_rung"] == REFERENCE_RUNG
    assert set(ladder["entries"]) == {"2", "4", str(REFERENCE_RUNG)}
    assert ladder["primary_rung"] in LADDER_RUNGS and 1 <= ladder["embedding_min_step"] <= JOB_KW["steps"]
    assert ladder["episodes"]["val"] == reference["episodes"]["val"]
    assert ladder["anchor"]["max_delta"] == 0.0 and ladder["objective_batches"] == 2
    assert ladder["reference_git_sha"] == reference["git_sha"]
    for key, entry in ladder["entries"].items():
        assert set(entry) == ENTRY_KEYS and entry["step"] == int(key)
        assert set(entry["objective"]) == OBJECTIVE_KEYS
        assert np.isfinite(entry["train_embedding"])
        assert set(entry["gate"]) == {"gap_final", "steps_degenerate"}
        assert set(entry["probe"]) == {"selection_r2", "measurable"}
        summary = entry["summary"]
        assert summary["windows"]["total"] == WINDOWS and summary["windows"]["clusters"] == CLUSTERS
        assert set(summary["curves"]) == set(CURVE_NAMES)
        assert np.asarray(summary["crossing"]["free"]).shape == (WINDOWS,)
        assert np.asarray(summary["margin"]["probe"]).shape == (WINDOWS, HORIZON)
    assert ladder["entries"][str(REFERENCE_RUNG)]["self_check"]["ok"] is True
    assert ladder["entries"][str(REFERENCE_RUNG)]["self_check"]["reference_position_max_delta"] == 0.0
    assert ladder["entries"]["2"]["self_check"] is None
    assert ladder["entries"][str(REFERENCE_RUNG)]["gate"]["gap_final"] == reference["position"]["gap_final"] or (
        np.isnan(ladder["entries"][str(REFERENCE_RUNG)]["gate"]["gap_final"]) and np.isnan(reference["position"]["gap_final"])
    )


def test_the_reference_entrys_train_embedding_is_the_reference_runs_own_smoothed_term(evaluated):
    ladder = load_record(evaluated.ladder / LADDER)
    reference = load_record(evaluated.out / RECORD)
    parts = reference["history"]["parts"]
    expected = float(np.mean([p["embedding"] for p in parts[-2:]]))  # --window 2 at the last step
    assert ladder["entries"][str(REFERENCE_RUNG)]["train_embedding"] == pytest.approx(expected)
    train = load_record(evaluated.ladder / TRAIN)
    expected_2 = float(np.mean([p["embedding"] for p in train["history"]["parts"][:2]]))
    assert ladder["entries"]["2"]["train_embedding"] == pytest.approx(expected_2)


def test_evaluate_without_a_train_record_is_exit_11(reference, capsys):
    assert _run(reference, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "run --phase train first" in capsys.readouterr().out


def test_a_doctored_reference_curve_is_exit_30_before_any_rung_is_evaluated(trained, monkeypatch, capsys):
    _doctor(trained.out / DIAGNOSTIC, lambda d: d["curves"]["reference_position"].__setitem__(
        0, d["curves"]["reference_position"][0] + 1e-3))
    monkeypatch.setattr(script, "evaluate_job", _never_evaluate)
    assert _run(trained, "--phase", "evaluate") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (trained.ladder / LADDER).exists()


def test_a_split_that_is_not_the_references_is_exit_12_and_a_protocol_flag_is_exit_14(trained, monkeypatch, capsys):
    monkeypatch.setattr(script, "evaluate_job", _never_evaluate)
    assert _run(trained, "--phase", "evaluate", "--context", "3") == script.EXIT_RECORD_MISMATCH
    assert "RECORD MISMATCH" in capsys.readouterr().out
    _doctor(trained.out / RECORD, lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]))
    assert _run(trained, "--phase", "evaluate") == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH" in capsys.readouterr().out


def test_a_rung_checkpoint_labelled_with_another_step_is_exit_33(trained, capsys):
    path = trained.ladder / "step2" / CHECKPOINT
    payload = torch.load(path, weights_only=True)
    payload["step"] = 4
    torch.save(payload, path)
    assert _run(trained, "--phase", "evaluate") == script.EXIT_RUNG_MISLABELLED
    out = capsys.readouterr().out
    assert "RUNG MISLABELLED" in out and "step=4" in out and "step=2" in out
    assert not (trained.ladder / LADDER).exists()


def test_a_missing_rung_checkpoint_is_exit_11(trained, capsys):
    (trained.ladder / "step4" / CHECKPOINT).unlink()
    assert _run(trained, "--phase", "evaluate") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out


def test_a_hard_evaluate_refuses_a_train_record_whose_anchor_is_not_zero_before_anything_runs(trained, monkeypatch, capsys):
    _doctor(trained.ladder / TRAIN, lambda t: t["anchor"].update({"max_delta": 0.5, "first_step": 3}))
    monkeypatch.setattr(script, "evaluate_job", _never_evaluate)
    monkeypatch.setattr(script, "prepare_cell", _never_evaluate)
    assert _run(trained, "--phase", "evaluate", "--anchor", "hard") == script.EXIT_ANCHOR_MISMATCH
    assert "first at step 3" in capsys.readouterr().out


def test_all_trains_then_evaluates_each_cell(reference, capsys):
    assert _run(reference, "--phase", "all", "--anchor", "hard") == script.EXIT_OK
    assert (reference.ladder / TRAIN).is_file() and (reference.ladder / LADDER).is_file()


# ---------------------------------------------------------------------------
# The helpers, pure.
# ---------------------------------------------------------------------------


def test_history_from_train_record_round_trips_what_history_at_needs():
    history = {
        "arm": "random_vit", "steps": 4, "loss": [4.0, 3.0, 2.0, 1.0],
        "parts": [{"embedding": e, "reward": 0.0, "continue": 0.0, "kl_dyn": 0.3, "kl_rep": 0.3}
                  for e in (0.4, 0.3, 0.2, 0.1)],
        "seconds": 8.0, "kl_dyn_max": 0.3, "kl_rate_above_free_bits": 1.0,
        "checkpoint_seconds": {2: 3.0, 4: 8.0},
    }
    reference = types.SimpleNamespace(record={"seq_len": 4, "git_sha": "abc"})
    record = script.train_record("random_vit", 1, history, (2, 4), 0.0, None,
                                 steps=4, policy="hard", reference=reference, device="cpu")
    assert record["checkpoint_seconds"] == {"2": 3.0, "4": 8.0} and record["seq_len"] == 4
    back = script.history_from_train_record(json.loads(json.dumps(record)))
    assert history_at(back, 2) == history_at(history, 2)
    assert history_at(back, 4)["seconds"] == 8.0


def test_smoothed_at_is_the_window_mean_ending_at_the_step():
    parts = [{"embedding": v} for v in (1.0, 2.0, 4.0, 8.0)]
    assert script.smoothed_at(parts, "embedding", 4, 2) == 6.0
    assert script.smoothed_at(parts, "embedding", 3, 2) == 3.0
    assert script.smoothed_at(parts, "embedding", 1, 100) == 1.0, "the window shrinks to the prefix"
    with pytest.raises(ValueError):
        script.smoothed_at(parts, "embedding", 0, 2)


def test_rung_cell_carries_the_rung_records_own_protocol_in_the_diagnostics_slot(evaluated):
    cell = script.rung_cell(evaluated.ladder / "step2", JOB.arm, JOB.seed)
    assert cell.diagnostic == {"context": CONTEXT, "horizon": HORIZON}
    assert cell.record["steps"] == 2 and cell.checkpoint.is_file()
    with pytest.raises(script.CellMissing, match="no checkpoint"):
        script.rung_cell(evaluated.ladder / "step3", JOB.arm, JOB.seed)


# ---------------------------------------------------------------------------
# read (Task 6): pooling glue on fabricated records, then the fixture run.
# ---------------------------------------------------------------------------

from mbfps.eval.ladder import DECISION_H  # noqa: E402


def _fake_entry(step: int, crossing, *, windows: int, horizon: int, measurable: bool, r2: float) -> dict:
    crossing = np.asarray(crossing, dtype=float)
    moved = np.ones((windows, horizon), dtype=bool)
    margin = np.tile(np.arange(1, horizon + 1, dtype=float), (windows, 1))
    return {
        "step": step, "record_git_sha": "ref",
        "gate": {"gap_final": 0.1, "steps_degenerate": 0},
        "probe": {"selection_r2": r2, "measurable": measurable},
        "objective": {k: 0.1 for k in OBJECTIVE_KEYS},
        "train_embedding": 0.1,
        "self_check": ({"reference_position_max_delta": 0.0, "persistence_position_max_delta": 0.0,
                        "windows_total_match": True, "windows_episode_match": True, "ok": True}
                       if step == REFERENCE_RUNG else None),
        "summary": {
            "windows": {"total": windows, "episode": [0, 0, 1, 1][:windows], "clusters": 2},
            "crossing": {"free": crossing.tolist(), "probe": crossing.tolist()},
            "margin": {"free": margin.tolist(), "probe": margin.tolist()},
            "moved": moved.tolist(),
            "first_moved": [1.0] * windows,
            "survival": {"free": [1.0] * (horizon + 1), "probe": [1.0] * (horizon + 1)},
            "trust_horizon": {"free": {"q50": 1, "q75": 1, "q90": 0}, "probe": {"q50": 1, "q75": 1, "q90": 0}},
            "counts": {"not_moved": [0] * horizon, "never_moved": 0},
        },
    }


def _fake_record(seed: int, *, primary: int, crossing_primary, crossing_reference,
                 measurable_reference: bool = True, r2: float = 0.5) -> dict:
    kw = dict(windows=4, horizon=3, r2=r2)
    entries = {
        "2": _fake_entry(2, crossing_primary if primary == 2 else [1, 1, 1, 1], measurable=True, **kw),
        "4": _fake_entry(4, crossing_primary if primary == 4 else [1, 1, 1, 1], measurable=True, **kw),
        str(REFERENCE_RUNG): _fake_entry(REFERENCE_RUNG, crossing_reference, measurable=measurable_reference, **kw),
    }
    return {
        "arm": "random_vit", "seed": seed, "context": 2, "horizon": 3, "device": "cpu",
        "torch_version": "t", "episodes": {"val": ["a", "b"]}, "primary_rung": primary,
        "rungs": [2, 4], "entries": entries,
    }


def test_survival_series_is_the_indicator_over_the_windows_that_ever_moved():
    r = _fake_record(0, primary=2, crossing_primary=[3, 3, float("nan"), 1], crossing_reference=[1, 1, 1, 1])
    s = script.survival_series(r, 2, "free", 2, "random_vit@primary")
    assert s.arm == "random_vit@primary" and s.rung == "val" and s.channel == "S/free"
    assert s.delta.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert s.changed.tolist() == [True, True, False, True]
    assert s.episode.tolist() == [0, 0, 1, 1] and s.windows_total == 4


def test_margin_series_masks_by_moved_at_h_not_ever_moved():
    r = _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1])
    r["entries"]["2"]["summary"]["moved"][1][1] = False
    s = script.margin_series(r, 2, "probe", 2, "random_vit@primary")
    assert s.channel == "margin/probe" and s.delta.tolist() == [2.0, 2.0, 2.0, 2.0]
    assert s.changed.tolist() == [True, False, True, True]


def test_arm_inputs_pairs_each_seeds_primary_rung_against_its_reference_on_the_same_windows():
    """Seed 0's primary is rung 2, seed 1's is rung 4; at h=2 the indicator is
    `crossing > 2`. Treatment seed-mean [1, .5, .5, 0] against control [0, 0,
    0, 0]: the paired mean is 0.5, over 2 episode clusters."""
    records = {
        ("random_vit", 0): _fake_record(0, primary=2, crossing_primary=[3, 3, 1, 1], crossing_reference=[1, 1, 1, 1]),
        ("random_vit", 1): _fake_record(1, primary=4, crossing_primary=[3, 1, 3, 1], crossing_reference=[1, 1, 1, 1]),
    }
    a = script.arm_inputs(records, "random_vit", [0, 1], h=2)
    assert a.primary_rungs == {0: 2, 1: 4}
    assert a.t_free.estimate == pytest.approx(0.5) and a.t_free.clusters == 2
    assert a.t_probe.estimate == pytest.approx(0.5)
    assert a.per_seed[0].t_free.estimate == pytest.approx(0.5) and a.per_seed[0].per_seed is None
    assert a.per_seed[1].t_free.estimate == pytest.approx(0.5) and a.per_seed[1].primary_rungs == {1: 4}


def test_a_cell_unmeasurable_at_either_rung_leaves_the_probe_channel_only():
    records = {
        ("random_vit", 0): _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1],
                                        measurable_reference=False),
    }
    a = script.arm_inputs(records, "random_vit", [0], h=2)
    assert a.t_free.estimate == pytest.approx(1.0)
    assert np.isnan(a.t_probe.estimate) and a.t_probe.clusters == 0
    # The sensitivity line drops a low-R2 cell the same way.
    low = _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1], r2=0.05)
    b = script.arm_inputs({("random_vit", 0): low}, "random_vit", [0], h=2, min_r2=0.1)
    assert np.isnan(b.t_probe.estimate) and b.t_free.estimate == pytest.approx(1.0)


def test_timing_inputs_reads_z_fam_off_the_val_clusters_and_orders_the_arms():
    records = {
        ("random_vit", 0): _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1]),
    }
    inputs = script.timing_inputs(records, arms=["random_vit"], seeds=[0], h=2)
    assert inputs.h == 2 and list(inputs.arms) == ["random_vit"]
    assert inputs.z_fam == pytest.approx(script.pooling.cluster_threshold(6, 2))


def test_objective_table_names_the_smallest_finite_rung_and_ignores_a_nan():
    """A NaN objective at the first rung must not pin the "smallest at step"
    line to it: `x < nan` is always False, so a naive running minimum sticks."""
    r = _fake_record(0, primary=2, crossing_primary=[3, 3, 3, 3], crossing_reference=[1, 1, 1, 1])
    r["objective_batches"], r["curve_window"] = 2, 100
    r["entries"]["2"]["objective"]["embedding"] = float("nan")
    r["entries"]["4"]["objective"]["embedding"] = 0.5
    r["entries"][str(REFERENCE_RUNG)]["objective"]["embedding"] = 0.3
    r["entries"]["2"]["objective"]["loss"] = 0.2
    r["entries"]["4"]["objective"]["loss"] = float("nan")
    r["entries"][str(REFERENCE_RUNG)]["objective"]["loss"] = 0.4
    text = script._objective_table({("random_vit", 0): r}, (2, 4))
    assert f"val embedding smallest at step {REFERENCE_RUNG}; val loss smallest at step 2" in text
    for step in ("2", "4", str(REFERENCE_RUNG)):
        r["entries"][step]["objective"]["embedding"] = float("nan")
    text = script._objective_table({("random_vit", 0): r}, (2, 4))
    assert "val embedding smallest at step n/a" in text


def test_decision_horizon_is_fifteen_unless_the_run_is_shorter():
    assert script.decision_horizon(45) == (DECISION_H, False)
    assert script.decision_horizon(3) == (3, True)


@pytest.fixture
def ladder_run(evaluated, capsys):
    status = _run(evaluated, "--phase", "read")
    out = capsys.readouterr().out
    assert status == script.EXIT_OK, out
    return types.SimpleNamespace(cell=evaluated, out=out, text=(evaluated.ladder / "ladder.txt").read_text())


def test_read_writes_ladder_txt_identical_to_stdout(ladder_run):
    assert ladder_run.text == ladder_run.out
    assert (ladder_run.cell.ladder / "ladder_curves.png").is_file()


def test_read_prints_every_block_in_order(ladder_run):
    text = ladder_run.text
    headers = [
        "--- anchor:", "--- primary rung per cell:", "--- reference self-check",
        "--- the gate at every rung:", "--- survival per rung:", "--- conditional survival per rung:",
        "--- the validation objective per rung", "  pooling: z_fam = cluster_threshold(6, 6)",
        "--- Reading T:", "--- T at the reported horizons", "--- sensitivity", "--- per seed",
        "figure=",
    ]
    positions = [text.index(h) for h in headers]
    assert positions == sorted(positions), "the blocks are printed in the spec's order"
    assert "verdict: random_vit" in text and "decided by:" in text
    assert f"h={HORIZON}" in text and "(pre-registered DECISION_H = 15)" in text, \
        "the fixture's horizon is 3, so the decision horizon is clamped and says so"
    assert "GATE PASSES" in text or "gap_closed(3)" in text
    assert "20000" in text


def test_a_single_seed_run_reads_this_seed_alone_per_seed_and_zero_of_one_pooled(ladder_run):
    text = ladder_run.text
    assert "  random_vit  s1:" in text
    # With one seed the pooled reading can never replicate: whatever the
    # fixture's contrast is, the status is NO DIFFERENCE and the rule says why
    # if it cleared.
    line = next(l for l in text.splitlines() if l.startswith("  verdict: random_vit"))
    assert "NO DIFFERENCE" in line
    assert ("does not clear" in line) or ("only 1 of 1 seeds" in line)


def test_read_refuses_a_ladder_record_whose_self_check_is_not_ok(evaluated, capsys):
    _doctor(evaluated.ladder / LADDER, lambda r: r["entries"][str(REFERENCE_RUNG)]["self_check"].update({"ok": False}))
    assert _run(evaluated, "--phase", "read") == script.EXIT_SELF_CHECK_FAILED
    assert "SELF-CHECK FAILED" in capsys.readouterr().out
    assert not (evaluated.ladder / "ladder.txt").exists()


def test_read_without_a_ladder_record_is_exit_11(trained, capsys):
    assert _run(trained, "--phase", "read") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out


def test_write_curves_handles_a_missing_matplotlib_by_costing_the_figure_only(evaluated, monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    records = {(JOB.arm, JOB.seed): load_record(evaluated.ladder / LADDER)}
    line = script.write_curves(records, tmp_path / "curves.png", LADDER_RUNGS)
    assert line.startswith("figure NOT written:") and not (tmp_path / "curves.png").exists()
