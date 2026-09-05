"""One study job: train, evaluate, emit a record the aggregation can read.

Nine of these records are the entire output of a 30-GPU-hour study, so every
test here is about a field that is silently wrong rather than absent.
"""

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval import study
from mbfps.eval.study import (
    NONFINITE_KEY,
    SPLIT_SEED,
    StudyJob,
    _summarise_reward,
    job_record_path,
    load_record,
    run_job,
    to_json_record,
    write_record,
)

JOB_KW = dict(steps=3, seq_len=4, context=2, horizon=3, device="cpu")

GAIN_KEYS = {
    "gain", "joint_r2", "embedding_r2", "ci_low", "ci_high", "confidence",
    "n_scored_windows", "ridge_selected", "joint_ridge", "embedding_ridge",
}


def _reject(constant):
    """`json.loads` calls this for the non-standard NaN/Infinity tokens.

    Python's parser accepts them by default, so without this hook a test that
    "the record parses" would pass on a file no strict parser can read -- which
    is exactly the failure this module's NaN policy exists to prevent.
    """
    raise AssertionError(f"record contains the non-JSON token {constant!r}")


def strict_loads(text: str):
    return json.loads(text, parse_constant=_reject)


@pytest.fixture
def record(tmp_path, small_buffer):
    """One completed job. ~2 s, so the assertions below share it.

    This is the LIVE record `run_job` returns -- real NaNs, no `nonfinite`
    map. `written` below is the same job's on-disk projection.
    """
    return run_job(StudyJob("random_vit", 0), small_buffer, tmp_path, **JOB_KW)


@pytest.fixture
def written(record):
    """What the file holds for the same job: sanitised, with the token map."""
    return to_json_record(record)


# --------------------------------------------------------------------------
# record path
# --------------------------------------------------------------------------

def test_job_record_path_is_unique_per_arm_and_seed(tmp_path):
    a = job_record_path(tmp_path, StudyJob("cnn", 0))
    b = job_record_path(tmp_path, StudyJob("cnn", 1))
    c = job_record_path(tmp_path, StudyJob("frozen_ssl", 0))
    assert len({a, b, c}) == 3
    assert a.suffix == ".json"


def test_job_record_path_names_both_the_arm_and_the_seed(tmp_path):
    """The driver resumes off these files. A name that drops either field
    collides two of the nine cells, and the second one is never run -- it is
    reported as the first one's numbers."""
    path = job_record_path(tmp_path, StudyJob("frozen_ssl", 2))
    assert "frozen_ssl" in path.name
    assert "2" in path.name.replace("frozen_ssl", "")
    assert path.parent == Path(tmp_path)


# --------------------------------------------------------------------------
# the record's contents
# --------------------------------------------------------------------------

def test_record_carries_everything_the_gate_needs(record):
    """A record missing a field silently drops one cell of the 3x3 table."""
    for key in ("arm", "seed", "steps", "seconds", "steps_per_second",
                "kl_rate_above_free_bits", "position", "angle", "filtering",
                "reward", "curves", "episodes", "probe", "split_seed"):
        assert key in record, f"record is missing {key!r}"
    for metric in ("position", "angle"):
        for key in ("gap_final", "gap_mean", "band_median",
                    "steps_floor_above_persistence", "steps_degenerate",
                    "gap_finite"):
            assert key in record[metric], f"{metric} is missing {key!r}"
    assert record["arm"] == "random_vit"
    assert record["seed"] == 0


def test_the_angle_summary_is_not_a_copy_of_the_position_summary(record):
    """Both metrics get the same guards, on their own curves. An earlier
    version computed all of this for position only."""
    assert record["angle"]["final_model"] != record["position"]["final_model"]
    assert record["angle"]["final_model"] == pytest.approx(
        record["curves"]["rssm_angle"][-1])
    assert record["position"]["final_model"] == pytest.approx(
        record["curves"]["rssm_position"][-1])


def test_the_record_carries_both_filtering_diagnostics(record):
    """Criterion 4 and the gain answer different questions and neither
    substitutes for the other: criterion 4 puts a ~160-bit latent against 2048
    encoder floats and can fail on the bottleneck alone, while the gain hands
    the raw embedding to both arms."""
    filtering = record["filtering"]
    assert set(filtering) == {"criterion_4", "gain"}
    assert set(filtering["criterion_4"]) == {
        "latent_r2", "embedding_r2", "latent_beats_embedding"}
    assert set(filtering["gain"]) == GAIN_KEYS


def test_the_record_says_which_ridge_each_gain_arm_selected(record):
    """The gain's SIGN is ridge-grid dependent -- measured -0.0706, -0.0208 and
    +0.0325 under three defensible selection designs, because RIDGES is a decade
    grid and the scored R^2 moves ~0.10 per decade while the effect is ~0.02.
    Without the selected ridges in the record that sensitivity is invisible
    across nine runs and the aggregate cannot be interpreted."""
    gain = record["filtering"]["gain"]
    from mbfps.eval.probe import RIDGES

    assert gain["joint_ridge"] in RIDGES
    assert gain["embedding_ridge"] in RIDGES
    assert isinstance(gain["ridge_selected"], bool)


def test_the_curves_are_as_long_as_the_horizon(record):
    """`horizon` reaching the rollout is what makes the curves this length; a
    call that let it default would report a different number of steps."""
    assert record["horizon"] == JOB_KW["horizon"]
    for name in ("rssm_position", "persistence_position", "floor_position",
                 "rssm_angle", "persistence_angle", "floor_angle"):
        assert len(record["curves"][name]) == JOB_KW["horizon"]
    assert record["position"]["n_steps"] == JOB_KW["horizon"]


def test_the_written_file_is_the_returned_records_json_projection(
    record, written, tmp_path
):
    """The file is `to_json_record(returned)` -- not the returned dict itself.

    The returned dict keeps its real NaNs (see
    `test_the_returned_record_does_not_crash_the_studys_own_driver`); the file
    is its lossy projection plus the map that inverts it. Nothing else may
    differ between them, or a caller reasoning about the returned record is
    reasoning about a different object from the one Task 6 aggregates.
    """
    path = job_record_path(tmp_path, StudyJob("random_vit", 0))
    assert path.is_file()
    assert strict_loads(path.read_text()) == written
    assert NONFINITE_KEY not in record, "the returned record is the live one"
    assert set(written) == set(record) | {NONFINITE_KEY}


# --------------------------------------------------------------------------
# reward accuracy
# --------------------------------------------------------------------------

def test_reward_accuracy_reports_its_own_degeneracy(small_buffer):
    """Spec 4.3 asks for reward accuracy, but this scenario's reward is
    near-constant: 19,417 of 19,424 real steps share one value and six carry the
    goal. A bare MSE would look precise and mean nothing, so the report must
    carry the baseline and the event count and flag the degeneracy itself."""
    import torch

    from mbfps.data.split import episode_split
    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    model.eval()
    _, val = episode_split(small_buffer.episode_paths(), val_fraction=0.2, seed=0)
    out = reward_accuracy(model, val, encoder_backbone(cfg.encoder),
                          torch.device("cpu"), limit=2)

    assert set(out) == {"mse", "baseline_mse", "r2", "n_steps",
                        "n_reward_events", "is_degenerate"}
    assert out["n_steps"] > 0
    assert out["baseline_mse"] >= 0.0
    assert isinstance(out["is_degenerate"], bool)


def test_reward_accuracy_flags_a_constant_reward_as_degenerate():
    """A constant target makes R^2 undefined; the flag is what stops a caller
    reporting a confident-looking number over it."""
    constant = _summarise_reward(np.zeros(50), np.zeros(50))
    assert constant["is_degenerate"] is True
    assert constant["n_reward_events"] == 0
    assert math.isnan(constant["r2"]), "a constant target has no R^2 to report"

    varied = _summarise_reward(np.arange(50, dtype=float), np.arange(50, dtype=float))
    assert varied["is_degenerate"] is False
    assert varied["r2"] > 0.99


def test_reward_summary_counts_the_events_and_the_baseline():
    """The baseline is the target's own variance: an MSE below it is the only
    thing that means the head learned anything."""
    true = np.zeros(1000)
    true[:2] = 10.0
    out = _summarise_reward(np.zeros(1000), true)
    assert out["n_reward_events"] == 2
    assert out["n_steps"] == 1000
    assert out["is_degenerate"] is True          # 998/1000 share the modal value
    assert out["baseline_mse"] == pytest.approx(np.var(true))
    assert out["mse"] == pytest.approx(np.mean(true ** 2))
    assert out["r2"] == pytest.approx(
        1.0 - np.sum(true ** 2) / np.sum((true - true.mean()) ** 2))


def test_reward_summary_refuses_arrays_it_cannot_compare():
    """Both messages are load-bearing. Without the guards numpy still raises --
    on an empty array it is `zero-size array to reduction operation maximum`,
    from three frames deep in a modal-share calculation -- and that is the
    message someone reads after a job has already burned its GPU hours. The
    `match` is what keeps these guards from being deletable no-ops."""
    with pytest.raises(ValueError, match="no reward steps to summarise"):
        _summarise_reward(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="are not aligned"):
        _summarise_reward(np.zeros(10), np.zeros(11))


# --------------------------------------------------------------------------
# the NaN policy
# --------------------------------------------------------------------------

def test_a_degenerate_record_is_written_as_valid_json(record, written, tmp_path):
    """`gap_final` is NaN by `metric_summary`'s contract whenever the band is
    non-positive, and this fixture's band IS non-positive. `json.dump` writes a
    bare `NaN` token for it, which is not JSON."""
    path = job_record_path(tmp_path, StudyJob("random_vit", 0))
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    strict_loads(text)      # raises via _reject if a bare token survived
    assert written[NONFINITE_KEY], "this fixture's band is degenerate by design"
    assert written["angle"]["gap_final"] is None
    assert written[NONFINITE_KEY]["angle.gap_final"] == "nan"
    # The map's own key is a cross-task contract: Task 6 reads nine of these
    # files and looks the map up by this literal name.
    assert "nonfinite" in strict_loads(text)


def test_an_undefined_gap_is_never_read_back_as_zero(record, written, tmp_path):
    """The distinction the policy exists to keep. NaN gap_final means the band
    was non-positive so the ratio is undefined; 0.0 means the model closed none
    of a real band. Averaging the first as the second across nine cells pulls
    the aggregate toward "no better than persistence" using cells that measured
    nothing at all."""
    restored = load_record(job_record_path(tmp_path, StudyJob("random_vit", 0)))
    for dotted in written[NONFINITE_KEY]:
        section, field = dotted.split(".", 1)
        assert written[section][field] is None
        assert restored[section][field] != 0.0
        assert not math.isfinite(restored[section][field])
    assert restored["position"]["n_steps"] == record["position"]["n_steps"]


def test_the_non_finite_policy_round_trips_every_kind(tmp_path):
    """null plus a token map, rather than null alone: null would lose
    NaN-versus-infinity, and an infinite error is a different bug from an
    undefined one."""
    original = {
        "nan": float("nan"),
        "inf": float("inf"),
        "-inf": float("-inf"),
        "zero": 0.0,
        "nested": {"deep": [1.0, float("nan"), 3.0]},
        "numpy": np.float32("nan"),
    }
    written = write_record(tmp_path / "r.json", original)
    text = (tmp_path / "r.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert strict_loads(text) == written
    assert written["nan"] is None and written["zero"] == 0.0
    assert written["nested"]["deep"] == [1.0, None, 3.0]

    back = load_record(tmp_path / "r.json")
    assert math.isnan(back["nan"]) and math.isnan(back["numpy"])
    assert back["inf"] == float("inf")
    assert back["-inf"] == float("-inf")
    assert back["zero"] == 0.0
    assert math.isnan(back["nested"]["deep"][1])
    assert back["nested"]["deep"][0] == 1.0


def test_numpy_scalars_are_written_as_plain_json(tmp_path):
    """`json.dump` cannot serialise numpy scalars at all, and several of these
    fields arrive as numpy types from the summaries."""
    written = write_record(tmp_path / "r.json", {
        "f": np.float32(1.5), "i": np.int64(7), "b": np.bool_(True),
        "a": np.arange(3),
    })
    assert written["f"] == 1.5 and isinstance(written["f"], float)
    assert written["i"] == 7 and isinstance(written["i"], int)
    assert written["b"] is True
    assert written["a"] == [0, 1, 2]
    assert strict_loads((tmp_path / "r.json").read_text()) == written


def test_write_record_refuses_to_emit_a_bare_nan_token(tmp_path, monkeypatch):
    """`allow_nan=False` is the belt to the sanitiser's braces. If the
    sanitiser ever regresses, this raises here rather than nine runs and thirty
    GPU-hours later, in the aggregation."""
    monkeypatch.setattr(study, "_sanitise", lambda value, path, found: value)
    with pytest.raises(ValueError):
        write_record(tmp_path / "r.json", {"gap_final": float("nan")})


def test_load_record_refuses_a_token_it_does_not_understand(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"x": None, NONFINITE_KEY: {"x": "maybe"}}))
    with pytest.raises(ValueError, match="maybe"):
        load_record(path)


def test_to_json_record_leaves_the_input_alone():
    """The caller's record still holds real NaNs afterwards; only the written
    copy is lossy."""
    original = {"position": {"gap_final": float("nan")}}
    clean = to_json_record(original)
    assert clean["position"]["gap_final"] is None
    assert math.isnan(original["position"]["gap_final"])
    assert NONFINITE_KEY not in original


# --------------------------------------------------------------------------
# reproducibility and the held-out split
# --------------------------------------------------------------------------

def test_two_runs_of_the_same_job_agree(tmp_path, small_buffer):
    """The whole study rests on this: same arm, same seed, same numbers.

    Compared through `to_json_record`, NOT on the live dicts: `run_job` now
    returns real NaNs and `nan != nan`, so a direct `==` on two identical
    degenerate records would report them as different. The projection maps
    every non-finite to `None` (equal to itself) and records WHICH token it
    was in the map, so `nan` and `inf` still cannot pass for one another.
    """
    first = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "a", **JOB_KW)
    second = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "b", **JOB_KW)
    a, b = to_json_record(first), to_json_record(second)
    assert a[NONFINITE_KEY] == b[NONFINITE_KEY]
    assert a[NONFINITE_KEY], "this fixture's band is degenerate; the map must be live"
    for section in ("position", "angle", "curves", "filtering", "reward", "probe"):
        assert a[section] == b[section], section


def test_different_seeds_give_different_results(tmp_path, small_buffer):
    a = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "a", **JOB_KW)
    b = run_job(StudyJob("random_vit", 1), small_buffer, tmp_path / "b", **JOB_KW)
    assert a["curves"]["rssm_position"] != b["curves"]["rssm_position"]
    assert a["seed"] == 0 and b["seed"] == 1


def test_the_held_out_episodes_do_not_move_with_the_job_seed(tmp_path, small_buffer):
    """`train_world_model` splits at seed 0. Splitting the evaluation at the
    JOB's seed would hand seed 1 a held-out set its own training run trained on,
    and the nine cells would no longer be scored on the same episodes."""
    a = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "a", **JOB_KW)
    b = run_job(StudyJob("random_vit", 1), small_buffer, tmp_path / "b", **JOB_KW)
    assert a["episodes"]["val"] == b["episodes"]["val"] != []
    assert a["episodes"]["train"] == b["episodes"]["train"] != []
    assert len(a["episodes"]["val"]) + len(a["episodes"]["train"]) == 6
    assert not set(a["episodes"]["val"]) & set(a["episodes"]["train"])
    assert a["split_seed"] == SPLIT_SEED == 0


def test_every_evaluation_runs_at_the_jobs_seed_and_the_jobs_window(
    tmp_path, small_buffer, monkeypatch
):
    """Each of these takes `seed`, `context` and `horizon`, and each of them
    silently accepts its own defaults. A rollout evaluated at seed 0 while the
    model trained at seed 2 is not that cell's number, and a probe fit at a
    different filtering depth from the rollout it is applied to is a
    distribution mismatch worth ~25 map units of position error.

    The job seed is 7, not 2: `JOB_KW["context"]` is 2, so at seed 2 the
    `seed=` and `context=` assertions below both read 2 and exchanging the two
    keyword VALUES at the call sites was a no-op this test could not see --
    the same coincidence-of-fixture-values that hid the probe-ridge swap."""
    seen: dict[str, dict] = {}
    assert 7 not in (JOB_KW["context"], JOB_KW["horizon"], JOB_KW["seq_len"]), (
        "the job seed must not coincide with any window value, or a seed/"
        "window exchange at the call sites is invisible here")

    def spy(name):
        real = getattr(study, name)

        def wrapper(*args, **kwargs):
            seen[name] = kwargs
            return real(*args, **kwargs)

        monkeypatch.setattr(study, name, wrapper)

    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain", "episode_split"):
        spy(name)

    run_job(StudyJob("random_vit", 7), small_buffer, tmp_path, **JOB_KW)

    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain"):
        assert name in seen, f"{name} was never called"
        assert seen[name]["seed"] == 7, f"{name} did not get the job's seed"
    for name in ("fit_probes", "evaluate_rollout", "filtering_report",
                 "filtering_gain"):
        assert seen[name]["context"] == JOB_KW["context"], name
        assert seen[name]["horizon"] == JOB_KW["horizon"], name
    # The split is the one exception, and deliberately so.
    assert seen["episode_split"]["seed"] == SPLIT_SEED


def test_a_checkpoint_from_another_job_is_refused(tmp_path, small_buffer, monkeypatch):
    """The record is written beside the checkpoint it describes. Loading one
    from a different arm or seed would attribute another cell's weights to this
    cell, and the state dict would load cleanly because the architecture is
    shared across seeds."""
    def fake_train(cfg, buffer, out_dir, log_every=100):
        import torch
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"arm": "cnn", "seed": 99, "state_dict": {}},
            out_dir / f"world_model_{cfg.arm}_seed{cfg.train.seed}.pt",
        )
        return {"steps": 3, "seconds": 1.0, "loss": [0.0],
                "kl_rate_above_free_bits": 0.0, "kl_dyn_max": 0.0}

    monkeypatch.setattr(study, "train_world_model", fake_train)
    with pytest.raises(ValueError, match="not this job's"):
        run_job(StudyJob("random_vit", 0), small_buffer, tmp_path, **JOB_KW)


# ---------------------------------------------------------------------------
# Review follow-up: the twelve confirmed findings of task 4's review. The
# record itself was right; its VALUES, its DEFAULTS and its returned FORM were
# unguarded, and a mutation of each survived the whole suite. Each test below
# names the mutation it kills. See .superpowers/sdd/task-4-report.md.
# ---------------------------------------------------------------------------


def test_the_returned_record_does_not_crash_the_studys_own_driver(record):
    """`scripts/run_study.py`'s per-job print, verbatim, on a degenerate cell.

    `run_job` used to return `write_record`'s SANITISED output, so
    `record["position"]["gap_final"]` was `None` rather than NaN and the
    driver's own format spec raised
    `unsupported format string passed to NoneType.__format__` -- AFTER the
    record was safely on disk. Cell one survived, the loop died, and the other
    eight jobs of an unattended overnight run never started. A float NaN
    formats as `+nan` and compares False, so the driver rides through.

    The formatting happens FIRST and unguarded, so a regression raises here
    exactly as it would in the driver rather than being reported as a failed
    type assertion. `+nan` in the rendered line is then what proves this
    fixture really is the degenerate case the bug needs.
    """
    line = (f"  steps/s={record['steps_per_second']:.2f} "
            f"kl_rate={record['kl_rate_above_free_bits']:.3f} "
            f"position_gap_final={record['position']['gap_final']:+.4f} "
            f"band_median={record['position']['band_median']:.2f}")
    assert "position_gap_final=+nan" in line, (
        "this fixture's position band is degenerate by design; without a "
        "non-finite gap_final this test cannot see the bug it exists for")
    gap = record["position"]["gap_final"]
    assert (gap > 0) is False, "the driver's gate arithmetic must not raise either"
    assert isinstance(gap, float) and math.isnan(gap)


# --------------------------------------------------------------------------
# the values the nine cells will actually run at
# --------------------------------------------------------------------------

def test_run_job_defaults_are_the_spec_values():
    """The driver calls `run_job(job, buffer, out, steps=..., seq_len=...,
    device=...)` and passes NEITHER context NOR horizon, so these defaults ARE
    the study's filtering depth and rollout length (spec line 68: condition on
    5 real frames, then imagine 45). `context: int = 5 -> 1` and
    `horizon: int = 45 -> 12` each survived the whole suite, because every test
    passes JOB_KW explicitly. `device: str = "mps" -> "cpu"` survived for the
    same reason and would silently take a rented GPU out of the run."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(run_job).parameters.items()
    }
    assert defaults["steps"] == 20_000
    assert defaults["seq_len"] == 64
    assert defaults["context"] == 5
    assert defaults["horizon"] == 45
    assert defaults["device"] == "mps"


def test_reward_accuracy_defaults_are_the_spec_values():
    """`run_job` never passes `limit`, so 20 is what the nine cells score at."""
    from mbfps.eval.study import reward_accuracy

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(reward_accuracy).parameters.items()
    }
    assert defaults["limit"] == 20
    assert defaults["seed"] == 0


def test_the_reward_limit_default_bounds_a_bare_call(small_buffer):
    """The default was not merely unpinned, it was UNREACHABLE: the one
    validation episode of this fixture scores identically at limit 1, 2 and 20.
    This calls `reward_accuracy` with no `limit` over three episodes, so
    `limit -> 1` truncates the target and the step count moves."""
    import torch

    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config
    from mbfps.data.episode import load_episode

    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    model.eval()
    paths = small_buffer.episode_paths()[:3]
    expected = sum(len(load_episode(p).rewards) for p in paths)
    out = reward_accuracy(model, paths, encoder_backbone(cfg.encoder),
                          torch.device("cpu"))
    assert out["n_steps"] == expected, (
        "the bare default must score all three episodes, not a truncation")


def test_the_record_reports_the_window_and_budget_the_job_ran_at(
    tmp_path, small_buffer
):
    """`"steps": 0`, `"seq_len": 0`, `"context": 0` and `"seconds": 0.0` all
    survived: the record is the study's only audit trail for the numbers each
    cell ran at, and every one of them was write-only.

    Run at FOUR DISTINCT values rather than the shared `record` fixture:
    `JOB_KW` has `steps=3` and `horizon=3`, so exchanging those two fields in
    the record is a numerical no-op and all four assertions below pass on a
    record that misreports both. Four distinct values make every pairwise
    exchange visible."""
    distinct = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
    assert len({v for k, v in distinct.items() if k != "device"}) == 4, (
        "two of these coincide, so exchanging the matching pair of record "
        "fields would be invisible to this test")

    record = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path,
                     **distinct)

    assert record["steps"] == distinct["steps"]
    assert record["seq_len"] == distinct["seq_len"]
    assert record["context"] == distinct["context"]
    assert record["horizon"] == distinct["horizon"]
    assert record["split_seed"] == SPLIT_SEED
    assert record["seconds"] > 0.0


def test_the_jobs_device_reaches_the_training_config_and_the_evaluation(
    tmp_path, small_buffer, monkeypatch
):
    """Three device mutations survived -- the default, `get_device(prefer="cpu")`
    and `get_config(..., device="cpu")` -- because every test in this file runs
    at device="cpu", so no assertion could tell "honoured" from "hardcoded".
    A study launched with `--device cuda` that silently trains on CPU is a
    33-hour loss. Run at a device string that is NOT the one a mutation would
    hardcode; `get_device` falls back to CPU wherever CUDA is absent, so this
    stays a CPU test."""
    real_get_config = study.get_config
    real_get_device = study.get_device
    real_world_model = study.WorldModel
    seen: dict = {}
    models: list = []

    def config_spy(arm, **overrides):
        seen["cfg_device"] = overrides.get("device")
        return real_get_config(arm, **overrides)

    def device_spy(prefer="mps"):
        seen["prefer"] = prefer
        return real_get_device(prefer=prefer)

    def model_spy(cfg):
        model = real_world_model(cfg)
        models.append(model)
        return model

    monkeypatch.setattr(study, "get_config", config_spy)
    monkeypatch.setattr(study, "get_device", device_spy)
    monkeypatch.setattr(study, "WorldModel", model_spy)

    run_job(StudyJob("random_vit", 0), small_buffer, tmp_path,
            **dict(JOB_KW, device="cuda"))

    assert seen["cfg_device"] == "cuda", (
        "the job's device never reached the training config, so train_world_model "
        "trained wherever TrainConfig defaults to")
    assert seen["prefer"] == "cuda", (
        "the job's device never reached get_device, so the evaluation ran on "
        "whatever get_device defaults to")
    assert models, "run_job never built the evaluation model"
    expected = real_get_device(prefer="cuda")
    assert next(models[0].parameters()).device.type == expected.type


def test_the_evaluation_holds_out_exactly_what_training_held_out(
    tmp_path, small_buffer, monkeypatch
):
    """The other half of the SPLIT_SEED coupling. The seed was factored out
    with a docstring about this exact hazard; the FRACTION stayed a duplicated
    literal in both modules, and `val_fraction=0.2 -> 0.4` in study.py alone
    left the suite green while the evaluation scored on episodes the model had
    trained on ('ep002' moves into the held-out set at 0.4). This runs both
    call sites in one process and requires them to agree on the shared
    constant."""
    import mbfps.training.world_model as wm
    from mbfps.data.split import VAL_FRACTION

    seen: dict[str, list] = {"study": [], "training": []}
    real_study_split = study.episode_split
    real_wm_split = wm.episode_split

    def study_spy(paths, val_fraction, seed):
        seen["study"].append(val_fraction)
        return real_study_split(paths, val_fraction, seed)

    def wm_spy(paths, val_fraction, seed):
        seen["training"].append(val_fraction)
        return real_wm_split(paths, val_fraction, seed)

    monkeypatch.setattr(study, "episode_split", study_spy)
    monkeypatch.setattr(wm, "episode_split", wm_spy)

    run_job(StudyJob("random_vit", 0), small_buffer, tmp_path, **JOB_KW)

    assert seen["training"], "train_world_model never split"
    assert seen["study"], "run_job never split"
    assert seen["study"] == seen["training"], (
        "the evaluation split at a different fraction from the one training "
        "used, so it scored on episodes the model saw")
    assert seen["study"] == [pytest.approx(VAL_FRACTION)]
    assert VAL_FRACTION == pytest.approx(0.2)


def test_the_record_names_the_held_out_episodes_not_the_training_ones(
    record, small_buffer
):
    """Exchanging the two comprehensions survived the whole suite: the only
    test on this field checked that the lists are stable across seeds, disjoint
    and sum to six, and all three stay true under a swap. Swapped, the record
    positively asserts the model was scored on the episodes it trained on and
    an auditor reading nine of them finds nothing wrong."""
    from mbfps.data.split import VAL_FRACTION, episode_split

    train_paths, val_paths = episode_split(
        small_buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    assert len(val_paths) != len(train_paths), (
        "a 3/3 split would make the swap invisible to this test")
    assert record["episodes"]["val"] == [p.name for p in val_paths]
    assert record["episodes"]["train"] == [p.name for p in train_paths]


def test_the_record_reports_the_training_history_it_was_given(
    tmp_path, small_buffer, monkeypatch
):
    """Four surviving mutations at once, all in fields nothing asserted on:
    the KL rate and the KL peak exchanged (`kl_rate < 0.5` is what tells the
    gate a run trained no dynamics prior, and the peak read as a rate means
    something else entirely), `loss_last20`'s `[-20:]` RESIZED in either
    direction (invisible at steps=3, a convergence number replaced by a
    training average -- or by a single noisy log point -- at 20,000), and
    `steps / seconds` inverted.

    The window is pinned in BOTH directions, which needs a non-constant tail.
    A tail of `[1.0] * 20` only caught WIDENING: narrowing to `[-1:]` or
    `[-2:]` has the same mean as `[-20:]` when every value in the tail is
    equal, so the assertion could not see it and both mutants survived. With
    twenty distinct values the four means are all different -- last 20 = 10.5,
    last two = 19.5, last one = 20.0, whole history = 40.33 -- so any resizing
    of the slice moves the number.
    """
    real_train = study.train_world_model
    tail = [float(i) for i in range(1, 21)]         # mean 10.5, last 20.0
    assert len({float(np.mean(tail)), float(np.mean(tail[-2:])),
                float(tail[-1])}) == 3, (
        "a constant tail makes the loss_last20 assertion blind to a narrowed "
        "window, which is the mutation this test exists to kill")

    def doctored(cfg, buffer, out_dir, log_every=100):
        history = real_train(cfg, buffer, out_dir=out_dir, log_every=log_every)
        history["steps"] = 1000
        history["seconds"] = 4.0
        history["loss"] = [100.0] * 10 + tail
        history["kl_rate_above_free_bits"] = 0.25
        history["kl_dyn_max"] = 7.5
        return history

    monkeypatch.setattr(study, "train_world_model", doctored)
    result = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path, **JOB_KW)

    assert result["kl_rate_above_free_bits"] == pytest.approx(0.25)
    assert result["kl_dyn_max"] == pytest.approx(7.5)
    assert result["steps_per_second"] == pytest.approx(250.0)
    assert result["loss_last20"] == pytest.approx(10.5)


def test_the_record_carries_the_probe_settings_it_measured_with(
    tmp_path, small_buffer, monkeypatch
):
    """`_probe_summary` could return `{}`, drop an r2 to None, or report each
    probe's ridge under the other's name, all with a green suite -- no test
    looked inside `record["probe"]` at all. It is the block that tells "the
    band is degenerate because the probe is noise" from "the model sits on its
    floor".

    The two RIDGE assertions need a fixture that can tell the two probes
    apart, and the honest fit cannot: measured here both probes select 1e7,
    the top of `probe.RIDGES`, because at steps=3 neither generalises and the
    maximum penalty always wins. While the two ridges are EQUAL, exchanging
    them in `_probe_summary` is a numerical no-op and the assertions pass on a
    swapped record. So the spy pins two DIFFERENT ridges on the real fits, and
    a guard below fails loudly if that ever collapses again.
    """
    real_fit = study.fit_probes
    seen: dict = {}

    def spy(*args, **kwargs):
        latent, embedding = real_fit(*args, **kwargs)
        # Only the "ridge" key is overridden. `w`, `mean` and `scale` are the
        # real fit and `apply_probe` never reads "ridge", so the rollout this
        # record describes is byte-for-byte the honest one; all that changes
        # is that the two reported penalties are now distinguishable.
        latent["ridge"], embedding["ridge"] = 1e1, 1e5
        seen["latent"], seen["embedding"] = latent, embedding
        return latent, embedding

    monkeypatch.setattr(study, "fit_probes", spy)
    result = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path, **JOB_KW)

    probe = result["probe"]
    assert set(probe) == {
        "latent_ridge", "latent_ridge_selected", "latent_selection_r2",
        "embedding_ridge", "embedding_ridge_selected", "embedding_selection_r2",
    }
    assert seen["latent"]["r2"] != seen["embedding"]["r2"], (
        "the two probes scored identically here, so a latent/embedding swap "
        "would be invisible to the two selection_r2 assertions below")
    assert seen["latent"]["ridge"] != seen["embedding"]["ridge"], (
        "the two probes report the SAME ridge here, so a latent/embedding "
        "swap would be invisible to the two ridge assertions below -- which "
        "is exactly the vacuous state this test was found in "
        f"(both at {seen['latent']['ridge']})")
    assert probe["latent_ridge"] == seen["latent"]["ridge"]
    assert probe["latent_selection_r2"] == seen["latent"]["r2"]
    assert probe["embedding_ridge"] == seen["embedding"]["ridge"]
    assert probe["embedding_selection_r2"] == seen["embedding"]["r2"]
    assert probe["latent_ridge_selected"] is True
    assert probe["embedding_ridge_selected"] is True


def test_probe_summary_says_whether_a_selection_r2_exists():
    """`fit_probe` only returns an `r2` when a selection split was used, so a
    bare `null` in the file meant EITHER "selection was skipped" or "the R^2
    was undefined" -- and an aggregation reading nine of these as floats turns
    both into NaN. The flag mirrors the gain block's own `ridge_selected`."""
    from mbfps.eval.study import _probe_summary

    summary = _probe_summary({"ridge": 1.0}, {"ridge": 10.0, "r2": 0.5})
    assert summary["latent_ridge_selected"] is False
    assert summary["latent_selection_r2"] is None
    assert summary["embedding_ridge_selected"] is True
    assert summary["embedding_selection_r2"] == 0.5


def test_the_record_carries_the_whole_reward_report(
    tmp_path, small_buffer, monkeypatch
):
    """`"reward": reward` reduced to `{"mse": reward["mse"]}` survived: the
    gate test only asserted `"reward" in record`. A bare MSE over this
    scenario's near-constant target reads as four-digit precision and means
    nothing without its baseline, its event count and its degeneracy flag."""
    real_reward = study.reward_accuracy
    seen: dict = {}

    def spy(*args, **kwargs):
        seen["out"] = real_reward(*args, **kwargs)
        return seen["out"]

    monkeypatch.setattr(study, "reward_accuracy", spy)
    result = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path, **JOB_KW)

    assert set(result["reward"]) == {
        "mse", "baseline_mse", "r2", "n_steps", "n_reward_events",
        "is_degenerate",
    }
    for key, value in seen["out"].items():
        got = result["reward"][key]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(got), key
        else:
            assert got == value, key


# --------------------------------------------------------------------------
# reward-accuracy semantics
# --------------------------------------------------------------------------

def test_degeneracy_is_a_property_of_the_target_not_the_prediction(record):
    """Computing the modal share over the PREDICTIONS survived: every existing
    case passes `predicted == true`, or a constant prediction against a
    degenerate target, so the two agree by accident. The dangerous direction is
    a model that collapses to a constant -- it would report is_degenerate on
    all nine cells and a reader would blame `my_way_home`'s reward rather than
    the arm's reward head, which is the confusion the flag exists to prevent."""
    on_a_constant_target = _summarise_reward(
        np.arange(50, dtype=float), np.zeros(50))
    assert on_a_constant_target["is_degenerate"] is True

    on_a_varied_target = _summarise_reward(
        np.zeros(50), np.arange(50, dtype=float))
    assert on_a_varied_target["is_degenerate"] is False

    assert isinstance(record["reward"]["is_degenerate"], bool)


def test_the_degeneracy_threshold_is_bracketed_on_both_sides():
    """0.99 was pinned only from above: the three existing cases sit at modal
    shares 1.0, 0.998 and 0.02, so ANY threshold in (0.02, 0.998] classified
    them identically and `0.99 -> 0.50` passed. Lowered, every arm is flagged
    degenerate and criterion 3 is suppressed across all nine cells."""
    assert study.DEGENERATE_REWARD_FRACTION == 0.99

    under = np.zeros(1000)
    under[:11] = np.arange(1, 12, dtype=float)          # modal share 0.989
    assert _summarise_reward(np.zeros(1000), under)["is_degenerate"] is False

    over = np.zeros(1000)
    over[:9] = np.arange(1, 10, dtype=float)            # modal share 0.991
    assert _summarise_reward(np.zeros(1000), over)["is_degenerate"] is True


def test_the_reward_rounding_resolves_a_fine_grained_target():
    """`np.round(true, 6)` feeds both the modal share and the event count, and
    `-> np.round(true, 0)` survived because both existing targets are on an
    integer scale. `my_way_home`'s living penalty is -0.0004: at 0 decimals
    every step collapses onto the modal value and the goal events vanish."""
    fine = np.arange(51, dtype=float) * 1e-4
    out = _summarise_reward(fine.copy(), fine)
    assert out["is_degenerate"] is False, (
        "a 1e-4 grid must not be rounded into a single constant value")
    assert out["n_reward_events"] == 50


def test_reward_accuracy_pairs_each_action_with_the_frame_it_led_to(
    small_buffer, monkeypatch
):
    """The third hand-rolled copy of `WorldModel.embed`'s action-time
    convention, and the only one with no test. `[:, 1:] -> [:, :-1]` survived:
    both slices have length T so the shape guard cannot see it, and a value
    pin cannot either -- measured, the two differ in the 6th significant digit
    of the MSE on this fixture and are bit-identical for the `cnn` arm. So
    compare the embeddings actually handed to `observe` against `model.embed`,
    which is where the convention is defined."""
    import torch

    from mbfps.data.episode import load_episode
    from mbfps.eval.rollout import source_for
    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config("random_vit", device="cpu", seed=0)
    model = WorldModel(cfg)
    model.eval()
    backbone = encoder_backbone(cfg.encoder)
    path = small_buffer.episode_paths()[0]

    captured: dict = {}
    real_observe = model.rssm.observe

    def spy(embeddings, actions):
        captured["embeddings"] = embeddings.detach().clone()
        captured["actions"] = actions.detach().clone()
        return real_observe(embeddings, actions)

    monkeypatch.setattr(model.rssm, "observe", spy)
    reward_accuracy(model, [path], backbone, torch.device("cpu"), limit=1)

    episode = load_episode(path)
    source = torch.as_tensor(source_for(model, path, episode, backbone))
    with torch.no_grad():
        window = {"obs": source.unsqueeze(0), "features": source.unsqueeze(0)}
        causal = model.embed(window)                       # embeddings[:, 1:]
        acausal = model.encoder(source).unsqueeze(0)[:, :-1]

    assert not torch.equal(causal, acausal), (
        "this episode's consecutive frames encode identically, so it cannot "
        "tell the causal slice from the acausal one and the assertion below "
        "would pass under either")
    assert torch.equal(captured["embeddings"], causal), (
        "reward_accuracy paired each action with the frame it was taken FROM, "
        "not the frame it led to -- an acausal posterior, and the reward "
        "accuracy of all nine cells measured against the wrong frame")
    assert captured["actions"].shape[1] == causal.shape[1]


# --------------------------------------------------------------------------
# the sanitiser's silent-corruption corners
# --------------------------------------------------------------------------

def test_a_dotted_record_key_is_refused_rather_than_aliasing_a_field(tmp_path):
    """The non-finite map addresses fields by dotted path, so a literal key
    `"a.b"` beside a real `{"a": {"b": ...}}` aliases it: measured, the real
    1.0 came back as NaN and the actual NaN stayed None, with nothing raised.
    No field is shaped like this today; nine JSON files are the whole study
    output, so the corner raises rather than corrupting a neighbour."""
    with pytest.raises(ValueError, match="contains a"):
        write_record(tmp_path / "r.json", {"a.b": float("nan"), "a": {"b": 1.0}})


def test_a_record_that_already_owns_the_map_key_is_refused(tmp_path):
    """`write_record({"nonfinite": {...}})` used to return `{"nonfinite": {}}`
    -- the caller's data discarded in silence."""
    with pytest.raises(ValueError, match=NONFINITE_KEY):
        write_record(tmp_path / "r.json", {NONFINITE_KEY: {"my": "data"}, "x": 1.0})


def test_a_zero_dimensional_numpy_value_is_written_as_a_scalar(tmp_path):
    """`np.ndarray` was matched by the list branch before the scalar branches,
    so a 0-d array raised `iteration over a 0-d array` -- a write-time crash
    after the GPU hours rather than a value coerced."""
    written = write_record(tmp_path / "r.json", {
        "z": np.array(3.5), "n": np.array(np.nan), "i": np.array(7),
    })
    assert written["z"] == 3.5 and isinstance(written["z"], float)
    assert written["i"] == 7 and isinstance(written["i"], int)
    assert written["n"] is None
    assert written[NONFINITE_KEY]["n"] == "nan"
    assert strict_loads((tmp_path / "r.json").read_text()) == written

