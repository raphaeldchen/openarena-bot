"""One study job: train, evaluate, emit a record the aggregation can read.

Nine of these records are the entire output of a 30-GPU-hour study, so every
test here is about a field that is silently wrong rather than absent.
"""

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
    """One completed job. ~2 s, so the assertions below share it."""
    return run_job(StudyJob("random_vit", 0), small_buffer, tmp_path, **JOB_KW)


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


def test_the_record_is_exactly_what_was_written(record, tmp_path):
    """The returned record IS the file. A caller comparing two jobs must be
    comparing what the aggregation will actually read."""
    path = job_record_path(tmp_path, StudyJob("random_vit", 0))
    assert path.is_file()
    assert strict_loads(path.read_text()) == record


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

def test_a_degenerate_record_is_written_as_valid_json(record, tmp_path):
    """`gap_final` is NaN by `metric_summary`'s contract whenever the band is
    non-positive, and this fixture's band IS non-positive. `json.dump` writes a
    bare `NaN` token for it, which is not JSON."""
    path = job_record_path(tmp_path, StudyJob("random_vit", 0))
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    strict_loads(text)      # raises via _reject if a bare token survived
    assert record[NONFINITE_KEY], "this fixture's band is degenerate by design"
    assert record["angle"]["gap_final"] is None
    assert record[NONFINITE_KEY]["angle.gap_final"] == "nan"


def test_an_undefined_gap_is_never_read_back_as_zero(record, tmp_path):
    """The distinction the policy exists to keep. NaN gap_final means the band
    was non-positive so the ratio is undefined; 0.0 means the model closed none
    of a real band. Averaging the first as the second across nine cells pulls
    the aggregate toward "no better than persistence" using cells that measured
    nothing at all."""
    restored = load_record(job_record_path(tmp_path, StudyJob("random_vit", 0)))
    for dotted in record[NONFINITE_KEY]:
        section, field = dotted.split(".", 1)
        assert record[section][field] is None
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
    """The whole study rests on this: same arm, same seed, same numbers."""
    first = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "a", **JOB_KW)
    second = run_job(StudyJob("random_vit", 0), small_buffer, tmp_path / "b", **JOB_KW)
    assert first["position"] == second["position"]
    assert first["angle"] == second["angle"]
    assert first["curves"] == second["curves"]
    assert first["filtering"] == second["filtering"]
    assert first["reward"] == second["reward"]


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
    distribution mismatch worth ~25 map units of position error."""
    seen: dict[str, dict] = {}

    def spy(name):
        real = getattr(study, name)

        def wrapper(*args, **kwargs):
            seen[name] = kwargs
            return real(*args, **kwargs)

        monkeypatch.setattr(study, name, wrapper)

    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain", "episode_split"):
        spy(name)

    run_job(StudyJob("random_vit", 2), small_buffer, tmp_path, **JOB_KW)

    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain"):
        assert name in seen, f"{name} was never called"
        assert seen[name]["seed"] == 2, f"{name} did not get the job's seed"
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
