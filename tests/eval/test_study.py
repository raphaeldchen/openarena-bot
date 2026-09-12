"""One study job: train, evaluate, emit a record the aggregation can read.

Nine of these records are the entire output of a 30-GPU-hour study, so every
test here is about a field that is silently wrong rather than absent.
"""

import inspect
import json
import math
import time
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

# ---------------------------------------------------------------------------
# The fixture's parameters, PAIRWISE DISTINCT BY CONSTRUCTION.
#
# Four mutations survived one round of review and they were not four bugs:
# they were one. Two of the values below used to coincide -- `steps == horizon
# == 3`, `job.seed == SPLIT_SEED == 0`, `job.seed == context == 2` -- so an
# assertion comparing them read the SAME NUMBER on both sides and could not
# fail, however carefully it was written. Exchanging the two quantities in
# `study.py` was then a numerical no-op and the suite stayed green.
#
# So the values are chosen once, here, so that no two of them are equal, and
# `test_the_fixture_parameters_are_pairwise_distinct_so_no_assertion_is_vacuous`
# below asserts that property itself. Reintroducing a collision fails THAT
# test loudly instead of silently hollowing out everything downstream.
# ---------------------------------------------------------------------------
JOB_ARM = "random_vit"      # the arm the shared `record` fixture runs at
OTHER_ARM = "pixel_ae"      # a second arm run below, so `"arm": job.arm` is
                            # not satisfied by hardcoding the first one
THIRD_ARM = "frozen_ssl"    # THE ONLY ARM WHOSE NAME IS NOT ITS BACKBONE'S.
                            # `random_vit` reads the `random_vit` cache and
                            # `pixel_ae` reads the `pixel_ae` cache, so on
                            # those two arms alone `backbone =
                            # encoder_backbone(cfg.encoder) -> backbone =
                            # job.arm` is a numerical no-op. This arm reads
                            # `dinov2`, and running it here is what makes
                            # that mutation visible at all.
THIRD_ARM_BACKBONE = "dinov2"
JOB_SEED = 1                # NOT SPLIT_SEED (0): `"split_seed": job.seed`
                            # is indistinguishable from the truth at seed 0
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
JOB = StudyJob(JOB_ARM, JOB_SEED)

#: Every scalar the record echoes or forwards, by the name it is known by.
#: All six must differ; see the invariant test.
PAIRWISE_DISTINCT_PARAMETERS = {
    "steps": JOB_KW["steps"],
    "seq_len": JOB_KW["seq_len"],
    "context": JOB_KW["context"],
    "horizon": JOB_KW["horizon"],
    "job.seed": JOB_SEED,
    "SPLIT_SEED": SPLIT_SEED,
}

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
    return run_job(JOB, small_buffer, tmp_path, **JOB_KW)


@pytest.fixture
def written(record):
    """What the file holds for the same job: sanitised, with the token map."""
    return to_json_record(record)


# --------------------------------------------------------------------------
# the fixture itself: the guard that kills the SPECIES, not the instance
# --------------------------------------------------------------------------

def test_the_fixture_parameters_are_pairwise_distinct_so_no_assertion_is_vacuous():
    """Every assertion in this module that says "field X carries quantity Y"
    can only fail if Y differs from the other quantities X might have been
    filled from. That is a property of the FIXTURE, not of the assertion, and
    it has been violated four times on this one file:

      * `steps == horizon == 3`, so `evaluate_rollout(horizon=horizon)` ->
        `horizon=steps` changed nothing and the curve-length assertions passed;
      * `job.seed == SPLIT_SEED == 0`, so `"split_seed": SPLIT_SEED` ->
        `job.seed` changed nothing -- while in the real study it would give
        each seed a DIFFERENT held-out set and the nine cells would stop being
        comparable;
      * `job.seed == context == 2`, so exchanging the `seed=` and `context=`
        keyword values at the evaluation call sites was invisible;
      * both probes selecting the same ridge, so reporting each under the
        other's name was invisible.

    Each was fixed as an instance and the species came back. This test is the
    guard on the species: if any two of the parameters below are ever made
    equal again, THIS fails loudly and by name, instead of quietly turning the
    assertions that depend on them into tautologies.
    """
    from mbfps.utils.config import ARMS

    collisions = {
        value: sorted(name for name, other in PAIRWISE_DISTINCT_PARAMETERS.items()
                      if other == value)
        for value in set(PAIRWISE_DISTINCT_PARAMETERS.values())
        if list(PAIRWISE_DISTINCT_PARAMETERS.values()).count(value) > 1
    }
    assert not collisions, (
        "these fixture parameters collide, so every assertion that tells one "
        "of them from the other is now a tautology and a mutation exchanging "
        f"them survives the suite: {collisions}")
    assert len(set(PAIRWISE_DISTINCT_PARAMETERS.values())) == 6, (
        "all six parameters must still be listed and compared")

    # The arm is categorical rather than numeric, and has the same failure
    # mode: while every `run_job` call in the file uses one arm, `"arm":
    # job.arm -> "arm": "random_vit"` mislabels all nine records and passes.
    assert JOB_ARM != OTHER_ARM
    assert {JOB_ARM, OTHER_ARM} <= set(ARMS)

    # THE SAME SPECIES ONE LEVEL DOWN, and the reason THIRD_ARM exists.
    #
    # `run_job` derives the feature cache from the CONFIG
    # (`encoder_backbone(cfg.encoder)`), never from the arm name, and
    # `mbfps/models/encoders.py` says why: "the arm name and the backbone name
    # are deliberately not assumed equal". But on the two arms this file used
    # to run, they ARE equal or irrelevant -- `random_vit`'s cache is called
    # `random_vit` and `pixel_ae`'s is called `pixel_ae` -- so `backbone =
    # job.arm` was a numerical no-op and survived the whole suite. In the study
    # it sends `frozen_ssl` to a cache that does not exist and kills all three
    # of that arm's seeds at `FileNotFoundError` hours in.
    #
    # So all three arms are run below, and the coincidence that hid the
    # mutation is asserted here by name rather than left to be rediscovered.
    from mbfps.models.encoders import encoder_backbone
    from mbfps.utils.config import get_config

    assert len({JOB_ARM, OTHER_ARM, THIRD_ARM}) == 3
    assert {JOB_ARM, OTHER_ARM, THIRD_ARM} == set(ARMS), (
        "every arm the study runs must be exercised here; the one that is "
        "missing is the one whose backbone mapping is unguarded")
    backbones = {
        arm: encoder_backbone(get_config(arm, device="cpu", seed=JOB_SEED).encoder)
        for arm in ARMS
    }
    assert backbones[THIRD_ARM] == THIRD_ARM_BACKBONE, backbones
    assert backbones[THIRD_ARM] != THIRD_ARM, (
        f"{THIRD_ARM!r} no longer reads a cache under a different name, so "
        "`backbone = job.arm` is invisible everywhere in this suite again")
    assert backbones[JOB_ARM] == JOB_ARM, (
        "this coincidence is the whole reason the third arm is needed: on "
        f"{JOB_ARM!r} the arm name and the backbone name are the same string")
    assert backbones[OTHER_ARM] == OTHER_ARM, (
        f"{OTHER_ARM!r}'s arm name and backbone name are the same string too, "
        "so it cannot discriminate either")


# --------------------------------------------------------------------------
# record path
# --------------------------------------------------------------------------

def test_job_record_path_is_unique_per_arm_and_seed(tmp_path):
    a = job_record_path(tmp_path, StudyJob("pixel_ae", 0))
    b = job_record_path(tmp_path, StudyJob("pixel_ae", 1))
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
    assert record["arm"] == JOB_ARM
    assert record["seed"] == JOB_SEED


def test_the_record_is_labelled_with_the_arm_the_job_asked_for(
    tmp_path, small_buffer
):
    """`"arm": job.arm -> "arm": "random_vit"` survived the whole suite for
    one reason only: every `run_job` call in this file ran `random_vit`, so
    the hardcoded string was always the right answer. On the study box it
    MISLABELS ALL NINE RECORDS -- Task 6 groups by this field, and three arms'
    results would be read as one arm's, with no way to tell after the fact.

    So this runs a SECOND arm. The assertion below can only pass if the label
    followed the job."""
    assert OTHER_ARM != JOB_ARM, (
        "both jobs run the same arm, so a hardcoded arm label passes")

    other = run_job(StudyJob(OTHER_ARM, JOB_SEED), small_buffer, tmp_path,
                    **JOB_KW)

    assert other["arm"] == OTHER_ARM
    assert other["seed"] == JOB_SEED
    # The checkpoint is named after the arm too, and the record is written
    # beside it: a record labelled with the wrong arm would still be found.
    assert OTHER_ARM in job_record_path(tmp_path,
                                        StudyJob(OTHER_ARM, JOB_SEED)).name


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
    path = job_record_path(tmp_path, JOB)
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
    path = job_record_path(tmp_path, JOB)
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
    restored = load_record(job_record_path(tmp_path, JOB))
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
    first = run_job(JOB, small_buffer, tmp_path / "a", **JOB_KW)
    second = run_job(JOB, small_buffer, tmp_path / "b", **JOB_KW)
    a, b = to_json_record(first), to_json_record(second)
    assert a[NONFINITE_KEY] == b[NONFINITE_KEY]
    assert a[NONFINITE_KEY], "this fixture's band is degenerate; the map must be live"
    for section in ("position", "angle", "curves", "filtering", "reward", "probe"):
        assert a[section] == b[section], section


def test_different_seeds_give_different_results(tmp_path, small_buffer):
    other_seed = JOB_SEED + 1
    a = run_job(JOB, small_buffer, tmp_path / "a", **JOB_KW)
    b = run_job(StudyJob(JOB_ARM, other_seed), small_buffer, tmp_path / "b",
                **JOB_KW)
    assert a["curves"]["rssm_position"] != b["curves"]["rssm_position"]
    assert a["seed"] == JOB_SEED and b["seed"] == other_seed


def test_the_held_out_episodes_do_not_move_with_the_job_seed(tmp_path, small_buffer):
    """`train_world_model` splits at SPLIT_SEED. Splitting the evaluation at
    the JOB's seed would hand each seed a held-out set its own training run
    trained on, and the nine cells would no longer be scored on the same
    episodes.

    Both jobs run at a seed that is NOT SPLIT_SEED. At the job seed 0 this
    test used to run at, `"split_seed": SPLIT_SEED -> job.seed` reported the
    same 0 either way and the last assertion could not fail -- while in the
    study it would silently give each of the three seeds a different held-out
    set."""
    seeds = (JOB_SEED, JOB_SEED + 1)
    assert SPLIT_SEED not in seeds, (
        "a job seed equal to SPLIT_SEED makes `split_seed: job.seed` "
        "indistinguishable from the truth")
    a = run_job(StudyJob(JOB_ARM, seeds[0]), small_buffer, tmp_path / "a", **JOB_KW)
    b = run_job(StudyJob(JOB_ARM, seeds[1]), small_buffer, tmp_path / "b", **JOB_KW)
    assert a["episodes"]["val"] == b["episodes"]["val"] != []
    assert a["episodes"]["train"] == b["episodes"]["train"] != []
    assert len(a["episodes"]["val"]) + len(a["episodes"]["train"]) == 6
    assert not set(a["episodes"]["val"]) & set(a["episodes"]["train"])
    assert a["split_seed"] == b["split_seed"] == SPLIT_SEED == 0
    assert (a["seed"], b["seed"]) == seeds


def test_every_evaluation_runs_at_the_jobs_seed_and_the_jobs_window(
    tmp_path, small_buffer, monkeypatch
):
    """Each of these takes `seed`, `context` and `horizon`, and each of them
    silently accepts its own defaults. A rollout evaluated at seed 0 while the
    model trained at seed 2 is not that cell's number, and a probe fit at a
    different filtering depth from the rollout it is applied to is a
    distribution mismatch worth ~25 map units of position error.

    The job seed is 7, distinct from EVERY other parameter this job runs at:
    at seed 2 the `seed=` and `context=` assertions below both read 2 and
    exchanging the two keyword VALUES at the call sites was a no-op this test
    could not see -- the same coincidence-of-fixture-values that hid the
    probe-ridge swap. `steps` is in the guard too: `horizon=horizon ->
    horizon=steps` in the `evaluate_rollout` call is invisible while the two
    are equal, which is exactly how that mutation survived."""
    seed = 7
    seen: dict[str, dict] = {}
    assert seed not in PAIRWISE_DISTINCT_PARAMETERS.values(), (
        "the job seed must not coincide with steps, seq_len, context, horizon "
        "or SPLIT_SEED, or an exchange of the matching pair of keyword values "
        f"at the call sites is invisible here: {PAIRWISE_DISTINCT_PARAMETERS}")
    assert JOB_KW["steps"] != JOB_KW["horizon"], (
        "`evaluate_rollout(horizon=steps)` reads the same number as "
        "`horizon=horizon` while these two coincide")

    def spy(name):
        real = getattr(study, name)

        def wrapper(*args, **kwargs):
            seen[name] = kwargs
            return real(*args, **kwargs)

        monkeypatch.setattr(study, name, wrapper)

    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain", "episode_split"):
        spy(name)

    run_job(StudyJob(JOB_ARM, seed), small_buffer, tmp_path, **JOB_KW)

    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain"):
        assert name in seen, f"{name} was never called"
        assert seen[name]["seed"] == seed, f"{name} did not get the job's seed"
    for name in ("fit_probes", "evaluate_rollout", "filtering_report",
                 "filtering_gain"):
        assert seen[name]["context"] == JOB_KW["context"], name
        assert seen[name]["horizon"] == JOB_KW["horizon"], name
    # The split is the one exception, and deliberately so.
    assert seen["episode_split"]["seed"] == SPLIT_SEED


def test_each_evaluation_gets_the_episode_set_it_is_supposed_to_get(
    tmp_path, small_buffer, monkeypatch
):
    """WHICH episodes reach each call had NO assertion at all, so exchanging
    `train_paths` and `val_paths` in the `filtering_gain` call survived the
    whole suite in silence -- it is not even a vacuous assertion, it is a
    missing one.

    Exchanged, the filtering probe is FIT on the episodes it then SCORES. That
    exact leak was measured in this project taking R^2 from -0.246 to 0.9997,
    so gate criterion 4's companion diagnostic would report a near-perfect
    number in all nine records and the gate would read as passed.

    `filtering_report` -- criterion 4 itself -- takes the same two arguments in
    the same order at the neighbouring call site and had the same exposure, so
    it is pinned here too, along with the three single-set calls: the probe is
    fit on TRAIN, the rollout and the reward are scored on VAL."""
    from mbfps.data.split import VAL_FRACTION, episode_split

    train_paths, val_paths = episode_split(
        small_buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    train, val = [p.name for p in train_paths], [p.name for p in val_paths]
    assert train and val and train != val, (
        "train and val name the same episodes here, so exchanging the two "
        "arguments would be invisible to every assertion below")

    seen: dict[str, tuple] = {}

    def spy(name):
        real = getattr(study, name)

        def wrapper(*args, **kwargs):
            seen[name] = args
            return real(*args, **kwargs)

        monkeypatch.setattr(study, name, wrapper)

    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain"):
        spy(name)

    run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    def names(name, position):
        assert name in seen, f"{name} was never called"
        return [Path(p).name for p in seen[name][position]]

    for name in ("filtering_report", "filtering_gain"):
        assert names(name, 1) == train, (
            f"{name} was FIT on the held-out episodes it then scores -- the "
            "leak that takes this diagnostic's R^2 from -0.246 to 0.9997")
        assert names(name, 2) == val, (
            f"{name} scored on the episodes the probe was fit on")
    assert names("fit_probes", 1) == train, (
        "the probe was fit on the held-out episodes")
    assert names("evaluate_rollout", 1) == val, (
        "the rollout was scored on episodes the model trained on")
    assert names("reward_accuracy", 1) == val, (
        "reward accuracy was scored on episodes the model trained on")


def _train_saving_a_checkpoint_labelled(arm, seed):
    """A stand-in trainer whose checkpoint is LABELLED for another cell.

    The file NAME is still this job's -- `run_job` finds it by name -- so the
    two labels inside are the only thing the guard has to go on. The weights
    are a real, loadable state dict for the arm actually being run, which is
    the point: a checkpoint from the wrong SEED of the same arm loads cleanly,
    so nothing downstream would ever notice the misattribution. With an empty
    state dict an escaped mutant would die at `load_state_dict` instead, and
    the test would pass for a reason that has nothing to do with the guard.
    """
    def fake_train(cfg, buffer, out_dir, log_every=100):
        import torch

        from mbfps.training.world_model import WorldModel

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"arm": arm, "seed": seed, "state_dict": WorldModel(cfg).state_dict()},
            out_dir / f"world_model_{cfg.arm}_seed{cfg.train.seed}.pt",
        )
        return {"steps": 3, "seconds": 1.0, "loss": [0.0],
                "kl_rate_above_free_bits": 0.0, "kl_dyn_max": 0.0}

    return fake_train


def test_a_checkpoint_from_another_job_is_refused(tmp_path, small_buffer, monkeypatch):
    """The record is written beside the checkpoint it describes. Loading one
    from a different arm or seed would attribute another cell's weights to this
    cell, and the state dict would load cleanly because the architecture is
    shared across seeds.

    This case is wrong in BOTH fields, so it cannot tell the two halves of the
    compound guard apart: either half alone still raises. Dropping either
    disjunct survived it. The two tests below are the ones that kill those
    mutants; this one stays because a wholly foreign checkpoint is still worth
    refusing, not because it discriminates.
    """
    monkeypatch.setattr(study, "train_world_model",
                        _train_saving_a_checkpoint_labelled(OTHER_ARM, 99))
    with pytest.raises(ValueError, match="not this job's"):
        run_job(JOB, small_buffer, tmp_path, **JOB_KW)


def test_a_checkpoint_wrong_in_the_arm_alone_is_refused(
    tmp_path, small_buffer, monkeypatch
):
    """Half of the compound guard, exercised ALONE.

    `checkpoint["arm"] != job.arm or checkpoint["seed"] != job.seed` was only
    ever tested with a checkpoint wrong in both fields (`pixel_ae`/99 against this
    job's `random_vit`/1), so both disjuncts were true at once and one
    `pytest.raises` could not say which had fired. Deleting either half left
    the suite green.

    Here the SEED matches and only the arm is wrong, so the seed disjunct is
    False and the arm disjunct is the only thing that can raise: under
    `if checkpoint.get("seed") != job.seed:` alone nothing raises, `run_job`
    evaluates another arm's weights as this cell, and this test goes red on
    DID NOT RAISE.
    """
    assert OTHER_ARM != JOB_ARM, "the arm must actually be wrong"
    monkeypatch.setattr(study, "train_world_model",
                        _train_saving_a_checkpoint_labelled(OTHER_ARM, JOB_SEED))
    with pytest.raises(ValueError, match="not this job's") as excinfo:
        run_job(JOB, small_buffer, tmp_path, **JOB_KW)
    assert f"is arm={OTHER_ARM!r} seed={JOB_SEED!r}" in str(excinfo.value), (
        "the message must name the checkpoint's own labels, so the reader can "
        "see WHICH half of the guard fired")


def test_a_checkpoint_wrong_in_the_seed_alone_is_refused(
    tmp_path, small_buffer, monkeypatch
):
    """The other half, exercised alone -- and the more dangerous one.

    The arm matches and only the seed is wrong, so the arm disjunct is False.
    Under `if checkpoint.get("arm") != job.arm:` alone nothing raises. That is
    precisely the misattribution the guard exists for: the architecture is
    shared across seeds, so a seed-2 checkpoint loads into the seed-1 cell
    without a murmur, and two of the nine records would describe the same
    trained model under different seed labels.
    """
    other_seed = JOB_SEED + 1
    assert other_seed != JOB_SEED, "the seed must actually be wrong"
    monkeypatch.setattr(study, "train_world_model",
                        _train_saving_a_checkpoint_labelled(JOB_ARM, other_seed))
    with pytest.raises(ValueError, match="not this job's") as excinfo:
        run_job(JOB, small_buffer, tmp_path, **JOB_KW)
    assert f"is arm={JOB_ARM!r} seed={other_seed!r}" in str(excinfo.value), (
        "the message must name the checkpoint's own labels, so the reader can "
        "see WHICH half of the guard fired")


def test_a_checkpoint_that_does_not_fit_the_model_is_refused(
    tmp_path, small_buffer, monkeypatch
):
    """`load_state_dict(..., strict=False)` -- the partial form of the defect
    the value comparison above closes.

    That comparison only walks the keys the checkpoint and the model SHARE, so
    it is silent about a key the checkpoint does not carry. Under
    `strict=False` a checkpoint whose keys no longer match the architecture --
    a renamed submodule, an arch change between the training run and the
    evaluation -- loads whatever it can and leaves the rest RANDOMLY
    INITIALISED, with nothing raised and the arm/seed guard still swearing the
    right checkpoint was used. The record then describes a half-trained
    network.

    The checkpoint here is labelled for this exact job and its weights are a
    real state dict for this arm, with ONE tensor removed. Under `strict=True`
    that raises; under `strict=False` `run_job` completes and this test goes
    red on DID NOT RAISE.
    """
    dropped: dict = {}

    def fake_train(cfg, buffer, out_dir, log_every=100):
        import torch

        from mbfps.training.world_model import WorldModel

        out_dir.mkdir(parents=True, exist_ok=True)
        state = WorldModel(cfg).state_dict()
        dropped["key"] = sorted(state)[0]
        del state[dropped["key"]]
        torch.save(
            {"arm": cfg.arm, "seed": cfg.train.seed, "state_dict": state},
            out_dir / f"world_model_{cfg.arm}_seed{cfg.train.seed}.pt",
        )
        return {"steps": 3, "seconds": 1.0, "loss": [0.0],
                "kl_rate_above_free_bits": 0.0, "kl_dyn_max": 0.0}

    monkeypatch.setattr(study, "train_world_model", fake_train)

    with pytest.raises(RuntimeError, match="Missing key") as excinfo:
        run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    # Check-it-can-fail: the removed tensor is a real, learned parameter of
    # this architecture, so under `strict=False` it would sit at its fresh
    # initialisation through every evaluation in the record.
    assert dropped["key"], "the stand-in trainer never saved a checkpoint"
    assert dropped["key"] in str(excinfo.value), (
        "the loader must name the tensor the checkpoint could not supply")


# ---------------------------------------------------------------------------
# Review follow-up: the twelve confirmed findings of task 4's review. The
# record itself was right; its VALUES, its DEFAULTS and its returned FORM were
# unguarded, and a mutation of each survived the whole suite. Each test below
# names the mutation it kills. See .superpowers/sdd/task-4-report.md.
# ---------------------------------------------------------------------------


def test_the_returned_record_does_not_crash_the_studys_own_driver(record):
    """The same format specs the driver's per-job print uses, on a degenerate cell.

    NOT the driver's block verbatim, and this docstring used to claim it was.
    The field names below (`steps/s=`, `kl_rate=`, `position_gap_final=`) are a
    hand-written paraphrase of `job_summary`'s (`steps_per_second=`,
    `kl_rate_above_free_bits=`, `gap_final=`), and nothing here calls
    `job_summary` at all -- so a "verbatim" claim was exactly the kind that
    stops someone adding the end-to-end guard that was in fact missing. The
    driver's own block is now pinned character for character by
    `test_job_summary_renders_every_field_from_the_place_it_claims_to`, and
    that main prints it by `test_main_prints_the_per_cell_summary_block`, both
    in `tests/eval/test_run_study.py`. What THIS test is for is unchanged and
    is `run_job`'s side of the contract: the record it returns must hold live
    floats, so that a degenerate cell formats rather than raising.

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

    Every value it reports is distinct from every other, by construction --
    that is what `PAIRWISE_DISTINCT_PARAMETERS` and its invariant test are
    for. While `steps` and `horizon` were both 3, exchanging those two fields
    was a numerical no-op and all four assertions below passed on a record
    that misreported both. The same holds for `split_seed`: at a job seed of
    0, `"split_seed": SPLIT_SEED -> job.seed` reported the right number by
    accident."""
    assert len(set(PAIRWISE_DISTINCT_PARAMETERS.values())) == 6, (
        "two of these coincide, so exchanging the matching pair of record "
        "fields would be invisible to this test")
    assert JOB_SEED != SPLIT_SEED, (
        "`split_seed: job.seed` reads the same number as the truth here")

    record = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    assert record["steps"] == JOB_KW["steps"]
    assert record["seq_len"] == JOB_KW["seq_len"]
    assert record["context"] == JOB_KW["context"]
    assert record["horizon"] == JOB_KW["horizon"]
    assert record["seed"] == JOB_SEED
    # The split seed is a CONSTANT, not the job's seed: it is what makes all
    # nine cells hold out the same episodes. Under `job.seed` each seed gets
    # its own split and the nine cells stop being comparable.
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
    stays a CPU test.

    The evaluation model's placement is checked on the ARGUMENT that reached
    `.to()`, not on the resulting parameters' device. Reading the parameters
    made the guard MACHINE-DEPENDENT and therefore useless here:
    `.to(torch_device) -> .to(torch.device("cpu"))` survived, because
    `get_device(prefer="cuda")` already IS cpu on a CUDA-less machine, so the
    tensors were where they were expected either way. That guard could only
    start guarding on the GPU box -- the one place nobody is watching it. The
    object `get_device` returned is a distinct object from any freshly built
    `torch.device`, so requiring THAT object at the call site fails on a
    hardcoded device on any machine, CUDA or not."""
    import torch

    real_get_config = study.get_config
    real_get_device = study.get_device
    real_world_model = study.WorldModel
    seen: dict = {}
    models: list = []
    forwarded: dict = {}

    # Where the device sits in each evaluation call. `evaluate_rollout` takes
    # it by keyword; the other four take it positionally.
    device_argument = {
        "fit_probes": 3,
        "reward_accuracy": 3,
        "filtering_report": 4,
        "filtering_gain": 4,
        "evaluate_rollout": "device",
    }

    def forward_spy(name, where):
        real = getattr(study, name)

        def wrapper(*args, **kwargs):
            forwarded[name] = (
                kwargs.get(where) if isinstance(where, str)
                else (args[where] if len(args) > where else kwargs.get("device"))
            )
            return real(*args, **kwargs)

        monkeypatch.setattr(study, name, wrapper)

    for name, where in device_argument.items():
        forward_spy(name, where)

    def config_spy(arm, **overrides):
        seen["cfg_device"] = overrides.get("device")
        return real_get_config(arm, **overrides)

    def device_spy(prefer="mps"):
        seen["prefer"] = prefer
        seen["device"] = real_get_device(prefer=prefer)
        return seen["device"]

    def model_spy(cfg):
        model = real_world_model(cfg)
        real_to = model.to

        def to_spy(*args, **kwargs):
            seen.setdefault("moved_to", []).append(
                args[0] if args else kwargs.get("device"))
            return real_to(*args, **kwargs)

        model.to = to_spy
        models.append(model)
        return model

    real_load = torch.load

    def load_spy(*args, **kwargs):
        seen.setdefault("loads", []).append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(study, "get_config", config_spy)
    monkeypatch.setattr(study, "get_device", device_spy)
    monkeypatch.setattr(study, "WorldModel", model_spy)
    monkeypatch.setattr(torch, "load", load_spy)

    run_job(JOB, small_buffer, tmp_path,
            **dict(JOB_KW, device="cuda"))

    assert seen["cfg_device"] == "cuda", (
        "the job's device never reached the training config, so train_world_model "
        "trained wherever TrainConfig defaults to")
    assert seen["prefer"] == "cuda", (
        "the job's device never reached get_device, so the evaluation ran on "
        "whatever get_device defaults to")
    assert models, "run_job never built the evaluation model"
    assert seen.get("moved_to"), "run_job never placed the evaluation model"
    # Check-it-can-fail: an equal-but-freshly-built device -- exactly what a
    # hardcoding mutation constructs -- is NOT the object `get_device`
    # returned, on this machine, where both of them are plain CPU.
    hardcoded = torch.device(seen["device"].type)
    assert hardcoded == seen["device"] and hardcoded is not seen["device"], (
        "this assertion cannot tell a hardcoded device from the requested "
        "one, so it would pass on a machine-independent regression")
    assert seen["moved_to"][0] is seen["device"], (
        "the evaluation model was moved to a device run_job built for itself "
        "rather than the one get_device returned for the job's --device; on a "
        "CUDA-less machine both are CPU and the model's own parameters cannot "
        f"show it (asked for {seen['device']}, got {seen['moved_to'][0]})")

    # The SAME species one line further down, and it survived here: the guard
    # above spies the argument reaching `.to()` at the model's construction,
    # but nothing looked at the `map_location=` of the `torch.load` two lines
    # below it, so `map_location=torch_device -> torch.device("cpu")` was
    # invisible. It is invisible NUMERICALLY too, on this machine, because
    # `get_device(prefer="cuda")` already is cpu here -- like the `.to()` bug
    # it could only start guarding on the GPU box, the one place nobody is
    # watching. So it is pinned the same way: by OBJECT IDENTITY against the
    # device `get_device` returned, which the `hardcoded is not seen["device"]`
    # check above has already shown a freshly-built device cannot satisfy.
    checkpoint_loads = [
        (args, kwargs) for args, kwargs in seen.get("loads", [])
        if Path(args[0] if args else kwargs["f"]).name.startswith("world_model_")
    ]
    assert len(checkpoint_loads) == 1, (
        "expected exactly one torch.load of the job's checkpoint, saw "
        f"{len(checkpoint_loads)}")
    load_kwargs = checkpoint_loads[0][1]
    assert load_kwargs.get("map_location") is seen["device"], (
        "the checkpoint was mapped onto a device run_job built for itself "
        "rather than the one get_device returned for the job's --device; on a "
        "CUDA-less machine both are CPU, so no tensor can show it "
        f"(asked for {seen['device']}, got {load_kwargs.get('map_location')})")

    # AND THE FIVE CALL SITES THAT FORWARD THE DEVICE ONWARD, which the two
    # checks above stopped short of. Every one of `fit_probes`,
    # `reward_accuracy`, `evaluate_rollout`, `filtering_report` and
    # `filtering_gain` is handed a device, and `torch_device ->
    # torch.device("cpu")` at any of them was invisible: `get_device(prefer=
    # "cuda")` already IS cpu on this machine, so no tensor and no number can
    # tell the two apart HERE. On the rented box it fits the probe, scores the
    # reward, runs the headline rollout or computes gate criterion 4 on CPU
    # tensors against a CUDA model -- a RuntimeError mid-study, or a silent
    # CPU evaluation, after the training hours are already paid. The identity
    # check above is what makes this bite on a CUDA-less machine too.
    assert set(forwarded) == set(device_argument), (
        f"not every evaluation was observed: {sorted(forwarded)}")
    for name, got in forwarded.items():
        assert got is seen["device"], (
            f"{name} was handed a device run_job built for itself rather than "
            "the one get_device returned for the job's --device; both are CPU "
            f"here, so only object identity can see it (asked for "
            f"{seen['device']}, got {got})")


def test_the_evaluated_model_is_the_trained_one_and_is_in_eval_mode(
    tmp_path, small_buffer, monkeypatch
):
    """NOTHING required the evaluated model to be the TRAINED model.

    `run_job` builds a fresh `WorldModel` at the eval site, loads the
    checkpoint, validates its arm and seed -- and then
    `model.load_state_dict(checkpoint["state_dict"])` could be deleted
    (`_ = checkpoint["state_dict"]`) and all 524 tests still passed. Under that
    mutation every number in all nine records -- probes, rollout, reward, both
    filtering diagnostics -- describes a RANDOMLY INITIALISED network, while
    the checkpoint is dutifully opened, checked against the job's arm and seed,
    and thrown away. The arm/seed guard would still swear the right checkpoint
    was used. That is the worst defect this file can carry: the study's only
    artifact becomes noise and nothing in it looks wrong.

    So the check is on the VALUES, not on a call: a spy on `load_state_dict`
    would be satisfied by a call whose effect is discarded. After `run_job` the
    evaluation model's own tensors must equal the checkpoint's.

    CHECK-IT-CAN-FAIL. A weights comparison is worthless if a fresh
    initialisation would satisfy it anyway -- and here the eval model is
    constructed from the same seeded config the trainer used, so its fresh
    weights are exactly what the mutant would leave in place. `moved` below is
    the set of tensors training actually changed, measured in this fixture, and
    the assertion is required to have a large one. Measured: 30 of 36 tensors
    differ (max |fresh - trained| = 0.087 on `encoder.bottleneck.bias`); the 6
    that do not are `rssm.prior_net.*`, the dynamics prior this tiny fixture
    leaves untrained -- which is exactly why a comparison on ONE arbitrarily
    chosen parameter could have been vacuous.

    `model.eval()` is pinned here too, and had no assertion at all: deleting it
    scores all nine cells with training-mode behaviour active, which is
    nondeterministic and not comparable across arms.

    ITS CHECK-IT-CAN-FAIL IS THE SUBTLE ONE, and the obvious assertion is
    vacuous. `model.training is False` AFTER `run_job` cannot fail no matter
    what this module does, because `evaluate_rollout` calls `model.eval()` for
    itself (`mbfps/eval/rollout.py:109`) on its way past -- deleting
    `study.py`'s own call leaves the model in eval mode by the time `run_job`
    returns, and the mutant survives the check. What it does NOT survive is
    being asked about the mode at the moment each consumer is CALLED:
    `fit_probes` and `reward_accuracy` run before `evaluate_rollout` and would
    be handed a model still in training mode. So the flag is sampled per call
    site, below, and the rollout's own self-defence is what proves the
    difference between the two forms of the assertion.
    """
    import torch

    real_world_model = study.WorldModel
    built: dict = {}
    mode_at_call: dict[str, bool] = {}

    def model_spy(cfg):
        model = real_world_model(cfg)
        built["fresh"] = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        built["model"] = model
        return model

    def mode_spy(name):
        real = getattr(study, name)

        def wrapper(*args, **kwargs):
            mode_at_call[name] = args[0].training
            return real(*args, **kwargs)

        monkeypatch.setattr(study, name, wrapper)

    monkeypatch.setattr(study, "WorldModel", model_spy)
    for name in ("fit_probes", "reward_accuracy", "evaluate_rollout",
                 "filtering_report", "filtering_gain"):
        mode_spy(name)

    run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    assert "model" in built, "run_job never built an evaluation model"
    # `nn.Module.to()` mutates in place and returns self, so this is the very
    # object every evaluation below was handed.
    model = built["model"]
    checkpoint = torch.load(
        tmp_path / f"world_model_{JOB_ARM}_seed{JOB_SEED}.pt",
        map_location="cpu", weights_only=True,
    )
    trained = checkpoint["state_dict"]
    fresh, evaluated = built["fresh"], model.state_dict()
    assert set(trained) == set(fresh) == set(evaluated) and trained, (
        "the checkpoint and the evaluation model do not even name the same "
        "tensors, so nothing below compares what it claims to")

    moved = sorted(k for k in trained if not torch.equal(fresh[k], trained[k]))
    assert len(moved) > len(trained) // 2, (
        "training barely moved this fixture's weights, so a freshly "
        "initialised model would satisfy the comparison below and it could "
        f"not fail: only {len(moved)} of {len(trained)} tensors differ")

    for key in trained:
        assert torch.equal(evaluated[key], trained[key]), (
            f"the evaluated model's {key!r} is not the checkpoint's -- the "
            "trained weights never reached the model that produced this "
            "record, so every number in it describes a different network")

    assert set(mode_at_call) == {"fit_probes", "reward_accuracy",
                                 "evaluate_rollout", "filtering_report",
                                 "filtering_gain"}, (
        f"not every evaluation was observed: {sorted(mode_at_call)}")
    for name, training in mode_at_call.items():
        assert training is False, (
            f"{name} was handed a model in TRAINING mode, so this cell was "
            "scored with training-time behaviour active -- nondeterministic, "
            "and not comparable with the other eight")
    # Check-it-can-fail, stated as an assertion so it cannot rot: the rollout
    # sets eval mode itself, so it is NOT among the call sites that can catch
    # a deleted `model.eval()`, and neither is the post-hoc flag below.
    from mbfps.eval import rollout as rollout_module

    assert "model.eval()" in inspect.getsource(rollout_module.evaluate_rollout), (
        "evaluate_rollout no longer sets eval mode itself; the note above "
        "about why the post-hoc check is vacuous needs revisiting")
    assert model.training is False, (
        "the evaluation model was left in training mode -- note this last "
        "check is the WEAK form and passes even with study.py's own eval() "
        "deleted; the per-call-site loop above is what guards the mutation")


def test_the_training_config_gets_the_budget_and_window_the_job_ran_at(
    tmp_path, small_buffer, monkeypatch
):
    """`get_config(..., steps=steps, seq_len=seq_len, ...)` had NO spy.

    Exchanging those two keyword values survived the whole suite. The record
    echoes `run_job`'s own PARAMETERS, never `cfg`, so `record["steps"] ==
    JOB_KW["steps"]` passes just as happily on a config that trained at the
    wrong budget. In the real study the mutant trains every cell at steps=64,
    seq_len=20000 -- a 312-times-shorter run on 312-times-longer sequences --
    while all nine records report 20000/64 and nothing else ever looks.

    The two values are pairwise-distinct fixture parameters (5 and 4), asserted
    below, so the exchange genuinely moves both assertions. The last two
    assertions are the ones that close the loop the record leaves open: the
    reported numbers must be the CONFIG's, not merely the arguments.
    """
    assert JOB_KW["steps"] != JOB_KW["seq_len"], (
        "these two read the same number here, so exchanging them at the "
        "get_config call site is invisible to this test")

    real_get_config = study.get_config
    seen: dict = {}

    def config_spy(arm, **overrides):
        cfg = real_get_config(arm, **overrides)
        seen["arm"] = arm
        seen["overrides"] = overrides
        seen["cfg"] = cfg
        return cfg

    monkeypatch.setattr(study, "get_config", config_spy)

    record = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    assert seen, "run_job never built a training config"
    assert seen["overrides"].get("steps") == JOB_KW["steps"], (
        "the training budget that reached get_config is not the job's")
    assert seen["overrides"].get("seq_len") == JOB_KW["seq_len"], (
        "the sequence length that reached get_config is not the job's")
    assert seen["cfg"].train.steps == JOB_KW["steps"]
    assert seen["cfg"].train.seq_len == JOB_KW["seq_len"]
    assert seen["cfg"].train.seed == JOB_SEED
    assert seen["arm"] == JOB_ARM
    assert record["steps"] == seen["cfg"].train.steps, (
        "the record reports a budget the model was not trained at")
    assert record["seq_len"] == seen["cfg"].train.seq_len, (
        "the record reports a sequence length the model was not trained at")


def test_the_recorded_seconds_cover_the_training_they_are_meant_to_price(
    tmp_path, small_buffer, monkeypatch
):
    """`record["seconds"] > 0.0` is a textbook vacuous assertion.

    It was the only thing said about the field, and it cannot fail for ANY
    placement of `started = time.perf_counter()`. Moving that line BELOW the
    `train_world_model` call survived the suite while making `seconds` the
    duration of the EVALUATION alone -- seconds of arithmetic reported for a
    cell that cost ~3.5 hours on the rented GPU, in the study's own cost
    record and its only evidence of what the nine runs were worth.

    So the trainer is made to take a known, extra half second and the record is
    required to have counted it. The naive form of this test would be
    `seconds > SLEEP`; the first assertion below shows why that is not enough
    -- the evaluation of this fixture alone already outlasts SLEEP, so the
    mutant would pass it. What separates them is the MARGIN over the
    evaluation-only stretch, which is the whole of training under the honest
    line and is nothing at all under the mutant.
    """
    sleep_for = 0.5
    real_train = study.train_world_model
    marks: dict = {}

    def slow_train(cfg, buffer, out_dir, log_every=100):
        history = real_train(cfg, buffer, out_dir=out_dir, log_every=log_every)
        time.sleep(sleep_for)
        marks["train_returned"] = time.perf_counter()
        return history

    monkeypatch.setattr(study, "train_world_model", slow_train)

    record = run_job(JOB, small_buffer, tmp_path, **JOB_KW)
    returned_at = time.perf_counter()

    assert "train_returned" in marks, "the stand-in trainer never ran"
    evaluation_only = returned_at - marks["train_returned"]
    assert evaluation_only > sleep_for, (
        "the evaluation is shorter than the sleep, so `seconds > sleep_for` "
        "would already tell the two timer placements apart and this test is "
        "not measuring what it claims to")
    assert record["seconds"] - evaluation_only > sleep_for / 2, (
        "`seconds` excludes the training it is supposed to price: it is "
        f"{record['seconds']:.3f}s against {evaluation_only:.3f}s of "
        "evaluation, so the timer starts after train_world_model returns")


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

    run_job(JOB, small_buffer, tmp_path, **JOB_KW)

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
    twenty distinct values the four means are all different, so any resizing
    of the slice moves the number.

    THE REDUCER OVER THE WINDOW IS PINNED TOO, and it was not. The tail used
    to be `[1.0 .. 20.0]`, whose mean AND median are both exactly 10.5, so
    `np.mean -> np.median` was a numerical no-op and survived -- the same
    fixture-coincidence species as `steps == horizon` above, one function
    deeper. `loss_last20` is the study's convergence number in all nine
    records and on a skewed tail the two statistics are ~27% apart. So the
    tail below is deliberately SKEWED: mean 14.5 against median 10.5, and the
    guard asserts that separation rather than trusting it.
    """
    real_train = study.train_world_model
    # Skewed on purpose: mean 14.5, median 10.5, last two 59.5, last one 100.0,
    # whole history 43.0 -- five different answers to "the training loss".
    tail = [float(i) for i in range(1, 20)] + [100.0]
    assert len({float(np.mean(tail)), float(np.mean(tail[-2:])),
                float(tail[-1])}) == 3, (
        "a constant tail makes the loss_last20 assertion blind to a narrowed "
        "window, which is the mutation this test exists to kill")
    assert float(np.mean(tail)) != float(np.median(tail)), (
        "the tail's mean and median coincide, so `np.mean -> np.median` over "
        "the window is a numerical no-op and the assertion below cannot see "
        "it -- which is exactly the state this fixture was found in")

    def doctored(cfg, buffer, out_dir, log_every=100):
        history = real_train(cfg, buffer, out_dir=out_dir, log_every=log_every)
        history["steps"] = 1000
        history["seconds"] = 4.0
        history["loss"] = [100.0] * 10 + tail
        history["kl_rate_above_free_bits"] = 0.25
        history["kl_dyn_max"] = 7.5
        return history

    monkeypatch.setattr(study, "train_world_model", doctored)
    result = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    assert result["kl_rate_above_free_bits"] == pytest.approx(0.25)
    assert result["kl_dyn_max"] == pytest.approx(7.5)
    assert result["steps_per_second"] == pytest.approx(250.0)
    assert result["loss_last20"] == pytest.approx(float(np.mean(tail)))
    assert result["loss_last20"] == pytest.approx(14.5)
    assert result["loss_last20"] != pytest.approx(float(np.median(tail))), (
        "the reported convergence number is the window's MEDIAN, not its mean")


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
    result = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

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
    result = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

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
    of the MSE on this fixture and were bit-identical for M3b's pixel arm. So
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



# ---------------------------------------------------------------------------
# Task 4, FINAL ROUND. The 19 Critical/Important findings of the last audit,
# plus the four Minors Task 5's argparse-based driver will hand strings to.
#
# Every one of these mutations was confirmed to leave 529/529 green. What they
# have in common is that the record still has the right SHAPE: the right keys,
# the right types, plausible magnitudes. Nine JSON files are the study's only
# artifact, so a field that is silently wrong is worse than one that is absent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [JOB_ARM, OTHER_ARM, THIRD_ARM])
def test_every_evaluation_reads_the_cache_the_arms_encoder_actually_uses(
    tmp_path, small_buffer, monkeypatch, arm
):
    """WHICH FEATURE CACHE the five evaluations read had no assertion at all.

    Two mutations of one line, both green across the whole suite:

      * `backbone = "dinov2"`. Demonstrated live on the real data: the
        record's `filtering.criterion_4.latent_beats_embedding` flips
        False -> True (latent_r2 -1.06e-05 -> +1.15e-03), so GATE CRITERION 4
        WOULD BE REPORTED AS PASSED in all nine records, off the wrong arm's
        cached features, and every curve and band moves with it.

      * `backbone = job.arm` -- the exact hazard `mbfps/models/encoders.py`
        documents ("the arm name and the backbone name are deliberately not
        assumed equal"). It was invisible by pure fixture coincidence:
        `random_vit`'s arm name IS its backbone name and so is `pixel_ae`'s,
        and those were the only two arms this file ever ran. `frozen_ssl`, the
        one arm where they differ, is the treatment arm of the study, and all
        three of its seeds would die at
        `FileNotFoundError('.features_frozen_ssl.npy')` hours in -- after the
        training time is paid, on the unattended box.

    So this runs ALL THREE ARMS and pins the backbone that reached each of the
    five evaluation calls. On `frozen_ssl` the mutation cannot even complete:
    the cache it names does not exist, which is precisely the study-box
    failure, reproduced here in two seconds.
    """
    from mbfps.data.loader import feature_suffix
    from mbfps.models.encoders import encoder_backbone
    from mbfps.utils.config import get_config

    expected = encoder_backbone(get_config(arm, device="cpu", seed=JOB_SEED).encoder)
    assert expected == {JOB_ARM: JOB_ARM, OTHER_ARM: OTHER_ARM,
                        THIRD_ARM: THIRD_ARM_BACKBONE}[arm], (
        f"{arm!r} no longer reads the cache this test was written against")

    # Check-it-can-fail, part one: a hardcoded "dinov2" is the WRONG answer
    # for two of the three arms, and `job.arm` is the wrong answer for the
    # third -- so between them the parametrisation moves.
    if arm != THIRD_ARM:
        assert expected != THIRD_ARM_BACKBONE, (
            "a hardcoded dinov2 backbone is indistinguishable from the truth "
            f"on {arm!r}")
    else:
        assert expected != arm, (
            f"the arm name and the backbone name coincide on {arm!r}, so "
            "`backbone = job.arm` is invisible here")

    # Check-it-can-fail, part two: every arm is a feature arm now, and
    # reading the OTHER cache must be a real change, not a relabelling. For
    # the ViT arms that is different numbers of the same shape; for
    # `pixel_ae` it is a different SHAPE -- (64, 32) rows against (64, 384)
    # -- which the loader's geometry check refuses outright.
    other_backbone = JOB_ARM if expected == THIRD_ARM_BACKBONE \
        else THIRD_ARM_BACKBONE
    episode = small_buffer.episode_paths()[0]
    mine = np.load(episode.with_suffix(feature_suffix(expected)))
    theirs = np.load(episode.with_suffix(feature_suffix(other_backbone)))
    assert mine.shape != theirs.shape or not np.array_equal(mine, theirs), (
        "the two feature caches hold identical data in this fixture, so "
        "reading the wrong one is a numerical no-op and nothing below "
        "guards anything")

    seen: dict = {}
    backbone_argument = {
        "fit_probes": 2,
        "reward_accuracy": 2,
        "filtering_report": 3,
        "filtering_gain": 3,
        "evaluate_rollout": "feature_backbone",
    }

    def spy(name, where):
        real = getattr(study, name)

        def wrapper(*args, **kwargs):
            seen[name] = (kwargs.get(where) if isinstance(where, str)
                          else args[where])
            return real(*args, **kwargs)

        monkeypatch.setattr(study, name, wrapper)

    for name, where in backbone_argument.items():
        spy(name, where)

    run_job(StudyJob(arm, JOB_SEED), small_buffer, tmp_path, **JOB_KW)

    assert set(seen) == set(backbone_argument), (
        f"not every evaluation was observed: {sorted(seen)}")
    for name, got in seen.items():
        assert got == expected, (
            f"{name} was pointed at the {got!r} feature cache while {arm!r}'s "
            f"encoder consumes {expected!r} -- every number this evaluation "
            "produces describes another arm's inputs")


def test_reward_accuracy_scores_the_reward_head_against_the_recorded_reward(
    small_buffer, monkeypatch
):
    """TWO Critical mutations at one call, both green across the suite.

      * `_summarise_reward(predicted, true)` -> `(true, predicted)`. Measured
        on three real `my_way_home` episodes: `is_degenerate` True -> False,
        `r2` NaN -> -47.11, `n_reward_events` 0 -> 1574; only `mse` survives,
        because it is symmetric. `is_degenerate` is the field that exists to
        stop a reader quoting a four-digit MSE over a near-constant target,
        and it INVERTS in all nine records.
        `test_degeneracy_is_a_property_of_the_target_not_the_prediction` pins
        `_summarise_reward` itself; nothing pinned the ORDER at the one call
        site that feeds it.

      * `model.heads(...)["reward"]` -> `["continue_logit"]`. The two heads
        have the same (B, T) shape, so no guard fires. Measured on real
        episodes the MSE goes 0.00258653 -> 0.00186895 -- a 28% BETTER-looking
        number for a head that never predicts reward.

    Both are killed by naming, for each argument, the tensor it must be.
    """
    import torch

    from mbfps.data.episode import load_episode
    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.models.heads import WorldModelHeads
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config(JOB_ARM, device="cpu", seed=JOB_SEED)
    model = WorldModel(cfg)
    model.eval()
    paths = small_buffer.episode_paths()[:2]

    head_outputs: list[dict] = []
    real_forward = WorldModelHeads.forward

    def forward_spy(self, latent):
        out = real_forward(self, latent)
        head_outputs.append(out)
        return out

    summarised: dict = {}
    real_summarise = study._summarise_reward

    def summarise_spy(*args, **kwargs):
        summarised["args"] = args
        return real_summarise(*args, **kwargs)

    monkeypatch.setattr(WorldModelHeads, "forward", forward_spy)
    monkeypatch.setattr(study, "_summarise_reward", summarise_spy)

    reward_accuracy(model, paths, encoder_backbone(cfg.encoder),
                    torch.device("cpu"), limit=2)

    assert len(head_outputs) == len(paths), (
        f"expected one head call per episode, saw {len(head_outputs)}")
    assert "args" in summarised, "_summarise_reward was never called"
    assert len(summarised["args"]) == 2, (
        "the summary is called positionally; this test reads the order off "
        "those two positions")
    got_predicted, got_true = summarised["args"]

    reward_head = np.concatenate(
        [out["reward"][0].float().cpu().numpy() for out in head_outputs])
    continue_head = np.concatenate(
        [out["continue_logit"][0].float().cpu().numpy() for out in head_outputs])
    recorded = np.concatenate([load_episode(p).rewards for p in paths])

    # Check-it-can-fail, three ways: the reward head and the continue head
    # really disagree, the prediction and the target really disagree, and the
    # two arguments have the SAME SHAPE, so exchanging them raises nothing.
    assert reward_head.shape == continue_head.shape
    assert not np.allclose(reward_head, continue_head), (
        "the reward head and the continue head agree on this fixture, so "
        "reading the wrong one is invisible to the assertion below")
    assert reward_head.shape == recorded.shape
    assert not np.allclose(reward_head, recorded), (
        "the prediction equals the target here, so exchanging the two "
        "arguments of _summarise_reward is a no-op and cannot be seen")

    assert np.array_equal(got_predicted, reward_head), (
        "reward accuracy did not summarise the REWARD head's output; the "
        "continue head has the same shape and yields a plausible, better "
        "looking MSE for a head that never predicts reward")
    assert not np.array_equal(got_predicted, continue_head)
    assert np.array_equal(got_true, recorded), (
        "the episode's recorded reward is not the second argument -- with the "
        "two exchanged, `is_degenerate` becomes a property of the model's "
        "predictions instead of the target and inverts in all nine records")


def test_reward_accuracys_own_inputs_are_placed_on_the_device_it_was_given(
    small_buffer, monkeypatch
):
    """The two per-episode input tensors, and the reason the study box dies.

    `torch.as_tensor(source).to(device)` -> `torch.as_tensor(source)`, and the
    same one line down for the actions, are byte-identical to pristine on CPU
    -- and every test in this suite runs `device="cpu"`, so 529/529 stayed
    green. Demonstrated on MPS, standing in for the rented CUDA box:
    `RuntimeError: Tensor for argument input is on cpu but expected on mps`.
    On the GPU box every job dies inside `reward_accuracy` AFTER paying its
    training hours -- 8.3 h for the pixel arm -- and the 33-hour unattended run
    produces nothing at all.

    The device test for `run_job` pins `.to()` and `map_location=` by OBJECT
    IDENTITY, which is what lets a device guard bite on a CUDA-less machine.
    The same trick reaches these two statements through a spy on
    `torch.Tensor.to`: the device object this test hands `reward_accuracy` is
    not equal-but-fresh, it is THE object, and nothing the mutation can
    construct satisfies `is`.
    """
    import torch

    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    device = torch.device("cpu")
    # Check-it-can-fail: an equal-but-freshly-built device -- exactly what a
    # hardcoding mutation constructs -- is NOT this object, on this machine,
    # where both of them are plain CPU.
    fresh = torch.device(device.type)
    assert fresh == device and fresh is not device, (
        "object identity can no longer tell a hardcoded device from the one "
        "the caller passed, so this test would pass on the GPU-box regression")

    cfg = get_config(JOB_ARM, device="cpu", seed=JOB_SEED)
    model = WorldModel(cfg)
    model.eval()

    placed: list = []
    real_to = torch.Tensor.to

    def to_spy(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, torch.device):
            placed.append((self.dtype, target))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_spy)
    reward_accuracy(model, small_buffer.episode_paths()[:1],
                    encoder_backbone(cfg.encoder), device, limit=1)
    monkeypatch.undo()

    assert len(placed) == 2, (
        "reward_accuracy places exactly two tensors per episode -- the encoder "
        "input and the action sequence. Seeing "
        f"{len(placed)} means one of them was left where numpy put it, which "
        f"is a RuntimeError on any non-CPU device: {placed}")
    dtypes = [dtype for dtype, _ in placed]
    assert torch.int64 in dtypes, (
        "the action tensor never reached a device: `.unsqueeze(0).to(device)` "
        "-> `.unsqueeze(0)`")
    assert any(dtype != torch.int64 for dtype in dtypes), (
        "the encoder input never reached a device: "
        "`torch.as_tensor(source).to(device)` -> `torch.as_tensor(source)`")
    for dtype, target in placed:
        assert target is device, (
            f"the {dtype} input was moved to a device reward_accuracy built "
            "for itself rather than the one it was handed; both are CPU here, "
            f"so only object identity can see it (got {target})")


def test_reward_accuracy_is_reproducible_from_its_own_seed(small_buffer):
    """`observe` SAMPLES from the posterior, so the RNG really moves the number.

    Two mutations, both green:

      * `torch.manual_seed(seed)` -> `torch.manual_seed(0)`. All three seeds of
        an arm then draw the same posterior sample stream, criterion 3's
        seed-to-seed spread is understated, and the three seeds stop being
        independent replicates for that field -- which is the only thing the
        3-seed design buys.
      * the line deleted outright. Reward accuracy then depends on whatever
        RNG state training and the probe fit happen to leave behind, so
        re-running a cell need not reproduce its own record -- while the
        driver resumes off these records precisely because a cell is supposed
        to be reproducible.

    The first assertion kills the hardcode; the second kills the deletion, and
    is only meaningful because the first has shown the stream matters.
    """
    import torch

    from mbfps.eval.study import reward_accuracy
    from mbfps.models.encoders import encoder_backbone
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import get_config

    cfg = get_config(JOB_ARM, device="cpu", seed=JOB_SEED)
    model = WorldModel(cfg)
    model.eval()
    backbone = encoder_backbone(cfg.encoder)
    paths = small_buffer.episode_paths()[:2]

    def score(seed):
        return reward_accuracy(model, paths, backbone, torch.device("cpu"),
                               limit=2, seed=seed)["mse"]

    at_one, at_two = score(1), score(2)
    assert at_one != at_two, (
        "two different seeds produce the same reward MSE, so the posterior is "
        "not being sampled here and neither assertion in this test can fail "
        "-- `torch.manual_seed(seed) -> torch.manual_seed(0)` would be a no-op")

    # Check-it-can-fail for the DELETION: the two ambient states below really
    # are different streams, so a reward_accuracy that did not reseed would
    # return two different numbers for one seed.
    torch.manual_seed(999)
    ambient_a = torch.rand(1).item()
    torch.manual_seed(12345)
    ambient_b = torch.rand(1).item()
    assert ambient_a != ambient_b, "the two ambient RNG states coincide"

    torch.manual_seed(999)
    from_ambient_a = score(1)
    torch.manual_seed(12345)
    from_ambient_b = score(1)
    assert from_ambient_a == from_ambient_b == at_one, (
        "the reward MSE moved with the AMBIENT RNG state, so this cell is not "
        "reproducible from its own seed: re-running it would not reproduce "
        "its own record, and the driver resumes off exactly that promise")


def test_the_reward_event_count_rounds_the_array_and_the_median_onto_one_grid():
    """`rounded` and the median it is compared against must share a grid.

    Two mutations, both green, both giving the same measured outcome on the
    real 59,511-step `my_way_home` reward stream: `n_reward_events` 21 ->
    59,511, i.e. EVERY STEP counted as a reward event.

      * `np.round(np.median(true), 6)` -> `np.median(true)`;
      * `np.round(true, 6)` -> `np.round(true, 12)`.

    The mechanism is the same in both: `my_way_home`'s living penalty is a
    float32 -0.0004, which widens to -0.00039999998989515007 in float64, so
    the rounded array and the unrounded (or finer-rounded) median stop being
    equal and every step differs from the modal value. The count that makes
    criterion 3 readable -- "six steps carry the goal" -- reports the exact
    opposite finding, at a magnitude nobody would flag as impossible.

    `test_the_reward_rounding_resolves_a_fine_grained_target` pins the decimals
    DOWNWARD only (6 -> 0), on a synthetic float64 grid where both roundings
    agree, so it cannot see either of these.
    """
    living_penalty = np.float32(-0.0004)
    true = np.full(1000, living_penalty, dtype=np.float32)
    true[:3] = 1.0                                  # three goal events

    # Check-it-can-fail: the float32 penalty is NOT representable on the
    # 6-decimal grid, which is the entire mechanism. On a target where the two
    # agree, both mutations are numerical no-ops.
    widened = np.asarray(true, dtype=np.float64)
    assert float(np.median(widened)) != float(np.round(np.median(widened), 6)), (
        "this target's modal value survives the float64 widening unchanged, "
        "so dropping the median's rounding is invisible here")
    assert float(np.round(widened, 12)[10]) != float(np.round(np.median(widened), 6)), (
        "a 12-decimal grid agrees with the 6-decimal median on this target, "
        "so widening the array's rounding is invisible here")

    out = _summarise_reward(np.zeros(1000, dtype=np.float64), true)
    assert out["n_reward_events"] == 3, (
        "the event count is not counting reward EVENTS: with the array and "
        "the median rounded onto different grids every step differs from the "
        "modal value and the count becomes the step count")
    assert out["n_reward_events"] != out["n_steps"]
    assert out["is_degenerate"] is True


def test_the_written_record_keeps_its_strings_and_its_booleans(
    record, written, tmp_path
):
    """`_sanitise`'s fallback `return value` -> `return None`, green across the
    whole suite AND across the suite with this file excluded.

    NO TEST READS A STRING OUT OF THE SANITISED PROJECTION: every assertion on
    `written` or on `strict_loads(text)` is a self-comparison, or is about a
    number or a None. Measured on a record of the real shape, the mutation
    writes `"arm": null` and every held-out episode name as `null`. Task 6
    GROUPS THE NINE FILES BY `arm`, so all three arms' results become
    indistinguishable, and the episode audit trail that makes "all nine cells
    were scored on the same episodes" checkable is erased.

    The bool branch is pinned here too. `isinstance(value, (bool, np.bool_))`
    -> `(np.bool_,)` lets a Python bool fall through to the int branch:
    measured, `latent_beats_embedding` False -> 0, `ridge_selected` True -> 1,
    `is_degenerate` True -> 1. Gate criterion 4's own verdict changes JSON
    TYPE in all nine files, and any aggregation using `is True`, a schema
    check or a strict type assertion reads it wrong.
    `test_numpy_scalars_are_written_as_plain_json` only ever passes `np.bool_`.
    """
    text = job_record_path(tmp_path, JOB).read_text()
    parsed = strict_loads(text)

    # Check-it-can-fail: these fields carry real strings in the live record,
    # so `None` in the projection is a genuine difference, not a re-encoding.
    assert isinstance(record["arm"], str) and record["arm"]
    assert record["episodes"]["val"] and record["episodes"]["train"]

    assert written["arm"] == JOB_ARM and isinstance(written["arm"], str), (
        f"the arm label reached the file as {written['arm']!r}; Task 6 groups "
        "the nine records by this field, so all three arms would be one group")
    assert parsed["arm"] == JOB_ARM, "the same, read back off disk"
    for split in ("train", "val"):
        assert written["episodes"][split] == record["episodes"][split]
        assert parsed["episodes"][split] == record["episodes"][split]
        assert all(isinstance(name, str) and name.endswith(".npz")
                   for name in written["episodes"][split]), (
            f"the held-out episode names were erased from the {split} list; "
            "the record's audit trail is what makes 'all nine cells were "
            "scored on the same episodes' checkable rather than assumed")

    # Booleans, in the record's own fields rather than in a synthetic dict.
    for section, field in (("reward", "is_degenerate"),
                           ("probe", "latent_ridge_selected"),
                           ("probe", "embedding_ridge_selected")):
        live = record[section][field]
        assert isinstance(live, bool), (section, field, type(live))
        assert written[section][field] is live, (
            f"{section}.{field} left the sanitiser as "
            f"{written[section][field]!r} rather than a JSON boolean")
    verdict = record["filtering"]["criterion_4"]["latent_beats_embedding"]
    assert isinstance(verdict, bool), (
        "gate criterion 4's verdict is no longer a Python bool, so the "
        "assertion below cannot see the int coercion it exists for")
    assert written["filtering"]["criterion_4"]["latent_beats_embedding"] is verdict
    assert parsed["filtering"]["criterion_4"]["latent_beats_embedding"] is verdict


def test_the_sanitiser_passes_strings_and_python_bools_through_unchanged(tmp_path):
    """The same two branches, at unit scale and with the JSON text inspected.

    `written["b"] is True` is the form that bites: `isinstance(1, int)` and
    `1 == True` are both true, so a `== True` assertion would pass on the
    integer the bool branch's mutation produces.
    """
    written = write_record(tmp_path / "r.json", {
        "arm": "frozen_ssl",
        "episodes": {"val": ["ep004.npz", "ep005.npz"]},
        "yes": True, "no": False,
        "one": 1, "zero": 0,
    })
    text = (tmp_path / "r.json").read_text()

    assert written["arm"] == "frozen_ssl"
    assert written["episodes"]["val"] == ["ep004.npz", "ep005.npz"]
    assert '"arm": "frozen_ssl"' in text
    assert "ep004.npz" in text

    assert written["yes"] is True and written["no"] is False
    assert '"yes": true' in text and '"no": false' in text
    # The ints are untouched, which is what makes the two branches distinct.
    assert written["one"] == 1 and written["one"] is not True
    assert written["zero"] == 0 and written["zero"] is not False


def test_the_out_dir_and_the_record_path_accept_the_strings_argparse_gives(
    tmp_path, small_buffer
):
    """Task 5's driver is built on `argparse`, so `--out-dir` arrives as a STR.

    Four coercions were unpinned, and the first is the one that would bite on
    the driver's very first action:

      * `job_record_path`: `Path(out_dir) / ...` -> `out_dir / ...`. THIS IS
        THE RESUME PATH. The driver calls it to decide whether an 8.3-hour
        cell has already been run, before anything else happens, and on a str
        it raises `TypeError: unsupported operand type(s) for /: 'str' and
        'str'`.
      * `run_job`: `out_dir = Path(out_dir)` deleted -> `out_dir.mkdir` on a
        str.
      * `write_record`: `path = Path(path)` deleted -> `path.parent` on a str.
      * `load_record`: `Path(path).read_text()` -> `path.read_text()`.

    Every existing test passes `tmp_path`, which is already a `Path`, so all
    four were invisible. Each mutation raises here.
    """
    out_dir = str(tmp_path / "study")

    path = job_record_path(out_dir, JOB)
    assert isinstance(path, Path)
    assert path.parent == Path(out_dir)
    assert path.name == f"result_{JOB_ARM}_seed{JOB_SEED}.json"
    # The resume decision the driver makes before it spends any GPU time.
    assert not path.exists()

    result = run_job(JOB, small_buffer, out_dir, **JOB_KW)

    assert path.is_file(), "the record was not written where the driver looks"
    assert job_record_path(out_dir, JOB).exists(), (
        "the resume check would re-run a cell that has already been paid for")

    restored = load_record(str(path))
    assert restored["arm"] == result["arm"] == JOB_ARM
    assert restored["seed"] == JOB_SEED

    elsewhere = tmp_path / "made" / "up" / "r.json"
    write_record(str(elsewhere), {"x": 1.0, "y": float("nan")})
    assert elsewhere.is_file(), "write_record did not create the parent chain"
    assert load_record(str(elsewhere))["x"] == 1.0


# --------------------------------------------------------------------------
# provenance: git_sha, device, encoder_params, and the full history (M3c)
# --------------------------------------------------------------------------
#
# The M3b write-up had to RECONSTRUCT three things after the fact: which code
# state produced the nine cells (from file mtimes), where each cell ran (from
# the log), and whether the pixel arm's KL ever cleared the free-bits floor
# (from a scalar rate, because the per-step history was thrown away). Each is
# now a field. Every guard below pins the field to a value the test computes
# INDEPENDENTLY -- the test's own `git rev-parse`, its own `numel` sum, the
# history it handed `train_world_model` -- because "is a string" and "is a
# list" are satisfied by a hardcoded literal.

#: What the record must hold under `history`, and the five loss terms every
#: `parts[i]` carries, spelled out as literals so a term dropped from
#: `WorldModel.forward`'s dict fails here by name.
HISTORY_KEYS = frozenset({"loss", "parts"})
PART_KEYS = frozenset({"embedding", "reward", "continue", "kl_dyn", "kl_rep"})

#: Spec 2.2's one stated asymmetry, per arm: `pixel_ae`'s bottleneck is
#: `Linear(32 -> 32)` = 32*32 + 32; the ViT arms' is `Linear(384 -> 32)` =
#: 384*32 + 32. A LITERAL, not a computation, so that a geometry change that
#: silently alters the bottleneck width shows up as a number, not as two
#: computations agreeing with each other.
EXPECTED_ENCODER_PARAMS = {
    "pixel_ae": 1_056,
    "frozen_ssl": 12_320,
    "random_vit": 12_320,
}


def _git_head(cwd: Path) -> str | None:
    """`git rev-parse HEAD` at `cwd`, run by the TEST, or None if git cannot."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def test_the_record_carries_the_sha_git_reports_for_the_code_that_ran(record):
    """`git_sha` must equal what `git rev-parse HEAD` says HERE, in this
    checkout -- the test runs git itself and compares. "Is a 40-character
    string" is satisfied by a literal pasted into `run_job`, and a literal is
    precisely the thing that would make nine records from two code states
    look like one.

    The sha is asked of the directory `study.py` lives in, not the process's
    cwd; the two are the same repository here, and this test pins the value,
    not the cwd (that is `test_git_sha_is_asked_of_the_code_not_the_cwd`).
    """
    repo_root = Path(__file__).resolve().parents[2]
    expected = _git_head(repo_root)
    if expected is None:
        pytest.skip("git cannot report HEAD for this checkout, so the value "
                    "cannot be pinned here; the failure path is tested below")
    assert len(expected) == 40 and set(expected) <= set("0123456789abcdef"), (
        f"the test's own git call returned {expected!r}, not a sha; the "
        "comparison below would be against garbage")
    assert record["git_sha"] == expected
    assert isinstance(record["git_sha"], str)
    assert record["git_sha"] != study.UNKNOWN_GIT_SHA


def test_git_sha_is_asked_of_the_code_not_the_cwd(monkeypatch):
    """`cwd=` must be `study.py`'s own directory. A study launched from
    `/scratch` must report the code's commit, not `/scratch`'s (or fail
    because `/scratch` is not a repository)."""
    import subprocess

    seen: dict = {}
    real_run = subprocess.run

    def spy(args, **kwargs):
        seen["args"] = list(args)
        seen["cwd"] = kwargs.get("cwd")
        return real_run(args, **kwargs)

    monkeypatch.setattr(study.subprocess, "run", spy)
    study._git_sha()
    assert seen["args"] == ["git", "rev-parse", "HEAD"]
    assert Path(seen["cwd"]).resolve() == Path(study.__file__).resolve().parent, (
        f"git was asked at {seen['cwd']!r}, not where the code lives")


@pytest.mark.parametrize("failure", [
    "no_git_executable", "git_hangs", "not_a_repository", "empty_stdout",
])
def test_git_sha_is_unknown_rather_than_an_exception_when_git_cannot_answer(
    monkeypatch, failure
):
    """A rented box without git must not lose a cell to a provenance field.

    Four ways the subprocess can fail to produce a sha, EACH EXERCISED ALONE
    so that no single guard covers for another: the executable is missing
    (`OSError`), it hangs (`TimeoutExpired`, a `SubprocessError`), it runs but
    the directory is not a repository (non-zero status, and git's usual
    empty stdout), and the degenerate zero-status-empty-stdout -- which a
    guard on the status alone would pass through as `""`.

    `not_a_repository` deliberately puts a non-sha ON stdout with the
    non-zero status, so the returncode guard is what has to catch it; the
    `or UNKNOWN` fallback alone would let "abc" through.
    """
    import subprocess
    from types import SimpleNamespace

    def fake_run(args, **kwargs):
        if failure == "no_git_executable":
            raise FileNotFoundError(2, "No such file or directory", "git")
        if failure == "git_hangs":
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))
        if failure == "not_a_repository":
            return SimpleNamespace(returncode=128, stdout="abc\n", stderr="fatal")
        return SimpleNamespace(returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(study.subprocess, "run", fake_run)
    assert study._git_sha() == study.UNKNOWN_GIT_SHA == "unknown"


def test_the_record_says_where_the_cell_ran_not_what_was_asked_for(
    tmp_path, small_buffer, monkeypatch
):
    """`device` is `str(torch_device)` -- the device `get_device` RESOLVED --
    not the `device` string the job was launched with. On a box without CUDA,
    `--device cuda` runs on the CPU, and a record that says "cuda" is the lie
    the field exists to prevent.

    `get_device` is pinned to CPU here so the guard is machine-independent:
    without the pin, `get_device(prefer="cuda")` already IS cpu on this Mac
    and would be cuda on the box, so `"device": device` would pass on the
    box for the wrong reason. With it, the request ("cuda") and the answer
    ("cpu") differ everywhere."""
    import torch

    requested = "cuda"
    resolved = torch.device("cpu")
    seen: dict = {}

    def pinned_get_device(prefer="mps"):
        seen["prefer"] = prefer
        return resolved

    monkeypatch.setattr(study, "get_device", pinned_get_device)
    result = run_job(JOB, small_buffer, tmp_path,
                     **dict(JOB_KW, device=requested))

    assert seen["prefer"] == requested, "run_job did not ask for the job's device"
    assert str(resolved) != requested, (
        "the pinned device spells the same as the request, so the assertion "
        "below cannot tell the resolved device from the requested string")
    assert result["device"] == str(resolved) == "cpu"
    assert isinstance(result["device"], str)
    assert result["device"] != requested


@pytest.mark.parametrize("arm", sorted(EXPECTED_ENCODER_PARAMS))
def test_encoder_params_counts_the_trainable_encoder_parameters_of_each_arm(
    tmp_path, small_buffer, arm
):
    """`encoder_params` is the sum of `numel` over the ENCODER's trainable
    parameters, computed here independently of `run_job` -- and equal to the
    spec's own stated number for the arm.

    Run for every arm because the value is the one place the arms differ and
    a literal (12,320) would pass on two of the three. The whole-model count
    is asserted different so that `model.parameters()` in place of
    `model.encoder.parameters()` is a visible mutation and not a coincidence.
    """
    from mbfps.training.world_model import WorldModel
    from mbfps.utils.config import ARMS, get_config

    assert set(EXPECTED_ENCODER_PARAMS) == set(ARMS), (
        "the expected counts must name exactly the study's arms; an arm "
        "without a stated count is an arm whose asymmetry is unrecorded")

    job = StudyJob(arm, JOB_SEED)
    result = run_job(job, small_buffer, tmp_path, **JOB_KW)

    model = WorldModel(get_config(arm, device="cpu", seed=JOB_SEED))
    expected = sum(p.numel() for p in model.encoder.parameters()
                   if p.requires_grad)
    whole_model = sum(p.numel() for p in model.parameters())
    assert expected != whole_model, (
        "the encoder is the whole model, so counting everything would pass")
    assert result["encoder_params"] == expected == EXPECTED_ENCODER_PARAMS[arm]
    assert isinstance(result["encoder_params"], int)
    assert not isinstance(result["encoder_params"], bool)


def test_encoder_params_excludes_frozen_parameters():
    """The `requires_grad` filter, alone. On every study arm all encoder
    parameters train, so `run_job` cannot show the filter; a module with one
    frozen tensor can. Without it a future encoder wrapping a frozen backbone
    would report the backbone's millions as "trained"."""
    import torch

    module = torch.nn.Module()
    module.trained = torch.nn.Parameter(torch.zeros(3, 4))          # 12
    module.frozen = torch.nn.Parameter(torch.zeros(100), requires_grad=False)
    module.register_buffer("stat", torch.zeros(1000))               # never
    assert study._trainable_parameters(module) == 12


def test_the_record_carries_the_full_history_at_the_jobs_own_length(record):
    """One `loss` and one five-term `parts` entry PER STEP, for every step
    the job trained -- `steps` entries, not the last twenty and not a mean.
    The M3b write-up's section 1 had to reconstruct the pixel arm's KL
    trajectory because this list was discarded."""
    history = record["history"]
    assert set(history) == HISTORY_KEYS
    assert isinstance(history["loss"], list)
    assert isinstance(history["parts"], list)
    assert len(history["loss"]) == JOB_KW["steps"]
    assert len(history["parts"]) == JOB_KW["steps"]
    # The length check alone cannot see a `[-20:]` slice over `study.py`: at
    # JOB_KW["steps"] == 5 (as at 20) the slice is the whole list. The slice
    # mutation is caught by exact equality against a 23-step history in
    # test_the_history_is_the_one_training_returned_and_not_a_summary_of_it.
    for value in history["loss"]:
        assert isinstance(value, float)
    for part in history["parts"]:
        assert set(part) == PART_KEYS, part
        assert all(isinstance(v, float) for v in part.values())
    # The scalar convergence number is derived from the SAME curve.
    assert record["loss_last20"] == pytest.approx(
        float(np.mean(history["loss"][-20:])))


def test_the_history_is_the_one_training_returned_and_not_a_summary_of_it(
    tmp_path, small_buffer, monkeypatch
):
    """Exact equality against a history the test built: 23 distinct losses
    and 23 distinct `parts` rows, longer than 20 (so a `[-20:]` slice
    shortens it), not equal to `steps` (so a `[:steps]` slice shortens it),
    with a NaN in one row so the non-finite policy is exercised on a value
    FOUR path segments deep -- `history.parts.<k>.kl_dyn` -- through the
    file and back."""
    real_train = study.train_world_model
    n = 23
    assert n not in PAIRWISE_DISTINCT_PARAMETERS.values() and n > 20
    losses = [100.0 + i for i in range(n)]
    nan_row = 7
    parts = [
        {"embedding": 1.0 + i, "reward": 2.0 + i, "continue": 3.0 + i,
         "kl_dyn": float("nan") if i == nan_row else 4.0 + i,
         "kl_rep": 5.0 + i}
        for i in range(n)
    ]

    def doctored(cfg, buffer, out_dir, log_every=100):
        history = real_train(cfg, buffer, out_dir=out_dir, log_every=log_every)
        history["loss"] = list(losses)
        history["parts"] = [dict(p) for p in parts]
        return history

    monkeypatch.setattr(study, "train_world_model", doctored)
    result = run_job(JOB, small_buffer, tmp_path, **JOB_KW)

    assert result["history"]["loss"] == losses
    assert len(result["history"]["parts"]) == n
    for i, (got, want) in enumerate(zip(result["history"]["parts"], parts)):
        assert set(got) == set(want) == PART_KEYS
        for key in PART_KEYS:
            if i == nan_row and key == "kl_dyn":
                assert math.isnan(got[key])
            else:
                assert got[key] == want[key]

    # Through the file and back: the NaN is a `null` plus a four-segment
    # dotted path, and `load_record` restores a float NaN, not 0.0 or None.
    written = to_json_record(result)
    dotted = f"history.parts.{nan_row}.kl_dyn"
    assert written["history"]["parts"][nan_row]["kl_dyn"] is None
    assert written[NONFINITE_KEY][dotted] == "nan"
    restored = load_record(job_record_path(tmp_path, JOB))
    assert math.isnan(restored["history"]["parts"][nan_row]["kl_dyn"])
    assert restored["history"]["parts"][nan_row]["kl_rep"] == 5.0 + nan_row
    assert restored["history"]["loss"] == losses


def test_a_nan_inside_parts_round_trips_through_the_nonfinite_map(tmp_path):
    """The dotted-path scheme at unit scale, on the record's new shape:
    `parts` is a LIST of dicts, so the path carries an integer segment
    (`history.parts.1.kl_dyn`), and `load_record` must index the list rather
    than look the string "1" up in a dict."""
    original = {
        "history": {
            "loss": [0.5, float("inf")],
            "parts": [{"kl_dyn": 0.1}, {"kl_dyn": float("nan"), "kl_rep": 0.2}],
        },
    }
    written = write_record(tmp_path / "r.json", original)
    assert written["history"]["parts"][1]["kl_dyn"] is None
    assert written["history"]["parts"][1]["kl_rep"] == 0.2
    assert written["history"]["loss"][1] is None
    assert written[NONFINITE_KEY] == {
        "history.loss.1": "inf",
        "history.parts.1.kl_dyn": "nan",
    }
    text = (tmp_path / "r.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert strict_loads(text) == written

    back = load_record(tmp_path / "r.json")
    assert math.isnan(back["history"]["parts"][1]["kl_dyn"])
    assert back["history"]["parts"][1]["kl_rep"] == 0.2
    assert back["history"]["parts"][0]["kl_dyn"] == 0.1
    assert back["history"]["loss"][1] == float("inf")
    # The caller's dict still holds the real NaN; only the copy is lossy.
    assert math.isnan(original["history"]["parts"][1]["kl_dyn"])
