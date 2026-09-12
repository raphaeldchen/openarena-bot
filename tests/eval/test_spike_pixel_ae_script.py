"""The spike that licenses the M3c study: three checks, one verdict, one exit status.

Spec section 3 says the nine-cell run happens only if a 2,000-step `pixel_ae`
run shows (1) an embedding loss at step 0 in the feature-arm band, (2) a
dynamics prior that trained on more than half its steps, and (3) cached
features a linear probe can read position from. The M3b pixel arm failed (1)
and (2) silently and cost 27 h before anyone looked. These tests are arranged
around the ways a verdict can be wrong without looking wrong: a check that
reads the last step instead of the first, a threshold that is inclusive where
the spec is strict, a check whose failure the others can mask, a probe scored
on the rows it was fit on, a side-by-side number that is not the number the
check judged, and an exit status that says 0 whatever the checks said.

The script is loaded by path, like `scripts/run_study.py` in its own tests:
`scripts/` is not a package.
"""

import importlib.util
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mbfps.data.episode import load_episode
from mbfps.data.loader import feature_suffix
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.probe import probe_targets
from mbfps.eval.study import SPLIT_SEED, load_record
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config

_SPEC = importlib.util.spec_from_file_location(
    "spike_pixel_ae_script",
    Path(__file__).resolve().parents[2] / "scripts" / "spike_pixel_ae.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

# The three study arms, AS A LITERAL. Parametrising over `ARMS` itself would
# let a fourth arm -- or a dropped one -- shrink this file's coverage without
# a failure; the equality assertion in the first test is what notices.
SPIKE_ARMS = ("pixel_ae", "frozen_ssl", "random_vit")

# The numbers the M3b study measured on its dead pixel arm, so the failing
# cases below are the failure that actually happened and not round numbers.
M3B_CNN_EMBEDDING_LOSS_STEP0 = 0.0008    # M3c design section 1, table 1
M3B_CNN_KL_RATE = 0.0009                 # 18 of 20,000 steps above the floor
M3B_CNN_KL_DYN_MAX = 0.800               # cnn/s1's peak, ABOVE the floor
M3B_CNN_PROBE_R2 = -0.036                # Task 2 results: a constant predictor

TINY = dict(steps=3, seq_len=6, batch_size=2, device="cpu")


def _history(embedding_step0=0.35, embedding_last=0.20, kl_rate=0.9,
             kl_dyn_max=0.7) -> dict:
    """A fabricated `train_world_model` history with the fields the checks read.

    Step 0 and the last step carry DIFFERENT embedding losses on purpose: a
    check that read `parts[-1]`, or the mean, would agree with one that read
    `parts[0]` on a history where they coincide.
    """
    return {
        "parts": [{"embedding": embedding_step0, "kl_dyn": 0.1},
                  {"embedding": embedding_last, "kl_dyn": 0.3}],
        "kl_rate_above_free_bits": kl_rate,
        "kl_dyn_max": kl_dyn_max,
        "loss": [1.0, 0.9],
    }


def _checks(embedding=0.35, kl_rate=0.9, r2=0.3) -> dict:
    return script.evaluate_checks(_history(embedding_step0=embedding,
                                           kl_rate=kl_rate), r2)


def _passed(checks: dict) -> dict:
    return {key: checks[key]["passed"] for key in script.CHECK_KEYS}


@pytest.fixture
def spike_buffer(small_buffer):
    """`small_buffer` plus a `pixel_ae` cache at the registry's (64, 32) rows.

    `tests/eval/conftest.py` writes the two ViT caches at (41, 64, 384). The
    loader validates rows against `BACKBONE_GEOMETRY`, so `pixel_ae` needs its
    own file at its own geometry or the arm cannot even build a batch.
    """
    rng = np.random.default_rng(1)
    for path in small_buffer.episode_paths():
        np.save(path.with_suffix(feature_suffix("pixel_ae")),
                rng.random((41, 64, 32)).astype(np.float16))
    return small_buffer


def _plant_position(path: Path) -> None:
    """Overwrite `path`'s pixel_ae cache so its first four columns ARE the targets.

    Everything else is zero, so the only way a probe on this cache scores
    R^2 > 0.9 is by reading THIS file and aligning its rows with the frames.
    """
    episode = load_episode(path)
    targets = probe_targets(episode.privileged, episode.privileged_keys)
    features = np.zeros((targets.shape[0], 64, 32), dtype=np.float16)
    features[:, 0, :4] = targets
    np.save(path.with_suffix(feature_suffix("pixel_ae")), features)


# --------------------------------------------------------------------------
# what the spike is
# --------------------------------------------------------------------------

def test_the_spike_is_pinned_to_pixel_ae_seed_0_and_2000_steps():
    """Spec section 3: ONE arm, ONE seed, 2,000 steps. The arm and seed are
    constants, not flags -- a `--arm frozen_ssl` spike would pass its own
    checks trivially and license nothing about the arm the study is for."""
    assert script.SPIKE_ARM == "pixel_ae"
    assert script.SPIKE_SEED == 0
    assert script.SPIKE_STEPS == 2_000
    assert set(SPIKE_ARMS) == set(ARMS), (
        "the study's arms changed; the side-by-side and the step-0 test below "
        "must cover every one of them")
    parser = script._parser()
    defaults = parser.parse_args([])
    assert defaults.steps == 2_000
    assert defaults.out == "runs/m3c_spike"
    for flag in ("--arm", "--seed"):
        with pytest.raises(SystemExit):
            parser.parse_args([flag, "0"])


# --------------------------------------------------------------------------
# check 1: embedding loss at step 0, in [0.1, 1.0]
# --------------------------------------------------------------------------

def test_embedding_check_flips_on_each_edge_alone():
    """Both edges of the band, each tested where the other cannot help."""
    assert script.check_embedding_loss([{"embedding": 0.35}])["passed"] is True
    # below the band: the M3b pixel arm's own number
    assert script.check_embedding_loss(
        [{"embedding": M3B_CNN_EMBEDDING_LOSS_STEP0}])["passed"] is False
    # above the band: a target so wide the RSSM would be fitting noise
    assert script.check_embedding_loss([{"embedding": 1.5}])["passed"] is False
    # the band is CLOSED at both ends ("in [0.1, 1.0]")
    assert script.check_embedding_loss([{"embedding": 0.1}])["passed"] is True
    assert script.check_embedding_loss([{"embedding": 1.0}])["passed"] is True
    assert script.check_embedding_loss([{"embedding": 0.0999}])["passed"] is False
    assert script.check_embedding_loss([{"embedding": 1.0001}])["passed"] is False


def test_embedding_check_reads_step_zero_not_the_last_step():
    """The failure section 1 documents is a target that is born collapsed. A
    check that reads the end of training would miss it: the loss can only
    fall from there, and 0.0008 -> 0.0003 is still 'in the band' on nothing."""
    born_dead = [{"embedding": M3B_CNN_EMBEDDING_LOSS_STEP0}, {"embedding": 0.5}]
    assert script.check_embedding_loss(born_dead)["passed"] is False
    born_alive = [{"embedding": 0.5}, {"embedding": M3B_CNN_EMBEDDING_LOSS_STEP0}]
    assert script.check_embedding_loss(born_alive)["passed"] is True
    assert script.check_embedding_loss(born_alive)["value"] == 0.5


def test_embedding_check_refuses_an_empty_history():
    with pytest.raises(ValueError, match="no steps"):
        script.check_embedding_loss([])


# --------------------------------------------------------------------------
# check 2: kl_rate_above_free_bits > 0.5
# --------------------------------------------------------------------------

def test_kl_check_flips_at_one_half():
    """Strict: `_kl_rate`'s own warning fires at `< 0.5`, and exactly one half
    is the prior having trained on precisely as many steps as it did not."""
    assert script.check_kl_rate(_history(kl_rate=0.51))["passed"] is True
    assert script.check_kl_rate(_history(kl_rate=0.5))["passed"] is False
    assert script.check_kl_rate(_history(kl_rate=0.49))["passed"] is False
    assert script.check_kl_rate(_history(kl_rate=M3B_CNN_KL_RATE))["passed"] is False


def test_kl_check_reads_the_rate_not_the_max():
    """`kl_dyn_max` is the number a max-based check would read, and it is the
    number that hid M3b's dead arm: cnn/s1 peaked at 0.800 -- four times the
    floor -- while clearing it on 16 of 20,000 steps."""
    history = _history(kl_rate=M3B_CNN_KL_RATE, kl_dyn_max=M3B_CNN_KL_DYN_MAX)
    assert history["kl_dyn_max"] > 0.5, "fixture: the max must be the misleading one"
    assert script.check_kl_rate(history)["passed"] is False
    assert script.check_kl_rate(history)["value"] == M3B_CNN_KL_RATE


# --------------------------------------------------------------------------
# check 3: held-out probe R^2 on the cached features > 0
# --------------------------------------------------------------------------

def test_probe_check_flips_at_zero():
    """Strict: R^2 of exactly 0 is the constant predictor at the mean, which is
    what the M3b pixel arm's probe collapsed to (-0.036 after ridge 1e7)."""
    assert script.check_probe_r2(0.01)["passed"] is True
    assert script.check_probe_r2(0.0)["passed"] is False
    assert script.check_probe_r2(M3B_CNN_PROBE_R2)["passed"] is False


def test_non_finite_values_fail_every_check():
    """NaN compares False against everything, and a check written as
    `not (value < low or value > high)` would pass it. Each check alone."""
    nan = float("nan")
    assert script.check_embedding_loss([{"embedding": nan}])["passed"] is False
    assert script.check_kl_rate(_history(kl_rate=nan))["passed"] is False
    assert script.check_probe_r2(nan)["passed"] is False
    assert script.all_passed(_checks(embedding=nan)) is False


# --------------------------------------------------------------------------
# the three together: independence, conjunction, decision, exit status
# --------------------------------------------------------------------------

def test_each_check_flips_independently_of_the_other_two():
    """Flip one input at a time; exactly that check must fail. A check that
    read another's field, or the verdict of another, would flip two."""
    assert _passed(_checks()) == {
        "embedding_loss_step0": True,
        "kl_rate_above_free_bits": True,
        "probe_r2_cached_features": True,
    }
    assert _passed(_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0)) == {
        "embedding_loss_step0": False,
        "kl_rate_above_free_bits": True,
        "probe_r2_cached_features": True,
    }
    assert _passed(_checks(kl_rate=M3B_CNN_KL_RATE)) == {
        "embedding_loss_step0": True,
        "kl_rate_above_free_bits": False,
        "probe_r2_cached_features": True,
    }
    assert _passed(_checks(r2=M3B_CNN_PROBE_R2)) == {
        "embedding_loss_step0": True,
        "kl_rate_above_free_bits": True,
        "probe_r2_cached_features": False,
    }


def test_all_passed_requires_all_three():
    """A conjunction, tested one failure at a time so `any` cannot pass."""
    assert script.all_passed(_checks()) is True
    assert script.all_passed(_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0)) is False
    assert script.all_passed(_checks(kl_rate=M3B_CNN_KL_RATE)) is False
    assert script.all_passed(_checks(r2=M3B_CNN_PROBE_R2)) is False


def test_all_passed_refuses_a_checks_dict_missing_a_check():
    checks = _checks()
    del checks["probe_r2_cached_features"]
    with pytest.raises(ValueError, match="probe_r2_cached_features"):
        script.all_passed(checks)


def test_decision_runs_the_study_only_when_all_three_pass():
    assert script.decision(_checks()) == script.RUN_STUDY


def test_decision_sends_a_lone_kl_failure_to_a_free_bits_recalibration():
    """Spec section 3: 'Only if the second fails does KL_FREE_BITS get
    revisited' -- the target is informative, the prior just did not clear the
    floor, so the floor is the thing to measure across all three arms."""
    assert script.decision(_checks(kl_rate=M3B_CNN_KL_RATE)) == \
        script.STOP_RECALIBRATE_FREE_BITS


def test_decision_calls_the_backbone_uninformative_on_check_1_or_3():
    """Each alone: an uninformative target (1) or features with no position
    in them (3) is a property of the backbone, and no floor can fix it."""
    assert script.decision(_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0)) == \
        script.STOP_BACKBONE_UNINFORMATIVE
    assert script.decision(_checks(r2=M3B_CNN_PROBE_R2)) == \
        script.STOP_BACKBONE_UNINFORMATIVE


def test_decision_backbone_verdict_dominates_a_kl_failure():
    """Checks 1 and 2 failing together is exactly the M3b pixel arm, and its
    lesson (section 1) is that lowering the floor cannot help a collapsed
    target. The recalibration route must not be offered."""
    both = _checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0, kl_rate=M3B_CNN_KL_RATE)
    assert script.decision(both) == script.STOP_BACKBONE_UNINFORMATIVE


def test_exit_status_is_zero_iff_all_pass():
    assert script.exit_status(_checks()) == 0
    for failing in (_checks(embedding=M3B_CNN_EMBEDDING_LOSS_STEP0),
                    _checks(kl_rate=M3B_CNN_KL_RATE),
                    _checks(r2=M3B_CNN_PROBE_R2)):
        assert script.exit_status(failing) == 10
    assert script.EXIT_OK == 0 and script.EXIT_SPIKE_FAILED == 10


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _fake_record(checks: dict) -> dict:
    return {
        "arm": "pixel_ae", "seed": 0, "steps": 2000, "device": "cpu",
        "git_sha": "0123abcd", "steps_per_second": 3.9, "kl_dyn_max": 0.7,
        # Distinct from section 1's 0.3165 / 0.4104 on purpose: `report` prints
        # those as a fixed reference on the same line, so a fake record that
        # reused them could not tell the arm's own value from the reference.
        "embedding_loss_step0": {"pixel_ae": 0.35, "frozen_ssl": 0.3211,
                                 "random_vit": 0.4088},
        "probe": {"backbone": "pixel_ae", "r2": 0.3, "ridge": 1e3,
                  "selection_r2": 0.31, "n_fit_episodes": 16,
                  "n_select_episodes": 4, "n_scored_episodes": 20,
                  "n_scored_rows": 10520, "n_columns": 2048},
        "checks": checks,
        "decision": script.decision(checks),
    }


def test_report_marks_each_check_with_its_own_verdict():
    """Three PASS when all pass; exactly one FAIL, on the failing check's own
    line, when one fails. Captured AND asserted, per line."""
    verdicts = lambda text: re.findall(r"\b(PASS|FAIL)\b", text)  # noqa: E731

    text = script.report(_fake_record(_checks()))
    assert verdicts(text) == ["PASS", "PASS", "PASS"]
    assert script.RUN_STUDY in text
    for arm in SPIKE_ARMS:
        assert f"{arm}=" in text, "the side-by-side must name every arm"
    # Each arm's OWN value under its name: a report that printed the pixel_ae
    # number three times would still name every arm.
    assert "pixel_ae=0.3500" in text
    assert "frozen_ssl=0.3211" in text and "frozen_ssl=0.3500" not in text
    assert "random_vit=0.4088" in text and "random_vit=0.3500" not in text

    text = script.report(_fake_record(_checks(kl_rate=M3B_CNN_KL_RATE)))
    assert verdicts(text) == ["PASS", "FAIL", "PASS"]
    line = next(l for l in text.splitlines() if l.startswith("check 2"))
    assert "FAIL" in line
    assert script.STOP_RECALIBRATE_FREE_BITS in text


# --------------------------------------------------------------------------
# the side-by-side: the number printed is the number train_world_model judges
# --------------------------------------------------------------------------

def test_step0_embedding_loss_matches_train_world_model_step_zero_on_every_arm(
        spike_buffer):
    """The side-by-side prints the step-0 embedding loss for all three arms
    from one forward pass each. It is only a comparison if that pass is the
    SAME pass `train_world_model` scores at step 0 -- same seed, same split,
    same loader draw, same posterior sample -- so the helper is held to
    `history["parts"][0]["embedding"]` on every arm, not just the spike's.

    The three values are then required to be pairwise distinct: a helper that
    ignored `cfg` and built `SPIKE_ARM` three times would return one number
    thrice, and on a fixture where the arms happened to coincide the equality
    assertions above could not tell.
    """
    seen = {}
    for arm in SPIKE_ARMS:
        cfg = get_config(arm, seed=0, **TINY)
        expected = train_world_model(cfg, spike_buffer, out_dir=None)
        got = script.step0_embedding_loss(cfg, spike_buffer)
        assert got == pytest.approx(expected["parts"][0]["embedding"], rel=1e-6), arm
        seen[arm] = round(got, 6)
    assert len(set(seen.values())) == 3, (
        f"fixture coincidence: two arms share a step-0 embedding loss {seen}; "
        "the per-arm equality assertions cannot distinguish them")


def test_step0_embedding_loss_runs_one_step_saves_nothing_and_reads_step_zero(
        monkeypatch):
    """The equality test above cannot see three things, because on the TINY
    fixture they change nothing: a helper that forgot `steps=1` (20,000 steps
    per arm on the real run, three times), one that passed an `out_dir` (a
    checkpoint written under the spike directory), and one that read
    `parts[-1]` (equal to `parts[0]` at one step). A spy in place of
    `train_world_model` pins all three."""
    calls = []

    def spy(cfg, buffer, out_dir, log_every=100):
        calls.append((cfg, buffer, out_dir, log_every))
        return {"parts": [{"embedding": 0.31}, {"embedding": 0.17}]}

    monkeypatch.setattr(script, "train_world_model", spy)
    # Seed 1, not the spike's 0: a helper that wrote `seed=SPIKE_SEED` into
    # the one-step config would be invisible at seed 0.
    cfg = get_config("frozen_ssl", seed=1, **TINY)
    assert script.step0_embedding_loss(cfg, "the buffer") == 0.31
    (seen_cfg, seen_buffer, seen_out_dir, seen_log_every), = calls
    assert seen_cfg.train.steps == 1
    assert seen_cfg.arm == "frozen_ssl" and seen_cfg.train.seed == 1
    assert replace(seen_cfg.train, steps=cfg.train.steps) == cfg.train, (
        "every TrainConfig field but steps must reach train_world_model unchanged")
    assert seen_buffer == "the buffer"
    assert seen_out_dir is None
    assert seen_log_every == 0


# --------------------------------------------------------------------------
# check 3's probe on the cache itself
# --------------------------------------------------------------------------

def test_cached_feature_probe_recovers_position_planted_in_the_pixel_ae_cache(
        spike_buffer):
    """Position planted in the pixel_ae cache -- and ONLY there; the dinov2
    and random_vit caches in this fixture are noise at a different geometry --
    must come back through the probe. Reading any other cache raises on
    geometry; reading this one with misaligned rows scores far lower."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    for path in train + val:
        _plant_position(path)
    result = script.cached_feature_probe(train, val, "pixel_ae")
    assert result["backbone"] == "pixel_ae"
    assert result["n_columns"] == 64 * 32
    assert result["r2"] > 0.9, result


def test_cached_feature_probe_scores_the_validation_episodes_not_the_fit_ones(
        spike_buffer):
    """Signal on the training episodes, noise on the held-out one: a probe
    scored on its own fit rows reads ~1.0 here; scored held out it reads
    nothing. 'Held-out' is the word in the spec's check."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    for path in train:
        _plant_position(path)          # val keeps the fixture's noise cache
    result = script.cached_feature_probe(train, val, "pixel_ae")
    assert result["r2"] < 0.5, result


def test_cached_feature_probe_refuses_a_cache_of_the_wrong_geometry(spike_buffer):
    """A stale cache from another geometry must be refused by name, not
    flattened into a probe of the wrong width that scores something."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    bad = train[0].with_suffix(feature_suffix("pixel_ae"))
    np.save(bad, np.zeros((41, 64, 33), dtype=np.float16))
    with pytest.raises(ValueError) as error:
        script.cached_feature_probe(train, val, "pixel_ae")
    message = str(error.value)
    assert "pixel_ae" in message and "(64, 32)" in message and "(64, 33)" in message
    assert bad.name in message


def test_cached_feature_probe_refuses_a_cache_that_is_not_one_row_per_frame(
        spike_buffer):
    """T rows against T+1 privileged frames: `probe_targets` would still
    return T+1 rows and a silent truncation would fit on shifted frames."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    short = train[0].with_suffix(feature_suffix("pixel_ae"))
    np.save(short, np.zeros((40, 64, 32), dtype=np.float16))
    with pytest.raises(ValueError, match="40 rows.*41 frames"):
        script.cached_feature_probe(train, val, "pixel_ae")


def test_cached_feature_probe_reports_the_split_it_used(spike_buffer):
    """Five training episodes with a 4-episode selection set leaves ONE to
    fit on; the record must say so rather than let a reader assume 16/4/20."""
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    assert (len(train), len(val)) == (5, 1)
    result = script.cached_feature_probe(train, val, "pixel_ae")
    assert result["n_fit_episodes"] == 1
    assert result["n_select_episodes"] == 4
    assert result["n_scored_episodes"] == 1
    assert result["n_scored_rows"] == 41
    assert result["ridge"] in (1e-1, 1e1, 1e3, 1e5, 1e7)
    assert math.isfinite(result["selection_r2"])


def test_cached_feature_probe_needs_an_episode_to_fit_on_beyond_the_selection_set(
        spike_buffer):
    train, val = episode_split(spike_buffer.episode_paths(),
                               val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    with pytest.raises(ValueError, match="fit"):
        script.cached_feature_probe(train[:4], val, "pixel_ae")
    with pytest.raises(ValueError, match="validation"):
        script.cached_feature_probe(train, [], "pixel_ae")


# --------------------------------------------------------------------------
# end to end, on CPU, three steps
# --------------------------------------------------------------------------

def test_main_writes_the_full_history_the_checks_and_the_verdict(
        spike_buffer, tmp_path, capsys):
    """One tiny run through `main`. What the record must carry: the FULL
    per-step history (open item 6), every check with its value equal to the
    field it was judged from, the side-by-side for every arm, and a decision.
    And the exit status must be the checks' verdict, not a constant 0.

    25 steps, not 3: `run_job` summarises `history["loss"][-20:]` as
    `loss_last20`, and a `main` that wrote that slice instead of the full list
    would be indistinguishable from it on any run of 20 steps or fewer."""
    out = tmp_path / "spike"
    status = script.main([
        "--data", str(spike_buffer.root), "--out", str(out),
        "--steps", "25", "--seq-len", "6", "--batch-size", "2", "--device", "cpu",
    ])
    record = load_record(script.spike_record_path(out))

    assert record["arm"] == "pixel_ae" and record["seed"] == 0
    assert record["steps"] == 25 and record["seq_len"] == 6
    assert record["split_seed"] == SPLIT_SEED
    assert record["device"] == "cpu"
    assert isinstance(record["git_sha"], str) and record["git_sha"]

    assert len(record["history"]["loss"]) == record["steps"]
    assert len(record["history"]["parts"]) == record["steps"]
    for part in record["history"]["parts"]:
        assert set(part) == {"embedding", "reward", "continue", "kl_dyn", "kl_rep"}

    assert set(record["checks"]) == set(script.CHECK_KEYS)
    assert record["checks"]["embedding_loss_step0"]["value"] == \
        record["history"]["parts"][0]["embedding"]
    assert record["checks"]["kl_rate_above_free_bits"]["value"] == \
        record["kl_rate_above_free_bits"]
    assert record["checks"]["probe_r2_cached_features"]["value"] == \
        record["probe"]["r2"]
    assert record["probe"]["backbone"] == "pixel_ae"

    assert set(record["embedding_loss_step0"]) == set(ARMS)
    assert record["embedding_loss_step0"]["pixel_ae"] == pytest.approx(
        record["history"]["parts"][0]["embedding"], rel=1e-6)
    assert record["decision"] == script.decision(record["checks"])
    assert (out / "world_model_pixel_ae_seed0.pt").is_file()

    # Fixture sanity, so the exit-10 path is genuinely exercised: 25 steps at
    # lr 1e-4 lift a 0.03-nat dyn KL to a 0.18 peak, never past the 0.20 floor.
    assert record["checks"]["kl_rate_above_free_bits"]["passed"] is False, (
        "the fixture now passes every check, so a `return 0` in main is invisible")
    assert status == script.EXIT_SPIKE_FAILED

    text = capsys.readouterr().out
    for label in ("check 1", "check 2", "check 3", "decision: "):
        assert label in text
