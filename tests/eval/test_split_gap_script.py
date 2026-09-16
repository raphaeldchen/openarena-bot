"""scripts/split_gap.py: load as trust_horizon does, the five checks in order,
three passes per cell, one record per cell; then (Task 6) the pooling, Reading
G, the learning curves and split_gap.txt.

The script is loaded by path. The cell under test is a REAL tiny cell on
`wide_buffer` -- thirty 40-step episodes, `run_job(..., steps=5, seq_len=4,
context=2, horizon=3, device="cpu")` -- and `diagnose_dynamics.main` writes
the diagnostic the self-check needs, so `max |delta| == 0.0` is asserted
against a diagnostic the ladder really wrote on this machine.

THE FIXTURE'S FACTS: `episode_split(30, 0.2, seed=0)` holds out SIX; the 24
training episodes split as probe_episodes' 20 + 4; `window_starts(40, 2, 3)`
cuts EIGHT windows per episode, so the strata carry 48 / 32 / 160 windows over
6 / 4 / 20 clusters. `pos_x = 3t`, `pos_y = -2t`: every window moves
sqrt(13) = 3.61 units per step, under the 5-unit threshold at h=1, over it
from h=2, so `first_moved` is 2 everywhere and no window never moves.
"""

import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import mbfps.eval.study as study
from mbfps.eval.probe import PROBE_EPISODE_LIMIT
from mbfps.eval.split_gap import CURVE_NAMES, STRATA
from mbfps.eval.study import StudyJob, job_record_path, load_record, run_job
from mbfps.utils.config import ARMS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("split_gap")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
WINDOWS_PER_EPISODE = 8
N_EPISODES = {"val": 6, "train_held": 4, "train_probe": PROBE_EPISODE_LIMIT}
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
SPLIT_GAP = "split_gap_random_vit_seed1.json"

EXPECTED_KEYS = {
    "arm", "seed", "context", "horizon", "split_seed", "device", "torch_version",
    "git_sha", "checkpoint_git_sha", "probe", "episodes", "self_check", "strata",
    "nonfinite",
}
STRATUM_KEYS = {
    "windows", "curves", "band", "moved", "first_moved", "crossing", "margin",
    "survival", "trust_horizon", "counts",
}


@pytest.fixture
def cell(tmp_path, wide_buffer, capsys):
    """One real cell on thirty episodes and the diagnostic the ladder writes for it."""
    out = tmp_path / "out"
    run_job(JOB, wide_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(wide_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=wide_buffer.root)


@pytest.fixture
def narrow_cell(tmp_path, small_buffer, capsys):
    """The SIX-episode cell: five training episodes, all of them the probe's.
    The strata cannot be built and the script must say so."""
    out = tmp_path / "out"
    run_job(JOB, small_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(small_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=small_buffer.root)


def _argv(cell, *extra: str) -> list[str]:
    return [
        "--out", str(cell.out), "--data", str(cell.data), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed), *extra,
    ]


def _doctor(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_refit(*args, **kwargs):
    """Patched onto `script._trust`: the refit lives in trust_horizon's
    `prepare_cell`, which split_gap calls by path."""
    raise AssertionError("fit_probes ran; this refusal must come before the refit")


FAULTS = {
    "diagnostic": lambda cell: (cell.out / DIAGNOSTIC).unlink(),
    "split": lambda cell: _doctor(
        cell.out / RECORD,
        lambda r: r["episodes"].__setitem__("val", ["ep_000099_len00040.npz"]),
    ),
    "record": lambda cell: _doctor(
        cell.out / RECORD,
        lambda r: r["curves"]["rssm_position"].__setitem__(
            0, r["curves"]["rssm_position"][0] + 6.5
        ),
    ),
    "self_check": lambda cell: _doctor(
        cell.out / DIAGNOSTIC,
        lambda d: d["curves"]["reference_position"].__setitem__(
            0, d["curves"]["reference_position"][0] + 1e-3
        ),
    ),
}


# ---------------------------------------------------------------------------
# The parser and the statuses.
# ---------------------------------------------------------------------------


def test_the_parser_defaults_match_trust_horizons_and_add_the_figure_and_window():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3_study_v2") and args.data == Path("data/my_way_home")
    assert args.device == "mps"
    assert args.context is None and args.horizon is None
    assert args.arms == list(ARMS) and args.seeds == [0, 1, 2]
    assert args.figure is None, "None -> <out>/learning_curves.png"
    assert args.window == 100
    with pytest.raises(SystemExit) as refused:
        script._parser().parse_args(["--arms", "cnn"])
    assert refused.value.code == 2


def test_the_reused_statuses_are_trust_horizons_and_thirty_one_is_new():
    assert script.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == trust.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == trust.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == trust.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == trust.EXIT_SELF_CHECK_FAILED == 30
    assert script.EXIT_STRATA_NOT_A_PARTITION == 31
    mine = {k: v for k, v in vars(script).items() if k.startswith("EXIT_")}
    assert set(mine) == {
        "EXIT_OK", "EXIT_NO_CHECKPOINTS", "EXIT_SPLIT_MISMATCH", "EXIT_RECORD_MISMATCH",
        "EXIT_SELF_CHECK_FAILED", "EXIT_STRATA_NOT_A_PARTITION",
    }
    assert len(set(mine.values())) == len(mine)
    assert 31 not in {v for k, v in vars(trust).items() if k.startswith("EXIT_")}
    assert 31 not in {v for k, v in vars(diagnose).items() if k.startswith("EXIT_")}


# ---------------------------------------------------------------------------
# A clean cell.
# ---------------------------------------------------------------------------


def test_a_clean_cell_exits_ok_and_writes_a_record_with_three_strata(cell):
    assert script.main(_argv(cell)) == script.EXIT_OK
    path = cell.out / SPLIT_GAP
    assert path.exists()
    record = load_record(path)
    assert set(record) == EXPECTED_KEYS
    assert (record["arm"], record["seed"]) == (JOB.arm, JOB.seed)
    assert (record["context"], record["horizon"], record["split_seed"]) == (CONTEXT, HORIZON, 0)
    assert record["device"] == "cpu" and record["torch_version"] == torch.__version__
    assert record["git_sha"] == study.git_sha()
    study_record = load_record(job_record_path(cell.out, JOB))
    assert record["checkpoint_git_sha"] == study_record["git_sha"]
    assert record["self_check"]["ok"] is True
    assert record["self_check"]["reference_position_max_delta"] == 0.0
    assert record["self_check"]["persistence_position_max_delta"] == 0.0

    episodes = record["episodes"]
    assert list(episodes) == list(STRATA)
    assert {k: len(v) for k, v in episodes.items()} == N_EPISODES
    assert episodes["val"] == study_record["episodes"]["val"]
    names = [n for v in episodes.values() for n in v]
    assert len(names) == len(set(names)) == 30, "a partition: disjoint and complete"

    for stratum in STRATA:
        block = record["strata"][stratum]
        assert set(block) == STRATUM_KEYS, stratum
        n = N_EPISODES[stratum] * WINDOWS_PER_EPISODE
        assert block["windows"] == {
            "total": n,
            "episode": [e for e in range(N_EPISODES[stratum]) for _ in range(WINDOWS_PER_EPISODE)],
            "clusters": N_EPISODES[stratum],
        }
        for channel in ("probe", "free"):
            assert len(block["crossing"][channel]) == n
            assert np.asarray(block["margin"][channel], dtype=float).shape == (n, HORIZON)
            assert len(block["survival"][channel]) == HORIZON + 1
            assert set(block["trust_horizon"][channel]) == {"q50", "q75", "q90"}
        assert set(block["curves"]) == set(CURVE_NAMES)
        assert all(len(block["curves"][name]) == HORIZON for name in CURVE_NAMES)
        assert set(block["band"]) == {"position", "angle"}
        assert "gap_final" in block["band"]["position"]
        assert np.asarray(block["moved"], dtype=bool).shape == (n, HORIZON)
        assert block["first_moved"] == [2.0] * n, "sqrt(13) per step: moved from h=2"
        assert block["counts"]["never_moved"] == 0
        assert block["counts"]["not_moved"] == [n, 0, 0]
    # The val stratum's band IS the study record's curve, bitwise.
    assert record["strata"]["val"]["curves"]["rssm_position"] == study_record["curves"]["rssm_position"]
    assert record["strata"]["val"]["curves"]["persistence_position"] == study_record["curves"]["persistence_position"]


def test_the_strata_are_probe_episodes_split_of_the_train_list(cell):
    """train_probe is the first PROBE_EPISODE_LIMIT training episodes in the
    split's order, train_held the rest -- the rule `fit_probes` used."""
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split

    assert script.main(_argv(cell)) == script.EXIT_OK
    record = load_record(cell.out / SPLIT_GAP)
    train, val = episode_split(
        ReplayBuffer(cell.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=0,
    )
    assert record["episodes"]["train_probe"] == [p.name for p in train[:PROBE_EPISODE_LIMIT]]
    assert record["episodes"]["train_held"] == [p.name for p in train[PROBE_EPISODE_LIMIT:]]
    assert record["episodes"]["val"] == [p.name for p in val]


def test_the_per_cell_line_names_the_strata_and_the_self_check(cell, capsys):
    assert script.main(_argv(cell)) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "random_vit seed 1:" in out
    assert "val 48 windows / 6 episodes" in out
    assert "train_held 32 windows / 4 episodes" in out
    assert "train_probe 160 windows / 20 episodes" in out
    assert "self-check max|delta| reference 0.0e+00 persistence 0.0e+00" in out


# ---------------------------------------------------------------------------
# The refusals, in order.
# ---------------------------------------------------------------------------


def test_a_missing_diagnostic_is_exit_11(cell, capsys):
    FAULTS["diagnostic"](cell)
    assert script.main(_argv(cell)) == script.EXIT_NO_CHECKPOINTS
    assert DIAGNOSTIC in capsys.readouterr().out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_six_episode_buffer_is_exit_31_before_any_refit(narrow_cell, capsys, monkeypatch):
    """Five training episodes, all the probe's: no train_held. Judged once,
    up front, before the first probe refit."""
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    assert script.main(_argv(narrow_cell)) == script.EXIT_STRATA_NOT_A_PARTITION
    out = capsys.readouterr().out
    assert "STRATA NOT A PARTITION" in out and "train_held is empty" in out
    assert not (narrow_cell.out / SPLIT_GAP).exists()


def test_a_probe_rule_that_disagrees_with_the_split_is_exit_31(cell, capsys, monkeypatch):
    """`probe_episodes` returning a `used` that is not the leading block of
    train -- the drift the partition check exists for."""
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    real = script.probe_episodes

    def skewed(paths, limit=PROBE_EPISODE_LIMIT):
        used, held = real(paths, limit)
        return used[1:] + held[:1], held[1:] + used[:1]

    monkeypatch.setattr(script, "probe_episodes", skewed)
    assert script.main(_argv(cell)) == script.EXIT_STRATA_NOT_A_PARTITION
    assert "leading block" in capsys.readouterr().out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_split_that_is_not_the_records_is_exit_12_before_the_refit(cell, capsys, monkeypatch):
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    FAULTS["split"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SPLIT_MISMATCH
    assert "SPLIT MISMATCH for random_vit seed 1" in capsys.readouterr().out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_protocol_that_disagrees_with_the_diagnostic_is_exit_14_before_the_refit(cell, capsys, monkeypatch):
    monkeypatch.setattr(script._trust, "fit_probes", _never_refit)
    assert script.main(_argv(cell, "--context", "3")) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out and "--context 3" in out


def test_a_study_record_the_rollout_no_longer_reproduces_is_exit_14(cell, capsys):
    FAULTS["record"](cell)
    assert script.main(_argv(cell)) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out and "device=cpu" in out
    assert not (cell.out / SPLIT_GAP).exists()


def test_a_doctored_reference_curve_is_exit_30_and_no_record_is_written(cell, capsys):
    FAULTS["self_check"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SELF_CHECK_FAILED
    out = capsys.readouterr().out
    assert "SELF-CHECK FAILED for random_vit seed 1" in out and "reference_position" in out
    assert not (cell.out / SPLIT_GAP).exists()


@pytest.mark.parametrize(
    "earlier, later, status",
    [
        ("split", "self_check", "EXIT_SPLIT_MISMATCH"),
        ("record", "self_check", "EXIT_RECORD_MISMATCH"),
        ("diagnostic", "split", "EXIT_NO_CHECKPOINTS"),
    ],
)
def test_the_checks_are_judged_in_the_order_11_31_12_14_30(cell, earlier, later, status):
    FAULTS[earlier](cell)
    FAULTS[later](cell)
    assert script.main(_argv(cell)) == getattr(script, status)
    assert not (cell.out / SPLIT_GAP).exists()


def test_an_absent_requested_cell_is_exit_11(cell, capsys):
    base = ["--out", str(cell.out), "--data", str(cell.data), "--device", "cpu"]
    assert script.main(base) == script.EXIT_NO_CHECKPOINTS
    assert "pixel_ae seed 0" in capsys.readouterr().out
