"""scripts/trust_horizon.py, part 1: load as diagnose_dynamics does, the four
checks in order, one reference pass, one record per cell.

The script is loaded by path, the way `test_diagnose_dynamics_script.py`
loads its script. Unlike that file, nothing here stubs the model: the cell
under test is the same tiny cell `test_aggregate.py` trains --
`run_job(..., steps=5, seq_len=4, context=2, horizon=3, device="cpu")` on the
shared `small_buffer` -- and `diagnose_dynamics.main` writes the diagnostic
for it, so the self-check's `max |delta| == 0.0` is asserted against a
diagnostic the ladder really wrote on the same machine, and every refusal is
produced by doctoring ONE file the way a real drift would.

THE FIXTURE'S FACTS, read off `conftest.small_buffer`, not off the script:
six 40-step episodes; `episode_split(val_fraction=0.2, seed=0)` holds out
ONE, so every window carries episode label 0, the folds of the scale
correction are unavailable and every clustered SE is NaN; `window_starts(40,
2, 3)` = range(0, 36, 5) cuts EIGHT windows from it; `pos_x = 3t`,
`pos_y = -2t`, so the true displacement over h steps is h * sqrt(13) = 3.61,
7.21, 10.82 map units -- below the 5-unit moved threshold at h=1, above it
from h=2 on, in every window.

The pure pieces -- `self_check`, `trust_record`, `write_trust_record` -- are
also driven from a fabricated `Trajectories` with answers worked by hand, so
a swapped argument (persistence for model error, the probe's channel for the
embedding's) is caught by a value and not by a shape.
"""

import dataclasses
import importlib.util
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import mbfps.eval.study as study
from mbfps.eval.diagnostics import Trajectories
from mbfps.eval.study import StudyJob, job_record_path, load_record, run_job
from mbfps.utils.config import ARMS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("trust_horizon")
diagnose = _load_script("diagnose_dynamics")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
"""`test_aggregate.py`'s REAL_JOB / REAL_JOB_KW: `random_vit` because its
backbone needs no downloaded weights; context 2 / horizon 3 because
`window_starts(40, 5, 45)` is empty on 40-step episodes."""
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
N_WINDOWS = 8
"""`window_starts(40, 2, 3)` = range(0, 36, 5) -> starts 0, 5, ..., 35."""
RECORD = "result_random_vit_seed1.json"
DIAGNOSTIC = "diagnostic_random_vit_seed1.json"
TRUST = "trust_random_vit_seed1.json"

EXPECTED_KEYS = {
    "arm", "seed", "context", "horizon", "split_seed", "device", "torch_version",
    "git_sha", "episodes", "windows", "probe", "self_check", "crossing", "margin",
    "ratio_probe", "ratio_raw", "ratio_free", "cosine", "displacement", "scale", "counts",
    "nonfinite",
}
"""The record's top-level keys, exactly. `nonfinite` is `write_record`'s map
and is on disk only; the live dict `trust_record` returns must NOT carry it
(`to_json_record` refuses a record that already does). `displacement` holds
the four norms the pooling's ratio of medians is built from."""


@pytest.fixture
def cell(tmp_path, small_buffer, capsys):
    """One real cell and the diagnostic the ladder writes for it.

    ~3 s: five training steps, the probe refit, and `diagnose_dynamics.main`
    with `--ks 1 3` (the sweep insists the horizon is among its ks). The
    ladder's own output is drained so a test reading stdout sees only
    `trust_horizon`'s.
    """
    out = tmp_path / "out"
    run_job(JOB, small_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(small_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    assert (out / DIAGNOSTIC).exists()
    capsys.readouterr()
    return types.SimpleNamespace(out=out, data=small_buffer.root)


def _argv(cell, *extra: str) -> list[str]:
    return [
        "--out", str(cell.out), "--data", str(cell.data), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed), *extra,
    ]


def _doctor(path: Path, edit) -> None:
    """Edit one JSON file in place through plain `json`, so the record's
    `nonfinite` map is carried through untouched and nothing re-sanitises."""
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _never_refit(*args, **kwargs):
    raise AssertionError("fit_probes ran; this refusal must come before the refit")


# One fault per file a real drift would touch. Each is applied to a fresh
# fixture, and the ordering test applies two at once.
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


def test_the_parser_defaults_to_the_study_directory_and_reads_the_protocol_from_the_diagnostic():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3_study_v2") and isinstance(args.out, Path)
    assert args.data == Path("data/my_way_home") and isinstance(args.data, Path)
    assert args.device == "mps"
    assert args.context is None and args.horizon is None, (
        "context and horizon default to whatever each cell's diagnostic was written at")
    assert args.arms == list(ARMS) == ["pixel_ae", "frozen_ssl", "random_vit"]
    assert args.seeds == [0, 1, 2]
    explicit = script._parser().parse_args(["--context", "2", "--horizon", "3", "--seeds", "1"])
    assert (explicit.context, explicit.horizon, explicit.seeds) == (2, 3, [1])
    with pytest.raises(SystemExit) as refused:
        script._parser().parse_args(["--arms", "cnn"])
    assert refused.value.code == 2


def test_the_reused_statuses_are_diagnose_dynamics_own_and_thirty_is_new():
    """11 / 12 / 14 carry `diagnose_dynamics.py`'s meanings, so a wrapper
    learns the same thing from either tool; 30 is this script's alone."""
    assert script.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == diagnose.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == diagnose.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == diagnose.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == 30
    mine = {k: v for k, v in vars(script).items() if k.startswith("EXIT_")}
    assert set(mine) == {
        "EXIT_OK", "EXIT_NO_CHECKPOINTS", "EXIT_SPLIT_MISMATCH",
        "EXIT_RECORD_MISMATCH", "EXIT_SELF_CHECK_FAILED",
    }
    assert len(set(mine.values())) == len(mine)
    theirs = {v for k, v in vars(diagnose).items() if k.startswith("EXIT_")}
    assert 30 not in theirs and 1 not in mine.values() and 2 not in mine.values()


# ---------------------------------------------------------------------------
# self_check, on fabricated trajectories.
# ---------------------------------------------------------------------------


def _fabricated() -> tuple[Trajectories, dict]:
    """Two windows, two steps, two episodes. Window 0 is a PERFECT predictor
    (imagined position == truth, floor == truth); window 1 is a PERSISTENCE
    CLONE (imagined position held at p_hat(0)). Every number below is chosen
    so the two rows have different signatures on every channel.

    Truth: window 0 moves (6, 0) then (12, 0) from (0, 0); window 1 moves
    (0, 6) then (0, 12) from (10, 10). Both are >= 5 map units at every h,
    so `moved` is all True.

      model error      row 0 [0, 0]    row 1 [6, 12]
      persistence err  row 0 [6, 12]   row 1 [6, 12]
      -> crossing (probe): row 0 never (3 = H + 1); row 1 ties never cross (3)
      -> margin: row 0 [6, 12]; row 1 [0, 0]
      -> ratio_probe / ratio_raw: row 0 [1, 1]; row 1 [0, 0]
      -> cosine: row 0 [1, 1]; row 1 NaN (|d_hat| = 0), counted as zero_displacement
      -> displacement: probe_hat row 0 [6, 12], row 1 [0, 0]; probe_real [6, 12] on
         both rows (the floor is the truth); free_hat / free_true are the two
         embedding norms below, as they are

    Embedding channel, chosen to DIFFER from the probe channel's answers:
      D_hat  row 0 [0, 0]   row 1 [5, 5]      (distance of e_hat(h) to e(h))
      D_0    row 0 [3, 4]   row 1 [3, 4]      (distance of e_hat(0) to e(h))
      -> crossing (free): row 0 never (3); row 1 5 > 3 at h=1 -> 1
      ||e_hat(h) - e_hat(0)||  row 0 [2, 4]   row 1 [1, 2]
      ||e(h) - e(0)||          row 0 [2, 4]   row 1 [2, 4]
      -> ratio_free: row 0 [1, 1]; row 1 [0.5, 0.5]

    Scale correction (episode labels 0 and 1 -> fold A = row 0, fold B = row 1):
      alpha_a fit on row 0: |alpha * d - d| = |alpha - 1| * |d| -> 1.0 at both h
      alpha_b fit on row 1: d_hat = 0, the error is |p_hat0 - p| whatever alpha
        -> a tie over the whole grid -> the smallest alpha, 0.0 -> BOUNDARY
      score_a (fit A, scored on row 1 with alpha 1): [6, 12]
      score_b (fit B, scored on row 0 with alpha 0): [6, 12]
      held_out: row 0 with alpha_b = 0 -> [6, 12]; row 1 with alpha_a = 1 -> [6, 12]
      boundary [True, True]; folds_available True

    The diagnostic's curves are the means of the rows above -- reference
    [3, 6], persistence [6, 12] -- and its floor [1, 1] gives a positive
    persistence-to-floor band (12 - 1 = 11) at the final step: measurable.
    """
    true_at_context = np.array([[0.0, 0.0], [10.0, 10.0]])
    true_positions = np.array([[[6.0, 0.0], [12.0, 0.0]], [[10.0, 16.0], [10.0, 22.0]]])
    positions_at_context = true_at_context.copy()
    positions = np.array([[[6.0, 0.0], [12.0, 0.0]], [[10.0, 10.0], [10.0, 10.0]]])
    traj = Trajectories(
        positions=positions,
        positions_at_context=positions_at_context,
        positions_real=true_positions.copy(),
        true_positions=true_positions,
        true_at_context=true_at_context,
        embedding_distance_to_truth=np.array([[0.0, 0.0], [5.0, 5.0]]),
        embedding_persistence_distance=np.array([[3.0, 4.0], [3.0, 4.0]]),
        embedding_displacement=np.array([[2.0, 4.0], [1.0, 2.0]]),
        true_embedding_displacement=np.array([[2.0, 4.0], [2.0, 4.0]]),
        window_episode=np.array([0, 1]),
        windows_total=2,
        reference_position=np.array([3.0, 6.0]),
        persistence_position=np.array([6.0, 12.0]),
    )
    diagnostic = {
        "context": 2,
        "horizon": 2,
        "episodes": {"val": ["ep_a.npz", "ep_b.npz"]},
        "probe": {"embedding_selection_r2": 0.31},
        "windows": {"total": 2, "episode": [0, 1]},
        "curves": {
            "reference_position": [3.0, 6.0],
            "persistence_position": [6.0, 12.0],
            "floor_position": [1.0, 1.0],
        },
    }
    return traj, diagnostic


def _with_curve(diagnostic: dict, name: str, values) -> dict:
    doctored = json.loads(json.dumps(diagnostic))
    doctored["curves"][name] = list(values)
    return doctored


def _with_windows(diagnostic: dict, total=None, episode="keep") -> dict:
    doctored = json.loads(json.dumps(diagnostic))
    if total is not None:
        doctored["windows"]["total"] = total
    if episode != "keep":
        doctored["windows"]["episode"] = episode
    return doctored


def test_self_check_reads_zero_when_the_trajectories_reproduce_the_diagnostic():
    traj, diagnostic = _fabricated()
    check = script.self_check(traj, diagnostic)
    assert check.ok is True
    assert check.failures() == []
    assert check.reference_position_max_delta == 0.0
    assert check.persistence_position_max_delta == 0.0
    assert check.windows_total_match is True and check.windows_episode_match is True
    assert check.record() == {
        "reference_position_max_delta": 0.0,
        "persistence_position_max_delta": 0.0,
        "windows_total_match": True,
        "windows_episode_match": True,
        "ok": True,
    }


def test_self_check_names_the_curve_and_the_step_of_the_largest_delta():
    """The rule is `max |delta| == 0.0`, the ladder's `record_reproduction`
    rule: a delta of 2^-40 (exactly representable beside 6.0, so the
    difference is exactly 2^-40) fails it. Each curve is judged on its own:
    the persistence curve doctored alone fails too, and the step named is
    the 1-based horizon step of the largest delta."""
    traj, diagnostic = _fabricated()
    tiny = 2.0 ** -40
    reference_only = script.self_check(
        traj, _with_curve(diagnostic, "reference_position", [3.0, 6.0 + tiny])
    )
    assert reference_only.ok is False
    assert reference_only.reference_position_max_delta == tiny
    assert reference_only.reference_position_step == 2
    assert reference_only.persistence_position_max_delta == 0.0
    assert reference_only.record()["ok"] is False
    (message,) = reference_only.failures()
    assert "reference_position" in message and "step 2" in message

    persistence_only = script.self_check(
        traj, _with_curve(diagnostic, "persistence_position", [6.5, 12.0])
    )
    assert persistence_only.ok is False
    assert persistence_only.reference_position_max_delta == 0.0
    assert persistence_only.persistence_position_max_delta == 0.5
    assert persistence_only.persistence_position_step == 1
    (message,) = persistence_only.failures()
    assert "persistence_position" in message and "step 1" in message


def test_self_check_judges_the_rows_the_record_is_built_from_not_the_passs_own_curve():
    """Spec 2.3: the mean over windows of |p_hat(h) - p(h)| and of
    |p_hat(0) - p(h)| -- computed HERE from the rows the record is built
    from -- must equal the diagnostic's curves. `Trajectories.reference_position`
    is the pass's own reduction of the same rows and equals this bitwise on a
    clean run (Task 3 pins it), but it is not what is judged: a regression
    that kept the curve right and the rows wrong would otherwise write a
    record. Window 0's imagined position at step 2 is moved by one map unit
    while the pass's curve stays [3, 6]: the row mean there is (1 + 12) / 2
    = 6.5, so the check reads 0.5 at step 2 and fails. Then window 1's anchor
    is moved by (0, 2) while the pass's persistence curve stays [6, 12]: its
    persistence errors become [4, 10], the means [5, 11], a delta of 1.0 at
    both steps, the first of which is named."""
    traj, diagnostic = _fabricated()
    positions = traj.positions.copy()
    positions[0, 1, 0] += 1.0
    rows_moved = script.self_check(dataclasses.replace(traj, positions=positions), diagnostic)
    assert rows_moved.ok is False
    assert rows_moved.reference_position_max_delta == 0.5
    assert rows_moved.reference_position_step == 2
    assert rows_moved.persistence_position_max_delta == 0.0

    at_context = traj.positions_at_context.copy()
    at_context[1, 1] += 2.0
    anchor_moved = script.self_check(
        dataclasses.replace(traj, positions_at_context=at_context), diagnostic
    )
    assert anchor_moved.ok is False
    assert anchor_moved.reference_position_max_delta == 0.0
    assert anchor_moved.persistence_position_max_delta == 1.0
    assert anchor_moved.persistence_position_step == 1


def test_self_check_refuses_a_curve_of_the_wrong_length_rather_than_raising():
    """A diagnostic written at another horizon has a curve of another length;
    `np.abs(a - b)` on mismatched shapes raises, and a traceback is exit 1 --
    the status reserved for a defect, not for a record that does not match."""
    traj, diagnostic = _fabricated()
    check = script.self_check(
        traj, _with_curve(diagnostic, "reference_position", [3.0, 6.0, 9.0])
    )
    assert check.ok is False
    assert check.reference_position_max_delta == math.inf
    assert check.reference_position_step == 0
    (message,) = check.failures()
    assert "reference_position" in message and "length" in message


def test_self_check_refuses_a_window_count_or_episode_index_that_differs():
    traj, diagnostic = _fabricated()
    wrong_total = script.self_check(traj, _with_windows(diagnostic, total=3))
    assert wrong_total.windows_total_match is False
    assert wrong_total.windows_episode_match is True
    assert wrong_total.ok is False
    assert any("windows.total" in m for m in wrong_total.failures())

    wrong_index = script.self_check(traj, _with_windows(diagnostic, episode=[0, 0]))
    assert wrong_index.windows_total_match is True
    assert wrong_index.windows_episode_match is False
    assert wrong_index.ok is False
    assert any("windows.episode" in m for m in wrong_index.failures())

    # The diagnostic writes `null` when the ladder carried no clustering; a
    # trust record cannot be clustered on nothing, so that is a mismatch too.
    no_index = script.self_check(traj, _with_windows(diagnostic, episode=None))
    assert no_index.windows_episode_match is False and no_index.ok is False


# ---------------------------------------------------------------------------
# trust_record and write_trust_record, on the fabricated trajectories.
# ---------------------------------------------------------------------------


def test_trust_record_wires_every_channel_the_way_the_spec_names_it():
    traj, diagnostic = _fabricated()
    record = script.trust_record(
        "frozen_ssl", 2, traj, diagnostic, context=2, horizon=2, device=torch.device("cpu")
    )
    assert set(record) == EXPECTED_KEYS - {"nonfinite"}, (
        "the live record must not carry `nonfinite`; write_record adds it")

    for key, value in {
        "arm": "frozen_ssl", "seed": 2, "context": 2, "horizon": 2, "split_seed": 0,
        "device": "cpu", "torch_version": torch.__version__, "git_sha": study.git_sha(),
        "episodes": {"val": ["ep_a.npz", "ep_b.npz"]},
        "windows": {"total": 2, "episode": [0, 1]},
        "probe": {"selection_r2": 0.31, "measurable": True},
        "self_check": {
            "reference_position_max_delta": 0.0, "persistence_position_max_delta": 0.0,
            "windows_total_match": True, "windows_episode_match": True, "ok": True,
        },
    }.items():
        assert record[key] == value, key

    equal = np.testing.assert_array_equal  # NaN == NaN by position
    equal(record["crossing"]["probe"], [3.0, 3.0])
    equal(record["crossing"]["free"], [3.0, 1.0])
    equal(record["margin"], [[6.0, 12.0], [0.0, 0.0]])
    equal(record["ratio_probe"], [[1.0, 1.0], [0.0, 0.0]])
    equal(record["ratio_raw"], [[1.0, 1.0], [0.0, 0.0]])
    equal(record["ratio_free"], [[1.0, 1.0], [0.5, 0.5]])
    equal(record["cosine"], [[1.0, 1.0], [np.nan, np.nan]])

    # The four norms the pooling's ratio of medians is built from: the
    # probe's imagined and REAL displacements (the floor is the truth here,
    # so row 1's real displacement is its true one), and the two embedding
    # norms as the pass reduced them.
    displacement = record["displacement"]
    assert set(displacement) == {"probe_hat", "probe_real", "free_hat", "free_true"}
    equal(displacement["probe_hat"], [[6.0, 12.0], [0.0, 0.0]])
    equal(displacement["probe_real"], [[6.0, 12.0], [6.0, 12.0]])
    equal(displacement["free_hat"], [[2.0, 4.0], [1.0, 2.0]])
    equal(displacement["free_true"], [[2.0, 4.0], [2.0, 4.0]])

    scale = record["scale"]
    assert set(scale) == {
        "alpha_a", "alpha_b", "score_a", "score_b", "held_out", "boundary", "folds_available",
    }
    equal(scale["alpha_a"], [1.0, 1.0])
    equal(scale["alpha_b"], [0.0, 0.0])
    equal(scale["score_a"], [6.0, 12.0])
    equal(scale["score_b"], [6.0, 12.0])
    equal(scale["held_out"], [[6.0, 12.0], [6.0, 12.0]])
    equal(scale["boundary"], [True, True])
    assert scale["folds_available"] is True

    counts = record["counts"]
    assert set(counts) == {"not_moved", "zero_displacement", "never_moved"}
    equal(counts["not_moved"], [0, 0])
    equal(counts["zero_displacement"], [1, 1])
    assert counts["never_moved"] == 0


def test_measurability_is_the_persistence_to_floor_band_at_the_final_step():
    """The ladder's `probe_is_measurable` rule, asked with no null band: the
    band is `persistence - floor` at the LAST step and must be positive. A
    floor that equals persistence there is a zero band -- not measurable --
    and the band is read off the persistence curve, never the reference."""
    traj, diagnostic = _fabricated()

    def measurable(floor):
        return script.trust_record(
            "frozen_ssl", 2, traj, _with_curve(diagnostic, "floor_position", floor),
            context=2, horizon=2, device="cpu",
        )["probe"]["measurable"]

    # persistence [6, 12]: band 12 - 8 = 4 > 0. (Off the reference curve,
    # [3, 6], it would read 6 - 8 < 0.)
    assert measurable([1.0, 8.0]) is True
    assert measurable([1.0, 12.0]) is False   # band exactly 0
    assert measurable([1.0, 13.0]) is False   # floor above persistence


def test_write_trust_record_names_both_the_arm_and_the_seed_and_round_trips_nan(tmp_path):
    """Written through `write_record`: atomic, strict JSON, `null` for a NaN
    with the token map that restores it -- never a bare `NaN` token."""
    record = {"arm": "random_vit", "seed": 1, "margin": [[float("nan"), 2.0]]}
    path = script.write_trust_record(tmp_path, record)
    assert path == tmp_path / "trust_random_vit_seed1.json"
    assert script.trust_record_path(tmp_path, "pixel_ae", 0) == tmp_path / "trust_pixel_ae_seed0.json"
    raw = json.loads(path.read_text())
    assert raw["margin"][0][0] is None and raw["nonfinite"] == {"margin.0.0": "nan"}
    back = load_record(path)
    assert math.isnan(back["margin"][0][0]) and back["margin"][0][1] == 2.0


# ---------------------------------------------------------------------------
# main, on the real tiny cell.
# ---------------------------------------------------------------------------


def test_a_clean_cell_exits_ok_and_writes_a_record_with_every_key_and_shape(cell):
    """Explicit `--context 2 --horizon 3`, EQUAL to the diagnostic's: a value
    that matches is not a mismatch. (The next test passes neither flag and
    exercises the defaults.)"""
    assert script.main(_argv(cell, "--context", str(CONTEXT), "--horizon", str(HORIZON))) == script.EXIT_OK
    path = cell.out / TRUST
    assert path.exists()
    record = load_record(path)
    assert set(record) == EXPECTED_KEYS

    assert (record["arm"], record["seed"]) == (JOB.arm, JOB.seed)
    assert (record["context"], record["horizon"]) == (CONTEXT, HORIZON)
    assert record["split_seed"] == 0
    assert record["device"] == "cpu"
    assert record["torch_version"] == torch.__version__
    assert record["git_sha"] == study.git_sha()
    study_record = load_record(job_record_path(cell.out, JOB))
    assert record["episodes"]["val"] == study_record["episodes"]["val"]
    assert len(record["episodes"]["val"]) == 1, "six episodes at val_fraction 0.2 hold out one"
    assert record["windows"] == {"total": N_WINDOWS, "episode": [0] * N_WINDOWS}

    assert len(record["crossing"]["probe"]) == N_WINDOWS
    assert len(record["crossing"]["free"]) == N_WINDOWS
    for key in ("margin", "ratio_probe", "ratio_raw", "ratio_free", "cosine"):
        assert np.asarray(record[key], dtype=float).shape == (N_WINDOWS, HORIZON), key
    assert set(record["displacement"]) == {"probe_hat", "probe_real", "free_hat", "free_true"}
    for key in ("probe_hat", "probe_real", "free_hat", "free_true"):
        assert np.asarray(record["displacement"][key], dtype=float).shape == (N_WINDOWS, HORIZON), key
    scale = record["scale"]
    assert set(scale) == {
        "alpha_a", "alpha_b", "score_a", "score_b", "held_out", "boundary", "folds_available",
    }
    for key in ("alpha_a", "alpha_b", "score_a", "score_b", "boundary"):
        assert len(scale[key]) == HORIZON, key
    assert np.asarray(scale["held_out"], dtype=float).shape == (N_WINDOWS, HORIZON)
    assert set(record["counts"]) == {"not_moved", "zero_displacement", "never_moved"}
    assert len(record["counts"]["not_moved"]) == HORIZON
    assert len(record["counts"]["zero_displacement"]) == HORIZON

    diagnostic = load_record(cell.out / DIAGNOSTIC)
    assert set(record["probe"]) == {"selection_r2", "measurable"}
    assert record["probe"]["selection_r2"] == diagnostic["probe"]["embedding_selection_r2"]
    band = diagnostic["curves"]["persistence_position"][-1] - diagnostic["curves"]["floor_position"][-1]
    assert record["probe"]["measurable"] is (band > 0.0)


def test_the_self_check_reads_zero_the_single_episode_is_disclosed_and_h1_never_moves(cell, capsys):
    """No `--context` / `--horizon`: both come from the diagnostic."""
    assert script.main(_argv(cell)) == script.EXIT_OK
    out = capsys.readouterr().out
    record = load_record(cell.out / TRUST)

    assert record["self_check"] == {
        "reference_position_max_delta": 0.0,
        "persistence_position_max_delta": 0.0,
        "windows_total_match": True,
        "windows_episode_match": True,
        "ok": True,
    }, "same windows, same rollout, same refit probe -- or this is not what the ladder measured"

    # ONE validation episode: no second fold, so the scale correction is NaN
    # and says so, and the run discloses it rather than printing a number.
    assert record["scale"]["folds_available"] is False
    for key in ("alpha_a", "alpha_b", "score_a", "score_b"):
        assert all(math.isnan(v) for v in record["scale"][key]), key
    assert all(math.isnan(v) for row in record["scale"]["held_out"] for v in row)
    assert record["scale"]["boundary"] == [False] * HORIZON
    assert f"{N_WINDOWS} windows from 1 validation episode;" in out
    assert "folds UNAVAILABLE" in out
    assert "random_vit seed 1" in out

    # pos_x = 3t, pos_y = -2t: h * sqrt(13) = 3.61 at h=1 (< 5, not moved in
    # any window), 7.21 and 10.82 after (moved in every window).
    assert record["counts"]["not_moved"] == [N_WINDOWS, 0, 0]
    assert record["counts"]["never_moved"] == 0
    for key in ("ratio_probe", "ratio_raw", "ratio_free", "cosine"):
        assert all(math.isnan(row[0]) for row in record[key]), f"{key} is defined at h=1"
    # Every window first moves at h=2, so a crossing is 2, 3, or never (4);
    # never NaN. The margin is unmasked and finite everywhere.
    assert all(c in (2.0, 3.0, 4.0) for c in record["crossing"]["probe"])
    assert all(c in (2.0, 3.0, 4.0) for c in record["crossing"]["free"])
    assert all(math.isfinite(v) for row in record["margin"] for v in row)


def test_a_missing_diagnostic_is_exit_11_naming_the_file(cell, capsys):
    FAULTS["diagnostic"](cell)
    assert script.main(_argv(cell)) == script.EXIT_NO_CHECKPOINTS
    out = capsys.readouterr().out
    assert DIAGNOSTIC in out and "random_vit seed 1" in out
    assert not (cell.out / TRUST).exists()


def test_a_doctored_reference_curve_is_exit_30_and_no_record_is_written(cell, capsys):
    """One value of `curves.reference_position` moved by 1e-3: the trust pass
    no longer reproduces the ladder's curve, the cell, the curve and the step
    are named, and NOTHING is written -- a record beside a failed self-check
    would be pooled by the next part as if it measured what the ladder did."""
    FAULTS["self_check"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SELF_CHECK_FAILED
    out = capsys.readouterr().out
    assert "SELF-CHECK FAILED for random_vit seed 1" in out
    assert "reference_position" in out and "step 1" in out
    assert not (cell.out / TRUST).exists()


def test_a_context_or_horizon_that_disagrees_with_the_diagnostic_is_refused_before_the_refit(
    cell, capsys, monkeypatch
):
    """Refused as `diagnose_dynamics.py` refuses it -- a RECORD MISMATCH, 14:
    a rollout at another context cannot reproduce the record's curve -- but
    up front, against the protocol the diagnostic says it was written at,
    not twenty seconds later after a probe refit at the wrong depth."""
    monkeypatch.setattr(script, "fit_probes", _never_refit)
    assert script.main(_argv(cell, "--context", "3")) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out
    assert "--context 3" in out and f"({CONTEXT})" in out
    assert script.main(_argv(cell, "--horizon", "4")) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "--horizon 4" in out and f"({HORIZON})" in out
    assert not (cell.out / TRUST).exists()


def test_a_split_that_is_not_the_records_is_exit_12_before_the_refit(cell, capsys, monkeypatch):
    monkeypatch.setattr(script, "fit_probes", _never_refit)
    FAULTS["split"](cell)
    assert script.main(_argv(cell)) == script.EXIT_SPLIT_MISMATCH
    out = capsys.readouterr().out
    assert "SPLIT MISMATCH for random_vit seed 1" in out
    assert "ep_000099_len00040.npz" in out, "the record's names are printed beside the split's"
    assert not (cell.out / TRUST).exists()


def test_a_study_record_the_rollout_no_longer_reproduces_is_exit_14(cell, capsys):
    """`curves.rssm_position` moved by 6.5 map units -- the size of the cpu
    miss `diagnose_dynamics.py` documents -- and the environment is named.
    The spec's 'wrong device' case cannot be staged on a cpu-trained cell;
    the doctored record produces the same status through the same
    `_max_delta` branch, and Task 7's pre-flight item 6 covers the device on
    the real run."""
    FAULTS["record"](cell)
    assert script.main(_argv(cell)) == script.EXIT_RECORD_MISMATCH
    out = capsys.readouterr().out
    assert "RECORD MISMATCH for random_vit seed 1" in out
    assert "device=cpu" in out
    assert not (cell.out / TRUST).exists()


@pytest.mark.parametrize(
    "earlier, later, status",
    [
        ("split", "self_check", "EXIT_SPLIT_MISMATCH"),
        ("record", "self_check", "EXIT_RECORD_MISMATCH"),
        ("diagnostic", "split", "EXIT_NO_CHECKPOINTS"),
    ],
)
def test_the_checks_are_judged_in_the_order_11_12_14_30(cell, earlier, later, status):
    """Two faults at once, and the EARLIER check's status is the one reported:
    the order separates a missing cell from a wrong split from an environment
    difference from a trust-pass defect, and a later status reported for an
    earlier fault sends the reader to the wrong place."""
    FAULTS[earlier](cell)
    FAULTS[later](cell)
    assert script.main(_argv(cell)) == getattr(script, status)
    assert not (cell.out / TRUST).exists()


def test_arms_and_seeds_select_the_cells_and_an_absent_cell_is_exit_11(cell, capsys):
    """Every REQUESTED cell must be present -- the readings need all of them --
    so the default nine on a directory holding one cell is 11, naming the
    first absent cell in `ARMS x seeds` order, before any refit."""
    base = ["--out", str(cell.out), "--data", str(cell.data), "--device", "cpu"]
    assert script.main(base) == script.EXIT_NO_CHECKPOINTS
    assert "pixel_ae seed 0" in capsys.readouterr().out
    assert script.main(base + ["--arms", "random_vit"]) == script.EXIT_NO_CHECKPOINTS
    assert "random_vit seed 0" in capsys.readouterr().out
    assert not (cell.out / TRUST).exists()
    assert script.main(base + ["--arms", "random_vit", "--seeds", "1"]) == script.EXIT_OK
    assert (cell.out / TRUST).exists()
    with pytest.raises(SystemExit) as refused:
        script.main(base + ["--arms", "cnn"])
    assert refused.value.code == 2
