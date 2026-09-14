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


# ---------------------------------------------------------------------------
# Pooling glue, the two readings, trust.txt  (Task 6)
# ---------------------------------------------------------------------------
# Nothing statistical lives in the glue: every estimator is `pooling.py`'s and
# every rule is `trust_readings.py`'s. What can go wrong here is WIRING -- the
# wrong series under the wrong mask into the right estimator -- so the tests
# below hand `pooled_inputs` six fabricated windows from four episodes, two
# seeds, three arms, at h = 3, and assert every field of `ReadingOneInputs`
# against a number worked by hand from the same six windows (the clustered
# standard error is `sqrt(G / (G - 1) * sum_g (sum of residuals in g)^2) / n`,
# `pooling.cluster_standard_error`'s own definition, written out per test).

# Task 4's header stays as it is: `script`, `diagnose`, `N_WINDOWS` (its eight
# fixture windows), the `cell` fixture, `_argv`, and the imports `math`,
# `types`, `Path`, `np`, `pytest`, `StudyJob`, `run_job` are all reused, and
# only the genuinely new imports are added.
import itertools

import mbfps.eval.pooling as pooling
from mbfps.eval.trust_readings import (
    ARMS_ORDER,
    CONTROL,
    FAMILY,
    TREATMENT,
    Contrast,
    Ratio,
    ReadingOneInputs,
)

NAN = float("nan")
N_FAB = 6
"""Six fabricated windows -- NOT Task 4's `N_WINDOWS`, which is the fixture's
eight and is read by Task 4's tests at import time."""
H_FAB = 3
HI = H_FAB - 1
EPISODE = [0, 0, 1, 1, 2, 3]
"""Four episodes over six windows: fold A (even labels) is episodes 0 and 2,
windows {0, 1, 4}; fold B (odd) is episodes 1 and 3, windows {2, 3, 5}. Four,
not two, so that each fold still has two clusters after window 0 is dropped."""
MOVED = [False, True, True, True, True, True]
"""Window 0 did not move at h = 3 in any cell; every probe-based series is NaN
there and `margin` -- finite everywhere by construction -- must be masked."""


def _tile(column):
    """One per-window column at h = 3, repeated to every step: the glue reads
    column `h - 1` only, and a value that leaked from another column would be
    the same value, so the tiling hides nothing the tests care about."""
    return np.tile(np.asarray(column, dtype=float)[:, None], (1, H_FAB)).tolist()


def _fab_record(arm, seed, *, margin, cosine, held, boundary, crossing_probe, crossing_free,
                probe_hat, probe_real, free_hat, free_true, measurable=True, r2=0.3):
    """A trust record in the contract's shape, from per-window values at h = 3."""
    moved = np.asarray(MOVED)
    hat, real = np.asarray(probe_hat, float), np.asarray(probe_real, float)
    e_hat, e_true = np.asarray(free_hat, float), np.asarray(free_true, float)
    return {
        "arm": arm, "seed": seed, "context": 2, "horizon": H_FAB, "split_seed": 0,
        "device": "cpu", "torch_version": "2.x", "git_sha": "0" * 7,
        "episodes": {"val": ["ep_a.npz", "ep_b.npz", "ep_c.npz", "ep_d.npz"]},
        "windows": {"total": N_FAB, "episode": list(EPISODE)},
        "probe": {"selection_r2": r2, "measurable": measurable},
        "self_check": {
            "reference_position_max_delta": 0.0, "persistence_position_max_delta": 0.0,
            "windows_total_match": True, "windows_episode_match": True, "ok": True,
        },
        "crossing": {"probe": list(crossing_probe), "free": list(crossing_free)},
        "margin": _tile(margin),
        "ratio_probe": _tile(np.where(moved, hat / real, NAN)),
        "ratio_raw": _tile(np.where(moved, 1.0, NAN)),
        "ratio_free": _tile(np.where(moved, e_hat / e_true, NAN)),
        "cosine": _tile(cosine),
        "scale": {
            "alpha_a": [1.0] * H_FAB, "alpha_b": [1.0] * H_FAB,
            "score_a": [0.0] * H_FAB, "score_b": [0.0] * H_FAB,
            "held_out": _tile(held), "boundary": list(boundary), "folds_available": True,
        },
        "displacement": {
            "probe_hat": _tile(hat), "probe_real": _tile(real),
            "free_hat": _tile(e_hat), "free_true": _tile(e_true),
        },
        "counts": {"not_moved": [1] * H_FAB, "zero_displacement": [0] * H_FAB, "never_moved": 0},
        "nonfinite": {},
    }


def _fab_records():
    """Three arms x two seeds. The numbers are chosen so every pooled field has
    a closed-form answer; each test names the ones it reads."""
    ones = [1.0] * N_FAB
    return {
        ("pixel_ae", 0): _fab_record(
            "pixel_ae", 0, margin=[3] * 6, cosine=[0.5] * 6, held=[10] * 6,
            boundary=[False] * 3, crossing_probe=[3] * 6, crossing_free=[4] * 6,
            probe_hat=[6] * 6, probe_real=[3] * 6, free_hat=[1] * 6, free_true=[2] * 6,
        ),
        ("pixel_ae", 1): _fab_record(
            "pixel_ae", 1, margin=[3] * 6, cosine=[0.5] * 6, held=[10] * 6,
            boundary=[False] * 3, crossing_probe=[3] * 6, crossing_free=[4] * 6,
            probe_hat=[2] * 6, probe_real=[1] * 6, free_hat=[2] * 6, free_true=[4] * 6,
        ),
        ("frozen_ssl", 0): _fab_record(
            "frozen_ssl", 0, margin=[50, 4, 2, 2, 4, 3], cosine=[1.0] * 6, held=[9] * 6,
            boundary=[False] * 3, crossing_probe=[4, 4, 2, NAN, 3, 4], crossing_free=[4] * 6,
            probe_hat=[100, 1, 1, 3, 3, 3], probe_real=[100, 2, 2, 2, 2, 2],
            free_hat=[3] * 6, free_true=[2] * 6,
        ),
        ("frozen_ssl", 1): _fab_record(
            "frozen_ssl", 1, margin=[52, 6, 4, 4, 6, 5],
            cosine=[0.5, 0.5, 0.5, 0.5, 0.5, NAN], held=[11] * 6,
            boundary=[False, False, True], crossing_probe=[4, 1, 2, 2, 3, 4],
            crossing_free=[4] * 6, probe_hat=[100, 2, 2, 2, 6, 6],
            probe_real=[100, 4, 4, 4, 4, 4], free_hat=[6] * 6, free_true=[4] * 6,
        ),
        ("random_vit", 0): _fab_record(
            "random_vit", 0, margin=ones, cosine=[0.25, 0.25, 0.75, 0.75, 0.25, 0.25],
            held=[0, 12, 14, 14, 16, 16], boundary=[False] * 3, crossing_probe=[1] * 6,
            crossing_free=[4] * 6, probe_hat=[0] * 6, probe_real=[5] * 6,
            free_hat=[1] * 6, free_true=[4] * 6,
        ),
        ("random_vit", 1): _fab_record(
            "random_vit", 1, margin=ones, cosine=[0.25, 0.25, 0.75, 0.75, 0.25, 0.25],
            held=[0, 12, 14, 14, 16, 16], boundary=[False] * 3, crossing_probe=[1] * 6,
            crossing_free=[4] * 6, probe_hat=[0] * 6, probe_real=[5] * 6,
            free_hat=[1] * 6, free_true=[4] * 6,
        ),
    }


# --- cell_series -------------------------------------------------------------


def test_cell_series_carries_the_moved_mask_as_changed_and_the_records_identity():
    """`changed` is what `pool_arm` and `paired_contrast` drop windows by, so
    the mask handed in must come out as handed in -- not `window_steps_changed
    > 0`, which is the ladder's mask and not the trust pass's. The identity
    fields are what `require_compatible` refuses a pool over, so each is
    asserted to be the RECORD's, not a default."""
    record = _fab_records()[("frozen_ssl", 1)]
    values = np.array([50.0, 4.0, 2.0, 2.0, 4.0, 3.0])
    changed = np.array(MOVED)
    series = script.cell_series("frozen_ssl", 1, "margin", values, changed, record)
    assert isinstance(series, pooling.CellSeries)
    assert series.arm == "frozen_ssl" and series.seed == 1
    assert series.rung == "margin" and series.channel == "position"
    np.testing.assert_array_equal(series.delta, values)
    np.testing.assert_array_equal(series.changed, changed)
    assert series.changed.dtype == bool
    np.testing.assert_array_equal(series.episode, np.array(EPISODE))
    assert series.embedding is None and series.noise is None
    assert series.windows_total == N_FAB
    assert series.val == ("ep_a.npz", "ep_b.npz", "ep_c.npz", "ep_d.npz")
    assert series.horizon == H_FAB and series.context == 2
    assert series.device == "cpu" and series.torch_version == "2.x"


def test_cell_series_refuses_a_series_that_is_not_one_value_per_window():
    """A `(n, H)` array handed in where `(n,)` was meant would broadcast
    silently inside `np.mean` and cluster by the wrong axis; a `changed` of the
    wrong length would drop the wrong windows. Both are refused by name."""
    record = _fab_records()[("pixel_ae", 0)]
    with pytest.raises(ValueError, match="must all be"):
        script.cell_series("pixel_ae", 0, "margin", np.zeros((6, 3)), np.ones(6, bool), record)
    with pytest.raises(ValueError, match="must all be"):
        script.cell_series("pixel_ae", 0, "margin", np.zeros(6), np.ones(5, bool), record)


# --- pooled_inputs -----------------------------------------------------------


def test_pooled_inputs_pairs_the_three_arms_as_a_minus_b_and_reads_a_finite_family_threshold():
    """The three unordered pairs in `ARMS_ORDER` order, each estimate `a - b`
    on the per-window SEED-MEAN margin over the MOVED windows, clustered by
    episode. frozen_ssl - random_vit: seed means [51, 5, 3, 3, 5, 4] minus
    [1, 1, 1, 1, 1, 1], window 0 dropped (not moved), leaves [4, 2, 2, 4, 3]
    on episodes [0, 1, 1, 2, 3]: mean 3, residuals [1, -1, -1, 1, 0], cluster
    sums [1, -2, 1, 0], sum of squares 6, times G / (G - 1) = 4 / 3 is 8, so
    se = sqrt(8) / 5 and z = 15 / (2 sqrt 2). pixel_ae - frozen_ssl: [3 - 5,
    3 - 3, 3 - 3, 3 - 5, 3 - 4] = [-2, 0, 0, -2, -1], mean -1, the same
    residual pattern, z = -5 / (2 sqrt 2). pixel_ae - random_vit is a constant
    2, se exactly 0, and `pooling._z` reads +inf there. `clusters` is the
    number of distinct episode labels the windows carry -- four -- not the
    kept-window count, and `cluster_threshold(FAMILY, 4)` is finite."""
    inputs, clusters = script.pooled_inputs(_fab_records(), h=H_FAB)
    assert isinstance(inputs, ReadingOneInputs)
    assert list(inputs.delta_contrast) == list(itertools.combinations(ARMS_ORDER, 2)) == [
        ("pixel_ae", "frozen_ssl"), ("pixel_ae", "random_vit"), ("frozen_ssl", "random_vit"),
    ]
    se = 2 * math.sqrt(2) / 5
    fr = inputs.delta_contrast[("frozen_ssl", "random_vit")]
    assert isinstance(fr, Contrast)
    assert fr.estimate == pytest.approx(3.0) and fr.se == pytest.approx(se)
    assert fr.z == pytest.approx(15 / (2 * math.sqrt(2))) and fr.n_windows == 5
    pf = inputs.delta_contrast[("pixel_ae", "frozen_ssl")]
    assert pf.estimate == pytest.approx(-1.0) and pf.se == pytest.approx(se)
    assert pf.z == pytest.approx(-5 / (2 * math.sqrt(2))) and pf.n_windows == 5
    pr = inputs.delta_contrast[("pixel_ae", "random_vit")]
    assert pr.estimate == pytest.approx(2.0) and pr.se == 0.0 and pr.z == math.inf
    assert clusters == 4
    z_fam = pooling.cluster_threshold(FAMILY, clusters)
    assert math.isfinite(z_fam) and z_fam == pooling.cluster_threshold(8, 4)


def test_pooled_inputs_cosine_and_corrected_contrasts_exclude_nan_windows_and_split_the_folds():
    """cos(3), frozen_ssl - random_vit on the seed-mean cosine: frozen_ssl
    seed 1 has a zero imagined displacement at window 5 (cosine NaN), so that
    window leaves the pairing along with unmoved window 0 -- kept [1, 2, 3, 4]
    on episodes [0, 1, 1, 2] with differences [0.75 - 0.25, 0.75 - 0.75,
    0.75 - 0.75, 0.75 - 0.25] = [0.5, 0, 0, 0.5]: mean 0.25, residuals
    [+-0.25], cluster sums [0.25, -0.5, 0.25], sum of squares 0.375, times
    3 / 2 is 0.5625, se = 0.75 / 4, z = 4 / 3. The held-out contrasts are on
    each FOLD's rows alone: fold A (even labels) keeps windows {1, 4} with
    seed-mean held-out [10, 10] - [12, 16] = [-2, -6] on episodes [0, 2],
    mean -4, cluster sums [2, -2], se = sqrt(2 * 8) / 2 = 2, z = -2; fold B
    keeps {2, 3, 5}: [10, 10, 10] - [14, 14, 16] = [-4, -4, -6] on [1, 1, 3],
    mean -14/3, residuals [2/3, 2/3, -4/3], cluster sums [4/3, -4/3], sum of
    squares 32/9, times 2 is 64/9, se = (8/3) / 3 = 8/9, z = -5.25."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    cos = inputs.cosine_contrast
    assert cos.estimate == pytest.approx(0.25) and cos.se == pytest.approx(0.1875)
    assert cos.z == pytest.approx(4 / 3) and cos.n_windows == 4
    a = inputs.corrected_contrast_a
    assert a.estimate == pytest.approx(-4.0) and a.se == pytest.approx(2.0)
    assert a.z == pytest.approx(-2.0) and a.n_windows == 2
    b = inputs.corrected_contrast_b
    assert b.estimate == pytest.approx(-14 / 3) and b.se == pytest.approx(8 / 9)
    assert b.z == pytest.approx(-5.25) and b.n_windows == 3


def test_pooled_inputs_crossing_contrasts_are_per_draw_with_the_seeds_stacked():
    """Spec 2.2: nothing about a crossing step is ever seed-averaged. The
    probe-channel contrast pairs frozen_ssl's twelve (window, seed) draws
    [4, 4, 2, NaN, 3, 4 | 4, 1, 2, 2, 3, 4] with random_vit's twelve 1s; the
    NaN draw (never moved) leaves alone, so eleven differences [3, 3, 1, 2, 3
    | 3, 0, 1, 1, 2, 3] on episodes [0, 0, 1, 2, 3 | 0, 0, 1, 1, 2, 3]: sum 22,
    mean 2, residuals [1, 1, -1, 0, 1 | 1, -2, -1, -1, 0, 1], cluster sums
    [1, -3, 0, 2], sum of squares 14, times 4 / 3 is 56 / 3, se = sqrt(56 / 3)
    / 11. Seed-averaging first would pair five windows, not eleven draws. The
    free channel is 4 against 4 everywhere: estimate 0, se 0, and `_z` reads
    0 / 0 as 0 over all twelve draws."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    probe = inputs.crossing_contrast_probe
    assert probe.estimate == pytest.approx(2.0)
    assert probe.se == pytest.approx(math.sqrt(56 / 3) / 11)
    assert probe.z == pytest.approx(22 * math.sqrt(3 / 56)) and probe.n_windows == 11
    free = inputs.crossing_contrast_free
    assert free.estimate == 0.0 and free.se == 0.0 and free.z == 0.0 and free.n_windows == 12


def test_pooled_inputs_ratios_are_ratios_of_medians_with_each_cell_in_units_of_its_own_denominator():
    """`pool_ratio`'s estimand, on the moved windows with the seeds stacked.
    frozen_ssl's probe ratio: seed 0 has |d_hat| [1, 1, 3, 3, 3] over
    |d_hat_real| [2, 2, 2, 2, 2] and seed 1 has [2, 2, 2, 6, 6] over [4, 4, 4,
    4, 4]; each cell in units of its own denominator median gives numerators
    [.5, .5, 1.5, 1.5, 1.5 | .5, .5, .5, 1.5, 1.5] with median 1.0 over
    denominators of median 1.0: ratio 1.0. A raw stacked median would read
    2.5 / 3. pixel_ae is 2x its denominator in both seeds (2.0, interval
    exactly [2, 2]); random_vit's numerator is 0 everywhere (0.0, [0, 0]).
    The probe-free ratio, which no probe touches: pixel_ae 0.5, frozen_ssl
    1.5, random_vit 0.25, each constant per window so each interval is the
    point."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    fr = inputs.ratio_probe["frozen_ssl"]
    assert isinstance(fr, Ratio)
    assert fr.estimate == pytest.approx(1.0) and fr.low <= 1.0 <= fr.high
    assert inputs.ratio_probe["pixel_ae"] == Ratio(estimate=2.0, low=2.0, high=2.0)
    assert inputs.ratio_probe["random_vit"] == Ratio(estimate=0.0, low=0.0, high=0.0)
    assert inputs.ratio_free["pixel_ae"] == Ratio(estimate=0.5, low=0.5, high=0.5)
    assert inputs.ratio_free["frozen_ssl"] == Ratio(estimate=1.5, low=1.5, high=1.5)
    assert inputs.ratio_free["random_vit"] == Ratio(estimate=0.25, low=0.25, high=0.25)
    assert list(inputs.ratio_probe) == list(inputs.ratio_free) == list(ARMS_ORDER)


def test_pooled_inputs_carries_alpha_boundaries_at_h_and_the_same_inputs_within_each_seed():
    """A boundary at h in ANY seed of an arm flags the arm (frozen_ssl seed 1
    is on the boundary at step 3 and nowhere else). The per-seed inputs are
    computed from that seed's cells alone and carry no seeds of their own:
    seed 0's frozen_ssl - random_vit margin is [4, 2, 2, 4, 3] - 1, mean 2,
    not the pooled 3; seed 0's frozen_ssl boundary is False."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    assert inputs.alpha_boundary == {"pixel_ae": False, "frozen_ssl": True, "random_vit": False}
    assert sorted(inputs.per_seed) == [0, 1]
    seed0, seed1 = inputs.per_seed[0], inputs.per_seed[1]
    assert isinstance(seed0, ReadingOneInputs) and seed0.per_seed is None and seed1.per_seed is None
    assert seed0.delta_contrast[("frozen_ssl", "random_vit")].estimate == pytest.approx(2.0)
    assert seed0.delta_contrast[("frozen_ssl", "random_vit")].n_windows == 5
    assert seed0.alpha_boundary["frozen_ssl"] is False
    assert seed1.alpha_boundary["frozen_ssl"] is True
    assert seed0.crossing_contrast_probe.n_windows == 5, "one seed: five finite draws"


def test_an_unmeasurable_cell_leaves_the_probe_based_pooling_and_only_that():
    """Spec 3.1: a cell enters the probe-based pooling iff its persistence-to-
    floor band is positive. random_vit seed 1 is marked unmeasurable and given
    a margin of 3 everywhere: excluded, random_vit's seed mean stays 1 and the
    contrast reads 3; included, it would be 2 and read 2. The per-draw probe
    crossing contrast pairs the seeds measurable in BOTH arms (seed 0 only:
    five finite draws), while the probe-free crossing contrast and ratio keep
    all twelve draws. With every cell unmeasurable there is nothing to pool:
    every probe-based field is NaN with zero windows, no boundary is flagged,
    and the probe-free fields are untouched."""
    records = _fab_records()
    records[("random_vit", 1)] = _fab_record(
        "random_vit", 1, margin=[3] * 6, cosine=[0.25] * 6, held=[0, 12, 14, 14, 16, 16],
        boundary=[False] * 3, crossing_probe=[1] * 6, crossing_free=[4] * 6,
        probe_hat=[0] * 6, probe_real=[5] * 6, free_hat=[1] * 6, free_true=[4] * 6,
        measurable=False,
    )
    inputs, _ = script.pooled_inputs(records, h=H_FAB)
    assert inputs.delta_contrast[("frozen_ssl", "random_vit")].estimate == pytest.approx(3.0)
    assert inputs.crossing_contrast_probe.n_windows == 5
    assert inputs.crossing_contrast_free.n_windows == 12
    assert inputs.ratio_free["random_vit"].estimate == pytest.approx(0.25)
    # The per-arm block pools the same measurable cells: random_vit's
    # pooled delta is seed 0's 1, not (1 + 3) / 2, and names the one seed.
    block = script.arm_summaries(records, inputs, h=H_FAB)
    assert block["random_vit"].delta.mean == pytest.approx(1.0)
    assert block["random_vit"].delta.seeds == (0,)

    for record in records.values():
        record["probe"]["measurable"] = False
    none, _ = script.pooled_inputs(records, h=H_FAB)
    nothing = script.arm_summaries(records, none, h=H_FAB)
    assert all(nothing[arm].delta is None and nothing[arm].cosine is None for arm in ARMS_ORDER)
    assert nothing["frozen_ssl"].ratio_free.estimate == pytest.approx(1.5)
    for pair in itertools.combinations(ARMS_ORDER, 2):
        assert math.isnan(none.delta_contrast[pair].z) and none.delta_contrast[pair].n_windows == 0
    assert math.isnan(none.cosine_contrast.z) and math.isnan(none.crossing_contrast_probe.z)
    assert math.isnan(none.corrected_contrast_a.z) and math.isnan(none.corrected_contrast_b.z)
    assert all(math.isnan(none.ratio_probe[arm].estimate) for arm in ARMS_ORDER)
    assert none.alpha_boundary == {arm: False for arm in ARMS_ORDER}
    assert none.crossing_contrast_free.n_windows == 12
    assert none.ratio_free["frozen_ssl"].estimate == pytest.approx(1.5)


def test_pooled_inputs_refuses_a_missing_cell_and_a_step_past_the_horizon():
    """Reading 1 is three arms at every seed; a pool over five cells printed
    under six cells' names is the table this repo has been burned by. And
    column h - 1 of a record with a shorter horizon is an IndexError at best
    and the wrong step at worst."""
    records = _fab_records()
    del records[("pixel_ae", 1)]
    with pytest.raises(KeyError, match="pixel_ae seed 1"):
        script.pooled_inputs(records, h=H_FAB)
    with pytest.raises(ValueError, match="no step h=4"):
        script.pooled_inputs(_fab_records(), h=4)


def test_the_r2_filter_marks_low_r2_cells_unmeasurable_without_touching_the_originals():
    """The sensitivity line of spec 3.1 recomputes every probe-based
    statistic with cells of selection R^2 < 0.1 excluded, and changes no
    verdict -- so the filter must produce NEW records and leave the ones the
    verdict was read from exactly as they were."""
    records = _fab_records()
    records[("frozen_ssl", 1)]["probe"]["selection_r2"] = 0.05
    filtered = script._r2_filtered(records, script.R2_SENSITIVITY)
    assert script.R2_SENSITIVITY == 0.1
    assert sorted(filtered) == sorted(records)
    assert filtered[("frozen_ssl", 1)]["probe"]["measurable"] is False
    assert filtered[("frozen_ssl", 0)]["probe"]["measurable"] is True
    assert records[("frozen_ssl", 1)]["probe"]["measurable"] is True
    assert filtered[("frozen_ssl", 1)]["margin"] is records[("frozen_ssl", 1)]["margin"], (
        "the series are shared, not copied: only the probe block is rewritten"
    )


# --- the per-arm block, pixel_ae's pair, the self-check table -----------------


def test_the_per_arm_block_pools_each_arms_own_margin_and_cosine_and_reads_the_counts():
    """Spec 5 prints "per-arm block, then contrasts, then the verdict lines":
    the block is `pool_arm` on ONE arm's measurable cells, the seed-mean
    per window over the moved-and-finite windows, clustered by episode.
    pixel_ae's margin is 3 in both seeds on every moved window: mean 3, se
    exactly 0, 5 windows. frozen_ssl's seed-mean margin over windows 1-5 is
    [5, 3, 3, 5, 4]: mean 4, residuals [1, -1, -1, 1, 0], cluster sums
    [1, -2, 1, 0], se = sqrt(8) / 5 -- the same ruler as its contrast with
    random_vit, whose margin is 1 everywhere. random_vit's cosine over the
    same windows is [0.25, 0.75, 0.75, 0.25, 0.25]: mean 0.45, residuals
    [-0.2, 0.3, 0.3, -0.2, -0.2], cluster sums [-0.2, 0.6, -0.2, -0.2], sum
    of squares 0.48, times 4 / 3 is 0.64, se = 0.8 / 5 = 0.16. frozen_ssl's
    cosine loses seed 1's NaN window 5 to the finiteness mask: four windows
    at (1.0 + 0.5) / 2. The ratios are `pooled_inputs`' own objects; R_raw is
    the median of the moved draws' per-window |d_hat| / |d| (1.0 everywhere
    in the fabricated records, NaN on the unmoved window and so excluded);
    the counts are summed over the arm's two cells."""
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    block = script.arm_summaries(_fab_records(), inputs, h=H_FAB)
    assert list(block) == list(ARMS_ORDER)
    assert all(isinstance(v, script.ArmSummary) for v in block.values())
    pa = block["pixel_ae"]
    assert isinstance(pa.delta, pooling.PooledMean)
    assert pa.delta.mean == pytest.approx(3.0) and pa.delta.se == 0.0 and pa.delta.windows == 5
    assert pa.cosine.mean == pytest.approx(0.5) and pa.cosine.se == 0.0
    fs = block["frozen_ssl"]
    assert fs.delta.mean == pytest.approx(4.0) and fs.delta.se == pytest.approx(2 * math.sqrt(2) / 5)
    assert fs.delta.windows == 5 and fs.delta.seeds == (0, 1)
    assert fs.cosine.mean == pytest.approx(0.75) and fs.cosine.windows == 4
    rv = block["random_vit"]
    assert rv.delta.mean == pytest.approx(1.0) and rv.delta.se == 0.0
    assert rv.cosine.mean == pytest.approx(0.45) and rv.cosine.se == pytest.approx(0.16)
    assert rv.cosine.windows == 5
    for arm in ARMS_ORDER:
        assert block[arm].ratio_probe == inputs.ratio_probe[arm]
        assert block[arm].ratio_free == inputs.ratio_free[arm]
        assert block[arm].ratio_raw == 1.0
        assert (block[arm].moved, block[arm].zero_displacement) == (10, 0)
    table = script._arm_table(block, H_FAB).splitlines()
    assert table[0].split()[0] == "arm" and len(table) == 4
    assert [line.split()[0] for line in table[1:]] == list(ARMS_ORDER)
    assert "n/a" not in "\n".join(table)


def test_pixel_aes_crossing_pair_is_computed_for_information_with_the_seeds_stacked():
    """Spec 3.2: "`pixel_ae`'s pair is printed for information." The same
    per-draw estimator as the probe control's contrast, `arm - random_vit`:
    pixel_ae's probe crossings are 3 in every draw against random_vit's 1 --
    twelve differences of exactly 2, se 0, and `_z` reads +inf -- and the
    free channel is 4 against 4, 0 over twelve draws. frozen_ssl's pair
    through the same function is the contrast `pooled_inputs` decides on, so
    the two must agree. Nothing reads it: `ReadingOneInputs` has no slot."""
    probe = script._informational_crossing(_fab_records(), "pixel_ae", "probe")
    assert isinstance(probe, Contrast)
    assert probe.estimate == pytest.approx(2.0) and probe.se == 0.0
    assert probe.z == math.inf and probe.n_windows == 12
    free = script._informational_crossing(_fab_records(), "pixel_ae", "free")
    assert free.estimate == 0.0 and free.n_windows == 12
    inputs, _ = script.pooled_inputs(_fab_records(), h=H_FAB)
    assert script._informational_crossing(_fab_records(), TREATMENT, "probe") == inputs.crossing_contrast_probe


def test_the_self_check_table_has_one_row_per_cell_in_arm_order_with_the_deltas_and_the_probe():
    """One row per record, in ARMS_ORDER then seed -- not alphabetical, which
    would put frozen_ssl first -- carrying both deltas, the window count,
    the episode-label match, the probe's R^2 and measurability, the
    never-moved count and `ok`. What Task 7 fills its self-check table from."""
    records = _fab_records()
    records[("random_vit", 1)]["probe"]["measurable"] = False
    records[("random_vit", 1)]["self_check"]["reference_position_max_delta"] = 2.0 ** -40
    lines = script._self_check_table(records).splitlines()
    assert lines[0].split()[0] == "cell" and len(lines) == 7
    assert [line.split()[0] for line in lines[1:]] == [
        f"{arm}/s{seed}" for arm in ARMS_ORDER for seed in (0, 1)
    ]
    assert lines[1].split()[1:3] == ["0.0e+00", "0.0e+00"]
    last = lines[-1].split()
    assert last[0] == "random_vit/s1" and last[1] == "9.1e-13" and "False" in last
    assert "True" in lines[1].split() and "0.300" in lines[1]


# --- survival_by_arm, write_readings ------------------------------------------


def test_survival_by_arm_stacks_the_seeds_per_channel():
    """`survival` over the (window, seed) draws that moved. frozen_ssl's probe
    crossings over both seeds are [4, 4, 2, NaN, 3, 4, 4, 1, 2, 2, 3, 4] --
    eleven finite draws, one at 1, three at 2, two at 3, five at 4 (= horizon
    + 1, never crossed): S(0) = 1, S(1) = 10/11, S(2) = 7/11, S(3) = 5/11.
    random_vit's probe crossings are all 1: S = [1, 0, 0, 0]. Every arm's
    free channel is 4 everywhere: S = [1, 1, 1, 1]."""
    curves = script.survival_by_arm(_fab_records())
    assert sorted(curves) == sorted(itertools.product(ARMS_ORDER, ("probe", "free")))
    np.testing.assert_allclose(curves[("frozen_ssl", "probe")], [1.0, 10 / 11, 7 / 11, 5 / 11])
    np.testing.assert_allclose(curves[("random_vit", "probe")], [1.0, 0.0, 0.0, 0.0])
    for arm in ARMS_ORDER:
        np.testing.assert_allclose(curves[(arm, "free")], [1.0, 1.0, 1.0, 1.0])


def test_write_readings_writes_trust_txt_under_out_and_returns_its_path(tmp_path):
    path = script.write_readings(tmp_path, "--- Reading 1 ---\nline\n")
    assert path == tmp_path / "trust.txt"
    assert path.read_text() == "--- Reading 1 ---\nline\n"


# --- main on the fixture: the NaN path ------------------------------------------


@pytest.fixture
def trust_run(tmp_path, small_buffer, capsys):
    """Three arms at seed 0 on the shared fixture (~3 s per cell: `run_job` for
    five steps, then `diagnose_dynamics.main` to write the diagnostic the
    self-check reads), then `trust_horizon.main` over all three. The fixture's
    split holds ONE validation episode, so every clustered standard error and
    `cluster_threshold(FAMILY, 1)` are NaN."""
    for arm in ARMS_ORDER:
        run_job(StudyJob(arm, 0), small_buffer, tmp_path,
                steps=5, seq_len=4, context=2, horizon=3, device="cpu")
        assert diagnose.main([
            "--out", str(tmp_path), "--data", str(small_buffer.root), "--arms", arm,
            "--seeds", "0", "--context", "2", "--horizon", "3", "--ks", "1", "3",
            "--device", "cpu",
        ]) == diagnose.EXIT_OK
    capsys.readouterr()
    status = script.main([
        "--out", str(tmp_path), "--data", str(small_buffer.root), "--arms", *ARMS_ORDER,
        "--seeds", "0", "--device", "cpu",
    ])
    return types.SimpleNamespace(status=status, out=capsys.readouterr().out, out_dir=tmp_path)


def test_the_fixture_run_writes_trust_txt_with_both_readings(trust_run):
    """Exit 0, and the text on disk is the text on screen, with this task's
    own section header for each reading -- so a reader of `trust.txt` alone
    can find both."""
    assert trust_run.status == script.EXIT_OK
    text = (trust_run.out_dir / "trust.txt").read_text()
    # The section order of `_readings_text`: the self-check table, the
    # pooling notes, the per-arm block, Reading 1 (Task 5's header, from
    # `format_reading_one` at THIS run's horizon), pixel_ae's h_x pair for
    # information, the sensitivity block, Reading 2 (Task 5's header).
    assert "--- self-check per cell" in text
    assert "--- per arm at h = 3" in text
    assert "--- Reading 1: does the h=3 gate reward slow drift?" in text
    assert "for information: h_x probe pixel_ae - random_vit" in text
    assert "--- Reading 2: the horizon M4 designs around" in text
    assert text.count("--- Reading 1:") == 1 and text.count("--- Reading 2:") == 1
    assert text.rstrip("\n") in trust_run.out
    order = [text.index(s) for s in (
        "--- self-check per cell", "clusters: 1 validation", "--- per arm at h = 3",
        "--- Reading 1:", "for information: h_x probe", "--- sensitivity", "--- Reading 2:",
    )]
    assert order == sorted(order)
    for arm in ARMS_ORDER:
        assert f"{arm}/s0" in text[:text.index("clusters:")], "the self-check table names every cell"


def test_the_fixture_run_says_why_every_contrast_is_n_a(trust_run):
    """One validation episode: the episode-clustered standard error needs at
    least two clusters, so every pooled z is NaN, every contrast is n/a, and
    Reading 1 cannot find a best-delta arm -- NOT_TESTABLE, by `reading_one`'s
    own order. The reason is printed by THIS script, before the formatted
    reading, so it does not depend on how the formatter renders a NaN."""
    text = (trust_run.out_dir / "trust.txt").read_text()
    assert "clusters: 1 validation episode(s)" in text
    assert "at least two" in text and "n/a" in text
    assert "reading 1: NOT_TESTABLE" in text
    assert "sensitivity" in text and "changes no verdict" in text


def test_a_single_arm_run_still_exits_ok_and_says_both_readings_are_not_computed(cell, capsys):
    """Task 4's fixture is ONE arm at one seed. Reading 1 pools all three
    arms and Reading 2 needs every arm's survival curve (`reading_two`
    refuses a missing `(arm, channel)` by name), so neither is computed --
    and the run says so under BOTH headers, still exits 0, still writes
    trust.txt and the self-check table, rather than ending in
    `reading_two`'s KeyError after nine clean cells."""
    assert script.main(_argv(cell)) == script.EXIT_OK
    text = (cell.out / "trust.txt").read_text()
    assert text.count("not computed:") == 2
    assert "Reading 1 pools all of" in text
    assert "Reading 2 needs every arm's survival curve" in text
    assert "['random_vit']" in text
    assert "--- self-check per cell" in text and "random_vit/s1" in text
    assert "--- Reading 1:" in text and "--- Reading 2:" in text
    assert text.rstrip("\n") in capsys.readouterr().out
