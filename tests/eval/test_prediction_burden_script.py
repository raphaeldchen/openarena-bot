"""M3m's measure phase: the ladder, the model-free baseline, one record per cell.

EVERY NUMBER IN THESE TESTS COMES FROM THE REAL CALL PATH. `measure_cell` runs
the real `regrounding_sweep` against the oracle-family models `test_rollout.py`
and `test_diagnostics.py` already use, so a record's margin can only have been
produced by the code that produces it. The fakes are confined to what needs nine
20,000-step checkpoints -- `load_cell` and `prepare_cell` -- and each of those is
made to BEHAVE (refuse a directory that lacks the checkpoint, hand back a model
whose records differ) rather than to record what it was called with. A previous
milestone's test proved a `frame_probe` was HANDED the right seed while its body
ignored it.

`measure_cell` DOES NOT RUN `evaluate_rollout`: the floor and the canonical
curves come off the `reference` it is handed, which is `Prepared.reference`, the
rollout `prepare_cell` verified against the study record. A test that calls
`measure_cell` directly builds that reference with the real `evaluate_rollout`
(`_reference`); the phase tests hand each cell a reference whose curves are
RECOGNISABLE, so the record can be asked which pass it carries.

THE RIG IS CHOSEN SO THAT NO ASSERTION IS DECIDED BY THE FIXTURE.

  * The agent WOBBLES about a straight line rather than moving along one. On a
    straight line the true displacement and the k=1 rung are both one constant
    step, the margin is exactly zero in every window, and `one - rows` is the
    same array as `rows - one`.
  * The probe is a linear fit over every episode, so the floor is a few map
    units and not zero, and the wobble gives each window its own displacement.
  * Nine episodes whose window counts are 1, 2, 0, 1, 3, 2, 1, 2, 1. The zero is
    length 50 == `context + horizon`, the one length where `< need` and
    `< need + 1` disagree. Two lengths are exact multiples of the stride.
  * Their FILE NAMES descend, so the order the paths are given in is NOT the
    order `sorted()` would put them in. Given lexicographically sorted names, a
    `sorted(val_paths)` traversal is the identity and no test could see it.
  * Nine episodes, not two: with two, a resample has three possible outcomes and
    a 2.5/97.5 percentile interval is just the smaller and larger episode mean.
"""

import importlib.util
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.data.episode import load_episode, save_episode
from mbfps.eval import burden
from mbfps.eval.diagnostics import (
    REGROUNDING_KS, action_intervention_ladder, regrounding_sweep,
)
from mbfps.eval.probe import fit_probe, probe_targets
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record
from mbfps.eval.windows import window_starts
from tests.eval.test_rollout import (
    KEYS, DriftingModel, OracleModel, synthetic_episode,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prediction_burden.py"


def _load():
    spec = importlib.util.spec_from_file_location("prediction_burden_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()

CONTEXT, HORIZON = 5, 45
NEED = CONTEXT + HORIZON
CPU = torch.device("cpu")
SEED = 3

LENGTHS = (70, 120, 50, 95, 170, 100, 60, 130, 75)
EXPECTED_WINDOWS = (1, 2, 0, 1, 3, 2, 1, 2, 1)


def _wobbling_episode(length: int, index: int):
    """`synthetic_episode`'s straight run, plus a wobble phased by `index`."""
    episode = synthetic_episode(length)
    t = np.arange(length + 1, dtype=np.float64)
    phase = 0.9 * index
    episode.privileged[:, 1] = (10.0 * t + 4.0 * np.sin(0.9 * t + phase)).astype(np.float32)
    episode.privileged[:, 2] = (-3.0 * t + 3.0 * np.cos(0.55 * t - phase)).astype(np.float32)
    return episode


def _episode_file(directory: Path, episode, name_index: int) -> Path:
    path = directory / f"ep_{name_index:06d}_len{episode.length:05d}.npz"
    save_episode(episode, path)
    return path


def _linear_probe(episodes) -> dict:
    """One linear map from the frame tag to the privileged state, over every
    episode -- not exact on a wobbling agent, which is the point."""
    tags = np.concatenate(
        [np.arange(e.privileged.shape[0], dtype=np.float64)[:, None] for e in episodes]
    )
    targets = np.concatenate([probe_targets(e.privileged, KEYS) for e in episodes])
    return fit_probe(tags, targets, ridge=1e-8)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    directory = tmp_path_factory.mktemp("burden_rig")
    episodes = [_wobbling_episode(n, i) for i, n in enumerate(LENGTHS)]
    # Descending names: the GIVEN order is the reverse of the sorted one.
    paths = [
        _episode_file(directory, e, len(LENGTHS) - 1 - i) for i, e in enumerate(episodes)
    ]
    assert paths != sorted(paths), "the fixture must distinguish sorted() from the given order"
    assert tuple(len(window_starts(n, CONTEXT, HORIZON)) for n in LENGTHS) == EXPECTED_WINDOWS
    return types.SimpleNamespace(
        paths=paths, episodes=episodes, probe=_linear_probe(episodes),
    )


STUDY_RECORD = {
    "steps": 20000, "kl_rate_above_free_bits": 0.9126, "kl_dyn_max": 2.951450824737549,
    "git_sha": "ca3e140772d6bc741d4d04312763afe3dd754166",
}


def _reference(rig, model, *, context=CONTEXT, horizon=HORIZON):
    """The rollout `prepare_cell` would hand back for `model`: the real
    `evaluate_rollout`, at the cell's seed and through the cell's probe."""
    return evaluate_rollout(
        model, rig.paths, rig.probe, context=context, horizon=horizon, seed=SEED,
        device=CPU, feature_backbone=None,
    )


def _cell_kwargs(rig, **over) -> dict:
    kwargs = dict(
        model=DriftingModel(), val_paths=rig.paths, probe=rig.probe, arm="pixel_ae",
        seed=SEED, context=CONTEXT, horizon=HORIZON, ks=REGROUNDING_KS, device=CPU,
        feature_backbone=None, study_record=dict(STUDY_RECORD),
    )
    kwargs.update(over)
    if "reference" not in over:
        kwargs["reference"] = _reference(
            rig, kwargs["model"], context=kwargs["context"], horizon=kwargs["horizon"],
        )
    return kwargs


@pytest.fixture(scope="module")
def record(rig):
    return script.measure_cell(**_cell_kwargs(rig))


def _independent_sweep(rig, model=None):
    """The real sweep, called again by the TEST, so what the record's numbers are
    compared with was not produced by the code under test."""
    return regrounding_sweep(
        model or DriftingModel(), rig.paths, rig.probe, ks=REGROUNDING_KS,
        context=CONTEXT, horizon=HORIZON, seed=SEED, device=CPU, feature_backbone=None,
    )


def _window_targets(paths) -> list[np.ndarray]:
    """Each window's `horizon + 1` privileged rows, in the traversal the sweep
    walks: the paths as given, `window_starts` within each."""
    out = []
    for path in paths:
        episode = load_episode(path)
        for start in window_starts(episode.length, CONTEXT, HORIZON):
            out.append(probe_targets(
                episode.privileged[start + CONTEXT : start + NEED + 1],
                episode.privileged_keys,
            ))
    return out


def _canonical(labels) -> np.ndarray:
    """The partition a label array induces, independent of the numbers used."""
    return np.unique(np.asarray(labels), return_inverse=True)[1]


# ---------------------------------------------------------------------------
# The exits.
# ---------------------------------------------------------------------------


def test_the_exit_codes_are_this_milestones_own():
    """39/40 are M3j's, 41/42 M3k's, 43/44 M3l's. A collision would make two
    tools report different failures under one number."""
    assert script.EXIT_CONTROL_BROKEN == 45
    assert script.EXIT_UNREADABLE == 46


# ---------------------------------------------------------------------------
# The model-free baseline, and the one thing that must stay aligned.
# ---------------------------------------------------------------------------


def test_baseline_rows_walks_the_given_order_and_the_shared_window_rule(tmp_path):
    """THE MUTATION THIS EXISTS FOR: `enumerate(val_paths)` ->
    `enumerate(sorted(val_paths))` in `baseline_rows`. The sweep retains its rows
    in the order IT walked, and nothing ties the two loops but this. The two
    episodes yield DIFFERENT window counts and their names are given in
    DESCENDING order, so `sorted()` swaps the blocks: the labels read [0, 0, 1]
    where the given order reads [0, 1, 1].
    """
    paths = [
        _episode_file(tmp_path, _wobbling_episode(70, 0), name_index=1),
        _episode_file(tmp_path, _wobbling_episode(120, 1), name_index=0),
    ]
    assert paths != sorted(paths), "the names must make sorted() a different order"
    expected = [len(window_starts(70, CONTEXT, HORIZON)), len(window_starts(120, CONTEXT, HORIZON))]
    assert expected[0] != expected[1], "fixture must distinguish the orders"

    rows, labels = script.baseline_rows(paths, context=CONTEXT, horizon=HORIZON)
    assert rows.shape == (sum(expected), HORIZON)
    # Labels are per window, in traversal order, so the FIRST block is the first
    # path's. A reordered traversal changes the block sizes.
    assert list(labels[: expected[0]]) == [0] * expected[0]
    assert list(labels[expected[0] :]) == [1] * expected[1]

    reversed_rows, _ = script.baseline_rows(paths[::-1], context=CONTEXT, horizon=HORIZON)
    assert not np.array_equal(rows, reversed_rows)


@pytest.mark.parametrize(
    "episode_index", [i for i, n in enumerate(EXPECTED_WINDOWS) if n],
)
def test_baseline_rows_are_the_true_one_step_displacement_in_every_window(rig, episode_index):
    """Model-free: the baseline is ground truth, so it must reproduce the
    displacement computed straight from the episode's own coordinates -- by a
    route that does not go through `probe_targets` or `burden`.

    THE MUTATION THIS EXISTS FOR: the slice `start + context : start + need + 1`
    moved one frame to `start + context + 1 : start + need + 2`. Every shape is
    unchanged and each row is still a plausible distance; the wobble makes
    consecutive displacements differ, so the shifted rows differ too. Every
    window of the episode is checked, so a `start` that does not advance by the
    stride is seen as well.

    ONE EPISODE PER CALL, deliberately. The length-100 episode is an exact
    multiple of the stride, so its last window ends on the final frame and a
    shifted slice runs off the end: stacked with the others that is a ragged
    array and a crash, which would fail this test for a reason that is not the
    one it names. Asked alone, every other episode fails it by VALUE.
    """
    episode = load_episode(rig.paths[episode_index])
    x = episode.privileged[:, 1].astype(np.float64)
    y = episode.privileged[:, 2].astype(np.float64)
    expected = []
    for start in window_starts(episode.length, CONTEXT, HORIZON):
        frames = slice(start + CONTEXT, start + NEED + 1)
        expected.append(np.hypot(np.diff(x[frames]), np.diff(y[frames])))
    expected = np.stack(expected)
    assert expected.shape == (EXPECTED_WINDOWS[episode_index], HORIZON)

    rows, _ = script.baseline_rows([rig.paths[episode_index]], context=CONTEXT, horizon=HORIZON)
    # A one-frame shift turns row j into what was row j + 1, so the fixture
    # reaches the mutation only if consecutive displacements differ.
    assert np.abs(np.diff(expected, axis=1)).max() > 0.5
    np.testing.assert_allclose(rows, expected, rtol=1e-6)


def test_baseline_rows_label_every_window_and_agree_with_the_sweeps_own_traversal(rig):
    """The tie between the two loops that CAN be measured. `regrounding_sweep`
    retains no episode labels, but `action_intervention_ladder` runs the same
    `_diagnose` traversal and does, so the baseline's labels are compared with
    the diagnostics' own -- on a rig with a zero-window episode in the middle
    (length 50, the `< need` boundary) and two exact stride multiples, the
    places a private copy of the window rule would disagree with the shared one.

    Compared as PARTITIONS: the diagnostics number episodes after skipping a
    too-short one, `baseline_rows` indexes `val_paths`, and clustering reads
    only which windows share a label.
    """
    rows, labels = script.baseline_rows(rig.paths, context=CONTEXT, horizon=HORIZON)
    ladder = action_intervention_ladder(
        DriftingModel(), rig.paths, rig.probe, arms=("shuffled",), context=CONTEXT,
        horizon=HORIZON, seed=SEED, device=CPU, feature_backbone=None,
    )
    assert labels.shape == (rows.shape[0],)
    assert np.array_equal(_canonical(labels), _canonical(ladder.window_episode))
    assert rows.shape == _independent_sweep(rig).window_position[1].shape
    # And the label of a window is the index of the path it came from, so the
    # zero-window episode (index 2) owns none.
    assert 2 not in set(labels.tolist())
    assert np.bincount(labels, minlength=len(LENGTHS)).tolist() == list(EXPECTED_WINDOWS)


def test_baseline_rows_refuse_when_no_validation_window_exists(tmp_path):
    path = _episode_file(tmp_path, _wobbling_episode(NEED, 0), name_index=0)
    with pytest.raises(SystemExit, match="no validation window"):
        script.baseline_rows([path], context=CONTEXT, horizon=HORIZON)


def test_measure_cell_refuses_a_baseline_that_does_not_align_with_the_sweep(
    rig, monkeypatch,
):
    """THE MUTATION THIS EXISTS FOR: deleting the `rows.shape != one.shape`
    refusal in `measure_cell`, after which a baseline cut on a different window
    rule subtracts row-for-row against the wrong windows and produces a
    plausible, wrong margin. With the check gone the subtraction raises a numpy
    broadcasting error instead of this refusal, so the test sees the difference.

    Both directions: one window too few and one too many.
    """
    real = script.baseline_rows

    def too_few(paths, *, context, horizon):
        rows, labels = real(paths, context=context, horizon=horizon)
        return rows[:-1], labels[:-1]

    def too_many(paths, *, context, horizon):
        rows, labels = real(paths, context=context, horizon=horizon)
        return np.vstack([rows, rows[:1]]), np.append(labels, labels[-1])

    for broken in (too_few, too_many):
        monkeypatch.setattr(script, "baseline_rows", broken)
        with pytest.raises(SystemExit) as caught:
            script.measure_cell(**_cell_kwargs(rig))
        assert "align" in str(caught.value), broken.__name__


# ---------------------------------------------------------------------------
# The stacked margin is `motion_margin`, and refuses what `motion_margin` would.
# ---------------------------------------------------------------------------


def test_the_stacked_margin_equals_motion_margin_window_by_window(rig, record):
    """The script subtracts stacked arrays for speed; `motion_margin` is the
    definition, and the k=1 rung it is given here comes from a SEPARATE call of
    the real sweep. If the two disagree the stacked path is wrong.

    THE MUTATION THIS EXISTS FOR: `one - rows` instead of `rows - one` in
    `measure_cell`, which flips the sign of the quantity Reading H's verdict is
    read from and leaves every shape and every interval intact. The wobble gives
    the margin both signs and a mean absolute value near 4 map units, so the
    flipped array is far from the right one in most windows.
    """
    one = _independent_sweep(rig).window_position[1]
    targets = _window_targets(rig.paths)
    margin = np.asarray(record["window_margin"], dtype=np.float64)
    assert margin.shape == one.shape == (len(targets), HORIZON)
    assert (margin > 0).any() and (margin < 0).any(), "the fixture must give both signs"
    for w, window in enumerate(targets):
        expected = burden.motion_margin(window, one[w])
        np.testing.assert_allclose(margin[w], expected, rtol=0, atol=1e-12, err_msg=f"window {w}")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_measure_cell_refuses_a_non_finite_baseline_by_name(rig, monkeypatch, bad):
    """`motion_margin` has a finiteness guard; the stacked subtraction does not,
    so the script carries its own. Without it the poisoned row reaches
    `margin_interval`, which raises a bare ValueError that names neither array.

    THE MUTATION THIS EXISTS FOR: deleting the `np.isfinite(rows)` refusal.
    """
    real = script.baseline_rows

    def poisoned(paths, *, context, horizon):
        rows, labels = real(paths, context=context, horizon=horizon)
        rows = rows.copy()
        rows[3, 10] = bad
        return rows, labels

    monkeypatch.setattr(script, "baseline_rows", poisoned)
    with pytest.raises(SystemExit) as caught:
        script.measure_cell(**_cell_kwargs(rig))
    assert "baseline" in str(caught.value)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_measure_cell_refuses_a_non_finite_k_one_rung_by_name(rig, monkeypatch, bad):
    """The same refusal on the other operand. The sweep's MEAN curves stay
    finite here -- only the retained per-window rows are poisoned -- which is the
    case the stacked subtraction would carry silently into every interval.

    THE MUTATION THIS EXISTS FOR: deleting the `np.isfinite(one)` refusal.
    """
    real = script.regrounding_sweep

    def poisoned(*args, **kwargs):
        sweep = real(*args, **kwargs)
        rows = sweep.window_position[1].copy()
        rows[2, 7] = bad
        return replace(sweep, window_position={**sweep.window_position, 1: rows})

    monkeypatch.setattr(script, "regrounding_sweep", poisoned)
    with pytest.raises(SystemExit) as caught:
        script.measure_cell(**_cell_kwargs(rig))
    assert "k=1" in str(caught.value)


def test_the_baseline_is_taken_after_the_sweep_and_never_before_it(rig, monkeypatch):
    """A rule the brief states, pinned because it is cheap to break by moving a
    line. It cannot change a number TODAY -- the baseline touches no generator
    and both passes reseed -- so this pins the rule, not an effect.

    THE MUTATION THIS EXISTS FOR: moving the `baseline_rows(...)` call above the
    `regrounding_sweep(...)` call in `measure_cell`.
    """
    calls = []

    def recording(name):
        real = getattr(script, name)

        def wrapper(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)

        return wrapper

    kwargs = _cell_kwargs(rig)
    for name in ("regrounding_sweep", "baseline_rows"):
        monkeypatch.setattr(script, name, recording(name))
    script.measure_cell(**kwargs)
    assert calls == ["regrounding_sweep", "baseline_rows"]


@pytest.mark.parametrize(
    "over, match",
    [
        ({"horizon": 30, "ks": (1, 30)}, "DECISION_H"),
        ({"ks": (5, 45)}, "k=1"),
    ],
)
def test_measure_cell_refuses_a_protocol_the_reading_cannot_be_taken_from(rig, over, match):
    """The status is read at `DECISION_H` and every margin is read off the k=1
    rung. Refused by name before any pass is paid for, rather than as an
    IndexError or a KeyError after the sweep."""
    with pytest.raises(SystemExit, match=match):
        script.measure_cell(**_cell_kwargs(rig, **over))


# ---------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------


def test_the_record_carries_every_protocol_parameter_the_reading_uses(record):
    """M3l shipped a headline interval whose DRAW COUNT was not on the record,
    so the permanent artefact could not be audited for it. Both the level and
    the count are recorded here from the start.

    THE MUTATION THIS EXISTS FOR: dropping any of these keys, or recording a
    literal 0.95 / 2000 instead of the module's constants.
    """
    assert record["confidence"] == burden.CONFIDENCE
    assert record["resamples"] == burden.RESAMPLES
    assert record["decision_h"] == burden.DECISION_H
    assert tuple(record["reported_h"]) == burden.REPORTED_H
    assert tuple(record["ks"]) == tuple(REGROUNDING_KS)
    assert record["identity_tolerance"] == burden.IDENTITY_TOLERANCE
    assert record["split_seed"] == SPLIT_SEED
    assert record["device"] == "cpu"
    assert record["git_sha"] == git_sha()
    for key in (
        "arm", "seed", "margin", "controls", "base_control", "burden_by_k",
        "compounding_by_k", "curves", "windows", "git_sha", "step",
        "kl_rate_above_free_bits", "kl_dyn_max", "window_margin", "episodes",
        "context", "horizon", "torch_version",
    ):
        assert key in record, key
    assert (record["arm"], record["seed"]) == ("pixel_ae", SEED)
    assert (record["context"], record["horizon"]) == (CONTEXT, HORIZON)


def test_the_record_carries_the_training_numbers_of_its_own_cell(rig):
    """Carried from the study record beside the checkpoint, strictly: a cell
    whose record lacks them is not a study record, and a default here would
    print a plausible number for a run nobody measured."""
    study = {
        "steps": 12345, "kl_rate_above_free_bits": 0.25, "kl_dyn_max": 7.5,
        "git_sha": "study-sha",
    }
    record = script.measure_cell(**_cell_kwargs(rig, study_record=study))
    assert record["step"] == 12345
    assert record["kl_rate_above_free_bits"] == 0.25
    assert record["kl_dyn_max"] == 7.5
    assert record["record_git_sha"] == "study-sha"

    with pytest.raises(KeyError):
        script.measure_cell(**_cell_kwargs(rig, study_record={"steps": 1}))


def test_the_record_names_its_episodes_and_clusters_its_windows(rig, record):
    assert record["episodes"]["val"] == [p.name for p in rig.paths]
    assert record["windows"]["total"] == sum(EXPECTED_WINDOWS)
    assert len(record["windows"]["episode"]) == record["windows"]["total"]
    assert np.bincount(
        record["windows"]["episode"], minlength=len(LENGTHS)
    ).tolist() == list(EXPECTED_WINDOWS)


def test_the_record_keeps_a_margin_interval_at_every_reported_horizon(record):
    assert sorted(int(h) for h in record["margin"]) == sorted(burden.REPORTED_H)
    window_margin = np.asarray(record["window_margin"])
    for h, entry in record["margin"].items():
        assert entry["ci_low"] <= entry["point"] <= entry["ci_high"], h
        # Never a zero-width interval: that would make the line above pass for
        # any point at all.
        assert entry["ci_low"] < entry["ci_high"], h
        assert entry["point"] == pytest.approx(window_margin[:, int(h) - 1].mean(), abs=1e-12)


def test_the_interval_is_taken_at_the_cells_seed_and_the_draw_count_the_record_states(
    rig, monkeypatch,
):
    """What the record SAYS about its interval is what the interval USED, checked
    by recomputing every interval from the record's own margins.

    A small `RESAMPLES` is patched in, so the recomputation can tell the stated
    count from `burden.RESAMPLES`'s 2000: a record that states one count and
    draws another is exactly what the permanent artefact could not be audited
    for. And the cell's seed is 3, not 0, so a bootstrap seeded with anything
    else lands on different bounds.

    THE MUTATIONS THIS EXISTS FOR: `margin_interval(..., seed=seed)` ->
    `seed=0` (the bounds move), and `resamples=RESAMPLES` dropped from the call
    (the default is `burden.RESAMPLES`, so the bounds move under the patch).
    """
    monkeypatch.setattr(script, "RESAMPLES", 40)
    record = script.measure_cell(**_cell_kwargs(rig))
    assert record["resamples"] == 40

    window_margin = np.asarray(record["window_margin"])
    groups = np.asarray(record["windows"]["episode"])

    def recompute(h, **over):
        options = dict(h=h, resamples=40, seed=SEED) | over
        return burden.margin_interval(window_margin, groups, **options)

    reached = False
    for h in burden.REPORTED_H:
        entry = record["margin"][str(h)]
        assert (entry["point"], entry["ci_low"], entry["ci_high"]) == recompute(h), h
        # The fixture REACHES the mutations: a different seed, and the shipped
        # draw count, each give different bounds at some horizon.
        other_seed = recompute(h, seed=0)
        other_draws = recompute(h, resamples=burden.RESAMPLES)
        reached = reached or other_seed[1:] != recompute(h)[1:] or other_draws[1:] != recompute(h)[1:]
    assert reached


def test_the_record_survives_a_round_trip_through_write_record(rig, record, tmp_path):
    """A record that is not identical before and after the write is a record the
    read phase would read differently from what was measured -- most
    treacherously through the keys, which JSON turns into strings."""
    written = script.write_record(tmp_path / "burden.json", record)
    assert {k: v for k, v in written.items() if k != "nonfinite"} == record
    assert load_record(tmp_path / "burden.json")["margin"] == record["margin"]


# ---------------------------------------------------------------------------
# The ladder: every rung comes from the sweep, and the decomposition is exact.
# ---------------------------------------------------------------------------


def test_every_rung_and_the_floor_in_the_record_are_the_real_passes(rig, record):
    sweep = _independent_sweep(rig)
    for k in REGROUNDING_KS:
        np.testing.assert_array_equal(record["curves"]["rungs"][str(k)], sweep.curve(k))
    np.testing.assert_array_equal(record["curves"]["floor_position"], sweep.reference.floor_position)
    np.testing.assert_array_equal(record["curves"]["rssm_position"], sweep.curve(HORIZON))
    # The rungs are distinct curves, so the comparison above cannot be satisfied
    # by one curve repeated.
    assert len({tuple(sweep.curve(k)) for k in REGROUNDING_KS}) == len(REGROUNDING_KS)


def test_burden_and_compounding_are_read_against_the_floor_and_the_k_one_rung(record):
    """`burden(k) = rung(k) - floor` and `compounding(k) = rung(k) - rung(1)`.

    THE MUTATION THIS EXISTS FOR: `compounding(sweep.curve(k), floor)` -- the
    drift `compounding`'s own docstring warns of, which silently returns
    `burden(k)` and makes the k=1 control read nonzero.
    """
    floor = np.asarray(record["curves"]["floor_position"])
    rungs = {k: np.asarray(record["curves"]["rungs"][str(k)]) for k in REGROUNDING_KS}
    for k in REGROUNDING_KS:
        np.testing.assert_array_equal(record["burden_by_k"][str(k)], rungs[k] - floor)
        np.testing.assert_array_equal(record["compounding_by_k"][str(k)], rungs[k] - rungs[1])
    assert np.abs(record["burden_by_k"]["1"]).max() > 0.0, "k=1 must sit off the floor"


def test_the_identity_residual_is_the_measured_maximum_over_the_rungs(record):
    """The residual is a MEASURED float. Hard-coding it to 0.0 is the defect M3l
    shipped in a legend line and has not yet fixed.

    THE MUTATION THIS EXISTS FOR: `controls["identity_residual"]` -> `0.0`. This
    fixture's curves sit a factor of 100 apart, outside the exact regime
    `IDENTITY_TOLERANCE` documents, so the true residual is a few units in the
    last place -- nonzero, and the test asserts that it is.
    """
    expected = max(
        float(np.max(np.abs(
            np.asarray(record["burden_by_k"][str(k)])
            - (np.asarray(record["burden_by_k"]["1"]) + np.asarray(record["compounding_by_k"][str(k)]))
        )))
        for k in REGROUNDING_KS
    )
    controls = record["controls"]
    assert isinstance(controls["identity_residual"], float)
    assert expected > 0.0, "the fixture must reach a nonzero residual"
    assert controls["identity_residual"] == expected
    assert controls["identity_residual"] <= burden.IDENTITY_TOLERANCE


def test_the_open_loop_control_is_read_against_the_reference_it_was_given(rig):
    """`open_loop_divergence` compares the k=horizon rung with the reference's own
    open-loop curve, so it must be read against THE REFERENCE `measure_cell` WAS
    HANDED. That reference is perturbed by 0.25 at one step, and the divergence
    must be 0.25.

    THE MUTATIONS THIS EXISTS FOR: `sweep.open_loop_divergence(sweep.reference)`
    (the sweep's own copy, which is never perturbed, reads 0.0), and a hard-coded
    `0.0`.
    """
    real = _reference(rig, DriftingModel())
    rssm = real.rssm_position.copy()
    rssm[2] += 0.25
    record = script.measure_cell(**_cell_kwargs(rig, reference=replace(real, rssm_position=rssm)))
    assert record["controls"]["open_loop_divergence"] == pytest.approx(0.25, abs=1e-9)


def test_the_controls_with_known_answers_are_measured_not_asserted(rig, record):
    """On the drifting model the open loop reproduces bitwise, k=1 is NOT the
    floor, and the k=1 rung's compounding is exactly zero. On the exact oracle
    k=1 IS bitwise the floor -- so the same field reads True there, which is
    what proves it is read off the sweep and not written down.

    THE MUTATIONS THIS EXISTS FOR: `"k_one_is_floor": False` as a literal (the
    oracle case reads True), and `compounding(sweep.curve(k), floor)` (the k=1
    entry is then `burden(1)`, not zero).
    """
    controls = record["controls"]
    assert controls["open_loop_divergence"] == 0.0
    assert controls["k_one_is_floor"] is False
    assert controls["compounding_at_k_one_max_abs"] == 0.0

    exact = script.measure_cell(**_cell_kwargs(rig, model=OracleModel()))
    assert exact["controls"]["k_one_is_floor"] is True
    assert exact["controls"]["compounding_at_k_one_max_abs"] == 0.0


def test_the_base_control_is_the_two_medians_at_the_decision_horizon(rig, record):
    """The median true displacement against the median floor error, at
    `DECISION_H` -- the pair `BurdenArm.base_ok` compares. Each is recomputed
    from its own source: the displacement from the episodes' coordinates, the
    floor from a separate call of the sweep."""
    sweep = _independent_sweep(rig)
    displacement = [
        burden.one_step_persistence(window)[burden.DECISION_H - 1]
        for window in _window_targets(rig.paths)
    ]
    base = record["base_control"]
    assert base["displacement_median"] == pytest.approx(float(np.median(displacement)), abs=1e-12)
    assert base["floor_median"] == pytest.approx(
        float(np.median(sweep.window_floor_position[:, burden.DECISION_H - 1])), abs=1e-12,
    )
    assert base["displacement_median"] != base["floor_median"]


# ---------------------------------------------------------------------------
# The phase driver.
# ---------------------------------------------------------------------------


STUDY = {
    ("pixel_ae", 0): {"steps": 20000, "kl_rate_above_free_bits": 0.9126, "kl_dyn_max": 2.95,
                      "git_sha": "study-a"},
    ("frozen_ssl", 0): {"steps": 20001, "kl_rate_above_free_bits": 0.746, "kl_dyn_max": 1.25,
                        "git_sha": "study-b"},
}
MODELS = {("pixel_ae", 0): DriftingModel, ("frozen_ssl", 0): OracleModel}


def _recognisable(real, index: int):
    """`real` with its three position curves replaced by values no pass over these
    models can produce: thousands of map units where a real floor is a few and a
    real rung is tens, each curve on its own offset and each CELL on its own base,
    so a record paired with another cell's reference is as visible as one that
    recomputed it. Binary fractions, so nothing is lost to a JSON round trip."""
    steps = np.arange(1, HORIZON + 1, dtype=np.float64)
    base = 1000.0 * (index + 1)
    return replace(
        real,
        rssm_position=base + 0.5 + 0.125 * steps,
        persistence_position=base + 0.25 + 0.0625 * steps,
        floor_position=base + 0.75 + 0.25 * steps,
    )


def _phase_env(tmp_path, monkeypatch, *, missing=(), refuse=None):
    """A real `ReplayBuffer`, the real split and the real `measure_cell`; only
    `load_cell` and `prepare_cell` are faked, because they need nine checkpoints.

    The fakes BEHAVE. `prepare_cell` refuses a cell unless `args.out` is the
    directory holding its checkpoint -- the study directory, which the record
    directory is not -- and it hands each cell a DIFFERENT model, so a record
    paired with the wrong cell reads differently.

    The `Prepared` it returns is as the real one is: `common` carries the
    protocol kwargs `evaluate_rollout` takes, and `reference` is a rollout of the
    cell's own model over the real split. Its position curves are then made
    RECOGNISABLE (`_recognisable`), so a record can be asked which pass it
    carries. `args.references[(arm, seed)]` is `(verified, recomputed)`: what
    `prepare_cell` handed over, and what a second `evaluate_rollout` would give.
    """
    data = tmp_path / "data"
    data.mkdir()
    episodes = [_wobbling_episode(60 + 7 * i, i) for i in range(20)]
    for index, episode in enumerate(episodes):
        save_episode(episode, data / f"ep_{index:06d}_len{episode.length:05d}.npz")
    source = tmp_path / "study"
    source.mkdir()
    for (arm, seed) in MODELS:
        (source / f"world_model_{arm}_seed{seed}.pt").write_bytes(b"")
    out = tmp_path / "burden_out"
    probe = _linear_probe(episodes)
    prepared_for = []
    references = {}

    def fake_load_cell(directory, arm, seed):
        if (arm, seed) in missing:
            raise script.CellMissing(f"{arm} seed {seed}: no checkpoint at {directory}")
        return types.SimpleNamespace(
            arm=arm, seed=seed, record=STUDY[(arm, seed)], diagnostic={},
            checkpoint=Path(directory) / f"world_model_{arm}_seed{seed}.pt",
        )

    def fake_prepare_cell(args, cell, device, train, val):
        prepared_for.append((cell.arm, cell.seed))
        if not (Path(args.out) / f"world_model_{cell.arm}_seed{cell.seed}.pt").exists():
            return script.EXIT_NO_CHECKPOINTS, None
        if refuse == (cell.arm, cell.seed):
            return script.EXIT_RECORD_MISMATCH, None
        model = MODELS[(cell.arm, cell.seed)]()
        common = dict(
            context=CONTEXT, horizon=HORIZON, seed=cell.seed, device=CPU,
            feature_backbone=None,
        )
        recomputed = evaluate_rollout(model, val, probe, **common)
        verified = _recognisable(recomputed, index=len(references))
        references[(cell.arm, cell.seed)] = (verified, recomputed)
        return script.EXIT_OK, types.SimpleNamespace(
            model=model, embedding_probe=probe, common=common, context=CONTEXT,
            horizon=HORIZON, reference=verified,
        )

    monkeypatch.setattr(script, "load_cell", fake_load_cell)
    monkeypatch.setattr(script, "prepare_cell", fake_prepare_cell)
    args = types.SimpleNamespace(
        out=out, source=source, data=data, device="cpu", context=None, horizon=None,
        arms=["pixel_ae", "frozen_ssl"], seeds=[0], references=references,
    )
    return args, prepared_for


def test_the_phase_writes_one_labelled_record_per_cell_from_that_cells_own_model(
    tmp_path, monkeypatch,
):
    """Nine records once went out serialised to ONE payload with the labels
    dropped. Here the two cells carry different models and different study
    records, and each file must hold its own cell's numbers.

    The checkpoint is looked for in `args.source`: the fake `prepare_cell`
    refuses a directory that does not hold it, and `args.out` does not.
    """
    args, _ = _phase_env(tmp_path, monkeypatch)
    assert script.measure_phase(args) == script.EXIT_OK
    assert sorted(p.name for p in args.out.iterdir()) == [
        "burden_frozen_ssl_seed0.json", "burden_pixel_ae_seed0.json",
    ]
    drifting = load_record(args.out / "burden_pixel_ae_seed0.json")
    exact = load_record(args.out / "burden_frozen_ssl_seed0.json")
    assert (drifting["arm"], drifting["seed"]) == ("pixel_ae", 0)
    assert (exact["arm"], exact["seed"]) == ("frozen_ssl", 0)
    assert drifting["controls"]["k_one_is_floor"] is False
    assert exact["controls"]["k_one_is_floor"] is True
    assert (drifting["step"], exact["step"]) == (20000, 20001)
    assert (drifting["kl_rate_above_free_bits"], exact["kl_rate_above_free_bits"]) == (0.9126, 0.746)
    assert drifting["git_sha"] == exact["git_sha"] == git_sha()
    assert drifting["seed"] == 0 and drifting["resamples"] == burden.RESAMPLES


def test_the_record_carries_the_reference_prepare_cell_verified(tmp_path, monkeypatch):
    """`prepare_cell` runs a val rollout to PROVE the loaded checkpoint reproduces
    the study record, and refuses when it does not. The floor and the canonical
    curves in a record must be THAT rollout's, not a second pass's: a second pass
    is identical in practice (`evaluate_rollout` seeds its sampler) but nothing
    checked it, and a record carrying it is no longer provably the reproduction.

    Asked of the record by VALUE. Each cell's `prepared.reference` carries
    recognisable position curves (`_recognisable`), so the record's `floor`,
    `rssm` and `persistence` curves equal them exactly, and the burden is read
    against that floor. Asserting `evaluate_rollout` was never CALLED would pass
    a refactor that calls it and discards the result; this does not.

    THE MUTATIONS THIS EXISTS FOR, each of which makes `measure_cell` carry a
    freshly computed `evaluate_rollout` instead of the verified one: a local
    `reference = evaluate_rollout(model, val_paths, probe, **common)` in
    `measure_cell` (the `reference=` it was handed is then ignored), and
    `measure_phase` passing `reference=evaluate_rollout(prepared.model, val,
    prepared.embedding_probe, **prepared.common)` rather than `prepared.reference`.
    The fixture REACHES both: `prepared.common` is the real kwargs, so both run,
    and the freshly computed curves differ from the verified ones (asserted below,
    so the test cannot pass because the two coincide).
    """
    args, _ = _phase_env(tmp_path, monkeypatch)
    assert script.measure_phase(args) == script.EXIT_OK
    assert sorted(args.references) == [("frozen_ssl", 0), ("pixel_ae", 0)]
    curves = ("floor_position", "rssm_position", "persistence_position")
    for (arm, seed), (verified, recomputed) in args.references.items():
        carried = load_record(args.out / f"burden_{arm}_seed{seed}.json")
        for name in curves:
            # Reached: a second pass gives other numbers, so equality below is
            # not the two passes agreeing.
            assert not np.array_equal(getattr(recomputed, name), getattr(verified, name)), name
            np.testing.assert_array_equal(
                carried["curves"][name], getattr(verified, name), err_msg=f"{arm} {name}",
            )
        # The ladder is read against the verified floor, rung by rung.
        for k in REGROUNDING_KS:
            np.testing.assert_array_equal(
                carried["burden_by_k"][str(k)],
                np.asarray(carried["curves"]["rungs"][str(k)]) - verified.floor_position,
            )
    # Each cell carries ITS OWN verified reference, not another cell's.
    pixel, frozen = (args.references[key][0] for key in (("pixel_ae", 0), ("frozen_ssl", 0)))
    assert not np.array_equal(pixel.floor_position, frozen.floor_position)


def test_the_phase_scores_the_studys_own_validation_split(tmp_path, monkeypatch):
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split

    args, _ = _phase_env(tmp_path, monkeypatch)
    script.measure_phase(args)
    _, val = episode_split(
        ReplayBuffer(args.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
    )
    record = load_record(args.out / "burden_pixel_ae_seed0.json")
    assert record["episodes"]["val"] == [p.name for p in val]
    assert record["windows"]["total"] == sum(
        len(window_starts(load_episode(p).length, CONTEXT, HORIZON)) for p in val
    )


def test_the_phase_names_a_missing_cell_and_measures_nothing(tmp_path, monkeypatch, capsys):
    args, prepared_for = _phase_env(tmp_path, monkeypatch, missing={("frozen_ssl", 0)})
    assert script.measure_phase(args) == script.EXIT_NO_CHECKPOINTS
    assert "frozen_ssl seed 0" in capsys.readouterr().out
    assert prepared_for == []
    assert not args.out.exists() or list(args.out.iterdir()) == []


def test_the_phase_stops_at_the_first_refusal_and_writes_nothing_for_it(tmp_path, monkeypatch):
    args, _ = _phase_env(tmp_path, monkeypatch, refuse=("frozen_ssl", 0))
    assert script.measure_phase(args) == script.EXIT_RECORD_MISMATCH
    assert sorted(p.name for p in args.out.iterdir()) == ["burden_pixel_ae_seed0.json"]


def test_the_phase_measures_a_repeated_arm_once(tmp_path, monkeypatch):
    args, prepared_for = _phase_env(tmp_path, monkeypatch)
    args.arms = ["pixel_ae", "pixel_ae"]
    args.seeds = [0, 0]
    assert script.measure_phase(args) == script.EXIT_OK
    assert prepared_for == [("pixel_ae", 0)]


def test_the_cell_line_prints_the_numbers_the_verdict_is_read_from(
    tmp_path, monkeypatch, capsys,
):
    args, _ = _phase_env(tmp_path, monkeypatch)
    script.measure_phase(args)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    record = load_record(args.out / "burden_pixel_ae_seed0.json")
    margin = record["margin"][str(burden.DECISION_H)]
    line = lines[0]
    assert line.startswith("pixel_ae seed 0:")
    for number in (margin["point"], margin["ci_low"], margin["ci_high"]):
        assert f"{number:+.4f}" in line
    assert f"{record['controls']['identity_residual']:.2e}" in line
    assert str(args.out / "burden_pixel_ae_seed0.json") in line


def test_main_takes_its_cells_and_directories_from_the_command_line(tmp_path, monkeypatch):
    """The flag names reach the fields `measure_phase` reads: a record lands in
    the directory `--out` named, for the cell `--arms`/`--seeds` named."""
    args, _ = _phase_env(tmp_path, monkeypatch)
    status = script.main([
        "--phase", "measure", "--source", str(args.source), "--out", str(args.out),
        "--data", str(args.data), "--device", "cpu", "--arms", "frozen_ssl", "--seeds", "0",
    ])
    assert status == script.EXIT_OK
    assert [p.name for p in args.out.iterdir()] == ["burden_frozen_ssl_seed0.json"]
