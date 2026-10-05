"""M3m's measure phase: the ladder, the model-free baseline, one record per cell -- and
the retired read phase, which prints the margin the old records stored and refuses.

EVERY NUMBER IN THESE TESTS COMES FROM THE REAL CALL PATH. `measure_cell` runs
the real `regrounding_sweep` against the oracle-family models `test_rollout.py`
and `test_diagnostics.py` already use, so a record's burden can only have been
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
    straight line the true displacement is one constant step in every window, so
    a baseline cut one frame late reads the same rows as one cut on time.
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

import copy
import importlib.util
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mbfps.data.episode import load_episode, save_episode
from mbfps.eval import burden, pooling
from mbfps.eval.diagnostics import (
    REGROUNDING_KS, action_intervention_ladder, regrounding_sweep,
)
from mbfps.eval.probe import apply_probe, fit_probe, probe_targets
from mbfps.eval.rollout import evaluate_rollout
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.eval.windows import window_starts
from tests.eval.test_rollout import (
    KEYS, STEP, DriftingModel, OracleModel, _OracleRSSM, oracle_probe, synthetic_episode,
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
GATE_H = 45
"""The M3 gate's own horizon, which the base control and the rulers are read at. A
literal here, not `script.DECISION_H`: an expected value never comes from the thing
under test, and a drifted constant must not be able to move both sides."""
REPORTED_HORIZONS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
"""The reporting grid M3m's records carry a margin at, spelled out for the same
reason as `GATE_H`."""
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


def _reference(rig, model, *, context=CONTEXT, horizon=HORIZON, feature_backbone=None):
    """The rollout `prepare_cell` would hand back for `model`: the real
    `evaluate_rollout`, at the cell's seed and through the cell's probe."""
    return evaluate_rollout(
        model, rig.paths, rig.probe, context=context, horizon=horizon, seed=SEED,
        device=CPU, feature_backbone=feature_backbone,
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
            feature_backbone=kwargs["feature_backbone"],
        )
    return kwargs


@pytest.fixture(scope="module")
def record(rig):
    return script.measure_cell(**_cell_kwargs(rig))


def _independent_sweep(rig, model=None, *, seed=SEED):
    """The real sweep, called again by the TEST, so what the record's numbers are
    compared with was not produced by the code under test."""
    return regrounding_sweep(
        model or DriftingModel(), rig.paths, rig.probe, ks=REGROUNDING_KS,
        context=CONTEXT, horizon=HORIZON, seed=seed, device=CPU, feature_backbone=None,
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


def _expected_k_one_rungs(rig) -> np.ndarray:
    """The k=1 rung of `DriftingModel` over the rig, window by window, derived
    WITHOUT the sweep: from the rig's linear probe over the episodes' frame tags.

    `DriftingModel` advances the frame tag by TWO per action. At k=1 every step is
    imagined from a state re-grounded on the real frame before it, so step `j` of a
    window cut at `start` imagines the tag of frame `start + context + j - 1`,
    plus two: the frame the step produced is `f = start + context + j`, and the
    model says `f + 1`. The rung is then the Euclidean distance, in the probe's
    first two columns (`pos_x`, `pos_y`), between the probe's reading of tag
    `f + 1` and the episode's true position at frame `f`.

    Every window of every episode, in the order the paths are GIVEN and
    `window_starts` within each, so row `w` is the window the baseline's row `w`
    is cut from. No `regrounding_sweep`, no `baseline_rows`, no `probe_targets`.
    """
    rows = []
    for path in rig.paths:
        episode = load_episode(path)
        for start in window_starts(episode.length, CONTEXT, HORIZON):
            frames = start + CONTEXT + np.arange(1, HORIZON + 1)
            predicted = apply_probe(rig.probe, (frames + 1.0)[:, None])[:, :2]
            true = episode.privileged[frames][:, 1:3].astype(np.float64)
            rows.append(np.linalg.norm(predicted - true, axis=1))
    return np.stack(rows)


def _canonical(labels) -> np.ndarray:
    """The partition a label array induces, read off the labels' RANK order.

    NOT independent of the numbers used -- `np.unique` ranks by value, not by first
    appearance, so two label arrays that number the same blocks in a different
    order would read as different partitions. That is harmless here because both
    inputs are monotone (a label array walks its episodes in order and never goes
    back), and it is a limit of this helper, not a property the callers rely on."""
    return np.unique(np.asarray(labels), return_inverse=True)[1]


# ---------------------------------------------------------------------------
# The exits.
# ---------------------------------------------------------------------------


def test_the_exit_codes_are_this_milestones_own():
    """39/40 are M3j's, 41/42 M3k's, 43/44 M3l's, and 45/46 are this script's, now
    HISTORICAL: nothing returns them, because the reading that did is retired. They
    stay defined so that a number M3m reported is not reissued to another failure,
    and because the exit-code registry test holds them."""
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
    refusal in `measure_cell`. It is the ONLY tie between the baseline's loop and
    the sweep's: nothing subtracts the two arrays any more, so with the refusal gone
    a baseline cut on a different window rule is no longer stopped by a numpy
    broadcasting error. It is written into the record instead -- the base control's
    median then taken over windows that are not the sweep's.

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


def test_the_k_one_rung_is_each_windows_own_in_the_order_given(rig, record):
    """Row `w` of the sweep is the window row `w` of the baseline is cut from, by
    VALUE. The shape refusal sees only a count that drifted, and a reorder that
    keeps the shape -- two episodes with the same number of windows swapped, or
    `sorted()` on names that happen to give the same block sizes -- passes it.

    The sweep's own k=1 rows, a second call of it, and the k=1 rung the record
    carries are each asked against `_expected_k_one_rungs`, which does not call the
    sweep. And the baseline curve the record carries is asked against the baseline
    computed here, window by window, from the episodes' own targets.

    THE MUTATIONS THIS EXISTS FOR, none of which a comparison against a second
    sweep could see: `val_paths = sorted(val_paths)` inside `regrounding_sweep`, and
    two equal-count episodes (0 and 3, one window each) swapped inside it -- the
    same wrong order on both sides of that comparison, a different one from the closed
    form. (Swapping rows of `one` inside `measure_cell` is no longer expressible:
    nothing there reads its values but the finiteness refusal.)
    """
    expected = _expected_k_one_rungs(rig)
    assert expected.shape == (sum(EXPECTED_WINDOWS), HORIZON)
    # Reached: every window's rung is its own, so a swap or a reorder moves values.
    assert len({tuple(row) for row in expected}) == expected.shape[0]
    assert np.abs(expected[0] - expected[3]).max() > 0.5, "windows 0 and 3 are the swapped pair"

    one = _independent_sweep(rig).window_position[1]
    np.testing.assert_allclose(one, expected, rtol=1e-6, atol=1e-9)

    np.testing.assert_allclose(
        record["curves"]["rungs"]["1"], expected.mean(axis=0), rtol=1e-6, atol=1e-9,
    )
    rows = np.stack([burden.one_step_persistence(w) for w in _window_targets(rig.paths)])
    assert rows.shape == expected.shape
    np.testing.assert_allclose(
        record["curves"]["one_step_persistence"], rows.mean(axis=0), rtol=1e-6, atol=1e-9,
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_measure_cell_refuses_a_non_finite_baseline_by_name(rig, monkeypatch, bad):
    """The baseline's rows become the base control's median and the recorded
    `one_step_persistence` curve. Without the refusal a poisoned row is written into
    that curve -- `study.write_record` nulls it, and the read meets a hole instead of
    the cause; with it the run stops before a record is written, naming the array.

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
    """The same refusal on the sweep's side. The sweep's MEAN curves stay finite
    here -- only the retained per-window rows are poisoned -- and those rows are
    what the two rulers are computed from, so without the refusal the poison is
    carried into the record by whichever step of them it sits in.

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
def test_measure_cell_refuses_a_protocol_the_reading_cannot_be_taken_from(
    rig, monkeypatch, over, match,
):
    """The base control and the rulers are read at `DECISION_H`, and every
    compounding and every ruler is read off the k=1 rung. Refused by name before
    any pass is paid for, rather than as an IndexError or a KeyError after the
    sweep.

    "BEFORE ANY PASS" IS ASKED, NOT ASSUMED: the sweep and the baseline are
    replaced by functions that fail the test if they are called, so the refusal
    has to win the race. Left to run, they let the refusal sit anywhere in the
    function and the test could not tell where.

    THE MUTATIONS THIS EXISTS FOR: `_require_a_readable_protocol(...)` moved to
    just after the `regrounding_sweep(...)` call, and to just after the
    `baseline_rows(...)` call. Both survived all 55 tests, because a refusal that
    comes late is still a refusal with the same words.
    """
    kwargs = _cell_kwargs(rig, **over)

    def paid_for(*args, **kwargs):
        raise AssertionError("a pass was paid for before the protocol was refused")

    monkeypatch.setattr(script, "regrounding_sweep", paid_for)
    monkeypatch.setattr(script, "baseline_rows", paid_for)
    with pytest.raises(SystemExit, match=match):
        script.measure_cell(**kwargs)


# ---------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------


def test_the_record_carries_every_protocol_parameter_it_was_taken_at(record):
    """The record states the horizon its controls were read at, the ladder, the
    grid, the tolerance the identity control was taken at, and the split -- so the
    permanent artefact can be audited for each.

    THE MUTATION THIS EXISTS FOR: dropping any of these keys. It pins that each
    is PRESENT and AGREES WITH the shipped value, and nothing more: a literal that
    equals the constant -- `"decision_h": 45` -- passes it, which is the one thing it
    cannot see. WHERE EACH VALUE IS READ FROM is pinned by
    `test_each_protocol_constant_is_recorded_from_the_modules_own_name`.

    The expected values are literals, and `DECISION_H`'s is asserted against one
    (`GATE_H`): it is a constant this script holds since M3n removed the one in
    `burden`, so a drifted copy must not be able to move both sides.
    """
    assert script.DECISION_H == GATE_H
    assert record["decision_h"] == GATE_H
    assert tuple(record["reported_h"]) == REPORTED_HORIZONS
    assert tuple(record["ks"]) == tuple(REGROUNDING_KS)
    assert record["identity_tolerance"] == 1e-9
    assert record["split_seed"] == SPLIT_SEED
    assert record["device"] == "cpu"
    assert record["git_sha"] == git_sha()
    for key in (
        "arm", "seed", "controls", "base_control", "burden_by_k", "compounding_by_k",
        "curves", "windows", "git_sha", "step", "kl_rate_above_free_bits", "kl_dyn_max",
        "episodes", "context", "horizon", "torch_version", "rulers",
    ):
        assert key in record, key
    assert (record["arm"], record["seed"]) == ("pixel_ae", SEED)
    assert (record["context"], record["horizon"]) == (CONTEXT, HORIZON)


@pytest.mark.parametrize(
    "constant, patched, key, expected",
    [
        ("DECISION_H", 30, "decision_h", 30),
        ("REPORTED_H", (1, 7, 45), "reported_h", [1, 7, 45]),
        ("IDENTITY_TOLERANCE", 1e-3, "identity_tolerance", 1e-3),
        ("SPLIT_SEED", 7, "split_seed", 7),
    ],
)
def test_each_protocol_constant_is_recorded_from_the_modules_own_name(
    rig, monkeypatch, constant, patched, key, expected,
):
    """The record states the constant THE MODULE HOLDS: the module's name is patched
    to a value the shipped constant is not, and the record must follow it.

    Comparing a record to the shipped value cannot do this. A literal `45` EQUALS it,
    so recording one passes every test that compares against the shipped value; only
    a patched module attribute tells a literal from a read.

    THE MUTATIONS THIS EXISTS FOR, one per case, each a literal where the module's
    constant stood: `"decision_h": 45`, `"reported_h": [1, 2, 3, 5, 8, 10, 15, 20,
    30, 45]`, `"identity_tolerance": 1e-9`, `"split_seed": 0`. `DECISION_H` is also
    read by the measurement itself (the base control and the rulers) and `REPORTED_H`
    by the protocol guard, so the patched values are ones the guard still accepts at
    the cell's horizon of 45.
    """
    assert getattr(script, constant) != patched
    monkeypatch.setattr(script, constant, patched)
    record = script.measure_cell(**_cell_kwargs(rig))
    assert record[key] == expected


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


@pytest.mark.parametrize(
    "missing", ["steps", "kl_rate_above_free_bits", "kl_dyn_max", "git_sha"],
)
def test_a_study_record_missing_a_training_number_is_refused_not_defaulted(rig, missing):
    """ONE KEY LEFT OUT PER CASE. A record built as `{"steps": 1}` lacks two keys
    at once, so the strict read of the second is what raises and a lenient read of
    the first -- `.get("steps", 0)`, or either KL field with a default -- changes
    nothing. Each key is the only one missing here, so each is pinned by itself.

    `git_sha` IS ONE OF THEM, because the read phase compares it across records
    as a protocol field. A default of "unknown" there is a value `study.git_sha()`
    itself writes when git cannot answer, so a record that merely LACKS the key
    would be indistinguishable from one written where git was unavailable, and
    cells from different code versions would pool into one finding with no
    refusal at all. (`study.UNKNOWN_GIT_SHA` present in the record is a different
    thing and still carried.)

    THE MUTATIONS THIS EXISTS FOR: `study_record["steps"]` ->
    `study_record.get("steps", 0)`, and the same for `kl_rate_above_free_bits` and
    `kl_dyn_max`, each of which survived the previous single `{"steps": 1}` case;
    and `study_record["git_sha"]` -> `study_record.get("git_sha", "unknown")`, the
    line as it was written, which survived every test."""
    study = {
        "steps": 12345, "kl_rate_above_free_bits": 0.25, "kl_dyn_max": 7.5,
        "git_sha": "study-sha",
    }
    del study[missing]
    with pytest.raises(KeyError, match=missing):
        script.measure_cell(**_cell_kwargs(rig, study_record=study))


def test_the_record_names_its_episodes_and_clusters_its_windows(rig, record):
    assert record["episodes"]["val"] == [p.name for p in rig.paths]
    assert record["windows"]["total"] == sum(EXPECTED_WINDOWS)
    assert len(record["windows"]["episode"]) == record["windows"]["total"]
    assert np.bincount(
        record["windows"]["episode"], minlength=len(LENGTHS)
    ).tolist() == list(EXPECTED_WINDOWS)


def test_the_record_survives_a_round_trip_through_write_record(rig, record, tmp_path):
    """A record that is not identical before and after the write is a record the
    read phase would read differently from what was measured -- most
    treacherously through the keys, which JSON turns into strings."""
    written = script.write_record(tmp_path / "burden.json", record)
    assert written["nonfinite"] == {}, "a measured cell must hold no non-finite value"
    assert {k: v for k, v in written.items() if k != "nonfinite"} == record
    assert load_record(tmp_path / "burden.json")["curves"]["rungs"] == record["curves"]["rungs"]


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


def test_the_floor_is_controlled_between_the_two_passes_that_read_it(rig, record):
    """The floor is read from two passes: `reference.floor_position` carries every
    burden and the identity residual, and `sweep.reference.floor_position` is what
    `k_one_is_floor` compares against. `open_loop_divergence == 0.0` shows the two
    agree on the RSSM curve; it says nothing about the floor curve. So the
    divergence between the two floors is recorded beside it.

    The reference handed in has its floor moved at two steps, neither of them the
    decision horizon, by -0.25 and +0.125: the divergence is the LARGEST ABSOLUTE
    difference over the horizon, 0.25, and a first step, a mean, a sum, an
    unsigned maximum or a read at `DECISION_H` each gives another number.

    THE MUTATIONS THIS EXISTS FOR: a hard-coded `0.0`; the sweep's floor compared
    with itself (it reads 0.0 against every reference, so only a perturbed one can
    tell it from the real control); another curve of the reference in the floor's
    place (nonzero on the unperturbed record); and `np.max` without `np.abs`,
    `np.mean`, or a read at one step in place of the maximum over the horizon.
    """
    assert record["controls"]["floor_divergence"] == 0.0
    assert type(record["controls"]["floor_divergence"]) is float

    real = _reference(rig, DriftingModel())
    floor = real.floor_position.copy()
    floor[7] -= 0.25
    floor[20] += 0.125
    moved = script.measure_cell(**_cell_kwargs(rig, reference=replace(real, floor_position=floor)))
    assert moved["controls"]["floor_divergence"] == pytest.approx(0.25, abs=1e-9)
    # Only the floor moved: the two controls are not the same number.
    assert moved["controls"]["open_loop_divergence"] == 0.0


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
        burden.one_step_persistence(window)[GATE_H - 1]
        for window in _window_targets(rig.paths)
    ]
    base = record["base_control"]
    assert base["displacement_median"] == pytest.approx(float(np.median(displacement)), abs=1e-12)
    assert base["floor_median"] == pytest.approx(
        float(np.median(sweep.window_floor_position[:, GATE_H - 1])), abs=1e-12,
    )
    assert base["displacement_median"] != base["floor_median"]


# --- the paired rulers -------------------------------------------------------
#
# The record keeps the MEAN curves and drops the per-window rows they came from, so
# the interval on the one conclusion the ladder carries (the share of the open-loop
# cost that is compounding) can only be computed from what is recorded at
# measurement time. `RegroundingSweep` already has the paired rulers; the record
# carries two of them, at `DECISION_H`.


def _expected_rulers(sweep, last, h):
    """The two rulers off a sweep the TEST ran, at 1-indexed horizon step `h`."""
    return (
        float(sweep.floor_margin_standard_error(1)[h - 1]),
        float(sweep.paired_standard_error(last, 1)[h - 1]),
    )


def test_the_rulers_are_the_sweeps_own_paired_standard_errors(rig, record):
    """`floor_margin_standard_error(1)` -- the ruler on `burden(1)` -- and
    `paired_standard_error(ks[-1], 1)` -- the ruler on `compounding(ks[-1])` --
    each at `DECISION_H`, compared by value with a second sweep the test ran.

    The two are told apart from the UNPAIRED spread of a single curve, which the
    sweep names `curve_standard_error` and documents as overstating a k-to-k bar
    by 1.7x to 3.9x: this fixture's paired rulers differ from every unpaired
    spread by far more than float noise, so a ruler taken from the wrong method
    cannot pass.

    THE MUTATIONS THIS EXISTS FOR: `curve_standard_error(1)` for the first and
    `curve_standard_error(ks[-1])` for the second (the unpaired spreads the spec
    calls an overstatement); `paired_standard_error(ks[-1], 3)` (the wrong second
    rung); `floor_margin_standard_error(ks[-1])` (the wrong rung against the
    floor). `paired_standard_error(1, ks[-1])` is an EQUIVALENT mutant: the spread of
    a difference is the spread of its negation.
    """
    sweep = _independent_sweep(rig)
    last = REGROUNDING_KS[-1]
    floor_margin, paired = _expected_rulers(sweep, last, GATE_H)
    assert record["rulers"]["floor_margin_standard_error"] == floor_margin
    assert record["rulers"]["paired_standard_error"] == paired
    assert all(type(v) is float for v in record["rulers"].values())

    h = GATE_H - 1
    unpaired = {k: float(sweep.curve_standard_error(k)[h]) for k in REGROUNDING_KS}
    for name, ruler in (("floor_margin", floor_margin), ("paired", paired)):
        for k, spread in unpaired.items():
            assert abs(ruler - spread) > 1e-3 * spread, (name, k)
    # ... and not another rung's paired ruler either.
    assert paired != float(sweep.paired_standard_error(REGROUNDING_KS[1], 1)[h])
    assert floor_margin != float(sweep.floor_margin_standard_error(last)[h])


def test_the_rulers_are_taken_at_the_decision_horizon_the_module_holds(rig, monkeypatch):
    """At `DECISION_H`, read off the module's name -- patched to 30 here, where the
    two rulers read differently than at step 45 -- and not at the last step, the
    first, or a literal 45.

    THE MUTATIONS THIS EXISTS FOR: a literal `45` or `-1` in place of
    `DECISION_H`, and `at_horizon(..., 1)`.
    """
    sweep = _independent_sweep(rig)
    last = REGROUNDING_KS[-1]
    assert _expected_rulers(sweep, last, 30) != _expected_rulers(sweep, last, 45)
    assert _expected_rulers(sweep, last, 30) != _expected_rulers(sweep, last, 1)
    monkeypatch.setattr(script, "DECISION_H", 30)
    rulers = script.measure_cell(**_cell_kwargs(rig))["rulers"]
    floor_margin, paired = _expected_rulers(sweep, last, 30)
    assert rulers == {
        "floor_margin_standard_error": floor_margin, "paired_standard_error": paired,
    }


def test_the_paired_ruler_belongs_to_the_last_rung_the_table_prints(rig):
    """`paired_standard_error` is the ruler on `compounding(ks[-1])`, the quantity
    the table's `comp_k` column prints, so it pairs the LAST rung of the ladder it
    was given with k=1 -- not a literal 45 and not the horizon. The ladder here ends
    in 3 with 45 in the middle, where the last rung and the horizon differ.

    THE MUTATIONS THIS EXISTS FOR: `paired_standard_error(45, 1)` and
    `paired_standard_error(horizon, 1)`, each of which equals the shipped line on
    every ladder that ends at the horizon, which is why this ladder does not.
    """
    ks = (1, 45, 3)
    sweep = regrounding_sweep(
        DriftingModel(), rig.paths, rig.probe, ks=ks, context=CONTEXT, horizon=HORIZON,
        seed=SEED, device=CPU, feature_backbone=None,
    )
    record = script.measure_cell(**_cell_kwargs(rig, ks=ks))
    floor_margin, paired = _expected_rulers(sweep, 3, GATE_H)
    assert record["rulers"]["paired_standard_error"] == paired
    assert paired != _expected_rulers(sweep, 45, GATE_H)[1]
    assert record["rulers"]["floor_margin_standard_error"] == floor_margin


# A horizon step is read through `burden.at_horizon` or it is not read at all.
# Its docstring says that indexing with `h` rather than `h - 1` "shifts every
# reported number by one step and breaks no shape, which is why this is a function
# rather than a convention" -- and for as long as every call site wrote its own
# `- 1`, replacing its body with a raise left the whole script's tests green. These
# tests patch the script's own name for it to a function that returns a number no
# measurement can produce and records the horizon it was asked for, so a site that
# indexes by hand reads a real number where the sentinel should be.
SENTINEL = 123456.789


def _sentinel_at_horizon(asked):
    def at_horizon(curve, h):
        asked.append(h)
        return SENTINEL
    return at_horizon


def test_measure_cell_reads_every_horizon_step_through_at_horizon(rig, monkeypatch):
    """`measure_cell`'s four horizon reads -- the two base-control medians (the
    displacement and the floor) and the two rulers -- are each taken at
    `DECISION_H` through `burden.at_horizon`.

    THE MUTATIONS THIS EXISTS FOR, one per read: `rows[:, DECISION_H - 1]`,
    `sweep.window_floor_position[:, DECISION_H - 1]`, and `[DECISION_H - 1]` on
    either ruler, written by hand again, each of which left the script's tests green
    before this one. Each read is a number that only `at_horizon` could have
    returned, so another site cannot cover for the one mutated.
    """
    asked = []
    monkeypatch.setattr(script, "at_horizon", _sentinel_at_horizon(asked))
    record = script.measure_cell(**_cell_kwargs(rig))
    assert record["base_control"] == {"displacement_median": SENTINEL, "floor_median": SENTINEL}
    assert record["rulers"] == {
        "floor_margin_standard_error": SENTINEL, "paired_standard_error": SENTINEL,
    }
    assert set(asked) == {GATE_H}
    # One call per window per median -- the median is of the windows, not of one
    # read -- and one per ruler.
    assert len(asked) == 2 * sum(EXPECTED_WINDOWS) + 2


# ---------------------------------------------------------------------------
# The arguments handed to the sweep are the ones it USES.
# ---------------------------------------------------------------------------


class _StochasticRSSM(_OracleRSSM):
    """`_OracleRSSM` whose `imagine` adds a standard normal draw to every step.

    THE FAKE EVERY OTHER TEST HERE CANNOT BE. The oracle-family models draw no
    random number, so a sweep handed `seed=0` and one handed the cell's seed
    return the same curves bit for bit, and nothing downstream can tell which it
    was given. `test_latent_capacity_script.py` names the same hazard "A
    STANDING HAZARD" for a `frame_probe` that was handed the right seed while its
    body ignored it: a test that shows an argument was PASSED shows nothing about
    whether it was USED. Here the draw is one `imagine` makes, `evaluate_rollout` and the sweep
    both seed the generator before the traversal, and a different seed gives
    different rungs -- by up to 5.57 map units at the open-loop rung on this rig."""

    def imagine(self, actions, state):
        tag = super().imagine(actions, state)["h"]
        tag = tag + torch.randn_like(tag)
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}


class StochasticModel(OracleModel):
    def __init__(self) -> None:
        super().__init__()
        self.rssm = _StochasticRSSM()


def test_the_sweep_is_run_at_the_cells_seed(rig):
    """`measure_cell` must hand `regrounding_sweep` the CELL'S seed. The
    reference it is given was rolled out at that seed, so the k=horizon rung --
    the same pass, which is what `open_loop_divergence` reads -- reproduces it
    bitwise ONLY if the sweep drew the same stream. Every rung is also asked by
    value, against a second sweep the test ran at the cell's seed.

    THE MUTATION THIS EXISTS FOR: `seed=seed` -> `seed=0` in the `common` dict
    `measure_cell` hands `regrounding_sweep`. It survived all 42 tests, because
    no model here drew a number. With it, the open-loop control reads
    5.569275918609585 where it should read 0.0.

    On correct code the divergence control WOULD flag the mutation -- at run
    time, in the record, after the whole nine-cell measurement had been paid for.
    This is the check that does not wait for that."""
    model = StochasticModel()
    at_cell = _independent_sweep(rig, model)
    at_zero = _independent_sweep(rig, model, seed=0)
    # Reached: the seeds really do give different curves, or equality below is
    # not the sweep reading its seed.
    assert not np.array_equal(at_cell.curve(HORIZON), at_zero.curve(HORIZON))
    assert np.abs(at_cell.curve(HORIZON) - at_zero.curve(HORIZON)).max() > 1.0

    record = script.measure_cell(**_cell_kwargs(rig, model=model))
    assert SEED != 0
    assert record["controls"]["open_loop_divergence"] == 0.0
    for k in REGROUNDING_KS:
        np.testing.assert_array_equal(
            record["curves"]["rungs"][str(k)], at_cell.curve(k), err_msg=f"k={k}",
        )


def _feature_rig(directory: Path, lengths=(70, 120), *, namespaced_offset=100.0, decoy_offset=200.0):
    """Straight-line episodes whose cached features carry the frame tag OFFSET by
    `namespaced_offset` under `random_vit`'s own namespaced suffix, and by
    `decoy_offset` under the suffix a backbone of `None` would name."""
    paths, episodes = [], []
    for index, length in enumerate(lengths):
        episode = synthetic_episode(length)
        path = _episode_file(directory, episode, name_index=index)
        tags = np.arange(length + 1, dtype=np.float32)
        np.save(path.with_suffix(".features_random_vit.npy"),
                (tags + namespaced_offset).reshape(-1, 1, 1, 1))
        np.save(path.with_suffix(".features_None.npy"),
                (tags + decoy_offset).reshape(-1, 1, 1, 1))
        paths.append(path)
        episodes.append(episode)
    return types.SimpleNamespace(paths=paths, episodes=episodes, probe=oracle_probe(episodes[0]))


def test_the_sweep_reads_the_feature_cache_of_the_backbone_it_was_handed(tmp_path):
    """A feature-input arm is fed its backbone's own namespaced cache. The caches
    here carry the frame tag offset by +100, so an exact oracle is exactly 100
    frames of true displacement wrong at every horizon step and every rung -- a
    closed form. A decoy cache under the suffix `None` would name carries +200,
    so a sweep that was not handed the backbone reads another number rather than
    merely crashing.

    THE MUTATION THIS EXISTS FOR: `feature_backbone=feature_backbone` ->
    `feature_backbone=None` in the `common` dict handed to `regrounding_sweep`,
    which survived every test because the pixel rig never reads a cache."""
    feature_rig = _feature_rig(tmp_path)
    model = OracleModel()
    model.input_kind = "features"
    record = script.measure_cell(**_cell_kwargs(
        feature_rig, model=model, feature_backbone="random_vit",
    ))
    for k in REGROUNDING_KS:
        np.testing.assert_allclose(
            record["curves"]["rungs"][str(k)], np.full(HORIZON, 100.0 * STEP), rtol=1e-4,
            err_msg=f"k={k}",
        )


def test_the_sweep_runs_on_the_device_it_was_handed(rig):
    """The device handed to `measure_cell` is the one the sweep traverses on, not
    whichever device the model's first parameter lives on.

    The model's only parameter is on the `meta` device, which no op here reads.
    Handed `cpu`, the sweep runs; handed `None` it falls back to the model's own
    device -- `meta` -- and `_rng_snapshot` refuses a device it has no verified
    generator state for. On the real run the model has been moved to the device
    handed in, so the two coincide: this pins the handoff, not a live hazard.

    THE MUTATION THIS EXISTS FOR: `device=device` -> `device=None` in the
    `common` dict handed to `regrounding_sweep`."""
    model = OracleModel()
    model.dummy = nn.Parameter(torch.zeros(1, device="meta"))
    assert next(model.parameters()).device.type == "meta"
    record = script.measure_cell(**_cell_kwargs(rig, model=model))
    assert record["device"] == "cpu"
    assert record["controls"]["open_loop_divergence"] == 0.0


# ---------------------------------------------------------------------------
# The phase driver.
# ---------------------------------------------------------------------------


PHASE_SEEDS = (1, 2)
"""TWO SEEDS PER ARM, AND NEITHER IS ZERO. With one seed per arm, and that seed 0,
`measure_phase` handing `measure_cell` `seed=0` and `burden_record_path(out, arm,
0)` were both indistinguishable from the real thing, and every test still passed.
The second is the expensive one: in the real nine-cell run it writes every arm's
seed-1 and seed-2 record to the seed-0 file name, so six of nine records are
overwritten and it surfaces only at the read phase, after the GPU time is spent.
Every cell here has a seed that differs from every other cell's, from 0, and from
the study's own `SPLIT_SEED`."""
STUDY = {
    ("pixel_ae", 1): {"steps": 20000, "kl_rate_above_free_bits": 0.9126, "kl_dyn_max": 2.95,
                      "git_sha": "study-a1"},
    ("pixel_ae", 2): {"steps": 20002, "kl_rate_above_free_bits": 0.8125, "kl_dyn_max": 2.5,
                      "git_sha": "study-a2"},
    ("frozen_ssl", 1): {"steps": 20001, "kl_rate_above_free_bits": 0.746, "kl_dyn_max": 1.25,
                        "git_sha": "study-b1"},
    ("frozen_ssl", 2): {"steps": 20003, "kl_rate_above_free_bits": 0.625, "kl_dyn_max": 1.75,
                        "git_sha": "study-b2"},
}
MODELS = {
    ("pixel_ae", 1): DriftingModel, ("pixel_ae", 2): DriftingModel,
    ("frozen_ssl", 1): OracleModel, ("frozen_ssl", 2): OracleModel,
}
CELLS = tuple(STUDY)
"""In the order `measure_phase` visits them: arm-major, seeds in the order given."""
assert set(STUDY) == set(MODELS) == {
    (arm, seed) for arm in ("pixel_ae", "frozen_ssl") for seed in PHASE_SEEDS
}
assert all(seed != 0 and seed != SPLIT_SEED for _, seed in CELLS)


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
        arms=["pixel_ae", "frozen_ssl"], seeds=list(PHASE_SEEDS), references=references,
    )
    return args, prepared_for


def test_the_phase_writes_one_labelled_record_per_cell_from_that_cells_own_model(
    tmp_path, monkeypatch,
):
    """Nine records once went out serialised to ONE payload with the labels
    dropped. Here the four cells carry two different models and four different
    study records, and each file must hold its own cell's numbers.

    THE SEED IS BOUND TWICE, because the cells' seeds are 1 and 2 (`PHASE_SEEDS`).
    The file NAME carries the cell's seed -- a name that carried 0 would write
    the second and third seed of every arm over one file, which the listing below
    is the only thing to see -- and the record carries it too, as `seed`.

    THE MUTATIONS THIS EXISTS FOR, each of which the one-seed-0 fixture let
    through with every test green: `measure_phase` handing `measure_cell`
    `seed=0` instead of `cell.seed` (the record's `seed` reads 0), and
    `burden_record_path(args.out, cell.arm, 0)` instead of `cell.seed` (the listing
    loses two files). Neither is seen through the numbers: the models here draw no
    random number, so the seed moves nothing but the label.

    The checkpoint is looked for in `args.source`: the fake `prepare_cell`
    refuses a directory that does not hold it, and `args.out` does not.
    """
    args, prepared_for = _phase_env(tmp_path, monkeypatch)
    assert script.measure_phase(args) == script.EXIT_OK
    assert prepared_for == list(CELLS)
    assert sorted(p.name for p in args.out.iterdir()) == [
        "burden_frozen_ssl_seed1.json", "burden_frozen_ssl_seed2.json",
        "burden_pixel_ae_seed1.json", "burden_pixel_ae_seed2.json",
    ]
    records = {
        (arm, seed): load_record(args.out / f"burden_{arm}_seed{seed}.json")
        for arm, seed in CELLS
    }
    for (arm, seed), carried in records.items():
        study = STUDY[(arm, seed)]
        assert (carried["arm"], carried["seed"]) == (arm, seed)
        assert carried["controls"]["k_one_is_floor"] is (arm == "frozen_ssl"), (arm, seed)
        assert carried["step"] == study["steps"], (arm, seed)
        assert carried["kl_rate_above_free_bits"] == study["kl_rate_above_free_bits"]
        assert carried["kl_dyn_max"] == study["kl_dyn_max"]
        assert carried["record_git_sha"] == study["git_sha"]
        assert carried["git_sha"] == git_sha()
        assert "margin" not in carried, (arm, seed)


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
    assert sorted(args.references) == sorted(CELLS)
    curves = ("floor_position", "rssm_position", "persistence_position")
    for (arm, seed), (verified, recomputed) in args.references.items():
        carried = load_record(args.out / f"burden_{arm}_seed{seed}.json")
        assert (carried["arm"], carried["seed"]) == (arm, seed)
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
    # Each cell carries ITS OWN verified reference, not another cell's -- and the
    # two seeds of one arm are two cells, so they are told apart as well.
    floors = [tuple(args.references[key][0].floor_position) for key in CELLS]
    assert len(set(floors)) == len(CELLS)


def test_the_phase_scores_the_studys_own_validation_split(tmp_path, monkeypatch):
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split

    args, _ = _phase_env(tmp_path, monkeypatch)
    script.measure_phase(args)
    _, val = episode_split(
        ReplayBuffer(args.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
    )
    record = load_record(args.out / "burden_pixel_ae_seed1.json")
    assert record["episodes"]["val"] == [p.name for p in val]
    assert record["windows"]["total"] == sum(
        len(window_starts(load_episode(p).length, CONTEXT, HORIZON)) for p in val
    )


def test_the_phase_names_a_missing_cell_and_measures_nothing(tmp_path, monkeypatch, capsys):
    args, prepared_for = _phase_env(tmp_path, monkeypatch, missing={("frozen_ssl", 2)})
    assert script.measure_phase(args) == script.EXIT_NO_CHECKPOINTS
    assert "frozen_ssl seed 2" in capsys.readouterr().out
    assert prepared_for == []
    assert not args.out.exists() or list(args.out.iterdir()) == []


def test_the_phase_stops_at_the_first_refusal_and_writes_nothing_for_it(tmp_path, monkeypatch):
    args, _ = _phase_env(tmp_path, monkeypatch, refuse=("frozen_ssl", 1))
    assert script.measure_phase(args) == script.EXIT_RECORD_MISMATCH
    assert sorted(p.name for p in args.out.iterdir()) == [
        "burden_pixel_ae_seed1.json", "burden_pixel_ae_seed2.json",
    ]


def test_the_phase_measures_a_repeated_arm_once(tmp_path, monkeypatch):
    args, prepared_for = _phase_env(tmp_path, monkeypatch)
    args.arms = ["pixel_ae", "pixel_ae"]
    args.seeds = [1, 1]
    assert script.measure_phase(args) == script.EXIT_OK
    assert prepared_for == [("pixel_ae", 1)]


def test_the_cell_line_prints_the_burden_and_the_controls_and_no_margin(
    tmp_path, monkeypatch, capsys,
):
    args, _ = _phase_env(tmp_path, monkeypatch)
    script.measure_phase(args)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(CELLS)
    last = str(REGROUNDING_KS[-1])
    for line, (arm, seed) in zip(lines, CELLS):
        record = load_record(args.out / f"burden_{arm}_seed{seed}.json")
        assert line.startswith(f"{arm} seed {seed}:"), line
        # No margin and no interval: the statistic is retired, and a line that still
        # printed one would be a number in the log that the record no longer holds.
        assert "margin" not in line, line
        # The burden printed is the LAST rung's, read at `GATE_H`: the rung and
        # the step are each written out here rather than read back off the record.
        by_step = record["burden_by_k"][last]
        printed = f"burden(k={last}) {by_step[GATE_H - 1]:+.4f};"
        assert printed in line, line
        # Reached: step 1 and step `GATE_H` read differently, so a line that
        # printed the first step fails; and on the drifting arm the first rung and
        # the last do too (on the exact oracle every rung is the same, so there the
        # rung is told apart only by the `k=` label above).
        assert f"{by_step[0]:+.4f}" != f"{by_step[GATE_H - 1]:+.4f}"
        if arm == "pixel_ae":
            first_rung = record["burden_by_k"]["1"][GATE_H - 1]
            assert f"{first_rung:+.4f}" != f"{by_step[GATE_H - 1]:+.4f}"
        assert f"{record['controls']['identity_residual']:.2e}" in line
        assert str(args.out / f"burden_{arm}_seed{seed}.json") in line


def test_the_cell_line_reads_its_burden_through_at_horizon(record, tmp_path, monkeypatch):
    """The burden the line prints is the one place the measure phase reads a step
    off a record rather than off an array, and it reads it through
    `burden.at_horizon`, the one place the 0- and 1-indexing differ is written.

    THE MUTATION THIS EXISTS FOR: `record['burden_by_k'][last][record['decision_h']
    - 1]` written by hand again.
    """
    asked = []
    monkeypatch.setattr(script, "at_horizon", _sentinel_at_horizon(asked))
    line = script._cell_line(record, tmp_path / "burden.json")
    assert f"burden(k={record['ks'][-1]}) {SENTINEL:+.4f};" in line
    assert asked == [record["decision_h"]]


def test_main_takes_its_cells_and_directories_from_the_command_line(tmp_path, monkeypatch):
    """The flag names reach the fields `measure_phase` reads: a record lands in
    the directory `--out` named, for the cell `--arms`/`--seeds` named."""
    args, _ = _phase_env(tmp_path, monkeypatch)
    status = script.main([
        "--phase", "measure", "--source", str(args.source), "--out", str(args.out),
        "--data", str(args.data), "--device", "cpu", "--arms", "frozen_ssl", "--seeds", "2",
    ])
    assert status == script.EXIT_OK
    assert [p.name for p in args.out.iterdir()] == ["burden_frozen_ssl_seed2.json"]


# ---------------------------------------------------------------------------
# The read phase, retired: nine records, printed superseded, then refused.
#
# THE POOL IS THE REAL RECORD, RELABELLED. Every cell starts as a deep copy of
# the record `measure_cell` writes (`record`), so the schema is the writer's own,
# and only the fields the read phase READS are then overwritten, each from a
# formula in the cell's index `c`: the stored `margin` the table prints, and the
# decision horizon it is read at. The expected table is built from the same
# formulas -- never from the records, and never through the script's formatter.
#
#   * Three arms x seeds 0, 1, 2: nine cells, no two alike in any number read.
#     The arms are given in `ARMS` order, whose index is NOT `sorted()` order
#     (`frozen_ssl` sorts first), so a read that sorts or enumerates differently
#     prints another cell's numbers on a row.
#   * The margin at every reported horizon but the decision horizon is a decoy
#     (about a thousand), so a read at the wrong horizon is not a plausible one.
#   * The decision horizon is also read at 30, not only 45: at 45 the step is the
#     LAST one, where `[h - 1]` and `[-1]` are one element.
# ---------------------------------------------------------------------------


READ_ARMS = ("pixel_ae", "frozen_ssl", "random_vit")
READ_SEEDS = (0, 1, 2)
FIRST_CELL, SECOND_CELL, LAST_CELL = ("frozen_ssl", 0), ("frozen_ssl", 1), ("random_vit", 2)
"""`sorted()` order of the nine cells -- the order `require_one_protocol` walks.
The LAST sorts after every other cell, so a check that looks at only the first
record or the first two sees nothing wrong with it."""


def _margin_triple(kind: str, c: int) -> tuple[float, float, float]:
    """`(point, ci_low, ci_high)`, binary fractions, a different one in every cell."""
    if kind == "up":
        point = 0.5 + c / 16.0
        return point, 0.125 + c / 32.0, point + 0.25
    if kind == "down":
        point = -0.5 - c / 16.0
        return point, point - 0.25, -0.125 - c / 32.0
    point = (c - 4) / 32.0
    return point, point - 0.25, point + 0.25


def _spec(kinds_by_arm) -> dict:
    """The stored margin each cell carries, from nothing but the cell's index."""
    spec = {}
    for a, (arm, kinds) in enumerate(kinds_by_arm.items()):
        for seed, kind in zip(READ_SEEDS, kinds, strict=True):
            c = a * len(READ_SEEDS) + seed
            point, low, high = _margin_triple(kind, c)
            spec[(arm, seed)] = {"c": c, "point": point, "low": low, "high": high}
    return spec


def _records_from(base, spec, *, decision_h=GATE_H) -> dict:
    """The record the measure phase wrote, relabelled per cell, WITH the `margin`
    that phase no longer writes -- which is what the nine on disk carry."""
    records = {}
    for (arm, seed), cell in spec.items():
        record = copy.deepcopy(base)
        record.update(arm=arm, seed=seed, decision_h=decision_h)
        record["margin"] = {
            str(h): (
                {"point": cell["point"], "ci_low": cell["low"], "ci_high": cell["high"]}
                if h == decision_h
                else {"point": 1000.0 + h, "ci_low": 999.0 + h, "ci_high": 1001.0 + h}
            )
            for h in REPORTED_HORIZONS
        }
        records[(arm, seed)] = record
    return records


def _read_args(tmp_path, records, *, arms=READ_ARMS, seeds=READ_SEEDS):
    """Write `records` under their file names -- spelled out here, not taken from
    `burden_record_path` -- and return the args `read_phase` takes."""
    out = Path(tmp_path) / "burden_out"
    out.mkdir(parents=True)
    for (arm, seed), record in records.items():
        write_record(out / f"burden_{arm}_seed{seed}.json", record)
    return types.SimpleNamespace(out=out, arms=list(arms), seeds=list(seeds))


def _named_cells(message: str, cells) -> set:
    return {(arm, seed) for arm, seed in cells if f"{arm} seed {seed}" in message}


MIXED = {
    # what each arm's three stored margins look like at the decision horizon:
    # whole interval above 0, whole interval at or below 0, or straddling it. The
    # signs differ across cells so that a column printed with the wrong format
    # (`+` dropped, a minus lost) is a different string, and `_margin_triple`
    # gives every cell its own value.
    "pixel_ae": ("down", "down", "straddle"),
    "frozen_ssl": ("up", "up", "straddle"),
    "random_vit": ("down", "down", "down"),
}
STORED_COLUMNS = ("arm", "seed", "margin", "ci_low", "ci_high")
STORED_WIDTHS = (13, 6, 11, 11, 11)


def _stored_row(values) -> str:
    """One printed row, spelled out here from the spec: right-aligned, unseparated,
    at the widths the header fixes -- not taken from the script's own constants."""
    return "".join(f"{str(v):>{w}}" for v, w in zip(values, STORED_WIDTHS, strict=True))


def _expected_stored_table(spec) -> list[str]:
    """Header and one row per cell in `sorted()` order, from the spec alone."""
    rows = [_stored_row(STORED_COLUMNS)]
    for (arm, seed), cell in sorted(spec.items()):
        rows.append(_stored_row((
            arm, seed, f"{cell['point']:+.4f}", f"{cell['low']:+.4f}", f"{cell['high']:+.4f}",
        )))
    return rows


def _read_refusal(args, capsys) -> tuple[str, str]:
    """`read_phase` over `args`: what it printed and what it was refused with."""
    with pytest.raises(SystemExit) as caught:
        script.read_phase(args)
    return capsys.readouterr().out, str(caught.value)


# --- records changed in the one way a refusal names ---------------------------


def _overwrite_record(args, cell, new) -> None:
    """Replace one cell's file in place -- under its literal name, as `_read_args`
    wrote it."""
    write_record(args.out / f"burden_{cell[0]}_seed{cell[1]}.json", new)


def _make_protocol_disagree(args, record) -> None:
    odd = _records_from(record, _spec(MIXED))[LAST_CELL]
    DISAGREEMENTS["git_sha"](odd)
    _overwrite_record(args, LAST_CELL, odd)


def _make_cell_missing(args, record) -> None:
    (args.out / f"burden_{LAST_CELL[0]}_seed{LAST_CELL[1]}.json").unlink()


# --- require_one_protocol ----------------------------------------------------


DISAGREEMENTS = {
    "git_sha": lambda r: r.__setitem__("git_sha", "0" * 40),
    "torch_version": lambda r: r.__setitem__("torch_version", r["torch_version"] + "+other"),
    "device": lambda r: r.__setitem__("device", "mps"),
    "step": lambda r: r.__setitem__("step", r["step"] + 1),
    "context": lambda r: r.__setitem__("context", r["context"] + 1),
    "horizon": lambda r: r.__setitem__("horizon", r["horizon"] + 1),
    "split_seed": lambda r: r.__setitem__("split_seed", r["split_seed"] + 1),
    "ks": lambda r: r.__setitem__("ks", r["ks"][:-1] + [r["ks"][-1] + 1]),
    "decision_h": lambda r: r.__setitem__("decision_h", r["decision_h"] - 1),
    "reported_h": lambda r: r.__setitem__("reported_h", r["reported_h"][:-1]),
    "identity_tolerance": lambda r: r.__setitem__("identity_tolerance", r["identity_tolerance"] * 10),
    "episodes.val": lambda r: r["episodes"].__setitem__("val", r["episodes"]["val"][::-1]),
    "windows.episode": lambda r: r["windows"]["episode"].__setitem__(
        -1, r["windows"]["episode"][-1] + 1
    ),
}
"""One way for a record to disagree with the others, per protocol field."""

REQUIRED_PROTOCOL_FIELDS = {
    "git_sha", "step", "context", "horizon", "ks", "decision_h", "reported_h",
    "identity_tolerance", "split_seed", "device",
}


def test_the_protocol_table_names_every_required_field_and_each_has_a_disagreement_case():
    """The table is a module-level tuple the function ITERATES and this test
    READS, so a field cannot be added to the comparison without this noticing it
    has no disagreement case below, and one cannot be dropped without the
    required set noticing. `git_sha` is in the required set because an earlier
    version of M3l's check omitted it and let records from different torch
    builds and code versions pool into one finding with no refusal at all."""
    names = [name for name, _ in script._PROTOCOL_FIELDS]
    assert len(names) == len(set(names))
    assert REQUIRED_PROTOCOL_FIELDS <= set(names)
    assert set(names) == set(DISAGREEMENTS)


@pytest.mark.parametrize("victim", [FIRST_CELL, SECOND_CELL, LAST_CELL], ids=["first", "second", "last"])
@pytest.mark.parametrize("field", sorted(DISAGREEMENTS))
def test_one_record_that_disagrees_on_any_protocol_field_is_refused_naming_it(
    record, field, victim,
):
    """Every field in the table, with the odd record at the front, second and
    last of the sorted nine. Last is the one a first-record-only check misses;
    second is the one a loop that starts one record too late misses. The message
    names the field AND the odd cell, and it names exactly two cells -- a message
    listing all nine would satisfy `victim in message` for any victim at all."""
    records = _records_from(record, _spec(MIXED))
    DISAGREEMENTS[field](records[victim])
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert f"disagree on {field}:" in message, message
    named = _named_cells(message, records)
    assert victim in named and len(named) == 2, (named, message)


def _long_episode_pool(record, odd) -> dict:
    """The nine records, every one describing 150 windows labelled `0..149`, and the
    LAST cell's labels then changed as `odd` says (`{index: label}`). 150 labels
    is far past what one printed line holds."""
    records = _records_from(record, _spec(MIXED))
    for cell in records.values():
        cell["windows"]["episode"] = list(range(150))
    for index, label in odd.items():
        records[LAST_CELL]["windows"]["episode"][index] = label
    return records


@pytest.mark.parametrize(
    "odd, said",
    [
        # The review's case: two 150-element lists that differ only in the LAST one.
        # Truncated to a line they print identically, and `(150 vs 150 rows)` would
        # say nothing either: the count is the same.
        pytest.param({149: 999}, "150 rows, first differing at index 149: 999 vs 149", id="last only"),
        # Two differences: the FIRST is the one named, not the last.
        pytest.param(
            {100: 777, 149: 999}, "150 rows, first differing at index 100: 777 vs 100",
            id="two differences",
        ),
    ],
)
def test_a_long_disagreement_names_the_first_index_where_the_values_differ(record, odd, said):
    """A refusal a person cannot act on is barely one. Both values are 150 labels;
    cut to a line they are the same text, so the message must say WHERE.

    THE MUTATIONS THIS EXISTS FOR: truncating each value to a line, which is what
    this printed before (`last only` fails -- the two truncations are
    character-for-character equal); naming the last differing index instead of the
    first (`two differences` fails); and printing the same index whatever the
    values are. The expected text is spelled out here, from the pool built above,
    not taken from the message."""
    records = _long_episode_pool(record, odd)
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert "random_vit seed 2 and frozen_ssl seed 0 disagree on windows.episode: " in message, message
    assert said in message, message


def test_a_long_disagreement_in_length_prints_both_lengths(record):
    """Two lists of different length, 151 labels against 150, that agree on every
    label both hold: no index among the labels they share is different, so the
    lengths are what the message must carry.

    THE MUTATION THIS EXISTS FOR: dropping the length branch, which leaves the two
    lists identical for as long as the shorter one runs, so that looking for a
    first differing index finds none and the fallback prints two truncations that
    read the same. `mine` is the later cell's, `reference` the first's."""
    records = _long_episode_pool(record, {})
    records[LAST_CELL]["windows"]["episode"].append(149)
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert "random_vit seed 2 and frozen_ssl seed 0 disagree on windows.episode: " in message, message
    assert "151 vs 150 rows" in message, message


def test_a_short_disagreement_is_still_printed_in_full(record):
    """A list that fits on a line is printed as it is: a ladder `[1, 3, 7, 15, 45]`
    against `[1, 3, 7, 15, 46]` is readable as two lists, and an index would be
    one more thing to look up.

    THE MUTATION THIS EXISTS FOR: taking the index form for every sequence."""
    records = _records_from(record, _spec(MIXED))
    reference = [int(k) for k in records[FIRST_CELL]["ks"]]
    odd = reference[:-1] + [reference[-1] + 1]
    records[LAST_CELL]["ks"] = odd
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    assert f"disagree on ks: {odd!r} vs {reference!r};" in str(caught.value)


def test_a_pool_that_agrees_on_the_protocol_is_accepted_whatever_else_differs(record):
    """Nothing but the table is compared. The per-cell numbers, the cell's own
    training provenance and its label all differ between cells by construction;
    comparing any of them would refuse the real nine."""
    records = _records_from(record, _spec(MIXED))
    for index, cell in enumerate(sorted(records)):
        records[cell]["record_git_sha"] = f"trained-at-{index}"
        records[cell]["kl_dyn_max"] = 1.0 + index
        records[cell]["kl_rate_above_free_bits"] = 0.5 + index / 32.0
    assert script.require_one_protocol(records) is None


@pytest.mark.parametrize("lacking", ["first", "last", "all"])
def test_a_record_without_a_protocol_field_is_refused_by_name_not_a_key_error(record, lacking):
    """A record that lacks a protocol field is refused BY NAME, not with a bare
    `KeyError` and not by defaulting the field.

    THE MUTATION THIS EXISTS FOR: reading a missing field as `None`. All three
    cases fail under it, in two different ways. With only one record lacking the
    field the default is still refused -- as a DISAGREEMENT with the other eight,
    naming `git_sha` and `None` -- and what tells that from the refusal by name is
    the message assertion (`lacks git_sha`), not the raise. `all` is the case where
    the default raises nothing at all: every record lacks `git_sha`, the nine agree
    on `None` and pool with no refusal."""
    records = _records_from(record, _spec(MIXED))
    victims = {"first": [FIRST_CELL], "last": [LAST_CELL], "all": sorted(records)}[lacking]
    for cell in victims:
        del records[cell]["git_sha"]
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert f"{victims[0][0]} seed {victims[0][1]} lacks git_sha" in message, message


@pytest.mark.parametrize("field, bad", [
    ("step", "not-a-number"),
    ("ks", ["1", "x"]),
])
def test_a_protocol_value_that_cannot_be_read_is_refused_by_name_not_a_value_error(
    record, field, bad,
):
    """A record whose protocol field is PRESENT but unreadable -- `int("not-a-number")`
    raises `ValueError`, not `KeyError` -- is refused by name, as one that lacks the
    field is. `_pick`'s docstring promises "a bare `KeyError` would not say which",
    and a bare `ValueError` says it no better; `scripts/motion_headroom.py`'s `_pick`
    catches it, and this one is held to the same contract.

    THE MUTATION THIS EXISTS FOR: narrowing `_pick`'s `except` back to
    `(KeyError, TypeError)`. The victim sorts last and the message is asserted, so
    a refusal that names no cell and no field does not pass."""
    records = _records_from(record, _spec(MIXED))
    records[LAST_CELL][field] = bad
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert f"random_vit seed 2 lacks {field}" in message, message


def test_an_empty_pool_is_refused_by_name():
    for call in (script.require_one_protocol, script.format_superseded_margin):
        with pytest.raises(SystemExit, match="no burden record"):
            call({})


def test_read_phase_refuses_records_that_disagree_on_the_protocol(record, tmp_path, capsys):
    """THE WIRING, not the function: `require_one_protocol` is tested above in
    isolation, and deleting its one call from `read_phase` leaves every one of
    those green. `git_sha` is the field the check once omitted: the records
    below differ in nothing else, so no other comparison -- the formatter's own
    agreement check is on `decision_h` alone -- can refuse them in its place.

    THE MUTATIONS THIS EXISTS FOR: deleting the `require_one_protocol(records)`
    call from `read_phase`, and dropping `git_sha` from `_PROTOCOL_FIELDS`. The
    victim sorts LAST, so a first-record-only check sees nothing; and nothing is
    printed first, because a pool that is not one measurement must not be shown
    as one table."""
    records = _records_from(record, _spec(MIXED))
    victim = sorted(records)[-1]
    assert victim == LAST_CELL
    records[victim]["git_sha"] = "0" * 40
    args = _read_args(tmp_path, records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(args)
    message = str(caught.value)
    assert "random_vit seed 2" in message and "git_sha" in message, message
    assert not (args.out / "burden.txt").exists()
    assert capsys.readouterr().out == ""


def test_the_formatter_refuses_a_pool_that_disagrees_on_the_decision_horizon(record):
    """One table holds ONE decision horizon for nine cells, so the pick is made where
    it is taken and does not depend on `require_one_protocol` having run first: a
    first-record pick would read every other cell at a horizon its own record does
    not name.

    THE MUTATION THIS EXISTS FOR: `int(items[0][1]["decision_h"])` without the
    agreement check in front of it. The odd record sorts last."""
    records = _records_from(record, _spec(MIXED))
    DISAGREEMENTS["decision_h"](records[LAST_CELL])
    with pytest.raises(SystemExit) as caught:
        script.format_superseded_margin(records)
    message = str(caught.value)
    assert "disagree on decision_h:" in message and "random_vit seed 2" in message, message


# --- the plan, and loading ---------------------------------------------------


def test_read_phase_names_the_first_missing_cell_and_exits_eleven(record, tmp_path, capsys):
    """Two cells are missing; the one the PLAN reaches first is the one named,
    and nothing is written."""
    records = _records_from(record, _spec(MIXED))
    del records[SECOND_CELL], records[LAST_CELL]
    args = _read_args(tmp_path, records)
    assert script.read_phase(args) == script.EXIT_NO_CHECKPOINTS
    out = capsys.readouterr().out
    assert "NO CELL" in out and "frozen_ssl seed 1" in out and "random_vit seed 2" not in out, out
    assert "--phase measure" in out
    assert not (args.out / "burden.txt").exists()
    assert "SUPERSEDED" not in out, "a pool with a hole is not printed as a table"


def test_only_the_planned_cells_are_read_and_printed(record, tmp_path, capsys):
    """The plan names the cells, and the records on disk outside it are not pooled:
    `--arms pixel_ae` is a legitimate thing to want from an old record, and the third
    arm's files are there and must not appear. No arm or seed minimum applies --
    those belonged to a verdict.

    THE MUTATION THIS EXISTS FOR: pooling whatever `burden_*.json` the directory
    holds, which prints nine rows for a plan of three."""
    spec = _spec(MIXED)
    args = _read_args(tmp_path, _records_from(record, spec), arms=("pixel_ae",))
    out, _ = _read_refusal(args, capsys)
    planned = {cell: spec[cell] for cell in spec if cell[0] == "pixel_ae"}
    assert len(planned) == 3
    lines = out.splitlines()
    at = lines.index(_expected_stored_table(planned)[0])
    assert lines[at : at + 4] == _expected_stored_table(planned)
    assert "frozen_ssl" not in out and "random_vit" not in out


@pytest.mark.parametrize(
    "swap, first_file, record_says, other_file",
    [
        # Only the SEED differs: both files are `pixel_ae`. `load_burden` walks the
        # plan arm by arm and seed by seed, so `seed1` is the first file it opens
        # whose record disagrees; `seed2`, the other half of the swap, comes later.
        pytest.param(
            (("pixel_ae", 1), ("pixel_ae", 2)),
            "burden_pixel_ae_seed1.json was read for pixel_ae seed 1",
            "arm='pixel_ae' seed=2",
            "burden_pixel_ae_seed2.json",
            id="seed half",
        ),
        # Only the ARM differs: both records say seed 0. `pixel_ae` is first in the
        # plan, so `burden_pixel_ae_seed0.json` -- which holds a `frozen_ssl` record
        # -- is opened before `burden_frozen_ssl_seed0.json` (which holds pixel_ae's).
        pytest.param(
            (("pixel_ae", 0), ("frozen_ssl", 0)),
            "burden_pixel_ae_seed0.json was read for pixel_ae seed 0",
            "arm='frozen_ssl' seed=0",
            "burden_frozen_ssl_seed0.json",
            id="arm half",
        ),
    ],
)
def test_a_record_filed_under_another_cells_name_is_refused(
    record, tmp_path, swap, first_file, record_says, other_file,
):
    """A swapped pair of files would pool one cell under another's name with every
    count still right.

    THE MUTATIONS THIS EXISTS FOR, one per case: dropping the `arm` comparison from
    `load_burden`'s filename check (the `arm half` case fails, and only it) and
    dropping the `seed` comparison (the `seed half` case fails, and only it). Each
    swap changes ONE of the two, so a check that tests the other cannot see it; the
    seed swap alone, which this test was first written as, left the arm half
    unpinned.

    The message must name the FIRST file `load_burden` meets that disagrees --
    `first_file` is spelled out here, in the plan's order, not read off the code --
    and must not name the other file of the swap: asserting `a or b` would accept
    either, and `"seed" in message` is true of any message that mentions a seed."""
    records = _records_from(record, _spec(MIXED))
    one, other = swap
    records[one], records[other] = records[other], records[one]
    args = _read_args(tmp_path, records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(args)
    message = str(caught.value)
    assert message.startswith(first_file), message
    assert f"but its record says {record_says}" in message, message
    assert other_file not in message, message


# --- what is written, and what is printed -----------------------------------


# --- main: the three phases --------------------------------------------------


def _main(monkeypatch, argv, *, measure=0, read=0):
    """`main(argv)` with BOTH phases replaced by recorders, returning
    `(status, phases_run)`. What this pins is the WIRING: which phase the flag runs,
    and whose status comes back."""
    ran = []

    def fake_measure(args):
        ran.append("measure")
        return measure

    def fake_read(args):
        ran.append("read")
        return read

    monkeypatch.setattr(script, "measure_phase", fake_measure)
    monkeypatch.setattr(script, "read_phase", fake_read)
    return script.main(argv), ran


def test_main_runs_the_phase_the_flag_names_and_returns_that_phases_status(monkeypatch):
    """Each phase alone, and each phase's own status: `read` answers 11 here so that
    a status taken from the other phase is a different number."""
    assert _main(monkeypatch, ["--phase", "measure"]) == (0, ["measure"])
    assert _main(monkeypatch, ["--phase", "read"]) == (0, ["read"])
    assert _main(monkeypatch, ["--phase", "read"], read=11) == (11, ["read"])
    assert _main(monkeypatch, ["--phase", "measure"], measure=14, read=0) == (14, ["measure"])


@pytest.mark.parametrize("status", [11, 12, 14, 30])
def test_a_failed_measure_is_the_exit_status_of_phase_measure_too(monkeypatch, status):
    """`--phase measure` is the only phase that pays for anything, and a refused cell
    is exactly where its status could be dropped on the floor: a `main` that returned
    `EXIT_OK` after it would print a success after a refused 30-minute run.

    THE MUTATION THIS EXISTS FOR: `return measure_phase(args)` -> `measure_phase(args)`
    followed by `return EXIT_OK`. The statuses are the literals `measure_phase` can
    return (11, 12, 14 from the shared cell checks, 30 from the self-check), not
    `script.EXIT_*`, and `read` is made to answer 0 so a status that was read from
    the wrong phase is a different number."""
    assert _main(monkeypatch, ["--phase", "measure"], measure=status, read=0) == (
        status, ["measure"],
    )


def test_the_default_phase_is_read_and_the_parser_names_the_three():
    """A default of `measure` would make a bare invocation pay for nine cells of
    GPU time; `read` costs none, prints the superseded margin of the records there
    are, and refuses by name when there are none."""
    args = script._parser().parse_args([])
    assert args.phase == "read"
    assert script.PHASES == ("all", "measure", "read")
    assert args.out == Path("runs/m3m_burden")
    with pytest.raises(SystemExit) as caught:
        script._parser().parse_args(["--phase", "readd"])
    assert caught.value.code == 2


def test_main_reads_the_arms_seeds_and_directory_the_command_line_names(
    monkeypatch, record, tmp_path, capsys,
):
    """The flag names reach the fields `read_phase` reads, through the real read
    phase: two of the three arms are asked for and only they are printed, and a
    second directory with no flag but `--out` prints all three (the default plan).
    The measure phase is made unreachable: a `--phase read` that fell into it would
    load the real checkpoints from `--source`'s default."""
    def forbidden(args):
        raise AssertionError("--phase read reached the measure phase")

    monkeypatch.setattr(script, "measure_phase", forbidden)
    args = _read_args(tmp_path, _records_from(record, _spec(MIXED)))
    argv = ["--phase", "read", "--out", str(args.out), "--arms", "pixel_ae", "frozen_ssl",
            "--seeds", "0", "1", "2"]
    with pytest.raises(SystemExit) as caught:
        script.main(argv)
    assert "scripts/motion_headroom.py" in str(caught.value)
    out = capsys.readouterr().out
    assert "pixel_ae" in out and "frozen_ssl" in out and "random_vit" not in out

    everything = _read_args(tmp_path / "all", _records_from(record, _spec(MIXED)))
    with pytest.raises(SystemExit):
        script.main(["--phase", "read", "--out", str(everything.out)])
    out = capsys.readouterr().out
    assert all(arm in out for arm in READ_ARMS)


# ---------------------------------------------------------------------------
# M3n retired the motion statistic: no new margin, and a read that refuses.
# ---------------------------------------------------------------------------


def test_the_record_carries_no_motion_margin_or_the_interval_it_was_drawn_with(record, tmp_path):
    """New records must not carry a `margin`. Old records keep theirs and stay
    readable; the point is to stop producing new contaminated numbers.

    ASKED OF THE WRITTEN RECORD'S KEYS, not of the source text of `measure_cell`:
    a source search for "margin" fails on any comment that uses the word, and says
    nothing about what the function returns. And top-level keys only -- the
    rulers are named `floor_margin_standard_error`, the margin over the FLOOR,
    which is `burden(1)`'s ruler and survives.

    `window_margin` is the per-window array the intervals were drawn from, and
    `confidence` / `resamples` describe an estimator that no longer runs: a record
    that names a level and a draw count for an interval it does not hold is worse
    than one that names neither.

    THE MUTATIONS THIS EXISTS FOR: restoring the `margin` loop (which carries the
    other three with it), and restoring any one of the four keys alone.

    The fixture is a REAL record, shown by the ladder keys it carries: a record
    that was empty would pass every absence below.
    """
    for kept in ("curves", "burden_by_k", "compounding_by_k", "controls", "base_control", "rulers"):
        assert kept in record, kept
    for gone in ("margin", "window_margin", "confidence", "resamples"):
        assert gone not in record, gone
    written = script.write_record(tmp_path / "burden.json", record)
    assert "margin" not in load_record(tmp_path / "burden.json")
    assert "margin" not in written


def test_the_script_no_longer_holds_what_only_reading_h_used():
    """The pieces of the verdict that lived HERE go with it: the table that built
    `BurdenInputs`, the exit map from its two refusal statuses, and the two plan
    checks that existed so a verdict could be taken.

    `EXIT_CONTROL_BROKEN` and `EXIT_UNREADABLE` stay as constants (45 and 46):
    they are in the exit-code registry, and a number M3m reported must not be
    reissued to another failure.
    """
    for gone in (
        "burden_inputs", "READ_EXITS", "require_readable_plan",
        "require_seeds_for_a_verdict",
    ):
        assert not hasattr(script, gone), f"{gone} must be removed"
    assert (script.EXIT_CONTROL_BROKEN, script.EXIT_UNREADABLE) == (45, 46)


def test_the_read_phase_prints_the_stored_margin_of_every_cell_and_then_refuses(
    record, tmp_path, capsys,
):
    """Reading H is retired. The records stay readable, so the margin each one
    STORED is printed -- every cell, at the decision horizon, in `sorted()` order
    -- and then the read refuses by name, `SystemExit` and status 1.

    THE FIXTURE DISCRIMINATES: nine cells, each with its own point and bounds, and
    a decoy of about a thousand at every other horizon, so a margin read at the
    wrong horizon, from another cell, or from another key is a different number.
    Distinct values are asserted rather than assumed.

    THE EXPECTED TABLE IS BUILT FROM THE SPEC, not by the script.
    """
    spec = _spec(MIXED)
    assert len({(c["point"], c["low"], c["high"]) for c in spec.values()}) == 9
    args = _read_args(tmp_path, _records_from(record, spec))
    out, message = _read_refusal(args, capsys)
    lines = out.splitlines()
    at = lines.index(_expected_stored_table(spec)[0])
    assert lines[at : at + 10] == _expected_stored_table(spec)
    assert "motion_headroom" in message


def test_the_stored_margin_is_printed_under_a_legend_that_says_it_is_superseded_and_why(
    record, tmp_path, capsys,
):
    """A reader of an old record is not left to rediscover the contamination.

    `motion_margin` subtracted the k=1 rung, read through the probe, from the true
    one-step displacement, a ground-truth quantity: on M3m's nine cells the median
    floor error alone was 88.29-224.36 against a displacement of 3.97, so the
    readout dominated the difference 22.2x-56.5x. A negative stored margin is that
    error, and the legend must NOT tell the reader it means the model copies --
    the sentence that made this hazard.

    The figures are M3m's own, quoted from its plan; they are a claim about those
    nine cells and the legend says so. The test pins the substance of each of the
    four things the legend has to say, not its wording.
    """
    args = _read_args(tmp_path, _records_from(record, _spec(MIXED)))
    out, _ = _read_refusal(args, capsys)
    assert out.startswith("--- SUPERSEDED"), out
    legend = out[out.index(_expected_stored_table(_spec(MIXED))[-1]):]
    lowered = legend.lower()
    assert "superseded" in lowered
    assert "22.2x-56.5x" in legend
    assert "88.29-224.36" in legend and "3.97" in legend
    assert "readout error" in lowered
    assert "does not mean the one-step map copies" in lowered
    assert "scripts/motion_headroom.py" in legend
    # A refusal is not a reading: no status, no verdict line, nothing that claims
    # one of the old four.
    for claim in ("verdict:", "PREDICTS", "COPIES", "INDETERMINATE", "UNREADABLE"):
        assert claim not in out, claim


def test_stdout_is_the_formatters_text_with_one_trailing_newline_and_reproducible(
    record, tmp_path, capsys,
):
    """The table's CONTENT is pinned from the spec above; this pins the WIRING: what
    is printed is the formatter's text, byte for byte, which ends in exactly one
    newline, and the same on a second read of the same records. Two reads in two
    directories, so a read that depended on anything but the records is a different
    string.

    THE MUTATIONS THIS EXISTS FOR: `print(text)` on a string that already ends in a
    newline, which doubles it, and `rstrip()` on the text, which loses it. Both are
    invisible to a test that compares the table's LINES, and to one that compares two
    reads of the same code (each strips the same bytes). The precondition is asserted:
    the formatter's text ends in exactly one newline."""
    records = _records_from(record, _spec(MIXED))
    expected = script.format_superseded_margin(records)
    assert expected.endswith("\n") and not expected.endswith("\n\n")
    first = _read_args(tmp_path / "a", records)
    second = _read_args(tmp_path / "b", records)
    out_first, _ = _read_refusal(first, capsys)
    out_second, _ = _read_refusal(second, capsys)
    assert out_first == out_second == expected


def test_the_refusal_names_the_replacement_and_says_no_reading_was_taken(
    record, tmp_path, capsys,
):
    """The message the operator is refused with is what the exit carries, so it
    must name the script that replaced the reading, and say that nothing was
    taken or written."""
    args = _read_args(tmp_path, _records_from(record, _spec(MIXED)))
    _, message = _read_refusal(args, capsys)
    assert "scripts/motion_headroom.py" in message
    assert "Reading H is retired" in message
    assert "no burden.txt" in message


def _make_no_change(args, record) -> None:
    """The retirement itself: the records are fine, and the read refuses anyway."""


UNTOUCHED_AFTER = {
    # name: (what to change first, what the read does -- an exit number, or SystemExit)
    "the retirement": (_make_no_change, SystemExit),
    "a missing cell": (_make_cell_missing, script.EXIT_NO_CHECKPOINTS),
    "a protocol disagreement": (_make_protocol_disagree, SystemExit),
}


@pytest.mark.parametrize("exit_way", sorted(UNTOUCHED_AFTER))
def test_the_retired_read_writes_nothing_and_removes_nothing(
    record, tmp_path, capsys, monkeypatch, exit_way,
):
    """The records directory is as shared as `runs/` is: a read that unlinked a
    `burden.txt` an EARLIER read left would delete an artefact the M3m spec may
    cite, to protect a reader from a reading this script can no longer take.

    THE MUTATIONS THIS EXISTS FOR: writing `burden.txt` (a refusal is not a
    reading), and the old `unlink(missing_ok=True)` at the top of `read_phase`, which
    every way out of the function used to pass through. Asked of all three ways out
    -- the retirement, a missing cell, a protocol disagreement -- so an unlink moved
    to any one of them is seen. The directory must hold, byte for byte, exactly what
    it held, and three bystanders -- a `burden.txt` beside `--out` and one in the
    current directory, and a `.txt` that is not `burden.txt` inside it -- are asked
    about as well, which an over-wide unlink would find.

    The one change a case makes is a record REMOVED or REWRITTEN, so `before` is
    taken after it."""
    change, outcome = UNTOUCHED_AFTER[exit_way]
    args = _read_args(tmp_path, _records_from(record, _spec(MIXED)))
    earlier = args.out / "burden.txt"
    earlier.write_text("a reading an earlier read left\n")
    (args.out / "notes.txt").write_text("not a reading\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    bystanders = [tmp_path / "burden.txt", cwd / "burden.txt"]
    for bystander in bystanders:
        bystander.write_text("not this one")

    change(args, record)
    before = {p.name: p.read_bytes() for p in args.out.iterdir()}
    assert "burden.txt" in before
    if isinstance(outcome, int):
        assert script.read_phase(args) == outcome
    else:
        with pytest.raises(outcome):
            script.read_phase(args)
    capsys.readouterr()

    assert {p.name: p.read_bytes() for p in args.out.iterdir()} == before
    for bystander in bystanders:
        assert bystander.read_text() == "not this one", bystander


def test_a_directory_with_no_earlier_reading_is_left_without_one(record, tmp_path, capsys):
    """The other half: the retirement is not a reading, so it must not CREATE a
    `burden.txt` where there was none."""
    args = _read_args(tmp_path, _records_from(record, _spec(MIXED)))
    before = sorted(p.name for p in args.out.iterdir())
    _read_refusal(args, capsys)
    assert sorted(p.name for p in args.out.iterdir()) == before
    assert "burden.txt" not in before


def test_the_margin_is_read_at_the_decision_horizon_the_records_name(record, tmp_path, capsys):
    """The decision horizon is the RECORDS', and at 30 the step is not the last
    one: a read at 45, at the first horizon, or at the largest key is another
    number. Every other horizon carries a decoy near a thousand."""
    spec = _spec(MIXED)
    records = _records_from(record, spec, decision_h=30)
    args = _read_args(tmp_path, records)
    out, _ = _read_refusal(args, capsys)
    assert "horizon 30" in out.splitlines()[0]
    lines = out.splitlines()
    assert lines[1 : 1 + 10] == _expected_stored_table(spec)


def test_a_record_written_after_the_retirement_is_printed_as_carrying_no_margin(
    record, tmp_path, capsys,
):
    """The measure phase no longer writes a `margin`, so a record it wrote has
    none to print. The read does not raise a bare `KeyError`: the cell is listed,
    its three columns say so, and the legend names what the dash means. Cells that
    DO carry one are unaffected -- only one of the nine is stripped.

    THE MUTATION THIS EXISTS FOR: `record["margin"]` where `record.get("margin")`
    stood, which crashes on every record the retired measure phase writes."""
    spec = _spec(MIXED)
    records = _records_from(record, spec)
    stripped = sorted(records)[4]
    del records[stripped]["margin"]
    args = _read_args(tmp_path, records)
    out, _ = _read_refusal(args, capsys)
    table = _expected_stored_table(spec)
    table[1 + 4] = _stored_row((stripped[0], stripped[1], "-", "-", "-"))
    lines = out.splitlines()
    at = lines.index(table[0])
    assert lines[at : at + 10] == table
    assert "no margin" in out.lower()

    # ... and a pool where every record has one does not print that line.
    full = _read_args(tmp_path / "full", _records_from(record, spec))
    out, _ = _read_refusal(full, capsys)
    assert "no margin" not in out.lower()


def test_phase_all_is_refused_by_name_before_anything_is_measured(monkeypatch):
    """`--phase all` was the measure and then the read. The read is retired, so
    `all` would pay for every cell and then refuse. It is refused instead, BY NAME
    and BEFORE the first cell, pointing at the two things that still work.

    `ran == []` is what says it came before the measure and not after it. The
    other two phases are unaffected."""
    ran = []
    monkeypatch.setattr(script, "measure_phase", lambda args: ran.append("measure") or 0)
    monkeypatch.setattr(script, "read_phase", lambda args: ran.append("read") or 0)
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "all"])
    message = str(caught.value)
    assert "scripts/motion_headroom.py" in message and "--phase measure" in message, message
    assert ran == []
    assert _main(monkeypatch, ["--phase", "measure"]) == (0, ["measure"])
    assert _main(monkeypatch, ["--phase", "read"]) == (0, ["read"])
