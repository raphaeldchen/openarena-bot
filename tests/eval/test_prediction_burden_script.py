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
from mbfps.eval import burden
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
    definition, and the k=1 rung it is given here is `_expected_k_one_rungs`'s
    closed form, NOT a call of the sweep. If the two disagree the stacked path is
    wrong.

    Not a second sweep: `regrounding_sweep` is called once inside `measure_cell`
    and once by a test, so a row order the sweep itself got wrong -- `sorted()`
    inside it, two episodes swapped -- would be the same wrong order on both sides
    and this comparison would pass. Windows the closed form takes in the given
    order are what make it a reference.

    THE MUTATION THIS EXISTS FOR: `one - rows` instead of `rows - one` in
    `measure_cell`, which flips the sign of the quantity Reading H's verdict is
    read from and leaves every shape and every interval intact. The wobble gives
    the margin both signs and a mean absolute value near 4 map units, so the
    flipped array is far from the right one in most windows.
    """
    one = _expected_k_one_rungs(rig)
    targets = _window_targets(rig.paths)
    margin = np.asarray(record["window_margin"], dtype=np.float64)
    assert margin.shape == one.shape == (len(targets), HORIZON)
    assert (margin > 0).any() and (margin < 0).any(), "the fixture must give both signs"
    for w, window in enumerate(targets):
        expected = burden.motion_margin(window, one[w])
        np.testing.assert_allclose(margin[w], expected, rtol=0, atol=1e-12, err_msg=f"window {w}")


def test_the_k_one_rung_is_each_windows_own_in_the_order_given(rig, record):
    """Row `w` of the sweep is the window row `w` of the baseline is cut from, by
    VALUE. The shape refusal sees only a count that drifted, and a reorder that
    keeps the shape -- two episodes with the same number of windows swapped, or
    `sorted()` on names that happen to give the same block sizes -- passes it.

    The sweep's own k=1 rows, a second call of it, and the margin the record
    carries are each asked against `_expected_k_one_rungs`, which does not call
    the sweep. The record's `window_margin` is `rows - one`, so it must equal the
    baseline's rows minus the closed form; that pins `one` INSIDE `measure_cell`
    as well as inside the sweep.

    THE MUTATIONS THIS EXISTS FOR, none of which `test_the_stacked_margin...` saw
    while its expectation came from a second sweep: `val_paths = sorted(val_paths)`
    inside `regrounding_sweep`; two equal-count episodes (0 and 3, one window each)
    swapped inside it; and rows 0 and 3 of `one` swapped inside `measure_cell`.
    """
    expected = _expected_k_one_rungs(rig)
    assert expected.shape == (sum(EXPECTED_WINDOWS), HORIZON)
    # Reached: every window's rung is its own, so a swap or a reorder moves values.
    assert len({tuple(row) for row in expected}) == expected.shape[0]
    assert np.abs(expected[0] - expected[3]).max() > 0.5, "windows 0 and 3 are the swapped pair"

    one = _independent_sweep(rig).window_position[1]
    np.testing.assert_allclose(one, expected, rtol=1e-6, atol=1e-9)

    rows = np.stack([burden.one_step_persistence(w) for w in _window_targets(rig.paths)])
    np.testing.assert_allclose(
        np.asarray(record["window_margin"]), rows - expected, rtol=1e-6, atol=1e-9,
    )
    np.testing.assert_allclose(
        record["curves"]["rungs"]["1"], expected.mean(axis=0), rtol=1e-6, atol=1e-9,
    )


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
def test_measure_cell_refuses_a_protocol_the_reading_cannot_be_taken_from(
    rig, monkeypatch, over, match,
):
    """The status is read at `DECISION_H` and every margin is read off the k=1
    rung. Refused by name before any pass is paid for, rather than as an
    IndexError or a KeyError after the sweep.

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


def test_the_record_carries_every_protocol_parameter_the_reading_uses(record):
    """M3l shipped a headline interval whose DRAW COUNT was not on the record,
    so the permanent artefact could not be audited for it. Both the level and
    the count are recorded here from the start.

    THE MUTATION THIS EXISTS FOR: dropping any of these keys. It pins that each
    is PRESENT and AGREES WITH the shipped constant, and nothing more: a literal
    that equals the constant -- `"confidence": 0.95` -- passes it, which is the
    one thing it cannot see. WHERE EACH VALUE IS READ FROM is pinned elsewhere:

      `resamples`
          test_the_interval_is_taken_at_the_cells_seed_and_the_draw_count_the_record_states
      `confidence`, `decision_h`, `reported_h`, `identity_tolerance`, `split_seed`
          test_each_protocol_constant_is_recorded_from_the_modules_own_name
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


@pytest.mark.parametrize(
    "constant, patched, key, expected",
    [
        ("CONFIDENCE", 0.9, "confidence", 0.9),
        ("DECISION_H", 30, "decision_h", 30),
        ("REPORTED_H", (1, 7, 45), "reported_h", [1, 7, 45]),
        ("IDENTITY_TOLERANCE", 1e-3, "identity_tolerance", 1e-3),
        ("SPLIT_SEED", 7, "split_seed", 7),
    ],
)
def test_each_protocol_constant_is_recorded_from_the_modules_own_name(
    rig, monkeypatch, constant, patched, key, expected,
):
    """The record states the constant THE MODULE HOLDS, checked the way the
    `RESAMPLES` test checks its own: the module's name is patched to a value the
    shipped constant is not, and the record must follow it.

    Comparing a record to `burden.CONFIDENCE` cannot do this. A literal `0.95`
    EQUALS it, so recording one passes every test that compares against the
    shipped value; only a patched module attribute tells a literal from a read.
    The brief's mutation for `confidence` also changed `burden.CONFIDENCE`, which
    makes it a two-file mutation the single-constant docstring never described.

    THE MUTATIONS THIS EXISTS FOR, one per case, each a literal where the module's
    constant stood, each of which survived all 50 tests: `"confidence": 0.95`,
    `"decision_h": 45`, `"reported_h": [1, 2, 3, 5, 8, 10, 15, 20, 30, 45]`,
    `"identity_tolerance": 1e-9`, `"split_seed": 0`. (`resamples` is the sixth, and
    is pinned by its own patching test.) `DECISION_H` and `REPORTED_H` are also
    read by the measurement itself, so the patched values are ones the protocol
    guard still accepts at the cell's horizon of 45.
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
    is the only thing to see -- and the record carries it too, as `seed` and as
    the seed its bootstrap interval was drawn at. The interval is recomputed from
    the record's own margins at the cell's seed and must be the one written, and
    the two seeds of one arm, which share a model and so a point estimate, must
    not share an interval.

    THE MUTATIONS THIS EXISTS FOR, each of which the one-seed-0 fixture let
    through with every test green: `measure_phase` handing `measure_cell`
    `seed=0` instead of `cell.seed`, and `burden_record_path(args.out, cell.arm,
    0)` instead of `cell.seed`.

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
        assert carried["resamples"] == burden.RESAMPLES
        h = burden.DECISION_H
        entry = carried["margin"][str(h)]
        assert (entry["point"], entry["ci_low"], entry["ci_high"]) == burden.margin_interval(
            np.asarray(carried["window_margin"]), np.asarray(carried["windows"]["episode"]),
            h=h, resamples=burden.RESAMPLES, seed=seed,
        ), (arm, seed)
    for arm in ("pixel_ae", "frozen_ssl"):
        first, second = (records[(arm, seed)]["margin"][str(burden.DECISION_H)] for seed in PHASE_SEEDS)
        # Reached: same model, same windows, so the point is shared and ONLY the
        # seed can move the bounds. If they coincided the recomputation above
        # would pass for any seed at all.
        assert first["point"] == second["point"], arm
        assert (first["ci_low"], first["ci_high"]) != (second["ci_low"], second["ci_high"]), arm


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


def test_the_cell_line_prints_the_numbers_the_verdict_is_read_from(
    tmp_path, monkeypatch, capsys,
):
    args, _ = _phase_env(tmp_path, monkeypatch)
    script.measure_phase(args)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(CELLS)
    last = str(REGROUNDING_KS[-1])
    for line, (arm, seed) in zip(lines, CELLS):
        record = load_record(args.out / f"burden_{arm}_seed{seed}.json")
        margin = record["margin"][str(burden.DECISION_H)]
        assert line.startswith(f"{arm} seed {seed}:"), line
        for number in (margin["point"], margin["ci_low"], margin["ci_high"]):
            assert f"{number:+.4f}" in line
        # The burden printed is the LAST rung's, read at `DECISION_H`: the rung and
        # the step are each written out here rather than read back off the record.
        by_step = record["burden_by_k"][last]
        printed = f"burden(k={last}) {by_step[burden.DECISION_H - 1]:+.4f};"
        assert printed in line, line
        # Reached: step 1 and step `DECISION_H` read differently, so a line that
        # printed the first step fails; and on the drifting arm the first rung and
        # the last do too (on the exact oracle every rung is the same, so there the
        # rung is told apart only by the `k=` label above).
        assert f"{by_step[0]:+.4f}" != f"{by_step[burden.DECISION_H - 1]:+.4f}"
        if arm == "pixel_ae":
            first_rung = record["burden_by_k"]["1"][burden.DECISION_H - 1]
            assert f"{first_rung:+.4f}" != f"{by_step[burden.DECISION_H - 1]:+.4f}"
        assert f"{record['controls']['identity_residual']:.2e}" in line
        assert str(args.out / f"burden_{arm}_seed{seed}.json") in line


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
# The read phase: nine records pooled into Reading H.
#
# THE POOL IS THE REAL RECORD, RELABELLED. Every cell starts as a deep copy of
# the record `measure_cell` writes (`record`), so the schema is the writer's own,
# and only the fields the read phase READS are then overwritten, each from a
# formula in the cell's index `c`. Nothing the reading reads is shared between
# two cells, and the expected `BurdenInputs` is built from the same formulas --
# never from the records, and never through `script.burden_inputs`.
#
#   * Three arms x seeds 0, 1, 2: nine cells, no two alike in any number read.
#     The arms are given in `ARMS` order, whose index is NOT `sorted()` order
#     (`frozen_ssl` sorts first), so a read that sorts or enumerates differently
#     reads another cell's numbers.
#   * `burden_by_k[k][j]` and `compounding_by_k[k][j]` hold a value that depends
#     on the cell, the rung AND the step, so a read at the wrong step, the wrong
#     rung, the wrong cell, or from the other array is a different number.
#   * The margin at every reported horizon but the decision horizon is a decoy
#     (about a thousand), so a read at the wrong horizon is not a plausible one.
#   * The decision horizon is also read at 30 and 20, not only 45: at 45 the
#     step index is the LAST one, where `[h - 1]` and `[-1]` are one element.
# ---------------------------------------------------------------------------


READ_ARMS = ("pixel_ae", "frozen_ssl", "random_vit")
READ_SEEDS = (0, 1, 2)
FIRST_CELL, SECOND_CELL, LAST_CELL = ("frozen_ssl", 0), ("frozen_ssl", 1), ("random_vit", 2)
"""`sorted()` order of the nine cells -- the order `require_one_protocol` walks.
The LAST sorts after every other cell, so a check that looks at only the first
record or the first two sees nothing wrong with it."""

KINDS = {
    # what each arm's three seeds clear at the decision horizon, in seed order:
    # "up" = the whole interval above 0, "down" = at or below 0, "straddle" = neither
    "PREDICTS_MOTION": {
        "pixel_ae": ("up", "up", "up"),
        "frozen_ssl": ("up", "straddle", "up"),
        "random_vit": ("down", "straddle", "down"),
    },
    "COPIES": {
        "pixel_ae": ("down", "down", "straddle"),
        "frozen_ssl": ("up", "up", "straddle"),
        "random_vit": ("down", "down", "down"),
    },
    "INDETERMINATE": {
        "pixel_ae": ("up", "up", "straddle"),
        "frozen_ssl": ("down", "down", "up"),
        "random_vit": ("straddle", "straddle", "up"),
    },
}
BROKEN_CONTROLS = {"identity_residual": 1e-3, "open_loop_divergence": 0.25, "k_one_is_floor": True}


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


def _burden_value(c: int, k: int, h: int) -> float:
    return 100.0 * c + k + h / 64.0


def _compounding_value(c: int, k: int, h: int) -> float:
    return c / 4.0 + k / 8.0 + h / 256.0


def _spec(kinds_by_arm, *, broken=None, stationary=None) -> dict:
    """The numbers the reading reads, per cell, from nothing but the cell's index.

    `broken=(cell, control)` misses one known answer in one cell and
    `stationary=cell` makes one cell's agent move less than the readout's error."""
    spec = {}
    for a, (arm, kinds) in enumerate(kinds_by_arm.items()):
        for seed, kind in zip(READ_SEEDS, kinds, strict=True):
            c = a * len(READ_SEEDS) + seed
            floor = 3.0 + c / 16.0
            point, low, high = _margin_triple(kind, c)
            spec[(arm, seed)] = {
                "c": c, "point": point, "low": low, "high": high,
                "identity_residual": (c + 1) * 2.0 ** -50,
                "open_loop_divergence": 0.0, "k_one_is_floor": False,
                "displacement_median": floor + 5.0 + c / 8.0, "floor_median": floor,
            }
    if broken is not None:
        cell, control = broken
        spec[cell][control] = BROKEN_CONTROLS[control]
    if stationary is not None:
        spec[stationary]["displacement_median"] = spec[stationary]["floor_median"] - 0.5
    return spec


def _scenario(status: str, **over) -> dict:
    """A spec that `burden.reading_burden` reads as `status`. The two refusals
    sit on top of a pool that would otherwise read PREDICTS_MOTION, which is what
    makes them refusals and not readings."""
    if status == "UNRESOLVED_CONTROL":
        over.setdefault("broken", (LAST_CELL, "identity_residual"))
    elif status == "UNREADABLE":
        over.setdefault("stationary", LAST_CELL)
    kinds = KINDS["PREDICTS_MOTION" if status in ("UNRESOLVED_CONTROL", "UNREADABLE") else status]
    return _spec(kinds, **over)


def _records_from(base, spec, *, decision_h=burden.DECISION_H, ks=REGROUNDING_KS) -> dict:
    records = {}
    for (arm, seed), cell in spec.items():
        c = cell["c"]
        record = copy.deepcopy(base)
        record.update(arm=arm, seed=seed, decision_h=decision_h, ks=list(ks))
        record["margin"] = {
            str(h): (
                {"point": cell["point"], "ci_low": cell["low"], "ci_high": cell["high"]}
                if h == decision_h
                else {"point": 1000.0 + h, "ci_low": 999.0 + h, "ci_high": 1001.0 + h}
            )
            for h in burden.REPORTED_H
        }
        steps = range(1, HORIZON + 1)
        record["burden_by_k"] = {str(k): [_burden_value(c, k, h) for h in steps] for k in ks}
        record["compounding_by_k"] = {
            str(k): [_compounding_value(c, k, h) for h in steps] for k in ks
        }
        record["controls"] = {
            **record["controls"],
            "identity_residual": cell["identity_residual"],
            "open_loop_divergence": cell["open_loop_divergence"],
            "k_one_is_floor": cell["k_one_is_floor"],
        }
        record["base_control"] = {
            "displacement_median": cell["displacement_median"],
            "floor_median": cell["floor_median"],
        }
        records[(arm, seed)] = record
    return records


def _expected_inputs(spec, *, decision_h=burden.DECISION_H, ks=REGROUNDING_KS):
    """What `burden_inputs` must return, built from the spec alone. The window
    counts are the rig's own (`EXPECTED_WINDOWS`), not read back off a record."""
    return burden.BurdenInputs(
        cells={
            cell: burden.BurdenArm(
                arm=cell[0], seed=cell[1], margin=s["point"], margin_low=s["low"],
                margin_high=s["high"],
                burden_by_k={k: _burden_value(s["c"], k, decision_h) for k in ks},
                compounding_by_k={k: _compounding_value(s["c"], k, decision_h) for k in ks},
                identity_residual=s["identity_residual"],
                open_loop_divergence=s["open_loop_divergence"],
                k_one_is_floor=s["k_one_is_floor"],
                displacement_median=s["displacement_median"], floor_median=s["floor_median"],
                clusters=sum(1 for n in EXPECTED_WINDOWS if n), rows=sum(EXPECTED_WINDOWS),
            )
            for cell, s in spec.items()
        },
        decision_h=decision_h, ks=tuple(ks),
    )


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


STATUSES = ("PREDICTS_MOTION", "COPIES", "INDETERMINATE", "UNRESOLVED_CONTROL", "UNREADABLE")


def test_the_read_fixtures_reach_the_status_they_are_named_for_and_sort_as_stated(record):
    """The fixtures' own check, taken from `reading_burden` over inputs built
    from the spec alone: a pool that read some other status than its name would
    make every exit-code test below test the wrong thing."""
    for status in STATUSES:
        spec = _scenario(status)
        assert burden.reading_burden(_expected_inputs(spec)).status == status, status
    # The refusals sit on a pool that decides on its own.
    assert burden.reading_burden(
        _expected_inputs(_spec(KINDS["PREDICTS_MOTION"]))
    ).status == "PREDICTS_MOTION"
    cells = sorted(_records_from(record, _scenario("COPIES")))
    assert (cells[0], cells[1], cells[-1]) == (FIRST_CELL, SECOND_CELL, LAST_CELL)
    assert cells != [(a, s) for a in READ_ARMS for s in READ_SEEDS], "ARMS order must not be sorted()"
    assert sum(1 for n in EXPECTED_WINDOWS if n) == 8 and sum(EXPECTED_WINDOWS) == 13


# --- READ_EXITS --------------------------------------------------------------


def test_read_exits_are_the_two_refusal_statuses_each_to_its_own_number():
    """THE MUTATION THIS EXISTS FOR: swapping 45 and 46, which would report a
    broken estimator as an unreadable plan. The numbers are literals here rather
    than `script.EXIT_*`, so a swap of the two CONSTANTS' own definitions is
    caught as well as a swap of the two values in the table."""
    assert script.READ_EXITS == {"UNRESOLVED_CONTROL": 45, "UNREADABLE": 46}
    assert script.READ_EXITS == {
        "UNRESOLVED_CONTROL": script.EXIT_CONTROL_BROKEN,
        "UNREADABLE": script.EXIT_UNREADABLE,
    }


@pytest.mark.parametrize(
    "status, code",
    [("PREDICTS_MOTION", 0), ("COPIES", 0), ("INDETERMINATE", 0),
     ("UNRESOLVED_CONTROL", 45), ("UNREADABLE", 46)],
)
def test_read_phase_returns_the_exit_of_the_status_the_records_read(
    record, tmp_path, capsys, status, code,
):
    """Every status from real records through the real read phase. A decisive
    status is a READING and exits 0 (a milestone that exited non-zero on a
    finding would make "the run worked" and "the news was good" one signal); the
    refusals exit with their own number, and the output says WHICH status
    produced it, so a swapped pair is a different line and not only a different
    number."""
    records = _records_from(record, _scenario(status))
    args = _read_args(tmp_path, records)
    assert script.read_phase(args) == code
    out = capsys.readouterr().out
    written = sorted(p.name for p in args.out.iterdir())
    if code == 0:
        assert f"verdict: {status.replace('_', ' ')} -- decided by:" in out, out
        assert written == sorted(
            [f"burden_{arm}_seed{seed}.json" for arm, seed in records] + ["burden.txt"]
        )
    else:
        assert status in out, out
        assert "verdict:" not in out, "a refusal is not a reading"
        assert "burden.txt" not in written


@pytest.mark.parametrize("control", sorted(BROKEN_CONTROLS))
def test_each_missed_control_alone_is_a_broken_control_that_names_its_cell(
    record, tmp_path, capsys, control,
):
    """Each of the three known answers, missed ALONE in the cell that sorts last
    of nine. Three controls and one `UNRESOLVED_CONTROL`: a record read with one
    control taken from another's key, or a flag defaulted, passes the two tests
    above and fails here. The refusal names that cell and no other."""
    spec = _scenario("PREDICTS_MOTION", broken=(LAST_CELL, control))
    args = _read_args(tmp_path, _records_from(record, spec))
    assert script.read_phase(args) == 45
    out = capsys.readouterr().out
    assert _named_cells(out, spec) == {LAST_CELL}, out
    assert "burden.txt" not in [p.name for p in args.out.iterdir()]


def test_a_stationary_cell_is_unreadable_and_named(record, tmp_path, capsys):
    spec = _scenario("PREDICTS_MOTION", stationary=SECOND_CELL)
    args = _read_args(tmp_path, _records_from(record, spec))
    assert script.read_phase(args) == 46
    out = capsys.readouterr().out
    assert _named_cells(out, spec) == {SECOND_CELL}, out
    assert "burden.txt" not in [p.name for p in args.out.iterdir()]


@pytest.mark.parametrize("seeds", [(0, 1), (1, 2), (2,)])
def test_an_arm_short_of_the_minimum_seeds_is_unreadable_not_a_raise(
    record, tmp_path, capsys, seeds,
):
    """A narrowed SEED plan is `reading_burden`'s business and comes back as
    `UNREADABLE`, naming each short arm with its count; it does not raise. The
    pool itself is sound -- every cell clears, none is stationary, no control is
    missed -- so the only thing that can produce 46 here is the seed count, and
    the three arms are named, not one."""
    args = _read_args(tmp_path, _records_from(record, _scenario("PREDICTS_MOTION")), seeds=seeds)
    assert script.read_phase(args) == 46
    out = capsys.readouterr().out
    for arm in READ_ARMS:
        assert f"{arm} carries {len(seeds)} seed(s), fewer than {burden.SEEDS_MINIMUM}" in out, out
    assert "burden.txt" not in [p.name for p in args.out.iterdir()]


def test_read_phase_writes_nothing_when_it_refuses(record, tmp_path, capsys):
    """A refusal must leave no burden.txt behind to be mistaken for a reading.

    THE MUTATION THIS EXISTS FOR: writing `burden.txt` before the status is
    checked. Asked of BOTH refusals, with the exit code asserted so the file is
    absent because the refusal was reached and not because something else
    stopped first."""
    for status in ("UNRESOLVED_CONTROL", "UNREADABLE"):
        args = _read_args(tmp_path / status, _records_from(record, _scenario(status)))
        assert script.read_phase(args) == script.READ_EXITS[status], status
        assert not (args.out / "burden.txt").exists(), status
        capsys.readouterr()


def _overwrite_record(args, cell, new) -> None:
    """Replace one cell's file in place -- under its literal name, as `_read_args`
    wrote it."""
    write_record(args.out / f"burden_{cell[0]}_seed{cell[1]}.json", new)


def _make_control_broken(args, record) -> None:
    _overwrite_record(
        args, LAST_CELL, _records_from(record, _scenario("UNRESOLVED_CONTROL"))[LAST_CELL],
    )


def _make_unreadable(args, record) -> None:
    _overwrite_record(
        args, LAST_CELL, _records_from(record, _scenario("UNREADABLE"))[LAST_CELL],
    )


def _make_protocol_disagree(args, record) -> None:
    odd = _records_from(record, _scenario("PREDICTS_MOTION"))[LAST_CELL]
    DISAGREEMENTS["git_sha"](odd)
    _overwrite_record(args, LAST_CELL, odd)


def _make_cell_missing(args, record) -> None:
    (args.out / f"burden_{LAST_CELL[0]}_seed{LAST_CELL[1]}.json").unlink()


def _make_plan_too_narrow(args, record) -> None:
    args.arms = ["pixel_ae"]


STALE_REFUSALS = {
    # name: (what to change after a good read, what the second read does)
    "UNRESOLVED_CONTROL": (_make_control_broken, 45),
    "UNREADABLE": (_make_unreadable, 46),
    "a missing cell": (_make_cell_missing, script.EXIT_NO_CHECKPOINTS),
    "a protocol disagreement": (_make_protocol_disagree, SystemExit),
    "a plan too narrow to read": (_make_plan_too_narrow, SystemExit),
}


@pytest.mark.parametrize("refusal", sorted(STALE_REFUSALS))
def test_a_refusal_removes_the_burden_txt_an_earlier_read_left_and_nothing_else(
    record, tmp_path, capsys, monkeypatch, refusal,
):
    """A reading from an earlier read of the SAME directory is not a reading of
    what is there now. If it survived a refusal, `burden.txt` would sit beside a
    line saying none was written, and nothing in the directory would say which of
    the two a later reader holds.

    THE MUTATIONS THIS EXISTS FOR: removing the unlink; making it conditional on
    reaching the status check (so only the two numbered refusals clear it, and a
    named `SystemExit` or a missing cell leave the file); and an unlink that
    reaches past `--out`. A successful read comes first, so the file is a real
    reading and its absence afterwards is the refusal's doing; the SECOND read is
    asked in the same directory, over records changed in the one way each case
    names.

    Three things sit where an over-wide unlink would find them: a `burden.txt`
    beside the output directory, one in the current directory, and a `.txt` file
    that is not `burden.txt` inside it. The directory must hold exactly what it
    held before the second read, less `burden.txt`."""
    change, outcome = STALE_REFUSALS[refusal]
    args = _read_args(tmp_path, _records_from(record, _scenario("PREDICTS_MOTION")))
    stale = args.out / "burden.txt"
    assert script.read_phase(args) == 0
    assert "verdict: PREDICTS MOTION -- decided by:" in stale.read_text()
    capsys.readouterr()

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    bystanders = [tmp_path / "burden.txt", cwd / "burden.txt", args.out / "notes.txt"]
    for bystander in bystanders:
        bystander.write_text("not this one")

    change(args, record)
    before = sorted(p.name for p in args.out.iterdir())
    assert "burden.txt" in before
    if isinstance(outcome, int):
        assert script.read_phase(args) == outcome
        out = capsys.readouterr().out
        if outcome in (45, 46):
            assert "no reading was taken, so no burden.txt was written" in out, out
        assert "verdict:" not in out
    else:
        with pytest.raises(outcome):
            script.read_phase(args)

    assert not stale.exists(), "the reading an earlier read left is still there"
    assert sorted(p.name for p in args.out.iterdir()) == [n for n in before if n != "burden.txt"]
    for bystander in bystanders:
        assert bystander.exists(), f"{bystander} was removed: it is not --out's burden.txt"
        assert bystander.read_text() == "not this one", bystander


def test_a_contradiction_between_the_decisive_statuses_propagates_and_writes_nothing(
    record, tmp_path,
):
    """`reading_burden` raises when two arms clear motion and two clear copies,
    and that is an arm count that has outgrown the bar -- an operator or shape
    error, not something found in the data. It must come out of `read_phase` as
    the ValueError it is: not caught into a status, not an exit number.

    THE MUTATION THIS EXISTS FOR: a `try/except ValueError` around the reading
    that returns `EXIT_UNREADABLE`. Four arms, which `read_phase` does not
    forbid -- it takes its arms from the command line and does not check them
    against `ARMS`."""
    four = {
        "pixel_ae": ("up", "up", "up"), "frozen_ssl": ("up", "up", "up"),
        "random_vit": ("down", "down", "down"), "fourth_arm": ("down", "down", "down"),
    }
    args = _read_args(tmp_path, _records_from(record, _spec(four)), arms=tuple(four))
    with pytest.raises(ValueError, match="contradict"):
        script.read_phase(args)
    assert not (args.out / "burden.txt").exists()


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
    "confidence": lambda r: r.__setitem__("confidence", 0.9),
    "resamples": lambda r: r.__setitem__("resamples", r["resamples"] + 1),
    "identity_tolerance": lambda r: r.__setitem__("identity_tolerance", r["identity_tolerance"] * 10),
    "episodes.val": lambda r: r["episodes"].__setitem__("val", r["episodes"]["val"][::-1]),
    "windows.episode": lambda r: r["windows"]["episode"].__setitem__(
        -1, r["windows"]["episode"][-1] + 1
    ),
}
"""One way for a record to disagree with the others, per protocol field."""

REQUIRED_PROTOCOL_FIELDS = {
    "git_sha", "step", "context", "horizon", "ks", "decision_h", "reported_h",
    "confidence", "resamples", "identity_tolerance", "split_seed", "device",
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
    records = _records_from(record, _scenario("COPIES"))
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
    records = _records_from(record, _scenario("COPIES"))
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
    records = _records_from(record, _scenario("COPIES"))
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
    records = _records_from(record, _scenario("COPIES"))
    for index, cell in enumerate(sorted(records)):
        records[cell]["record_git_sha"] = f"trained-at-{index}"
        records[cell]["kl_dyn_max"] = 1.0 + index
        records[cell]["kl_rate_above_free_bits"] = 0.5 + index / 32.0
    assert script.require_one_protocol(records) is None


@pytest.mark.parametrize("lacking", ["first", "last", "all"])
def test_a_record_without_a_protocol_field_is_refused_by_name_not_a_key_error(record, lacking):
    """A record that lacks a protocol field is refused BY NAME, not with a bare
    `KeyError` and not by defaulting the field.

    `all` is the case a default would pass: if every record lacks `git_sha` and
    a missing field is read as `None`, the nine agree on `None` and pool with no
    refusal. With only one record lacking it the default would disagree with the
    other eight and be refused anyway, which is why the two single-record cases
    cannot tell a refusal from a default."""
    records = _records_from(record, _scenario("COPIES"))
    victims = {"first": [FIRST_CELL], "last": [LAST_CELL], "all": sorted(records)}[lacking]
    for cell in victims:
        del records[cell]["git_sha"]
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert f"{victims[0][0]} seed {victims[0][1]} lacks git_sha" in message, message


def test_an_empty_pool_is_refused_by_name():
    for call in (script.require_one_protocol, script.burden_inputs):
        with pytest.raises(SystemExit, match="no burden record"):
            call({})


def test_read_phase_refuses_records_that_disagree_on_the_protocol(record, tmp_path, capsys):
    """THE WIRING, not the function: `require_one_protocol` is tested above in
    isolation, and deleting its one call from `read_phase` leaves every one of
    those green. `git_sha` is the field the check once omitted: the records
    below differ in nothing else, so no other comparison -- `burden_inputs`'s
    own, `reading_burden`'s -- can refuse them in its place.

    THE MUTATIONS THIS EXISTS FOR: deleting the `require_one_protocol(records)`
    call from `read_phase`, and dropping `git_sha` from `_PROTOCOL_FIELDS`. The
    victim sorts LAST, so a first-record-only check sees nothing; and nothing is
    written, because a pool that cannot be read must leave no artefact claiming
    it was."""
    records = _records_from(record, _scenario("PREDICTS_MOTION"))
    victim = sorted(records)[-1]
    assert victim == LAST_CELL
    records[victim]["git_sha"] = "0" * 40
    args = _read_args(tmp_path, records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(args)
    message = str(caught.value)
    assert "random_vit seed 2" in message and "git_sha" in message, message
    assert not (args.out / "burden.txt").exists()
    assert "verdict:" not in capsys.readouterr().out


# --- burden_inputs -----------------------------------------------------------


@pytest.mark.parametrize(
    "status, decision_h, ks",
    [(status, burden.DECISION_H, REGROUNDING_KS) for status in STATUSES]
    + [("COPIES", 30, REGROUNDING_KS), ("INDETERMINATE", 20, (1, 3, 7))],
)
def test_burden_inputs_reads_every_field_from_its_own_key_at_the_decision_horizon(
    record, tmp_path, status, decision_h, ks,
):
    """Compared with `BurdenInputs` built from the spec's formulas -- not from the
    records, and not by `reading_burden` -- both in memory and after a round trip
    through the files the measure phase writes.

    At `decision_h=30` and `20` the horizon step is not the last one: a read at
    `[-1]` or at `[h]` is another number, where at 45 `[-1]` and `[h - 1]` are the
    same element. The margin is read at the DECISION horizon of the records, and
    the rungs are the records' own `ks`, whose last one is not 45 in the third."""
    spec = _scenario(status)
    records = _records_from(record, spec, decision_h=decision_h, ks=ks)
    expected = _expected_inputs(spec, decision_h=decision_h, ks=ks)
    assert script.burden_inputs(records) == expected
    args = _read_args(tmp_path, records)
    loaded = script.load_burden(args.out, READ_ARMS, READ_SEEDS)
    assert script.burden_inputs(loaded) == expected


@pytest.mark.parametrize("field", ["decision_h", "ks"])
def test_burden_inputs_refuses_a_pool_that_disagrees_on_the_one_value_it_takes(record, field):
    """`BurdenInputs` holds ONE decision horizon and ONE ladder for nine cells, so
    the pick is made where it is taken, and does not depend on
    `require_one_protocol` having run first."""
    records = _records_from(record, _scenario("COPIES"))
    DISAGREEMENTS[field](records[LAST_CELL])
    with pytest.raises(SystemExit) as caught:
        script.burden_inputs(records)
    message = str(caught.value)
    assert f"disagree on {field}:" in message and "random_vit seed 2" in message, message


# --- the plan, and loading ---------------------------------------------------


def test_read_phase_refuses_a_plan_narrower_than_the_bar_before_loading_anything(tmp_path):
    """`--arms pixel_ae` cannot be read: Reading H needs `ARMS_REQUIRED` arms, and
    falling through would print INDETERMINATE under a sentence about one arm --
    a status the plan produced and the data did not. It raises, and BEFORE the
    records are looked for: the directory below does not exist, so a plan check
    made after the load would return 11 instead.

    `--arms a a a` is ONE arm: counting the list would let it through."""
    for arms in (["pixel_ae"], ["pixel_ae", "pixel_ae", "pixel_ae"]):
        args = types.SimpleNamespace(out=tmp_path / "absent", arms=arms, seeds=[0, 1, 2])
        with pytest.raises(SystemExit) as caught:
            script.read_phase(args)
        message = str(caught.value)
        assert f"ARMS_REQUIRED={burden.ARMS_REQUIRED}" in message, message
        assert "1 arm(s)" in message, message


def test_two_arms_of_the_nine_are_a_readable_plan_and_only_they_are_pooled(
    record, tmp_path, capsys,
):
    """Two arms is the bar, so it is readable; and the third arm's records are
    on disk and outside the plan, so they must not be pooled."""
    args = _read_args(
        tmp_path, _records_from(record, _scenario("PREDICTS_MOTION")),
        arms=("pixel_ae", "frozen_ssl"),
    )
    assert script.read_phase(args) == 0
    out = capsys.readouterr().out
    assert "in 2 of 2 arms (frozen_ssl, pixel_ae)" in out, out
    assert "random_vit" not in out


def test_read_phase_names_the_first_missing_cell_and_exits_eleven(record, tmp_path, capsys):
    """Two cells are missing; the one the PLAN reaches first is the one named,
    and nothing is written."""
    records = _records_from(record, _scenario("COPIES"))
    del records[SECOND_CELL], records[LAST_CELL]
    args = _read_args(tmp_path, records)
    assert script.read_phase(args) == script.EXIT_NO_CHECKPOINTS
    out = capsys.readouterr().out
    assert "NO CELL" in out and "frozen_ssl seed 1" in out and "random_vit seed 2" not in out, out
    assert "--phase measure" in out
    assert not (args.out / "burden.txt").exists()


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
    records = _records_from(record, _scenario("COPIES"))
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


def test_burden_txt_is_the_formatters_text_byte_for_byte_and_stdout_is_the_same_string(
    record, tmp_path, capsys,
):
    """THE MUTATIONS THIS EXISTS FOR: `write_text(text.rstrip())`, and `print(text)`
    on a string that already ends in a newline. Both are invisible to a test that
    compares two reads of the SAME code (each read strips the same bytes), so the
    expected text is built here, from the spec, by the formatter -- not by the
    read phase -- and compared with the file and with stdout separately.

    The precondition is asserted: the text ends in exactly one newline, which is
    what `rstrip()` would lose and `print()` would double."""
    spec = _scenario("PREDICTS_MOTION")
    inputs = _expected_inputs(spec)
    expected = burden.format_reading_burden(burden.reading_burden(inputs), inputs)
    assert expected.endswith("\n") and not expected.endswith("\n\n")
    args = _read_args(tmp_path, _records_from(record, spec))
    assert script.read_phase(args) == 0
    assert (args.out / "burden.txt").read_bytes() == expected.encode()
    assert capsys.readouterr().out == expected


def test_the_reading_is_byte_identical_on_two_reads(record, tmp_path):
    """The point of the two-phase split: a reading reproducible from the records
    without a GPU. Two independent reads, in two directories, of the same nine
    records.

    THE MUTATION THIS EXISTS FOR: `rstrip()` on the text. Two reads of the SAME
    code strip the same bytes and agree with each other, so agreement alone
    cannot see it -- both files are ALSO compared with the formatter's text,
    built here from the spec, which ends in the newline `rstrip()` would lose."""
    spec = _scenario("PREDICTS_MOTION")
    inputs = _expected_inputs(spec)
    expected = burden.format_reading_burden(burden.reading_burden(inputs), inputs).encode()
    assert expected.endswith(b"\n") and not expected.endswith(b"\n\n")
    records = _records_from(record, spec)
    first, second = _read_args(tmp_path / "a", records), _read_args(tmp_path / "b", records)
    assert script.read_phase(first) == script.read_phase(second) == 0
    a, b = (first.out / "burden.txt").read_bytes(), (second.out / "burden.txt").read_bytes()
    assert a == b
    assert a == expected


# --- main: the three phases --------------------------------------------------


def _main(monkeypatch, argv, *, measure=0, read=0):
    """`main(argv)` with BOTH phases replaced by recorders, returning
    `(status, phases_run)`. What this pins is the WIRING: which phase runs, in
    which order, and whether a failed measure stops the read."""
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


def test_main_runs_the_phases_the_flag_names_and_returns_the_last_status(monkeypatch):
    assert _main(monkeypatch, ["--phase", "measure"]) == (0, ["measure"])
    assert _main(monkeypatch, ["--phase", "read"]) == (0, ["read"])
    assert _main(monkeypatch, ["--phase", "all"]) == (0, ["measure", "read"])
    assert _main(monkeypatch, ["--phase", "read"], read=45) == (45, ["read"])
    assert _main(monkeypatch, ["--phase", "all"], read=46) == (46, ["measure", "read"])


def test_a_failed_measure_stops_phase_all_before_it_reads(monkeypatch):
    """A measure that returned 14 wrote no record for its cell, and a read after
    it would be over a pool missing that cell."""
    assert _main(monkeypatch, ["--phase", "all"], measure=14, read=0) == (14, ["measure"])


@pytest.mark.parametrize("status", [11, 12, 14, 30])
def test_a_failed_measure_is_the_exit_status_of_phase_measure_too(monkeypatch, status):
    """`--phase measure` has no read after it, and that is exactly where a failed
    measure's status could be dropped on the floor: the guard that stops the read
    in `--phase all` is also what RETURNS the status, and a `--phase measure` that
    fell through to `EXIT_OK` would print a success after a refused 30-minute run.

    THE MUTATION THIS EXISTS FOR: narrowing `main`'s `if status != EXIT_OK:` to
    `and args.phase == "all"`. The statuses are the literals `measure_phase` can
    return (11, 12, 14 from the shared cell checks, 30 from the self-check), not
    `script.EXIT_*`, and `read` is made to answer 0 so a status that was read from
    the wrong phase is a different number."""
    assert _main(monkeypatch, ["--phase", "measure"], measure=status, read=0) == (
        status, ["measure"],
    )


def test_the_default_phase_is_read_and_the_parser_names_the_three():
    """A default of `measure` would make a bare invocation pay for nine cells of
    GPU time; `read` costs none and refuses by name when there are no records."""
    args = script._parser().parse_args([])
    assert args.phase == "read"
    assert script.PHASES == ("all", "measure", "read")
    assert args.out == Path("runs/m3m_burden")
    with pytest.raises(SystemExit) as caught:
        script._parser().parse_args(["--phase", "readd"])
    assert caught.value.code == 2


def test_phase_all_refuses_a_plan_it_could_not_read_before_measuring_anything(monkeypatch):
    """`--phase all --arms pixel_ae` would be the whole measure followed by a
    refusal at the read. `--phase measure` is allowed the same plan: the
    milestone's smoke is one cell."""
    ran = []
    monkeypatch.setattr(script, "measure_phase", lambda args: ran.append("measure") or 0)
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "all", "--arms", "pixel_ae"])
    assert f"ARMS_REQUIRED={burden.ARMS_REQUIRED}" in str(caught.value)
    assert ran == []
    assert _main(monkeypatch, ["--phase", "measure", "--arms", "pixel_ae"]) == (0, ["measure"])


@pytest.mark.parametrize(
    "seeds, distinct",
    [
        pytest.param(
            [str(i) for i in range(burden.SEEDS_MINIMUM - 1)], burden.SEEDS_MINIMUM - 1,
            id="one short of the minimum",
        ),
        pytest.param(["5"], 1, id="one seed"),
        # `--seeds a a a` is ONE seed: counting the list would let it through, and
        # the read would then find an arm with one seed.
        pytest.param(["5"] * burden.SEEDS_MINIMUM, 1, id="enough listed, one distinct"),
    ],
)
def test_phase_all_refuses_too_few_seeds_before_measuring_anything(
    monkeypatch, seeds, distinct,
):
    """`--phase all --seeds 0 1` would measure all six cells and then come back 46:
    every arm carries two seeds, and `reading_burden` refuses an arm short of
    `SEEDS_MINIMUM`. The plan is known at the start, so the refusal is made there.

    THE MUTATIONS THIS EXISTS FOR: dropping the check from `main`; counting the
    seeds as listed instead of as distinct (the third case); and making the
    check `<=` (see the boundary test below). It raises BY NAME rather than
    returning 46 -- the numbered exits report what was found in the data -- and
    `ran == []` is what says it came BEFORE the measure, not after it."""
    ran = []
    monkeypatch.setattr(script, "measure_phase", lambda args: ran.append("measure") or 0)
    monkeypatch.setattr(script, "read_phase", lambda args: ran.append("read") or 0)
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "all", "--seeds", *seeds])
    message = str(caught.value)
    assert f"SEEDS_MINIMUM={burden.SEEDS_MINIMUM}" in message, message
    assert f"{distinct} seed(s)" in message, message
    assert ran == []


def test_a_seed_plan_short_of_the_minimum_is_refused_only_where_it_would_waste_a_measure(
    monkeypatch,
):
    """The seed check belongs to `--phase all` and to nothing else.

    `--phase measure` is the milestone's smoke and may run one seed. `--phase read`
    must REACH `read_phase` with a short plan, because that is where the short-arm
    46 comes from (`test_an_arm_short_of_the_minimum_seeds_is_unreadable_not_a_raise`
    drives it): a check in front of it would make the status unreachable. And
    exactly `SEEDS_MINIMUM` seeds is a plan -- the boundary a `<=` would refuse.

    THE MUTATIONS THIS EXISTS FOR: applying the check to `--phase measure` (the
    first line fails), applying it to `--phase read` (the second), and `<=` for `<`
    (the third)."""
    short = ["--seeds", *(str(i) for i in range(burden.SEEDS_MINIMUM - 1))]
    assert _main(monkeypatch, ["--phase", "measure", *short]) == (0, ["measure"])
    assert _main(monkeypatch, ["--phase", "read", *short], read=46) == (46, ["read"])
    enough = ["--seeds", *(str(i) for i in range(burden.SEEDS_MINIMUM))]
    assert _main(monkeypatch, ["--phase", "all", *enough]) == (0, ["measure", "read"])


def test_main_reads_the_arms_seeds_and_directory_the_command_line_names(
    monkeypatch, record, tmp_path, capsys,
):
    """The flag names reach the fields `read_phase` reads, through the real read
    phase. The measure phase is made unreachable: a `--phase read` that fell into
    it would load the real checkpoints from `--source`'s default."""
    def forbidden(args):
        raise AssertionError("--phase read reached the measure phase")

    monkeypatch.setattr(script, "measure_phase", forbidden)
    args = _read_args(tmp_path, _records_from(record, _scenario("PREDICTS_MOTION")))
    argv = ["--phase", "read", "--out", str(args.out), "--arms", "pixel_ae", "frozen_ssl",
            "--seeds", "0", "1", "2"]
    assert script.main(argv) == 0
    assert "in 2 of 2 arms" in capsys.readouterr().out
    broken = _read_args(tmp_path / "broken", _records_from(record, _scenario("UNRESOLVED_CONTROL")))
    assert script.main(["--phase", "read", "--out", str(broken.out)]) == 45
