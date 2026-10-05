"""M3n's script: the measure phase, the record it writes, and the read phase.

THREE KINDS OF FIXTURE, each used where the others cannot reach.

  * A HAND-BUILT SWEEP (`_hand_sweep`). `measure_cell` is handed a
    `RegroundingSweep` whose per-window rows are written out by the test, so
    every interval, share, secondary figure and control can be asked against an
    answer that never passed through the sweep. Every row is a multiple of 1/64,
    so every sum and difference in the identity is exact and `==` is the right
    comparison. The sweep is built to DISCRIMINATE (`test_the_hand_sweep_...`):
    six clusters, headroom that is negative at some (k, h), exactly zero at one,
    positive-but-unresolvable at another and resolvable elsewhere. A fixture in
    which the headroom is positive everywhere cannot tell a share gated on the
    point from one gated on the interval, and the first rig tried here was that
    fixture.
  * THE REAL SWEEP over the wobbling-agent rig `test_prediction_burden_script.py`
    uses, with the oracle-family models from `test_rollout.py`. It is where
    `seed`, `device`, `feature_backbone` and `ks` are shown to REACH the sweep,
    and where the closed form for `hold`, `rung` and `floor` -- derived from the
    rig's own linear probe, without the sweep -- is compared with what the sweep
    returned.
  * RECORDS ON DISK for the read phase, each a deep copy of the record
    `measure_cell` writes with only the fields the reading reads overwritten, so
    the schema is the writer's own.

THE READ PHASE'S EXPECTATIONS ARE BUILT FROM THE SPEC, never from
`script.headroom_inputs`: `_expected_inputs` writes the `HeadroomInputs` a pool
must read as from the same formulas that wrote the records.
"""

import contextlib
import copy
import importlib.util
import io
import json
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mbfps.data.episode import load_episode, save_episode
from mbfps.eval import headroom as H
from mbfps.eval import pooling
from mbfps.eval.diagnostics import REGROUNDING_KS, RegroundingSweep, ground_step, regrounding_sweep
from mbfps.eval.probe import apply_probe, fit_probe, probe_targets
from mbfps.eval.rollout import RolloutResult, evaluate_rollout
from mbfps.eval.study import SPLIT_SEED, git_sha, load_record, write_record
from mbfps.eval.windows import window_starts
from tests.eval.test_rollout import (
    KEYS, STEP, DriftingModel, OracleModel, _OracleRSSM, oracle_probe, synthetic_episode,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "motion_headroom.py"


def _load():
    spec = importlib.util.spec_from_file_location("motion_headroom_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()

CONTEXT, HORIZON = 5, 45
CPU = torch.device("cpu")
SEED = 3
"""The cell's seed. It is neither 0 nor `BOOTSTRAP_SEED`, so a bootstrap drawn at
the cell's seed, at 0, or at the bootstrap seed are three different draws."""
BOOTSTRAP_SEED = 7
"""The run's BASE seed, `--bootstrap-seed`. No interval is drawn at it: each cell
draws at `script.cell_bootstrap_seed(BOOTSTRAP_SEED, arm, seed)`."""
SMALL = 40
"""A draw count the tests patch in, so recomputing an interval costs milliseconds
and so a record can be told to have used the PATCHED count and not 2000."""

LENGTHS = (70, 120, 50, 95, 170, 100, 60, 130, 75)
EXPECTED_WINDOWS = (1, 2, 0, 1, 3, 2, 1, 2, 1)

STUDY_RECORD = {
    "steps": 20000, "kl_rate_above_free_bits": 0.9126, "kl_dyn_max": 2.951450824737549,
    "git_sha": "ca3e140772d6bc741d4d04312763afe3dd754166",
}


def _q(x) -> np.ndarray:
    """Round to a multiple of 1/64: every sum and difference of these is exact in
    float64 at the magnitudes used here, so the identity residual of a fixture
    built from them is exactly 0.0 and expectations can use `==`."""
    return np.round(np.asarray(x, dtype=np.float64) * 64.0) / 64.0


# ---------------------------------------------------------------------------
# The hand-built sweep.
# ---------------------------------------------------------------------------

ZERO_CONTROL_NAMES = (
    "persistence_divergence", "k_invariance_at_h1", "open_loop_divergence", "floor_divergence",
)
"""The four controls whose known answer is exactly 0.0, spelled out here: a test
that iterated `ZERO_CONTROL_NAMES` would shrink with it when one was dropped."""

HAND_KS = (1, 3, 45)
HAND_GROUPS = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5])
HAND_NAMES = [f"ep_{i:06d}.npz" for i in (5, 4, 3, 2, 1, 0)]
"""Six validation episodes, named in DESCENDING order so that a record's
`episodes.val` is not what `sorted()` would give."""

NEGATIVE_STEPS = [[1, 2], [1, 8], [3, 5]]
"""The (k, h) cells where `_hand_sweep`'s headroom is negative, by construction.
Written out, not derived: they are the answer `negative_headroom_steps` is
asked against."""
ZERO_CELL = (3, 10)
"""headroom is EXACTLY 0.0 here, at every window: the cell where `point < 0.0`
and `point <= 0.0` disagree, and where a share gated on `>= 0` divides by zero."""
UNRESOLVED_CELL = (1, 3)
"""headroom's point is positive and its interval includes zero: the cell where a
gate on the point and a gate on the interval disagree."""


def _hand_arrays() -> dict:
    rng = np.random.default_rng(20261004)
    n, steps = len(HAND_GROUPS), HORIZON
    floor = _q(100.0 + 40.0 * rng.random((n, steps)))
    shift = {k: np.full(steps, 6.0) for k in HAND_KS}
    for k, h in NEGATIVE_STEPS:
        shift[k][h - 1] = -4.0
    hold, rung = {}, {}
    for k in HAND_KS:
        hold[k] = floor + _q(shift[k] + rng.normal(0.0, 0.75, (n, steps)))
        rung[k] = floor + _q(3.0 + rng.normal(0.0, 0.75, (n, steps)))
    k, h = UNRESOLVED_CELL
    hold[k][:, h - 1] = floor[:, h - 1] + np.where(np.isin(HAND_GROUPS, [1, 2, 3]), 11.0, -9.0)
    k, h = ZERO_CELL
    hold[k][:, h - 1] = floor[:, h - 1]
    # `k_invariance_at_h1`: at h=1 every k grounds at step 0.
    for k in HAND_KS:
        hold[k][:, 0], rung[k][:, 0] = hold[1][:, 0], rung[1][:, 0]
    return {"floor": floor, "hold": hold, "rung": rung}


def _hand_sweep(arrays=None) -> RegroundingSweep:
    """A `RegroundingSweep` whose rows are `_hand_arrays`, with every control at
    its known answer: the k=45 hold IS the persistence curve, the k=45 rung IS
    the open-loop curve, and the three share one floor."""
    a = arrays or _hand_arrays()
    zeros = np.zeros(HORIZON)
    reference = RolloutResult(
        horizon=np.arange(1, HORIZON + 1),
        rssm_position=a["rung"][45].mean(axis=0),
        persistence_position=a["hold"][45].mean(axis=0),
        floor_position=a["floor"].mean(axis=0),
        rssm_angle=zeros, persistence_angle=zeros, floor_angle=zeros,
    )
    return RegroundingSweep(
        ks=HAND_KS, horizon=HORIZON, reference=reference,
        position={k: a["rung"][k].mean(axis=0) for k in HAND_KS},
        angle={k: zeros for k in HAND_KS},
        window_position=a["rung"], window_floor_position=a["floor"],
        hold_position={k: a["hold"][k].mean(axis=0) for k in HAND_KS},
        window_hold_position=a["hold"],
        window_episode=HAND_GROUPS.copy(), windows_total=len(HAND_GROUPS),
    )


def _rows(sweep, k) -> dict:
    """The three differences, as the definition writes them -- not through the
    script, and not through `eval.headroom`."""
    hold, rung = sweep.window_hold_position[k], sweep.window_position[k]
    floor = sweep.window_floor_position
    return {"headroom": hold - floor, "skill": hold - rung, "deficit": rung - floor}


def _cell_kwargs(**over) -> dict:
    kwargs = dict(
        model=None, val_paths=[Path("/data") / name for name in HAND_NAMES], probe=None,
        arm="pixel_ae", seed=SEED, context=CONTEXT, horizon=HORIZON, ks=HAND_KS,
        device=CPU, feature_backbone=None, study_record=dict(STUDY_RECORD),
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    kwargs.update(over)
    return kwargs


def _hand_measure(monkeypatch, sweep=None, *, reference=None, small=True, **over) -> dict:
    """`measure_cell` over a hand-built sweep. The sweep stands in for
    `regrounding_sweep` and nothing else is replaced; `reference` defaults to a
    copy of the sweep's own, which is the clean case."""
    sweep = sweep or _hand_sweep()
    monkeypatch.setattr(script, "regrounding_sweep", lambda *a, **k: sweep)
    if small:
        monkeypatch.setattr(script, "RESAMPLES", SMALL)
    reference = reference if reference is not None else copy.deepcopy(sweep.reference)
    return script.measure_cell(**_cell_kwargs(reference=reference, **over))


@pytest.fixture(scope="module")
def hand():
    return _hand_sweep()


@pytest.fixture(scope="module")
def hand_record(hand):
    """The record `measure_cell` writes for the hand sweep: the read-phase
    fixtures' base, so the schema is the writer's own."""
    with pytest.MonkeyPatch.context() as patch:
        return _hand_measure(patch, copy.deepcopy(hand))


HAND_CELL = ("pixel_ae", SEED)
"""The `(arm, seed)` `_cell_kwargs` builds every fixture record for."""


def _cell_seed(cell=HAND_CELL) -> int:
    """The seed `cell`'s intervals are drawn at, from the run's base seed."""
    return script.cell_bootstrap_seed(BOOTSTRAP_SEED, *cell)


def _interval(rows, groups, h, *, resamples=SMALL, seed=None) -> dict:
    """`clustered_interval` at `seed`, which defaults to the fixture cell's own
    derived seed -- the one its record's intervals were drawn at."""
    point, low, high = pooling.clustered_interval(
        rows, groups, h=h, resamples=resamples, seed=_cell_seed() if seed is None else seed,
    )
    return {"point": point, "ci_low": low, "ci_high": high}


# ---------------------------------------------------------------------------
# The exits, and the one path that must carry the seed.
# ---------------------------------------------------------------------------


def test_the_exit_codes_are_this_milestones_own():
    """39/40 are M3j's, 41/42 M3k's, 43/44 M3l's, 45/46 M3m's. A collision would
    make two tools report different failures under one number. The numbers are
    literals, so a swap of the two constants' own definitions is seen."""
    assert script.EXIT_UNREADABLE_HEADROOM == 47
    assert script.EXIT_NO_MAJORITY == 48


def test_read_exits_are_the_two_non_placements_each_to_its_own_number():
    """Keyed on exactly the statuses that are NOT a placement of the model. The
    four placements exit 0: each is an answer."""
    assert script.READ_EXITS == {"UNREADABLE": 47, H.NO_MAJORITY: 48}
    assert set(script.READ_EXITS).isdisjoint({"BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS"})


def test_the_zero_controls_are_the_four_the_spec_names():
    """`persistence_divergence`, `k_invariance_at_h1`, `open_loop_divergence` and
    `floor_divergence` are each required to be exactly 0.0; `identity_residual` is
    held to a tolerance instead. A control dropped from the tuple is a control
    never read, so the tuple is compared with the four names written out here.

    THE MUTATION THIS EXISTS FOR, run against it: deleting one name from
    `ZERO_CONTROLS`."""
    assert tuple(script.ZERO_CONTROLS) == ZERO_CONTROL_NAMES


def test_the_record_path_separates_seeds_and_arms(tmp_path):
    """Three seeds of one arm must not collide on one filename.

    M3m's fixtures all used seed 0, which hid exactly this: in the real run
    `burden_record_path(out, arm, 0)` would have written all three seeds of an
    arm to one file, destroying 6 of 9 records and surfacing only at the read
    phase after the GPU time had been spent. The arm is held to the same
    standard, and the names are spelled out so the read-phase fixtures, which
    write by literal name, agree with what the script looks for.
    """
    arms = ("pixel_ae", "frozen_ssl", "random_vit")
    paths = {
        (arm, seed): script.headroom_record_path(tmp_path, arm, seed)
        for arm in arms for seed in (0, 1, 2)
    }
    assert len(set(paths.values())) == 9
    for (arm, seed), path in paths.items():
        assert path == tmp_path / f"headroom_{arm}_seed{seed}.json"


# ---------------------------------------------------------------------------
# The per-cell bootstrap seed: nine cells, nine independent resamplings.
# ---------------------------------------------------------------------------

NINE_CELLS = tuple(
    (arm, seed) for arm in ("pixel_ae", "frozen_ssl", "random_vit") for seed in (0, 1, 2)
)
"""The shipped plan, spelled out. A test iterating `ARMS` x `SEEDS` would shrink
with either one."""


@pytest.mark.parametrize("base", [0, BOOTSTRAP_SEED, 12345])
def test_the_derived_seed_differs_for_every_one_of_the_nine_cells(base):
    """The verdict counts nine cells as nine independent readings, and cells that
    resample the same episode draws are one reading nine times. Every `(arm, seed)`
    gets its own seed at each of three bases, base 0 being the CLI default.

    THE MUTATIONS THIS EXISTS FOR, each run against it: returning `base_seed`
    unchanged (the shipped defect), and dropping `arm` or `seed` from the digest --
    either leaves three or nine cells on one seed."""
    derived = {cell: script.cell_bootstrap_seed(base, *cell) for cell in NINE_CELLS}
    assert len(derived) == 9
    assert len(set(derived.values())) == 9, derived
    assert base not in derived.values()


def test_the_derived_seed_is_stable_and_is_not_drawn_from_python_hash():
    """The same `(base, arm, seed)` is the same integer on every call, process and
    platform, so any single interval of a record reproduces from the record. The
    two literals are what `zlib.crc32` of `b"base:arm:seed"` returns; a `hash()`
    swapped back in is salted per process (`PYTHONHASHSEED`) and misses them, as
    does any other digest.

    THE MUTATION THIS EXISTS FOR, run against it: `zlib.crc32(...)` ->
    `hash(...) & 0xFFFFFFFF`."""
    assert script.cell_bootstrap_seed(0, "pixel_ae", 0) == 235428016
    assert script.cell_bootstrap_seed(7, "frozen_ssl", 2) == 3554091843
    assert script.cell_bootstrap_seed(0, "pixel_ae", 0) == script.cell_bootstrap_seed(
        0, "pixel_ae", 0,
    )
    for cell in NINE_CELLS:
        value = script.cell_bootstrap_seed(0, *cell)
        assert isinstance(value, int) and 0 <= value < 2**32, cell


def test_the_derived_seed_moves_when_the_base_moves_for_every_cell():
    """`--bootstrap-seed` must stay a lever: a different base is a different run,
    for every cell, not only for one.

    THE MUTATION THIS EXISTS FOR, run against it: leaving `base_seed` out of the
    digest, which makes every run of the script draw the same nine seeds."""
    for cell in NINE_CELLS:
        assert script.cell_bootstrap_seed(0, *cell) != script.cell_bootstrap_seed(1, *cell), cell
        assert script.cell_bootstrap_seed(7, *cell) != script.cell_bootstrap_seed(8, *cell), cell


def test_one_cell_draws_every_interval_at_its_one_derived_seed(hand, monkeypatch):
    """The independence that matters is BETWEEN cells. Within a cell all three
    differences at every `(k, h)` are drawn at the cell's one seed, and it is the
    derived one -- not the base, and not the cell's own `seed`.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `seed=cell_seed` ->
    `seed=bootstrap_seed`, and deriving a fresh seed per `(k, h)`."""
    drawn = []
    real = script.clustered_interval

    def spy(*args, **kwargs):
        drawn.append(kwargs["seed"])
        return real(*args, **kwargs)

    monkeypatch.setattr(script, "clustered_interval", spy)
    record = _hand_measure(monkeypatch, copy.deepcopy(hand))
    assert len(drawn) == len(HAND_KS) * len(H.REPORTED_H) * 3
    assert set(drawn) == {record["cell_bootstrap_seed"]} == {_cell_seed()}


def test_two_cells_on_the_same_rows_share_every_point_estimate_and_not_their_bounds(
    hand, monkeypatch,
):
    """The point estimate is seed-free, so on identical rows two cells must agree
    on every `point` exactly and may differ only in the bounds. That is what tells
    nine independent draws from nine identical ones: with the seed shared, every
    bound would be equal too, and a reader could not see it.

    The rows are the SAME OBJECT for both cells, so a difference in a bound is the
    seed and nothing else. The decision cell is where it is asserted for all three
    differences, on both bounds; the control is the same cell measured twice, which
    must be identical to the last bit.

    THE MUTATION THIS EXISTS FOR, run against it: `seed=cell_seed` ->
    `seed=bootstrap_seed`, under which the bounds below are equal."""
    first = _hand_measure(monkeypatch, copy.deepcopy(hand), arm="pixel_ae", seed=0)
    second = _hand_measure(monkeypatch, copy.deepcopy(hand), arm="frozen_ssl", seed=1)
    again = _hand_measure(monkeypatch, copy.deepcopy(hand), arm="pixel_ae", seed=0)
    assert first["cell_bootstrap_seed"] != second["cell_bootstrap_seed"]
    assert first["bootstrap_seed"] == second["bootstrap_seed"] == BOOTSTRAP_SEED
    assert first["intervals"] == again["intervals"]

    moved = 0
    for k in HAND_KS:
        for h in H.REPORTED_H:
            for name in ("headroom", "skill", "deficit"):
                a = first["intervals"][str(k)][str(h)][name]
                b = second["intervals"][str(k)][str(h)][name]
                assert a["point"] == b["point"], (k, h, name)
                moved += (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])
                if (k, h) == (H.DECISION_K, H.DECISION_H):
                    assert a["ci_low"] != b["ci_low"], name
                    assert a["ci_high"] != b["ci_high"], name
    assert moved > 0


# ---------------------------------------------------------------------------
# measure_cell, over a sweep whose rows the test wrote.
# ---------------------------------------------------------------------------


def test_the_hand_sweep_discriminates_what_the_assertions_below_are_about(hand):
    """The fixture's own check, taken before anything is asserted about a
    result and from `pooling.clustered_interval` directly -- not from the script.

    Six clusters; an interval of positive width on all three differences at the
    decision cell; a cell whose headroom is negative at the POINT, one where it
    is exactly zero, one where the point is positive and the interval is not
    above zero, and the rest resolvably positive; a nonzero standard error.
    """
    assert len(set(HAND_GROUPS.tolist())) >= 3
    low, high = {}, {}
    for k in HAND_KS:
        rows = _rows(hand, k)
        for h in H.REPORTED_H:
            # At the draw count the tests run at AND at the shipped one: what
            # the fixture is shown to discriminate at is what it is read at.
            for draws in (SMALL, 2000):
                hd = _interval(rows["headroom"], HAND_GROUPS, h, resamples=draws)
                if draws == SMALL:
                    low[(k, h)], high[(k, h)] = hd["ci_low"], hd["ci_high"]
                if (k, h) == (H.DECISION_K, H.DECISION_H):
                    for name in rows:
                        cell = _interval(rows[name], HAND_GROUPS, h, resamples=draws)
                        assert cell["ci_low"] < cell["ci_high"], name
                point = rows["headroom"][:, h - 1].mean()
                assert (point < 0.0) == ([k, h] in NEGATIVE_STEPS), (k, h)
                assert (point == 0.0) == ((k, h) == ZERO_CELL), (k, h)
                unresolved = point > 0.0 and hd["ci_low"] <= 0.0
                assert unresolved == ((k, h) == UNRESOLVED_CELL), (k, h, draws)
    assert any(low[key] > 0.0 for key in low), "some cell must pass the gate"
    assert np.all(RegroundingSweep.standard_error(_rows(hand, 1)["skill"]) > 0.0)
    # The two controls whose fixtures are rows the sweep itself checks.
    assert hand.persistence_divergence() == 0.0
    assert hand.k_invariance_at_h1() == 0.0


def test_every_interval_is_the_clustered_interval_of_its_own_rows_at_every_k_and_h(
    hand, monkeypatch,
):
    """Each of the three differences at every (k, h), recomputed here from the
    hand-built rows by `pooling.clustered_interval` at THE CELL'S DERIVED seed and
    the draw count the record states, and compared with `==`.

    The cell's seed (3), 0, the run's base seed (7) and the derived seed are four
    different draws, and so are `SMALL` and the shipped 2000: the reached check
    below shows the bounds move under each, so an equality that survives is the
    interval read from the right seed, the right count, the right rows and the
    right clusters.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `seed=cell_seed` ->
    `seed=bootstrap_seed` (the base, which is what every cell drew at before the
    derivation), `-> seed=seed` and `-> seed=0` in the `clustered_interval` call,
    `h=h` -> `h=1`, and `groups` -> `np.arange(n)`.
    """
    record = _hand_measure(monkeypatch, copy.deepcopy(hand))
    assert record["resamples"] == SMALL
    assert set(record["intervals"]) == {str(k) for k in HAND_KS}
    reached = {"seed": False, "zero": False, "base": False, "draws": False, "groups": False}
    for k in HAND_KS:
        rows = _rows(hand, k)
        assert set(record["intervals"][str(k)]) == {str(h) for h in H.REPORTED_H}
        for h in H.REPORTED_H:
            carried = record["intervals"][str(k)][str(h)]
            assert set(carried) == {"headroom", "skill", "deficit"}
            for name, array in rows.items():
                expected = _interval(array, HAND_GROUPS, h)
                assert carried[name] == expected, (k, h, name)
                moved = lambda **over: _interval(array, HAND_GROUPS, h, **over) != expected
                reached["seed"] |= moved(seed=SEED)
                reached["zero"] |= moved(seed=0)
                reached["base"] |= moved(seed=BOOTSTRAP_SEED)
                reached["draws"] |= moved(resamples=2000)
                reached["groups"] |= (
                    pooling.clustered_interval(
                        array, np.arange(len(HAND_GROUPS)), h=h, resamples=SMALL,
                        seed=_cell_seed(),
                    )[1:] != (expected["ci_low"], expected["ci_high"])
                )
    assert all(reached.values()), reached


def test_the_share_is_skill_over_headroom_only_where_the_gate_passed(hand, monkeypatch):
    """`None` where `headroom`'s interval is not above zero, and the quotient of
    the two POINT estimates where it is.

    The gate is the INTERVAL's lower bound, not the point. At `UNRESOLVED_CELL`
    the headroom point is positive (a gate on the point passes it and prints a
    share of a quantity indistinguishable from zero); at `ZERO_CELL` it is
    exactly 0.0 (a gate on `>= 0` divides by zero). Both are in the fixture, and
    the expected value is read off the rows, not off the record's own intervals.

    THE MUTATIONS THIS EXISTS FOR, each run against it: the gate `hd["ci_low"] >
    0.0` -> `hd["point"] > 0.0`, and `-> hd["ci_low"] >= 0.0`.
    """
    record = _hand_measure(monkeypatch, copy.deepcopy(hand))
    numbers = nones = 0
    for k in HAND_KS:
        rows = _rows(hand, k)
        for h in H.REPORTED_H:
            carried = record["share"][str(k)][str(h)]
            gate = _interval(rows["headroom"], HAND_GROUPS, h)["ci_low"] > 0.0
            if gate:
                expected = rows["skill"][:, h - 1].mean() / rows["headroom"][:, h - 1].mean()
                assert carried == expected, (k, h)
                numbers += 1
            else:
                assert carried is None, (k, h)
                nones += 1
    assert numbers > 0 and nones >= 4  # three negative, one unresolved, one zero cell
    assert record["share"][str(UNRESOLVED_CELL[0])][str(UNRESOLVED_CELL[1])] is None
    assert record["share"][str(ZERO_CELL[0])][str(ZERO_CELL[1])] is None


def test_negative_headroom_steps_lists_every_k_h_below_zero_in_loop_order(hand, monkeypatch):
    """The (k, h) cells whose headroom POINT is below zero, k-major and h
    ascending -- compared with the literal list the fixture was built to give.
    `ZERO_CELL` is exactly 0.0 and must NOT be listed.

    THE MUTATION THIS EXISTS FOR: `hd["point"] < 0.0` -> `<= 0.0`.
    """
    record = _hand_measure(monkeypatch, copy.deepcopy(hand))
    assert record["controls"]["negative_headroom_steps"] == NEGATIVE_STEPS
    assert list(ZERO_CELL) not in record["controls"]["negative_headroom_steps"]


def test_the_secondary_figure_is_the_recorded_multiple_of_one_standard_error(
    hand, monkeypatch,
):
    """`SECONDARY_SIGMAS` times `RegroundingSweep.standard_error` of each
    difference's rows, over every horizon step -- ONE standard error, doubled by
    the constant and by nothing else. The constant is patched to 3, so a literal
    2 and a literal 1 are both different numbers, and the expected value is
    computed here from the rows (`std(ddof=1) / sqrt(n)`), not by calling
    `standard_error`.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `SECONDARY_SIGMAS *
    standard_error(...)` -> `2 * standard_error(...)`, and -> `standard_error(...)`
    alone.
    """
    monkeypatch.setattr(script, "SECONDARY_SIGMAS", 3)
    record = _hand_measure(monkeypatch, copy.deepcopy(hand))
    assert list(record["secondary"]) == ["iid_2se"]
    for k in HAND_KS:
        for name, array in _rows(hand, k).items():
            one = array.std(axis=0, ddof=1) / np.sqrt(array.shape[0])
            # Reached: a nonzero ruler at every step but the one the fixture
            # makes exactly flat (`ZERO_CELL`'s headroom is 0.0 in every window).
            flat = 1 if (k, name) == (ZERO_CELL[0], "headroom") else 0
            assert np.count_nonzero(one == 0.0) == flat, (k, name)
            carried = record["secondary"]["iid_2se"][str(k)][name]
            assert len(carried) == HORIZON
            np.testing.assert_allclose(carried, 3.0 * one, rtol=1e-12, atol=0.0)


def test_each_control_is_measured_from_the_object_it_names_not_asserted(hand, monkeypatch):
    """Four controls with a known answer of exactly 0.0, each made nonzero by
    ONE perturbation of known size so the record can be asked for the size.

      persistence_divergence  the k=45 hold curve moved by 0.75 at one step
      k_invariance_at_h1      one window's k=45 hold at h=1 moved by 0.5
      open_loop_divergence    the VERIFIED reference's rssm curve moved by 0.25
                              (position) and 0.5 (angle): the larger wins
      floor_divergence        the VERIFIED reference's floor moved by 0.125

    The sizes are binary fractions, so each is read back with `==`. A control
    hard-coded to 0.0, or read off the wrong object, reads 0.0 here.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `"floor_divergence"`
    hard-coded to `0.0`; computed as the sweep's own floor against itself; and
    `"open_loop_divergence"` read against the sweep's reference instead of the
    one handed in.
    """
    arrays = _hand_arrays()
    sweep = _hand_sweep(arrays)
    sweep.hold_position[45] = sweep.hold_position[45].copy()
    sweep.hold_position[45][6] += 0.75
    sweep.window_hold_position[45] = arrays["hold"][45].copy()
    sweep.window_hold_position[45][3, 0] += 0.5
    verified = copy.deepcopy(sweep.reference)
    verified.persistence_position = sweep.reference.persistence_position
    verified.rssm_position = verified.rssm_position.copy()
    verified.rssm_position[4] += 0.25
    verified.rssm_angle = verified.rssm_angle.copy()
    verified.rssm_angle[9] += 0.5
    verified.floor_position = verified.floor_position.copy()
    verified.floor_position[11] += 0.125
    # `persistence_divergence` compares `hold_position[45]` with the SWEEP's own
    # reference, which still carries the unperturbed curve: reached below.
    assert sweep.persistence_divergence() == 0.75
    assert sweep.k_invariance_at_h1() == 0.5
    record = _hand_measure(monkeypatch, sweep, reference=verified)
    controls = record["controls"]
    assert controls["persistence_divergence"] == 0.75
    assert controls["k_invariance_at_h1"] == 0.5
    assert controls["open_loop_divergence"] == 0.5
    assert controls["floor_divergence"] == 0.125


def test_the_clean_sweep_reads_every_control_at_exactly_its_known_answer(hand, hand_record):
    """The four zeros and the identity, from a sweep built to satisfy them."""
    controls = hand_record["controls"]
    for name in ZERO_CONTROL_NAMES:
        assert controls[name] == 0.0, name
    assert controls["identity_residual"] == 0.0
    assert script.broken_controls(controls) == []


def test_the_identity_residual_is_judged_against_the_modules_own_tolerance(
    hand_record, monkeypatch,
):
    """`broken_controls` holds the residual to `IDENTITY_TOLERANCE` -- the name the
    module holds, so a patched value moves the line. With it at 1e-5 a residual of
    1e-6 passes (it would fail against a literal 1e-09) and one of 1e-4 does not.

    THE MUTATION THIS EXISTS FOR, run against it: `IDENTITY_TOLERANCE` -> the
    literal `1e-09` in `broken_controls`."""
    monkeypatch.setattr(script, "IDENTITY_TOLERANCE", 1e-5)
    controls = copy.deepcopy(hand_record["controls"])
    controls["identity_residual"] = 1e-6
    assert script.broken_controls(controls) == []
    controls["identity_residual"] = 1e-4
    assert len(script.broken_controls(controls)) == 1


def test_the_protocol_check_reads_the_modules_own_decision_cell_and_grid(hand, monkeypatch):
    """The refusal before any pass is paid for reads `DECISION_K`, `DECISION_H` and
    `REPORTED_H` from the module, not as literals: each is patched to a value the
    protocol then cannot satisfy, with the sweep replaced by one that raises if it
    is reached.

    THE MUTATIONS THIS EXISTS FOR, each run against it, one per case: `DECISION_K`
    -> `1` in the `ks` check; `*REPORTED_H` -> `45` and `DECISION_H` -> `1` in the
    horizon check."""
    def unreachable(*args, **kwargs):
        raise AssertionError("the sweep was run for a protocol that cannot be read")

    monkeypatch.setattr(script, "regrounding_sweep", unreachable)
    for name, value, said in (
        ("DECISION_K", 2, "omits k=2"),
        ("REPORTED_H", (1, 60), "up to 60"),
        ("DECISION_H", 50, "DECISION_H=50"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(script, name, value)
            with pytest.raises(SystemExit) as caught:
                script.measure_cell(**_cell_kwargs(reference=copy.deepcopy(hand.reference)))
        assert said in str(caught.value), (name, str(caught.value))


def test_the_identity_residual_is_the_largest_over_the_ks_not_the_first_the_last_or_the_sum(
    hand, monkeypatch,
):
    """`skill` is faulted on the three `triple_residual` calls -- one per k, in
    the order the ks are given -- by 0.25, then 0.5 (negative), then 0.125. The
    middle one is the largest, so the record must read exactly 0.5: the first
    reads 0.25, the last 0.125, the sum 0.875 and the mean 0.2916...

    The fault goes in through the one place a real plumbing fault would, one of
    the three differences returning a wrong number at one step, as
    `test_headroom.py` does. `headroom.skill` is patched inside the `headroom`
    module, which is the name `triple_residual` calls; the rows `measure_cell`
    differences itself go through the script's own binding and are untouched.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `float(np.max(residuals))`
    -> `float(np.sum(residuals))`, and -> `residuals[-1]`.
    """
    calls = []
    real = H.skill
    bumps = (0.25, -0.5, 0.125)

    def faulted(hold, rung):
        bump = np.zeros(hold.shape)
        bump[2, 5] = bumps[len(calls)]
        calls.append(None)
        return real(hold, rung) + bump

    monkeypatch.setattr(H, "skill", faulted)
    record = _hand_measure(monkeypatch, copy.deepcopy(hand))
    assert len(calls) == len(HAND_KS)
    assert record["controls"]["identity_residual"] == 0.5


def test_a_nan_identity_residual_at_any_k_is_not_swallowed_by_the_reduction(hand, monkeypatch):
    """The reduction over the ks is `np.max`, which PROPAGATES a NaN. Python's
    `max` does not: `max([0.25, nan, 0.125])` is 0.25, because every comparison
    with NaN is False, so a NaN in the middle of the list is skipped and the
    residual reads as passing. The brief's `max(residuals)` was that.

    The fault is a NaN on the middle k's `triple_residual` call (the one place
    rows reach it finite is the fault itself: `checked_pair` refuses a NaN row,
    so this is the only way to put one in), and the record must read NaN -- which
    `broken_controls` then names.

    THE MUTATION THIS EXISTS FOR, run against it: `float(np.max(residuals))` ->
    `float(max(residuals))`."""
    calls = []
    real = H.skill
    bumps = (0.25, float("nan"), 0.125)

    def faulted(hold, rung):
        bump = np.zeros(hold.shape)
        bump[2, 5] = bumps[len(calls)]
        calls.append(None)
        return real(hold, rung) + bump

    monkeypatch.setattr(H, "skill", faulted)
    record = _hand_measure(monkeypatch, copy.deepcopy(hand))
    assert len(calls) == len(HAND_KS)
    assert np.isnan(record["controls"]["identity_residual"])
    assert script.broken_controls(record["controls"]) == ["identity_residual=nan"]


def test_the_record_carries_the_verified_reference_and_the_sweeps_own_rungs_and_holds(
    hand, monkeypatch,
):
    """The three canonical position curves are the VERIFIED reference's, which
    `prepare_cell` proved reproduces the study record; the rungs and the holds
    are the sweep's own. The reference handed in carries curves the sweep's does
    not, so a record that read them off `sweep.reference` is a different number.

    THE MUTATION THIS EXISTS FOR: `reference.floor_position` ->
    `sweep.reference.floor_position` (and the same for the other two curves)
    in the record's `curves`.
    """
    verified = copy.deepcopy(hand.reference)
    steps = np.arange(1, HORIZON + 1, dtype=np.float64)
    verified.rssm_position = 1000.5 + 0.125 * steps
    verified.persistence_position = 2000.25 + 0.0625 * steps
    verified.floor_position = 3000.75 + 0.25 * steps
    record = _hand_measure(monkeypatch, copy.deepcopy(hand), reference=verified)
    curves = record["curves"]
    assert curves["rssm_position"] == verified.rssm_position.tolist()
    assert curves["persistence_position"] == verified.persistence_position.tolist()
    assert curves["floor_position"] == verified.floor_position.tolist()
    assert curves["rssm_position"] != hand.reference.rssm_position.tolist()
    assert set(curves["rungs"]) == set(curves["holds"]) == {str(k) for k in HAND_KS}
    for k in HAND_KS:
        assert curves["rungs"][str(k)] == hand.position[k].tolist()
        assert curves["holds"][str(k)] == hand.hold_position[k].tolist()
    assert len({tuple(curves["holds"][str(k)]) for k in HAND_KS}) == len(HAND_KS)
    assert len({tuple(curves["rungs"][str(k)]) for k in HAND_KS}) == len(HAND_KS)


# (module name, patched value, record key, value the record must carry)
CONSTANTS = [
    ("CONFIDENCE", 0.9, "confidence", 0.9),
    ("RESAMPLES", 7, "resamples", 7),
    ("DECISION_H", 2, "decision_h", 2),
    ("DECISION_K", 3, "decision_k", 3),
    ("IDENTITY_TOLERANCE", 1e-5, "identity_tolerance", 1e-5),
    ("SECONDARY_SIGMAS", 3, "secondary_sigmas", 3),
    ("REPORTED_H", (1, 2, 45), "reported_h", [1, 2, 45]),
    ("SPLIT_SEED", 5, "split_seed", 5),
    ("DISPLACEMENT_RECORDED_ONLY", 1.5, "displacement_median", 1.5),
]


@pytest.mark.parametrize("name, patched, key, expected", CONSTANTS, ids=[c[0] for c in CONSTANTS])
def test_each_recorded_constant_moves_when_the_modules_own_name_moves(
    hand, monkeypatch, name, patched, key, expected,
):
    """The record states the constant THE MODULE HOLDS. Comparing a record with
    `headroom.CONFIDENCE` cannot tell a read from a literal -- a literal `0.95`
    EQUALS it -- so the module's name is patched to a value the shipped constant
    is not, and the record must follow it. One constant per case: a combined
    patch would let a literal in one place hide behind the right value in another.

    Five record constants were literals surviving all 50 tests in M3m.

    THE MUTATIONS THIS EXISTS FOR, one per case, each a literal where the
    constant stood: `"confidence": 0.95`, `"resamples": 2000`, `"decision_h": 1`,
    `"decision_k": 1`, `"identity_tolerance": 1e-9`, `"secondary_sigmas": 2`. The
    other three are the same defect in the three record fields that are not one of
    the milestone's six. `RESAMPLES` and `REPORTED_H` also drive the measurement,
    so `intervals` follows them too (checked below).
    """
    assert getattr(script, name) != patched
    monkeypatch.setattr(script, name, patched)
    record = _hand_measure(monkeypatch, copy.deepcopy(hand), small=(name != "RESAMPLES"))
    assert record[key] == expected
    if name == "REPORTED_H":
        assert all(set(by_h) == {"1", "2", "45"} for by_h in record["intervals"].values())
        assert all(set(by_h) == {"1", "2", "45"} for by_h in record["share"].values())


def test_the_recorded_confidence_is_the_level_the_estimator_takes():
    """`CONFIDENCE` describes `pooling.percentile_interval`; it does not configure
    it, and the record states it beside every interval. `test_burden.py` pins
    `burden.CONFIDENCE` against the estimator and nothing pinned `headroom`'s, so a
    change to the percentiles would leave every M3n record naming a level the
    estimator did not take -- the defect M3l shipped and then fixed.

    THE MUTATION THIS EXISTS FOR, run against it: `[2.5, 97.5]` -> `[5.0, 95.0]` in
    `pooling.percentile_interval`."""
    replicates = np.arange(10001, dtype=np.float64)
    low, high, _ = pooling.percentile_interval(replicates)
    tail = (1.0 - H.CONFIDENCE) / 2.0
    assert low == pytest.approx(np.percentile(replicates, 100 * tail))
    assert high == pytest.approx(np.percentile(replicates, 100 * (1 - tail)))
    assert (low, high) == (250.0, 9750.0)


def test_the_interval_is_drawn_at_the_draw_count_the_record_states(hand, monkeypatch):
    """The record's `resamples` is what the bootstrap USED. With the constant at
    7 the interval equals the recomputation at 7 draws and not at 2000.

    THE MUTATION THIS EXISTS FOR: `resamples=RESAMPLES` -> `resamples=2000` in
    the `clustered_interval` call, with the record still reading `RESAMPLES`:
    a record that states one count and draws another is exactly what a permanent
    artefact cannot be audited for.
    """
    monkeypatch.setattr(script, "RESAMPLES", 7)
    record = _hand_measure(monkeypatch, copy.deepcopy(hand), small=False)
    rows = _rows(hand, 1)["headroom"]
    carried = [record["intervals"]["1"][str(h)]["headroom"] for h in H.REPORTED_H]
    assert carried == [_interval(rows, HAND_GROUPS, h, resamples=7) for h in H.REPORTED_H]
    assert carried != [_interval(rows, HAND_GROUPS, h, resamples=2000) for h in H.REPORTED_H]


def test_the_record_carries_the_training_numbers_and_provenance_of_its_own_cell(hand, monkeypatch):
    study = {
        "steps": 12345, "kl_rate_above_free_bits": 0.25, "kl_dyn_max": 7.5,
        "git_sha": "study-sha",
    }
    record = _hand_measure(monkeypatch, copy.deepcopy(hand), study_record=study)
    assert record["step"] == 12345
    assert record["kl_rate_above_free_bits"] == 0.25
    assert record["kl_dyn_max"] == 7.5
    assert record["record_git_sha"] == "study-sha"
    assert record["git_sha"] == git_sha()
    assert record["torch_version"] == torch.__version__
    assert record["device"] == "cpu"
    assert (record["arm"], record["seed"]) == ("pixel_ae", SEED)
    assert record["bootstrap_seed"] == BOOTSTRAP_SEED
    assert record["cell_bootstrap_seed"] == _cell_seed(("pixel_ae", SEED))
    assert record["cell_bootstrap_seed"] != BOOTSTRAP_SEED
    assert (record["context"], record["horizon"]) == (CONTEXT, HORIZON)
    assert record["split_seed"] == SPLIT_SEED
    assert record["ks"] == list(HAND_KS)
    assert record["reported_h"] == [1, 2, 3, 5, 8, 10, 15, 20, 30, 45]
    assert (record["decision_k"], record["decision_h"]) == (1, 1)
    assert (record["confidence"], record["resamples"]) == (0.95, SMALL)
    assert record["identity_tolerance"] == 1e-09 and record["secondary_sigmas"] == 2
    assert record["displacement_median"] == 3.9694722203504225


@pytest.mark.parametrize(
    "missing", ["steps", "kl_rate_above_free_bits", "kl_dyn_max", "git_sha"],
)
def test_a_study_record_missing_a_training_number_is_refused_not_defaulted(
    hand, monkeypatch, missing,
):
    """ONE KEY LEFT OUT PER CASE, so each strict read is pinned by itself.
    `git_sha` is one of them because the read phase compares it across records as
    a protocol field: a default of "unknown" is a value `study.git_sha()` itself
    writes when git cannot answer, so a record that merely LACKS the key would be
    indistinguishable from one written where git was unavailable.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `study_record["steps"]`
    -> `study_record.get("steps", 0)`, and `study_record["git_sha"]` ->
    `study_record.get("git_sha", "unknown")`."""
    study = dict(STUDY_RECORD)
    del study[missing]
    with pytest.raises(KeyError, match=missing):
        _hand_measure(monkeypatch, copy.deepcopy(hand), study_record=study)


def test_the_record_names_its_episodes_and_clusters_its_windows_by_the_sweeps_own_labels(
    hand, hand_record,
):
    """`windows.episode` is the sweep's `window_episode` copied through, in the
    order it walked, and `episodes.val` the validation paths in the order given
    (descending file names here, so `sorted()` would be a different list).
    Nothing is recovered by a second walk.

    THE MUTATION THIS EXISTS FOR: `groups = np.arange(n)` in `measure_cell`, which
    would make every window its own cluster -- here 13 labels where there are 6.
    """
    assert hand_record["windows"]["episode"] == HAND_GROUPS.tolist()
    assert hand_record["windows"]["total"] == len(HAND_GROUPS)
    assert hand_record["episodes"]["val"] == HAND_NAMES
    assert hand_record["episodes"]["val"] != sorted(HAND_NAMES)
    assert len(set(hand_record["windows"]["episode"])) == 6


def test_the_record_has_the_brief_schema_less_the_key_write_record_owns(hand_record):
    """The brief's schema ends `"nonfinite": 0`. `study.write_record` owns the
    top-level key `nonfinite` -- it is the map from the dotted path of each value
    it nulled to the token it came from -- and RAISES for a record that already
    carries one, so a count under that name would have crashed the first write of
    the real run, after the GPU time was spent. The record carries every other key
    of the schema, and the one on disk is the sanitiser's, `{}`
    (`test_the_record_survives_a_round_trip_through_write_record`).

    THE MUTATION THIS EXISTS FOR: putting `"nonfinite": <anything>` back into the
    record."""
    assert "nonfinite" not in hand_record
    assert set(hand_record) == {
        "arm", "seed", "step", "git_sha", "record_git_sha", "torch_version", "device",
        "context", "horizon", "ks", "reported_h", "decision_k", "decision_h",
        "confidence", "resamples", "identity_tolerance", "secondary_sigmas",
        "bootstrap_seed", "cell_bootstrap_seed", "split_seed", "displacement_median",
        "episodes", "windows", "curves", "intervals", "share", "secondary", "controls",
        "kl_dyn_max", "kl_rate_above_free_bits",
    }
    assert set(hand_record["curves"]) == {
        "floor_position", "persistence_position", "rssm_position", "rungs", "holds",
    }
    assert set(hand_record["controls"]) == {
        "persistence_divergence", "k_invariance_at_h1", "open_loop_divergence",
        "floor_divergence", "identity_residual", "negative_headroom_steps",
    }


def test_the_record_survives_a_round_trip_through_write_record(hand_record, tmp_path):
    """A record that is not identical before and after the write is one the read
    phase would read differently from what was measured -- most treacherously
    through the keys, which JSON turns into strings (they are strings already)."""
    path = tmp_path / "headroom.json"
    written = write_record(path, hand_record)
    assert written["nonfinite"] == {}
    assert load_record(path) == {**hand_record, "nonfinite": {}}


@pytest.mark.parametrize("where", ["hold", "rung", "floor"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_measure_cell_refuses_a_non_finite_row_by_name_before_any_interval(
    hand, monkeypatch, where, bad,
):
    """The per-window rows reach `headroom`, `skill`, `deficit` and the bootstrap;
    each of those raises a bare ValueError that names neither the cell nor the
    array. A row poisoned in ONE array -- the hold, the rung or the floor -- is
    refused here, naming the cell, with nothing written.

    The poisoned value sits in a COLUMN `clustered_interval` is not asked about at
    h=1 and not on the decision path, so the check has to read the whole row.

    THE MUTATION THIS EXISTS FOR: deleting the non-finite refusal (the call then
    raises ValueError, not SystemExit).
    """
    arrays = _hand_arrays()
    poisoned = copy.deepcopy(arrays)
    target = {"hold": poisoned["hold"][3], "rung": poisoned["rung"][3], "floor": poisoned["floor"]}[where]
    target[4, 33] = bad
    sweep = _hand_sweep(arrays)
    sweep = replace(
        sweep, window_hold_position=poisoned["hold"], window_position=poisoned["rung"],
        window_floor_position=poisoned["floor"],
    )
    with pytest.raises(SystemExit) as caught:
        _hand_measure(monkeypatch, sweep)
    message = str(caught.value)
    assert "pixel_ae seed 3" in message and "non-finite" in message, message


@pytest.mark.parametrize(
    "horizon, ks, said",
    [
        (30, HAND_KS, "horizon of 30"),
        (HORIZON, (3, 45), "omits k=1"),
        (HORIZON, (1, 3), "omits the horizon"),
    ],
)
def test_measure_cell_refuses_a_protocol_the_reading_cannot_be_taken_from(
    hand, monkeypatch, horizon, ks, said,
):
    """Refused BY NAME before any pass is paid for. Asked afterwards, the first is
    an IndexError out of the 45-step interval, the second a KeyError out of the
    decision cell's intervals and the third a ValueError out of the sweep.

    The sweep is replaced by one that raises if it is reached, so the refusal is
    shown to come first.

    THE MUTATION THIS EXISTS FOR: deleting `_require_a_readable_protocol(...)`.
    """
    def unreachable(*args, **kwargs):
        raise AssertionError("the sweep was run for a protocol that cannot be read")

    monkeypatch.setattr(script, "regrounding_sweep", unreachable)
    with pytest.raises(SystemExit) as caught:
        script.measure_cell(**_cell_kwargs(
            reference=copy.deepcopy(hand.reference), horizon=horizon, ks=ks,
        ))
    assert said in str(caught.value)


# ---------------------------------------------------------------------------
# measure_cell over the REAL sweep: what reaches it, and the closed form.
# ---------------------------------------------------------------------------


def _wobbling_episode(length: int, index: int):
    """`synthetic_episode`'s straight run, plus a wobble phased by `index`. On a
    straight line every window has the same displacement and every difference is
    one constant per step; the wobble gives each window its own."""
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
    episode: not exact on a wobbling agent, which is the point."""
    tags = np.concatenate(
        [np.arange(e.privileged.shape[0], dtype=np.float64)[:, None] for e in episodes]
    )
    targets = np.concatenate([probe_targets(e.privileged, KEYS) for e in episodes])
    return fit_probe(tags, targets, ridge=1e-8)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    """Nine episodes whose window counts are 1, 2, 0, 1, 3, 2, 1, 2, 1 (the zero
    is length 50 == `context + horizon`), with DESCENDING file names so the order
    the paths are given in is not the order `sorted()` would put them in."""
    directory = tmp_path_factory.mktemp("headroom_rig")
    episodes = [_wobbling_episode(n, i) for i, n in enumerate(LENGTHS)]
    paths = [
        _episode_file(directory, e, len(LENGTHS) - 1 - i) for i, e in enumerate(episodes)
    ]
    assert paths != sorted(paths), "the fixture must distinguish sorted() from the given order"
    assert tuple(len(window_starts(n, CONTEXT, HORIZON)) for n in LENGTHS) == EXPECTED_WINDOWS
    return types.SimpleNamespace(paths=paths, episodes=episodes, probe=_linear_probe(episodes))


class _StochasticRSSM(_OracleRSSM):
    """`_OracleRSSM` whose `imagine` adds a standard normal draw to every step,
    drawn ONE STEP AT A TIME, as a real RSSM samples its categorical latent.

    THE FAKE THE OTHER MODELS CANNOT BE. An oracle-family model draws no random
    number, so a sweep handed `seed=0` and one handed the cell's seed return the
    same curves bit for bit, and nothing downstream can tell which it was given.
    A test that shows an argument was PASSED shows nothing about whether it was
    USED.

    The draw is per step, not one tensor for the whole segment, and that is
    load-bearing for `k_invariance_at_h1`: `torch.randn` of a `(1, 45, 1)` tensor
    and of a `(1, 1, 1)` one do not begin with the same number (the vectorised
    path starts at 16 elements), so a single draw per call makes the first step
    differ between k=1 and k=45 -- a fake that FAILS a control the real model
    passes, which is how an earlier version of this fixture read 14.87 on it."""

    def imagine(self, actions, state):
        tag = super().imagine(actions, state)["h"]
        noise = torch.cat(
            [torch.randn(tag.shape[0], 1, tag.shape[2]) for _ in range(tag.shape[1])], dim=1
        )
        tag = tag + noise
        return {"h": tag, "z": tag, "latent": torch.cat([tag, tag], dim=-1)}


class StochasticModel(OracleModel):
    def __init__(self) -> None:
        super().__init__()
        self.rssm = _StochasticRSSM()


def _reference(rig, model, *, feature_backbone=None, seed=SEED):
    """The rollout `prepare_cell` would hand back for `model`: the real
    `evaluate_rollout`, at the cell's seed and through the cell's probe."""
    return evaluate_rollout(
        model, rig.paths, rig.probe, context=CONTEXT, horizon=HORIZON, seed=seed,
        device=CPU, feature_backbone=feature_backbone,
    )


def _real_kwargs(rig, **over) -> dict:
    kwargs = _cell_kwargs(
        model=DriftingModel(), val_paths=rig.paths, probe=rig.probe,
        ks=REGROUNDING_KS,
    )
    kwargs.update(over)
    if "reference" not in over:
        kwargs["reference"] = _reference(
            rig, kwargs["model"], feature_backbone=kwargs["feature_backbone"],
        )
    return kwargs


def _independent_sweep(rig, model=None, *, seed=SEED, ks=REGROUNDING_KS):
    """The real sweep, called again by the TEST, so what the record's numbers are
    compared with was not produced by the code under test."""
    return regrounding_sweep(
        model or DriftingModel(), rig.paths, rig.probe, ks=ks, context=CONTEXT,
        horizon=HORIZON, seed=seed, device=CPU, feature_backbone=None,
    )


def _expected_rows(rig, k):
    """`hold`, `rung` and `floor` of `DriftingModel` over the rig, window by
    window, derived WITHOUT the sweep, from the rig's linear probe over the
    episodes' frame tags.

    `DriftingModel` advances the frame tag by TWO per action. A window cut at
    `start` scores horizon step `h` on frame `f = start + context + h`, and the
    state a segment is imagined from was grounded at `g = ground_step(k, h)`
    steps in, i.e. at the tag `start + context + g`. So

        floor(h) = |probe(f)            - true(f)|
        hold(h)  = |probe(start+ctx+g)  - true(f)|
        rung(h)  = |probe(start+ctx+g + 2 * (h - g)) - true(f)|

    in the probe's first two columns (`pos_x`, `pos_y`). Every window of every
    episode, in the order the paths are GIVEN and `window_starts` within each.
    """
    hold, rung, floor = [], [], []
    steps = np.arange(1, HORIZON + 1)
    ground = np.array([ground_step(k, int(h)) for h in steps])

    def probed(tags):
        return apply_probe(rig.probe, np.asarray(tags, dtype=np.float64)[:, None])[:, :2]

    for path in rig.paths:
        episode = load_episode(path)
        for start in window_starts(episode.length, CONTEXT, HORIZON):
            frames = start + CONTEXT + steps
            true = episode.privileged[frames][:, 1:3].astype(np.float64)
            hold.append(np.linalg.norm(probed(start + CONTEXT + ground) - true, axis=1))
            rung.append(np.linalg.norm(
                probed(start + CONTEXT + ground + 2 * (steps - ground)) - true, axis=1))
            floor.append(np.linalg.norm(probed(frames) - true, axis=1))
    return np.stack(hold), np.stack(rung), np.stack(floor)


def _expected_labels() -> list[int]:
    """One label per window, in traversal order: the index, among the episodes
    that HAVE a window, of the episode it was cut from."""
    labels, index = [], 0
    for length in LENGTHS:
        count = len(window_starts(length, CONTEXT, HORIZON))
        if count:
            labels += [index] * count
            index += 1
    return labels


@pytest.fixture(scope="module")
def real_record(rig):
    """The record at the shipped constants -- 2000 resamples and all five rungs."""
    return script.measure_cell(**_real_kwargs(rig))


def test_the_rig_discriminates_hold_rung_and_floor_window_by_window(rig):
    """The closed form is not a constant, and the three differences are not
    degenerate: windows differ, the rungs differ across k, and at the decision
    cell every difference has a spread across episodes to bootstrap."""
    hold, rung, floor = _expected_rows(rig, 1)
    assert hold.shape == (sum(EXPECTED_WINDOWS), HORIZON)
    assert len({tuple(row) for row in floor}) == hold.shape[0]
    assert len({tuple(_expected_rows(rig, k)[1][0]) for k in REGROUNDING_KS}) == len(REGROUNDING_KS)
    labels = np.asarray(_expected_labels())
    assert len(set(labels.tolist())) >= 3
    for rows in (hold - floor, hold - rung, rung - floor):
        assert rows[:, 0].std() > 0.0
    point, low, high = pooling.clustered_interval(
        hold - rung, labels, h=1, resamples=SMALL, seed=_cell_seed(),
    )
    assert low < high


def test_the_real_sweeps_rows_are_the_closed_form_and_the_record_carries_them(
    rig, real_record,
):
    """Per window, per k, `hold`, `rung` and `floor` from the real sweep equal the
    closed form -- which reads the probe and the episodes and nothing else -- and
    the record's curves are the means of those rows.

    This is the check the hand-built sweep cannot make: that `measure_cell` is
    handed rows for the right windows in the right order, by VALUE. A reorder that
    keeps the shape (two equal-count episodes swapped; `sorted()` on names whose
    block sizes happen to agree) passes any shape check and moves these numbers.
    """
    sweep = _independent_sweep(rig)
    floor_rows = None
    for k in REGROUNDING_KS:
        hold, rung, floor = _expected_rows(rig, k)
        np.testing.assert_allclose(sweep.window_hold_position[k], hold, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(sweep.window_position[k], rung, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(sweep.window_floor_position, floor, rtol=1e-6, atol=1e-9)
        np.testing.assert_array_equal(real_record["curves"]["holds"][str(k)], sweep.hold_position[k])
        np.testing.assert_array_equal(real_record["curves"]["rungs"][str(k)], sweep.curve(k))
        np.testing.assert_allclose(
            real_record["curves"]["holds"][str(k)], hold.mean(axis=0), rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(
            real_record["curves"]["rungs"][str(k)], rung.mean(axis=0), rtol=1e-6, atol=1e-9)
        floor_rows = floor
    assert real_record["windows"]["total"] == sum(EXPECTED_WINDOWS)
    assert real_record["episodes"]["val"] == [p.name for p in rig.paths]
    # Labels agree as PARTITIONS, which is all clustering reads.
    carried = np.asarray(real_record["windows"]["episode"])
    expected = np.asarray(_expected_labels())
    np.testing.assert_array_equal(
        np.unique(carried, return_inverse=True)[1], np.unique(expected, return_inverse=True)[1],
    )
    assert floor_rows is not None


def test_every_real_interval_is_drawn_from_the_sweeps_rows_at_the_stated_seed(
    rig, real_record,
):
    """The 150 intervals of the shipped run -- five rungs, ten horizons, three
    differences -- recomputed from a second sweep the test ran, at the shipped
    2000 draws and the cell's derived seed. Compared with `==`: same rows, same
    seed, same draws, same bits."""
    assert real_record["resamples"] == H.RESAMPLES == 2000
    sweep = _independent_sweep(rig)
    for k in REGROUNDING_KS:
        for name, rows in _rows(sweep, k).items():
            for h in H.REPORTED_H:
                assert real_record["intervals"][str(k)][str(h)][name] == _interval(
                    rows, sweep.window_episode, h, resamples=2000,
                ), (k, h, name)


def test_the_real_rig_reads_every_control_at_its_known_answer(real_record):
    """Four zeros the sweep is built to satisfy, and an identity at rounding.

    `identity_residual` is a MEASURED float, not a constant: on the rig it is
    rounding noise, below `IDENTITY_TOLERANCE`, and not hard-coded to 0.0."""
    controls = real_record["controls"]
    for name in ZERO_CONTROL_NAMES:
        assert controls[name] == 0.0, name
    assert 0.0 <= controls["identity_residual"] < H.IDENTITY_TOLERANCE
    assert script.broken_controls(controls) == []


def test_the_sweep_is_run_at_the_cells_seed(rig, monkeypatch):
    """`measure_cell` must hand `regrounding_sweep` the CELL'S seed. The reference
    it is given was rolled out at that seed, so the k=horizon rung reproduces it
    bitwise ONLY if the sweep drew the same stream. Every rung is also asked by
    value against a second sweep the test ran at the cell's seed.

    THE MUTATION THIS EXISTS FOR: `seed=seed` -> `seed=0` in the call to
    `regrounding_sweep`. On correct code the `open_loop_divergence` control would
    flag it at run time, in the record, after the whole nine-cell measurement had
    been paid for; this is the check that does not wait for that.
    """
    monkeypatch.setattr(script, "RESAMPLES", SMALL)
    model = StochasticModel()
    at_cell = _independent_sweep(rig, model)
    at_zero = _independent_sweep(rig, model, seed=0)
    # Reached: the seeds really do give different curves.
    assert np.abs(at_cell.curve(HORIZON) - at_zero.curve(HORIZON)).max() > 1.0
    record = script.measure_cell(**_real_kwargs(rig, model=model))
    assert SEED != 0
    for name in ZERO_CONTROL_NAMES:
        assert record["controls"][name] == 0.0, name
    for k in REGROUNDING_KS:
        np.testing.assert_array_equal(
            record["curves"]["rungs"][str(k)], at_cell.curve(k), err_msg=f"k={k}",
        )
        np.testing.assert_array_equal(
            record["curves"]["holds"][str(k)], at_cell.hold_position[k], err_msg=f"k={k}",
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


def test_the_sweep_reads_the_feature_cache_of_the_backbone_it_was_handed(tmp_path, monkeypatch):
    """A feature-input arm is fed its backbone's own namespaced cache. The caches
    carry the frame tag offset by +100, so an exact oracle is exactly 100 frames
    of true displacement wrong at every rung and every step (a closed form) and
    the HOLD at step `g` is wrong by `|100 + g - h|` frames. A decoy cache under
    the suffix `None` would name carries +200, so a sweep that was not handed the
    backbone reads another number rather than merely crashing.

    THE MUTATION THIS EXISTS FOR: `feature_backbone=feature_backbone` ->
    `feature_backbone=None` in the call to `regrounding_sweep`, which survives
    every test over the pixel rig because the pixel rig never reads a cache.
    """
    monkeypatch.setattr(script, "RESAMPLES", SMALL)
    feature_rig = _feature_rig(tmp_path)
    model = OracleModel()
    model.input_kind = "features"
    record = script.measure_cell(**_cell_kwargs(
        model=model, val_paths=feature_rig.paths, probe=feature_rig.probe,
        ks=REGROUNDING_KS, feature_backbone="random_vit",
        reference=evaluate_rollout(
            model, feature_rig.paths, feature_rig.probe, context=CONTEXT, horizon=HORIZON,
            seed=SEED, device=CPU, feature_backbone="random_vit",
        ),
    ))
    steps = np.arange(1, HORIZON + 1)
    for k in REGROUNDING_KS:
        ground = np.array([ground_step(k, int(h)) for h in steps])
        np.testing.assert_allclose(
            record["curves"]["rungs"][str(k)], np.full(HORIZON, 100.0 * STEP), rtol=1e-4,
            err_msg=f"rung k={k}",
        )
        np.testing.assert_allclose(
            record["curves"]["holds"][str(k)], STEP * np.abs(100.0 + ground - steps),
            rtol=1e-4, err_msg=f"hold k={k}",
        )
    assert record["controls"]["open_loop_divergence"] == 0.0


def test_the_sweep_runs_on_the_device_it_was_handed(rig, monkeypatch):
    """The device handed to `measure_cell` is the one the sweep traverses on, not
    whichever device the model's first parameter lives on. The model's only
    parameter is on `meta`, which no op here reads: handed `cpu` the sweep runs,
    handed `None` it falls back to the model's own device and `_rng_snapshot`
    refuses a device it has no verified generator state for.

    THE MUTATION THIS EXISTS FOR: `device=device` -> `device=None` in the call to
    `regrounding_sweep`."""
    monkeypatch.setattr(script, "RESAMPLES", SMALL)
    model = OracleModel()
    model.dummy = nn.Parameter(torch.zeros(1, device="meta"))
    assert next(model.parameters()).device.type == "meta"
    record = script.measure_cell(**_real_kwargs(rig, model=model))
    assert record["device"] == "cpu"
    assert record["controls"]["open_loop_divergence"] == 0.0


def test_the_ladder_swept_is_the_ks_handed_in(rig, monkeypatch):
    """`--ks 1,45` is the smoke's ladder: the record carries those two rungs and
    no others, the sweep is ASKED for those two, and its k-invariance control still
    reads 0.0 over them.

    What the sweep is asked for is seen through a wrapper that records the `ks` it
    was called with and returns the real sweep -- the argument's PASS-THROUGH, and
    the most that can be pinned: a sweep run over five rungs where two were asked
    for returns the same two the record reads, so nothing in the record can tell
    them apart (that mutation is an equivalent one, caught only by the wrapper).

    THE MUTATIONS THIS EXISTS FOR, each run against it: `ks=ks` ->
    `ks=REGROUNDING_KS` in the call to `regrounding_sweep` (the wrapper);
    `"ks": [int(k) for k in ks]` -> `list(REGROUNDING_KS)` in the record, and
    `for k in ks:` -> `for k in REGROUNDING_KS:` in the interval loop."""
    monkeypatch.setattr(script, "RESAMPLES", SMALL)
    real = script.regrounding_sweep
    asked = []

    def watching(*args, **kwargs):
        asked.append(tuple(kwargs["ks"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(script, "regrounding_sweep", watching)
    record = script.measure_cell(**_real_kwargs(rig, ks=(1, 45)))
    assert asked == [(1, 45)]
    assert record["ks"] == [1, 45]
    assert set(record["curves"]["rungs"]) == set(record["curves"]["holds"]) == {"1", "45"}
    assert set(record["intervals"]) == set(record["share"]) == {"1", "45"}
    assert record["controls"]["k_invariance_at_h1"] == 0.0


# ---------------------------------------------------------------------------
# The phase driver.
# ---------------------------------------------------------------------------

PHASE_SEEDS = (1, 2)
"""TWO SEEDS PER ARM, AND NEITHER IS ZERO. With one seed per arm and that seed 0,
`measure_phase` handing `measure_cell` `seed=0` and `headroom_record_path(out,
arm, 0)` were both indistinguishable from the real thing. The second is the
expensive one: in the real nine-cell run it writes every arm's seed-1 and seed-2
record to the seed-0 file name, so six of nine records are overwritten and it
surfaces only at the read phase, after the GPU time is spent. Every cell here
has a seed that differs from every other cell's, from 0, from the study's own
`SPLIT_SEED` and from `BOOTSTRAP_SEED`."""
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
    ("frozen_ssl", 1): StochasticModel, ("frozen_ssl", 2): StochasticModel,
}
CELLS = tuple(STUDY)
"""In the order `measure_phase` visits them: arm-major, seeds in the order given."""
assert set(STUDY) == set(MODELS) == {
    (arm, seed) for arm in ("pixel_ae", "frozen_ssl") for seed in PHASE_SEEDS
}
assert all(seed not in (0, SPLIT_SEED, BOOTSTRAP_SEED) for _, seed in CELLS)


def _recognisable(real, index: int):
    """`real` with its three position curves replaced by values no pass over these
    models can produce, each curve on its own offset and each CELL on its own
    base, so a record paired with another cell's reference is as visible as one
    that recomputed it. Binary fractions, so nothing is lost to a JSON round trip."""
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
    paired with the wrong cell reads differently. The reference it returns is a
    rollout of the cell's own model over the real split with its position curves
    made RECOGNISABLE; `args.references[(arm, seed)]` is `(verified, recomputed)`.
    """
    monkeypatch.setattr(script, "RESAMPLES", SMALL)
    data = tmp_path / "data"
    data.mkdir()
    episodes = [_wobbling_episode(60 + 7 * i, i) for i in range(20)]
    for index, episode in enumerate(episodes):
        save_episode(episode, data / f"ep_{index:06d}_len{episode.length:05d}.npz")
    source = tmp_path / "study"
    source.mkdir()
    for (arm, seed) in MODELS:
        (source / f"world_model_{arm}_seed{seed}.pt").write_bytes(b"")
    out = tmp_path / "headroom_out"
    probe = _linear_probe(episodes)
    prepared_for, references = [], {}

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
            context=CONTEXT, horizon=HORIZON, seed=cell.seed, device=CPU, feature_backbone=None,
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
        arms=["pixel_ae", "frozen_ssl"], seeds=list(PHASE_SEEDS), ks=list(REGROUNDING_KS),
        bootstrap_seed=BOOTSTRAP_SEED, references=references, probe=probe,
    )
    return args, prepared_for


def _val_split(args):
    from mbfps.data.buffer import ReplayBuffer
    from mbfps.data.split import VAL_FRACTION, episode_split

    return episode_split(
        ReplayBuffer(args.data, capacity_transitions=10**9).episode_paths(),
        val_fraction=VAL_FRACTION, seed=SPLIT_SEED,
    )[1]


def test_the_phase_writes_one_labelled_record_per_cell_from_that_cells_own_model(
    tmp_path, monkeypatch,
):
    """Nine records once went out serialised to ONE payload with the labels
    dropped. Here the four cells carry two different models and four different
    study records, and each file must hold its own cell's numbers.

    THE SEED IS BOUND THREE TIMES, because the cells' seeds are 1 and 2 and the
    bootstrap seed is 7: in the file NAME, in the record (`seed`), and in the
    sweep the cell's model was run through. The decision cell's intervals are
    recomputed from a second sweep the test runs at the CELL'S seed and drawn at
    the seed derived from the BOOTSTRAP seed and that cell's `(arm, seed)`, and
    must be the ones written.

    THE MUTATIONS THIS EXISTS FOR, each run against it, each of which a
    one-seed-0 fixture lets through: `measure_phase` handing `measure_cell`
    `seed=0` instead of `cell.seed`, `headroom_record_path(args.out, cell.arm, 0)`
    instead of `cell.seed`, and `bootstrap_seed=0` instead of `args.bootstrap_seed`.
    """
    args, prepared_for = _phase_env(tmp_path, monkeypatch)
    assert script.measure_phase(args) == script.EXIT_OK
    assert prepared_for == list(CELLS)
    assert sorted(p.name for p in args.out.iterdir()) == [
        "headroom_frozen_ssl_seed1.json", "headroom_frozen_ssl_seed2.json",
        "headroom_pixel_ae_seed1.json", "headroom_pixel_ae_seed2.json",
    ]
    val = _val_split(args)
    seen = set()
    for arm, seed in CELLS:
        carried = load_record(args.out / f"headroom_{arm}_seed{seed}.json")
        study = STUDY[(arm, seed)]
        assert (carried["arm"], carried["seed"]) == (arm, seed)
        assert carried["bootstrap_seed"] == BOOTSTRAP_SEED
        assert carried["cell_bootstrap_seed"] == _cell_seed((arm, seed)), (arm, seed)
        assert carried["step"] == study["steps"], (arm, seed)
        assert carried["kl_rate_above_free_bits"] == study["kl_rate_above_free_bits"]
        assert carried["kl_dyn_max"] == study["kl_dyn_max"]
        assert carried["record_git_sha"] == study["git_sha"]
        assert carried["git_sha"] == git_sha()
        assert carried["resamples"] == SMALL
        assert carried["episodes"]["val"] == [p.name for p in val]
        sweep = regrounding_sweep(
            MODELS[(arm, seed)](), val, args.probe, ks=REGROUNDING_KS, context=CONTEXT,
            horizon=HORIZON, seed=seed, device=CPU, feature_backbone=None,
        )
        rows = _rows(sweep, 1)
        for name, array in rows.items():
            expected = _interval(array, sweep.window_episode, 1, seed=_cell_seed((arm, seed)))
            assert carried["intervals"]["1"]["1"][name] == expected, (arm, seed, name)
            seen.add((arm, seed, name, tuple(expected.values())))
    assert len(seen) == len(CELLS) * 3
    # Reached: the two seeds of the stochastic arm read different intervals, so
    # the equality above was the sweep read at each cell's own seed.
    first, second = (
        load_record(args.out / f"headroom_frozen_ssl_seed{s}.json")["intervals"]["1"]["1"]
        for s in PHASE_SEEDS
    )
    assert first["deficit"] != second["deficit"]


def test_the_record_carries_the_reference_prepare_cell_verified(tmp_path, monkeypatch):
    """`prepare_cell` runs a val rollout to PROVE the loaded checkpoint reproduces
    the study record, and refuses when it does not. The canonical curves in a
    record must be THAT rollout's, not a second pass's.

    Each cell's `prepared.reference` carries recognisable position curves, and
    the record's three canonical curves equal them exactly. Asserting
    `evaluate_rollout` was never CALLED would pass a refactor that calls it and
    discards the result; this does not.

    THE MUTATIONS THIS EXISTS FOR: a local `reference = evaluate_rollout(...)` in
    `measure_cell`, and `measure_phase` passing a fresh `evaluate_rollout`
    instead of `prepared.reference`. The freshly computed curves differ from the
    verified ones (asserted below), so the test cannot pass by their coinciding.
    """
    args, _ = _phase_env(tmp_path, monkeypatch)
    assert script.measure_phase(args) == script.EXIT_OK
    assert sorted(args.references) == sorted(CELLS)
    for (arm, seed), (verified, recomputed) in args.references.items():
        carried = load_record(args.out / f"headroom_{arm}_seed{seed}.json")
        for name in ("floor_position", "rssm_position", "persistence_position"):
            assert not np.array_equal(getattr(recomputed, name), getattr(verified, name)), name
            np.testing.assert_array_equal(
                carried["curves"][name], getattr(verified, name), err_msg=f"{arm} {seed} {name}",
            )
    floors = [tuple(args.references[key][0].floor_position) for key in CELLS]
    assert len(set(floors)) == len(CELLS)


def test_the_phase_scores_the_studys_own_validation_split(tmp_path, monkeypatch):
    args, _ = _phase_env(tmp_path, monkeypatch)
    script.measure_phase(args)
    val = _val_split(args)
    record = load_record(args.out / "headroom_pixel_ae_seed1.json")
    assert record["episodes"]["val"] == [p.name for p in val]
    assert record["windows"]["total"] == sum(
        len(window_starts(load_episode(p).length, CONTEXT, HORIZON)) for p in val
    )
    assert record["split_seed"] == SPLIT_SEED
    assert len(set(record["windows"]["episode"])) >= 3


def test_the_phase_hands_the_ks_and_the_bootstrap_seed_it_was_given_to_the_cell(
    tmp_path, monkeypatch,
):
    """`--ks` and `--bootstrap-seed` reach `measure_cell`: the record carries the
    ladder asked for and the seed the bootstrap was drawn at."""
    args, _ = _phase_env(tmp_path, monkeypatch)
    args.ks, args.bootstrap_seed, args.arms, args.seeds = [1, 45], 11, ["pixel_ae"], [2]
    assert script.measure_phase(args) == script.EXIT_OK
    record = load_record(args.out / "headroom_pixel_ae_seed2.json")
    assert record["ks"] == [1, 45]
    assert record["bootstrap_seed"] == 11
    assert record["cell_bootstrap_seed"] == script.cell_bootstrap_seed(11, "pixel_ae", 2)
    assert set(record["curves"]["rungs"]) == {"1", "45"}


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
        "headroom_pixel_ae_seed1.json", "headroom_pixel_ae_seed2.json",
    ]


def test_the_phase_measures_a_repeated_arm_once(tmp_path, monkeypatch):
    args, prepared_for = _phase_env(tmp_path, monkeypatch)
    args.arms = ["pixel_ae", "pixel_ae"]
    args.seeds = [1, 1]
    assert script.measure_phase(args) == script.EXIT_OK
    assert prepared_for == [("pixel_ae", 1)]


def test_a_cell_that_raises_stops_the_phase_and_is_not_swallowed(tmp_path, monkeypatch):
    """The OTHER way a failed cell can exit 0: an exception caught and dropped.
    The cell that raised wrote nothing, the cells after it are never measured,
    and the exception reaches the caller (a traceback, status 1) rather than
    being turned into a quiet `EXIT_OK` after a run that cost GPU time.

    THE MUTATION THIS EXISTS FOR: wrapping the `measure_cell` call in `try: ...
    except Exception: continue`.
    """
    args, prepared_for = _phase_env(tmp_path, monkeypatch)
    real = script.measure_cell

    def explode_on_the_second_cell(*a, **k):
        if k["arm"] == "pixel_ae" and k["seed"] == 2:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(script, "measure_cell", explode_on_the_second_cell)
    with pytest.raises(RuntimeError, match="boom"):
        script.main([
            "--phase", "measure", "--source", str(args.source), "--out", str(args.out),
            "--data", str(args.data), "--device", "cpu", "--arms", "pixel_ae",
            "frozen_ssl", "--seeds", "1", "2",
        ])
    assert prepared_for == [("pixel_ae", 1), ("pixel_ae", 2)]
    assert [p.name for p in args.out.iterdir()] == ["headroom_pixel_ae_seed1.json"]


def test_the_cell_line_prints_the_numbers_the_verdict_is_read_from(
    tmp_path, monkeypatch, capsys,
):
    args, _ = _phase_env(tmp_path, monkeypatch)
    script.measure_phase(args)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(CELLS)
    for line, (arm, seed) in zip(lines, CELLS):
        path = args.out / f"headroom_{arm}_seed{seed}.json"
        record = load_record(path)
        at = record["intervals"]["1"]["1"]
        assert line.startswith(f"{arm} seed {seed}:"), line
        assert "at k=1, h=1" in line
        for name in ("headroom", "skill", "deficit"):
            point, low, high = (at[name][key] for key in ("point", "ci_low", "ci_high"))
            assert f"{name} {point:+8.3f} [{low:+8.3f},{high:+8.3f}]" in line, (name, line)
        # The reference in this environment is the RECOGNISABLE one, thousands of
        # map units from anything a pass could produce, so exactly the two
        # controls read against it are flagged -- and the two the sweep checks
        # against itself are not.
        assert "CONTROLS BROKEN: open_loop_divergence=" in line, line
        assert ", floor_divergence=" in line and "persistence_divergence" not in line
        assert "k_invariance_at_h1" not in line
        assert f"{record['controls']['identity_residual']:.2e}" in line
        assert str(path) in line


@pytest.mark.parametrize("decision", [(1, 1), (3, 10), (45, 45)])
def test_the_cell_line_is_read_at_the_records_own_decision_cell_and_says_the_controls_hold(
    hand_record, tmp_path, decision,
):
    """The line prints the three differences at the decision cell THE RECORD NAMES
    -- (1, 1), a middle cell and the last -- so a line that read a fixed (k, h)
    prints another cell's numbers. Each cell's intervals differ (it is the hand
    sweep's own), and the controls are the clean ones.

    THE MUTATION THIS EXISTS FOR, run against it: `record["decision_k"]` ->
    the literal `"1"` in `_cell_line`."""
    k, h = decision
    record = copy.deepcopy(hand_record)
    record.update(decision_k=k, decision_h=h)
    line = script._cell_line(record, tmp_path / "x.json")
    assert "controls hold" in line and "BROKEN" not in line, line
    assert f"at k={k}, h={h} " in line and "0.00e+00" in line
    assert "13 windows from 6 episodes" in line
    at = record["intervals"][str(k)][str(h)]
    for name in ("headroom", "skill", "deficit"):
        assert f"{name} {_triple_text(at[name])}" in line, (name, line)
    if decision != (1, 1):
        other = hand_record["intervals"]["1"]["1"]["headroom"]
        assert f"headroom {_triple_text(other)}" not in line


def _triple_text(interval) -> str:
    return f"{interval['point']:+8.3f} [{interval['ci_low']:+8.3f},{interval['ci_high']:+8.3f}]"


@pytest.mark.parametrize(
    "name", ["persistence_divergence", "k_invariance_at_h1", "open_loop_divergence", "floor_divergence"],
)
@pytest.mark.parametrize("bad", [np.nan, 0.5, -0.5], ids=["nan", "positive", "negative"])
def test_the_cell_line_flags_a_control_that_missed_its_known_answer(hand_record, tmp_path, name, bad):
    """The measure phase RECORDS a missed control and carries on, so the cell
    line is the only place an operator sees it before the read. Read with `!=
    0.0`: NaN is the case that matters, because `persistence_divergence` and
    `k_invariance_at_h1` propagate NaN by design and `nan > 0` is False.

    THE MUTATION THIS EXISTS FOR: `broken_controls` testing `> 0.0`."""
    record = copy.deepcopy(hand_record)
    record["controls"][name] = bad
    line = script._cell_line(record, tmp_path / "x.json")
    assert "CONTROLS BROKEN" in line and name in line, line
    assert "controls hold" not in line
    assert script.broken_controls(record["controls"]) == [f"{name}={bad!r}"]


def test_the_identity_residual_is_judged_against_the_tolerance_and_nan_fails_it(hand_record):
    """`identity_residual` is held to `IDENTITY_TOLERANCE`, not to zero: rounding
    noise below it passes and a residual above it does not. NaN must FAIL, and
    `nan > tolerance` is False -- so the comparison is written `not residual <=
    tolerance`.

    THE MUTATION THIS EXISTS FOR: `not residual <= IDENTITY_TOLERANCE` ->
    `residual > IDENTITY_TOLERANCE`."""
    controls = copy.deepcopy(hand_record["controls"])
    controls["identity_residual"] = H.IDENTITY_TOLERANCE / 2
    assert script.broken_controls(controls) == []
    controls["identity_residual"] = H.IDENTITY_TOLERANCE
    assert script.broken_controls(controls) == []
    controls["identity_residual"] = H.IDENTITY_TOLERANCE * 2
    assert len(script.broken_controls(controls)) == 1
    controls["identity_residual"] = float("nan")
    assert len(script.broken_controls(controls)) == 1


def test_main_takes_its_cells_and_directories_from_the_command_line(tmp_path, monkeypatch):
    """The flag names reach the fields `measure_phase` reads, in BOTH spellings
    of a list -- `--arms a,b` and `--arms a b` are one plan (the run's command
    line, `--arms frozen_ssl,pixel_ae,random_vit --seeds 0,1,2 --ks 1,3,5,15,45`,
    is the comma form)."""
    args, _ = _phase_env(tmp_path, monkeypatch)
    status = script.main([
        "--phase", "measure", "--source", str(args.source), "--out", str(args.out),
        "--data", str(args.data), "--device", "cpu", "--arms", "frozen_ssl", "--seeds", "2",
        "--ks", "1,45", "--bootstrap-seed", "5",
    ])
    assert status == script.EXIT_OK
    assert [p.name for p in args.out.iterdir()] == ["headroom_frozen_ssl_seed2.json"]
    record = load_record(args.out / "headroom_frozen_ssl_seed2.json")
    assert (record["ks"], record["bootstrap_seed"]) == ([1, 45], 5)


# ---------------------------------------------------------------------------
# The read phase: records on disk, pooled into the reading.
#
# EVERY CELL STARTS AS A DEEP COPY of the record `measure_cell` writes for the
# hand sweep (`hand_record`), so the schema is the writer's own, and only the
# decision cell's three intervals are then overwritten, from a formula in the
# cell's index `c`. The expected `HeadroomInputs` is built from the same
# formulas -- never from the records and never through `script.headroom_inputs`.
#
#   * Three arms x seeds 0, 1, 2: nine cells, none alike in any number the
#     reading reads (`_decorate` shifts every number by 0.001 * c, which no
#     placement is within 0.1 of changing). The arms are given in `ARMS` order,
#     whose index is NOT `sorted()` order (`frozen_ssl` sorts first).
#   * The placements' triples are (point, ci_low, ci_high).
# ---------------------------------------------------------------------------

READ_ARMS = ("pixel_ae", "frozen_ssl", "random_vit")
READ_SEEDS = (0, 1, 2)
READ_CELLS = [(arm, seed) for arm in READ_ARMS for seed in READ_SEEDS]
FIRST_CELL, SECOND_CELL, LAST_CELL = ("frozen_ssl", 0), ("frozen_ssl", 1), ("random_vit", 2)
"""`sorted()` order of the nine cells -- the order `require_one_protocol` walks."""

TRIPLES = {
    "BETWEEN": dict(headroom=(5.0, 3.0, 7.0), skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5)),
    "AT_PERFECT": dict(headroom=(5.0, 3.0, 7.0), skill=(2.0, 1.0, 3.0), deficit=(0.1, -0.5, 0.7)),
    "AT_COPYING": dict(headroom=(5.0, 3.0, 7.0), skill=(0.1, -0.5, 0.7), deficit=(4.9, 3.0, 6.8)),
    "AMBIGUOUS": dict(headroom=(5.0, 3.0, 7.0), skill=(0.2, -0.5, 0.9), deficit=(0.2, -0.4, 0.8)),
    # The gate runs first: skill and deficit are both resolvably positive here
    # and the cell still reads UNREADABLE, because headroom's interval is not
    # above zero (the brief's own triple).
    "UNREADABLE": dict(headroom=(0.4, -0.9, 1.7), skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5)),
}
NAMES = ("headroom", "skill", "deficit")


def _decorate(triple, c):
    return tuple(value + 0.001 * c for value in triple)


def _spec(placements) -> dict:
    """`{cell: {"c", "triples"}}` for nine placements in `READ_CELLS` order."""
    assert len(placements) == len(READ_CELLS)
    return {
        cell: {"c": c, "triples": TRIPLES[placement]}
        for c, (cell, placement) in enumerate(zip(READ_CELLS, placements))
    }


def _uniform_spec(headroom, skill, deficit) -> dict:
    return {
        cell: {"c": c, "triples": dict(headroom=headroom, skill=skill, deficit=deficit)}
        for c, cell in enumerate(READ_CELLS)
    }


def _triples_of(entry) -> dict:
    return {name: _decorate(entry["triples"][name], entry["c"]) for name in NAMES}


def _records_from(base, spec, *, decision=(1, 1)) -> dict:
    k, h = decision
    records = {}
    for (arm, seed), entry in spec.items():
        record = copy.deepcopy(base)
        record.update(arm=arm, seed=seed, decision_k=k, decision_h=h)
        at = record["intervals"][str(k)][str(h)]
        for name, triple in _triples_of(entry).items():
            at[name] = dict(zip(("point", "ci_low", "ci_high"), triple))
        records[(arm, seed)] = record
    return records


def _expected_inputs(spec, *, decision=(1, 1), ks=HAND_KS, clusters=None) -> H.HeadroomInputs:
    """What `headroom_inputs` must return, built from the spec alone. The cluster
    count is the hand sweep's own six."""
    clusters = clusters or (lambda cell: len(set(HAND_GROUPS.tolist())))
    return H.HeadroomInputs(
        cells={
            cell: H.HeadroomCell(
                arm=cell[0], seed=cell[1],
                headroom=H.Interval(*_triples_of(entry)["headroom"]),
                skill=H.Interval(*_triples_of(entry)["skill"]),
                deficit=H.Interval(*_triples_of(entry)["deficit"]),
                clusters=clusters(cell),
            )
            for cell, entry in spec.items()
        },
        decision_k=decision[0], decision_h=decision[1], ks=tuple(ks),
    )


def _write_records(out, records) -> Path:
    """Write `records` under their file names -- spelled out here, not taken from
    `headroom_record_path`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for (arm, seed), record in records.items():
        write_record(out / f"headroom_{arm}_seed{seed}.json", record)
    return out


def _check_discrimination(spec, records):
    """The write helpers' own check, taken before they return: nine records, at
    least three clusters in each, an interval of positive width in every one of
    the three differences, and no two cells carrying the same decision intervals."""
    assert len(records) == 9
    seen = set()
    for (arm, seed), record in records.items():
        assert len(set(record["windows"]["episode"])) >= 3
        at = record["intervals"][str(record["decision_k"])][str(record["decision_h"])]
        for name in NAMES:
            assert at[name]["ci_low"] < at[name]["ci_high"], (arm, seed, name)
        seen.add(tuple(tuple(at[name].values()) for name in NAMES))
    assert len(seen) == 9, "two cells carry the same numbers"


def _write_nine_records(tmp_path, *, headroom, skill, deficit, base) -> dict:
    """Three arms x three seeds, the same (point, ci_low, ci_high) triples in
    every cell up to a per-cell shift of 0.001 * c. Returns the records."""
    spec = _uniform_spec(headroom, skill, deficit)
    records = _records_from(base, spec)
    _check_discrimination(spec, records)
    _write_records(tmp_path, records)
    return records


def _write_split_records(tmp_path, base) -> dict:
    """Four BETWEEN, four UNREADABLE and one AT_PERFECT: no placement reaches the
    bar of five."""
    spec = _spec(["BETWEEN"] * 4 + ["UNREADABLE"] * 4 + ["AT_PERFECT"])
    records = _records_from(base, spec)
    _check_discrimination(spec, records)
    _write_records(tmp_path, records)
    return records


def _read_args(out, *, arms=READ_ARMS, seeds=READ_SEEDS):
    return types.SimpleNamespace(out=Path(out), arms=list(arms), seeds=list(seeds))


def _capture_read_stdout(out) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        script.main(["--phase", "read", "--out", str(out)])
    return buffer.getvalue()


def _named_cells(message: str, cells) -> set:
    return {(arm, seed) for arm, seed in cells if f"{arm} seed {seed}" in message}


def test_the_read_fixtures_reach_the_placement_they_are_named_for(hand_record, tmp_path):
    """The fixtures' own check, taken from `reading_headroom` over inputs built
    from the spec alone: a pool that read some other placement than its name
    would make every exit-code test below test the wrong thing."""
    for placement in TRIPLES:
        reading = H.reading_headroom(_expected_inputs(_spec([placement] * 9)))
        assert reading.verdict == placement, placement
    spec = _spec(["BETWEEN"] * 4 + ["UNREADABLE"] * 4 + ["AT_PERFECT"])
    split = H.reading_headroom(_expected_inputs(spec))
    assert split.verdict == H.NO_MAJORITY
    assert split.rule.startswith("no placement reached 5 of 9 cells")
    cells = sorted(spec)
    assert (cells[0], cells[1], cells[-1]) == (FIRST_CELL, SECOND_CELL, LAST_CELL)
    assert cells != READ_CELLS, "ARMS order must not be sorted()"
    _write_nine_records(
        tmp_path, headroom=(5.0, 3.0, 7.0), skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5),
        base=hand_record,
    )
    assert len(list(Path(tmp_path).glob("headroom_*_seed*.json"))) == 9


def test_read_returns_47_when_a_majority_of_cells_is_unreadable(hand_record, tmp_path, capsys):
    """Exit 47 is the instrument's verdict, not the model's -- and it is a
    FINDING: the table and the verdict are printed and written, not withheld."""
    _write_nine_records(
        tmp_path, headroom=(0.4, -0.9, 1.7), skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5),
        base=hand_record,
    )
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == script.EXIT_UNREADABLE_HEADROOM == 47
    out = capsys.readouterr().out
    assert "verdict: UNREADABLE" in out
    assert (tmp_path / "headroom.txt").read_text() == out


def test_read_returns_48_when_no_placement_holds_a_majority(hand_record, tmp_path, capsys):
    """Exit 48 with a four-four-one split across nine cells, and the RULE is
    printed -- it is the only line that says which of the two causes of
    `NO_MAJORITY` this is."""
    _write_split_records(tmp_path, hand_record)
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == script.EXIT_NO_MAJORITY == 48
    out = capsys.readouterr().out
    assert "verdict: NO_MAJORITY" in out
    assert "no placement reached 5 of 9 cells" in out
    assert (tmp_path / "headroom.txt").read_text() == out


def test_read_prints_the_rule_that_tells_the_two_no_majorities_apart(
    hand_record, tmp_path, monkeypatch, capsys,
):
    """`NO_MAJORITY` is returned both when no placement reached the bar and when
    one did but its cells spanned fewer than `ARMS_REQUIRED` arms, and both leave
    as exit 48. The verdict line is IDENTICAL in the two outputs; only `rule`
    differs, so the read phase must print it -- to stdout and to the file.

    The second shape is not one a 3x3 plan can build (no arm holds more than
    three cells, so five cells always span two arms), and `read_phase` cannot
    build any other: the loader is replaced by one returning arms of 3, 3 and 9
    seeds, with the nine-seed arm reading BETWEEN alone, which reaches the bar of
    8 of 15 by itself.

    THE MUTATION THIS EXISTS FOR: printing `reading.verdict` without the rule.
    """
    _write_split_records(tmp_path / "a", hand_record)
    assert script.main(["--phase", "read", "--out", str(tmp_path / "a")]) == 48
    no_bar = capsys.readouterr().out

    uneven = {}
    cells = (
        [("pixel_ae", s) for s in range(3)]
        + [("frozen_ssl", s) for s in range(3)]
        + [("random_vit", s) for s in range(9)]
    )
    base = _records_from(hand_record, _spec(["BETWEEN"] * 9))[READ_CELLS[0]]
    for arm, seed in cells:
        record = copy.deepcopy(base)
        record.update(arm=arm, seed=seed, nonfinite={})  # as `load_record` returns it
        at = record["intervals"]["1"]["1"]
        for name in NAMES:
            triple = TRIPLES["BETWEEN" if arm == "random_vit" else "UNREADABLE"][name]
            at[name] = dict(zip(("point", "ci_low", "ci_high"), triple))
        uneven[(arm, seed)] = record
    out = tmp_path / "b"
    out.mkdir()
    monkeypatch.setattr(script, "load_headroom", lambda directory, arms, seeds: uneven)
    assert script.read_phase(_read_args(out, seeds=range(9))) == script.EXIT_NO_MAJORITY
    one_arm = capsys.readouterr().out

    assert "verdict: NO_MAJORITY" in no_bar and "verdict: NO_MAJORITY" in one_arm
    assert "spanned only 1 arm (random_vit)" in one_arm
    assert "spanned only" not in no_bar and "no placement reached" not in one_arm
    assert (out / "headroom.txt").read_text() == one_arm
    assert (tmp_path / "a" / "headroom.txt").read_text() == no_bar


@pytest.mark.parametrize("placement", ["BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS"])
def test_each_model_placement_exits_zero_and_is_written(hand_record, tmp_path, capsys, placement):
    """Four answers, four zeros: each is an answer, and AMBIGUOUS is an answer
    about the ruler at a readable cell, a different claim from 47's about the
    instrument.

    THE MUTATION THIS EXISTS FOR: adding any of the four to `READ_EXITS`."""
    records = _records_from(hand_record, _spec([placement] * 9))
    _write_records(tmp_path, records)
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"verdict: {placement}" in out
    assert (tmp_path / "headroom.txt").read_text() == out


def test_the_reading_is_the_formatters_text_byte_for_byte_and_the_same_on_two_reads(
    hand_record, tmp_path,
):
    """The reading must not depend on dict ordering or a fresh RNG draw, and it
    is compared with text built from the SPEC path -- `format_reading_headroom`
    over inputs written from the spec, never through the script -- not with a
    second call of the same code: two reads strip the same bytes, so
    byte-identity against itself cannot catch an `rstrip()`, and `print(text)` on
    a string that already ends in a newline doubles it. The precondition is
    asserted: the text ends in exactly one newline.

    THE MUTATIONS THIS EXISTS FOR, each run against it: `write_text(text.rstrip())`
    and `print(text)` without `end=""`.
    """
    spec = _spec(["BETWEEN"] * 9)
    inputs = _expected_inputs(spec)
    expected = H.format_reading_headroom(H.reading_headroom(inputs), inputs)
    assert expected.endswith("\n") and not expected.endswith("\n\n")
    records = _records_from(hand_record, spec)
    first = _write_records(tmp_path / "a", records)
    second = _write_records(tmp_path / "b", records)
    one, two = _capture_read_stdout(first), _capture_read_stdout(second)
    assert one == two == expected
    assert (first / "headroom.txt").read_text() == (second / "headroom.txt").read_text() == expected
    assert (first / "headroom.txt").read_bytes() == expected.encode()


# --- headroom_inputs ---------------------------------------------------------


@pytest.mark.parametrize("decision", [(1, 1), (3, 10), (45, 45)])
def test_headroom_inputs_reads_every_cell_from_its_own_record_at_the_decision_cell(
    hand_record, decision,
):
    """Each cell's three intervals come from ITS record, at the decision cell the
    record names -- (1, 1), a middle cell and the last -- and are compared
    with the `HeadroomInputs` the spec writes. Windows differ per cell: cell `c`
    has `3 + c % 4` episodes of `2 + c % 3` windows each, so a `clusters` read
    from the first record, or from the wrong cell, is a different number; and
    `clusters` is the count of DISTINCT labels, not of windows.

    THE MUTATIONS THIS EXISTS FOR, each run against it: reading
    `intervals[str(DECISION_K)][str(DECISION_H)]` instead of the record's own
    decision cell, and `clusters=len(labels)`.
    """
    spec = _spec([
        "BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS", "UNREADABLE",
        "BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS",
    ])
    records = _records_from(hand_record, spec, decision=decision)
    cluster_count = {}
    for c, cell in enumerate(READ_CELLS):
        episodes, windows = 3 + c % 4, 2 + c % 3
        records[cell]["windows"]["episode"] = np.repeat(np.arange(episodes), windows).tolist()
        cluster_count[cell] = episodes
    assert len(set(cluster_count.values())) > 1
    inputs = script.headroom_inputs(records)
    assert inputs == _expected_inputs(
        spec, decision=decision, clusters=lambda cell: cluster_count[cell],
    )
    # Reached: where the decision cell is not (1, 1), the (1, 1) entry the first
    # cell's record carries is a different interval from the one it was read at.
    if decision != (1, 1):
        decoy = H.Interval(**records[READ_CELLS[0]]["intervals"]["1"]["1"]["headroom"])
        assert decoy != inputs.cells[READ_CELLS[0]].headroom


def test_headroom_inputs_refuses_a_null_episode_label_and_never_clusters_by_window(hand_record):
    """No fallback to `arange(n)`. A record with no clustering must stop the
    read, not silently become the window-level bootstrap
    `pooling.clustered_interval`'s first paragraph rules out: treating every
    window as its own episode would print a narrower interval and a more
    confident verdict.

    Asked of `headroom_inputs` DIRECTLY, because the read phase's protocol check
    refuses a null label first (`windows.episode` is a protocol field) and so
    stands in front of this refusal in any test that goes through it. A record
    with a null label and a window count that would make `clusters` a usable
    number is the case a fallback survives.

    THE MUTATION THIS EXISTS FOR: `labels = np.arange(record["windows"]["total"])`
    in place of the refusal.
    """
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    victim = ("pixel_ae", 1)
    records[victim]["windows"]["episode"] = None
    assert records[victim]["windows"]["total"] >= 3
    with pytest.raises(SystemExit) as caught:
        script.headroom_inputs(records)
    message = str(caught.value)
    assert "pixel_ae seed 1" in message and "episode labels" in message, message


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("field", ["point", "ci_low", "ci_high"])
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_headroom_inputs_refuses_a_non_finite_interval_at_the_decision_cell(
    hand_record, name, field, bad,
):
    """`Interval.resolvably_positive` is `ci_low > 0.0`, False for NaN, so a NaN
    DEFICIT beside a resolvable headroom and skill reads `AT_PERFECT` -- a verdict
    about a model derived from a broken number. `clustered_interval` refuses
    non-finite rows, so that should be unreachable; here it is made provably so
    by refusing the record, whichever of the nine numbers is the broken one.

    `reading_headroom` DOES return `AT_PERFECT` for such a cell (shown on the
    module's own types in the next test), so a refusal that did not happen would
    be a wrong verdict and not an exception.

    THE MUTATIONS THIS EXISTS FOR, each run against it: no finiteness check, and
    a check over `headroom` only."""
    spec = _spec(["BETWEEN"] * 9)
    records = _records_from(hand_record, spec)
    victim = ("pixel_ae", 1)
    records[victim]["intervals"]["1"]["1"][name][field] = bad
    with pytest.raises(SystemExit) as caught:
        script.headroom_inputs(records)
    message = str(caught.value)
    assert "pixel_ae seed 1" in message and f"{name}.{field}" in message, message


def test_a_nan_deficit_beside_a_resolvable_headroom_and_skill_would_read_at_perfect():
    """The reason the refusal above exists, shown on the module's own types: this
    is what `reading_headroom` does with the NaN if it is let through."""
    cell = H.HeadroomCell(
        arm="a", seed=0, headroom=H.Interval(5.0, 3.0, 7.0), skill=H.Interval(2.0, 1.0, 3.0),
        deficit=H.Interval(NAN, NAN, NAN), clusters=6,
    )
    assert cell.placement() == "AT_PERFECT"


def test_headroom_inputs_refuses_a_non_finite_interval_anywhere_not_only_at_the_decision_cell(
    hand_record,
):
    """A record whose interval values are not ALL finite is not one this script's
    measure wrote, whichever cell the NaN is in.

    THE MUTATION THIS EXISTS FOR: a finiteness check restricted to the decision
    cell."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    records[("random_vit", 0)]["intervals"]["45"]["30"]["skill"]["ci_high"] = NAN
    with pytest.raises(SystemExit) as caught:
        script.headroom_inputs(records)
    assert "random_vit seed 0" in str(caught.value)


@pytest.mark.parametrize("field", ["decision_k", "decision_h", "ks"])
def test_headroom_inputs_refuses_a_pool_that_disagrees_on_the_one_value_it_takes(
    hand_record, field,
):
    """The decision cell and the ladder are the values `HeadroomInputs` holds once
    for nine cells, so a disagreement on either is refused HERE rather than
    depending on `require_one_protocol` having run first: a first-record pick
    would otherwise read every other cell at a (k, h) its own record does not
    name. A table whose rows were read at different (k, h) is not a table."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    odd = records[LAST_CELL]
    odd[field] = {"decision_k": 3, "decision_h": 2, "ks": [1, 3]}[field]
    with pytest.raises(SystemExit) as caught:
        script.headroom_inputs(records)
    assert f"disagree on {field}" in str(caught.value)


# --- the refusals the read phase makes by name -------------------------------


def test_read_phase_refuses_a_plan_narrower_than_the_bar_before_loading_anything(tmp_path):
    """`--arms pixel_ae` cannot be read: a verdict needs `ARMS_REQUIRED` arms.
    It is refused BEFORE the records are looked for -- the directory does not
    exist, so a plan check made after the load would return 11 instead -- and
    `--arms a a a` is ONE arm: counting the list would let it through."""
    for arms in (["pixel_ae"], ["pixel_ae", "pixel_ae", "pixel_ae"]):
        with pytest.raises(SystemExit) as caught:
            script.read_phase(_read_args(tmp_path / "absent", arms=arms))
        message = str(caught.value)
        assert f"ARMS_REQUIRED={H.ARMS_REQUIRED}" in message and "1 arm(s)" in message, message


def test_an_arm_short_of_the_minimum_seeds_raises_and_is_never_caught_into_an_exit(
    hand_record, tmp_path,
):
    """`reading_headroom` RAISES for an arm with fewer than `SEEDS_MINIMUM` seeds,
    naming the arm and its seeds. That is a shape error -- `--phase read --seeds 0
    1` -- not something found in the data, so it reaches the caller as the
    ValueError it is: not a status, not an exit number, and nothing is written.

    THE MUTATION THIS EXISTS FOR: a `try/except ValueError` around the reading
    that returns 47 or 48."""
    _write_records(tmp_path, _records_from(hand_record, _spec(["BETWEEN"] * 9)))
    with pytest.raises(ValueError, match=r"SEEDS_MINIMUM=3.*'pixel_ae': \[0, 1\]"):
        script.read_phase(_read_args(tmp_path, seeds=(0, 1)))
    assert not (tmp_path / "headroom.txt").exists()


def test_read_phase_names_the_first_missing_cell_and_exits_eleven(hand_record, tmp_path, capsys):
    """Two cells are missing; the one the PLAN reaches first is the one named,
    and nothing is written."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    del records[SECOND_CELL], records[LAST_CELL]
    _write_records(tmp_path, records)
    assert script.read_phase(_read_args(tmp_path)) == script.EXIT_NO_CHECKPOINTS == 11
    out = capsys.readouterr().out
    assert "NO CELL" in out and "frozen_ssl seed 1" in out and "random_vit seed 2" not in out, out
    assert "--phase measure" in out
    assert not (tmp_path / "headroom.txt").exists()


@pytest.mark.parametrize(
    "swap, first_file, record_says, other_file",
    [
        pytest.param(
            (("pixel_ae", 1), ("pixel_ae", 2)),
            "headroom_pixel_ae_seed1.json was read for pixel_ae seed 1",
            "arm='pixel_ae' seed=2", "headroom_pixel_ae_seed2.json", id="seed half",
        ),
        pytest.param(
            (("pixel_ae", 0), ("frozen_ssl", 0)),
            "headroom_pixel_ae_seed0.json was read for pixel_ae seed 0",
            "arm='frozen_ssl' seed=0", "headroom_frozen_ssl_seed0.json", id="arm half",
        ),
    ],
)
def test_a_record_filed_under_another_cells_name_is_refused(
    hand_record, tmp_path, swap, first_file, record_says, other_file,
):
    """A swapped pair of files would pool one cell under another's name with every
    count still right. Each swap changes ONE of arm and seed, so a check that
    tests the other cannot see it.

    THE MUTATIONS THIS EXISTS FOR, one per case: dropping the `arm` comparison from
    `load_headroom`'s filename check (the `arm half` fails) and dropping the `seed`
    comparison (the `seed half` fails)."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    one, other = swap
    records[one], records[other] = records[other], records[one]
    _write_records(tmp_path, records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(_read_args(tmp_path))
    message = str(caught.value)
    assert message.startswith(first_file), message
    assert f"but its record says {record_says}" in message, message
    assert other_file not in message, message


DISAGREEMENTS = {
    "git_sha": lambda r: r.__setitem__("git_sha", "0" * 40),
    "torch_version": lambda r: r.__setitem__("torch_version", r["torch_version"] + "+other"),
    "device": lambda r: r.__setitem__("device", "mps"),
    "step": lambda r: r.__setitem__("step", r["step"] + 1),
    "context": lambda r: r.__setitem__("context", r["context"] + 1),
    "horizon": lambda r: r.__setitem__("horizon", r["horizon"] + 1),
    "split_seed": lambda r: r.__setitem__("split_seed", r["split_seed"] + 1),
    "ks": lambda r: r.__setitem__("ks", r["ks"][:-1] + [r["ks"][-1] + 1]),
    "reported_h": lambda r: r.__setitem__("reported_h", r["reported_h"][:-1]),
    "decision_k": lambda r: r.__setitem__("decision_k", 3),
    "decision_h": lambda r: r.__setitem__("decision_h", 2),
    "confidence": lambda r: r.__setitem__("confidence", 0.9),
    "resamples": lambda r: r.__setitem__("resamples", r["resamples"] + 1),
    "identity_tolerance": lambda r: r.__setitem__("identity_tolerance", r["identity_tolerance"] * 10),
    "secondary_sigmas": lambda r: r.__setitem__("secondary_sigmas", 3),
    "bootstrap_seed": lambda r: r.__setitem__("bootstrap_seed", r["bootstrap_seed"] + 1),
    "episodes.val": lambda r: r["episodes"].__setitem__("val", r["episodes"]["val"][::-1]),
    "windows.episode": lambda r: r["windows"]["episode"].__setitem__(
        -1, r["windows"]["episode"][-1] + 1
    ),
}
"""One way for a record to disagree with the others, per protocol field."""

REQUIRED_PROTOCOL_FIELDS = {
    "git_sha", "torch_version", "device", "step", "context", "horizon", "split_seed",
    "ks", "reported_h", "decision_k", "decision_h", "confidence", "resamples",
    "identity_tolerance", "secondary_sigmas", "bootstrap_seed",
}


def test_the_protocol_table_names_every_required_field_and_each_has_a_disagreement_case():
    """The table is a module-level tuple the function ITERATES and this test
    READS, so a field cannot be added to the comparison without this noticing it
    has no disagreement case, and one cannot be dropped without the required set
    noticing. `git_sha` is required because an earlier version of M3l's check
    omitted it and let records from different torch builds and code versions pool
    into one finding with no refusal at all, and `bootstrap_seed` because every
    interval was drawn at it."""
    names = [name for name, _ in script._PROTOCOL_FIELDS]
    assert len(names) == len(set(names))
    assert REQUIRED_PROTOCOL_FIELDS <= set(names)
    assert set(names) == set(DISAGREEMENTS)


@pytest.mark.parametrize("victim", [FIRST_CELL, SECOND_CELL, LAST_CELL], ids=["first", "second", "last"])
@pytest.mark.parametrize("field", sorted(DISAGREEMENTS))
def test_one_record_that_disagrees_on_any_protocol_field_is_refused_naming_it(
    hand_record, field, victim,
):
    """Every field in the table, with the odd record at the front, second and
    last of the sorted nine. Last is the one a first-record-only check misses;
    second is the one a loop that starts one record too late misses. The message
    names the field AND the odd cell, and exactly two cells -- a message listing
    all nine would satisfy `victim in message` for any victim at all."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    DISAGREEMENTS[field](records[victim])
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert f"disagree on {field}:" in message, message
    named = _named_cells(message, records)
    assert victim in named and len(named) == 2, (named, message)


@pytest.mark.parametrize("lacking", [name for name, _ in script._PROTOCOL_FIELDS])
def test_a_record_without_a_protocol_field_is_refused_by_name_not_a_key_error(
    hand_record, lacking,
):
    """A record WITHOUT a field is refused by name, not defaulted: a default would
    make a record that merely lacks the field pass for one that agrees."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    record = records[SECOND_CELL]
    if lacking == "windows.episode":
        record["windows"]["episode"] = None
    elif lacking == "episodes.val":
        del record["episodes"]["val"]
    else:
        del record[lacking]
    with pytest.raises(SystemExit) as caught:
        script.require_one_protocol(records)
    message = str(caught.value)
    assert "frozen_ssl seed 1" in message and f"lacks {lacking}" in message, message


def test_read_phase_refuses_records_that_disagree_on_the_protocol_and_writes_nothing(
    hand_record, tmp_path,
):
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    DISAGREEMENTS["git_sha"](records[LAST_CELL])
    _write_records(tmp_path, records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(_read_args(tmp_path))
    assert "disagree on git_sha" in str(caught.value)
    assert not (tmp_path / "headroom.txt").exists()


def test_the_brief_null_episode_record_is_refused_at_the_read_phase(hand_record, tmp_path):
    """The brief's own case, through `main`: one record's `windows.episode` is
    null, and the read must stop rather than print a verdict. (It is the protocol
    check that stops it here; the refusal in `headroom_inputs` itself is asked
    directly above.)"""
    _write_nine_records(
        tmp_path, headroom=(5.0, 3.0, 7.0), skill=(2.0, 1.0, 3.0), deficit=(3.0, 1.5, 4.5),
        base=hand_record,
    )
    path = script.headroom_record_path(tmp_path, "pixel_ae", 1)
    record = json.loads(path.read_text())
    record["windows"]["episode"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "read", "--out", str(tmp_path)])
    assert caught.value.code != 0
    assert "pixel_ae seed 1" in str(caught.value) and "windows.episode" in str(caught.value)
    assert not (tmp_path / "headroom.txt").exists()


@pytest.mark.parametrize("name", ["persistence_divergence", "k_invariance_at_h1", "open_loop_divergence", "floor_divergence"])
@pytest.mark.parametrize("bad", [NAN, 1e-3, -1e-3], ids=["nan", "positive", "negative"])
def test_a_control_that_missed_its_known_answer_is_refused_by_name_whatever_its_sign(
    hand_record, tmp_path, name, bad,
):
    """THE TEST THE `> 0` DEFECT SURVIVES EVERYTHING ELSE.
    `persistence_divergence` and `k_invariance_at_h1` propagate NaN by design,
    and `nan > 0` is False, so a control read with `> 0` reports a broken
    measurement as passing and the read goes on to print a verdict about a model.
    Every one of the four is read with `!= 0.0` here, and NaN is refused beside
    a positive and a NEGATIVE value (the latter is impossible by construction --
    each is a max of absolute values -- and is refused all the same, which is
    what `!=` says and `>` does not).

    The refusal names the cell and the control, and nothing is written.

    THE MUTATION THIS EXISTS FOR: `broken_controls` reading a control with `> 0`
    (or `> 0.0`) instead of `!= 0.0`."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    records[SECOND_CELL]["controls"][name] = bad
    _write_records(tmp_path, records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(_read_args(tmp_path))
    message = str(caught.value)
    assert "frozen_ssl seed 1" in message and name in message, message
    assert not (tmp_path / "headroom.txt").exists()


@pytest.mark.parametrize(
    "name", [*ZERO_CONTROL_NAMES, "identity_residual"],
)
def test_a_nan_control_is_refused_as_that_control_without_help_from_the_nonfinite_map(
    hand_record, tmp_path, monkeypatch, name,
):
    """A NaN control is ALSO refused by the sanitiser's `nonfinite` map, because a
    record written with a NaN in it carries that map -- so every read-phase test
    that goes through `write_record` has a second line of defence behind the
    comparison it means to test, and a `> 0` that let the NaN through would be
    caught by the wrong check. Here the loader hands the read phase records with
    the NaN in the control and an EMPTY map, which `load_record` cannot produce, so
    the refusal can only be the control's own comparison.

    THE MUTATIONS THIS EXISTS FOR, each run against it: a zero control read with
    `> 0`, and the identity residual read with `residual > IDENTITY_TOLERANCE`."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    for record in records.values():
        record["nonfinite"] = {}
    records[SECOND_CELL]["controls"][name] = NAN
    monkeypatch.setattr(script, "load_headroom", lambda directory, arms, seeds: records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(_read_args(tmp_path))
    message = str(caught.value)
    assert "frozen_ssl seed 1" in message and f"{name}=nan" in message, message
    assert "non-finite" not in message


@pytest.mark.parametrize("bad", [NAN, H.IDENTITY_TOLERANCE * 2, INF], ids=["nan", "above", "inf"])
def test_an_identity_residual_beyond_the_tolerance_or_not_a_number_is_refused(
    hand_record, tmp_path, bad,
):
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    records[LAST_CELL]["controls"]["identity_residual"] = bad
    _write_records(tmp_path, records)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(_read_args(tmp_path))
    assert "random_vit seed 2" in str(caught.value) and "identity_residual" in str(caught.value)


@pytest.mark.parametrize(
    "where",
    [("secondary", "iid_2se", "3", "skill", 7), ("curves", "floor_position", 11)],
    ids=["secondary", "curves"],
)
def test_a_record_carrying_any_non_finite_value_is_refused_for_the_one_it_is(
    hand_record, tmp_path, where,
):
    """`measure_cell` refuses a non-finite row before it writes, so a record with
    a non-finite value ANYWHERE -- here in two places no other refusal reads, the
    secondary figure and the canonical curves -- is not one it wrote. The
    sanitiser's `nonfinite` map names the dotted path, and so does the refusal.

    THE MUTATION THIS EXISTS FOR: deleting the `nonfinite` check from
    `require_sound_records`."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    node = records[FIRST_CELL]
    for key in where[:-2]:
        node = node[key]
    node[where[-2]][where[-1]] = NAN
    _write_records(tmp_path, records)
    path = ".".join(str(part) for part in where)
    with pytest.raises(SystemExit) as caught:
        script.read_phase(_read_args(tmp_path))
    message = str(caught.value)
    assert "frozen_ssl seed 0" in message and path in message, message


def test_negative_headroom_steps_are_recorded_and_do_not_stop_the_read(hand_record, tmp_path, capsys):
    """`negative_headroom_steps` is a recorded list that the readability gate
    reads, not a control checked against a constant: a record carrying a long
    list still reads."""
    records = _records_from(hand_record, _spec(["BETWEEN"] * 9))
    for record in records.values():
        record["controls"]["negative_headroom_steps"] = [[1, 2], [1, 8], [3, 5]]
    _write_records(tmp_path, records)
    assert script.read_phase(_read_args(tmp_path)) == 0
    assert "verdict: BETWEEN" in capsys.readouterr().out


def test_two_arms_of_the_nine_are_a_readable_plan_and_only_they_are_pooled(
    hand_record, tmp_path, capsys,
):
    """Two arms is the bar, so it is readable; the third arm's records are on disk
    and outside the plan, so they must not be pooled."""
    _write_records(tmp_path, _records_from(hand_record, _spec(["BETWEEN"] * 9)))
    assert script.read_phase(_read_args(tmp_path, arms=("pixel_ae", "frozen_ssl"))) == 0
    out = capsys.readouterr().out
    assert "BETWEEN in 6 of 6 cells" in out, out
    assert "random_vit" not in out


# --- a refusal leaves no reading behind --------------------------------------


def _overwrite(args, cell, record) -> None:
    write_record(args.out / f"headroom_{cell[0]}_seed{cell[1]}.json", record)


def _make_control_broken(args, base):
    records = _records_from(base, _spec(["BETWEEN"] * 9))
    records[LAST_CELL]["controls"]["persistence_divergence"] = NAN
    _overwrite(args, LAST_CELL, records[LAST_CELL])


def _make_protocol_disagree(args, base):
    records = _records_from(base, _spec(["BETWEEN"] * 9))
    DISAGREEMENTS["git_sha"](records[LAST_CELL])
    _overwrite(args, LAST_CELL, records[LAST_CELL])


def _make_cell_missing(args, base):
    (args.out / f"headroom_{LAST_CELL[0]}_seed{LAST_CELL[1]}.json").unlink()


def _make_plan_too_narrow(args, base):
    args.arms = ["pixel_ae"]


def _make_seeds_short(args, base):
    args.seeds = [0, 1]


STALE_REFUSALS = {
    # name: (what to change after a good read, what the second read does)
    "a broken control": (_make_control_broken, SystemExit),
    "a protocol disagreement": (_make_protocol_disagree, SystemExit),
    "a missing cell": (_make_cell_missing, 11),
    "a plan too narrow to read": (_make_plan_too_narrow, SystemExit),
    "an arm short of seeds": (_make_seeds_short, ValueError),
}


@pytest.mark.parametrize("refusal", sorted(STALE_REFUSALS))
def test_a_refusal_removes_the_headroom_txt_an_earlier_read_left_and_nothing_else(
    hand_record, tmp_path, capsys, monkeypatch, refusal,
):
    """A reading from an earlier read of the SAME directory is not a reading of
    what is there now. If it survived a refusal, `headroom.txt` would sit beside
    a refusal and nothing in the directory would say which of the two a later
    reader holds. The file is the current reading or it is absent.

    A successful read comes first, so the file is a real reading and its absence
    afterwards is the refusal's doing. The unlink happens before ANY refusal is
    reached -- the plan check, the load, the protocol check, the reading's own
    ValueError -- and touches only `<--out>/headroom.txt`: a `headroom.txt` beside
    the output directory, one in the current directory and a `.txt` that is not
    `headroom.txt` inside it must all survive.

    THE MUTATIONS THIS EXISTS FOR: removing the unlink, and moving it after the
    checks that refuse."""
    change, outcome = STALE_REFUSALS[refusal]
    out = tmp_path / "out"
    _write_records(out, _records_from(hand_record, _spec(["BETWEEN"] * 9)))
    args = _read_args(out)
    stale = out / "headroom.txt"
    assert script.read_phase(args) == 0
    assert "verdict: BETWEEN" in stale.read_text()
    capsys.readouterr()

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    bystanders = [tmp_path / "headroom.txt", cwd / "headroom.txt", out / "notes.txt"]
    for bystander in bystanders:
        bystander.write_text("not this one")

    change(args, hand_record)
    before = sorted(p.name for p in out.iterdir())
    assert "headroom.txt" in before
    if isinstance(outcome, int):
        assert script.read_phase(args) == outcome
        assert "verdict:" not in capsys.readouterr().out
    else:
        with pytest.raises(outcome):
            script.read_phase(args)

    assert not stale.exists(), "the reading an earlier read left is still there"
    assert sorted(p.name for p in out.iterdir()) == [n for n in before if n != "headroom.txt"]
    for bystander in bystanders:
        assert bystander.read_text() == "not this one", bystander


# ---------------------------------------------------------------------------
# main: the three phases, the plan gates, and the command line.
# ---------------------------------------------------------------------------


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
    assert _main(monkeypatch, ["--phase", "read"], read=47) == (47, ["read"])
    assert _main(monkeypatch, ["--phase", "all"], read=48) == (48, ["measure", "read"])


def test_a_failed_measure_stops_phase_all_before_it_reads(monkeypatch):
    """A measure that returned 14 wrote no record for its cell, and a read after
    it would be over a pool missing that cell."""
    assert _main(monkeypatch, ["--phase", "all"], measure=14, read=0) == (14, ["measure"])


@pytest.mark.parametrize("status", [11, 12, 14, 30])
def test_a_failed_measure_is_the_exit_status_of_phase_measure_too(monkeypatch, status):
    """`--phase measure` has no read after it, and that is exactly where a failed
    measure's status could be dropped on the floor: the guard that stops the read
    in `--phase all` is also what RETURNS the status, and a `--phase measure` that
    fell through to `EXIT_OK` would print a success after a refused 20-minute run
    -- the operator reads success and the records are not there.

    The statuses are literals (11, 12, 14 from the shared cell checks, 30 from the
    self-check), not `script.EXIT_*`, and `read` answers 0 so a status read from
    the wrong phase is a different number.

    THE MUTATION THIS EXISTS FOR, run against it: narrowing `main`'s `if status
    != EXIT_OK:` to `and args.phase == "all"`."""
    assert _main(monkeypatch, ["--phase", "measure"], measure=status, read=0) == (
        status, ["measure"],
    )


def test_a_refused_cell_is_a_nonzero_exit_through_main_with_the_real_phase(tmp_path, monkeypatch):
    """The same property one level down, through the real `measure_phase` and the
    real `main`: a cell `prepare_cell` refuses is a returned 14, and not 0. The
    brief's test patched `measure_cell` to raise and expected `SystemExit`; this
    script, like `prediction_burden.py`, RETURNS statuses and lets an exception
    propagate (see `test_a_cell_that_raises_stops_the_phase_and_is_not_swallowed`).

    THE MUTATION THIS EXISTS FOR, run against it: `return status` -> `continue`
    where `measure_phase` meets a cell `prepare_cell` refused."""
    args, _ = _phase_env(tmp_path, monkeypatch, refuse=("frozen_ssl", 1))
    status = script.main([
        "--phase", "measure", "--source", str(args.source), "--out", str(args.out),
        "--data", str(args.data), "--device", "cpu", "--arms", "pixel_ae", "frozen_ssl",
        "--seeds", "1", "2",
    ])
    assert status == script.EXIT_RECORD_MISMATCH == 14


def test_the_default_phase_is_read_and_the_parser_names_the_three():
    """A default of `measure` would make a bare invocation pay for nine cells of
    GPU time; `read` costs none and refuses by name when there are no records."""
    args = script._parser().parse_args([])
    assert args.phase == "read"
    assert script.PHASES == ("all", "measure", "read")
    assert args.out == Path("runs/m3n_motion")
    with pytest.raises(SystemExit) as caught:
        script._parser().parse_args(["--phase", "readd"])
    assert caught.value.code == 2


def test_the_parser_defaults_are_the_full_nine_cell_plan():
    args = script._parser().parse_args([])
    assert args.arms == ["pixel_ae", "frozen_ssl", "random_vit"]
    assert args.seeds == [0, 1, 2]
    assert args.ks == list(REGROUNDING_KS)
    assert args.bootstrap_seed == 0
    assert args.context is None and args.horizon is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--arms", "frozen_ssl,pixel_ae,random_vit", "--seeds", "0,1,2", "--ks", "1,3,5,15,45"],
        ["--arms", "frozen_ssl", "pixel_ae", "random_vit", "--seeds", "0", "1", "2",
         "--ks", "1", "3", "5", "15", "45"],
        ["--arms", "frozen_ssl,pixel_ae", "random_vit", "--seeds", "0,1", "2",
         "--ks", "1,3", "5,15", "45"],
    ],
    ids=["commas", "spaces", "mixed"],
)
def test_a_list_flag_reads_commas_and_spaces_alike(argv):
    """The plan's own commands pass `--arms frozen_ssl,pixel_ae,random_vit --seeds
    0,1,2 --ks 1,3,5,15,45` -- one comma-joined token -- and `prediction_burden`'s
    spelling is space-separated. Read as a single token, `"0,1,2"` is not an
    integer and `"frozen_ssl,pixel_ae,random_vit"` is not an arm, so a parser that
    took only one spelling would refuse the run's own command line after the
    checkpoints were already loaded."""
    args = script._parser().parse_args(argv)
    assert args.arms == ["frozen_ssl", "pixel_ae", "random_vit"]
    assert args.seeds == [0, 1, 2]
    assert args.ks == [1, 3, 5, 15, 45]
    assert all(isinstance(value, int) for value in args.seeds + args.ks)


def test_a_list_flag_drops_repeats_and_refuses_what_it_cannot_read():
    args = script._parser().parse_args(["--arms", "pixel_ae,pixel_ae", "--seeds", "1,1,2", "--ks", "45,1,45"])
    assert (args.arms, args.seeds, args.ks) == (["pixel_ae"], [1, 2], [45, 1])
    for argv in (["--arms", "pixel_aee"], ["--seeds", "zero"], ["--ks", "1,,x"], ["--arms", ","]):
        with pytest.raises(SystemExit) as caught:
            script._parser().parse_args(argv)
        assert caught.value.code == 2, argv


def test_phase_all_refuses_a_plan_it_could_not_read_before_measuring_anything(monkeypatch):
    """`--phase all --arms pixel_ae` would be the whole measure followed by a
    refusal at the read. `--phase measure` is allowed the same plan: the
    milestone's smoke is one cell.

    THE MUTATION THIS EXISTS FOR: dropping `require_readable_plan` from `main`."""
    ran = []
    monkeypatch.setattr(script, "measure_phase", lambda args: ran.append("measure") or 0)
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "all", "--arms", "pixel_ae"])
    assert f"ARMS_REQUIRED={H.ARMS_REQUIRED}" in str(caught.value)
    assert ran == []
    assert _main(monkeypatch, ["--phase", "measure", "--arms", "pixel_ae"]) == (0, ["measure"])


@pytest.mark.parametrize(
    "seeds, distinct",
    [
        pytest.param([str(i) for i in range(H.SEEDS_MINIMUM - 1)], H.SEEDS_MINIMUM - 1,
                     id="one short of the minimum"),
        pytest.param(["5"], 1, id="one seed"),
        # `--seeds a a a` is ONE seed: counting the list would let it through, and
        # the read would then find an arm with one seed.
        pytest.param(["5"] * H.SEEDS_MINIMUM, 1, id="enough listed, one distinct"),
    ],
)
def test_phase_all_refuses_too_few_seeds_before_measuring_anything(monkeypatch, seeds, distinct):
    """`--phase all --seeds 0 1` would measure all six cells and then raise at the
    read: every arm carries two seeds, and `reading_headroom` refuses an arm short
    of `SEEDS_MINIMUM`. The plan is known at the start, so the refusal is made
    there -- by name, and `ran == []` is what says it came BEFORE the measure.

    THE MUTATIONS THIS EXISTS FOR, each run against it: dropping the check from
    `main`; and counting the seeds as listed instead of as distinct -- which takes
    BOTH dedups out, the parser's (`_Csv`) and `_plan`'s, because either alone
    leaves the third case refused (`test_the_phase_measures_a_repeated_arm_once`
    is what sees `_plan`'s on its own)."""
    ran = []
    monkeypatch.setattr(script, "measure_phase", lambda args: ran.append("measure") or 0)
    monkeypatch.setattr(script, "read_phase", lambda args: ran.append("read") or 0)
    with pytest.raises(SystemExit) as caught:
        script.main(["--phase", "all", "--seeds", *seeds])
    message = str(caught.value)
    assert f"SEEDS_MINIMUM={H.SEEDS_MINIMUM}" in message, message
    assert f"{distinct} seed(s)" in message, message
    assert ran == []


def test_a_seed_plan_short_of_the_minimum_is_refused_only_where_it_would_waste_a_measure(
    monkeypatch,
):
    """The seed check belongs to `--phase all` and to nothing else: `--phase
    measure` is the milestone's smoke and may run one seed, and `--phase read`
    must REACH `read_phase` with a short plan (it is where the arm-short
    ValueError comes from, which a check in front of it would make unreachable).
    Exactly `SEEDS_MINIMUM` seeds is a plan -- the boundary a `<=` would refuse.

    THE MUTATIONS THIS EXISTS FOR, each run against it: the check applied to
    `--phase measure`, applied to `--phase read`, and `<=` for `<`."""
    short = ["--seeds", *(str(i) for i in range(H.SEEDS_MINIMUM - 1))]
    assert _main(monkeypatch, ["--phase", "measure", *short]) == (0, ["measure"])
    assert _main(monkeypatch, ["--phase", "read", *short], read=48) == (48, ["read"])
    enough = ["--seeds", *(str(i) for i in range(H.SEEDS_MINIMUM))]
    assert _main(monkeypatch, ["--phase", "all", *enough]) == (0, ["measure", "read"])


def test_main_reads_the_arms_seeds_and_directory_the_command_line_names(
    monkeypatch, hand_record, tmp_path, capsys,
):
    """The flag names reach the fields `read_phase` reads, through the real read
    phase. The measure phase is made unreachable: a `--phase read` that fell into
    it would load the real checkpoints from `--source`'s default."""
    def forbidden(args):
        raise AssertionError("--phase read reached the measure phase")

    monkeypatch.setattr(script, "measure_phase", forbidden)
    _write_records(tmp_path / "good", _records_from(hand_record, _spec(["BETWEEN"] * 9)))
    argv = ["--phase", "read", "--out", str(tmp_path / "good"), "--arms", "pixel_ae,frozen_ssl",
            "--seeds", "0,1,2"]
    assert script.main(argv) == 0
    assert "BETWEEN in 6 of 6 cells" in capsys.readouterr().out
    _write_records(tmp_path / "unreadable", _records_from(hand_record, _spec(["UNREADABLE"] * 9)))
    assert script.main(["--phase", "read", "--out", str(tmp_path / "unreadable")]) == 47
