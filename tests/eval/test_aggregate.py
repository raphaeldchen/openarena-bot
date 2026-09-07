"""The cross-seed aggregation and the M3 gate verdict.

This is the last computation before a human reads a number off a 33-hour study,
so the tests below are written against the ways a WRONG number can look right:

  * a column that holds its neighbour's value -- every fixture number is
    pairwise distinct, and `test_the_fixture_values_are_pairwise_distinct_...`
    asserts that property itself, because an assertion whose two sides are the
    same number cannot fail;
  * a per-seed column indexed by POSITION rather than by seed, which puts seed
    2's gap under "seed 1" as soon as one cell is missing;
  * a record whose filename and contents name different cells, which is the
    only failure here with no symptom at all;
  * a criterion quietly dropped from the conjunction, or averaged away;
  * a mean whose denominator moved when a cell measured nothing.

WHY THE FIXTURE NUMBERS ARE GENERATED RATHER THAN WRITTEN OUT ONE BY ONE.
Twenty-odd fields across nine cells plus six curves is more than four hundred
values, and a hand-written table that size acquires collisions faster than
anyone notices them. So each FIELD gets a hand-written base, 100 apart, and
each CELL adds an offset -- 0 to 13, unevenly spaced on purpose (see
SEED_OFFSET). The bases are the part a human chooses and the part that can
therefore be wrong: two bases set within 13 of each other collide, and the
invariant test fails by name. It has already caught three. The generated part cannot
drift on its own, and the bases stay readable in the rendered tables: a 2100 in
a column means `latent_r2`, a 1000 means `position.gap_final`, so the
character-for-character table assertions below can be checked by eye against
the field each column claims to hold.
"""

import builtins
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval import aggregate
from mbfps.eval.aggregate import (
    CURVE_NAMES,
    GATE_CRITERIA,
    GATE_METRIC,
    MAX_DEGENERATE_STEPS,
    MislabelledRecord,
    RecordsUnusable,
    UnreadableRecord,
    evaluate_gate,
    load_records,
    mean_curve,
    per_arm,
    record_cell,
    record_names_its_file,
    ridge_groups,
)
from mbfps.eval.study import (
    StudyJob,
    job_record_path,
    run_job,
    write_record,
)
from mbfps.eval.summary import METRICS
from mbfps.utils.config import ARMS

_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report_study = _load_script("report_study")
run_study = _load_script("run_study")

SEEDS = (0, 1, 2)

# ---------------------------------------------------------------------------
# THE FIXTURE, PAIRWISE DISTINCT BY CONSTRUCTION.
# ---------------------------------------------------------------------------

FIXTURE_HORIZON = 4
"""How many imagined steps the fixture's curves cover.

Small, and DIFFERENT from `FIXTURE_CONTEXT`: `_curves_ok` checks the curve
length against the record's `horizon`, and a fixture where the two agreed could
not tell that check from one against `context`.
"""
FIXTURE_CONTEXT = 2
FIXTURE_STEPS = 11
FIXTURE_SEQ_LEN = 6
"""Configuration fields the aggregation neither reads nor renders.

Deliberately outside the distinctness table below: nothing compares them to
anything, so a distinct value here would be decoration. `horizon` is the
exception and is checked against the curve length, which is why it differs from
`context`.
"""

FIELD_BASE = {
    "position.gap_final": 1000,
    "position.gap_mean": 1100,
    "position.band_median": 1200,
    "angle.gap_final": 1300,
    "angle.gap_mean": 1400,
    "angle.band_median": 1500,
    "steps_per_second": 1600,
    "kl_rate_above_free_bits": 1700,
    "kl_dyn_max": 1800,
    "loss_last20": 1900,
    "filtering.criterion_4.latent_r2": 2100,
    "filtering.criterion_4.embedding_r2": 2200,
    "filtering.gain.gain": 2300,
    "filtering.gain.ci_low": 2400,
    "filtering.gain.ci_high": 2500,
    "filtering.gain.joint_r2": 2600,
    "filtering.gain.embedding_r2": 2700,
    "filtering.gain.n_scored_windows": 2800,
    "reward.mse": 2900,
    "reward.baseline_mse": 3000,
    "reward.r2": 3100,
    "reward.n_reward_events": 3200,
    "reward.n_steps": 3300,
    # THE TWO COUNT COLUMNS, WHICH USED TO BE ZERO IN EVERY CELL OF BOTH.
    # `degenerate` counts steps whose band was POSITIVE but numerically junk
    # (the gate criterion) and `floor>=pers` counts steps whose band was
    # NON-POSITIVE (which is why `gap_closed` is NaN by contract there, and in
    # a study whose bands are non-positive it is the only column that explains
    # a row of +nan). They mean opposite things and sit side by side, and with
    # 0 in both, in all nine cells, each column rendered the other's field
    # without a single assertion changing -- the L1 species, in the headline
    # table. `steps_degenerate` cannot be given a value here (MAX_DEGENERATE_
    # STEPS is 0, so a non-zero one would make this fixture a study that fails
    # its own gate), so the distinctness has to come from this side.
    "position.steps_floor_above_persistence": 3400,
    "angle.steps_floor_above_persistence": 3500,
    # `seconds` is rendered as HOURS at two decimals, so its cells have to be
    # more than 18 seconds apart or nine distinct values render as one string
    # -- which they did, and the whole wall_h column read 0.56 for every cell.
    # Its own range because `FIELD_SCALE` spreads it over 1300.
    "seconds": 20_000,
}
"""One base per field a mutation could exchange for another, 100 apart.

100 apart because a cell adds up to 13 to its field's base; two bases closer
than that would let one cell of one field equal another cell of another, and
every assertion telling the two apart would become a tautology.
"""

FIELD_SCALE = {"seconds": 100}
"""Fields whose cells need spreading further than one unit apart.

Only `seconds`, and only because it is the one field the report renders through
a TRANSFORM: `wall_h` divides by 3600 and prints two decimals, so cells one
second apart are distinct as values and identical as text. Distinctness has to
hold in whatever form the assertion actually reads.
"""

SEED_OFFSET = (0, 1, 3)
"""What each seed adds to its field's base. DELIBERATELY NOT EQUALLY SPACED.

With 0, 1, 2 the mean of seeds 0 and 2 equals the mean of all three, so every
"the denominator moved" assertion would be reading the same number either way
-- the L1 species, arithmetic rather than typographic. 0, 1, 3 gives 1001.5
against 1001.3333.
"""

CURVE_BASE = {
    "rssm_position": 5000,
    "persistence_position": 5200,
    "floor_position": 5400,
    "rssm_angle": 5600,
    "persistence_angle": 5800,
    "floor_angle": 6000,
}
"""One base per curve. A cell adds 10x its offset and a step adds 1.

200 apart, not 100: the largest offset is 10, so a cell contributes up to 100
and bases 100 apart put `rssm_angle` of the first cell exactly on top of
`floor_position` of the last. The invariant test found that, which is what it
is for. The whole six-curve, nine-cell, four-step block is now distinct within
its own range and distinct from `FIELD_BASE`'s.
"""

JOINT_RIDGE = 1e3
EMBEDDING_RIDGE = 1e5
ALT_JOINT_RIDGE = 1e7
ALT_EMBEDDING_RIDGE = 1e1
"""The two ridge decades, DIFFERENT from each other on purpose.

They are outside the distinctness table because every cell of a coherent study
holds the same pair -- that is what `gain_ridges_agree` means -- so they cannot
vary per cell. What they must not do is equal each other: `joint_ridge` and
`embedding_ridge` sit in adjacent columns of the filtering table, and a
rendering that printed one under the other's heading would be invisible if they
agreed. Both are real values from `probe.RIDGES`.

`ALT_*` are the decades a cell moves TO when a test needs two cells to disagree
on ONE part of `ridge_groups`'s three-part key. All four must be pairwise
distinct, and distinct as ".1e" TEXT, or a test asserting that a cell moved
group would read the same number either way; the invariant test asserts it.
The key is `(ridge_selected, joint_ridge, embedding_ridge)` and every fixture
cell used to carry the same three, so two of its three parts could be dropped
without a single test noticing -- W07 merged cells that used different
embedding ridges into one group, and W08 merged a cell that ran ridge
selection with one that did not, silencing `ridge_block`'s second warning.
"""

#: The three booleans the report renders, as the RENDERING fixture holds them.
#
# `False == 0` in Python, so booleans cannot live in a distinctness table of
# numbers; they get this guard instead. Two rules, and the invariant test
# asserts both:
#
#   * within one table, no two booleans may agree, or a rendering that printed
#     one under the other's heading is invisible. `latent_beats_embedding` and
#     `ridge_selected` share the filtering table and therefore differ;
#     `is_degenerate` is alone in the reward table and is only required to
#     differ from the literal its own mutation would write.
#   * each is the NEGATION of the literal a mutation replacing it would use --
#     a bool has two values, so one rendering excludes only one literal, and
#     the both-ways tests below render the block again with all three flipped.
RENDERED_BOOLEANS = {
    "filtering.criterion_4.latent_beats_embedding": False,
    "filtering.gain.ridge_selected": True,
    "reward.is_degenerate": True,
}


def _offset(arm, seed) -> int:
    """This cell's addend, distinct for every (arm, seed) of the study."""
    arm_index = ARMS.index(arm) if arm in ARMS else len(ARMS)
    seed_index = SEEDS.index(seed) if seed in SEEDS else len(SEEDS)
    seed_offset = (SEED_OFFSET[seed_index] if seed_index < len(SEED_OFFSET)
                   else max(SEED_OFFSET) + 1)
    return arm_index * (max(SEED_OFFSET) + 2) + seed_offset


def _value(field: str, arm, seed) -> float:
    return float(
        FIELD_BASE[field] + _offset(arm, seed) * FIELD_SCALE.get(field, 1))


def _curve(name: str, arm, seed) -> list[float]:
    base = CURVE_BASE[name] + 10 * _offset(arm, seed)
    return [float(base + step) for step in range(FIXTURE_HORIZON)]


def _metric_block(metric: str, arm, seed) -> dict:
    """One `metric_summary`-shaped block.

    `steps_degenerate` is 0 because the fixture is a study that PASSES and the
    allowance is zero; `steps_floor_above_persistence` is not gated and is
    therefore distinct per cell and per metric, so the two adjacent count
    columns of the metric table can be told apart.
    """
    return {
        "final_model": 1.0,
        "final_persistence": 2.0,
        "final_floor": 0.5,
        "band_min": 1.0,
        "band_median": _value(f"{metric}.band_median", arm, seed),
        "band_max": 3.0,
        "relative_min": 0.4,
        "relative_median": 0.5,
        "relative_max": 0.6,
        "steps_floor_above_persistence": int(
            _value(f"{metric}.steps_floor_above_persistence", arm, seed)),
        # Zero in every cell, and it has to be: `MAX_DEGENERATE_STEPS` is 0, so
        # any other value would make this a study that fails `band_is_usable`.
        # Its neighbour above carries the distinctness for the pair.
        "steps_degenerate": 0,
        # PRESENT FOR RECORD FIDELITY AND DELIBERATELY NOT ASSERTED ON.
        # `metric_summary` writes it and `scripts/run_study.py` renders it as
        # `finite=`, but neither `aggregate` nor `report_study` reads it --
        # they count finite `gap_final`s themselves and publish that as
        # `n_seeds_with_finite_gap`. So it is outside the distinctness table
        # for the same reason `FIXTURE_STEPS` is: nothing here compares it to
        # anything, and a distinct value would be decoration. Written down
        # rather than left silent, because a fixture field nothing asserts on
        # is otherwise indistinguishable from one whose assertion was lost.
        "gap_finite": FIXTURE_HORIZON,
        "gap_mean": _value(f"{metric}.gap_mean", arm, seed),
        "gap_min": 0.1,
        "gap_max": 0.9,
        "gap_final": _value(f"{metric}.gap_final", arm, seed),
        "n_steps": FIXTURE_HORIZON,
    }


def _record(arm, seed, *, latent_wins=True, reward_degenerate=True,
            ridge_selected=True, joint_ridge=JOINT_RIDGE,
            embedding_ridge=EMBEDDING_RIDGE):
    """A complete record for one cell, shaped exactly as `run_job` writes one.

    `filtering` is NESTED -- `{"criterion_4": {...}, "gain": {...}}` -- because
    that is what `run_job` writes: gate criterion 4 and its bottleneck-free
    companion are BOTH recorded, and an aggregation reading
    `filtering["latent_beats_embedding"]` would raise on every real record.
    """
    return {
        "arm": arm,
        "seed": seed,
        "steps": FIXTURE_STEPS,
        "seq_len": FIXTURE_SEQ_LEN,
        "context": FIXTURE_CONTEXT,
        "horizon": FIXTURE_HORIZON,
        "split_seed": 0,
        "seconds": _value("seconds", arm, seed),
        "steps_per_second": _value("steps_per_second", arm, seed),
        "kl_rate_above_free_bits": _value("kl_rate_above_free_bits", arm, seed),
        "kl_dyn_max": _value("kl_dyn_max", arm, seed),
        "loss_last20": _value("loss_last20", arm, seed),
        "episodes": {"train": ["ep_train.npz"], "val": ["ep_val.npz"]},
        "probe": {
            "latent_ridge": JOINT_RIDGE,
            "latent_ridge_selected": True,
            "latent_selection_r2": 0.5,
            "embedding_ridge": EMBEDDING_RIDGE,
            "embedding_ridge_selected": True,
            "embedding_selection_r2": 0.6,
        },
        "position": _metric_block("position", arm, seed),
        "angle": _metric_block("angle", arm, seed),
        "filtering": {
            "criterion_4": {
                "latent_r2": _value(
                    "filtering.criterion_4.latent_r2", arm, seed),
                "embedding_r2": _value(
                    "filtering.criterion_4.embedding_r2", arm, seed),
                "latent_beats_embedding": latent_wins,
            },
            "gain": {
                "gain": _value("filtering.gain.gain", arm, seed),
                "joint_r2": _value("filtering.gain.joint_r2", arm, seed),
                "embedding_r2": _value(
                    "filtering.gain.embedding_r2", arm, seed),
                "ci_low": _value("filtering.gain.ci_low", arm, seed),
                "ci_high": _value("filtering.gain.ci_high", arm, seed),
                "confidence": 0.95,
                "n_scored_windows": int(
                    _value("filtering.gain.n_scored_windows", arm, seed)),
                "ridge_selected": ridge_selected,
                "joint_ridge": joint_ridge,
                "embedding_ridge": embedding_ridge,
            },
        },
        "reward": {
            "mse": _value("reward.mse", arm, seed),
            "baseline_mse": _value("reward.baseline_mse", arm, seed),
            "r2": _value("reward.r2", arm, seed),
            "n_steps": int(_value("reward.n_steps", arm, seed)),
            "n_reward_events": int(
                _value("reward.n_reward_events", arm, seed)),
            "is_degenerate": reward_degenerate,
        },
        "curves": {name: _curve(name, arm, seed) for name in CURVE_NAMES},
    }


def _study(**kwargs) -> list[dict]:
    """All nine cells: three arms, three seeds, every value distinct."""
    return [_record(arm, seed, **kwargs) for arm in ARMS for seed in SEEDS]


def _rendering_study() -> list[dict]:
    """The nine cells with the booleans set as `RENDERED_BOOLEANS` says.

    Not the same fixture as `_study()`: the gate-passing study needs
    `latent_beats_embedding` True, and a rendering fixture needs it to be the
    OPPOSITE of the literal a mutation would write. `False` here is also what
    the real trained model measured.
    """
    return _study(
        latent_wins=RENDERED_BOOLEANS[
            "filtering.criterion_4.latent_beats_embedding"],
        reward_degenerate=RENDERED_BOOLEANS["reward.is_degenerate"],
        ridge_selected=RENDERED_BOOLEANS["filtering.gain.ridge_selected"],
    )


def _write(tmp_path, records) -> Path:
    """Write each record to the name its own arm and seed ask for.

    Through `write_record`, never `json.dumps`: only that adds the `nonfinite`
    map that `load_record` inverts, and a fixture that wrote plain JSON could
    not tell a NaN restored from a NaN that was never lost.
    """
    for record in records:
        write_record(
            job_record_path(tmp_path, StudyJob(record["arm"], record["seed"])),
            record,
        )
    return tmp_path


def _cell(records, arm, seed) -> dict:
    """The one record for this cell, so a test never indexes a list by luck."""
    matches = [r for r in records if (r["arm"], r["seed"]) == (arm, seed)]
    assert len(matches) == 1, f"{arm}/s{seed} appears {len(matches)} times"
    return matches[0]


def _row(text: str, prefix: str) -> str:
    """The one line of `text` starting with `prefix`.

    A `in output` assertion is answered by any line that happens to contain the
    substring -- a banner, a neighbouring row, the same value under another
    heading. This pins the row down first and the caller then compares the
    WHOLE line, so nothing else in the report can satisfy it.
    """
    matches = [line for line in text.splitlines() if line.startswith(prefix)]
    assert len(matches) == 1, (
        f"{len(matches)} lines start with {prefix!r}; the assertion below "
        f"would be answered by whichever one came first:\n{text}")
    return matches[0]


# ---------------------------------------------------------------------------
# the fixture itself: the guard on the SPECIES, not the instance
# ---------------------------------------------------------------------------

def test_the_fixture_values_are_pairwise_distinct_so_no_assertion_is_vacuous():
    """Every "this column holds that field" assertion in this module can only
    fail if the two fields hold different numbers. That is a property of the
    fixture, and Tasks 4 and 5 lost ten review rounds to it."""
    values: dict[str, float] = {}
    for arm in ARMS:
        for seed in SEEDS:
            for field in FIELD_BASE:
                values[f"{field}@{arm}/s{seed}"] = _value(field, arm, seed)
            for name in CURVE_NAMES:
                for step, value in enumerate(_curve(name, arm, seed)):
                    values[f"curve.{name}[{step}]@{arm}/s{seed}"] = value

    seen: dict[float, list[str]] = {}
    for name, value in values.items():
        seen.setdefault(value, []).append(name)
    collisions = {v: names for v, names in seen.items() if len(names) > 1}
    assert not collisions, (
        "these fixture values collide, so every assertion telling one from "
        f"the other is now a tautology: {collisions}")
    assert len(values) == len(FIELD_BASE) * 9 + len(CURVE_NAMES) * 9 * FIXTURE_HORIZON
    assert len(values) == 450, (
        "a field was dropped from the table rather than made distinct; the "
        "count is spelled out so that deleting a colliding entry cannot be "
        "mistaken for fixing it")

    # The rendered values must also be distinct AS TEXT: two floats that differ
    # in the ninth decimal render identically under "+.4f", and the table
    # assertions compare text.
    assert len({f"{v:+.4f}" for v in values.values()}) == len(values)

    # ...and distinct through the one field the report TRANSFORMS before
    # rendering it. `wall_h` is `seconds / 3600` at two decimals, and nine
    # values one second apart print as one string: the column read 0.56 in
    # every row until `FIELD_SCALE` spread them.
    hours = {f"{_value('seconds', arm, seed) / 3600.0:.2f}"
             for arm in ARMS for seed in SEEDS}
    assert len(hours) == 9

    # An arm's three seeds must not be equally spaced, or the mean over two of
    # them equals the mean over all three and every assertion about a
    # denominator that moved reads the same number either way.
    gaps = [_value("position.gap_final", "cnn", seed) for seed in SEEDS]
    assert (gaps[0] + gaps[2]) / 2 != sum(gaps) / 3

    # The two ridges share a table row and must not agree; see JOINT_RIDGE.
    # All four decades, because `ridge_groups`'s key is the PAIR plus the flag
    # and a test that moves one cell to `ALT_JOINT_RIDGE` proves nothing if
    # that is the value the cell already held.
    ridges = [JOINT_RIDGE, EMBEDDING_RIDGE, ALT_JOINT_RIDGE,
              ALT_EMBEDDING_RIDGE]
    assert len(set(ridges)) == 4
    assert len({f"{r:.1e}" for r in ridges}) == 4

    # The two count columns of the metric table mean OPPOSITE things and sit
    # side by side. `steps_degenerate` is pinned at 0 by the gate's own
    # allowance, so the pair is only distinguishable if the other one is not.
    for arm in ARMS:
        for seed in SEEDS:
            for metric in METRICS:
                block = _metric_block(metric, arm, seed)
                assert block["steps_degenerate"] == 0
                assert block["steps_floor_above_persistence"] != 0, (
                    "with 0 in both, each of the two adjacent count columns "
                    "renders the other's field and no assertion changes")
    assert len({
        _metric_block(metric, arm, seed)["steps_floor_above_persistence"]
        for arm in ARMS for seed in SEEDS for metric in METRICS}) == 18

    # The booleans, which cannot be in the table above (`False == 0`).
    assert all(v is True or v is False for v in RENDERED_BOOLEANS.values())
    assert (RENDERED_BOOLEANS["filtering.criterion_4.latent_beats_embedding"]
            is not RENDERED_BOOLEANS["filtering.gain.ridge_selected"]), (
        "these two booleans share the filtering table, so a rendering that "
        "printed one under the other's heading is invisible if they agree")

    # `_curves_ok` compares the curve length against `horizon`; a fixture where
    # `context` equalled `horizon` could not tell that check from one against
    # `context`.
    assert FIXTURE_CONTEXT != FIXTURE_HORIZON
    assert len(_curve("rssm_position", "cnn", 0)) == FIXTURE_HORIZON

    # All three arms, including `frozen_ssl` -- the arm whose backbone name
    # differs from its own and the one Task 4's tests never ran.
    assert {r["arm"] for r in _study()} == set(ARMS) == {
        "cnn", "frozen_ssl", "random_vit"}
    assert {r["seed"] for r in _study()} == set(SEEDS)
    assert len(_study()) == 9


def test_the_aggregation_expects_the_cells_the_driver_runs():
    """A private copy of either tuple would let the driver and the gate
    disagree about what the study is: nine cells run and six expected reports
    NOT PASSED on a complete study; the other way round calls a 3x2 study
    complete."""
    assert aggregate.ARMS is ARMS
    assert aggregate.SEEDS == run_study.SEEDS
    assert aggregate.METRICS is METRICS
    assert set(CURVE_NAMES) == {
        "rssm_position", "persistence_position", "floor_position",
        "rssm_angle", "persistence_angle", "floor_angle"}, (
        "the six curves are what spec criterion 2 asks to be drawn on the "
        "same axes; derived from METRICS, pinned here against a literal")
    assert GATE_METRIC in METRICS


def test_the_gate_reports_exactly_these_criteria():
    """Pinned against a hand-written literal rather than against
    `GATE_CRITERIA` itself.

    Parametrising over the collection under test is how a criterion deleted
    from the gate deletes its own test case: the suite stays green and the
    conjunction is over one fewer term. This list is written out by hand, so
    dropping `filtering_beats_embedding` -- the one criterion the real study
    fails -- fails HERE, by name.
    """
    assert GATE_CRITERIA == (
        "all_nine_cells_present",
        "no_duplicate_cells",
        "every_record_is_a_study_cell",
        "beats_persistence",
        "band_is_usable",
        "filtering_beats_embedding",
        "reward_reported",
        "curves_produced",
    )
    verdict = evaluate_gate(_study())
    assert tuple(verdict["criteria"]) == GATE_CRITERIA
    assert set(verdict["not_evaluated"]) == {"invariant_tests_green"}, (
        "spec section 4 criterion 5 cannot be computed from records, and a "
        "criterion nobody checked must not be reported by being absent")


# ---------------------------------------------------------------------------
# load_records: what is on disk, read the one way that inverts the writer
# ---------------------------------------------------------------------------

def test_load_records_reads_every_cell_in_the_directory(tmp_path):
    records = load_records(_write(tmp_path, _study()))
    assert len(records) == 9
    assert {(r["arm"], r["seed"]) for r in records} == {
        (arm, seed) for arm in ARMS for seed in SEEDS}


def test_load_records_restores_the_non_finite_floats_null_stands_for(tmp_path):
    """THE DEFECT THIS PROJECT ALREADY PAID FOR ONCE, ONE LAYER UP.

    `write_record` cannot write a bare NaN -- it is not JSON -- so it writes
    `null` plus a `nonfinite` map naming the dotted path and the token. Only
    `study.load_record` inverts that. Read with `json.loads`, `gap_final` comes
    back as `None`, and the first thing the aggregation does with it is compare
    it to zero, which raises `TypeError`. NaN here is the NORMAL path:
    `gap_closed` is NaN by contract whenever the band is non-positive.

    Both directions are asserted, because "not None" alone would be satisfied
    by any number at all, including a 0.0 substituted for the undefined value.
    """
    records = _study()
    _cell(records, "frozen_ssl", 1)["position"]["gap_final"] = float("nan")
    _cell(records, "frozen_ssl", 1)["reward"]["r2"] = float("-inf")
    _write(tmp_path, records)

    raw = json.loads(
        (tmp_path / "result_frozen_ssl_seed1.json").read_text())
    assert raw["position"]["gap_final"] is None, (
        "the fixture must really have gone through the null-plus-map "
        "encoding, or this test proves nothing about restoring it")
    assert raw["nonfinite"]["position.gap_final"] == "nan"

    loaded = _cell(load_records(tmp_path), "frozen_ssl", 1)
    assert loaded["position"]["gap_final"] is not None
    assert math.isnan(loaded["position"]["gap_final"])
    assert loaded["reward"]["r2"] == float("-inf")
    # ...and the finite neighbours are untouched, so the restore did not write
    # over the wrong path.
    assert loaded["position"]["gap_mean"] == _value(
        "position.gap_mean", "frozen_ssl", 1)


def test_a_nan_record_survives_the_whole_aggregation(tmp_path):
    """The end-to-end version of the test above: a NaN cell must not raise.

    Reading with `json.loads` leaves `None`, and `gaps > 0` on a `None` raises
    `TypeError` -- after the GPU hours, in the last step of the pipeline.
    """
    records = _study()
    _cell(records, "cnn", 2)["position"]["gap_final"] = float("nan")
    verdict = evaluate_gate(load_records(_write(tmp_path, records)))
    assert verdict["criteria"]["beats_persistence"] is False
    assert verdict["per_arm"]["cnn"]["n_seeds_with_finite_gap"] == 2


def test_load_records_ignores_files_that_are_not_records(tmp_path):
    """The glob is what keeps the driver's lock file and its quarantined
    mislabelled records out of the aggregation."""
    _write(tmp_path, _study())
    (tmp_path / "study.lock").write_text('{"pid": 1}')
    (tmp_path / "result_cnn_seed0.json.mislabelled").write_text(
        json.dumps(_record("random_vit", 1)))
    (tmp_path / "world_model_cnn_seed0.pt").write_bytes(b"not json")
    (tmp_path / "notes.txt").write_text("hello")
    assert len(load_records(tmp_path)) == 9


def test_load_records_refuses_a_record_whose_arm_disagrees_with_its_filename(
        tmp_path):
    """Reporting one arm's numbers under another arm's name is the worst
    outcome this study has, and the only one with no symptom."""
    _write(tmp_path, _study())
    path = tmp_path / "result_cnn_seed0.json"
    path.write_text(json.dumps({**_record("frozen_ssl", 0), "nonfinite": {}}))
    with pytest.raises(MislabelledRecord) as error:
        load_records(tmp_path)
    assert "result_cnn_seed0.json" in str(error.value)
    assert "arm='frozen_ssl'" in str(error.value)
    assert "result_frozen_ssl_seed0.json" in str(error.value)


def test_load_records_refuses_a_record_whose_seed_disagrees_with_its_filename(
        tmp_path):
    """The seed alone, so the arm check cannot be what caught it."""
    _write(tmp_path, _study())
    path = tmp_path / "result_random_vit_seed2.json"
    path.write_text(json.dumps({**_record("random_vit", 1), "nonfinite": {}}))
    with pytest.raises(MislabelledRecord) as error:
        load_records(tmp_path)
    message = str(error.value)
    assert "result_random_vit_seed2.json" in message
    assert "seed=1" in message
    assert "result_random_vit_seed1.json" in message


def test_load_records_refuses_a_record_whose_seed_is_not_an_integer(tmp_path):
    """`seed: "0"` satisfies every f-string comparison and no set membership.

    Left alone it would pass the filename check -- `f"seed{'0'}"` is
    `"seed0"` -- and then be counted as a MISSING cell with the file sitting
    right there. A study reported incomplete with all nine records present is
    a morning nobody gets back.
    """
    _write(tmp_path, _study())
    path = tmp_path / "result_cnn_seed0.json"
    path.write_text(json.dumps({**_record("cnn", 0), "seed": "0",
                                "nonfinite": {}}))
    with pytest.raises(MislabelledRecord) as error:
        load_records(tmp_path)
    assert "seed='0'" in str(error.value), (
        "the message must show the TYPE; str() renders 0 and '0' identically "
        "and the operator cannot act on a message that hides the difference")


def test_load_records_refuses_a_file_it_cannot_read(tmp_path):
    """A file that cannot be parsed is not skipped: it may be the cell the
    table is missing, and a nine-cell study with eight records and no message
    is how a gate gets computed on a study nobody ran."""
    _write(tmp_path, _study())
    (tmp_path / "result_cnn_seed1.json").write_text('{"arm": "cnn", ')
    with pytest.raises(UnreadableRecord) as error:
        load_records(tmp_path)
    assert "result_cnn_seed1.json" in str(error.value)
    assert isinstance(error.value, RecordsUnusable)


#: The four ways a file can parse as JSON and still come apart inside
#: `load_record`, one per exception type its non-finite restore step raises.
#:
#: The restore step is not an exotic path: NaN is what `gap_final` holds
#: whenever the persistence-to-floor band is non-positive, so every real record
#: of a study with a non-positive band goes through it. `load_records` used to
#: catch only `(OSError, ValueError)`, so all four escaped `main()` as a
#: traceback and exit 1 -- the one status this pipeline's numbering reserves
#: for "nobody caught this" -- instead of the named refusal and EXIT_UNREADABLE.
#:
#: The expected type name is carried BESIDE each shape and asserted, so the
#: four cases cannot collapse into four spellings of one branch: a fix that
#: caught only `TypeError` would leave three of them failing by name.
UNREADABLE_SHAPES = [
    (
        "TypeError",
        {"arm": "cnn", "seed": 1, "position": None,
         "nonfinite": {"position.gap_final": "nan"}},
    ),
    ("AttributeError", [1, 2, 3]),
    (
        "KeyError",
        {"arm": "cnn", "seed": 1, "nonfinite": {"reward.mse": "nan"}},
    ),
    (
        "IndexError",
        {"arm": "cnn", "seed": 1, "curves": {"rssm_position": []},
         "nonfinite": {"curves.rssm_position.0": "nan"}},
    ),
]


@pytest.mark.parametrize("expected_type,content", UNREADABLE_SHAPES)
def test_load_records_refuses_a_record_the_restore_step_cannot_rebuild(
        tmp_path, expected_type, content):
    """A file that is valid JSON and is not a well-formed record.

    This is the shape a hand-edited, half-converted or older-format record has,
    and `load_record`'s non-finite restore is the step most likely to meet one:
    it walks each dotted path in the record's `nonfinite` map, so a `null`
    block on the way down raises `TypeError`, a JSON list or scalar has no
    `.get` (`AttributeError`), a path naming a field that is gone raises
    `KeyError` and one naming a list index past the end raises `IndexError`.
    None of the four is an `OSError` or a `ValueError`.
    """
    _write(tmp_path, _study())
    (tmp_path / "result_cnn_seed1.json").write_text(json.dumps(content))
    with pytest.raises(UnreadableRecord) as error:
        load_records(tmp_path)
    message = str(error.value)
    assert "result_cnn_seed1.json" in message
    assert expected_type in message, (
        "the message names the exception the file degraded from, so a guard "
        "that caught one of the four and let the others through fails here "
        "by name rather than passing on whichever one it did catch")
    assert "Re-run that cell" in message, (
        "the remedy for an unreadable record is to re-run the cell; a "
        "mislabelled one must be moved by hand, and the two must not swap")
    assert isinstance(error.value, RecordsUnusable)


def test_a_json_scalar_where_a_record_belongs_is_unreadable_not_mislabelled(
        tmp_path):
    """The two refusals want OPPOSITE responses from the operator -- re-run the
    cell, or move the file by hand -- so a file that is not a record at all
    must not arrive as a mislabelling."""
    _write(tmp_path, _study())
    (tmp_path / "result_cnn_seed1.json").write_text("42")
    with pytest.raises(UnreadableRecord) as error:
        load_records(tmp_path)
    assert "AttributeError" in str(error.value)
    assert not isinstance(error.value, MislabelledRecord)


def test_the_mislabelled_refusal_names_the_directory_on_its_remedy_line(
        tmp_path):
    """The remedy tells the operator to move files; a remedy that does not say
    WHERE cost this project a directory once already (`stale_report`).

    The directory is asserted on the remedy line specifically, not anywhere in
    the message: the banner names it too, so a check over the whole string
    would be answered by the banner and a remedy line that lost the path would
    still pass.
    """
    _write(tmp_path, _study())
    (tmp_path / "result_cnn_seed0.json").write_text(
        json.dumps({**_record("frozen_ssl", 0), "nonfinite": {}}))
    with pytest.raises(MislabelledRecord) as error:
        load_records(tmp_path)
    remedy = _row(str(error.value), "Remedy:")
    assert str(tmp_path) in remedy


def test_record_names_its_file_accepts_the_writers_own_name(tmp_path):
    """The other half of the guard: the check must be able to say YES.

    A `record_names_its_file` that returned False always would satisfy every
    refusal test above and reject the whole study.
    """
    record = _record("frozen_ssl", 2)
    path = job_record_path(tmp_path, StudyJob("frozen_ssl", 2))
    assert record_names_its_file(record, path) is True
    assert record_names_its_file(record, tmp_path / "result_cnn_seed2.json") is False


def test_record_cell_reads_the_arm_and_the_seed():
    assert record_cell(_record("random_vit", 1)) == ("random_vit", 1)
    assert record_cell({"arm": "cnn"}) is None
    assert record_cell({"seed": 0}) is None
    assert record_cell({"arm": "cnn", "seed": "0"}) is None
    assert record_cell({"arm": "cnn", "seed": True}) is None, (
        "True is an int in Python and would read as seed 1")


# ---------------------------------------------------------------------------
# the aggregation against a record `run_job` really wrote
# ---------------------------------------------------------------------------

REAL_JOB = StudyJob("random_vit", 1)
REAL_JOB_KW = dict(steps=5, seq_len=4, context=2, horizon=3, device="cpu")
"""One real cell, trained for five steps on the shared small buffer.

`random_vit` because its backbone needs no downloaded weights -- the same
choice `tests/eval/test_study.py` makes for its own `run_job` fixture, and the
reason this file's other tests carry `frozen_ssl` in every fixture instead.
"""


def test_the_aggregation_reads_a_record_run_job_really_wrote(
        tmp_path, small_buffer):
    """THE ONE TEST THAT PINS THE TWO MODULES TOGETHER.

    Every other test in this file is written against a hand-built record shaped
    like `run_job`'s. If `run_job` moved a block tomorrow, all of them would
    still pass and the aggregation would `KeyError` on the real study -- which
    is exactly the defect this task was handed: the brief read
    `record["filtering"]["latent_beats_embedding"]` while `run_job` writes
    `filtering: {criterion_4: {...}, gain: {...}}`.

    So this runs a real cell, reads it back off disk the way the report does,
    and asserts every field the aggregation depends on was FOUND -- not that it
    has any particular value. A NaN `gap_final` is a legitimate result for a
    five-step model; `None` is not, and neither is a NaN that came from a key
    that is not there any more.
    """
    run_job(REAL_JOB, small_buffer, tmp_path, **REAL_JOB_KW)
    records = load_records(tmp_path)
    assert len(records) == 1, (
        "load_records refused or missed the record run_job wrote; the glob, "
        "the filename convention and the arm/seed check all have to agree "
        "with the writer")
    summary = per_arm(records)[REAL_JOB.arm]

    assert summary["seeds"] == [REAL_JOB.seed]
    assert summary["n_seeds"] == 1
    # D1: criterion 4 lives in the NESTED block, and the flag is a real bool
    # rather than the None `_flag` returns for a path that is not there.
    assert summary["latent_beats_embedding_by_seed"][0] in (True, False)
    for key in ("latent_r2_by_seed", "embedding_r2_by_seed", "gain_by_seed",
                "gain_ci_low_by_seed", "gain_ci_high_by_seed",
                "reward_mse_by_seed", "reward_baseline_mse_by_seed",
                "steps_per_second_by_seed"):
        assert not math.isnan(summary[key][0]), f"{key} was not found"
    assert summary["gain_ridges"][0]["ridge_selected"] in (True, False)
    assert not math.isnan(summary["gain_ridges"][0]["joint_ridge"])
    assert not math.isnan(summary["gain_ridges"][0]["embedding_ridge"])
    assert summary["curves_ok"] is True, (
        "the six curves and the horizon they are checked against are both "
        "`run_job`'s; if this fails they have drifted apart")
    assert summary["reward_reported"] is True
    assert not math.isnan(summary["max_degenerate_steps"])
    for metric in METRICS:
        # D2: the gap may legitimately be NaN on a five-step model. What it may
        # not be is None -- which is what a `json.loads` read leaves behind, and
        # what `gaps > 0` raises TypeError on.
        gap = summary["metrics"][metric]["gap_final_by_seed"][0]
        assert isinstance(gap, float)

    # And the whole report renders off it without raising.
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["all_nine_cells_present"] is False
    assert verdict["missing_cells"] == sorted(
        {(arm, seed) for arm in ARMS for seed in SEEDS}
        - {(REAL_JOB.arm, REAL_JOB.seed)})
    text = report_study.report(records, verdict, tmp_path)
    assert _row(text, "GATE:") == "GATE: NOT PASSED"
    assert _row(text, "  records") == (
        "  records     1 of 9 expected cells (3 arms x 3 seeds)")
    # The cell's own row in each table -- located per table, because
    # "random_vit/s1" prefixes a row in four of them.
    assert "MISSING" not in _row(report_study.gain_table(records),
                                 "random_vit/s1")
    assert "MISSING" not in _row(report_study.reward_table(records),
                                 "random_vit/s1")
    assert _row(report_study.metric_table(records, verdict["per_arm"]),
                "cnn         position").count("MISSING") == 3


# ---------------------------------------------------------------------------
# per_arm
# ---------------------------------------------------------------------------

def test_per_arm_summarises_every_arm_including_frozen_ssl():
    summary = per_arm(_study())
    assert set(summary) == set(ARMS)
    assert summary["frozen_ssl"]["n_seeds"] == 3
    assert summary["frozen_ssl"]["seeds"] == [0, 1, 2]


def test_per_arm_orders_the_seeds_rather_than_the_directory_listing():
    """`gap_final_by_seed[0]` must be seed 0's number whatever order the
    records arrived in; the list is printed against seed headings."""
    records = [_record("cnn", seed) for seed in (2, 0, 1)]
    summary = per_arm(records)["cnn"]
    assert summary["seeds"] == [0, 1, 2]
    assert summary["gap_final_by_seed"] == [
        _value("position.gap_final", "cnn", seed) for seed in (0, 1, 2)]


def test_per_arm_aggregation_is_nan_aware():
    """A NaN gap must not poison an arm's mean -- and must not be counted as a
    seed that beat persistence either."""
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 1)["position"]["gap_final"] = float("nan")
    summary = per_arm(records)["cnn"]
    expected = (_value("position.gap_final", "cnn", 0)
                + _value("position.gap_final", "cnn", 2)) / 2
    assert summary["gap_final_mean"] == pytest.approx(expected)
    assert summary["n_seeds_with_finite_gap"] == 2
    assert summary["n_seeds"] == 3, (
        "the denominator the mean was NOT taken over is reported beside it, "
        "or a mean over one seed reads as a mean over three")
    assert summary["all_seeds_positive"] is False


def test_a_mean_over_no_finite_seed_is_nan_rather_than_an_exception():
    records = [_record("cnn", seed) for seed in SEEDS]
    for seed in SEEDS:
        _cell(records, "cnn", seed)["position"]["gap_final"] = float("nan")
    summary = per_arm(records)["cnn"]
    assert math.isnan(summary["gap_final_mean"])
    assert summary["n_seeds_with_finite_gap"] == 0


def test_a_negative_gap_is_not_a_positive_seed():
    """The `(gaps > 0).all()` half of the unanimity, alone."""
    records = [_record("frozen_ssl", seed) for seed in SEEDS]
    _cell(records, "frozen_ssl", 2)["position"]["gap_final"] = -0.01
    summary = per_arm(records)["frozen_ssl"]
    assert summary["all_seeds_positive"] is False
    assert summary["n_seeds_with_finite_gap"] == 3, (
        "a negative gap is finite; if this said 2 the test above would be "
        "exercising the NaN half instead of this one")


def test_an_infinite_gap_is_not_a_positive_seed():
    """The `finite.all()` half of the unanimity, alone -- and the ONLY case
    that half decides.

    `+inf > 0` is True, so `(gaps > 0).all()` passes on it. An infinite
    `gap_closed` is what a band that underflowed to zero produces; the record
    format has a token for it precisely because it happens. It is not a seed
    that beat persistence by any amount anyone measured.
    """
    records = [_record("random_vit", seed) for seed in SEEDS]
    _cell(records, "random_vit", 0)["position"]["gap_final"] = float("inf")
    summary = per_arm(records)["random_vit"]
    assert summary["all_seeds_positive"] is False
    assert (np.array(summary["gap_final_by_seed"]) > 0).all(), (
        "if the fixture's infinity were negative this would be the same test "
        "as the negative-gap one above")


def test_a_none_where_a_number_belongs_does_not_crash_or_count_as_positive():
    """What a bare `json.loads` leaves behind. `load_records` restores the real
    float, but a caller that assembled records another way must degrade to
    "undefined", not raise `TypeError` inside a comparison."""
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 0)["position"]["gap_final"] = None
    summary = per_arm(records)["cnn"]
    assert summary["all_seeds_positive"] is False
    assert summary["n_seeds_with_finite_gap"] == 2


def test_a_gap_of_exactly_zero_closed_none_of_the_band():
    """THE BOUNDARY THE HEADLINE CRITERION IS DEFINED AT.

    Spec section 4 criterion 1 asks for `gap_closed > 0`, and the strictness of
    that inequality is the whole difference between "the world model learned
    something" and "the world model is persistence": a gap of exactly 0.0 means
    the model's final-step error equals the persistence baseline's, i.e. it
    closed NONE of the persistence-to-floor band. That is a realistic output
    for a decoder that collapsed onto copying the last observed state, and it
    is the null hypothesis this whole study exists to reject. The suite pinned
    a negative gap, a NaN gap and a +inf gap and never the boundary itself, so
    `(gaps > 0)` could soften to `(gaps >= 0)` with nothing failing.
    """
    records = _study()
    _cell(records, "frozen_ssl", 1)["position"]["gap_final"] = 0.0
    summary = per_arm(records)["frozen_ssl"]
    assert summary["all_seeds_positive"] is False
    assert summary["n_seeds_with_finite_gap"] == 3, (
        "0.0 is finite; if this said 2 the `finite.all()` half would be what "
        "failed and the boundary would still be untested")
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["beats_persistence"] is False
    assert verdict["passed"] is False


def test_a_bool_where_a_number_belongs_is_a_flag_and_not_a_measurement():
    """`True` is an `int` in Python, so a boolean where `gap_final` belongs
    would be averaged in as 1.0 -- finite, strictly positive, and therefore a
    seed that "beat persistence" on the strength of a flag. `_is_number`
    excludes `bool` for exactly that, and nothing exercised the exclusion.

    The same `_num` feeds the degeneracy counts, the reward numbers and the
    gain, so a bool anywhere a number belongs is promoted across the whole
    aggregation, not just here.
    """
    records = _study()
    _cell(records, "cnn", 0)["position"]["gap_final"] = True
    summary = per_arm(records)["cnn"]
    assert math.isnan(summary["gap_final_by_seed"][0])
    assert summary["n_seeds_with_finite_gap"] == 2
    assert summary["all_seeds_positive"] is False
    assert evaluate_gate(records)["criteria"]["beats_persistence"] is False
    assert bool(True) > 0, (
        "spelled out: were `True` averaged in it would be 1.0, which is both "
        "finite and strictly positive -- the criterion would PASS on it")


def test_the_flat_keys_are_the_gate_metrics_own_and_not_angles():
    """The flat `gap_final_*` keys are position's. Position and angle carry
    different numbers in the fixture, so a flat view that came to hold angle's
    fails here rather than reading plausibly."""
    summary = per_arm(_study())["cnn"]
    position = summary["metrics"]["position"]
    angle = summary["metrics"]["angle"]
    assert summary["gap_final_by_seed"] == position["gap_final_by_seed"]
    assert summary["gap_final_mean"] == position["gap_final_mean"]
    assert summary["all_seeds_positive"] == position["all_seeds_positive"]
    assert summary["n_seeds_with_finite_gap"] == position["n_seeds_with_finite_gap"]
    assert position["gap_final_by_seed"] != angle["gap_final_by_seed"]
    assert GATE_METRIC == "position"


def test_per_arm_reports_angle_beside_position_without_mixing_them():
    summary = per_arm(_study())["random_vit"]
    for metric in METRICS:
        assert summary["metrics"][metric]["gap_final_by_seed"] == [
            _value(f"{metric}.gap_final", "random_vit", seed) for seed in SEEDS]
        assert summary["metrics"][metric]["gap_mean_by_seed"] == [
            _value(f"{metric}.gap_mean", "random_vit", seed) for seed in SEEDS]


def test_an_angle_gap_does_not_move_the_position_mean():
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 0)["angle"]["gap_final"] = float("nan")
    summary = per_arm(records)["cnn"]
    assert summary["n_seeds_with_finite_gap"] == 3
    assert summary["metrics"]["angle"]["n_seeds_with_finite_gap"] == 2


def test_per_arm_counts_a_degenerate_position_band():
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 1)["position"]["steps_degenerate"] = 7
    summary = per_arm(records)["cnn"]
    assert summary["metrics"]["position"]["max_degenerate_steps"] == 7
    assert summary["metrics"]["angle"]["max_degenerate_steps"] == 0
    assert summary["max_degenerate_steps"] == 7
    assert summary["any_degenerate"] is True


def test_per_arm_counts_a_degenerate_angle_band():
    """Angle alone. Task 1 of this plan exists because the degeneracy guards
    were computed for position only; a gate that re-checked position alone
    would reintroduce that at the last step of the study."""
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 1)["angle"]["steps_degenerate"] = 5
    summary = per_arm(records)["cnn"]
    assert summary["metrics"]["position"]["max_degenerate_steps"] == 0
    assert summary["metrics"]["angle"]["max_degenerate_steps"] == 5
    assert summary["max_degenerate_steps"] == 5
    assert summary["any_degenerate"] is True


def test_a_record_missing_its_degeneracy_count_has_not_shown_a_usable_band():
    records = [_record("cnn", seed) for seed in SEEDS]
    del _cell(records, "cnn", 2)["position"]["steps_degenerate"]
    summary = per_arm(records)["cnn"]
    assert math.isnan(summary["max_degenerate_steps"])
    assert summary["any_degenerate"] is True


def test_a_record_missing_its_ANGLE_degeneracy_count_is_no_different():
    """The same guard for the OTHER metric, and it is not a duplicate.

    `max()` with a NaN in it is ORDER-DEPENDENT -- `max([nan, 0.0])` is `nan`
    but `max([0.0, nan])` is `0.0`, because `0.0 > nan` and `nan > 0.0` are
    both False -- and `METRICS` is `("position", "angle")`. The sibling test
    above deletes POSITION's count, which puts the NaN first, so an unguarded
    `max(degenerate)` returns NaN there anyway and the guard it is testing can
    be deleted with a green suite. Deleting ANGLE's count puts the NaN second,
    where only the explicit NaN check can catch it.
    """
    records = _study()
    del _cell(records, "cnn", 2)["angle"]["steps_degenerate"]
    summary = per_arm(records)["cnn"]
    assert math.isnan(summary["metrics"]["angle"]["max_degenerate_steps"])
    assert summary["metrics"]["position"]["max_degenerate_steps"] == 0, (
        "position's count is present and zero, so it is the NaN in SECOND "
        "position that this test turns on -- exactly the one an unguarded "
        "max() would step over")
    assert math.isnan(summary["max_degenerate_steps"])
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["band_is_usable"] is False
    assert verdict["passed"] is False


def test_per_arm_reports_the_band_median_and_the_floor_above_persistence_count():
    """Two fields nothing in this module ever read.

    `band_median_by_seed` was computed, exposed in the API and asserted
    nowhere; `max_steps_floor_above_persistence` is a PRINTED column of the
    headline table (`floor>=pers`) whose field was equally unguarded, so it
    could read `steps_degenerate` instead. Those two counts mean opposite
    things -- `degenerate` is a band that was positive and numerically junk,
    `floor>=pers` is a band that was NON-POSITIVE, which is why `gap_closed` is
    NaN by contract there. In a study whose bands are non-positive, that column
    is the only place the report explains a row full of +nan.
    """
    summary = per_arm(_study())["frozen_ssl"]
    for metric in METRICS:
        block = summary["metrics"][metric]
        assert block["band_median_by_seed"] == [
            _value(f"{metric}.band_median", "frozen_ssl", seed)
            for seed in SEEDS]
        assert block["max_steps_floor_above_persistence"] == max(
            _value(f"{metric}.steps_floor_above_persistence", "frozen_ssl",
                   seed)
            for seed in SEEDS)
        assert block["max_degenerate_steps"] == 0
        assert block["max_steps_floor_above_persistence"] != block[
            "max_degenerate_steps"], (
            "the two counts have to differ in the fixture or each column is "
            "free to render the other's field")
        assert block["band_median_by_seed"] != block["gap_mean_by_seed"]


def test_per_arm_reads_criterion_4_out_of_its_nested_block():
    """`run_job` writes `filtering: {criterion_4: {...}, gain: {...}}`, and
    criterion 4's flag is inside the first. A reader of
    `filtering["latent_beats_embedding"]` finds nothing on a real record."""
    summary = per_arm(_study())["cnn"]
    assert summary["filtering_all_pass"] is True
    assert summary["latent_r2_by_seed"] == [
        _value("filtering.criterion_4.latent_r2", "cnn", seed)
        for seed in SEEDS]
    assert summary["embedding_r2_by_seed"] == [
        _value("filtering.criterion_4.embedding_r2", "cnn", seed)
        for seed in SEEDS]
    assert summary["latent_beats_embedding_by_seed"] == [True, True, True]


def test_one_losing_seed_costs_the_arm_criterion_4():
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 1)["filtering"]["criterion_4"][
        "latent_beats_embedding"] = False
    summary = per_arm(records)["cnn"]
    assert summary["filtering_all_pass"] is False
    assert summary["latent_beats_embedding_by_seed"] == [True, False, True]


def test_a_missing_criterion_4_flag_is_not_a_pass():
    """"The record does not say" and "the record says no" are different
    findings and only one of them is a measurement; both fail the criterion,
    and `all()` over an absent key would have passed it."""
    records = [_record("cnn", seed) for seed in SEEDS]
    del _cell(records, "cnn", 0)["filtering"]["criterion_4"][
        "latent_beats_embedding"]
    summary = per_arm(records)["cnn"]
    assert summary["filtering_all_pass"] is False
    assert summary["latent_beats_embedding_by_seed"] == [None, True, True]


@pytest.mark.parametrize("not_a_bool", [1, 0, "false", "true", []])
def test_a_criterion_4_flag_that_is_not_a_boolean_is_not_an_answer(not_a_bool):
    """`_flag`'s `isinstance(node, (bool, np.bool_))` check, alone.

    It is what makes "the record does not carry a real boolean" distinguishable
    from "the record says no", and nothing exercised it. Criterion 4's flag is
    the ONE criterion the measured study fails, so this is the number the whole
    milestone turns on: without the check, a JSON `1` reads as True and -- much
    worse -- so does the STRING "false", because `bool("false")` is True. The
    same `_flag` feeds `any_reward_degenerate` and `ridge_selected`, the R1
    disclosure, so a non-boolean there becomes a measured answer too.
    """
    records = _study()
    _cell(records, "cnn", 0)["filtering"]["criterion_4"][
        "latent_beats_embedding"] = not_a_bool
    summary = per_arm(records)["cnn"]
    assert summary["latent_beats_embedding_by_seed"] == [None, True, True]
    assert summary["filtering_all_pass"] is False
    assert evaluate_gate(records)["criteria"][
        "filtering_beats_embedding"] is False


def test_per_arm_reports_the_gain_with_its_interval():
    """R1: the gain is the bottleneck-free companion to criterion 4 and is
    meaningless without its interval."""
    summary = per_arm(_study())["frozen_ssl"]
    assert summary["gain_by_seed"] == [
        _value("filtering.gain.gain", "frozen_ssl", seed) for seed in SEEDS]
    assert summary["gain_ci_low_by_seed"] == [
        _value("filtering.gain.ci_low", "frozen_ssl", seed) for seed in SEEDS]
    assert summary["gain_ci_high_by_seed"] == [
        _value("filtering.gain.ci_high", "frozen_ssl", seed) for seed in SEEDS]
    assert summary["gain_mean"] == pytest.approx(
        float(np.mean(summary["gain_by_seed"])))
    assert summary["n_seeds_with_finite_gain"] == 3


def test_a_confidence_interval_entirely_below_zero_excludes_zero():
    """The `high < 0` half of the disjunction, alone. This is the sign the real
    study measured: gain -0.0208, 95% CI [-0.0395, -0.0040]."""
    records = [_record("cnn", 0)]
    records[0]["filtering"]["gain"].update(
        gain=-0.0208, ci_low=-0.0395, ci_high=-0.0040)
    assert per_arm(records)["cnn"]["gain_ci_excludes_zero_by_seed"] == [True]


def test_a_confidence_interval_entirely_above_zero_excludes_zero():
    """The `low > 0` half, alone."""
    records = [_record("cnn", 0)]
    records[0]["filtering"]["gain"].update(
        gain=0.0325, ci_low=0.0040, ci_high=0.0610)
    assert per_arm(records)["cnn"]["gain_ci_excludes_zero_by_seed"] == [True]


def test_a_confidence_interval_straddling_zero_does_not_exclude_it():
    records = [_record("cnn", 0)]
    records[0]["filtering"]["gain"].update(
        gain=0.001, ci_low=-0.0100, ci_high=0.0200)
    assert per_arm(records)["cnn"]["gain_ci_excludes_zero_by_seed"] == [False]


@pytest.mark.parametrize("ci_low,ci_high", [
    (float("nan"), -0.0040),
    (-0.0395, float("nan")),
    (float("nan"), float("nan")),
])
def test_an_interval_with_one_end_is_not_an_interval(ci_low, ci_high):
    """`low > 0.0 or high < 0.0` is a DISJUNCTION, so ONE finite end on the
    right side of zero satisfies it on its own -- and a gain whose lower bound
    was never computed was therefore reported as SIGNIFICANT.

    The report script has always rendered this case as "n/a", saying why in its
    own docstring: an interval with one end is not an interval, and rendering
    it as `False` would say "the interval covers zero", which nobody measured.
    The aggregation -- which is the API a write-up or a later task reads --
    said `True`. The two disagreed with no test comparing them.
    """
    records = [_record("cnn", 0)]
    records[0]["filtering"]["gain"].update(
        gain=-0.0208, ci_low=ci_low, ci_high=ci_high)
    assert per_arm(records)["cnn"]["gain_ci_excludes_zero_by_seed"] == [None]


def test_the_api_and_the_column_read_the_interval_the_same_way():
    """The aggregation's `gain_ci_excludes_zero_by_seed` and the report's
    `CI excl 0` column are two implementations of one rule, and they used to
    give opposite answers on a one-ended interval. Compared over the cases that
    separate them, including the MEASURED interval."""
    cases = [
        (-0.0395, -0.0040),          # the measured study: entirely below zero
        (0.0040, 0.0610),            # entirely above zero
        (-0.0100, 0.0200),           # straddling zero
        (float("nan"), -0.0040),     # one end never computed
        (-0.0395, float("nan")),
    ]
    rendered = {None: "n/a", True: "True", False: "False"}
    for low, high in cases:
        records = [_record("cnn", 0)]
        records[0]["filtering"]["gain"].update(ci_low=low, ci_high=high)
        api = per_arm(records)["cnn"]["gain_ci_excludes_zero_by_seed"][0]
        assert rendered[api] == report_study._excludes_zero(low, high), (
            f"the API and the column disagree about [{low}, {high}]")
    assert {report_study._excludes_zero(low, high)
            for low, high in cases} == {"True", "False", "n/a"}, (
        "the five cases must produce all three renderings, or two "
        "implementations that both answered one thing for everything would "
        "satisfy the loop above")


def test_per_arm_flags_seeds_that_selected_different_ridges():
    """R1, the reason it is a requirement: the scored R^2 moves ~0.10 per
    decade while the gain is ~0.02, so the gain's SIGN moves with the selected
    decade -- measured -0.0706, -0.0208 and +0.0325 for one model. A mean over
    seeds that selected different decades is a mean over different estimators
    and nothing in the number itself says so."""
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 2)["filtering"]["gain"][
        "joint_ridge"] = ALT_JOINT_RIDGE
    summary = per_arm(records)["cnn"]
    assert summary["gain_ridges_agree"] is False
    assert summary["gain_ridges"][2]["joint_ridge"] == ALT_JOINT_RIDGE
    assert summary["gain_ridges"][0]["joint_ridge"] == JOINT_RIDGE


def test_per_arm_says_the_ridges_agree_when_they_do():
    """The other half: a flag that was always False would satisfy the test
    above and warn on every real study."""
    summary = per_arm([_record("cnn", seed) for seed in SEEDS])["cnn"]
    assert summary["gain_ridges_agree"] is True
    assert [r["embedding_ridge"] for r in summary["gain_ridges"]] == [
        EMBEDDING_RIDGE] * 3


def test_a_seed_that_skipped_ridge_selection_disagrees_with_one_that_did_not():
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 0)["filtering"]["gain"]["ridge_selected"] = False
    summary = per_arm(records)["cnn"]
    assert summary["gain_ridges_agree"] is False
    assert summary["gain_ridges"][0]["ridge_selected"] is False
    assert summary["gain_ridges"][1]["ridge_selected"] is True


def test_per_arm_reports_the_reward_numbers_and_the_degenerate_flag():
    summary = per_arm(_study(reward_degenerate=False))["random_vit"]
    assert summary["reward_mse_by_seed"] == [
        _value("reward.mse", "random_vit", seed) for seed in SEEDS]
    assert summary["reward_r2_by_seed"] == [
        _value("reward.r2", "random_vit", seed) for seed in SEEDS]
    assert summary["reward_baseline_mse_by_seed"] == [
        _value("reward.baseline_mse", "random_vit", seed) for seed in SEEDS]
    assert summary["any_reward_degenerate"] is False
    assert summary["reward_reported"] is True


def test_a_degenerate_reward_target_is_flagged_without_hiding_the_numbers():
    summary = per_arm(_study(reward_degenerate=True))["cnn"]
    assert summary["any_reward_degenerate"] is True
    assert summary["reward_reported"] is True, (
        "spec criterion 3 asks for the accuracy to be REPORTED, and "
        "my_way_home's reward is near-constant by construction")


def test_a_record_with_no_reward_block_has_not_reported_one():
    """The arm's OTHER seeds say their target was not degenerate.

    That is the whole point of the fixture here: with the default
    `reward_degenerate=True` the `any()` is satisfied by the two healthy
    records whatever the third one does, so the assertion below could not fail
    and `is not False` could be weakened to `is True` with a green suite --
    which is exactly what mutation A28 did. A record that never said whether
    its target was constant has not earned the benefit of the doubt, and this
    is what says so.
    """
    records = [_record("cnn", seed, reward_degenerate=False)
               for seed in SEEDS]
    del _cell(records, "cnn", 1)["reward"]
    summary = per_arm(records)["cnn"]
    assert summary["reward_reported"] is False
    assert summary["any_reward_degenerate"] is True
    assert per_arm([_record("cnn", 0, reward_degenerate=False)])["cnn"][
        "any_reward_degenerate"] is False, (
        "and the flag must be able to say False, or the assertion above is "
        "satisfied by a constant")


def test_a_reward_block_scored_over_no_steps_has_not_reported_one():
    """An MSE over zero steps is a number with nothing behind it. `mse` and
    `baseline_mse` are both present and finite here, so the `n_steps > 0` half
    of the check is the only thing that can fail."""
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 2)["reward"]["n_steps"] = 0
    summary = per_arm(records)["cnn"]
    assert summary["reward_reported"] is False
    assert not math.isnan(summary["reward_mse_by_seed"][2])


def test_a_reward_mse_that_never_computed_has_not_reported_accuracy():
    """The `not isnan(mse)` term, ALONE.

    `reward_reported` is a three-term conjunction and only its third term ever
    had an isolating test; the one test that could fail it through the two mse
    terms deleted the WHOLE reward block, so all three failed at once and
    either of the first two could be dropped with a green suite. A NaN mse is
    not exotic here -- `_summarise_reward`'s own contract produces one when the
    target has no variance, which is `my_way_home`'s normal case -- and a cell
    that reported no usable accuracy would then read `reward_reported PASS`.
    """
    records = _study()
    _cell(records, "cnn", 1)["reward"]["mse"] = float("nan")
    summary = per_arm(records)["cnn"]
    assert summary["reward_reported"] is False
    assert not math.isnan(summary["reward_baseline_mse_by_seed"][1]), (
        "the baseline is finite here, so this test turns on the mse term and "
        "not on its neighbour")
    assert summary["reward_r2_by_seed"][1] == _value("reward.r2", "cnn", 1)
    assert evaluate_gate(records)["criteria"]["reward_reported"] is False


def test_a_reward_baseline_that_never_computed_has_not_reported_accuracy():
    """The `not isnan(baseline_mse)` term, ALONE. An MSE over a near-constant
    target looks precise and means nothing without the baseline beside it, so
    a cell that lost the baseline has not reported its accuracy either."""
    records = _study()
    del _cell(records, "frozen_ssl", 2)["reward"]["baseline_mse"]
    summary = per_arm(records)["frozen_ssl"]
    assert summary["reward_reported"] is False
    assert math.isnan(summary["reward_baseline_mse_by_seed"][2])
    assert not math.isnan(summary["reward_mse_by_seed"][2]), (
        "the mse is finite here, so this test turns on the baseline term and "
        "not on its neighbour")
    assert summary["reward_mse_by_seed"][2] == _value(
        "reward.mse", "frozen_ssl", 2)
    assert evaluate_gate(records)["criteria"]["reward_reported"] is False


def test_per_arm_reports_throughput_and_the_worst_kl_rate():
    summary = per_arm(_study())["frozen_ssl"]
    assert summary["steps_per_second_by_seed"] == [
        _value("steps_per_second", "frozen_ssl", seed) for seed in SEEDS]
    assert summary["kl_rate_min"] == min(
        _value("kl_rate_above_free_bits", "frozen_ssl", seed)
        for seed in SEEDS)


def test_curves_are_checked_for_every_reference_and_both_metrics():
    assert per_arm(_study())["cnn"]["curves_ok"] is True
    for name in CURVE_NAMES:
        records = [_record("cnn", seed) for seed in SEEDS]
        del _cell(records, "cnn", 0)["curves"][name]
        assert per_arm(records)["cnn"]["curves_ok"] is False, (
            f"{name} is one of the three references spec criterion 2 asks to "
            "be drawn on the same axes")


def test_a_curve_shorter_than_the_horizon_is_not_a_complete_rollout():
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 1)["curves"]["rssm_position"] = [1.0, 2.0]
    assert per_arm(records)["cnn"]["curves_ok"] is False


def test_curves_all_shorter_than_the_horizon_are_still_incomplete():
    """The `lengths == {horizon}` check, not merely `len(lengths) == 1`: six
    curves that agree with each other and disagree with the horizon are a
    truncated rollout plotted as a complete one."""
    records = [_record("cnn", seed) for seed in SEEDS]
    for name in CURVE_NAMES:
        _cell(records, "cnn", 1)["curves"][name] = [1.0, 2.0]
    assert per_arm(records)["cnn"]["curves_ok"] is False


@pytest.mark.parametrize("block", [None, 7, [1, 2], "x"])
def test_a_curves_block_that_is_not_a_dict_has_produced_no_curves(block):
    """The `not isinstance(curves, dict)` half, alone.

    Every other curve test reaches INTO the dict -- `record["curves"][name]` --
    so a record whose WHOLE `curves` key came back as a null or a scalar was
    never constructed anywhere, and the guard could return True with a green
    suite: gate criterion 2 (the error-vs-horizon curve of each arm with
    persistence and floor on the same axes) reported as MET with nothing on
    disk to draw. It is the same corruption species `_get` and `_num` are
    guarded against for `position`, `filtering` and `reward`; it was missed
    for `curves`.

    The gap list is asserted, not just the boolean: "all six curves are
    missing" is the finding, and a guard that answered False for some other
    reason would satisfy a bare `is False`.
    """
    records = _study()
    _cell(records, "cnn", 1)["curves"] = block
    assert aggregate.curve_gaps(_cell(records, "cnn", 1)) == list(CURVE_NAMES)
    assert per_arm(records)["cnn"]["curves_ok"] is False
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["curves_produced"] is False
    assert verdict["passed"] is False


def test_a_record_with_no_curves_key_at_all_has_produced_no_curves():
    """The absent block, beside the null one above, because the two are NOT
    interchangeable everywhere.

    `curve_gaps` sees `None` for both, so this half of it is one branch. But
    `mean_curve` read `record.get("curves", {})`, and there the two part
    company: an absent key takes the `{}` default and answers `None` safely,
    while a stored `null` is returned as itself and `.get(name)` on it raises
    `AttributeError`. Pinning only the shape that happens to be safe there is
    how that crash survived.
    """
    records = _study()
    del _cell(records, "frozen_ssl", 0)["curves"]
    assert aggregate.curve_gaps(_cell(records, "frozen_ssl", 0)) == list(
        CURVE_NAMES)
    assert per_arm(records)["frozen_ssl"]["curves_ok"] is False
    assert evaluate_gate(records)["criteria"]["curves_produced"] is False


def test_curve_gaps_names_only_the_curves_that_are_short():
    """`curves_produced` is the one criterion the report cannot attribute to a
    cell from a table, so the gate carries the NAMES. A list that named all six
    whenever any one was short would be no more use than the bare boolean."""
    record = _record("cnn", 0)
    assert aggregate.curve_gaps(record) == []
    del record["curves"]["floor_angle"]
    record["curves"]["rssm_position"] = [1.0, 2.0]
    assert aggregate.curve_gaps(record) == ["rssm_position", "floor_angle"], (
        "in CURVE_NAMES order, and only the two that are actually short")


@pytest.mark.parametrize("curve,why", [
    ("1234", "a JSON string"),
    ({"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0}, "a mapping"),
])
def test_a_curve_that_is_not_a_LIST_has_not_been_produced(curve, why):
    """The `not isinstance(curve, list)` term of the three-term guard, ALONE.

    Every other curve test in this module deletes the name, shortens the list
    or replaces the whole `curves` block; nothing anywhere put a NON-LIST in a
    curve SLOT, so that term was never the deciding one. The fixtures here are
    chosen to be exactly `FIXTURE_HORIZON` long, so `len(curve) != horizon` is
    FALSE for both and the isinstance term is the only thing that can answer:
    without it a record whose `rssm_position` came back as the four-character
    string "1234" reports a complete curve set, `curves_produced` PASSES, and
    `write_figure` then hands `plt.plot` four characters where a rollout
    belongs.
    """
    record = _record("cnn", 0)
    assert len(curve) == FIXTURE_HORIZON, (
        f"{why} of exactly the horizon's length is the point: a shorter one "
        "would be caught by the length term and prove nothing")
    record["curves"]["rssm_position"] = curve
    assert aggregate.curve_gaps(record) == ["rssm_position"], (
        "only the slot that is not a list, and it IS one of the gaps")
    assert aggregate._curves_ok(record) is False


def test_six_EMPTY_curves_at_a_horizon_of_zero_are_still_no_curves():
    """The `not curve` term, alone, and the only shape that can reach it.

    At any positive horizon an empty list is already caught by
    `len(curve) != horizon`, so the term can only be the deciding one when the
    record says its horizon was 0 -- where `len([]) != 0` is False and a
    record carrying six empty curves would otherwise report a complete
    error-vs-horizon curve set with nothing at all to draw.
    """
    record = _record("cnn", 0)
    record["horizon"] = 0
    record["curves"] = {name: [] for name in CURVE_NAMES}
    assert aggregate.curve_gaps(record) == list(CURVE_NAMES)


@pytest.mark.parametrize("horizon,length", [
    # A float: `len(curve) != 4.0` is FALSE for a four-step curve, so without
    # the guard a record whose horizon came back as `4.0` -- which is what a
    # JSON `"horizon": 4.0` or a half-converted record holds -- reports every
    # curve complete.
    (float(FIXTURE_HORIZON), FIXTURE_HORIZON),
    # A bool: `isinstance(True, int)` is True, so only the explicit
    # `isinstance(horizon, bool)` half excludes it, and `len(curve) != True`
    # is False for a one-step curve.
    (True, 1),
])
def test_a_horizon_that_is_not_an_INT_leaves_every_curve_unverifiable(
        horizon, length):
    """The moved guard's RETURN VALUE: all six names, not an empty list.

    The guard used to sit after the length loop and answer a bool; it now sits
    before the loop and answers `list(CURVE_NAMES)`, and nothing exercised
    that. The two shapes here are the ones where the length comparison AGREES
    with a horizon it should never have been allowed to compare against, so
    they are the two that tell the guard's answer apart from the loop's: with
    the guard gone, both report a complete curve set and gate criterion 2 --
    the error-vs-horizon curve of each arm with persistence and floor on the
    same axes -- is reported as MET by a record that never said how many steps
    it rolled out.
    """
    records = _study()
    broken = _cell(records, "cnn", 1)
    broken["horizon"] = horizon
    broken["curves"] = {
        name: [float(step) for step in range(length)] for name in CURVE_NAMES}
    assert aggregate.curve_gaps(broken) == list(CURVE_NAMES), (
        "a horizon nothing can be checked against makes every curve "
        "unverifiable, which is not the same as none being short")
    assert per_arm(records)["cnn"]["curves_ok"] is False
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["curves_produced"] is False
    assert verdict["passed"] is False
    assert _row(report_study.gate_block(verdict), "  CURVES INCOMPLETE") == (
        "  CURVES INCOMPLETE (1 of 9): cnn/s1 "
        f"({', '.join(CURVE_NAMES)})")


# ---------------------------------------------------------------------------
# ridge_groups and mean_curve
# ---------------------------------------------------------------------------

def test_ridge_groups_puts_a_coherent_study_in_one_group():
    groups = ridge_groups(_study())
    assert len(groups) == 1
    assert groups[0]["n_cells"] == 9
    assert groups[0]["joint_ridge"] == JOINT_RIDGE
    assert groups[0]["embedding_ridge"] == EMBEDDING_RIDGE
    assert groups[0]["ridge_selected"] is True
    assert sorted(groups[0]["cells"]) == sorted(
        (arm, seed) for arm in ARMS for seed in SEEDS)


def test_ridge_groups_splits_cells_that_selected_different_decades():
    records = _study()
    for seed in SEEDS:
        _cell(records, "random_vit", seed)["filtering"]["gain"][
            "joint_ridge"] = ALT_JOINT_RIDGE
    groups = ridge_groups(records)
    assert len(groups) == 2
    assert [g["n_cells"] for g in groups] == [6, 3]
    assert [g["joint_ridge"] for g in groups] == [JOINT_RIDGE,
                                                  ALT_JOINT_RIDGE]
    assert groups[1]["cells"] == [("random_vit", seed) for seed in SEEDS]


def test_ridge_groups_reports_each_groups_own_mean_gain():
    records = _study()
    for seed in SEEDS:
        _cell(records, "cnn", seed)["filtering"]["gain"]["gain"] = -0.0706
    groups = ridge_groups(records)
    means = {g["joint_ridge"]: g["gain_mean"] for g in groups}
    assert len(groups) == 1
    assert means[JOINT_RIDGE] == pytest.approx(
        float(np.mean([-0.0706] * 3 + [
            _value("filtering.gain.gain", arm, seed)
            for arm in ("frozen_ssl", "random_vit") for seed in SEEDS])))


def test_ridge_groups_does_not_open_a_group_per_broken_cell():
    """NaN != NaN, so keying the grouping on the raw float would give every
    record with a missing ridge a group of its own."""
    records = _study()
    for arm in ARMS:
        for seed in SEEDS:
            del _cell(records, arm, seed)["filtering"]["gain"]["joint_ridge"]
    groups = ridge_groups(records)
    assert len(groups) == 1
    assert math.isnan(groups[0]["joint_ridge"])
    assert groups[0]["n_cells"] == 9


def test_two_cells_that_used_different_EMBEDDING_ridges_are_two_groups():
    """The second of the grouping key's three parts, alone.

    `gain = joint_r2 - embedding_r2`, so two cells that priced the EMBEDDING
    probe at different decades are two different estimators even when their
    joint ridges agree -- exactly the "mean over different estimators" whose
    sign R1 says is not interpretable. Every fixture cell used to carry the
    same joint ridge, the same embedding ridge and the same flag, so two of
    the key's three parts could be dropped with nothing failing. The only
    existing splitting test moves the JOINT ridge, which is the one part that
    was already pinned. Here the joint ridge is asserted EQUAL across the two
    groups, so the split can only have come from the embedding half.
    """
    records = _study()
    _cell(records, "cnn", 0)["filtering"]["gain"][
        "embedding_ridge"] = ALT_EMBEDDING_RIDGE
    groups = ridge_groups(records)
    assert len(groups) == 2
    assert {g["joint_ridge"] for g in groups} == {JOINT_RIDGE}, (
        "the joint ridge agrees, so a key that had lost the embedding half "
        "would have put all nine cells in one group")
    assert {g["embedding_ridge"] for g in groups} == {
        EMBEDDING_RIDGE, ALT_EMBEDDING_RIDGE}
    assert sorted(g["n_cells"] for g in groups) == [1, 8]
    assert [g["cells"] for g in groups if g["n_cells"] == 1] == [[("cnn", 0)]]


def test_a_cell_that_skipped_selection_is_not_grouped_with_one_that_did_not():
    """The FLAG, the third part of the key, alone -- both ridges agree here.

    `ridge_block`'s "at least one cell computed its gain with NO ridge
    selection" warning is driven by `group['ridge_selected'] is not True` over
    GROUPS, not over cells, so a key that merged the two would render a single
    group reading `True` and suppress the warning entirely: a gain computed at
    `fit_probe`'s default penalty averaged in beside selected ones with
    nothing saying so. The existing rendering test sets the flag False for ALL
    nine cells, which is one group either way.
    """
    records = _study()
    _cell(records, "cnn", 0)["filtering"]["gain"]["ridge_selected"] = False
    groups = ridge_groups(records)
    assert len(groups) == 2
    assert {g["joint_ridge"] for g in groups} == {JOINT_RIDGE}
    assert {g["embedding_ridge"] for g in groups} == {EMBEDDING_RIDGE}, (
        "both ridges agree, so the split is the flag's doing alone")
    assert [g["ridge_selected"] for g in groups] == [True, False]
    block = report_study.ridge_block(groups)
    assert "NO ridge selection" in _row(block, "WARNING: at least one cell")


def test_a_group_publishes_the_denominator_its_mean_was_taken_over():
    """`gain_mean` is a `nanmean`, and `n_cells` counts every cell in the
    group whether its gain was computed or not.

    Both module docstrings promise that nothing is dropped from a denominator
    without the denominator being printed -- `_metric_summary` publishes
    `n_seeds_with_finite_gap`, `per_arm` publishes `n_seeds_with_finite_gain`
    -- and this row was the exception: `cells 9` printed beside a mean over
    six, with all nine named as members.
    """
    records = _study()
    for seed in SEEDS:
        del _cell(records, "cnn", seed)["filtering"]["gain"]["gain"]
    [group] = ridge_groups(records)
    assert group["n_cells"] == 9
    assert group["n_finite_gains"] == 6
    survivors = [
        _value("filtering.gain.gain", arm, seed)
        for arm in ARMS if arm != "cnn" for seed in SEEDS
    ]
    assert group["gain_mean"] == pytest.approx(sum(survivors) / 6)
    assert group["gain_mean"] != pytest.approx(
        sum(survivors) / 9), (
        "spelled out: the numerator is over six and so is the denominator; a "
        "mean that divided by nine would be a different number")


def test_a_group_whose_every_gain_is_undefined_reports_a_zero_denominator():
    """The all-NaN guard on that mean, and the denominator beside it.

    `np.nanmean` over an all-NaN slice is a RuntimeWarning on the way to the
    same NaN; the guard is what keeps that out of `study.log`. The row still
    says `cells 9`, so `finite 0` is the only thing telling the reader that
    the `nan` is a mean of nothing rather than a mean of nine nans."""
    records = _study()
    for arm in ARMS:
        for seed in SEEDS:
            del _cell(records, arm, seed)["filtering"]["gain"]["gain"]
    [group] = ridge_groups(records)
    assert group["n_cells"] == 9
    assert group["n_finite_gains"] == 0
    assert math.isnan(group["gain_mean"])


def test_mean_curve_averages_across_the_seeds_of_one_arm():
    records = [_record("cnn", seed) for seed in SEEDS]
    curve = mean_curve(records, "rssm_position")
    expected = np.mean(
        [_curve("rssm_position", "cnn", seed) for seed in SEEDS], axis=0)
    assert curve.tolist() == pytest.approx(expected.tolist())
    assert len(curve) == FIXTURE_HORIZON
    assert curve.tolist() != _curve("rssm_position", "cnn", 0), (
        "if the seeds' curves were equal this would pass on a mean that "
        "returned the first record's curve")


def test_mean_curve_refuses_a_ragged_set_rather_than_averaging_horizons():
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 1)["curves"]["rssm_position"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="different lengths"):
        mean_curve(records, "rssm_position")


def test_mean_curve_names_the_cell_whose_curve_is_MISSING():
    """The `missing` guard, which is not the ragged-set guard.

    A record short of ONE of its six curves is exactly what `curves_produced`
    exists to detect, so the gate is already reporting FAIL by the time the
    figure is drawn. Without this guard the length loop raises `TypeError`
    ("object of type 'NoneType' has no len()"), which `write_figure` does NOT
    catch -- it catches `ValueError` and `OSError` -- so the process dies
    AFTER the verdict has been printed and the exit status is lost. The
    ragged-set test above covers the `ValueError` half only.
    """
    records = [_record("cnn", seed) for seed in SEEDS]
    del _cell(records, "cnn", 1)["curves"]["rssm_position"]
    with pytest.raises(ValueError) as error:
        mean_curve(records, "rssm_position")
    assert str(error.value) == "no 'rssm_position' curve for [('cnn', 1)]", (
        "the cell is named; 'a curve is missing somewhere in nine records' is "
        "not a message anyone can act on")
    assert "different lengths" not in str(error.value), (
        "a missing curve and a ragged set are two findings with two remedies")


def test_mean_curve_survives_a_curves_block_that_is_not_a_dict():
    """`record.get("curves", {})` answers with the block that IS there, so a
    `curves` key holding a null or a scalar reached `.get(name)` and raised
    `AttributeError` -- which `write_figure` does not catch either."""
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 0)["curves"] = None
    _cell(records, "cnn", 2)["curves"] = 7
    with pytest.raises(ValueError) as error:
        mean_curve(records, "rssm_position")
    assert str(error.value) == (
        "no 'rssm_position' curve for [('cnn', 0), ('cnn', 2)]")


def test_mean_curve_refuses_a_name_that_is_not_one_of_the_six():
    with pytest.raises(ValueError):
        mean_curve([_record("cnn", 0)], "rssm_health")


# ---------------------------------------------------------------------------
# evaluate_gate
# ---------------------------------------------------------------------------

def test_the_gate_passes_when_every_criterion_is_met():
    verdict = evaluate_gate(_study())
    assert verdict["criteria"] == {name: True for name in GATE_CRITERIA}
    assert verdict["passed"] is True
    assert verdict["missing_cells"] == []
    assert verdict["n_records"] == 9
    assert verdict["n_expected"] == 9


def test_a_passing_study_still_reports_its_degenerate_reward_target():
    """The gate must not be gated on a property of the scenario that was known
    before the study started, and it must not hide it either."""
    verdict = evaluate_gate(_study(reward_degenerate=True))
    assert verdict["passed"] is True
    assert verdict["per_arm"]["cnn"]["any_reward_degenerate"] is True


def test_the_gate_fails_when_one_seed_of_one_arm_is_negative():
    """Unanimity, deliberately -- with n=3 a t-test has 2 degrees of freedom
    and would dress a weak result in strong-looking notation."""
    records = _study()
    _cell(records, "frozen_ssl", 1)["position"]["gap_final"] = -0.01
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["beats_persistence"] is False
    assert verdict["passed"] is False
    assert all(verdict["criteria"][name] for name in GATE_CRITERIA
               if name != "beats_persistence"), (
        "one negative seed must cost exactly one criterion; a gate whose "
        "criteria all move together reports one finding under eight names")


def test_the_gate_fails_when_a_whole_arm_is_missing():
    """`all()` over the arms that HAVE records is True for an arm with none.
    The expensive arm is 25 of the study's 33 hours and is the one that does
    not finish."""
    records = [r for r in _study() if r["arm"] != "cnn"]
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["beats_persistence"] is False
    assert verdict["criteria"]["band_is_usable"] is False
    assert verdict["criteria"]["filtering_beats_embedding"] is False
    assert verdict["criteria"]["reward_reported"] is False
    assert verdict["criteria"]["curves_produced"] is False
    assert verdict["criteria"]["all_nine_cells_present"] is False
    assert verdict["passed"] is False


def test_the_gate_fails_when_one_seed_is_missing_and_names_the_cell():
    """Nine cells or no verdict -- averaging over a silently varying subset is
    how a 3x3 study becomes a 3x2 study nobody notices."""
    records = [r for r in _study()
               if (r["arm"], r["seed"]) != ("random_vit", 1)]
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["all_nine_cells_present"] is False
    assert verdict["passed"] is False
    assert verdict["missing_cells"] == [("random_vit", 1)]
    assert verdict["n_records"] == 8


def test_the_gate_fails_on_a_duplicated_cell():
    """One cell counted twice is a mean over a denominator nobody chose."""
    records = _study()
    records.append(_record("cnn", 0))
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["no_duplicate_cells"] is False
    assert verdict["duplicate_cells"] == [("cnn", 0)]
    assert verdict["criteria"]["all_nine_cells_present"] is True, (
        "a duplicate must cost its own criterion and not be reported as a "
        "missing cell")
    assert verdict["passed"] is False


def test_the_gate_fails_on_a_record_that_is_not_a_study_cell():
    """A record for a seed the study never ran is not evidence about the study,
    and silently ignoring it is how a directory nobody checked gets
    aggregated."""
    records = _study()
    records.append(_record("cnn", 0) | {"seed": 17})
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["every_record_is_a_study_cell"] is False
    assert verdict["unexpected_cells"] == ["arm='cnn' seed=17"]
    assert verdict["passed"] is False


def test_a_missing_cell_beside_a_stray_file_is_not_a_complete_study():
    """Nine records is not nine CELLS, and this is the shape the study will
    actually meet: `result_cnn_seed2.json` never got written and a
    `result_cnn_seed7.json` is left over from an earlier `--arms cnn --seeds 7`
    run into the same directory.

    Completeness is `expected - present`, deliberately not a count: a superset
    or `len(present) >= len(expected)` test passes on this set. Neither of the
    two existing completeness tests can tell the two forms apart -- one removes
    a record (eight, both forms fail) and one adds a tenth (ten, both forms
    pass). Under the count form the gate reports `all_nine_cells_present` PASS
    while the very next lines of the same block print `MISSING CELLS: cnn/s2`,
    a self-contradictory verdict naming the wrong cause for a NOT PASSED.
    """
    records = [r for r in _study() if (r["arm"], r["seed"]) != ("cnn", 2)]
    records.append(_record("cnn", 0) | {"seed": 7})
    verdict = evaluate_gate(records)
    assert verdict["n_records"] == 9, (
        "nine records, so a criterion that counted rather than compared sets "
        "would read this study as complete")
    assert verdict["criteria"]["all_nine_cells_present"] is False
    assert verdict["criteria"]["every_record_is_a_study_cell"] is False
    assert verdict["criteria"]["no_duplicate_cells"] is True, (
        "the stray cell is its own finding and must not be reported as a "
        "duplicate of the one it stands in for")
    assert verdict["missing_cells"] == [("cnn", 2)]
    assert verdict["unexpected_cells"] == ["arm='cnn' seed=7"]
    assert verdict["passed"] is False


def test_the_gate_fails_on_a_record_that_names_no_cell_at_all():
    """The other half of the same criterion: a record whose seed is a string
    names no cell, which is different from naming one outside the study."""
    records = _study()
    records.append(_record("cnn", 0) | {"seed": "0"})
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["every_record_is_a_study_cell"] is False
    assert verdict["unexpected_cells"] == ["arm='cnn' seed='0'"]


def test_the_gate_fails_on_a_degenerate_position_band():
    """A ratio over a numerically degenerate denominator is not evidence."""
    records = _study()
    _cell(records, "cnn", 2)["position"]["steps_degenerate"] = 7
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["band_is_usable"] is False
    assert verdict["criteria"]["beats_persistence"] is True, (
        "a degenerate band must cost its own criterion alone")
    assert verdict["passed"] is False


def test_the_gate_fails_on_a_degenerate_angle_band():
    """Angle alone, because the position count is what a re-implementation
    would check and the angle band is the one Task 1 exists for."""
    records = _study()
    _cell(records, "random_vit", 0)["angle"]["steps_degenerate"] = 3
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["band_is_usable"] is False
    assert verdict["passed"] is False


def test_the_degenerate_threshold_is_zero_steps():
    """Pinned against a literal, not against the constant: raising it to 45
    would make every band "usable" and the criterion vacuous."""
    assert MAX_DEGENERATE_STEPS == 0
    records = _study()
    _cell(records, "cnn", 0)["position"]["steps_degenerate"] = 1
    assert evaluate_gate(records)["criteria"]["band_is_usable"] is False


def test_the_gate_fails_when_the_filtering_probe_loses_for_any_arm():
    records = _study()
    _cell(records, "random_vit", 2)["filtering"]["criterion_4"][
        "latent_beats_embedding"] = False
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["filtering_beats_embedding"] is False
    assert verdict["passed"] is False
    assert verdict["criteria"]["beats_persistence"] is True


def test_the_gate_fails_when_a_record_reports_no_reward_accuracy():
    records = _study()
    del _cell(records, "frozen_ssl", 0)["reward"]
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["reward_reported"] is False
    assert verdict["passed"] is False


def test_the_gate_fails_when_an_arms_curves_are_incomplete():
    records = _study()
    del _cell(records, "cnn", 1)["curves"]["floor_angle"]
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["curves_produced"] is False
    assert verdict["passed"] is False


def test_the_gate_reads_not_passed_on_the_studys_real_shaped_result(tmp_path):
    """THE MEASURED RESULT, and the one this whole task must not soften.

    Criterion 4 fails on the real trained model -- latent R^2 +0.1193 against
    the encoder embedding's +0.3287 -- and the bottleneck-free companion
    agrees: gain -0.0208 with a 95% CI of [-0.0395, -0.0040], which excludes
    zero on the wrong side. Everything else about the study is healthy. The
    honest verdict is NOT PASSED, printed as legibly as a pass, and this test
    is what stops that being arranged away.
    """
    records = _study()
    for record in records:
        record["filtering"]["criterion_4"].update(
            latent_r2=0.1193, embedding_r2=0.3287, latent_beats_embedding=False)
        record["filtering"]["gain"].update(
            gain=-0.0208, ci_low=-0.0395, ci_high=-0.0040)
    verdict = evaluate_gate(load_records(_write(tmp_path, records)))

    assert verdict["passed"] is False
    assert verdict["criteria"]["filtering_beats_embedding"] is False
    assert all(verdict["criteria"][name] for name in GATE_CRITERIA
               if name != "filtering_beats_embedding"), (
        "exactly one criterion fails on this study; a gate that failed the "
        "others too would be reporting one finding eight times")
    for arm in ARMS:
        assert verdict["per_arm"][arm]["gain_ci_excludes_zero_by_seed"] == [
            True, True, True]
        assert verdict["per_arm"][arm]["gain_mean"] == pytest.approx(-0.0208)

    text = report_study.report(records, verdict, tmp_path)
    assert _row(text, "GATE:") == "GATE: NOT PASSED"
    assert _row(text, "  [FAIL]") == "  [FAIL] filtering_beats_embedding"


# ---------------------------------------------------------------------------
# scripts/report_study.py
# ---------------------------------------------------------------------------

EXPECTED_EXIT_STATUS = {
    "EXIT_OK": 0,
    "EXIT_NO_RECORDS": 7,
    "EXIT_MISLABELLED": 8,
    "EXIT_UNREADABLE": 9,
    "EXIT_GATE_NOT_PASSED": 10,
}
"""Pinned against hand-written literals, by name set as well as by value.

An `EXIT_*` added later without a line here is one that is free to land on 2 --
argparse's own usage status -- or on top of another of the five, and the whole
point of these codes is a wrapper reading them with nobody watching.
"""


def test_every_exit_status_is_distinct_and_none_of_them_is_argparses_own():
    actual = {name: getattr(report_study, name)
              for name in EXPECTED_EXIT_STATUS}
    assert actual == EXPECTED_EXIT_STATUS
    assert len(set(actual.values())) == len(actual)
    assert 2 not in set(actual.values()), (
        "2 is argparse's usage status: a wrapper could not tell a misspelt "
        "flag from a gate that did not pass")
    assert 1 not in set(actual.values()), (
        "1 is what an uncaught traceback exits with")
    driver = {name: getattr(run_study, name)
              for name in dir(run_study) if name.startswith("EXIT_")}
    shared = {status for name, status in actual.items() if status
              in set(driver.values())
              and driver.get(name) != status}
    assert not shared, (
        "these statuses mean one thing in the driver and another here, and "
        f"the two scripts run in the same shell: {shared}")


def test_the_out_flag_stays_the_string_argparse_produces():
    """Task 5 learned this one the hard way: everything downstream coerces,
    and a `type=Path` here would mean the tests exercise a type the command
    line never produces."""
    args = report_study._parser().parse_args(["--out", "runs/somewhere"])
    assert args.out == "runs/somewhere"
    assert isinstance(args.out, str)
    assert report_study._parser().parse_args([]).out == "runs/m3_study"
    assert report_study._parser().parse_args([]).figure is None


def test_the_metric_table_puts_each_seeds_gap_in_its_own_column():
    """CHARACTER FOR CHARACTER, and readable by base: the 1000-series is
    `position.gap_final` and the 1300-series is `angle.gap_final`, so a column
    holding its neighbour's field shows up as a different thousand."""
    records = _rendering_study()
    table = report_study.metric_table(records, per_arm(records))
    assert _row(table, "arm ") == (
        "arm         metric          seed 0      seed 1      seed 2"
        "        mean  finite  unanimous  degenerate  floor>=pers")
    assert _row(table, "cnn         position") == (
        "cnn         position    +1000.0000  +1001.0000  +1003.0000"
        "  +1001.3333     3/3       True           0         3403")
    assert _row(table, "cnn         angle") == (
        "cnn         angle       +1300.0000  +1301.0000  +1303.0000"
        "  +1301.3333     3/3       True           0         3503")
    assert _row(table, "random_vit  position") == (
        "random_vit  position    +1010.0000  +1011.0000  +1013.0000"
        "  +1011.3333     3/3       True           0         3413")


def test_the_metric_table_marks_a_missing_cell_in_its_own_column():
    """THE ONE THAT MATTERS MOST. With seed 1 absent, an arm's
    `gap_final_by_seed` has two entries, and printing that list under three
    seed headings puts SEED 2'S NUMBER IN SEED 1'S COLUMN -- a full-looking
    table describing a study nobody ran."""
    records = [r for r in _rendering_study()
               if (r["arm"], r["seed"]) != ("cnn", 1)]
    table = report_study.metric_table(records, per_arm(records))
    assert _row(table, "cnn         position") == (
        "cnn         position    +1000.0000     MISSING  +1003.0000"
        "  +1001.5000     2/3       True           0         3403")


def test_the_metric_table_survives_an_arm_with_no_records_at_all():
    records = [r for r in _rendering_study() if r["arm"] != "frozen_ssl"]
    table = report_study.metric_table(records, per_arm(records))
    assert _row(table, "frozen_ssl  position") == (
        "frozen_ssl  position       MISSING     MISSING     MISSING"
        "         n/a     0/3        n/a         n/a          n/a")


def test_the_metric_table_shows_a_nan_cell_and_the_denominator_that_moved():
    """The mean over seeds 0 and 2 is +1001.5000 and the mean over all three is
    +1001.3333: the fixture's seeds are unequally spaced precisely so that the
    denominator moving CHANGES THE NUMBER, rather than the two agreeing and the
    assertion reading the same value either way."""
    records = _rendering_study()
    _cell(records, "cnn", 1)["position"]["gap_final"] = float("nan")
    table = report_study.metric_table(records, per_arm(records))
    assert _row(table, "cnn         position") == (
        "cnn         position    +1000.0000        +nan  +1003.0000"
        "  +1001.5000     2/3      False           0         3403")


def test_the_criterion_4_table_puts_the_latent_against_the_embedding():
    """The 2100-series is criterion 4's `latent_r2` and the 2200-series its
    `embedding_r2`; the gain block's own two levels are the 2600 and 2700
    series and belong in the other table. A column holding the wrong one shows
    up as a different thousand."""
    table = report_study.criterion_4_table(_rendering_study())
    assert _row(table, "cell") == (
        "cell                latent_r2     embed_r2   latent_beats_embedding")
    assert _row(table, "frozen_ssl/s0") == (
        "frozen_ssl/s0      +2105.0000   +2205.0000                    False")


def test_the_criterion_4_table_renders_its_boolean_the_other_way():
    """A bool has two values, so one rendering excludes only one literal."""
    table = report_study.criterion_4_table(_study(latent_wins=True))
    assert _row(table, "frozen_ssl/s0") == (
        "frozen_ssl/s0      +2105.0000   +2205.0000                     True")


def test_the_gain_table_carries_both_levels_the_interval_and_the_ridges():
    """R1, every column of it. The gain is ~0.02 while the levels are ~0.31, so
    the difference alone cannot tell a small gap between two good probes from a
    small gap between two useless ones."""
    table = report_study.gain_table(_rendering_study())
    assert _row(table, "cell") == (
        "cell                joint_r2    embed_r2        gain"
        "                     95% CI  CI excl 0  joint_ridge  embed_ridge"
        "  selected  windows")
    assert _row(table, "frozen_ssl/s0") == (
        "frozen_ssl/s0     +2605.0000  +2705.0000  +2305.0000"
        "   [+2405.0000, +2505.0000]       True      1.0e+03      1.0e+05"
        "      True     2805")


def test_the_gain_table_renders_its_boolean_the_other_way():
    table = report_study.gain_table(_study(ridge_selected=False))
    assert _row(table, "frozen_ssl/s0") == (
        "frozen_ssl/s0     +2605.0000  +2705.0000  +2305.0000"
        "   [+2405.0000, +2505.0000]       True      1.0e+03      1.0e+05"
        "     False     2805")


def test_the_gain_table_says_n_a_for_an_interval_with_one_end_missing():
    """An interval with one end is not an interval, and rendering that as
    `False` would say "the interval covers zero", which nobody measured."""
    records = _rendering_study()
    _cell(records, "cnn", 0)["filtering"]["gain"]["ci_high"] = float("nan")
    row = _row(report_study.gain_table(records), "cnn/s0")
    assert row == (
        "cnn/s0            +2600.0000  +2700.0000  +2300.0000"
        "         [+2400.0000, +nan]        n/a      1.0e+03      1.0e+05"
        "      True     2800")


def test_the_CI_column_reads_an_interval_below_zero_AND_one_above():
    """THE COLUMN THE MEASURED RESULT IS READ OFF, and each half of the
    disjunction that renders it, ALONE.

    `bool(low > 0.0 or high < 0.0)` has two terms and the fixture's interval
    is [+2400, +2500] -- BOTH ENDS POSITIVE -- so `low > 0.0` answered every
    assertion in this module and the second term was never the deciding one.
    The half that hides there is the half the real study exercises: the
    measured bottleneck-free gain is -0.0208 with a 95% CI of
    [-0.0395, -0.0040], entirely BELOW zero, which is the finding. With only
    the first term the column prints `False` for that interval -- "the
    interval covers zero", "the negative gain is not distinguishable from
    zero" -- the opposite of what was measured, in the column a reader looks
    at to decide whether the negative gain is real, corroborating the one
    criterion the milestone fails.

    So: the measured interval, where only `high < 0.0` is true; one entirely
    above zero, where only `low > 0.0` is true; and one straddling zero, where
    neither is. Three rows, three renderings, no term carried by its
    neighbour.
    """
    records = _rendering_study()
    _cell(records, "cnn", 0)["filtering"]["gain"].update(
        gain=-0.0208, ci_low=-0.0395, ci_high=-0.0040)
    _cell(records, "cnn", 1)["filtering"]["gain"].update(
        gain=0.0325, ci_low=0.0040, ci_high=0.0610)
    _cell(records, "cnn", 2)["filtering"]["gain"].update(
        gain=-0.0100, ci_low=-0.0300, ci_high=0.0200)
    table = report_study.gain_table(records)
    assert _row(table, "cnn/s0") == (
        "cnn/s0            +2600.0000  +2700.0000     -0.0208"
        "         [-0.0395, -0.0040]       True      1.0e+03      1.0e+05"
        "      True     2800"), (
        "the MEASURED study: the interval is entirely below zero, so only "
        "`high < 0.0` can be what decided this row")
    assert _row(table, "cnn/s1") == (
        "cnn/s1            +2601.0000  +2701.0000     +0.0325"
        "         [+0.0040, +0.0610]       True      1.0e+03      1.0e+05"
        "      True     2801"), (
        "entirely above zero, so only `low > 0.0` can be what decided it")
    assert _row(table, "cnn/s2") == (
        "cnn/s2            +2603.0000  +2703.0000     -0.0100"
        "         [-0.0300, +0.0200]      False      1.0e+03      1.0e+05"
        "      True     2803"), (
        "straddling zero: neither term is true, so a column that always said "
        "True would fail here rather than pass on the two rows above")
    assert report_study._excludes_zero(-0.0395, -0.0040) == "True"
    assert report_study._excludes_zero(0.0040, 0.0610) == "True"
    assert report_study._excludes_zero(-0.0300, 0.0200) == "False"


def test_the_ridge_block_lists_the_decade_every_gain_was_computed_at():
    block = report_study.ridge_block(ridge_groups(_rendering_study()))
    assert _row(block, "selected") == (
        "selected    joint_ridge  embed_ridge  cells  finite  gain_mean"
        "  members")
    assert _row(block, "True") == (
        "True            1.0e+03      1.0e+05      9       9 +2306.3333  "
        "cnn/s0 cnn/s1 cnn/s2 frozen_ssl/s0 frozen_ssl/s1 frozen_ssl/s2 "
        "random_vit/s0 random_vit/s1 random_vit/s2")
    assert "WARNING" not in block, (
        "nine cells that agree must not be warned about, or the warning "
        "means nothing when they do not")


def test_the_ridge_block_warns_when_the_cells_did_not_agree():
    """R1. There is no correction for this, only the disclosure."""
    records = _rendering_study()
    for seed in SEEDS:
        _cell(records, "cnn", seed)["filtering"]["gain"][
            "joint_ridge"] = ALT_JOINT_RIDGE
    block = report_study.ridge_block(ridge_groups(records))
    warning = _row(block, "WARNING: the nine cells")
    assert "SIGN is not interpretable" in warning
    assert len(block.splitlines()) == 4


def test_the_ridge_block_warns_when_a_cell_skipped_selection_entirely():
    """The second warning, alone: a gain computed at `fit_probe`'s default
    penalty is not the same estimator as a selected one."""
    block = report_study.ridge_block(
        ridge_groups(_study(ridge_selected=False)))
    assert _row(block, "False") .startswith("False           1.0e+03")
    warning = _row(block, "WARNING: at least one cell")
    assert "NO ridge selection" in warning
    assert not any(line.startswith("WARNING: the nine cells")
                   for line in block.splitlines()), (
        "one group cannot have disagreed with itself")


def test_the_ridge_block_warns_when_a_cell_does_not_SAY_whether_it_selected():
    """`is not True`, and the half of it an explicit `False` cannot reach.

    The warning is deliberately written to cover BOTH "the cell says it did
    not select" and "the cell does not say whether it did" -- `_flag` keeps
    those apart on purpose, and only the first was tested, because the sibling
    above builds its fixture with an explicit `False`, on which `is False` and
    `is not True` agree. With the flag DELETED the group renders `n/a` in the
    `selected` column and `is False` stops warning: the reader is shown a
    grouping of gains that may not be the same estimator, with nothing saying
    so.
    """
    records = _rendering_study()
    for seed in SEEDS:
        del _cell(records, "cnn", seed)["filtering"]["gain"]["ridge_selected"]
    block = report_study.ridge_block(ridge_groups(records))
    assert _row(block, "n/a") == (
        "n/a             1.0e+03      1.0e+05      3       3 +2301.3333  "
        "cnn/s0 cnn/s1 cnn/s2")
    assert "NO ridge selection" in _row(block, "WARNING: at least one cell")


def test_the_ridge_block_does_not_warn_about_a_decade_its_own_table_refutes():
    """M9. A named warning the table beside it refutes is one a reader learns
    to discount, and this block is the whole of R1's disclosure.

    `ridge_groups` keys on `(ridge_selected, joint_ridge, embedding_ridge)`,
    so two groups can differ in the FLAG alone. The first warning asked
    `len(groups) > 1`, so it fired on that split and printed "the nine cells
    did not all select the same ridge decade" directly under two rows whose
    `joint_ridge` and `embed_ridge` columns hold THE SAME TWO NUMBERS -- the
    only two columns the sentence is about.

    What that split IS about is the second warning, and it still fires: the
    disclosure is not lost, it is told truthfully. The sibling above, where
    the two groups really do sit at different decades, is the other rendering.
    """
    records = _rendering_study()
    for seed in SEEDS:
        _cell(records, "cnn", seed)["filtering"]["gain"][
            "ridge_selected"] = False
    block = report_study.ridge_block(ridge_groups(records))
    assert _row(block, "False") == (
        "False           1.0e+03      1.0e+05      3       3 +2301.3333  "
        "cnn/s0 cnn/s1 cnn/s2")
    assert _row(block, "True") == (
        "True            1.0e+03      1.0e+05      6       6 +2308.8333  "
        "frozen_ssl/s0 frozen_ssl/s1 frozen_ssl/s2 "
        "random_vit/s0 random_vit/s1 random_vit/s2"), (
        "two groups, and both ridge columns identical in both rows -- which "
        "is what makes the decade warning below a claim the table refutes")
    assert not any(line.startswith("WARNING: the nine cells")
                   for line in block.splitlines()), (
        "no cell moved decade; the flag is what split the groups")
    assert "NO ridge selection" in _row(block, "WARNING: at least one cell"), (
        "and the warning that IS about the flag still fires")


def test_the_ridge_block_prints_the_denominator_beside_the_count_of_cells():
    """`cells 9` beside a mean over six was the one row in the report that
    broke the module docstring's promise that the denominator of every mean is
    printed beside it. The three cells whose gain was never computed are still
    named as members, which is why the two columns have to differ."""
    records = _rendering_study()
    for seed in SEEDS:
        del _cell(records, "cnn", seed)["filtering"]["gain"]["gain"]
    block = report_study.ridge_block(ridge_groups(records))
    assert _row(block, "selected") == (
        "selected    joint_ridge  embed_ridge  cells  finite  gain_mean"
        "  members")
    assert _row(block, "True") == (
        "True            1.0e+03      1.0e+05      9       6 +2308.8333  "
        "cnn/s0 cnn/s1 cnn/s2 frozen_ssl/s0 frozen_ssl/s1 frozen_ssl/s2 "
        "random_vit/s0 random_vit/s1 random_vit/s2"), (
        "nine cells, six of them in the mean, all nine named -- and the mean "
        "is +2308.8333 rather than the nine-cell +2306.3333, so a mean taken "
        "over the wrong set would not render this row")


def test_the_ridge_block_names_a_record_that_names_no_cell():
    """`record_cell` is None for a record whose seed is a string, and
    `for arm, seed in group['cells']` cannot unpack that. The fallback keeps a
    damaged record costing a legible table row rather than the report -- which
    is drawn after the verdict is already on the operator's screen.

    The stray carries seed "7", not "0": a record that renamed itself after
    the cell it was copied from would render `cnn/s0` twice and the row would
    read the same whether the fallback ran or not.
    """
    records = _rendering_study()
    records.append(_record("cnn", 0) | {"seed": "7"})
    assert _row(report_study.ridge_block(ridge_groups(records)), "True") == (
        "True            1.0e+03      1.0e+05     10      10 +2305.7000  "
        "cnn/s0 cnn/s1 cnn/s2 frozen_ssl/s0 frozen_ssl/s1 frozen_ssl/s2 "
        "random_vit/s0 random_vit/s1 random_vit/s2 cnn/s7")


def test_the_reward_table_reports_the_numbers_and_the_target_flag():
    table = report_study.reward_table(_rendering_study())
    assert _row(table, "cell") == (
        "cell                       mse    baseline_mse           r2   events"
        "    steps  degenerate_target")
    assert _row(table, "random_vit/s2") == (
        "random_vit/s2       2913.00000      3013.00000   +3113.0000     3213"
        "     3313               True")


def test_the_reward_table_renders_its_boolean_the_other_way():
    table = report_study.reward_table(_study(reward_degenerate=False))
    assert _row(table, "random_vit/s2").endswith("     3313              False")


def test_the_training_table_reports_throughput_per_cell():
    table = report_study.training_table(_rendering_study())
    assert _row(table, "cell") == (
        "cell               steps/s   kl_rate  kl_dyn_max  loss_last20   wall_h")
    assert _row(table, "frozen_ssl/s1") == (
        "frozen_ssl/s1      1606.00  1706.000    1806.000    1906.0000     5.72")


def test_the_gate_block_renders_a_failing_verdict():
    records = _study()
    _cell(records, "cnn", 0)["filtering"]["criterion_4"][
        "latent_beats_embedding"] = False
    block = report_study.gate_block(evaluate_gate(records))
    assert _row(block, "  [FAIL]") == "  [FAIL] filtering_beats_embedding"
    assert _row(block, "GATE:") == "GATE: NOT PASSED"
    assert len([line for line in block.splitlines()
                if line.startswith("  [PASS]")]) == len(GATE_CRITERIA) - 1


def test_the_gate_block_renders_a_passing_verdict():
    """The other rendering. A block that said NOT PASSED unconditionally would
    satisfy every failure test in this module."""
    block = report_study.gate_block(evaluate_gate(_study()))
    assert _row(block, "GATE:") == "GATE: PASSED"
    assert len([line for line in block.splitlines()
                if line.startswith("  [PASS]")]) == len(GATE_CRITERIA)
    assert "[FAIL]" not in block


def test_the_gate_block_names_the_criteria_it_could_not_evaluate():
    """Spec section 4 has five criteria and one of them is the pytest suite.
    Printing only the computable ones is how a five-criterion gate gets
    reported as a four-criterion pass."""
    block = report_study.gate_block(evaluate_gate(_study()))
    assert _row(block, "  [ n/a]") == "  [ n/a] invariant_tests_green"
    # The reason is wrapped for the terminal, so the phrase is looked for in
    # the unwrapped text; against the wrapped block a phrase that straddled a
    # line break would fail for the wrong reason.
    unwrapped = " ".join(line.strip() for line in block.splitlines())
    assert "spec section 4 criterion 5" in unwrapped
    assert "run it and read its exit status" in unwrapped


def test_the_gate_block_names_every_missing_cell():
    records = [r for r in _study()
               if (r["arm"], r["seed"]) not in {("cnn", 1), ("random_vit", 2)}]
    block = report_study.gate_block(evaluate_gate(records))
    assert _row(block, "  MISSING CELLS") == (
        "  MISSING CELLS (2 of 9): cnn/s1, random_vit/s2")


def test_the_gate_block_names_a_duplicated_and_an_unexpected_cell():
    records = _study()
    records.append(_record("cnn", 0))
    records.append(_record("cnn", 0) | {"seed": 17})
    block = report_study.gate_block(evaluate_gate(records))
    assert _row(block, "  DUPLICATE CELLS") == "  DUPLICATE CELLS: cnn/s0"
    assert _row(block, "  RECORDS THAT ARE NOT") == (
        "  RECORDS THAT ARE NOT STUDY CELLS: arm='cnn' seed=17")


def test_the_gate_block_names_the_cell_and_the_curve_that_came_up_short():
    """`curves_produced` was the ONE criterion a reader could not locate.

    Every other failing criterion is attributable from the report:
    `beats_persistence` and `band_is_usable` have per-seed columns and their
    own `unanimous`/`degenerate` columns, `filtering_beats_embedding` and
    `reward_reported` have their own tables, and the three cell-accounting
    criteria name their cells. A short `floor_angle` in one cell printed
    `  [FAIL] curves_produced` and nothing else -- the word "curve" appeared
    nowhere else in the output -- leaving the operator to open nine JSON files
    to find which of six curves in which cell is missing.
    """
    records = _study()
    del _cell(records, "frozen_ssl", 1)["curves"]["floor_angle"]
    _cell(records, "cnn", 0)["curves"]["rssm_position"] = [1.0, 2.0]
    # A record that names NO cell, also short a curve. `record_cell` answers
    # None for it, and `f"{cell[0]}/s{cell[1]}"` cannot unpack a None -- so the
    # line that is supposed to name a broken record has to be able to name the
    # most broken one. It is listed by its own self-description, the same text
    # the RECORDS THAT ARE NOT STUDY CELLS line uses, with the seed's `repr`
    # visible so `seed=7` and `seed='7'` cannot read alike.
    stray = _record("cnn", 0) | {"seed": "7"}
    del stray["curves"]["floor_position"]
    records.append(stray)
    verdict = evaluate_gate(records)
    assert verdict["criteria"]["curves_produced"] is False
    block = report_study.gate_block(verdict)
    assert _row(block, "  CURVES INCOMPLETE") == (
        "  CURVES INCOMPLETE (3 of 9): cnn/s0 (rssm_position), "
        "frozen_ssl/s1 (floor_angle), arm='cnn' seed='7' (floor_position)"), (
        "each cell with the name of ITS OWN short curve: a line that named "
        "all six for any of them would be no more use than the bare FAIL. "
        "The count is of incomplete records against the nine cells EXPECTED, "
        "which is why a tenth record can make it read 3 of 9")
    assert _row(block, "  [FAIL] curves_produced") == (
        "  [FAIL] curves_produced")


def test_the_gate_block_says_nothing_about_curves_when_all_six_are_there():
    """The other rendering. A line printed unconditionally would be satisfied
    by every complete study as well as by every broken one."""
    block = report_study.gate_block(evaluate_gate(_study()))
    assert "CURVES INCOMPLETE" not in block
    assert "curve" not in block.replace("curves_produced", "")


def test_the_report_states_the_policy_the_means_were_taken_under():
    """Either policy is defensible; neither is defensible in silence."""
    records = _study()
    text = report_study.report(records, evaluate_gate(records), "runs/x")
    assert "means skip cells whose gap_closed is undefined" in text
    assert "the finite column is the denominator" in text.replace("\n", " ")
    assert "fails band_is_usable instead" in text
    assert _row(text, "  --out") == "  --out       runs/x"
    assert _row(text, "  records") == (
        "  records     9 of 9 expected cells (3 arms x 3 seeds)")


#: Every row of every per-cell table, as a LITERAL label, in the order the
#: study owes them: three arms x three seeds, the middle arm in the middle.
_CELL_ROWS = tuple(f"{arm}/s{seed}" for arm in ARMS for seed in SEEDS)

#: The metric table's rows: one per arm and metric, its label column being
#: `f"{arm:<12}{metric:<10}"`.
_METRIC_ROWS = tuple(
    f"{arm:<12}{metric:<10}" for arm in ARMS for metric in METRICS)

#: The seven sections of the report: the section header, the first line of the
#: block that must sit DIRECTLY under it, and the label prefix of every row of
#: that block, in order. `None` for the gate block, whose lines are asserted
#: against `GATE_CRITERIA` at the foot of the test instead.
#:
#: The tables are individually pinned character-for-character above; what this
#: pins is the JOIN between them and the DOCUMENT, and the SET OF ROWS each one
#: prints. Five of the seven could be deleted from `report()` -- the training
#: table, the criterion-4 table, the gain table, the ridge block and the reward
#: table, i.e. both artefacts R1 exists to produce and the table carrying the
#: number the milestone fails on -- leaving their section headers behind, and
#: the suite stayed green, because every one of them was tested only by being
#: called directly.
#:
#: NOT ONE OF THESE EXPECTATIONS IS PRODUCED BY THE CODE UNDER TEST. The first
#: version of this table carried a CALLABLE per section and built each expected
#: body by calling the very renderer the assertion was meant to judge, so a
#: mutated renderer answered the assertion with its own output: with the last
#: arm, the last seed or a named arm dropped from `criterion_4_table` or
#: `training_table`, the shortened table still sat under its header and matched
#: the shortened expectation, and the suite stayed green while the criterion-4
#: table lost three cells two lines under a header still reading "records 9 of
#: 9 expected cells". That is the L7 species living inside the fix for L7.
#: Here every column header is a string typed out in full and every row list is
#: built from ARMS, SEEDS and METRICS -- constants of `mbfps.utils.config` and
#: `mbfps.eval.summary`, pinned to their literal values in the test below and
#: in `tests/utils/test_config.py`, not owned by `scripts/report_study.py`. A
#: renderer that drops, reorders or renames a row cannot make the expectation
#: move with it.
REPORT_SECTIONS = [
    ("--- gap_closed at the final horizon step, by arm and seed "
     "(fraction of the persistence-to-floor band) ---",
     "arm         metric          seed 0      seed 1      seed 2"
     "        mean  finite  unanimous  degenerate  floor>=pers",
     _METRIC_ROWS),
    ("--- training ---",
     "cell               steps/s   kl_rate  kl_dyn_max  loss_last20   wall_h",
     _CELL_ROWS),
    ("--- filtering, gate criterion 4: does the posterior latent beat the "
     "raw encoder embedding of the same frame? ---",
     "cell                latent_r2     embed_r2   latent_beats_embedding",
     _CELL_ROWS),
    ("--- filtering, the bottleneck-free companion: what appending h to "
     "that same embedding buys ---",
     "cell                joint_r2    embed_r2        gain"
     "                     95% CI  CI excl 0  joint_ridge  embed_ridge"
     "  selected  windows",
     _CELL_ROWS),
    ("--- the ridge decade each gain was computed at ---",
     "selected    joint_ridge  embed_ridge  cells  finite  gain_mean  members",
     # One group: every cell of both fixtures below carries the same three-part
     # ridge key, so the block is its column header and exactly one row.
     ("True",)),
    ("--- reward prediction (spec criterion 3: reported per arm, and "
     "deliberately not gated -- the target is near-constant by "
     "construction) ---",
     "cell                       mse    baseline_mse           r2   events"
     "    steps  degenerate_target",
     _CELL_ROWS),
    # The gate block's first line depends on the verdict rather than on the
    # study's shape, so the caller spells it out; its rows are asserted
    # against GATE_CRITERIA at the foot of the test below.
    ("--- M3 exit gate (spec section 4) ---", None, None),
]


def _section_body(text: str, header: str) -> list[str]:
    """The lines directly under `header`, to the blank line that ends them.

    The section headers are distinct, so this can only return the block that
    belongs to the header it was asked for; a body picked out by matching its
    own text would be satisfied by any table rendering the same rows.
    """
    lines = text.splitlines()
    assert lines.count(header) == 1, (
        f"{lines.count(header)} lines are {header!r}; the block below it "
        f"cannot be identified:\n{text}")
    body = []
    for line in lines[lines.index(header) + 1:]:
        if not line:
            break
        body.append(line)
    return body


def _assert_sections(text: str, gate_first_line: str) -> None:
    """Every section's header, its column header and its full row set."""
    for header, column_header, rows in REPORT_SECTIONS:
        if column_header is None:
            column_header = gate_first_line
        assert f"{header}\n{column_header}" in text, (
            f"the section header {header!r} is in the report and the table it "
            "names is not directly under it")
        if rows is None:
            continue
        body = _section_body(text, header)
        assert len(body) - 1 == len(rows), (
            f"{header!r} printed {len(body) - 1} rows under its column "
            f"header and the study owes {len(rows)}:\n" + "\n".join(body))
        # The length is asserted FIRST: `zip` truncates to the shorter of the
        # two, so a table short of a row would otherwise compare only the rows
        # it still has and agree with itself.
        assert [line[:len(row)] for line, row in zip(body[1:], rows)] == list(
            rows), (
            f"{header!r} does not print one row per cell, in ARMS x SEEDS "
            "order:\n" + "\n".join(body))


def test_every_section_of_the_report_is_actually_in_the_report():
    """Header, column header and one row per cell, for all seven sections.

    The join is asserted as the literal `f"{header}\\n{column_header}"` rather
    than as two `in` checks: a header line and a row that both appear
    somewhere is satisfied by a report whose sections have been shuffled.
    """
    assert ARMS == ("cnn", "frozen_ssl", "random_vit"), (
        "the row expectations are built from ARMS, METRICS and SEEDS, which "
        "the renderer reads too; pinned literally here so that a constant "
        "losing an entry cannot move both sides of the comparison together")
    assert METRICS == ("position", "angle")
    assert aggregate.SEEDS == SEEDS == (0, 1, 2)
    records = _rendering_study()
    verdict = evaluate_gate(records)
    text = report_study.report(records, verdict, "runs/x")
    _assert_sections(text, "  [PASS] all_nine_cells_present")
    assert [line for line in text.splitlines() if line.startswith("---")] == [
        header for header, _, _ in REPORT_SECTIONS], (
        "seven sections, in this order, and no eighth: a header with no table "
        "under it is what every one of these deletions left behind")
    gate = _section_body(text, REPORT_SECTIONS[-1][0])
    assert [line[len("  [PASS] "):] for line in gate
            if line.startswith(("  [PASS] ", "  [FAIL] "))] == list(
        GATE_CRITERIA), (
        "every criterion, by name, in GATE_CRITERIA order -- a conjunction "
        "reported with one of its terms missing is a five-criterion gate "
        "reported as a four-criterion pass")
    assert text.splitlines()[-1] == "GATE: NOT PASSED", (
        "the verdict is the LAST line of the report: the rendering fixture "
        "has latent_beats_embedding False, which is what the real trained "
        "model measured")


def test_all_five_per_cell_tables_MARK_a_missing_cell_rather_than_dropping_it():
    """THE MARKER, IN ALL FIVE TABLES. Only the metric table's was pinned.

    Deleting the `MISSING` line from `training_table`, `criterion_4_table`,
    `gain_table` or `reward_table` -- each of which prints it for a cell with
    no record and then `continue`s -- left the whole suite green. With
    frozen_ssl's three records absent the shipped criterion-4 table prints
    three MISSING rows between the cnn and the random_vit rows; without that
    line it prints six rows and nothing else: a full-looking, evenly spaced
    table describing a study nobody ran, in the table carrying the number the
    milestone fails on, two lines under a header that still says how many
    cells are missing.

    The four markers are at four DIFFERENT column widths -- 10, 13, 12 and 14
    -- so no one of these four expectations can be satisfied by another
    table's row, and the section loop re-runs here over the incomplete study,
    so a table that dropped the row instead of marking it is short by three.
    """
    records = [r for r in _rendering_study() if r["arm"] != "frozen_ssl"]
    verdict = evaluate_gate(records)
    text = report_study.report(records, verdict, "runs/x")
    assert _row(text, "  records") == (
        "  records     6 of 9 expected cells (3 arms x 3 seeds)")
    assert _row(text, "  MISSING CELLS") == (
        "  MISSING CELLS (3 of 9): frozen_ssl/s0, frozen_ssl/s1, "
        "frozen_ssl/s2")
    _assert_sections(text, "  [FAIL] all_nine_cells_present")
    assert _row(text, "frozen_ssl  position") == (
        "frozen_ssl  position       MISSING     MISSING     MISSING"
        "         n/a     0/3        n/a         n/a          n/a")
    assert _row(text, "frozen_ssl  angle") == (
        "frozen_ssl  angle          MISSING     MISSING     MISSING"
        "         n/a     0/3        n/a         n/a          n/a")
    for header, marker in (
        ("--- training ---", "frozen_ssl/s0      MISSING"),
        ("--- filtering, gate criterion 4: does the posterior latent beat "
         "the raw encoder embedding of the same frame? ---",
         "frozen_ssl/s0         MISSING"),
        ("--- filtering, the bottleneck-free companion: what appending h to "
         "that same embedding buys ---",
         "frozen_ssl/s0        MISSING"),
        ("--- reward prediction (spec criterion 3: reported per arm, and "
         "deliberately not gated -- the target is near-constant by "
         "construction) ---",
         "frozen_ssl/s0          MISSING"),
    ):
        body = _section_body(text, header)
        assert body[4:7] == [
            marker,
            marker.replace("/s0", "/s1"),
            marker.replace("/s0", "/s2"),
        ], (
            f"the three cells of the arm that never ran are marked MISSING, "
            f"in their own rows, between cnn's and random_vit's, under "
            f"{header!r}:\n" + "\n".join(body))


def test_the_report_renders_the_ridge_grouping_out_of_the_VERDICT():
    """R1's disclosure has three independent ways to vanish -- the block
    deleted from `report()`, the section's data emptied at the verdict, and
    the warnings' condition weakened -- and this is the middle one.

    `report()` renders `verdict["ridge_groups"]`, so a verdict carrying `[]`
    prints the section header, the column header, no group, no members, no
    mean and NEITHER warning. Every ridge test called `ridge_groups()`
    directly, so nothing noticed.
    """
    records = _rendering_study()
    verdict = evaluate_gate(records)
    groups = verdict["ridge_groups"]
    assert [g["n_cells"] for g in groups] == [9]
    assert groups[0]["cells"] == [
        (arm, seed) for arm in ARMS for seed in SEEDS]
    assert groups[0]["gain_mean"] == pytest.approx(
        sum(_value("filtering.gain.gain", arm, seed)
            for arm in ARMS for seed in SEEDS) / 9)
    text = report_study.report(records, verdict, "runs/x")
    assert _row(text, "True            1.0e+03").endswith("random_vit/s2"), (
        "the members are in the printed report, not only in the return value "
        "of a function the report might not be calling")


def test_a_non_numeric_leaf_costs_its_own_column_and_not_the_report():
    """`_fmt` and `_count` degrade to `str(value)` on a value that is not a
    number, and neither `except` clause was ever entered.

    The closest existing test replaces whole BLOCKS with scalars, and `_get`
    answers `None` for a scalar block, so both formatters take their
    `value is None` branch and the except never runs. Reaching it needs a
    non-numeric value at a LEAF -- `gap_final` holding the string "1.2.3", or
    `n_scored_windows` holding "many" -- which is what a hand-edited or
    half-converted record has. Raising there loses the WHOLE report to one bad
    field after the 33 GPU hours are already paid for.
    """
    assert report_study._fmt("1.2.3", "+.4f") == "1.2.3"
    assert report_study._count("many") == "many"
    assert report_study._fmt(None, "+.4f") == "n/a", (
        "the None branch and the except branch are two different guards; "
        "spelled out here so a fix that collapsed them shows up")
    records = _rendering_study()
    _cell(records, "cnn", 1)["position"]["gap_final"] = "1.2.3"
    _cell(records, "cnn", 2)["filtering"]["gain"]["n_scored_windows"] = "many"
    verdict = evaluate_gate(records)
    text = report_study.report(records, verdict, "runs/x")
    assert _row(text, "cnn         position") == (
        "cnn         position    +1000.0000       1.2.3  +1003.0000"
        "  +1001.5000     2/3      False           0         3403"), (
        "the bad leaf is printed as the text it holds, in its own column, and "
        "the two good seeds either side of it are untouched")
    assert _row(text, "cnn/s2            +2603.0000").endswith("     many")
    assert _row(text, "GATE:") == "GATE: NOT PASSED", (
        "a string where a number belongs is not a measurement, so the "
        "criterion that read it fails -- the report is what must survive")


def test_the_report_says_how_many_of_the_nine_cells_it_found():
    records = [r for r in _study() if r["arm"] != "cnn"]
    text = report_study.report(records, evaluate_gate(records), "runs/x")
    assert _row(text, "  records") == (
        "  records     6 of 9 expected cells (3 arms x 3 seeds)")


# --- main ------------------------------------------------------------------

def test_main_prints_the_report_and_exits_ok_on_a_passing_study(
        tmp_path, capsys):
    status = report_study.main(["--out", str(_write(tmp_path, _study()))])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_OK
    assert _row(out, "GATE:") == "GATE: PASSED"
    assert _row(out, "cnn         position").startswith(
        "cnn         position    +1000.0000  +1001.0000  +1003.0000")


def test_main_exits_gate_not_passed_when_the_gate_does_not_pass(
        tmp_path, capsys):
    records = _study()
    _cell(records, "cnn", 0)["filtering"]["criterion_4"][
        "latent_beats_embedding"] = False
    status = report_study.main(["--out", str(_write(tmp_path, records))])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_GATE_NOT_PASSED
    assert _row(out, "GATE:") == "GATE: NOT PASSED"
    assert _row(out, "  [FAIL]") == "  [FAIL] filtering_beats_embedding"
    assert _row(out, "cnn         position").startswith(
        "cnn         position    +1000.0000"), (
        "the whole report is printed BEFORE the status is decided; a gate "
        "that did not pass is a result, not an error that suppresses output")


def test_main_gates_over_the_STUDYS_arms_not_over_the_ones_that_ran(
        tmp_path, capsys):
    """THE WORST OUTCOME THIS FILE CAN PRODUCE: a milestone declared met on a
    study that did not run.

    `evaluate_gate` judges every criterion over `arms` and not over whichever
    arms happen to have records -- its docstring says so and says why -- and
    eight mutations of that contract inside the function are killed. Nothing
    pinned the ONE call site that supplies `arms`. `main` calling
    `evaluate_gate(records, arms=the arms it found)` prints GATE: PASSED,
    returns EXIT_OK, reports "6 of 9" as "6 of 6", lists no missing cells and
    marks every criterion PASS.

    The cnn arm is 25 of the study's 33 hours and is the one that does not
    finish on a rented box. The only main-level incomplete-study test removes
    a single SEED, which leaves all three arm names present and therefore
    cannot tell the two calls apart.

    Every per-arm criterion is asserted, not just the cell-accounting ones: an
    arm with no record fails all five, and it is the five that would otherwise
    be read off the two cheap arms that did finish.
    """
    records = [r for r in _study() if r["arm"] != "cnn"]
    status = report_study.main(["--out", str(_write(tmp_path, records))])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_GATE_NOT_PASSED
    assert _row(out, "GATE:") == "GATE: NOT PASSED"
    assert _row(out, "  records") == (
        "  records     6 of 9 expected cells (3 arms x 3 seeds)")
    assert _row(out, "  MISSING CELLS") == (
        "  MISSING CELLS (3 of 9): cnn/s0, cnn/s1, cnn/s2")
    for criterion in ("all_nine_cells_present", "beats_persistence",
                      "band_is_usable", "filtering_beats_embedding",
                      "reward_reported", "curves_produced"):
        assert _row(out, f"  [FAIL] {criterion}") == f"  [FAIL] {criterion}"
    assert "[PASS]" in out, (
        "the two arms that DID finish still pass what they can -- a report "
        "that failed everything would satisfy this test without judging "
        "anything over the missing arm")
    assert _row(out, "cnn         position") == (
        "cnn         position       MISSING     MISSING     MISSING"
        "         n/a     0/3        n/a         n/a          n/a"), (
        "the arm that never ran has a row of MISSING, not an absent row: an "
        "arm that is not in the table is an arm nobody looks for")


def test_main_gates_over_the_STUDYS_seeds_not_over_the_ones_that_ran(
        tmp_path, capsys):
    """The other half of the same call site, and it fails the same way.

    `evaluate_gate(records, seeds=the seeds it found)` is the seed-shaped twin
    of the arms mutation above: a study that lost seed 2 in all three arms --
    one overnight run that never started, three cells -- would be gated as a
    complete 3x2 study and report `6 of 6`. Removing a seed from ONE arm, which
    is what the existing incomplete-study test does, leaves seed 2 present in
    the other two arms and so cannot tell the two calls apart, exactly as
    removing one seed cannot tell the arms mutation apart.
    """
    records = [r for r in _study() if r["seed"] != 2]
    status = report_study.main(["--out", str(_write(tmp_path, records))])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_GATE_NOT_PASSED
    assert _row(out, "GATE:") == "GATE: NOT PASSED"
    assert _row(out, "  records") == (
        "  records     6 of 9 expected cells (3 arms x 3 seeds)")
    assert _row(out, "  MISSING CELLS") == (
        "  MISSING CELLS (3 of 9): cnn/s2, frozen_ssl/s2, random_vit/s2")
    assert _row(out, "  [FAIL] all_nine_cells_present") == (
        "  [FAIL] all_nine_cells_present")
    assert _row(out, "cnn         position") == (
        "cnn         position    +1000.0000  +1001.0000     MISSING"
        "  +1000.5000     2/3       True           0         3401"), (
        "the seed the study owes is still a column, and the finite "
        "denominator is still 3 -- 2/2 would read as a complete arm")


def test_main_reports_an_incomplete_study_rather_than_aggregating_it(
        tmp_path, capsys):
    records = [r for r in _study() if (r["arm"], r["seed"]) != ("cnn", 2)]
    status = report_study.main(["--out", str(_write(tmp_path, records))])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_GATE_NOT_PASSED
    assert _row(out, "  MISSING CELLS") == "  MISSING CELLS (1 of 9): cnn/s2"
    assert _row(out, "  records") == (
        "  records     8 of 9 expected cells (3 arms x 3 seeds)")


def test_main_names_the_directory_it_ACTUALLY_read(tmp_path, capsys):
    """The `--out` line is the outermost label on the nine numbers below it.

    `main` passes `args.out` to `report`, and every `main` test asserted a gate
    row, a metric row or an error line -- never the header -- so `main` could
    print a directory it never opened and stay green. The operator is told to
    move files between study directories and re-run, so a header naming the
    wrong one is the same failure class as the mislabelled record the whole
    module refuses to aggregate: a number under the wrong label has no symptom.
    """
    where = tmp_path / "some_other_study"
    where.mkdir()
    report_study.main(["--out", str(_write(where, _study()))])
    out = capsys.readouterr().out
    assert _row(out, "  --out") == f"  --out       {where}"
    assert "runs/m3_study" not in out, (
        "the parser's DEFAULT is runs/m3_study; a header hard-coded to it "
        "would render correctly on every run that used the default")


def test_the_script_exits_with_the_status_main_returned(tmp_path):
    """`raise SystemExit(main())`, the one line the suite never runs.

    The test module loads this script with `spec_from_file_location`, so
    `__name__` is "report_study" and the `if __name__ == "__main__"` guard
    never fires; every status test calls `main()` and asserts its RETURN
    VALUE. The five statuses exist so an unattended wrapper can tell a
    milestone that did not pass from a report that could not be produced, and
    the whole of that design is delivered by that single unexecuted line --
    drop the `raise` and every one of them becomes 0.

    Three cases, because a process that always exited 0, or always 1, would
    satisfy any one of them alone.
    """
    script = _ROOT / "scripts" / "report_study.py"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")

    def status(out_dir):
        return subprocess.run(
            [sys.executable, str(script), "--out", str(out_dir)],
            capture_output=True, text=True, env=env, cwd=str(_ROOT),
        ).returncode

    assert status(_write(tmp_path / "passing", _study())) == (
        report_study.EXIT_OK) == 0
    failing = _study()
    _cell(failing, "cnn", 0)["filtering"]["criterion_4"][
        "latent_beats_embedding"] = False
    assert status(_write(tmp_path / "failing", failing)) == (
        report_study.EXIT_GATE_NOT_PASSED), (
        "a milestone that did not pass reporting 0 to the wrapper is the "
        "whole reason these statuses are numbered the way they are")
    assert status(tmp_path / "nothing_here") == report_study.EXIT_NO_RECORDS
    assert len({report_study.EXIT_OK, report_study.EXIT_GATE_NOT_PASSED,
                report_study.EXIT_NO_RECORDS}) == 3


def test_main_refuses_a_directory_with_no_records(tmp_path, capsys):
    status = report_study.main(["--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_NO_RECORDS
    assert f"--out {tmp_path} holds no result_*_seed*.json" in out
    assert "GATE:" not in out, (
        "no records is not a gate verdict; printing one would be a milestone "
        "reported from an empty directory")


def test_main_says_when_the_out_path_does_not_exist_at_all(tmp_path, capsys):
    """A mistyped --out and a study that has not run yet want opposite
    responses from the operator, and both used to print the same line."""
    missing = tmp_path / "typo"
    status = report_study.main(["--out", str(missing)])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_NO_RECORDS
    assert f"--out {missing} does not exist" in out


def test_a_record_whose_block_is_a_scalar_does_not_kill_the_report():
    """The `not isinstance(node, dict)` half of the printing guard, alone.

    `key not in node` RAISES `TypeError` on a scalar rather than answering, and
    a record whose whole `position` block came back as a `null` is exactly what
    a half-converted or hand-edited file holds. Losing the report to that after
    33 hours is the failure this guard exists for.
    """
    records = _rendering_study()
    _cell(records, "cnn", 1)["position"] = None
    _cell(records, "cnn", 2)["filtering"] = 7
    _cell(records, "frozen_ssl", 0)["reward"] = "no reward"
    # The whole report renders, and every table is asserted on its own: a
    # per-cell row prefix like "cnn/s2" appears in four of them, so a search
    # over the whole text would be answered by whichever table came first.
    report_study.report(records, evaluate_gate(records), "runs/x")
    verdict = evaluate_gate(records)
    assert _row(report_study.metric_table(records, verdict["per_arm"]),
                "cnn         position") == (
        "cnn         position    +1000.0000         n/a  +1003.0000"
        "  +1001.5000     2/3      False         n/a          n/a")
    assert _row(report_study.criterion_4_table(records), "cnn/s2") == (
        "cnn/s2                    n/a          n/a                      n/a")
    assert _row(report_study.gain_table(records), "cnn/s2") == (
        "cnn/s2                   n/a         n/a         n/a"
        "                 [n/a, n/a]        n/a          n/a          n/a"
        "       n/a      n/a")
    assert _row(report_study.reward_table(records), "frozen_ssl/s0") == (
        "frozen_ssl/s0              n/a             n/a          n/a      n/a"
        "      n/a                n/a")


def test_a_record_one_field_short_does_not_kill_the_report():
    """The `key not in node` half, alone: the block is a real dict and only the
    field is gone."""
    records = _rendering_study()
    del _cell(records, "cnn", 1)["position"]["gap_final"]
    verdict = evaluate_gate(records)
    assert _row(report_study.metric_table(records, verdict["per_arm"]),
                "cnn         position") == (
        "cnn         position    +1000.0000         n/a  +1003.0000"
        "  +1001.5000     2/3      False           0         3403")


def test_main_refuses_a_mislabelled_record_and_prints_the_remedy(
        tmp_path, capsys):
    _write(tmp_path, _study())
    (tmp_path / "result_cnn_seed0.json").write_text(
        json.dumps({**_record("random_vit", 2), "nonfinite": {}}))
    status = report_study.main(["--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_MISLABELLED
    assert "result_cnn_seed0.json" in out
    assert "arm='random_vit'" in out
    assert "GATE:" not in out


def test_main_refuses_a_record_it_cannot_read(tmp_path, capsys):
    _write(tmp_path, _study())
    (tmp_path / "result_frozen_ssl_seed2.json").write_text("{")
    status = report_study.main(["--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_UNREADABLE
    assert "result_frozen_ssl_seed2.json" in out
    assert "GATE:" not in out


def test_main_writes_the_figure_when_asked(tmp_path, capsys):
    figure = tmp_path / "figures" / "curves.png"
    status = report_study.main([
        "--out", str(_write(tmp_path, _study())), "--figure", str(figure)])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_OK
    assert figure.exists() and figure.stat().st_size > 0
    assert _row(out, "figure=") == f"figure={figure}"


def test_the_figure_skips_a_record_that_names_no_cell(tmp_path):
    """`record_cell` returns None for a record whose seed is a string, and
    indexing that None is a crash in the very last step of the pipeline. The
    record is not plotted; refusing the whole directory is `load_records`'s
    job, and by the time a figure is being drawn the report is already
    printed."""
    figure = tmp_path / "curves.png"
    # FIRST and last. `drawn` asks `any(...)`, which short-circuits on the
    # first arm it matches, so a malformed record at the END of the list is
    # never reached and the guard it is supposed to exercise is never run --
    # mutation R21 survived on exactly that. One at each end evaluates it
    # before any match and after every match.
    stray = _record("cnn", 0) | {"seed": "0"}
    records = [stray] + _study() + [stray]
    assert report_study.write_figure(records, figure) == f"figure={figure}"
    assert figure.exists()


def test_the_figure_draws_the_model_against_BOTH_baselines(tmp_path,
                                                           monkeypatch):
    """Spec criterion 2 asks for the curve of each arm with both baselines on
    the SAME axes, and nothing asserted anything about the figure's CONTENT --
    only that a non-empty PNG appeared.

    Dropping the persistence line leaves a picture of the model against the
    floor, which is the single most flattering way to draw a study that did
    not pass: an arm that never beat persistence looks like a success story,
    and the figure is the artefact most likely to end up in a write-up on its
    own, detached from the table that would have contradicted it.

    The spy reads the axes at `savefig`, so it sees what was actually about to
    be written rather than what the code says it drew.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    drawn = {}
    real_savefig = Figure.savefig

    def spy(self, *args, **kwargs):
        drawn["labels"] = [[line.get_label() for line in ax.get_lines()]
                           for ax in self.axes]
        drawn["titles"] = [ax.get_title() for ax in self.axes]
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    figure = tmp_path / "curves.png"
    assert report_study.write_figure(_study(), figure) == f"figure={figure}"
    assert drawn["titles"] == [f"{arm} (mean of 3 seed(s))" for arm in ARMS], (
        "one panel per arm, in ARMS order, so a label below belongs to the "
        "arm the panel is titled with")
    assert drawn["labels"] == [["persistence", "rssm", "floor"]] * len(ARMS), (
        "the model AND both baselines, on every panel: persistence is the "
        "line criterion 1 is measured against and floor is the ceiling on "
        "what any model could do")


def test_a_figure_with_no_arm_to_draw_is_a_message_and_not_a_blank_canvas(
        tmp_path):
    """`plt.subplots(1, 0, ...)` is a figure with no axes and it saves
    happily, so without the guard an empty directory produces a blank PNG and
    a `figure=` line that says one was drawn."""
    figure = tmp_path / "curves.png"
    assert report_study.write_figure([], figure) == (
        "figure NOT written: no arm has a record")
    assert not figure.exists()


def test_a_figure_that_cannot_be_drawn_does_not_cost_the_verdict(
        tmp_path, capsys):
    """The gate verdict is already printed above the figure line. A ragged
    curve set must cost the picture, not the number."""
    records = _study()
    _cell(records, "cnn", 1)["curves"]["rssm_position"] = [1.0, 2.0]
    status = report_study.main([
        "--out", str(_write(tmp_path, records)),
        "--figure", str(tmp_path / "curves.png")])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_GATE_NOT_PASSED, (
        "the incomplete curves fail curves_produced; the figure is not what "
        "decided the status")
    assert _row(out, "figure NOT written").startswith(
        "figure NOT written: 'rssm_position' curves have different lengths")
    assert _row(out, "GATE:") == "GATE: NOT PASSED"
    assert not (tmp_path / "curves.png").exists()


@pytest.mark.parametrize("first_missing,gaps,mangle", [
    # `write_figure` plots persistence, then the model, then the floor, so the
    # name in the FIGURE's message is the first of the three that is gone,
    # while the GATE's line names every curve the record is short of. Both are
    # carried beside each shape and asserted, so the three cases cannot
    # collapse into three spellings of one branch -- one curve gone is one
    # name in the gate line, and a `curves` block that is not a dict is all
    # six.
    ("rssm_position", ["rssm_position"],
     lambda record: record["curves"].pop("rssm_position")),
    ("persistence_position", list(CURVE_NAMES),
     lambda record: record.update(curves=None)),
    ("persistence_position", list(CURVE_NAMES),
     lambda record: record.update(curves=7)),
])
def test_a_record_with_no_curve_to_draw_costs_the_picture_not_the_status(
        tmp_path, capsys, first_missing, gaps, mangle):
    """The MISSING curve, which is a different failure from the ragged one.

    A record short of one of its six curves, or whose whole `curves` block
    came back as a null or a scalar, used to reach `len(None)` or `None.get`
    inside `mean_curve` -- `TypeError` and `AttributeError`, neither of which
    `write_figure` catches, because it catches `ValueError` and `OSError`. The
    figure is drawn AFTER the verdict is printed, so the process died with a
    traceback with the whole report already on the operator's screen and the
    exit status replaced by 1 -- the one status this pipeline's numbering
    reserves for "nobody caught this", and exactly what the gate's own FAIL on
    `curves_produced` was already telling them about.
    """
    records = _study()
    mangle(_cell(records, "cnn", 1))
    status = report_study.main([
        "--out", str(_write(tmp_path, records)),
        "--figure", str(tmp_path / "curves.png")])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_GATE_NOT_PASSED, (
        "not 1: a traceback and a gate that did not pass are the two things "
        "an unattended wrapper most needs to tell apart")
    assert _row(out, "figure NOT written") == (
        f"figure NOT written: no {first_missing!r} curve for [('cnn', 1)]")
    assert _row(out, "GATE:") == "GATE: NOT PASSED"
    assert _row(out, "  CURVES INCOMPLETE") == (
        f"  CURVES INCOMPLETE (1 of 9): cnn/s1 ({', '.join(gaps)})")
    assert not (tmp_path / "curves.png").exists()


def test_a_missing_matplotlib_costs_the_FIGURE_and_not_the_exit_status(
        tmp_path):
    """I3. The imports used to sit OUTSIDE the try that guards every other way
    of failing to draw, so the whole report printed correctly and the process
    then died with an uncaught `ImportError`: exit 1.

    1 is what this module's EXIT_* docstring reserves for "an uncaught
    traceback", and the statuses exist so an unattended wrapper can tell a
    milestone that did not pass from a report that could not be produced. A
    rented GPU box may well not have matplotlib, and matplotlib is the one
    dependency nothing else in the report needs -- so the gate's own verdict,
    already printed in full above, was replaced by the one status that says
    nobody caught this.

    Exercised with an UNIMPORTABLE matplotlib in a real subprocess rather than
    by patching `write_figure`: the defect is in which statements the `try`
    covers, and a mock of the function under test cannot see that.
    """
    stub = tmp_path / "no_matplotlib"
    stub.mkdir()
    (stub / "matplotlib.py").write_text(
        "raise ImportError(\"No module named 'matplotlib'\")\n")
    records = _study()
    _cell(records, "cnn", 0)["filtering"]["criterion_4"][
        "latent_beats_embedding"] = False
    figure = tmp_path / "curves.png"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    done = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "report_study.py"),
         "--out", str(_write(tmp_path / "study", records)),
         "--figure", str(figure)],
        capture_output=True, text=True, env=env, cwd=str(_ROOT),
    )
    assert "Traceback" not in done.stderr, done.stderr
    assert done.returncode == report_study.EXIT_GATE_NOT_PASSED, (
        "the gate's own status, not 1: the verdict was already on the "
        f"operator's screen.\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}")
    assert _row(done.stdout, "GATE:") == "GATE: NOT PASSED"
    assert _row(done.stdout, "figure NOT written") == (
        "figure NOT written: No module named 'matplotlib'"), (
        "reported the way every other figure failure already is")
    assert _row(done.stdout, "cnn         position").startswith(
        "cnn         position    +1000.0000"), (
        "and the whole report is still printed")
    assert not figure.exists()


def test_the_figure_is_drawn_AFTER_the_verdict_is_printed(tmp_path, capsys):
    """M7. A test's own message says the figure is drawn after the verdict so
    that a figure failure cannot cost the report; the ORDER itself was
    unpinned, and reversing the two statements in `main` left the suite green.

    Both renderings, because the figure line is the last line either way: the
    figure that could not be drawn is the case the ordering exists for, and
    the one that was drawn is the case a report printed second would still
    look right in.
    """
    ragged = _study()
    _cell(ragged, "cnn", 1)["curves"]["rssm_position"] = [1.0, 2.0]
    report_study.main(["--out", str(_write(tmp_path / "ragged", ragged)),
                       "--figure", str(tmp_path / "ragged.png")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "GATE: NOT PASSED"
    assert lines[-1].startswith("figure NOT written: "), (
        "a figure that could not be drawn must be reported UNDER a verdict "
        "the operator has already read, never in front of it")

    drawn = tmp_path / "drawn.png"
    report_study.main(["--out", str(_write(tmp_path / "good", _study())),
                       "--figure", str(drawn)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "GATE: PASSED"
    assert lines[-1] == f"figure={drawn}"


def test_every_line_main_prints_is_FLUSHED(tmp_path, capsys, monkeypatch):
    """M8. Every `print` in the script carries `flush=True` deliberately, so
    the report reaches a pipe or a log before anything else can happen to the
    process -- the figure is drawn after it, and drawing is the step most
    likely to die. None of the five was pinned.

    The spy records the keyword arguments each call was ACTUALLY made with,
    which is the only place `flush=True` is observable: it changes nothing a
    captured stream can be asked about afterwards.
    """
    calls = []
    real_print = builtins.print

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", spy)
    # All five call sites: the report, the figure line, the empty directory,
    # the mislabelled record and the unreadable one.
    report_study.main(["--out", str(_write(tmp_path / "study", _study())),
                       "--figure", str(tmp_path / "curves.png")])
    report_study.main(["--out", str(tmp_path / "empty_directory")])
    mislabelled = _write(tmp_path / "mislabelled", _study())
    (mislabelled / "result_cnn_seed0.json").write_text(
        json.dumps({**_record("random_vit", 2), "nonfinite": {}}))
    report_study.main(["--out", str(mislabelled)])
    unreadable = _write(tmp_path / "unreadable", _study())
    (unreadable / "result_frozen_ssl_seed2.json").write_text("{")
    report_study.main(["--out", str(unreadable)])
    assert len(calls) == 5, (
        "five prints, one per call site; a site that stopped printing would "
        f"otherwise leave the survivors to answer this test:\n{calls}")
    assert calls == [{"flush": True}] * 5, (
        "every one of them flushed: the report has to be on the pipe before "
        "the figure is attempted")


def test_main_does_not_write_a_figure_unless_asked(tmp_path, capsys):
    report_study.main(["--out", str(_write(tmp_path, _study()))])
    assert "figure" not in capsys.readouterr().out
    assert list(tmp_path.glob("*.png")) == []
