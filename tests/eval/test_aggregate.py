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

import importlib.util
import json
import math
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
"""The two ridge decades, DIFFERENT from each other on purpose.

They are outside the distinctness table because every cell of a coherent study
holds the same pair -- that is what `gain_ridges_agree` means -- so they cannot
vary per cell. What they must not do is equal each other: `joint_ridge` and
`embedding_ridge` sit in adjacent columns of the filtering table, and a
rendering that printed one under the other's heading would be invisible if they
agreed. Both are real values from `probe.RIDGES`.
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
    """One `metric_summary`-shaped block. Counts are 0: the fixture is a study
    that passes, and every non-zero count is set by the test that needs it."""
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
        "steps_floor_above_persistence": 0,
        "steps_degenerate": 0,
        "gap_finite": FIXTURE_HORIZON,
        "gap_mean": _value(f"{metric}.gap_mean", arm, seed),
        "gap_min": 0.1,
        "gap_max": 0.9,
        "gap_final": _value(f"{metric}.gap_final", arm, seed),
        "n_steps": FIXTURE_HORIZON,
    }


def _record(arm, seed, *, latent_wins=True, reward_degenerate=True,
            ridge_selected=True):
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
                "joint_ridge": JOINT_RIDGE,
                "embedding_ridge": EMBEDDING_RIDGE,
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
    assert len(values) == 432, (
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
    assert JOINT_RIDGE != EMBEDDING_RIDGE
    assert f"{JOINT_RIDGE:.1e}" != f"{EMBEDDING_RIDGE:.1e}"

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


def test_per_arm_flags_seeds_that_selected_different_ridges():
    """R1, the reason it is a requirement: the scored R^2 moves ~0.10 per
    decade while the gain is ~0.02, so the gain's SIGN moves with the selected
    decade -- measured -0.0706, -0.0208 and +0.0325 for one model. A mean over
    seeds that selected different decades is a mean over different estimators
    and nothing in the number itself says so."""
    records = [_record("cnn", seed) for seed in SEEDS]
    _cell(records, "cnn", 2)["filtering"]["gain"]["joint_ridge"] = 1e7
    summary = per_arm(records)["cnn"]
    assert summary["gain_ridges_agree"] is False
    assert summary["gain_ridges"][2]["joint_ridge"] == 1e7
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
            "joint_ridge"] = 1e7
    groups = ridge_groups(records)
    assert len(groups) == 2
    assert [g["n_cells"] for g in groups] == [6, 3]
    assert [g["joint_ridge"] for g in groups] == [JOINT_RIDGE, 1e7]
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
        "  +1001.3333     3/3       True           0            0")
    assert _row(table, "cnn         angle") == (
        "cnn         angle       +1300.0000  +1301.0000  +1303.0000"
        "  +1301.3333     3/3       True           0            0")
    assert _row(table, "random_vit  position") == (
        "random_vit  position    +1010.0000  +1011.0000  +1013.0000"
        "  +1011.3333     3/3       True           0            0")


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
        "  +1001.5000     2/3       True           0            0")


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
        "  +1001.5000     2/3      False           0            0")


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


def test_the_ridge_block_lists_the_decade_every_gain_was_computed_at():
    block = report_study.ridge_block(ridge_groups(_rendering_study()))
    assert _row(block, "selected") == (
        "selected    joint_ridge  embed_ridge  cells  gain_mean  members")
    assert _row(block, "True") == (
        "True            1.0e+03      1.0e+05      9 +2306.3333  "
        "cnn/s0 cnn/s1 cnn/s2 frozen_ssl/s0 frozen_ssl/s1 frozen_ssl/s2 "
        "random_vit/s0 random_vit/s1 random_vit/s2")
    assert "WARNING" not in block, (
        "nine cells that agree must not be warned about, or the warning "
        "means nothing when they do not")


def test_the_ridge_block_warns_when_the_cells_did_not_agree():
    """R1. There is no correction for this, only the disclosure."""
    records = _rendering_study()
    for seed in SEEDS:
        _cell(records, "cnn", seed)["filtering"]["gain"]["joint_ridge"] = 1e7
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


def test_main_reports_an_incomplete_study_rather_than_aggregating_it(
        tmp_path, capsys):
    records = [r for r in _study() if (r["arm"], r["seed"]) != ("cnn", 2)]
    status = report_study.main(["--out", str(_write(tmp_path, records))])
    out = capsys.readouterr().out
    assert status == report_study.EXIT_GATE_NOT_PASSED
    assert _row(out, "  MISSING CELLS") == "  MISSING CELLS (1 of 9): cnn/s2"
    assert _row(out, "  records") == (
        "  records     8 of 9 expected cells (3 arms x 3 seeds)")


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
        "  +1001.5000     2/3      False           0            0")


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


def test_main_does_not_write_a_figure_unless_asked(tmp_path, capsys):
    report_study.main(["--out", str(_write(tmp_path, _study()))])
    assert "figure" not in capsys.readouterr().out
    assert list(tmp_path.glob("*.png")) == []
