"""`mbfps.eval.ladder`: the pure rules of M3f, one mutation at a time.

`primary_rung` is spec 2.2; `anchor_delta` is spec 2.4; `reading_timing` is
spec 3.3 in the table's precedence. Every rule is exercised by an input that
only that rule can decide, so a rule dropped or reordered fails a named test.
"""

import math

import pytest

from mbfps.eval.ladder import (
    CURVE_WINDOW,
    DECISION_H,
    FAMILY,
    OBJECTIVE_BATCHES,
    R2_SENSITIVITY,
    REFERENCE_RUNG,
    REPORTED_H,
    RUNGS,
    SEEDS_REQUIRED,
    STEPS,
    ArmInputs,
    Status,
    TimingInputs,
    anchor_delta,
    format_reading_timing,
    gate_passes,
    primary_rung,
    reading_timing,
)
from mbfps.eval.split_gap import StratumContrast, train_held_passes_gate

Z_FAM = 2.89
"""`cluster_threshold(6, 24)` on the shipped split, to two decimals; the
tests only need a bar that 3.0 clears and 2.0 does not."""


def test_the_pre_registered_constants_are_the_specs():
    assert RUNGS == (1000, 2000, 3000, 4000, 5000)
    assert STEPS == 5000 and STEPS >= max(RUNGS)
    assert REFERENCE_RUNG == 20000
    assert DECISION_H == 15 and REPORTED_H == (5, 15, 45)
    assert FAMILY == 6 and SEEDS_REQUIRED == 2
    assert R2_SENSITIVITY == 0.1 and CURVE_WINDOW == 100 and OBJECTIVE_BATCHES == 50
    assert gate_passes is train_held_passes_gate


# ---------------------------------------------------------------------------
# primary_rung
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_step, rung", [
    (2045, 2000), (3555, 4000), (4510, 5000),        # frozen_ssl s0 / s1 / s2
    (1610, 2000), (4535, 5000), (3275, 3000),        # random_vit s0 / s1 / s2
    (8359, 5000), (19096, 5000), (16873, 5000),      # pixel_ae, beyond the ladder
])
def test_primary_rung_on_the_shipped_minima_is_the_specs_table(min_step, rung):
    assert primary_rung(min_step) == rung


def test_primary_rung_ties_go_to_the_earlier_rung_and_the_ends_clamp():
    assert primary_rung(1500) == 1000
    assert primary_rung(2500) == 2000
    assert primary_rung(1) == 1000
    assert primary_rung(10**9) == 5000
    assert primary_rung(3, rungs=(2, 4)) == 2
    assert primary_rung(5, rungs=(4, 2)) == 4, "the rungs need not be sorted"


def test_primary_rung_refuses_no_rungs_or_a_step_before_the_first_step():
    with pytest.raises(ValueError, match="rung"):
        primary_rung(100, rungs=())
    with pytest.raises(ValueError, match="1-based"):
        primary_rung(0)


# ---------------------------------------------------------------------------
# anchor_delta
# ---------------------------------------------------------------------------


def test_anchor_delta_is_zero_with_no_step_when_the_prefixes_are_identical():
    assert anchor_delta([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 9.0], 3) == (0.0, None)


def test_anchor_delta_is_the_largest_difference_and_names_the_first_differing_step():
    delta, step = anchor_delta([1.0, 2.5, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], 4)
    assert delta == 1.0 and step == 2
    assert anchor_delta([1.0, 2.5, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], 1) == (0.0, None)


def test_anchor_delta_treats_a_nan_as_a_difference_and_reports_it_as_infinite():
    delta, step = anchor_delta([1.0, float("nan")], [1.0, 2.0], 2)
    assert math.isinf(delta) and step == 2
    delta, step = anchor_delta([1.0, float("nan")], [1.0, float("nan")], 2)
    assert math.isinf(delta) and step == 2, "NaN is not equal to NaN; the anchor is exact equality"


def test_anchor_delta_refuses_a_history_shorter_than_the_steps_it_is_asked_about():
    with pytest.raises(ValueError, match="shorter"):
        anchor_delta([1.0, 2.0], [1.0, 2.0, 3.0], 3)
    with pytest.raises(ValueError, match="shorter"):
        anchor_delta([1.0, 2.0, 3.0], [1.0, 2.0], 3)
    with pytest.raises(ValueError, match="steps"):
        anchor_delta([1.0], [1.0], 0)


# ---------------------------------------------------------------------------
# Reading T
# ---------------------------------------------------------------------------


def _contrast(z: float, clusters: int = 24) -> StratumContrast:
    return StratumContrast(estimate=0.01 * z, se=0.01, z=z, clusters=clusters)


def _leaf(seed: int, free_z: float, probe_z: float = 0.0, rung: int = 2000) -> ArmInputs:
    return ArmInputs(
        t_free=_contrast(free_z), t_probe=_contrast(probe_z),
        primary_rungs={seed: rung}, per_seed=None,
    )


def _arm(free_z: float, probe_z: float = 0.0, seed_free=(4.0, 4.0, 4.0)) -> ArmInputs:
    leaves = {s: _leaf(s, z) for s, z in enumerate(seed_free)}
    return ArmInputs(
        t_free=_contrast(free_z), t_probe=_contrast(probe_z),
        primary_rungs={s: 2000 for s in leaves}, per_seed=leaves,
    )


def _inputs(**arms) -> TimingInputs:
    return TimingInputs(arms=arms, z_fam=Z_FAM, h=DECISION_H)


def _read(arm: ArmInputs):
    return reading_timing(_inputs(a=arm)).arms["a"]


def test_opposite_sign_clears_are_unresolved_and_nothing_below_is_read():
    r = _read(_arm(free_z=4.0, probe_z=-4.0))
    assert r.status is Status.UNRESOLVED_PROBE
    assert "opposite signs" in r.rule
    assert _read(_arm(free_z=-4.0, probe_z=4.0)).status is Status.UNRESOLVED_PROBE
    # Same-sign clears are NOT unresolved.
    assert _read(_arm(free_z=4.0, probe_z=4.0)).status is Status.EARLIER_BETTER


def test_a_positive_free_clear_replicated_in_two_seeds_is_earlier_better():
    r = _read(_arm(free_z=4.0, seed_free=(4.0, 4.0, 0.5)))
    assert r.status is Status.EARLIER_BETTER
    assert (r.seeds_up, r.seeds_total) == (2, 3)
    assert "2 of 3 seeds" in r.rule


def test_a_negative_free_clear_replicated_in_two_seeds_is_earlier_worse():
    r = _read(_arm(free_z=-4.0, seed_free=(-4.0, 0.5, -4.0)))
    assert r.status is Status.EARLIER_WORSE
    assert (r.seeds_down, r.seeds_total) == (2, 3)
    assert "2 of 3 seeds" in r.rule


def test_a_pooled_clear_in_one_seed_only_is_no_difference_and_the_sentence_says_so():
    r = _read(_arm(free_z=4.0, seed_free=(4.0, 0.5, 0.5)))
    assert r.status is Status.NO_DIFFERENCE
    assert "only 1 of 3 seeds" in r.rule and f">= {SEEDS_REQUIRED} required" in r.rule
    r = _read(_arm(free_z=-4.0, seed_free=(-4.0, 0.5, 0.5)))
    assert r.status is Status.NO_DIFFERENCE and "only 1 of 3 seeds" in r.rule


def test_an_interval_covering_zero_is_no_difference():
    r = _read(_arm(free_z=1.0, seed_free=(4.0, 4.0, 4.0)))
    assert r.status is Status.NO_DIFFERENCE
    assert "does not clear" in r.rule
    assert r.seeds_up == 3, "the seeds still counted; the pooled bar decided"


def test_the_bar_is_strict_and_a_nan_or_infinite_z_never_clears():
    assert _read(_arm(free_z=Z_FAM)).status is Status.NO_DIFFERENCE
    assert _read(_arm(free_z=3.0)).status is Status.EARLIER_BETTER
    assert _read(_arm(free_z=float("nan"))).status is Status.NO_DIFFERENCE
    assert _read(_arm(free_z=float("inf"))).status is Status.NO_DIFFERENCE
    assert _read(_arm(free_z=4.0, probe_z=float("-inf"))).status is Status.EARLIER_BETTER, \
        "an infinite probe z is not a clear, so it cannot make the arm unresolved"


def test_a_single_seed_leaf_has_a_vacuous_replication_clause():
    """The M3e correction, built in: a leaf read alone (per_seed=None) is not
    held to SEEDS_REQUIRED -- there is nothing to replicate across."""
    up = _read(_leaf(0, free_z=4.0))
    assert up.status is Status.EARLIER_BETTER and (up.seeds_up, up.seeds_total) == (1, 1)
    assert "this seed alone" in up.rule
    down = _read(_leaf(0, free_z=-4.0))
    assert down.status is Status.EARLIER_WORSE and (down.seeds_down, down.seeds_total) == (1, 1)
    flat = _read(_leaf(0, free_z=1.0))
    assert flat.status is Status.NO_DIFFERENCE and (flat.seeds_up, flat.seeds_total) == (0, 1)


def test_an_empty_per_seed_dict_is_zero_of_zero_and_never_replicates():
    arm = ArmInputs(t_free=_contrast(4.0), t_probe=_contrast(0.0), primary_rungs={}, per_seed={})
    r = _read(arm)
    assert r.status is Status.NO_DIFFERENCE and (r.seeds_up, r.seeds_total) == (0, 0)
    assert "only 0 of 0 seeds" in r.rule


def test_arms_are_read_independently_and_in_the_callers_order():
    reading = reading_timing(_inputs(
        random_vit=_arm(4.0), pixel_ae=_arm(-4.0, seed_free=(-4.0, -4.0, -4.0)), frozen_ssl=_arm(0.5),
    ))
    assert list(reading.arms) == ["random_vit", "pixel_ae", "frozen_ssl"]
    assert [r.status for r in reading.arms.values()] == [
        Status.EARLIER_BETTER, Status.EARLIER_WORSE, Status.NO_DIFFERENCE,
    ]
    assert reading.h == DECISION_H and reading.z_fam == Z_FAM


def test_format_prints_the_contrasts_the_primary_rungs_and_a_verdict_per_arm():
    inputs = _inputs(frozen_ssl=_arm(4.0), random_vit=_arm(0.5))
    text = format_reading_timing(reading_timing(inputs), inputs)
    assert f"Reading T" in text and f"h={DECISION_H}" in text and f"z_fam = {Z_FAM:.2f}" in text
    assert "frozen_ssl" in text and "EARLIER BETTER" in text
    assert "random_vit" in text and "NO DIFFERENCE" in text
    assert "s0 2000" in text, "the primary rung per seed is printed beside the verdict"
    assert "decided by:" in text
