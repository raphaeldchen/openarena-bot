"""mbfps.eval.sharper, the pure half of M3h: Reading N (is the rollout
noise-limited, and at which temperature) and, from Task 5, Reading M (does
training it sharper help).

Every contrast is fabricated and every expected status is typed by hand, so a
rule that changes has to break a named test rather than agree with itself. The
contrast type is `split_gap.StratumContrast`, the same reduction of a pooled
contrast Readings G, T and S are decided on.
"""

import numpy as np

from mbfps.eval.split_gap import StratumContrast
from mbfps.eval.sharper import (
    ARMS_REQUIRED,
    DECISION_H,
    IDENTITY_STEPS,
    REFERENCE_TAU,
    REPORTED_H,
    SEEDS_REQUIRED,
    STEPS,
    SWEEP_FAMILY,
    RETRAIN_FAMILY,
    TAU_CANDIDATES,
    TAU_GRID,
    ArmTau,
    SweepArm,
    SweepInputs,
    SweepReading,
    SweepStatus,
    TauInputs,
    format_reading_noise,
    reading_noise,
)

Z_FAM = 3.06


def c(z: float, estimate: float | None = None) -> StratumContrast:
    if estimate is None:
        estimate = float("nan") if not np.isfinite(z) else 0.01 * z
    return StratumContrast(estimate=estimate, se=0.01, z=z, clusters=24)


def tau_inputs(pooled: float, seeds=None) -> TauInputs:
    """One (arm, tau): the pooled contrast and its three per-seed leaves,
    which default to the pooled z so a pooled clear replicates by default."""
    seeds = (pooled, pooled, pooled) if seeds is None else seeds
    return TauInputs(free=c(pooled), per_seed={i: c(z) for i, z in enumerate(seeds)})


def arm(**by_tau) -> SweepArm:
    """`arm(t07=..., t05=..., t03=..., t00=...)` -- keyword spellings of the
    grid, so a test reads as the table does."""
    spelling = {"t07": 0.7, "t05": 0.5, "t03": 0.3, "t00": 0.0}
    taus = {spelling[k]: (v if isinstance(v, TauInputs) else tau_inputs(v)) for k, v in by_tau.items()}
    for tau in TAU_CANDIDATES:
        taus.setdefault(tau, tau_inputs(0.0))
    taus.setdefault(0.0, tau_inputs(0.0))
    return SweepArm(taus=taus)


def read(**arms) -> SweepReading:
    return reading_noise(SweepInputs(arms=arms, z_fam=Z_FAM, h=15))


def test_the_pre_registered_constants_are_the_specs():
    assert TAU_GRID == (1.0, 0.7, 0.5, 0.3, 0.0)
    assert TAU_CANDIDATES == (0.7, 0.5, 0.3)
    from mbfps.models.rssm import SAMPLE_TEMPERATURE

    assert REFERENCE_TAU == SAMPLE_TEMPERATURE == 1.0 and REFERENCE_TAU not in TAU_CANDIDATES
    assert 0.0 not in TAU_CANDIDATES, "argmax is an endpoint, never a candidate"
    assert set(TAU_CANDIDATES) | {REFERENCE_TAU, 0.0} == set(TAU_GRID)
    assert SWEEP_FAMILY == len(TAU_CANDIDATES) * 3 == 9
    assert RETRAIN_FAMILY == 6 and SEEDS_REQUIRED == 2 and ARMS_REQUIRED == 2
    assert DECISION_H == 15 and REPORTED_H == (5, 15, 45)
    assert STEPS == 20000 and IDENTITY_STEPS == 500


def test_two_arms_clearing_at_one_tau_read_noise_limited_at_that_tau():
    reading = read(pixel_ae=arm(t05=5.0), frozen_ssl=arm(t05=4.0), random_vit=arm())
    assert isinstance(reading, SweepReading)
    assert reading.status is SweepStatus.NOISE_LIMITED
    assert reading.tau_star == 0.5
    assert reading.arms_clearing[0.5] == ("pixel_ae", "frozen_ssl")
    assert "tau=0.5" in reading.rule and "2 of 3 arms" in reading.rule
    cell = reading.cells[("pixel_ae", 0.5)]
    assert isinstance(cell, ArmTau) and cell.clears_up and cell.seeds_up == 3


def test_one_arm_alone_is_not_enough_and_the_sentence_names_it():
    reading = read(pixel_ae=arm(t05=9.0), frozen_ssl=arm(), random_vit=arm())
    assert reading.status is SweepStatus.NOT_NOISE_LIMITED
    assert reading.tau_star is None
    assert "pixel_ae" in reading.rule and "1 of 3 arms" in reading.rule


def test_a_pooled_clear_that_replicates_in_one_seed_does_not_count_for_its_arm():
    thin = tau_inputs(5.0, seeds=(5.0, 1.0, 1.0))
    reading = read(pixel_ae=arm(t05=thin), frozen_ssl=arm(t05=5.0), random_vit=arm())
    assert reading.status is SweepStatus.NOT_NOISE_LIMITED
    assert reading.cells[("pixel_ae", 0.5)].seeds_up == 1
    assert not reading.cells[("pixel_ae", 0.5)].counts
    assert reading.cells[("frozen_ssl", 0.5)].counts


def test_the_largest_pooled_contrast_among_the_clearing_taus_wins():
    small, large = tau_inputs(4.0), tau_inputs(9.0)
    assert small.free.z == 4.0 and large.free.z == 9.0
    reading = read(
        pixel_ae=arm(t07=small, t03=large),
        frozen_ssl=arm(t07=small, t03=large),
        random_vit=arm(),
    )
    assert reading.status is SweepStatus.NOISE_LIMITED
    assert reading.tau_star == 0.3, "0.3 pools the larger contrast"


def test_a_tie_between_two_clearing_taus_goes_to_the_larger_temperature():
    """The milder intervention, pre-registered: an equal estimate at 0.7 and
    0.3 retrains at 0.7."""
    tied = {"t07": tau_inputs(5.0), "t03": tau_inputs(5.0)}
    reading = read(pixel_ae=arm(**tied), frozen_ssl=arm(**tied), random_vit=arm())
    assert reading.status is SweepStatus.NOISE_LIMITED and reading.tau_star == 0.7


def test_two_arms_clearing_negatively_read_sharper_worse():
    reading = read(pixel_ae=arm(t03=-6.0), frozen_ssl=arm(t03=-5.0), random_vit=arm())
    assert reading.status is SweepStatus.SHARPER_WORSE
    assert reading.tau_star is None
    assert "tau=0.3" in reading.rule and "-" in reading.rule


def test_a_positive_clear_in_two_arms_outranks_a_negative_clear_elsewhere():
    """Precedence: the sweep exists to find a helpful temperature, and a
    different temperature hurting does not withdraw one that helps."""
    reading = read(
        pixel_ae=arm(t07=5.0, t03=-6.0), frozen_ssl=arm(t07=5.0, t03=-6.0), random_vit=arm()
    )
    assert reading.status is SweepStatus.NOISE_LIMITED and reading.tau_star == 0.7


def test_the_endpoint_never_decides_anything():
    """tau = 0 may be in the inputs and is reported; it cannot make the status
    NOISE_LIMITED and cannot be tau_star."""
    reading = read(pixel_ae=arm(t00=9.0), frozen_ssl=arm(t00=9.0), random_vit=arm(t00=9.0))
    assert reading.status is SweepStatus.NOT_NOISE_LIMITED and reading.tau_star is None
    assert reading.cells[("pixel_ae", 0.0)].clears_up, "still reported"


def test_z_exactly_at_the_bar_does_not_clear_and_a_nan_never_does():
    at_bar = read(pixel_ae=arm(t05=Z_FAM), frozen_ssl=arm(t05=Z_FAM), random_vit=arm())
    assert at_bar.status is SweepStatus.NOT_NOISE_LIMITED
    nan = read(
        pixel_ae=arm(t05=tau_inputs(float("nan"))),
        frozen_ssl=arm(t05=tau_inputs(float("nan"))),
        random_vit=arm(),
    )
    assert nan.status is SweepStatus.NOT_NOISE_LIMITED
    infinite = read(pixel_ae=arm(t05=float("inf")), frozen_ssl=arm(t05=float("inf")), random_vit=arm())
    assert infinite.status is SweepStatus.NOT_NOISE_LIMITED


def test_format_reading_noise_prints_every_cell_and_the_verdict():
    inputs = SweepInputs(
        arms={"pixel_ae": arm(t05=5.0), "frozen_ssl": arm(t05=4.0), "random_vit": arm()},
        z_fam=Z_FAM, h=15,
    )
    text = format_reading_noise(reading_noise(inputs), inputs)
    assert text.startswith("--- Reading N: is the rollout noise-limited at h=15")
    assert "z_fam = 3.06" in text
    for tau in TAU_GRID[1:]:
        assert f"{tau:.1f}" in text
    assert "verdict: NOISE LIMITED" in text and "tau=0.5" in text
    assert text.endswith("\n")


def test_the_table_carries_the_down_tally_the_sharper_worse_verdict_is_read_from():
    """Under SHARPER_WORSE the verdict sentence is a statement about the DOWN
    seeds ("tau=0.3 clears -3.06 in 3 of 3 arms"), so the table has to print
    that tally. Carrying only the up one made every row of that verdict read
    `0/3 ... no`, which reads as "it replicated in no seed" -- the opposite.

    The shape here is M3h's own reading: all three arms clear downward at
    tau = 0.3, replicated in 2 of 3 seeds for pixel_ae and 3 of 3 elsewhere.
    """
    inputs = SweepInputs(
        arms={
            "pixel_ae": arm(t03=tau_inputs(-5.0, seeds=(-5.0, -5.0, 1.0))),
            "frozen_ssl": arm(t03=-5.0),
            "random_vit": arm(t03=-5.0),
        },
        z_fam=Z_FAM, h=15,
    )
    reading = reading_noise(inputs)
    assert reading.status is SweepStatus.SHARPER_WORSE
    assert "in 3 of 3 arms" in reading.rule
    text = format_reading_noise(reading, inputs)

    header = text.splitlines()[1]
    assert header.split() == [
        "arm", "tau", "estimate", "se", "z", "seeds", "up", "seeds", "dn", "clears", "counts", "up"
    ]
    rows = {
        (parts[0], parts[1]): parts
        for parts in (line.split() for line in text.splitlines()[2:])
        if len(parts) >= 9 and parts[0] in inputs.arms
    }
    # arm, tau, estimate, se, z, seeds up, seeds dn, clears, counts up
    assert rows[("pixel_ae", "0.3")][:9] == [
        "pixel_ae", "0.3", "-0.0500", "0.0100", "-5.00", "0/3", "2/3", "down", "no"
    ]
    assert rows[("frozen_ssl", "0.3")][5:9] == ["0/3", "3/3", "down", "no"]
    assert rows[("random_vit", "0.3")][5:9] == ["0/3", "3/3", "down", "no"]
    # A temperature that moved neither way carries both tallies at zero.
    assert rows[("pixel_ae", "0.7")][5:9] == ["0/3", "0/3", "no", "no"]
    # The down tally the verdict counted is the one the table prints: three
    # arms with at least SEEDS_REQUIRED down seeds.
    down = sum(1 for a in inputs.arms if int(rows[(a, "0.3")][6].split("/")[0]) >= SEEDS_REQUIRED)
    assert down == 3 and reading.arms_against[0.3] == ("pixel_ae", "frozen_ssl", "random_vit")


# ---------------------------------------------------------------------------
# Reading M (spec 3.3): does a model TRAINED at tau* roll out better than the
# one it replaces? The same four-status shape as M3f's Reading T, with the
# probe channel as the control and the single-seed clause built in.
# ---------------------------------------------------------------------------

from mbfps.eval.sharper import (  # noqa: E402
    RetrainArm,
    RetrainArmReading,
    RetrainInputs,
    RetrainReading,
    RetrainStatus,
    format_reading_sharper,
    reading_sharper,
)

M_Z_FAM = 2.89


def retrain_arm(free=5.0, probe=5.0, seeds=None) -> RetrainArm:
    seeds = (free, free, free) if seeds is None else seeds
    return RetrainArm(
        free=c(free), probe=c(probe),
        per_seed={i: RetrainArm(free=c(z), probe=c(probe), per_seed=None) for i, z in enumerate(seeds)},
    )


def read_m(**arms) -> RetrainReading:
    return reading_sharper(RetrainInputs(arms=arms, z_fam=M_Z_FAM, h=15, tau=0.5))


def test_a_positive_clear_replicated_in_two_seeds_reads_sharper_better():
    reading = read_m(pixel_ae=retrain_arm(free=5.0, seeds=(5.0, 5.0, 1.0)))
    r = reading.arms["pixel_ae"]
    assert isinstance(r, RetrainArmReading) and r.status is RetrainStatus.SHARPER_BETTER
    assert r.seeds_up == 2 and r.seeds_total == 3
    assert "2 of 3 seeds" in r.rule and "+5.00" in r.rule
    assert reading.tau == 0.5 and reading.h == 15


def test_a_negative_clear_replicated_reads_sharper_worse():
    r = read_m(a=retrain_arm(free=-5.0, probe=-5.0)).arms["a"]
    assert r.status is RetrainStatus.SHARPER_WORSE
    assert r.rule.startswith("M_free z -5.00 < -2.89 pooled and in 3 of 3 seeds")


def test_channels_clearing_with_opposite_signs_are_unresolved_through_the_probe():
    r = read_m(a=retrain_arm(free=5.0, probe=-5.0)).arms["a"]
    assert r.status is RetrainStatus.UNRESOLVED_PROBE
    assert "opposite signs" in r.rule


def test_a_pooled_clear_that_does_not_replicate_reads_no_difference_and_says_why():
    r = read_m(a=retrain_arm(free=5.0, seeds=(5.0, 1.0, 1.0))).arms["a"]
    assert r.status is RetrainStatus.NO_DIFFERENCE
    assert "only 1 of 3 seeds" in r.rule and f">= {SEEDS_REQUIRED}" in r.rule


def test_no_clear_at_all_reads_no_difference():
    r = read_m(a=retrain_arm(free=1.0, probe=1.0)).arms["a"]
    assert r.status is RetrainStatus.NO_DIFFERENCE
    assert r.rule == "M_free z +1.00 does not clear +-2.89"


def test_a_single_seed_leaf_read_alone_has_a_vacuous_replication_clause():
    leaf = RetrainArm(free=c(5.0), probe=c(5.0), per_seed=None)
    r = read_m(a=leaf).arms["a"]
    assert r.status is RetrainStatus.SHARPER_BETTER
    assert "this seed alone" in r.rule and r.seeds_total == 1


def test_z_exactly_at_the_bar_does_not_clear_and_a_nan_never_does_on_the_retrain():
    # Named apart from Reading N's test of the same rule above: two
    # module-level functions of one name shadow each other, and the first
    # silently stops running.

    assert read_m(a=retrain_arm(free=M_Z_FAM)).arms["a"].status is RetrainStatus.NO_DIFFERENCE
    assert read_m(a=retrain_arm(free=float("nan"))).arms["a"].status is RetrainStatus.NO_DIFFERENCE
    assert read_m(a=retrain_arm(free=float("inf"))).arms["a"].status is RetrainStatus.NO_DIFFERENCE


def test_arms_are_read_independently_and_in_the_callers_order():
    reading = read_m(random_vit=retrain_arm(free=-5.0, probe=-5.0), pixel_ae=retrain_arm())
    assert list(reading.arms) == ["random_vit", "pixel_ae"]
    assert reading.arms["random_vit"].status is RetrainStatus.SHARPER_WORSE
    assert reading.arms["pixel_ae"].status is RetrainStatus.SHARPER_BETTER


def test_format_reading_sharper_prints_both_channels_and_the_verdicts():
    inputs = RetrainInputs(
        arms={"pixel_ae": retrain_arm(free=5.0), "frozen_ssl": retrain_arm(free=1.0, probe=1.0)},
        z_fam=M_Z_FAM, h=15, tau=0.5,
    )
    text = format_reading_sharper(reading_sharper(inputs), inputs)
    assert text.startswith("--- Reading M: the retrain at tau=0.5 against the M3c cells at h=15")
    assert "z_fam = 2.89" in text
    assert text.count("free") >= 2 and text.count("probe") >= 2
    assert "verdict: pixel_ae    SHARPER BETTER" in text
    assert "verdict: frozen_ssl  NO DIFFERENCE" in text
    assert text.endswith("\n")
