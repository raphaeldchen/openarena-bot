"""The pre-registered rules of Reading 1 and Reading 2, on fabricated pooled inputs.

Every test here builds its inputs by hand and states the answer by hand;
nothing reads an expectation from the function under test. The baseline
fixture is the slow-drift story told exactly as spec 3.2 would find it
supported -- `random_vit` best on Δ(45) against each other arm and least
moving under both ratio channels, the treatment's cosine contrast clearing
the bar, its held-out corrected error smaller on both folds, no α on a grid
boundary, the h× control silent -- and every test after it mutates ONE rule's
input and states what the rule must then say. The bar `z_fam` is a hand-chosen
3.0: the rules read whatever bar they are given (`cluster_threshold(8, 24)` is
3.01 on the shipped split, and that is Task 6's to compute).
"""

import math

import numpy as np
import pytest

from mbfps.eval.trust_readings import (
    ARMS_ORDER,
    CONTROL,
    FAMILY,
    Q_PREREGISTERED,
    Q_REPORTED,
    TREATMENT,
    ConditionResult,
    Contrast,
    Ratio,
    ReadingOne,
    ReadingOneInputs,
    ReadingTwo,
    Status,
    best_delta_arm,
    condition_i,
    condition_ii,
    condition_iii,
    format_reading_one,
    format_reading_two,
    least_moving_arm,
    probe_control,
    reading_one,
    reading_two,
)
from mbfps.utils.config import ARMS

Z_FAM = 3.0
NAN = float("nan")
HORIZON = 45


def _c(z: float, n: int = 229) -> Contrast:
    """A contrast whose z is `z`; se fixed at 0.1 so the estimate is 0.1 z."""
    return Contrast(estimate=0.1 * z, se=0.1, z=z, n_windows=n)


def _r(estimate: float) -> Ratio:
    return Ratio(estimate=estimate, low=estimate - 0.1, high=estimate + 0.1)


def _leaf(**overrides) -> ReadingOneInputs:
    """One seed's (or the pooled) inputs: the slow-drift story, supported.

    Keys are oriented `(a, b)` with `a` earlier in ARMS_ORDER, so `random_vit`
    -- the best-Δ arm -- is always the SUBTRACTED arm: its two contrasts read
    -4 and -5 here and must be flipped to +4 and +5 to be read as
    `random_vit - other`."""
    fields = dict(
        delta_contrast={
            ("pixel_ae", "frozen_ssl"): _c(-1.0),
            ("pixel_ae", "random_vit"): _c(-4.0),
            ("frozen_ssl", "random_vit"): _c(-5.0),
        },
        ratio_probe={"pixel_ae": _r(0.9), "frozen_ssl": _r(1.3), "random_vit": _r(0.4)},
        ratio_free={"pixel_ae": _r(0.8), "frozen_ssl": _r(1.2), "random_vit": _r(0.3)},
        cosine_contrast=_c(4.0),
        corrected_contrast_a=_c(-4.0),
        corrected_contrast_b=_c(-3.5),
        alpha_boundary={"pixel_ae": False, "frozen_ssl": False, "random_vit": False},
        crossing_contrast_probe=_c(1.0),
        crossing_contrast_free=_c(0.5),
        per_seed=None,
    )
    fields.update(overrides)
    return ReadingOneInputs(**fields)


def _inputs(pooled: dict | None = None, seeds: dict | None = None) -> ReadingOneInputs:
    """Pooled inputs from `pooled` overrides over three per-seed leaves;
    `seeds` maps a seed to the overrides for that seed's leaf alone."""
    seeds = seeds or {}
    return _leaf(**(pooled or {}), per_seed={s: _leaf(**seeds.get(s, {})) for s in (0, 1, 2)})


def _step(k: int) -> np.ndarray:
    """A survival curve that is 1 through h = k and 0 after: H*_q = k for every q <= 1."""
    return np.array([1.0] * (k + 1) + [0.0] * (HORIZON - k))


def _curves(**per_key: np.ndarray) -> dict:
    """All six (arm, channel) curves, defaulting to 'never crosses' (S = 1
    everywhere, H*_q = 45); `per_key` overrides by `arm__channel`."""
    curves = {(arm, channel): np.ones(HORIZON + 1) for arm in ARMS_ORDER for channel in ("probe", "free")}
    for name, curve in per_key.items():
        arm, channel = name.rsplit("__", 1)
        curves[(arm, channel)] = curve
    return curves


# --- Constants ---------------------------------------------------------------


def test_the_constants_are_the_specs():
    """ARMS_ORDER is the study's arm order restated (the module imports
    nothing but numpy and trust); the family is the eight contrasts of spec
    3.1; q = 0.75 is pre-registered and printed among 0.5 and 0.9."""
    assert ARMS_ORDER == ("pixel_ae", "frozen_ssl", "random_vit") == ARMS
    assert (TREATMENT, CONTROL) == ("frozen_ssl", "random_vit")
    assert FAMILY == 8
    assert Q_PREREGISTERED == 0.75
    assert Q_REPORTED == (0.5, 0.75, 0.9)
    assert [s.name for s in Status] == [
        "SUPPORTED", "NOT_SUPPORTED", "NOT_TESTABLE", "UNRESOLVED_PROBE", "UNRESOLVED_ALPHA"
    ]


# --- (i): the best-Δ arm ------------------------------------------------------


def test_best_delta_arm_is_the_arm_whose_delta_contrast_clears_z_fam_against_each_other_arm():
    """random_vit - pixel_ae is +4 and random_vit - frozen_ssl is +5, both
    above 3.0; no other arm clears even one. Both of random_vit's contrasts
    are keyed the other way round, so a lookup that forgets to flip the sign
    reads -4 and -5 and finds no arm."""
    assert best_delta_arm(_leaf().delta_contrast, Z_FAM) == "random_vit"


def test_best_delta_arm_clearing_one_contrast_but_not_the_other_is_none_and_reading_one_is_not_testable():
    """random_vit still clears pixel_ae (+4) but its contrast with frozen_ssl
    drops to +2: "if no arm clears both, (i) is undecidable at this precision
    and Reading 1 is not testable, never supported." Everything else in the
    fixture holds, so any status but NOT_TESTABLE means a rule was skipped."""
    table = dict(_leaf().delta_contrast)
    table[("frozen_ssl", "random_vit")] = _c(-2.0)
    assert best_delta_arm(table, Z_FAM) is None
    result = reading_one(_inputs(pooled={"delta_contrast": table}), Z_FAM)
    assert result.status is Status.NOT_TESTABLE
    assert result.best_delta_arm is None
    assert result.conditions[0].holds is None
    assert "undecidable at this precision" in result.conditions[0].detail
    assert result.reason.startswith("(i) undecidable at this precision: no arm")
    assert result.reason.count("undecidable") == 1, "the name and the detail, never the word twice"
    # (ii) and (iii) were still evaluated and reported.
    assert result.conditions[1].holds is True and result.conditions[2].holds is True


def test_best_delta_arm_reads_a_pair_keyed_the_other_way_round_with_its_sign_flipped():
    """The same table keyed `(random_vit, other)`: +4 and +5 are read
    directly and must NOT be flipped. Together with the test above this pins
    the flip to the reversed orientation only."""
    reversed_keys = {
        ("frozen_ssl", "pixel_ae"): _c(+1.0),
        ("random_vit", "pixel_ae"): _c(+4.0),
        ("random_vit", "frozen_ssl"): _c(+5.0),
    }
    assert best_delta_arm(reversed_keys, Z_FAM) == "random_vit"


def test_best_delta_arm_refuses_a_missing_pair_and_an_inconsistent_table():
    """A pair absent from the table is a KeyError naming it, not "does not
    clear"; a table keyed in both orientations with values that let two arms
    win is a ValueError, not the first arm in ARMS_ORDER."""
    table = dict(_leaf().delta_contrast)
    del table[("pixel_ae", "random_vit")]
    with pytest.raises(KeyError) as excinfo:
        best_delta_arm(table, Z_FAM)
    assert "no Δ(45) contrast for the pair" in str(excinfo.value)
    assert "'pixel_ae'" in str(excinfo.value) and "'random_vit'" in str(excinfo.value)
    inconsistent = {
        ("pixel_ae", "frozen_ssl"): _c(+4.0), ("frozen_ssl", "pixel_ae"): _c(+4.0),
        ("pixel_ae", "random_vit"): _c(+4.0), ("random_vit", "pixel_ae"): _c(+4.0),
        ("frozen_ssl", "random_vit"): _c(+4.0), ("random_vit", "frozen_ssl"): _c(+4.0),
    }
    with pytest.raises(ValueError, match="inconsistent"):
        best_delta_arm(inconsistent, Z_FAM)


def test_a_z_exactly_at_z_fam_does_not_clear_on_any_rule():
    """Every comparison is strict: z == z_fam is not `z > z_fam`. Pinned on
    (i), (ii) and (iii) separately, because each has its own comparison."""
    table = dict(_leaf().delta_contrast)
    table[("frozen_ssl", "random_vit")] = _c(-3.0)
    assert best_delta_arm(table, Z_FAM) is None
    assert condition_ii(_leaf(cosine_contrast=_c(3.0)), Z_FAM).holds is False
    assert condition_iii(_leaf(corrected_contrast_b=_c(-3.0)), Z_FAM).holds is None


# --- (i): the least-moving arm ------------------------------------------------


def test_least_moving_arm_is_the_smallest_probe_ratio_when_the_free_channel_agrees():
    """random_vit reads 0.4 under R_probe and 0.3 under R_free, the smallest
    in each; the answer is the arm, and condition (i) holds because it is
    also the best-Δ arm."""
    leaf = _leaf()
    assert least_moving_arm(leaf.ratio_probe, leaf.ratio_free) == "random_vit"
    result = condition_i(leaf, Z_FAM)
    assert (result.name, result.holds) == ("(i)", True)
    assert "best-Δ arm random_vit" in result.detail and "least-moving arm random_vit" in result.detail


def test_least_moving_arm_differing_between_channels_leaves_condition_i_unresolved_through_the_probe():
    """R_probe still says random_vit (0.4) but R_free now says pixel_ae
    (0.1): "it must be the same arm under R_free(45), or (i) is unresolved
    through the probe." The condition is None, not False, and Reading 1 is
    UNRESOLVED_PROBE even though (ii) and (iii) hold."""
    free = dict(_leaf().ratio_free)
    free["pixel_ae"] = _r(0.1)
    assert least_moving_arm(_leaf().ratio_probe, free) is None
    result = condition_i(_leaf(ratio_free=free), Z_FAM)
    assert result.holds is None
    assert "unresolved through the probe" in result.detail
    assert "random_vit under R_probe(45) but pixel_ae under R_free(45)" in result.detail
    verdict = reading_one(_inputs(pooled={"ratio_free": free}), Z_FAM)
    assert verdict.status is Status.UNRESOLVED_PROBE
    assert verdict.best_delta_arm == "random_vit" and verdict.least_moving_arm is None
    assert verdict.reason.startswith("(i) unresolved through the probe: the least-moving arm")
    assert verdict.reason.count("unresolved") == 1, "the name and the detail, never the word twice"


def test_least_moving_arm_is_undefined_on_a_nan_or_a_tied_ratio():
    """A NaN estimate cannot be compared, and an exact tie has no single
    smallest arm: both are None. The degenerate values are placed where a
    plain `min` would still name an arm that AGREES with the other channel
    -- frozen_ssl's probe ratio NaN leaves min at random_vit, which is the
    free channel's answer; a pixel_ae/random_vit tie in BOTH channels leaves
    min at pixel_ae in both -- so a missing guard returns an arm, not None."""
    leaf = _leaf()
    nan_probe = dict(leaf.ratio_probe)
    nan_probe["frozen_ssl"] = _r(NAN)
    assert least_moving_arm(nan_probe, leaf.ratio_free) is None
    tied_probe = dict(leaf.ratio_probe)
    tied_free = dict(leaf.ratio_free)
    tied_probe["pixel_ae"], tied_free["pixel_ae"] = _r(0.4), _r(0.3)
    assert least_moving_arm(tied_probe, tied_free) is None
    assert condition_i(_leaf(ratio_probe=tied_probe, ratio_free=tied_free), Z_FAM).holds is None


def test_condition_i_holds_iff_the_least_moving_arm_is_the_best_delta_arm():
    """frozen_ssl moves least under both channels (0.1, 0.1) while random_vit
    is still the best-Δ arm: (i) is decided -- False -- and the detail names
    both arms."""
    probe = dict(_leaf().ratio_probe)
    free = dict(_leaf().ratio_free)
    probe["frozen_ssl"], free["frozen_ssl"] = _r(0.1), _r(0.1)
    result = condition_i(_leaf(ratio_probe=probe, ratio_free=free), Z_FAM)
    assert result.holds is False
    assert "best-Δ arm random_vit" in result.detail
    assert "least-moving arm frozen_ssl" in result.detail
    assert result.detail.endswith("different arms")


# --- (ii) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "z, holds",
    [(3.0001, True), (2.9999, False), (4.0, True), (-4.0, False), (NAN, False), (math.inf, False)],
)
def test_condition_ii_reads_the_cosine_z_strictly_above_z_fam(z, holds):
    """`z > z_fam`, signed: a large NEGATIVE cosine contrast (the treatment
    points the wrong way) does not hold, and neither does a NaN (fewer than
    two clusters) or an infinite z (a zero standard error)."""
    result = condition_ii(_leaf(cosine_contrast=_c(z)), Z_FAM)
    assert result.name == "(ii)" and result.holds is holds
    assert "cos(45) frozen_ssl-random_vit" in result.detail


# --- (iii) --------------------------------------------------------------------


@pytest.mark.parametrize(
    "z_a, z_b, holds",
    [
        (-4.0, -3.5, True),      # both folds below -z_fam
        (-4.0, -1.0, None),      # fold A only: not holds, not fails
        (-1.0, -4.0, None),      # fold B only
        (-1.0, -1.0, None),      # neither fold resolved
        (-4.0, +4.0, False),     # a fold above +z_fam fails whatever the other says
        (+4.0, -4.0, False),
        (+3.5, +1.0, False),
        (NAN, -4.0, None),       # a NaN fold is undecided, not a failure
    ],
)
def test_condition_iii_needs_both_folds_below_minus_z_fam(z_a, z_b, holds):
    """"holds iff z < -z_fam ...; fails iff z > z_fam; undecided otherwise.
    (iii) holds only if it holds on both folds." """
    result = condition_iii(
        _leaf(corrected_contrast_a=_c(z_a), corrected_contrast_b=_c(z_b)), Z_FAM
    )
    assert result.name == "(iii)" and result.holds is holds
    assert "fold A" in result.detail and "fold B" in result.detail
    if holds is None:
        assert result.detail.startswith("undecided")


def test_condition_iii_is_unreadable_on_a_boundary_alpha_of_the_treatment_or_the_control_but_not_of_pixel_ae():
    """"unreadable -- Reading 1 unresolved -- if either arm's α is on the
    grid boundary on either fold": either arm of the CONTRAST. The control's
    boundary and the treatment's each make (iii) None with "unreadable" and
    the verdict UNRESOLVED_ALPHA, even with both folds far below -z_fam;
    pixel_ae's boundary decides nothing and the reading stays SUPPORTED."""
    for arm in (CONTROL, TREATMENT):
        flags = {"pixel_ae": False, "frozen_ssl": False, "random_vit": False, arm: True}
        result = condition_iii(_leaf(alpha_boundary=flags), Z_FAM)
        assert result.holds is None and result.detail.startswith("unreadable"), arm
        assert arm in result.detail
        verdict = reading_one(_inputs(pooled={"alpha_boundary": flags}), Z_FAM)
        assert verdict.status is Status.UNRESOLVED_ALPHA, arm
        assert verdict.reason.startswith("(iii) unreadable: alpha on the grid boundary")
        assert verdict.reason.count("unreadable") == 1, "the name and the detail, never the word twice"
    pixel_only = {"pixel_ae": True, "frozen_ssl": False, "random_vit": False}
    assert condition_iii(_leaf(alpha_boundary=pixel_only), Z_FAM).holds is True
    assert reading_one(_inputs(pooled={"alpha_boundary": pixel_only}), Z_FAM).status is Status.SUPPORTED


# --- The probe control --------------------------------------------------------


@pytest.mark.parametrize(
    "z_probe, z_free, fires",
    [
        (+4.0, -4.0, True),
        (-4.0, +4.0, True),
        (+4.0, +4.0, False),    # same sign: the channels agree
        (-4.0, -4.0, False),
        (+4.0, -1.0, False),    # the free channel is inside the bar
        (+1.0, -4.0, False),    # the probe channel is inside the bar
        (0.0, -4.0, False),     # a tie contradicts nothing
        (NAN, -4.0, False),
        (+3.0, -4.0, False),    # exactly at the bar does not clear
    ],
)
def test_probe_control_fires_only_when_both_channels_clear_with_opposite_signs(z_probe, z_free, fires):
    """"if both channels clear z_fam with opposite signs, Reading 1 is
    unresolved through the probe ... A tie, or an interval covering 0 in
    either channel, contradicts nothing." """
    leaf = _leaf(crossing_contrast_probe=_c(z_probe), crossing_contrast_free=_c(z_free))
    assert probe_control(leaf, Z_FAM) is fires


def test_probe_control_firing_is_unresolved_through_the_probe_whatever_the_conditions_said():
    """Every condition holds pooled and in all three seeds; the control alone
    fires, and the verdict is UNRESOLVED_PROBE with the two z's in the
    reason -- and the conditions are still reported as holding."""
    fire = {"crossing_contrast_probe": _c(+4.0), "crossing_contrast_free": _c(-4.0)}
    result = reading_one(_inputs(pooled=fire), Z_FAM)
    assert result.status is Status.UNRESOLVED_PROBE
    assert result.reason.startswith("probe control")
    assert "probe z +4.00" in result.reason and "free z -4.00" in result.reason
    assert [c.holds for c in result.conditions] == [True, True, True]
    assert result.per_seed_agreement == {"(i)": 3, "(ii)": 3, "(iii)": 3}


# --- Per-seed agreement and the verdict ----------------------------------------


def test_per_seed_agreement_three_two_and_one_of_three():
    """Pooled everything holds throughout. 3/3 seeds: SUPPORTED. Seed 2's
    cosine contrast drops to +1: (ii) holds in 2 of 3 -- still SUPPORTED,
    "at least two of the three seeds". Seeds 1 and 2 both drop: 1 of 3,
    NOT_SUPPORTED, and the reason says which condition and the count."""
    three = reading_one(_inputs(), Z_FAM)
    assert three.status is Status.SUPPORTED
    assert three.per_seed_agreement == {"(i)": 3, "(ii)": 3, "(iii)": 3}
    assert three.best_delta_arm == "random_vit" and three.least_moving_arm == "random_vit"
    assert "(ii) 3/3" in three.reason

    two = reading_one(_inputs(seeds={2: {"cosine_contrast": _c(1.0)}}), Z_FAM)
    assert two.status is Status.SUPPORTED
    assert two.per_seed_agreement == {"(i)": 3, "(ii)": 2, "(iii)": 3}

    one = reading_one(
        _inputs(seeds={1: {"cosine_contrast": _c(1.0)}, 2: {"cosine_contrast": _c(1.0)}}), Z_FAM
    )
    assert one.status is Status.NOT_SUPPORTED
    assert one.per_seed_agreement == {"(i)": 3, "(ii)": 1, "(iii)": 3}
    assert [c.holds for c in one.conditions] == [True, True, True]
    assert "(ii) holds in 1 of 3 seeds" in one.reason and "2 of 3 required" in one.reason


def test_an_undecided_seed_does_not_count_as_agreeing():
    """In seeds 1 and 2 the best-Δ arm is undecidable (random_vit - frozen_ssl
    reads +2): (i) is None there, which is not "holds". Agreement on (i) is
    1 of 3 and the reading is NOT_SUPPORTED; counting None as not-False
    would call it 3 of 3."""
    table = dict(_leaf().delta_contrast)
    table[("frozen_ssl", "random_vit")] = _c(-2.0)
    result = reading_one(
        _inputs(seeds={1: {"delta_contrast": table}, 2: {"delta_contrast": table}}), Z_FAM
    )
    assert result.status is Status.NOT_SUPPORTED
    assert result.per_seed_agreement == {"(i)": 1, "(ii)": 3, "(iii)": 3}
    assert "(i) holds in 1 of 3 seeds" in result.reason


def test_a_pooled_condition_failing_is_not_supported_even_when_every_seed_agrees_and_ii_alone_is_named_informative():
    """The pooled cosine contrast reads +1 while all three seeds hold: the
    rule is "all hold pooled AND ...", so NOT_SUPPORTED. And "(ii) failing
    alone is the informative failure: the prior does not drift slowly, it
    moves wrong" -- the reason says so when (ii) is the only failure, and
    does not when (i) fails beside it."""
    only_ii = reading_one(_inputs(pooled={"cosine_contrast": _c(1.0)}), Z_FAM)
    assert only_ii.status is Status.NOT_SUPPORTED
    assert only_ii.per_seed_agreement == {"(i)": 3, "(ii)": 3, "(iii)": 3}
    assert only_ii.reason.startswith("pooled: (ii) not holding")
    assert "it moves wrong" in only_ii.reason

    probe = dict(_leaf().ratio_probe)
    free = dict(_leaf().ratio_free)
    probe["frozen_ssl"], free["frozen_ssl"] = _r(0.1), _r(0.1)
    i_and_ii = reading_one(
        _inputs(pooled={"cosine_contrast": _c(1.0), "ratio_probe": probe, "ratio_free": free}), Z_FAM
    )
    assert i_and_ii.status is Status.NOT_SUPPORTED
    assert i_and_ii.reason.startswith("pooled: (i), (ii) not holding")
    assert "it moves wrong" not in i_and_ii.reason


def test_the_status_precedence_is_probe_then_alpha_then_testable_then_least_moving_then_the_conditions():
    """Four inputs, each carrying every failure of the rungs below it."""
    undecidable = dict(_leaf().delta_contrast)
    undecidable[("frozen_ssl", "random_vit")] = _c(-2.0)
    disagreeing_free = dict(_leaf().ratio_free)
    disagreeing_free["pixel_ae"] = _r(0.1)
    control_boundary = {"pixel_ae": False, "frozen_ssl": False, "random_vit": True}
    everything = dict(
        crossing_contrast_probe=_c(+4.0), crossing_contrast_free=_c(-4.0),
        alpha_boundary=control_boundary, delta_contrast=undecidable,
        ratio_free=disagreeing_free, cosine_contrast=_c(1.0),
    )
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.UNRESOLVED_PROBE

    del everything["crossing_contrast_probe"], everything["crossing_contrast_free"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.UNRESOLVED_ALPHA

    del everything["alpha_boundary"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.NOT_TESTABLE

    del everything["delta_contrast"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.UNRESOLVED_PROBE

    del everything["ratio_free"]
    assert reading_one(_inputs(pooled=everything), Z_FAM).status is Status.NOT_SUPPORTED


def test_reading_one_refuses_inputs_without_per_seed_entries():
    """The two-of-three rule cannot be evaluated on a leaf; a leaf passed as
    the pooled inputs is a ValueError, not a SUPPORTED with 0 of 0 seeds."""
    with pytest.raises(ValueError, match="per_seed"):
        reading_one(_leaf(), Z_FAM)


# --- Reading 2 -----------------------------------------------------------------


def test_reading_two_on_a_bimodal_survival_curve():
    """Half the draws cross at h× = 2, half never (h× = 46): S(0) = S(1) = 1,
    S(2..45) = 0.5. H*_0.5 is 45 (S(45) = 0.5 >= 0.5), H*_0.75 is 1 and
    H*_0.9 is 1 -- the median reading says "trust it to the end", the
    pre-registered quantile says "one step". The other five curves never
    cross, so H*_min is this arm's 1."""
    bimodal = np.array([1.0, 1.0] + [0.5] * 44)
    assert bimodal.shape == (HORIZON + 1,)
    result = reading_two(_curves(frozen_ssl__free=bimodal))
    assert isinstance(result, ReadingTwo)
    assert result.horizons[("frozen_ssl", "free", 0.5)] == 45
    assert result.horizons[("frozen_ssl", "free", 0.75)] == 1
    assert result.horizons[("frozen_ssl", "free", 0.9)] == 1
    assert result.horizons[("frozen_ssl", "probe", 0.75)] == 45
    assert result.h_min == 1
    assert set(result.horizons) == {(a, ch, q) for a in ARMS_ORDER for ch in ("probe", "free") for q in Q_REPORTED}
    np.testing.assert_array_equal(result.survival[("frozen_ssl", "free")], bimodal)


def test_reading_two_on_an_even_count_curve_is_an_integer_with_no_rounding():
    """Four draws, h× = {3, 3, 7, 7}: S = 1 for h < 3, 0.5 for 3 <= h < 7,
    0 from h = 7. H*_0.5 is 6 -- the largest h with S >= 0.5 -- an int, not
    np.median's interpolated 5.0; H*_0.75 and H*_0.9 are 2."""
    even = np.array([1.0] * 3 + [0.5] * 4 + [0.0] * 39)
    assert even.shape == (HORIZON + 1,)
    result = reading_two(_curves(pixel_ae__probe=even))
    assert result.horizons[("pixel_ae", "probe", 0.5)] == 6
    assert type(result.horizons[("pixel_ae", "probe", 0.5)]) is int
    assert result.horizons[("pixel_ae", "probe", 0.5)] != np.median([3, 3, 7, 7])
    assert result.horizons[("pixel_ae", "probe", 0.75)] == 2
    assert result.horizons[("pixel_ae", "probe", 0.9)] == 2
    assert result.h_min == 45


def test_h_min_is_the_minimum_of_the_probe_free_h_star_at_the_preregistered_q():
    """Probe-free H*_0.75: pixel_ae 10, frozen_ssl 1 (the bimodal curve,
    whose H*_0.5 is 45), random_vit 12 -> H*_min = 1. The probe channel
    reads 0, 20, 2 and must not enter: over both channels the answer would
    be 0, at q = 0.5 it would be 10, with max in place of min 12."""
    bimodal = np.array([1.0, 1.0] + [0.5] * 44)
    result = reading_two(_curves(
        pixel_ae__probe=_step(0), frozen_ssl__probe=_step(20), random_vit__probe=_step(2),
        pixel_ae__free=_step(10), frozen_ssl__free=bimodal, random_vit__free=_step(12),
    ))
    assert [result.horizons[(a, "free", 0.75)] for a in ARMS_ORDER] == [10, 1, 12]
    assert [result.horizons[(a, "probe", 0.75)] for a in ARMS_ORDER] == [0, 20, 2]
    assert result.horizons[("frozen_ssl", "free", 0.5)] == 45
    assert result.h_min == 1
    assert type(result.h_min) is int


def test_reading_two_refuses_a_missing_arm_or_channel_and_a_q_list_without_the_preregistered_q():
    """A curve missing for random_vit/free is a KeyError naming it -- a
    minimum over two arms would print as one over three -- and a `qs`
    without 0.75 is a ValueError: H*_min is defined at the pre-registered q."""
    curves = _curves()
    del curves[("random_vit", "free")]
    with pytest.raises(KeyError, match=r"survival curve.*\('random_vit', 'free'\)"):
        reading_two(curves)
    with pytest.raises(ValueError, match="pre-registered q 0.75"):
        reading_two(_curves(), qs=(0.5, 0.9))


# --- Formatting ------------------------------------------------------------------


def test_format_reading_one_prints_every_condition_detail_every_seed_and_the_deciding_rule():
    """Seed 2 fails (ii) with z = +1 and the pooled control fires, so the
    printout must carry: the pooled details of all three conditions, the
    probe-control line with both z's, a block per seed holding that seed's
    own details (seed 2's (ii) reads z +1.00, the others +4.00), the
    agreement counts, and the verdict line naming the status and the rule
    that decided it."""
    inputs = _inputs(
        pooled={"crossing_contrast_probe": _c(+4.0), "crossing_contrast_free": _c(-4.0)},
        seeds={2: {"cosine_contrast": _c(1.0)}},
    )
    result = reading_one(inputs, Z_FAM, h=45)
    assert result.status is Status.UNRESOLVED_PROBE
    text = format_reading_one(result, inputs, Z_FAM, h=45)
    lines = text.splitlines()
    assert lines[0].startswith("--- Reading 1: does the h=45 gate reward slow drift?")
    assert "z_fam 3.00" in lines[0] and "family 8" in lines[0]
    for condition in result.conditions:
        assert condition.detail in text, condition.name
        assert f"{condition.name:<6}holds" in text, condition.name
    assert "probe control: fires" in text and "probe z +4.00" in text and "free z -4.00" in text
    for seed in (0, 1, 2):
        assert f"seed {seed}:" in text
    seed_2 = text[text.index("seed 2:"):]
    assert f"{'(ii)':<6}fails" in seed_2 and "z +1.00 not > z_fam 3.00" in seed_2
    assert text.count("z +4.00 > z_fam 3.00") == 3   # pooled, seed 0, seed 1
    assert "per-seed agreement: (i) 3/3, (ii) 2/3, (iii) 3/3  (2 of 3 required)" in text
    assert "best-Δ arm: random_vit; least-moving arm: random_vit" in text
    assert lines[-1] == f"verdict: Reading 1 {Status.UNRESOLVED_PROBE.value} -- decided by: {result.reason}"
    # `h` names the step everywhere it is printed and decides nothing: the
    # same inputs read at h = 3 give the same status with every "45" now "3".
    at_three = reading_one(inputs, Z_FAM, h=3)
    assert at_three.status is result.status
    text_three = format_reading_one(at_three, inputs, Z_FAM, h=3)
    assert text_three.splitlines()[0].startswith("--- Reading 1: does the h=3 gate reward slow drift?")
    assert "cos(3) frozen_ssl-random_vit" in text_three and "held-out c(3)" in text_three
    assert "(45)" not in text_three and "h=45" not in text_three


def test_format_reading_two_prints_every_survival_value_every_horizon_and_h_min():
    """One line per arm and channel carrying H*_0.5, H*_0.75, H*_0.9 and all
    46 values of S(h); then H*_min with the probe-based H*_0.75 beside it."""
    bimodal = np.array([1.0, 1.0] + [0.5] * 44)
    result = reading_two(_curves(frozen_ssl__free=bimodal, random_vit__probe=_step(7)))
    text = format_reading_two(result)
    lines = text.splitlines()
    assert lines[0].startswith("--- Reading 2: the horizon M4 designs around")
    assert "no verdict" in lines[0]
    assert lines[1].split()[:5] == ["arm", "channel", "H*_0.5", "H*_0.75", "H*_0.9"]
    body = lines[2:8]
    assert [line.split()[:2] for line in body] == [
        [arm, channel] for arm in ARMS_ORDER for channel in ("probe", "free")
    ]
    frozen_free = next(line for line in body if line.startswith(f"{'frozen_ssl':<12}{'free':<9}"))
    tokens = frozen_free.split()
    assert tokens[2:5] == ["45", "1", "1"]
    assert tokens[5:] == ["1.00", "1.00"] + ["0.50"] * 44
    random_probe = next(line for line in body if line.startswith(f"{'random_vit':<12}{'probe':<9}"))
    assert random_probe.split()[2:5] == ["7", "7", "7"]
    assert lines[-1].startswith("H*_min = 1  (min over arms of the probe-free H*_0.75")
    assert "pixel_ae 45, frozen_ssl 45, random_vit 7" in lines[-1]
