"""mbfps.eval.split_gap: the pure functions of the M3e diagnostic, pinned on
fabricated inputs with hand-worked answers.

`strata_partition` on synthetic path lists, each of its refusals provoked one
at a time. `stratum_summary` on the SAME fabricated trajectories
`test_trust_horizon_script.py` uses (two windows, two steps, a biased probe),
so every crossing and margin below is a number that file already derives by
hand -- plus a `band` with curves chosen so `gap_closed` is a clean fraction.
"""

from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.diagnostics import Trajectories
from mbfps.eval.rollout import RolloutResult
from mbfps.eval.split_gap import (
    ArmInputs,
    ArmReading,
    CHANNELS,
    CURVE_NAMES,
    DECISION,
    DECISION_H,
    FAMILY,
    GapInputs,
    GapReading,
    REPORTED_H,
    SEEDS_REQUIRED,
    STRATA,
    Status,
    StratumContrast,
    TERMS,
    StrataNotAPartition,
    clears,
    format_reading_gap,
    learning_curve_summary,
    q_key,
    reading_gap,
    strata_partition,
    stratum_summary,
    train_held_passes_gate,
)
from mbfps.eval.trust_readings import Q_REPORTED


def test_the_pre_registered_constants_are_the_specs():
    """Spec 2.1 / 3.1 / 3.2 / 2.3, as literals: these are not tuned after the run."""
    assert STRATA == ("val", "train_held", "train_probe")
    assert DECISION == ("train_held", "val")
    assert CHANNELS == ("probe", "free")
    assert CURVE_NAMES == ("rssm_position", "persistence_position", "floor_position",
                           "rssm_angle", "persistence_angle", "floor_angle")
    assert DECISION_H == 15 and REPORTED_H == (5, 15, 45) and DECISION_H in REPORTED_H
    assert FAMILY == 6 and SEEDS_REQUIRED == 2
    assert TERMS == ("embedding", "reward", "continue", "kl_dyn", "kl_rep")
    assert [q_key(q) for q in Q_REPORTED] == ["q50", "q75", "q90"], (
        "JSON keys may not contain '.'; write_record refuses them")


# ---------------------------------------------------------------------------
# strata_partition
# ---------------------------------------------------------------------------


def _paths(n: int) -> list[Path]:
    return [Path(f"ep_{i:06d}_len00526.npz") for i in range(n)]


def test_strata_partition_returns_the_three_lists_in_the_callers_order():
    """30 episodes: val = every fifth, train = the rest (24), used = train[:20]
    -> train_held = train[20:] (4). Order is preserved, never sorted."""
    everything = _paths(30)
    val = everything[::5]
    train = [p for p in everything if p not in val]
    used = train[:20]
    strata = strata_partition(everything, train, val, used)
    assert list(strata) == list(STRATA)
    assert strata["val"] == val
    assert strata["train_probe"] == used
    assert strata["train_held"] == train[20:] and len(strata["train_held"]) == 4
    assert sum(len(v) for v in strata.values()) == 30


def test_strata_partition_refuses_an_overlap_a_gap_a_non_prefix_and_an_empty_decision_stratum():
    everything = _paths(30)
    val = everything[::5]
    train = [p for p in everything if p not in val]
    used = train[:20]
    with pytest.raises(StrataNotAPartition, match="share"):
        strata_partition(everything, train, val + [train[0]], used)
    with pytest.raises(StrataNotAPartition, match="do not cover"):
        strata_partition(everything + [Path("ep_999999_len00526.npz")], train, val, used)
    with pytest.raises(StrataNotAPartition, match="do not cover"):
        strata_partition(everything, train[1:], val, train[1:21])
    with pytest.raises(StrataNotAPartition, match="leading block"):
        strata_partition(everything, train, val, train[1:21])
    # Six episodes, one held out: the probe takes all five training episodes.
    six = _paths(6)
    with pytest.raises(StrataNotAPartition, match="train_held is empty"):
        strata_partition(six, six[1:], six[:1], six[1:])


# ---------------------------------------------------------------------------
# stratum_summary, on fabricated trajectories.
# ---------------------------------------------------------------------------


BIAS = np.array([0.0, 9.0])


def _band() -> RolloutResult:
    """Means of the two rows below -- reference [2, 10.5], persistence
    [9.5, 31] -- with a floor of [1, 1]: gap_closed position = [7.5/8.5,
    20.5/30]. Angle curves chosen so gap_closed angle = [20/25, 20/35]."""
    return RolloutResult(
        horizon=np.array([1, 2]),
        rssm_position=np.array([2.0, 10.5]),
        persistence_position=np.array([9.5, 31.0]),
        floor_position=np.array([1.0, 1.0]),
        rssm_angle=np.array([10.0, 20.0]),
        persistence_angle=np.array([30.0, 40.0]),
        floor_angle=np.array([5.0, 5.0]),
    )


def _fabricated(*, stationary_third_window: bool = False) -> Trajectories:
    """Window 0 a perfect predictor, window 1 a persistence clone, the probe
    biased by BIAS at the anchor (the numbers of
    test_trust_horizon_script._fabricated). With the flag, a THIRD window that
    never moves: its every quantity must be NaN or counted, never pooled.

      persistence err  row 0 [15, 41]  row 1 [4, 21]
      model err        row 0 [0, 0]    row 1 [4, 21]
      -> crossing probe [3, 3] (never; ties never cross); margin probe
         row 0 [15, 41], row 1 [0, 0]
      D_hat row 0 [0, 0] row 1 [5, 5]; D_0 [3, 4] both rows
      -> crossing free [3, 1]; margin free row 0 [3, 4], row 1 [-2, -1]
    """
    true_at_context = np.array([[0.0, 0.0], [10.0, 10.0]])
    true_positions = np.array([[[12.0, 0.0], [40.0, 0.0]], [[10.0, 15.0], [10.0, 40.0]]])
    positions = np.array([[[12.0, 0.0], [40.0, 0.0]], [[10.0, 19.0], [10.0, 19.0]]])
    d_hat = np.array([[0.0, 0.0], [5.0, 5.0]])
    d_0 = np.array([[3.0, 4.0], [3.0, 4.0]])
    e_disp = np.array([[2.0, 4.0], [1.0, 2.0]])
    e_true = np.array([[2.0, 4.0], [2.0, 4.0]])
    episode = np.array([0, 1])
    if stationary_third_window:
        true_at_context = np.vstack([true_at_context, [[5.0, 5.0]]])
        true_positions = np.vstack([true_positions, [[[5.0, 5.0], [5.0, 5.0]]]])
        positions = np.vstack([positions, [[[5.0, 14.0], [5.0, 14.0]]]])
        d_hat = np.vstack([d_hat, [[1.0, 1.0]]])
        d_0 = np.vstack([d_0, [[1.0, 1.0]]])
        e_disp = np.vstack([e_disp, [[0.0, 0.0]]])
        e_true = np.vstack([e_true, [[0.0, 0.0]]])
        episode = np.array([0, 1, 1])
    positions_at_context = true_at_context + BIAS
    d = true_positions - true_at_context[:, None, :]
    n = true_positions.shape[0]
    return Trajectories(
        positions=positions,
        positions_at_context=positions_at_context,
        positions_real=positions_at_context[:, None, :] + 0.5 * d,
        true_positions=true_positions,
        true_at_context=true_at_context,
        embedding_distance_to_truth=d_hat,
        embedding_persistence_distance=d_0,
        embedding_displacement=e_disp,
        true_embedding_displacement=e_true,
        window_episode=episode,
        windows_total=n,
        reference_position=np.linalg.norm(positions - true_positions, axis=-1).mean(axis=0),
        persistence_position=np.linalg.norm(
            positions_at_context[:, None, :] - true_positions, axis=-1
        ).mean(axis=0),
        band=_band(),
    )


def test_stratum_summary_reduces_the_pass_through_trusts_functions():
    s = stratum_summary(_fabricated(), horizon=2)
    assert set(s) == {"windows", "curves", "band", "moved", "first_moved", "crossing",
                      "margin", "survival", "trust_horizon", "counts"}
    assert s["windows"] == {"total": 2, "episode": [0, 1], "clusters": 2}
    assert s["moved"].all() and s["moved"].shape == (2, 2)
    np.testing.assert_array_equal(s["first_moved"], [1.0, 1.0])
    np.testing.assert_array_equal(s["crossing"]["probe"], [3.0, 3.0])
    np.testing.assert_array_equal(s["crossing"]["free"], [3.0, 1.0])
    np.testing.assert_array_equal(s["margin"]["probe"], [[15.0, 41.0], [0.0, 0.0]])
    np.testing.assert_array_equal(s["margin"]["free"], [[3.0, 4.0], [-2.0, -1.0]])
    np.testing.assert_array_equal(s["survival"]["probe"], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(s["survival"]["free"], [1.0, 0.5, 0.5])
    assert s["trust_horizon"]["probe"] == {"q50": 2, "q75": 2, "q90": 2}
    assert s["trust_horizon"]["free"] == {"q50": 2, "q75": 0, "q90": 0}
    np.testing.assert_array_equal(s["counts"]["not_moved"], [0, 0])
    assert s["counts"]["never_moved"] == 0


def test_stratum_summary_carries_the_band_and_its_gap_closed():
    s = stratum_summary(_fabricated(), horizon=2)
    assert set(s["curves"]) == set(CURVE_NAMES)
    np.testing.assert_array_equal(s["curves"]["floor_position"], [1.0, 1.0])
    np.testing.assert_array_equal(s["curves"]["rssm_angle"], [10.0, 20.0])
    position, angle = s["band"]["position"], s["band"]["angle"]
    assert position["gap_final"] == pytest.approx(20.5 / 30.0)
    assert position["steps_degenerate"] == 0 and position["steps_floor_above_persistence"] == 0
    assert position["gap_finite"] == 2 and position["n_steps"] == 2
    assert angle["gap_final"] == pytest.approx(20.0 / 35.0)
    assert angle["final_persistence"] == 40.0


def test_stratum_summary_counts_a_window_that_never_moves_and_pools_nothing_from_it():
    """Row 2 stands still: NaN crossing in both channels, NaN h0, counted in
    `never_moved` and in `not_moved` at every step, and `survival` is
    unchanged because `trust.survival` drops NaN crossings."""
    s = stratum_summary(_fabricated(stationary_third_window=True), horizon=2)
    assert s["windows"] == {"total": 3, "episode": [0, 1, 1], "clusters": 2}
    assert not s["moved"][2].any()
    assert np.isnan(s["first_moved"][2]) and np.isfinite(s["first_moved"][:2]).all()
    assert np.isnan(s["crossing"]["probe"][2]) and np.isnan(s["crossing"]["free"][2])
    np.testing.assert_array_equal(s["survival"]["free"], [1.0, 0.5, 0.5])
    np.testing.assert_array_equal(s["counts"]["not_moved"], [1, 1])
    assert s["counts"]["never_moved"] == 1


def test_stratum_summary_refuses_a_pass_without_a_band_and_a_horizon_that_is_not_the_rows():
    import dataclasses

    with pytest.raises(ValueError, match="band"):
        stratum_summary(dataclasses.replace(_fabricated(), band=None), horizon=2)
    with pytest.raises(ValueError, match="horizon"):
        stratum_summary(_fabricated(), horizon=3)


# ---------------------------------------------------------------------------
# Reading G, on fabricated pooled tables. z_fam is 3.0 throughout; every
# fabricated z is either clearly above (+4), clearly below (+1), clearly
# inverted (-4), or exactly the bar (3.0, which must NOT clear).
# ---------------------------------------------------------------------------

Z_FAM = 3.0


def _contrast(z: float) -> StratumContrast:
    return StratumContrast(estimate=0.1 * z, se=0.1, z=z, clusters=24)


def _arm(free_z, probe_z, *, seeds=(4.0, 4.0, 4.0), gate=(0.2, 0.3, 0.1)) -> ArmInputs:
    """Pooled contrasts plus three per-seed leaves (each with its own free z
    and the SAME probe z) and train_held's gap_closed(45) per seed."""
    per_seed = {
        seed: ArmInputs(
            gap_free=_contrast(z), gap_probe=_contrast(probe_z),
            train_held_gap_final={seed: gate[seed]}, per_seed=None,
        )
        for seed, z in enumerate(seeds)
    }
    return ArmInputs(
        gap_free=_contrast(free_z), gap_probe=_contrast(probe_z),
        train_held_gap_final={s: gate[s] for s in range(3)}, per_seed=per_seed,
    )


def _inputs(**arms) -> GapInputs:
    return GapInputs(arms=arms, z_fam=Z_FAM, h=DECISION_H)


def test_clears_is_strict_and_never_on_a_nan_or_infinite_z():
    assert clears(3.01, Z_FAM) and not clears(3.0, Z_FAM) and not clears(2.99, Z_FAM)
    assert not clears(float("nan"), Z_FAM) and not clears(float("inf"), Z_FAM)
    assert not clears(4.0, float("nan")), "a NaN bar cannot be cleared"


def test_train_held_passes_gate_is_spec_4_1_unanimity_with_nan_not_positive():
    assert train_held_passes_gate({0: 0.2, 1: 0.01, 2: 0.9})
    assert not train_held_passes_gate({0: 0.2, 1: 0.0, 2: 0.9}), "zero is not > 0"
    assert not train_held_passes_gate({0: 0.2, 1: -0.1, 2: 0.9})
    assert not train_held_passes_gate({0: 0.2, 1: float("nan"), 2: 0.9}), (
        "a NaN gap_closed is a non-positive band and is not > 0")
    assert not train_held_passes_gate({0: 0.2, 1: float("inf"), 2: 0.9}), (
        "an infinite gap_closed is not a measurement; the isfinite guard is what excludes it")
    assert not train_held_passes_gate({}), "no seed is not unanimity"


def test_memorisation_needs_the_pooled_gap_two_seeds_and_the_train_side_gate():
    reading = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0)))
    r = reading.arms["frozen_ssl"]
    assert r.status is Status.MEMORISATION
    assert r.free_clears_up and not r.probe_clears_up and not r.probe_clears_down
    assert (r.seeds_clearing, r.seeds_total, r.train_held_unanimous) == (3, 3, True)
    assert "passes" in r.rule and "4.1" in r.rule
    assert (reading.h, reading.z_fam) == (DECISION_H, Z_FAM)


def test_partial_gap_when_one_train_held_seed_does_not_beat_persistence_at_45():
    r = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0, gate=(0.2, -0.05, 0.1)))).arms["frozen_ssl"]
    assert r.status is Status.PARTIAL_GAP and not r.train_held_unanimous
    assert "FAILS" in r.rule
    nan = reading_gap(_inputs(a=_arm(4.0, 1.0, gate=(0.2, float("nan"), 0.1)))).arms["a"]
    assert nan.status is Status.PARTIAL_GAP, "a NaN band on one seed is not unanimity"


def test_a_pooled_gap_that_only_one_seed_shows_is_no_gap():
    r = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0, seeds=(4.0, 1.0, 1.0)))).arms["frozen_ssl"]
    assert r.status is Status.NO_GAP and r.seeds_clearing == 1
    assert "only 1 of 3" in r.rule
    two = reading_gap(_inputs(frozen_ssl=_arm(4.0, 1.0, seeds=(4.0, 4.0, 1.0)))).arms["frozen_ssl"]
    assert two.status is Status.MEMORISATION and two.seeds_clearing == 2, "two of three suffice"


def test_no_gap_when_the_free_contrast_does_not_clear_even_if_the_probe_does():
    r = reading_gap(_inputs(random_vit=_arm(1.0, 4.0))).arms["random_vit"]
    assert r.status is Status.NO_GAP and r.probe_clears_up and not r.free_clears_up
    assert "does not clear" in r.rule
    bar = reading_gap(_inputs(random_vit=_arm(3.0, 1.0))).arms["random_vit"]
    assert bar.status is Status.NO_GAP, "z equal to the bar does not clear"


def test_inverted_gap_when_train_held_is_worse_than_val():
    r = reading_gap(_inputs(pixel_ae=_arm(-4.0, -1.0))).arms["pixel_ae"]
    assert r.status is Status.INVERTED_GAP and r.free_clears_down
    also_probe = reading_gap(_inputs(pixel_ae=_arm(-4.0, -4.0))).arms["pixel_ae"]
    assert also_probe.status is Status.INVERTED_GAP, "both channels inverted agree; not unresolved"


def test_unresolved_probe_when_the_two_channels_clear_with_opposite_signs_and_it_wins_precedence():
    up_down = reading_gap(_inputs(pixel_ae=_arm(4.0, -4.0))).arms["pixel_ae"]
    assert up_down.status is Status.UNRESOLVED_PROBE and "opposite signs" in up_down.rule
    down_up = reading_gap(_inputs(pixel_ae=_arm(-4.0, 4.0))).arms["pixel_ae"]
    assert down_up.status is Status.UNRESOLVED_PROBE
    # Precedence: the SAME inputs that would be MEMORISATION become unresolved
    # when the probe twin clears the other way.
    assert reading_gap(_inputs(a=_arm(4.0, 1.0))).arms["a"].status is Status.MEMORISATION
    assert reading_gap(_inputs(a=_arm(4.0, -4.0))).arms["a"].status is Status.UNRESOLVED_PROBE


def test_reading_gap_reads_every_arm_independently_in_the_callers_order():
    reading = reading_gap(_inputs(
        pixel_ae=_arm(-4.0, -1.0), frozen_ssl=_arm(4.0, 1.0), random_vit=_arm(1.0, 1.0),
    ))
    assert list(reading.arms) == ["pixel_ae", "frozen_ssl", "random_vit"]
    assert [r.status for r in reading.arms.values()] == [
        Status.INVERTED_GAP, Status.MEMORISATION, Status.NO_GAP,
    ]


def test_reading_gap_on_a_single_seed_leaf_has_a_vacuous_replication_clause():
    """A single-seed leaf (`per_seed=None`), read alone -- the per-seed line
    of `scripts/split_gap.py` -- has nothing to replicate across, so a
    clearing `G_free` reaches MEMORISATION or PARTIAL_GAP on its own, always
    with (seeds_clearing, seeds_total) == (1, 1) and "this seed alone" named
    in the rule."""
    memorisation = reading_gap(_inputs(a=ArmInputs(
        gap_free=_contrast(4.0), gap_probe=_contrast(1.0),
        train_held_gap_final={0: 0.2}, per_seed=None,
    ))).arms["a"]
    assert memorisation.status is Status.MEMORISATION
    assert (memorisation.seeds_clearing, memorisation.seeds_total) == (1, 1)
    assert "this seed alone" in memorisation.rule

    partial = reading_gap(_inputs(a=ArmInputs(
        gap_free=_contrast(4.0), gap_probe=_contrast(1.0),
        train_held_gap_final={0: -0.2}, per_seed=None,
    ))).arms["a"]
    assert partial.status is Status.PARTIAL_GAP
    assert (partial.seeds_clearing, partial.seeds_total) == (1, 1)
    assert "this seed alone" in partial.rule

    no_gap = reading_gap(_inputs(a=ArmInputs(
        gap_free=_contrast(1.0), gap_probe=_contrast(1.0),
        train_held_gap_final={0: 0.2}, per_seed=None,
    ))).arms["a"]
    assert no_gap.status is Status.NO_GAP
    assert (no_gap.seeds_clearing, no_gap.seeds_total) == (0, 1)

    # The pooled path (a dict, possibly empty) is unchanged: 0 of 0 seeds
    # does not replicate, and the rule still names it that way.
    pooled_empty = reading_gap(_inputs(a=ArmInputs(
        gap_free=_contrast(4.0), gap_probe=_contrast(1.0),
        train_held_gap_final={0: 0.2}, per_seed={},
    ))).arms["a"]
    assert pooled_empty.status is Status.NO_GAP
    assert (pooled_empty.seeds_clearing, pooled_empty.seeds_total) == (0, 0)
    assert "only 0 of 0 seeds" in pooled_empty.rule


def test_format_reading_gap_prints_each_arm_with_its_status_and_rule():
    inputs = _inputs(frozen_ssl=_arm(4.0, 1.0), random_vit=_arm(1.0, 1.0))
    text = format_reading_gap(reading_gap(inputs), inputs)
    assert f"h={DECISION_H}" in text and f"z_fam = {Z_FAM:.2f}" in text
    assert "frozen_ssl" in text and "MEMORISATION" in text
    assert "random_vit" in text and "NO GAP" in text
    assert "decided by:" in text
    assert "train_held - val" in text
    for column in ("free", "probe"):
        assert column in text


# ---------------------------------------------------------------------------
# learning_curve_summary, on a synthetic history.
# ---------------------------------------------------------------------------


def _history(n: int = 400) -> dict:
    """embedding descends 1 -> 0.5 linearly, reward ascends 0 -> 1, the other
    three terms are flat; loss is their sum."""
    i = np.arange(n, dtype=float)
    embedding = 1.0 - 0.5 * i / (n - 1)
    reward = i / (n - 1)
    parts = [
        {"embedding": float(embedding[k]), "reward": float(reward[k]), "continue": 0.7,
         "kl_dyn": 0.3, "kl_rep": 0.3}
        for k in range(n)
    ]
    loss = [sum(p.values()) for p in parts]
    return {"loss": loss, "parts": parts}


def test_learning_curve_summary_reads_quarters_and_the_smoothed_minimum():
    s = learning_curve_summary(_history(400), window=100)
    assert (s["steps"], s["window"], s["quarter"]) == (400, 100, 100)
    assert set(s["terms"]) == {"loss", *TERMS}
    emb = s["terms"]["embedding"]
    assert emb["last_quarter_mean"] == pytest.approx(1.0 - 0.5 * 349.5 / 399)
    assert emb["preceding_quarter_mean"] == pytest.approx(1.0 - 0.5 * 249.5 / 399)
    assert emb["descending"] and emb["change_pct"] < 0
    assert emb["smoothed_min_step"] == 400, "still descending: the minimum is the last window"
    rew = s["terms"]["reward"]
    assert not rew["descending"] and rew["change_pct"] > 0
    assert rew["smoothed_min_step"] == 100, "ascending: the minimum is the FIRST full window"
    flat = s["terms"]["kl_dyn"]
    assert flat["change_pct"] == pytest.approx(0.0) and not flat["descending"]
    assert s["terms"]["loss"]["last_quarter_mean"] == pytest.approx(
        emb["last_quarter_mean"] + rew["last_quarter_mean"] + 1.3
    )


def test_learning_curve_summary_shrinks_the_window_to_the_history_and_refuses_junk():
    short = learning_curve_summary(_history(8), window=100)
    assert short["window"] == 8 and short["quarter"] == 2
    with pytest.raises(ValueError, match="four"):
        learning_curve_summary(_history(3))
    with pytest.raises(ValueError, match="window"):
        learning_curve_summary(_history(8), window=0)
    broken = _history(8)
    broken["parts"] = broken["parts"][:-1]
    with pytest.raises(ValueError, match="disagree"):
        learning_curve_summary(broken)
