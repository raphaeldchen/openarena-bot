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
    CHANNELS,
    CURVE_NAMES,
    DECISION,
    DECISION_H,
    FAMILY,
    REPORTED_H,
    SEEDS_REQUIRED,
    STRATA,
    TERMS,
    StrataNotAPartition,
    q_key,
    strata_partition,
    stratum_summary,
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
