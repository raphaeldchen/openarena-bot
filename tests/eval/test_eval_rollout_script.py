"""The reporting layer must guard BOTH metrics, not just position.

Every degeneracy guard was written for position and none for angle, so
`angle_gap_closed_final` was printed as a bare number while its curve ranged
+8.68 to -23.04 with one NaN step.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_rollout_script",
    Path(__file__).resolve().parents[2] / "scripts" / "eval_rollout.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

from mbfps.eval.rollout import RolloutResult


def _result(pos, pers_pos, floor_pos, ang, pers_ang, floor_ang) -> RolloutResult:
    n = len(pos)
    return RolloutResult(
        horizon=np.arange(1, n + 1),
        rssm_position=np.array(pos, dtype=float),
        persistence_position=np.array(pers_pos, dtype=float),
        floor_position=np.array(floor_pos, dtype=float),
        rssm_angle=np.array(ang, dtype=float),
        persistence_angle=np.array(pers_ang, dtype=float),
        floor_angle=np.array(floor_ang, dtype=float),
    )


def test_band_report_accepts_a_metric_and_reads_the_matching_curves():
    """Asking for angle must read the angle curves, not the position ones."""
    result = _result(
        pos=[10.0, 10.0], pers_pos=[20.0, 20.0], floor_pos=[0.0, 0.0],
        ang=[5.0, 5.0], pers_ang=[9.0, 9.0], floor_ang=[1.0, 1.0],
    )
    position = script.band_report(result, "position")
    angle = script.band_report(result, "angle")
    assert position["band_median"] == pytest.approx(20.0)
    assert angle["band_median"] == pytest.approx(8.0)
    assert position["gap_mean"] == pytest.approx(0.5)
    assert angle["gap_mean"] == pytest.approx(0.5)


def test_band_report_counts_floor_above_persistence_for_angle():
    result = _result(
        pos=[1.0], pers_pos=[2.0], floor_pos=[0.0],
        ang=[1.0], pers_ang=[2.0], floor_ang=[3.0],   # floor ABOVE persistence
    )
    assert script.band_report(result, "position")["steps_floor_above_persistence"] == 0
    assert script.band_report(result, "angle")["steps_floor_above_persistence"] == 1


def test_band_report_counts_nan_gap_steps_for_angle():
    """A non-positive band yields NaN; the count must be reported, not hidden."""
    result = _result(
        pos=[1.0, 1.0], pers_pos=[2.0, 2.0], floor_pos=[0.0, 0.0],
        ang=[1.0, 1.0], pers_ang=[2.0, 2.0], floor_ang=[2.0, 0.0],  # first band == 0
    )
    angle = script.band_report(result, "angle")
    assert angle["gap_finite"] == 1
    assert angle["n_steps"] == 2


def test_band_report_flags_a_degenerate_positive_band_for_angle():
    """A band that is positive but numerically negligible makes the ratio
    meaningless -- measured, it produced values like 125.2 and 9.06."""
    result = _result(
        pos=[1.0], pers_pos=[2.0], floor_pos=[0.0],
        ang=[1.0], pers_ang=[2.0], floor_ang=[2.0 - 1e-9],
    )
    assert script.band_report(result, "angle")["steps_degenerate"] == 1


def test_band_report_rejects_an_unknown_metric():
    result = _result([1.0], [2.0], [0.0], [1.0], [2.0], [0.0])
    with pytest.raises(ValueError, match="position.*angle"):
        script.band_report(result, "velocity")


def test_print_report_emits_both_metrics(capsys):
    """The angle block must carry the same guards as the position block."""
    result = _result(
        pos=[1.0, 1.0], pers_pos=[2.0, 2.0], floor_pos=[0.0, 0.0],
        ang=[1.0, 1.0], pers_ang=[2.0, 2.0], floor_ang=[0.0, 0.0],
    )
    script.print_report(
        result,
        {"position": script.band_report(result, "position"),
         "angle": script.band_report(result, "angle")},
    )
    out = capsys.readouterr().out
    for metric in ("position", "angle"):
        assert f"{metric}_band_width" in out
        assert f"{metric}_gap_closed" in out
        assert f"{metric}_steps_with_floor_above_persistence" in out


def test_degenerate_threshold_is_pinned_at_one_part_in_a_thousand():
    """The threshold's magnitude is load-bearing, not just its existence.

    Its docstring justifies 1e-3 by measured gap_closed values of 125.2, 9.06
    and -0.68 at a relative band around 1e-4. Nothing else in this file pins
    the number itself -- `DEGENERATE = 1.0` passes every other test here."""
    assert script.DEGENERATE == 1e-3


def test_band_report_does_not_flag_a_healthy_band_as_degenerate():
    """A healthy band (relative width 0.24, the measured median at 20k steps)
    must NOT be counted as degenerate. Under `DEGENERATE = 1.0` this band
    (0.24 < 1.0) would wrongly be flagged, even though the threshold-value
    test above already catches that mutation on its own."""
    result = _result(
        pos=[1.0], pers_pos=[2.0], floor_pos=[0.0],
        ang=[1.0], pers_ang=[100.0], floor_ang=[76.0],  # band=24, relative=0.24
    )
    assert script.band_report(result, "angle")["steps_degenerate"] == 0


def test_band_report_final_fields_read_the_last_horizon_step():
    """final_model, final_persistence, final_floor and gap_final must all read
    index [-1] -- the LAST horizon step -- not [0]. Every curve here differs
    from its first to its last element so an [0]-vs-[-1] swap cannot hide."""
    result = _result(
        pos=[1.0, 2.0, 3.0], pers_pos=[10.0, 20.0, 40.0], floor_pos=[0.0, 1.0, 2.0],
        ang=[5.0, 6.0, 9.0], pers_ang=[50.0, 60.0, 90.0], floor_ang=[0.0, 5.0, 10.0],
    )
    position = script.band_report(result, "position")
    angle = script.band_report(result, "angle")

    assert position["final_model"] == pytest.approx(3.0)
    assert position["final_persistence"] == pytest.approx(40.0)
    assert position["final_floor"] == pytest.approx(2.0)
    assert position["gap_final"] == pytest.approx((40.0 - 3.0) / (40.0 - 2.0))

    assert angle["final_model"] == pytest.approx(9.0)
    assert angle["final_persistence"] == pytest.approx(90.0)
    assert angle["final_floor"] == pytest.approx(10.0)
    assert angle["gap_final"] == pytest.approx((90.0 - 9.0) / (90.0 - 10.0))


def test_print_report_prints_each_metrics_own_numbers(capsys):
    """The angle block must print angle's own numbers, not position's.

    Gives position and angle numerically distinct curves so a hardcoded
    `r = reports["position"]` inside the loop -- which keeps the labels
    correct but substitutes position's values under the angle heading --
    is caught even though both blocks' labels still read correctly."""
    result = _result(
        pos=[1.0, 1.0], pers_pos=[2.0, 2.0], floor_pos=[0.0, 0.0],
        ang=[3.0, 3.0], pers_ang=[8.0, 8.0], floor_ang=[4.0, 4.0],
    )
    reports = {"position": script.band_report(result, "position"),
               "angle": script.band_report(result, "angle")}
    script.print_report(result, reports)
    out = capsys.readouterr().out
    angle_block = out.split("--- angle")[1]

    assert f"rssm={3.0:9.2f}" in angle_block
    assert f"persistence={8.0:9.2f}" in angle_block
    assert f"floor={4.0:9.2f}" in angle_block
    assert f"mean={1.25:+.4f}" in angle_block
    assert f"final={1.25:+.4f}" in angle_block

    assert f"rssm={1.0:9.2f}" not in angle_block
    assert f"persistence={2.0:9.2f}" not in angle_block
    assert f"floor={0.0:9.2f}" not in angle_block
    assert f"mean={0.5:+.4f}" not in angle_block
    assert f"final={0.5:+.4f}" not in angle_block


def test_band_report_counts_zero_band_as_floor_above_persistence():
    """A zero band means the floor exactly equals persistence -- still no
    headroom, and must be counted alongside a strictly negative band. This
    pins the `band <= 0` boundary against a weakening to `band < 0`."""
    result = _result(
        pos=[1.0], pers_pos=[2.0], floor_pos=[2.0],  # band == 0 exactly
        ang=[1.0], pers_ang=[2.0], floor_ang=[0.0],
    )
    assert script.band_report(result, "position")["steps_floor_above_persistence"] == 1


def test_band_report_flags_degeneracy_by_relative_size_not_absolute_size():
    """DEGENERATE is a share of persistence, not an absolute width -- a band
    of 5 next to a persistence error of 10000 is relatively tiny (0.05%) even
    though 5 is nowhere near DEGENERATE=1e-3 measured on its own. Mutating the
    check from `relative < DEGENERATE` to `band < DEGENERATE` passes every
    other test here (their bands and relatives are both tiny at once) but
    must fail this one."""
    result = _result(
        pos=[1.0], pers_pos=[2.0], floor_pos=[0.0],
        ang=[1.0], pers_ang=[10_000.0], floor_ang=[9_995.0],  # band=5, relative=5e-4
    )
    assert script.band_report(result, "angle")["steps_degenerate"] == 1
