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
