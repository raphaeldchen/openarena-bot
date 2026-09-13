"""Band statistics and degeneracy guards shared by every rollout consumer.

Both `scripts/eval_rollout.py` (a single run) and the nine-cell study job
(Task 4) consume `metric_summary` for "position" and "angle" alike. Keeping
two copies of this computation -- one in the script, one in the study job --
is exactly how the degeneracy threshold drifts between what a single run
reports and what the nine-cell study records, so this is the only copy of
either.
"""

import numpy as np

METRICS = ("position", "angle")

DEGENERATE = 1e-3
"""Below this share of the persistence error, a POSITIVE band is still junk.

`gap_closed` already returns NaN for a non-positive band, but a band that is
positive and merely tiny divides just fine and returns nonsense: measured on
an untrained model with a poor probe, a band ~1e-4 wide produced `gap_closed`
values of 125.2, 9.06 and -0.68. A non-positive band already returns NaN; a
numerically degenerate POSITIVE one does not, and is counted here so no
consumer reports the ratio as if it meant something.
"""


def metric_summary(result, metric: str) -> dict:
    """Band statistics and degeneracy counts for one metric.

    `metric` is "position" or "angle". Both get the same guards: an earlier
    version computed all of this for position only and printed the angle
    ratio bare, while that curve ranged +8.68 to -23.04 with a NaN step in
    it.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    model = getattr(result, f"rssm_{metric}")
    persistence = getattr(result, f"persistence_{metric}")
    floor = getattr(result, f"floor_{metric}")
    gap = getattr(result, f"{metric}_gap_closed")()

    band = persistence - floor
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(persistence == 0, np.nan, band / persistence)
    finite = np.isfinite(gap)
    return {
        "final_model": float(model[-1]),
        "final_persistence": float(persistence[-1]),
        "final_floor": float(floor[-1]),
        "band_min": float(band.min()),
        "band_median": float(np.median(band)),
        "band_max": float(band.max()),
        "relative_min": float(np.nanmin(relative)),
        "relative_median": float(np.nanmedian(relative)),
        "relative_max": float(np.nanmax(relative)),
        "steps_floor_above_persistence": int((band <= 0).sum()),
        "steps_degenerate": int(((band > 0) & (relative < DEGENERATE)).sum()),
        "gap_finite": int(finite.sum()),
        "gap_mean": float(np.nanmean(gap)) if finite.any() else float("nan"),
        "gap_min": float(np.nanmin(gap)) if finite.any() else float("nan"),
        "gap_max": float(np.nanmax(gap)) if finite.any() else float("nan"),
        "gap_final": float(gap[-1]),
        "n_steps": int(len(gap)),
    }
