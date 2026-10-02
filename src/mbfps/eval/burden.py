"""The prediction-burden decomposition -- milestone M3m.

M3l refuted the bottleneck lever: the latent carries at most 1.11% of its
160-bit ceiling. That left the OBJECTIVE lever, standing by elimination. The
hypothesis inherited with it -- "the loss never asks for motion" -- is false as
stated: `kl_rate_above_free_bits` runs 0.746-0.976 across the nine cells, so
the dynamics prior trained on 75-98% of steps.

What survives is narrower. The loss asks for ONE-step latent agreement; the
gate reads 45-step rollout error. Evaluated at every horizon, `gap_closed(h)`
is positive in 8 of 9 cells at h=1 and 0 of 9 by h=20, so the failing criterion
is the tail of a curve that starts out working.

TWO AXES, NEVER CONFLATED. `k` is the RE-GROUNDING PERIOD -- how often
observation corrects the rollout, over `diagnostics.REGROUNDING_KS` -- and `h`
is the HORIZON STEP. Both were called `k` in an early draft of the spec. Write
`burden(k, h)`.

This module holds no torch and no I/O: every number Reading H reports is a
function of arrays, so it is testable without a checkpoint. `scripts/
prediction_burden.py` owns the model, the device and the record schema.
"""

import numpy as np

DECISION_H: int = 45
"""The horizon the status is read at -- the M3 gate's own horizon.

Not a free choice: `gap_closed(45)` is the criterion that has failed since M3c,
so a reading at any other horizon would answer a question the gate does not
ask. It must appear in `REPORTED_H`, or the status would be read at a horizon
the table never prints.
"""

REPORTED_H: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
"""The reporting grid, as raw levels, so a reader may apply a different horizon
rather than inheriting `DECISION_H`."""

ARMS_REQUIRED: int = 2
"""Of three. Matches every milestone from M3i onward."""

IDENTITY_TOLERANCE: float = 1e-9
"""The bound on the decomposition's floating-point residual.

`burden(k) = burden(1) + compounding(k)` is algebraic -- the floor cancels --
so at the magnitudes this milestone measures (floor in 100-250, curves monotone),
the residual is exactly 0.0. The three subtractions each fall in the Sterbenz
regime and are exact. At magnitudes outside this regime (e.g., curves near 1.7e8),
the residual is nonzero: 2.9802322387695312e-08 at that scale. This tolerance
bounds those wider regimes and leaves margin for other measurements.
"""

CONFIDENCE: float = 0.95
"""The interval's level. `pooling.percentile_interval` takes the 2.5/97.5
percentiles, so this constant DESCRIBES that function rather than configuring
it -- a test pins the two together, because a record that names a level the
estimator did not take is worse than a record that names none."""

RESAMPLES: int = 2000
"""Episode draws per interval. `pooling.pool_ratio` uses the same count."""


def at_horizon(curve: np.ndarray, h: int) -> float:
    """`curve` at horizon step `h`, ONE-INDEXED.

    The curves are 0-indexed arrays; every record and every sentence in this
    project counts horizon steps from 1. Indexing with `h` rather than `h - 1`
    shifts every reported number by one step and breaks no shape, which is why
    this is a function rather than a convention.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if not 1 <= h <= curve.size:
        raise ValueError(
            f"horizon step must be in 1..{curve.size}, got {h}"
        )
    return float(curve[h - 1])


def _checked_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both curves as finite float64 of equal length.

    NumPy broadcasts a length-1 array against a length-45 one without
    complaint, so a curve read from the wrong record key would yield a
    full-length result that is nonsense. The length check is the only thing
    standing between that and a plausible-looking table.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(
            f"the two curves must have the same length; got {left.shape} and {right.shape}"
        )
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("every curve value must be finite")
    return left, right


def burden(curve_k: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """`burden(k, .)` -- what predicting at re-grounding period `k` costs over
    the floor, per horizon step.

    The floor is a ZERO-step posterior at every step, seeing the frame it is
    scored on. It is not the k -> 0 limit of the ladder: k=1 is a one-step
    PRIOR from a state grounded one frame earlier, so k=1 must sit STRICTLY
    above the floor. `RegroundingSweep.is_bitwise_the_floor(1)` is the check,
    and it reads False on all nine shipped cells.
    """
    curve_k, floor = _checked_pair(curve_k, floor)
    return curve_k - floor


def compounding(curve_k: np.ndarray, curve_one: np.ndarray) -> np.ndarray:
    """`compounding(k, .)` -- what it costs that correction arrives every `k`
    steps rather than every step.

    Against the k=1 CURVE, not against the floor: the floor cancels out of
    `burden(k) - burden(1)`, and computing it against the floor instead would
    silently return `burden(k)` and make the k=1 control read nonzero.
    """
    curve_k, curve_one = _checked_pair(curve_k, curve_one)
    return curve_k - curve_one


def identity_residual(
    curve_k: np.ndarray, curve_one: np.ndarray, floor: np.ndarray
) -> float:
    """The largest absolute violation of
    `burden(k) == burden(1) + compounding(k)` over the horizon.

    Algebraic, so this measures floating point and nothing else -- which is
    exactly why it is worth recording: a nonzero residual beyond
    `IDENTITY_TOLERANCE` means one of the three curves is not what its name
    says, not that arithmetic failed.
    """
    whole = burden(curve_k, floor)
    parts = burden(curve_one, floor) + compounding(curve_k, curve_one)
    return float(np.max(np.abs(whole - parts)))
