"""The prediction-burden ladder -- milestone M3m's decomposition, kept by M3n.

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

WHERE THE MOTION STATISTIC WENT. This module once also held M3m's
`motion_margin` -- the true one-step displacement minus the k=1 rung -- and
Reading H, the status read from it. Both were removed in M3n, and
`eval.headroom` is what replaced them: its three differences are all taken in
probe space, on the same windows, over one denominator. `motion_margin`
subtracted a probe-space model error (a median floor error of 88-224 map units on
M3m's nine cells) from a ground-truth displacement (3.97), so the readout error
dominated the difference 22.2x-56.5x, the base control failed 9 of 9 and Reading H
shipped UNREADABLE. It was removed rather than deprecated, because a live
function whose legend says a reading of -116 means COPIES is a standing trap,
and conditioning its legend was the right fix only while the records had to stay
readable. What is left is the ladder -- `burden`, `compounding`,
`identity_residual` and what they share -- which was right and which `headroom`
builds on: its `deficit` IS `burden`, and it imports `checked_pair` and
`strict_majority` from here. M3m's records still carry the `margin` they stored,
and `scripts/prediction_burden.py --phase read` prints it under a legend saying
it is superseded and why.

This module loads no checkpoint, touches no device and reads no file: every
number it returns is a function of arrays, so it is testable without a GPU.
`scripts/prediction_burden.py` owns the model, the device and the record schema.

IT IS NOT TORCH-FREE IN ITS IMPORT GRAPH, and that is deliberate. It imports
`probe.position_error` so that the position metric has ONE definition, and
`probe` imports torch at module level. Duplicating `position_error` here to
keep the import clean would put a second Euclidean distance in the codebase --
the kind of drift `evaluate_rollout`'s own comments record as having already
destroyed a signal once, when the model and the floor were probed through
differently fitted pipelines. One shared definition beats a clean import.
"""

import numpy as np

from mbfps.eval.probe import position_error

REPORTED_H: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
"""The reporting grid, as raw levels, so a reader may apply any horizon on it
rather than being handed one. `headroom.REPORTED_H` repeats it unchanged, so the
two milestones' tables sit on the same grid."""

IDENTITY_TOLERANCE: float = 1e-9
"""The bound on the decomposition's floating-point residual.

`burden(k) = burden(1) + compounding(k)` is algebraic -- the floor cancels --
so the residual is rounding and nothing else. It is exactly 0.0 whenever the
two summand differences, `curve_one - floor` and `curve_k - curve_one`, are
exact: their sum is then `curve_k - floor` in real arithmetic, which is what
`burden(k)` rounds. Only those two subtractions need be exact, not all three.

Sterbenz's lemma makes a subtraction exact when its operands are within a
factor of two. For monotone curves (floor <= curve_one <= curve_k) that covers
both summand differences whenever `curve_k <= 2 * floor`, so inside that ratio
the residual is exactly 0.0.

Above that ratio exactness is not guaranteed, and the lemma does not stretch to
cover a range: floor=106.90575077107698, curve_one=110.35124079639952,
curve_k=238.65922676171652 is monotone and inside 100-250, yet its ratio is 2.23
and its residual is 2.842170943040401e-14 -- 2**-45, one unit in the last place
at that magnitude. With every value in 100-250 the residual is three roundings
of at most 2**-46 each, so it cannot exceed 3 * 2**-46 = 4.3e-14, and 1e-9 sits
more than four orders of magnitude above that.

That is the regime this milestone reads, and it is not wholly inside the exact
one. Across the nine cells of `runs/m3_study_v2`, `max(k45_position) /
min(floor_position)` runs from 1.15 to 2.205, and `pixel_ae_seed0` at 2.205 is
outside it: 8 of its 45 steps at k=45 fall beyond the lemma's reach. The
residual read off that cell is still 0.0, as it is for every rung on all nine
cells -- but that is the data's doing, not a guarantee.

The tolerance does NOT bound wide-spread inputs. Curves near 1.7e8 over a floor
near 5e6 reach 2.9802322387695312e-08, about thirty times this tolerance;
`test_the_identity_residual_is_measured_and_its_tolerance_is_reachable` pins
exactly that.
"""


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


def checked_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both curves as finite float64 of equal length.

    NumPy broadcasts a length-1 array against a length-45 one without
    complaint, so a curve read from the wrong record key would yield a
    full-length result that is nonsense. The length check is the only thing
    standing between that and a plausible-looking table.

    PUBLIC because `eval.headroom` consumes it. A second shape guard in this
    codebase would be the wrong answer; `headroom`, `skill` and `deficit` all
    pass through this one.
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
    curve_k, floor = checked_pair(curve_k, floor)
    return curve_k - floor


def compounding(curve_k: np.ndarray, curve_one: np.ndarray) -> np.ndarray:
    """`compounding(k, .)` -- what it costs that correction arrives every `k`
    steps rather than every step.

    Against the k=1 CURVE, not against the floor: the floor cancels out of
    `burden(k) - burden(1)`, and computing it against the floor instead would
    silently return `burden(k)` and make the k=1 control read nonzero.
    """
    curve_k, curve_one = checked_pair(curve_k, curve_one)
    return curve_k - curve_one


def identity_residual(
    curve_k: np.ndarray, curve_one: np.ndarray, floor: np.ndarray
) -> float:
    """The largest absolute violation of
    `burden(k) == burden(1) + compounding(k)` over the horizon.

    Algebraic, so on the magnitudes this milestone measures it reads floating
    point and nothing else. A residual beyond `IDENTITY_TOLERANCE` there means
    one of the three curves is not what its name says.

    THAT READING IS CONDITIONAL ON THE MAGNITUDES, and the condition is the one
    `IDENTITY_TOLERANCE` documents: it holds while the operand pairs stay within
    a factor of two, where the worst residual measured is 4.3e-14. On
    wide-spread inputs the cancellation alone exceeds the tolerance -- 2.98e-08
    at magnitudes near 1.7e8 -- so there a large residual IS arithmetic, and
    says nothing about the curves. Position errors run 100-250 units, four
    orders below that, which is why this is a usable control here and would not
    be on arbitrary data.
    """
    whole = burden(curve_k, floor)
    parts = burden(curve_one, floor) + compounding(curve_k, curve_one)
    return float(np.max(np.abs(whole - parts)))


def scored_targets(window_targets: np.ndarray) -> np.ndarray:
    """The rows the model is scored on, given the window's `horizon + 1` rows.

    THE SLICE, stated so the off-by-one is hard to write. `evaluate_rollout`
    scores horizon step `j + 1` against
    `privileged[start + context + 1 + j]`, and its comment records that this
    was verified with an oracle model rather than argued: with
    `start + context` instead, a PERFECT predictor carries a constant error of
    one step of true displacement at every horizon step. The caller therefore
    passes `privileged[start + context : start + need + 1]`, whose row 0 is the
    last CONTEXT frame -- so the scored rows are `[1:]`, and the baseline and
    the model are scored on the identical frames by construction.
    """
    window_targets = np.asarray(window_targets, dtype=np.float64)
    if window_targets.ndim != 2 or window_targets.shape[0] < 2:
        raise ValueError(
            "window_targets must be (horizon + 1, K) with at least two rows; "
            f"got {window_targets.shape}"
        )
    return window_targets[1:]


def one_step_persistence(window_targets: np.ndarray) -> np.ndarray:
    """The error of predicting each scored frame by the frame before it.

    This IS the true one-step displacement -- ground truth, no model involved.
    M3m subtracted the k=1 rung from it, and that statistic is gone (see the
    module docstring); it is read now only by the script's base control and
    recorded as a curve, and its median, `headroom.DISPLACEMENT_RECORDED_ONLY`, is
    used by no statistic.

    NOT the recorded `persistence_position`. That copies the position at `t`,
    the last frame the OPEN LOOP saw, so comparing it with the k=1 rung -- which
    is re-grounded every step and has seen `h - 1` frames more -- would hand the
    rung a win on information advantage rather than on prediction. Spec 2.2
    forbids that comparison by name.
    """
    window_targets = np.asarray(window_targets, dtype=np.float64)
    scored = scored_targets(window_targets)
    return position_error(window_targets[:-1], scored)


SEEDS_MINIMUM: int = 3
"""The seed bar M3m's Reading H refused an arm below, by name, rather than
tallying it. Nothing in this module enforces it now that Reading H is gone;
`headroom.SEEDS_MINIMUM` is the copy M3n's reading enforces.

`strict_majority(1) == 1`, so without a bar a single lucky cell would establish
an arm -- the M3j trap where `--arms random_vit` printed a row reading
`clears = up` beside a verdict of NO DIFFERENCE.
"""


def strict_majority(n: int) -> int:
    """More than half of `n`, COMPUTED from the seed count present.

    Never a stored constant: `SEEDS_REQUIRED = 2` is a majority at 3 seeds and
    a minority at 5, and M3l's design was reworked for exactly that.
    """
    if n < 1:
        raise ValueError(f"a majority needs at least one seed, got {n}")
    return n // 2 + 1
