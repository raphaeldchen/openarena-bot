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

This module loads no checkpoint, touches no device and reads no file: every
number Reading H reports is a function of arrays, so it is testable without a
GPU. `scripts/prediction_burden.py` owns the model, the device and the record
schema.

IT IS NOT TORCH-FREE IN ITS IMPORT GRAPH, and that is deliberate. It imports
`probe.position_error` so that the position metric has ONE definition, and
`probe` imports torch at module level. Duplicating `position_error` here to
keep the import clean would put a second Euclidean distance in the codebase --
the kind of drift `evaluate_rollout`'s own comments record as having already
destroyed a signal once, when the model and the floor were probed through
differently fitted pipelines. One shared definition beats a clean import.
"""

from dataclasses import dataclass

import numpy as np

from mbfps.eval.pooling import episode_bootstrap, percentile_interval
from mbfps.eval.probe import position_error

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

    This IS the true one-step displacement -- ground truth, no model involved,
    which is what makes `motion_margin` a level rather than a difference of two
    estimates.

    NOT the recorded `persistence_position`. That copies the position at `t`,
    the last frame the OPEN LOOP saw, so comparing it with the k=1 rung -- which
    is re-grounded every step and has seen `h - 1` frames more -- would hand the
    rung a win on information advantage rather than on prediction. Spec 2.2
    forbids that comparison by name.
    """
    window_targets = np.asarray(window_targets, dtype=np.float64)
    scored = scored_targets(window_targets)
    return position_error(window_targets[:-1], scored)


def motion_margin(window_targets: np.ndarray, curve_one: np.ndarray) -> np.ndarray:
    """`one_step_persistence - the k=1 rung`, per horizon step.

    Positive means one prior step from a posterior-grounded state beats
    assuming the agent did not move.
    """
    baseline, curve_one = _checked_pair(
        one_step_persistence(window_targets), curve_one
    )
    return baseline - curve_one


def margin_interval(
    window_margin: np.ndarray,
    groups: np.ndarray,
    *,
    h: int,
    resamples: int = RESAMPLES,
    seed: int,
) -> tuple[float, float, float]:
    """`(point, ci_low, ci_high)` for the mean margin at horizon step `h`.

    THE RESAMPLING UNIT IS THE EPISODE. There are 229 windows over 24 episodes
    on every shipped cell, and consecutive Doom frames are near-duplicates, so
    a window-level bootstrap counts correlated observations as independent ones
    and returns an interval several times too narrow.

    `seed` IS KEYWORD-REQUIRED AND HAS NO DEFAULT. A defaulted seed is how a
    previous milestone shipped every cell drawing the same resamples; the
    point estimate is seed-free, so only the bounds can move, and a reader
    cannot tell nine identical draws from nine independent ones by looking.

    `groups` must carry one label per window. A record whose ladder carried no
    clustering stores `windows.episode` as null, and its own comment requires a
    reader to refuse rather than treat every window as its own episode --
    falling back to `arange(n)` here would convert this into the window-level
    bootstrap the first paragraph rules out.

    `window_margin` must be finite EVERYWHERE, not only in the column read at
    `h`. A NaN that reaches the mean returns `(nan, nan, nan)` -- no exception,
    no indication of which input was bad -- and that triple is the quantity
    Reading H's verdict is read from, so a silent one is a silent wrong verdict.
    """
    window_margin = np.asarray(window_margin, dtype=np.float64)
    groups = np.asarray(groups)
    if window_margin.ndim != 2:
        raise ValueError(
            f"window_margin must be (windows, horizon); got {window_margin.shape}"
        )
    if not np.isfinite(window_margin).all():
        raise ValueError("every window_margin value must be finite")
    if groups.ndim != 1 or groups.size != window_margin.shape[0]:
        raise ValueError(
            "groups must carry one label per window; got "
            f"{groups.shape} for {window_margin.shape[0]} windows"
        )
    if np.unique(groups).size < 2:
        raise ValueError(
            "an episode-clustered bootstrap needs at least two episodes; got "
            f"{np.unique(groups).size}"
        )
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")

    column = window_margin[:, h - 1] if 1 <= h <= window_margin.shape[1] else None
    if column is None:
        raise ValueError(
            f"horizon step must be in 1..{window_margin.shape[1]}, got {h}"
        )
    point = float(column.mean())
    replicates = np.array([
        float(column[index].mean())
        for index in episode_bootstrap(groups, resamples, seed)
    ])
    low, high, _se = percentile_interval(replicates)
    return point, low, high


SEEDS_MINIMUM: int = 3
"""An arm with fewer seeds is refused by name rather than tallied.

`strict_majority(1) == 1`, so without this a single lucky cell would establish
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


@dataclass(frozen=True)
class BurdenArm:
    """One cell: one arm at one seed, at `DECISION_H`."""

    arm: str
    seed: int
    margin: float
    margin_low: float
    margin_high: float
    burden_by_k: dict[int, float]
    compounding_by_k: dict[int, float]
    identity_residual: float
    open_loop_divergence: float
    k_one_is_floor: bool
    displacement_median: float
    floor_median: float
    clusters: int
    rows: int

    def __post_init__(self) -> None:
        if self.margin_low > self.margin_high:
            raise ValueError(
                f"{self.arm} seed {self.seed}: an interval cannot have "
                f"ci_low {self.margin_low} above ci_high {self.margin_high}"
            )

    @property
    def controls_ok(self) -> bool:
        """Every known answer hit. Checked before any status is tallied."""
        return (
            abs(self.identity_residual) <= IDENTITY_TOLERANCE
            and self.open_loop_divergence == 0.0
            and not self.k_one_is_floor
        )

    @property
    def base_ok(self) -> bool:
        """True motion exceeds the readout's own error, so a margin is
        detectable at all. Self-calibrating: a ratio of two measured
        quantities, not a magic constant."""
        return self.displacement_median > self.floor_median


@dataclass(frozen=True)
class BurdenInputs:
    cells: dict[tuple[str, int], BurdenArm]
    decision_h: int
    ks: tuple[int, ...]


@dataclass(frozen=True)
class BurdenStatus:
    status: str
    rule: str
    arms_motion: tuple[str, ...]
    arms_copies: tuple[str, ...]
    seeds_total: dict[str, int]


def _fall_through_rule(
    arms_motion: tuple[str, ...], arms_copies: tuple[str, ...],
    decision_h: int, n_arms: int,
) -> str:
    """The `INDETERMINATE` sentence, built from the two tallies it is a verdict on.

    THE CONDITION IS THAT NEITHER TALLY REACHED `ARMS_REQUIRED`, and nothing
    stronger. The first draft said "the motion_margin interval straddles 0",
    which holds only when no arm cleared; it was printed verbatim when one arm
    cleared motion alone and when two arms cleared in opposite directions, where
    the clearing arms' intervals do not straddle anything. So the sentence
    reports what DID clear, and names which of three situations this is: no arm
    clearing either way, the clearing arms all pointing one way but too few, or
    the arms splitting. The second and third are different results and the
    artefact has to let a reader tell them apart.

    The sentence ships in `burden.txt`; its wording is specification.
    """
    def tally(arms: tuple[str, ...]) -> str:
        return f"{len(arms)} of {n_arms} arms ({', '.join(arms) or 'none'})"

    if arms_motion and arms_copies:
        situation = (
            "the arms clear in opposite directions, so the reading is split, "
            "not merely short"
        )
    elif arms_motion or arms_copies:
        short = ARMS_REQUIRED - len(arms_motion or arms_copies)
        situation = (
            "every arm that clears does so in the same direction, and the "
            f"reading falls short of the bar by {short} arm(s)"
        )
    else:
        situation = (
            "no arm has a strict majority of its seeds clearing 0 in either "
            "direction"
        )
    return (
        f"neither decisive status reaches {ARMS_REQUIRED} arms at horizon "
        f"{decision_h}. An arm clears motion when the whole motion_margin "
        "interval is above 0, and copies when it is at or below 0, in a "
        f"strict majority of its seeds. Here motion is cleared by "
        f"{tally(arms_motion)} and copies by {tally(arms_copies)}: "
        f"{situation}. This status is the fall-through, not a bar that was "
        "cleared, so it arrived by default rather than by evidence and "
        "licenses no positive claim in either direction"
    )


def reading_burden(inputs: BurdenInputs) -> BurdenStatus:
    """Reading H: is the rollout compounding, or did the one-step map never
    learn motion?

    PRECEDENCE. `UNRESOLVED_CONTROL` outranks everything: a reading taken from
    an estimator that missed a known answer is not a weaker reading, it is not
    a reading. `UNREADABLE` comes next, because a cell where the agent barely
    moved would read COPIES for a reason that has nothing to do with the
    objective. Only then are the two decisive statuses tallied, and
    `INDETERMINATE` is the fall-through.

    THE TWO DECISIVE STATUSES CANNOT BOTH REACH THE BAR AT THREE ARMS. With
    `ARMS_REQUIRED = 2` and three arms, 2 + 2 > 3, and no arm sits in both
    tallies (an interval cannot be both above 0 and at or below 0). That is a
    fact about the arm COUNT, not about the statuses: with four arms, two
    clearing motion and two clearing copies is a contradiction, and an order of
    checks would resolve it by statement order rather than by evidence. So it
    raises, naming both sets. It is unreachable today and becomes reachable only
    if the arm count grows; if it does, the reading has to be redesigned, not
    patched with a preference.

    An EMPTY cell set raises too: it has no arm to read or to name, so neither
    the control check nor the short-arm refusal can catch it, and it would
    otherwise fall through to INDETERMINATE under a sentence about "0 arms".

    THE ASYMMETRY, AND WHERE IT HOLDS. Both decisive statuses are levels against
    an exactly known baseline -- the true one-step displacement -- measured on
    the same windows. That makes a reading in EITHER direction evidence ONLY IN A
    CELL THE BASE CONTROL ADMITS, where the median true one-step displacement
    exceeds the median floor error. The margin subtracts a probe-space quantity
    (the k=1 rung) from a ground-truth one that pays no readout error, so it
    carries the readout error with it: outside the gate a negative margin IS
    that error, and a perfect one-step predictor would read negative too. On M3m's
    own nine cells the gate refused all nine, and the margin sat within 0.1% to
    5.8% of the value a perfect predictor would have read (`## Task 7 results`,
    in the plan). That is why `UNREADABLE` outranks both statuses above, and why
    the sentences below are conditioned on it. Inside the gate this is the
    structural difference from M3k, whose statistic spoke in one direction only,
    and from M3l, whose `bits_carried` upper-bounds the joint and was
    trustworthy only below a cut.
    """
    if not inputs.cells:
        raise ValueError(
            "Reading H was given an empty cell set: there is no arm to read, "
            "so no status can be taken and none is returned"
        )
    by_arm: dict[str, list[BurdenArm]] = {}
    for cell in inputs.cells.values():
        by_arm.setdefault(cell.arm, []).append(cell)
    seeds_total = {arm: len(cells) for arm, cells in by_arm.items()}

    broken = sorted(
        f"{c.arm} seed {c.seed}" for c in inputs.cells.values() if not c.controls_ok
    )
    if broken:
        return BurdenStatus(
            status="UNRESOLVED_CONTROL",
            rule=(
                "a control with a known answer was missed in "
                f"{', '.join(broken)}: the identity residual must stay within "
                f"{IDENTITY_TOLERANCE:g}, the k=45 rung must reproduce the "
                "record bitwise (open_loop_divergence 0.0), and the k=1 rung "
                "must sit strictly above the floor. A reading taken from an "
                "estimator that missed a known answer is not a weaker reading, "
                "it is not a reading"
            ),
            arms_motion=(), arms_copies=(), seeds_total=seeds_total,
        )

    stationary = sorted(
        f"{c.arm} seed {c.seed}" for c in inputs.cells.values() if not c.base_ok
    )
    short = sorted(arm for arm, n in seeds_total.items() if n < SEEDS_MINIMUM)
    if stationary or short:
        causes = []
        if stationary:
            causes.append(
                "the median true one-step displacement does not exceed the "
                f"median floor error in {', '.join(stationary)}, so no method "
                "could detect motion prediction there"
            )
        if short:
            causes.append(
                "; ".join(
                    f"{arm} carries {seeds_total[arm]} seed(s), fewer than "
                    f"{SEEDS_MINIMUM}" for arm in short
                )
                + ", and an arm short of the minimum is refused by name rather "
                "than tallied"
            )
        return BurdenStatus(
            status="UNREADABLE", rule="; ".join(causes),
            arms_motion=(), arms_copies=(), seeds_total=seeds_total,
        )

    def clearing(predicate) -> tuple[str, ...]:
        return tuple(sorted(
            arm for arm, cells in by_arm.items()
            if sum(1 for c in cells if predicate(c)) >= strict_majority(len(cells))
        ))

    arms_motion = clearing(lambda c: c.margin_low > 0.0)
    arms_copies = clearing(lambda c: c.margin_high <= 0.0)

    if len(arms_motion) >= ARMS_REQUIRED and len(arms_copies) >= ARMS_REQUIRED:
        raise ValueError(
            f"PREDICTS_MOTION ({', '.join(arms_motion)}) and COPIES "
            f"({', '.join(arms_copies)}) each reach {ARMS_REQUIRED} arms at "
            f"horizon {inputs.decision_h}: the two decisive statuses "
            "contradict each other, and the reading cannot be taken. No arm "
            f"is in both tallies, so both reach {ARMS_REQUIRED} only with at "
            f"least {2 * ARMS_REQUIRED} arms, and there are {len(by_arm)}: "
            "the arm count has outgrown the bar"
        )

    if len(arms_motion) >= ARMS_REQUIRED:
        return BurdenStatus(
            status="PREDICTS_MOTION",
            rule=(
                f"the whole motion_margin interval clears 0 at horizon "
                f"{inputs.decision_h} in {len(arms_motion)} of "
                f"{len(by_arm)} arms ({', '.join(arms_motion)}), each in a "
                "strict majority of its seeds: one prior step from the true "
                "state beats assuming the agent did not move, so the one-step "
                "map predicts real motion and what fails is rolling it "
                "forward. A multi-step or overshooting objective is the "
                "indicated intervention"
            ),
            arms_motion=arms_motion, arms_copies=arms_copies, seeds_total=seeds_total,
        )
    if len(arms_copies) >= ARMS_REQUIRED:
        return BurdenStatus(
            status="COPIES",
            rule=(
                f"the whole motion_margin interval sits at or below 0 at "
                f"horizon {inputs.decision_h} in {len(arms_copies)} of "
                f"{len(by_arm)} arms ({', '.join(arms_copies)}), each in a "
                "strict majority of its seeds: one prior step from the TRUE "
                "state is no better than assuming stillness, and the base "
                "control passed in every cell (the median true one-step "
                "displacement exceeds the median floor error), so that "
                "margin is not only the readout error. A longer-horizon term "
                "cannot rescue this and the target itself must change"
            ),
            arms_motion=arms_motion, arms_copies=arms_copies, seeds_total=seeds_total,
        )
    return BurdenStatus(
        status="INDETERMINATE",
        rule=_fall_through_rule(
            arms_motion, arms_copies, inputs.decision_h, len(by_arm)
        ),
        arms_motion=arms_motion, arms_copies=arms_copies, seeds_total=seeds_total,
    )


READING_COLUMNS: tuple[str, ...] = (
    "arm", "seed", "margin", "ci_low", "ci_high", "burden_k", "comp_k", "clears",
)
"""`burden_k` and `comp_k` are `burden(k = ks[-1], h = decision_h)`: the rung of
the ladder is the LAST re-grounding period and the horizon is the decision
horizon, and both happen to be 45 in production. The labels carry neither
number, because a header is a constant and the values are not -- a label saying
`45` would not say which axis, and would lie the day `ks[-1]` changed. The
legend, which is built per call, states both."""
READING_WIDTHS: tuple[int, ...] = (13, 6, 11, 11, 11, 11, 11, 17)


def _row(values, widths) -> str:
    return "".join(f"{str(v):>{w}}" for v, w in zip(values, widths, strict=True))


def format_reading_burden(reading: BurdenStatus, inputs: BurdenInputs) -> str:
    """Reading H as `burden.txt` carries it, byte for byte.

    Every number in the caption and the legend is interpolated from the module
    -- the decision horizon, the identity tolerance, the arm bar -- so a
    constant that drifts cannot leave a stale literal behind. M3l shipped a
    legend saying "floor exactly 0" over a 1e-9 check and ~1e-13 values, and it
    is still an open follow-up.
    """
    lines = [
        f"--- Reading H: the prediction burden at horizon {inputs.decision_h} "
        f"(re-grounding periods k = {', '.join(str(k) for k in inputs.ks)})",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for (arm, seed), cell in sorted(inputs.cells.items()):
        clears = (
            "motion" if cell.margin_low > 0.0
            else "copies" if cell.margin_high <= 0.0
            else "-"
        )
        lines.append(_row(
            (
                arm, seed,
                f"{cell.margin:+.4f}", f"{cell.margin_low:+.4f}",
                f"{cell.margin_high:+.4f}",
                f"{cell.burden_by_k[inputs.ks[-1]]:+.4f}",
                f"{cell.compounding_by_k[inputs.ks[-1]]:+.4f}",
                clears,
            ),
            READING_WIDTHS,
        ))
    lines += [
        "  margin = the true one-step displacement minus the k=1 rung, so "
        "POSITIVE means one prior step from the true state beats assuming the "
        "agent did not move. The displacement is ground truth but the rung is "
        "read through the probe, so the margin carries the readout error: a "
        "reading in EITHER direction is evidence only where the base control "
        "passed, the median true one-step displacement exceeding the median "
        f"floor error at horizon {inputs.decision_h}, and below that gate a "
        "negative margin is the readout error",
        f"  {READING_COLUMNS[5]}/{READING_COLUMNS[6]} = "
        f"burden(k={inputs.ks[-1]}, h={inputs.decision_h}) and "
        f"compounding(k={inputs.ks[-1]}, h={inputs.decision_h}), where k is the "
        "re-grounding period, the last of those listed above, and h is the "
        "horizon step; compounding(k=1) is 0 by construction and the identity "
        f"residual is held within {IDENTITY_TOLERANCE:g}",
        f"  a status needs a strict majority of each arm's seeds in at least "
        f"{ARMS_REQUIRED} of {len(reading.seeds_total)} arms",
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}",
    ]
    return "\n".join(lines) + "\n"
