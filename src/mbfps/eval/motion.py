"""M3i: does the posterior latent encode MOTION, or only absolute position?

M3g measured that the frame injects 0.290-0.489 nats beyond what the
teacher-forced prior already predicted, summed over all 32 groups, and that
copying the previous latent predicts the next one BETTER than the model's own
prior does (persist 0.824-0.866 against teacher 0.796-0.845). M3h then ruled
out the prior's sampling temperature. Between them they leave the three
remaining prior-side levers pulling on a stage with about four tenths of a nat
of headroom.

So this module asks a different question. The latent demonstrably carries
absolute position -- `latent_selection_r2` 0.18-0.34 across the M3c records --
and a latent that encodes "which corridor am I in" can score well on a
position probe while carrying nothing about step-to-step DISPLACEMENT. The M3
gate scores the imagined trajectory against PERSISTENCE, staying put, so a
latent without displacement cannot beat it whatever the prior does.

Everything here is pure: arrays in, arrays and readings out. No torch, no I/O,
no record schema. `scripts/latent_motion.py` owns all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mbfps.eval.pooling import cluster_threshold
from mbfps.eval.stages import entropy_by_group

# Pre-registered (spec 3.1, 3.2). Three arms, each contrast within a cell
# against that cell's own persistence baseline -- no arm is ranked against
# another, so the family is the arms and not their pairs.
MOTION_FAMILY: int = 3
SEEDS_REQUIRED: int = 2
ARMS_REQUIRED: int = 2

# Reported at every one of these; DECIDED only at `split_gap.DECISION_H`, the
# horizon every milestone since M3e has decided at.
K_REPORTED: tuple[int, ...] = (1, 5, 15, 30, 45)

# The permutation control's seed. Fixed so the control is reproducible: a
# control that draws a fresh permutation each run is a control whose refusal
# cannot be repeated.
CONTROL_SEED: int = 0


def motion_threshold(clusters: int) -> float:
    """The z Reading D must clear: the project's cluster-robust Bonferroni
    bar over `MOTION_FAMILY`, read against t(G-1) for G episode clusters.
    2.582 on the shipped 24-episode split."""
    return cluster_threshold(MOTION_FAMILY, clusters)


def displacement(positions, k: int) -> np.ndarray:
    """`p(t+k) - p(t)` per window, as a VECTOR in map units. `(n, 2)`.

    Taken from the window's first scored step, so every window contributes
    exactly one displacement at each k and the rows stay alignable with the
    per-window masks the pooling clusters on. A window shorter than `k + 1`
    steps is refused rather than truncated: a short window silently scored at
    a smaller k would be a different horizon pooled as if it were this one.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must be (n, steps, 2), got {positions.shape}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if positions.shape[1] < k + 1:
        raise ValueError(
            f"k={k} needs {k + 1} steps per window, got {positions.shape[1]}"
        )
    return positions[:, k, :] - positions[:, 0, :]


def contrast_series(predicted, true) -> np.ndarray:
    """Per window, `||true|| - ||predicted - true||`: how much closer the
    probe's displacement is than predicting no displacement at all. `(n,)`.

    Positive means the latent beat staying put. The persistence baseline is
    the ZERO prediction, so its error is `||true||` exactly -- which makes a
    probe that outputs zeros score exactly 0.0, the fixed point the reading is
    read against. Deliberately the same shape as `gap_closed`: a paired,
    per-window, model-against-persistence contrast, so `pooling.paired_contrast`
    reads it unchanged and clusters it on the same episodes.
    """
    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    if predicted.shape != true.shape:
        raise ValueError(
            f"predicted {predicted.shape} and true {true.shape} must be the same shape"
        )
    error_persist = np.linalg.norm(true, axis=-1)
    error_model = np.linalg.norm(predicted - true, axis=-1)
    return error_persist - error_model


@dataclass(frozen=True)
class MotionArm:
    """One arm's pooled contrast at one k, with the seed tallies BOTH ways.

    `seeds_down` is carried beside `seeds_up` because M3h shipped a Reading N
    table that printed only the up tally while its verdict was read from the
    down one: every row said `0/3` under a verdict asserting "3 of 3 arms",
    and the natural misreading was the opposite of the truth.
    """

    estimate: float
    se: float
    z: float
    seeds_up: int
    seeds_down: int
    seeds_total: int

    def clears_up(self, z_fam: float) -> bool:
        return self.z >= z_fam and self.seeds_up >= SEEDS_REQUIRED

    def clears_down(self, z_fam: float) -> bool:
        return self.z <= -z_fam and self.seeds_down >= SEEDS_REQUIRED

    def leaks(self, z_fam: float) -> bool:
        """For a CONTROL arm: cleared the bar in either direction. Two-sided
        on purpose -- a permuted pairing that is reliably worse than chance is
        as much a broken instrument as one that is better, and only the seed
        tallies are ignored here because a control has no result to replicate.
        """
        return abs(self.z) >= z_fam


@dataclass(frozen=True)
class MotionInputs:
    """Everything Reading D is decided on: the real arms, the permuted
    control, the bar, the horizon and the cluster count."""

    arms: dict[str, MotionArm]
    control: dict[str, MotionArm]
    z_fam: float
    k: int
    clusters: int


@dataclass(frozen=True)
class MotionStatus:
    status: str
    rule: str
    arms_up: tuple[str, ...]
    arms_down: tuple[str, ...]
    leaked: tuple[str, ...]


def reading_displacement(inputs: MotionInputs) -> MotionStatus:
    """Does the posterior latent encode displacement (spec 3.2)?

    Precedence, and it is the point: UNRESOLVED_CONTROL outranks every
    result. The control's pairing was permuted, so it cannot carry signal; a
    control that clears means the instrument is reading structure that does
    not exist, and the reading it would otherwise have printed is precisely
    the one not to trust. A suppressed reading reports no arms at all rather
    than reporting them beside a warning nobody reads.
    """
    z_fam = float(inputs.z_fam)
    leaked = tuple(sorted(a for a, arm in inputs.control.items() if arm.leaks(z_fam)))
    if leaked:
        return MotionStatus(
            status="UNRESOLVED_CONTROL",
            rule=(
                f"the permuted control cleared +/-{z_fam:.2f} in {', '.join(leaked)} at "
                f"k = {inputs.k}; the pairing it scores cannot carry signal, so the "
                f"instrument is reading structure that is not there and no reading is taken"
            ),
            arms_up=(), arms_down=(), leaked=leaked,
        )

    up = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_up(z_fam)))
    down = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_down(z_fam)))

    if len(up) >= ARMS_REQUIRED:
        return MotionStatus(
            status="MOTION_ENCODED",
            rule=(
                f"the latent beats staying put by more than +{z_fam:.2f} in "
                f"{len(up)} of {len(inputs.arms)} arms ({', '.join(up)}) at k = {inputs.k}, "
                f"each in at least {SEEDS_REQUIRED} of its seeds"
            ),
            arms_up=up, arms_down=down, leaked=(),
        )
    if not up and len(down) >= ARMS_REQUIRED:
        return MotionStatus(
            status="NO_MOTION",
            rule=(
                f"no arm beats staying put at k = {inputs.k}, and the latent is WORSE than "
                f"staying put by more than -{z_fam:.2f} in {len(down)} of {len(inputs.arms)} "
                f"arms ({', '.join(down)})"
            ),
            arms_up=(), arms_down=down, leaked=(),
        )
    return MotionStatus(
        status="NO_DIFFERENCE",
        rule=(
            f"no {ARMS_REQUIRED} arms clear +/-{z_fam:.2f} at k = {inputs.k} in at least "
            f"{SEEDS_REQUIRED} seeds each; the latent is indistinguishable from staying put"
        ),
        arms_up=up, arms_down=down, leaked=(),
    )


# The reading table's columns, in printed order. Declared once so a test can
# assert the header against THIS and the rows against the same index -- the
# caption-column pairing that three shipped defects on this project all broke.
READING_COLUMNS: tuple[str, ...] = (
    "arm", "estimate", "se", "z", "up", "dn", "clears",
)

# Field widths for READING_COLUMNS, in the same order. Kept beside the names
# so the header and the rows -- two separate f-strings -- cannot drift apart,
# and sized so a realistic value cannot equal or exceed its width and glue
# onto the previous column with no separator (the arm-name-overflow defect
# this project already shipped once, relocated to a numeric column: e.g. a
# 7-char "120/120" seed tally in what was a 6-wide field).
READING_WIDTHS: tuple[int, ...] = (12, 13, 8, 8, 8, 8, 8)


def format_reading_displacement(reading: MotionStatus, inputs: MotionInputs) -> str:
    """Reading D as it is printed and written to `motion.txt`, byte for byte.

    The caption names the reduction and the horizon, because a table whose
    header does not say what its columns hold is how this project has shipped
    a wrong number three times.
    """
    lines = [
        f"--- Reading D: does the latent encode displacement at k = {inputs.k} "
        f"(per-window ||true|| - ||predicted - true||, map units, against staying put); "
        f"z_fam = {inputs.z_fam:.2f} over {inputs.clusters} clusters ---",
        "  " + "".join(
            f"{name:>{width}}" for name, width in zip(
                READING_COLUMNS, READING_WIDTHS, strict=True,
            )
        ),
    ]
    w_arm, w_est, w_se, w_z, w_up, w_dn, w_clears = READING_WIDTHS
    for arm in sorted(inputs.arms):
        cell = inputs.arms[arm]
        if cell.clears_up(inputs.z_fam):
            clears = "up"
        elif cell.clears_down(inputs.z_fam):
            clears = "down"
        else:
            clears = "no"
        lines.append(
            f"  {arm:>{w_arm}}{cell.estimate:>{w_est}.4f}{cell.se:>{w_se}.4f}"
            f"{cell.z:>{w_z}.2f}"
            f"{f'{cell.seeds_up}/{cell.seeds_total}':>{w_up}}"
            f"{f'{cell.seeds_down}/{cell.seeds_total}':>{w_dn}}{clears:>{w_clears}}"
        )
    lines.append(
        "  control (permuted pairing, cannot carry signal): "
        + ", ".join(
            f"{arm} z={inputs.control[arm].z:+.2f}" for arm in sorted(inputs.control)
        )
    )
    lines.append(f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}")
    return "\n".join(lines) + "\n"


def latent_description(post_logits, prior_logits) -> dict:
    """The descriptive block (spec 2.3): what the latent looks like, with no
    threshold attached to any of it.

    Reported beside Reading D and deciding nothing. Every one of these is
    written into the record so the next milestone quotes an artefact rather
    than a scratch measurement -- which is exactly how M3h came to ship a
    premise that was never measured as stated.

    `live_groups` is the mean over windows of how many groups change argmax at
    least once within the window: a latent whose groups are mostly constant is
    far smaller than G x C in effect, however much capacity it nominally has.
    """
    post = np.asarray(post_logits, dtype=float)
    prior = np.asarray(prior_logits, dtype=float)
    if post.ndim != 4 or prior.ndim != 4:
        raise ValueError(
            f"expected (n, steps, groups, classes); got post {post.shape}, prior {prior.shape}"
        )
    entropy = entropy_by_group(post)
    modes = post.argmax(axis=-1)                       # (n, steps, groups)
    changes = (modes[:, 1:, :] != modes[:, :-1, :]).any(axis=1)   # (n, groups)
    return {
        "entropy_by_group": [float(x) for x in entropy],
        "entropy_mean": float(np.mean(entropy)),
        "entropy_max": float(np.log(post.shape[-1])),
        "live_groups": float(changes.sum(axis=-1).mean()),
        "top1_posterior": float(_top1_mass(post)),
        "top1_prior": float(_top1_mass(prior)),
    }


def _top1_mass(logits: np.ndarray) -> float:
    """Mean probability the most likely class in each group carries. 1/C for a
    uniform group, 1.0 for a sharp one."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return float(probs.max(axis=-1).mean())
