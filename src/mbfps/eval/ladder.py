"""The checkpoint ladder's pure rules -- milestone M3f.

Three things, none of which touches torch or a file:

  * spec 2.2 -- `primary_rung`: the rung nearest a cell's smoothed
    `embedding` minimum, ties to the earlier;
  * spec 2.4 -- `anchor_delta`: does the retrain's loss curve equal the
    original run's, step for step, and where does it first differ;
  * spec 3.3 -- Reading T: per arm, the paired rung contrast on the free
    channel against `z_fam`, its probe twin as the control, the replication
    across seeds, and one status with the sentence that decided it.

The contrasts arrive as `split_gap.StratumContrast` (estimate, se, z,
clusters) -- the same reduction of a pooled contrast Reading G reads -- built
by `scripts/checkpoint_ladder.py` from `pooling.paired_contrast`.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from mbfps.eval.split_gap import DECISION_H, StratumContrast, clears, fmt_z, train_held_passes_gate

RUNGS: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)
"""The checkpoint steps saved during the retrain (spec 2.1)."""
STEPS: int = 5000
"""The retrain's length; every rung lies at or before it."""
REFERENCE_RUNG: int = 20000
"""The M3c study's checkpoint, read from `--reference`, never retrained."""
# DECISION_H is M3e's, imported above: one decision horizon (15) for both
# readings, so `split_gap.decision_horizon` serves this script unchanged.
REPORTED_H: tuple[int, ...] = (5, 15, 45)
FAMILY: int = 6
"""Reading T's family: three arms x two channels at one horizon (spec 3.1)."""
SEEDS_REQUIRED: int = 2
R2_SENSITIVITY: float = 0.1
CURVE_WINDOW: int = 100
"""The moving-mean width the primary rung's minimum is read at -- M3e's."""
OBJECTIVE_BATCHES: int = 50
"""Validation draws per rung for `val_objective` (spec 2.3)."""

gate_passes = train_held_passes_gate
"""Spec 3.3's gate line: `gap_closed(45)` on position > 0 in EVERY seed, a
NaN never > 0 -- the same predicate Reading G applied to `train_held`, under
the name this reading uses."""


def primary_rung(min_step: int, rungs=RUNGS) -> int:
    """The rung nearest `min_step`; ties go to the EARLIER rung; a minimum
    before the first rung reads the first and one beyond the last reads the
    last (spec 2.2). `rungs` need not be sorted."""
    ordered = sorted(int(r) for r in rungs)
    if not ordered:
        raise ValueError("primary_rung needs at least one rung")
    min_step = int(min_step)
    if min_step < 1:
        raise ValueError(f"min_step is a 1-based training step, got {min_step}")
    best = ordered[0]
    for rung in ordered[1:]:
        if abs(rung - min_step) < abs(best - min_step):
            best = rung
    return best


def anchor_delta(retrain_loss, reference_loss, steps: int) -> tuple[float, int | None]:
    """`max |retrain - reference|` over the first `steps` per-step losses and
    the 1-based step at which they first differ, or `(0.0, None)` when the
    two prefixes are equal element for element. Equality is exact: a NaN on
    either side is a difference (NaN != NaN) and is reported as an infinite
    delta. Either history shorter than `steps` is refused."""
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    a = np.asarray(list(retrain_loss)[:steps], dtype=float)
    b = np.asarray(list(reference_loss)[:steps], dtype=float)
    if a.size < steps or b.size < steps:
        raise ValueError(
            f"a history is shorter than the {steps} steps the anchor covers: retrain "
            f"{a.size}, reference {b.size}"
        )
    differing = ~(a == b)
    if not differing.any():
        return 0.0, None
    first = int(np.argmax(differing)) + 1
    diff = np.abs(a - b)
    if not np.isfinite(diff[differing]).all():
        return float("inf"), first
    return float(diff.max()), first


# ---------------------------------------------------------------------------
# Reading T -- the timing hypothesis (spec 3.3), over pooled inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmInputs:
    """One arm's paired rung contrasts at the decision horizon
    (`primary - reference`), the primary rung each seed was read at, and the
    same inputs within each seed alone (leaves carry `per_seed=None`)."""

    t_free: StratumContrast
    t_probe: StratumContrast
    primary_rungs: dict[int, int]
    per_seed: "dict[int, ArmInputs] | None"


@dataclass(frozen=True)
class TimingInputs:
    arms: dict[str, ArmInputs]
    z_fam: float
    h: int


class Status(str, Enum):
    """Spec 3.3's four outcomes, in the table's order of precedence."""

    UNRESOLVED_PROBE = "unresolved through the probe"
    EARLIER_BETTER = "earlier better"
    EARLIER_WORSE = "earlier worse"
    NO_DIFFERENCE = "no difference"


@dataclass(frozen=True)
class ArmReading:
    """`seeds_up` / `seeds_down` / `seeds_total`: the replication the status
    was decided on. A pooled reading (`per_seed` a dict, possibly empty)
    counts that dict's clearing leaves; a single-seed leaf read alone
    (`per_seed=None`) has a vacuous replication clause -- `seeds_total = 1`,
    `seeds_up`/`seeds_down` 1 where its own contrast clears -- so a clearing
    leaf reads EARLIER_BETTER or EARLIER_WORSE on its own."""

    arm: str
    status: Status
    rule: str
    free_clears_up: bool
    free_clears_down: bool
    probe_clears_up: bool
    probe_clears_down: bool
    seeds_up: int
    seeds_down: int
    seeds_total: int


@dataclass(frozen=True)
class TimingReading:
    arms: dict[str, ArmReading]
    h: int
    z_fam: float


def _arm_reading(arm: str, a: ArmInputs, z_fam: float) -> ArmReading:
    """The rules of spec 3.3, in the table's precedence, each status carrying
    the sentence that decided it."""
    free_up, free_down = clears(a.t_free.z, z_fam), clears(-a.t_free.z, z_fam)
    probe_up, probe_down = clears(a.t_probe.z, z_fam), clears(-a.t_probe.z, z_fam)
    if a.per_seed is None:
        # A single-seed leaf, read alone: there is nothing to replicate across.
        seeds_up, seeds_down, seeds_total = int(free_up), int(free_down), 1
        replicated_up = replicated_down = True
        up_words = down_words = "this seed alone"
    else:
        seeds_up = sum(1 for leaf in a.per_seed.values() if clears(leaf.t_free.z, z_fam))
        seeds_down = sum(1 for leaf in a.per_seed.values() if clears(-leaf.t_free.z, z_fam))
        seeds_total = len(a.per_seed)
        replicated_up = seeds_up >= SEEDS_REQUIRED
        replicated_down = seeds_down >= SEEDS_REQUIRED
        up_words = f"{seeds_up} of {seeds_total} seeds"
        down_words = f"{seeds_down} of {seeds_total} seeds"
    fz, pz, bar = fmt_z(a.t_free.z), fmt_z(a.t_probe.z), f"{z_fam:.2f}"
    if (free_up and probe_down) or (free_down and probe_up):
        status = Status.UNRESOLVED_PROBE
        rule = f"T_free z {fz} and T_probe z {pz} both clear +-{bar} with opposite signs"
    elif free_up and replicated_up:
        status = Status.EARLIER_BETTER
        rule = f"T_free z {fz} > {bar} pooled and in {up_words}"
    elif free_down and replicated_down:
        status = Status.EARLIER_WORSE
        rule = f"T_free z {fz} < -{bar} pooled and in {down_words}"
    elif free_up:
        status = Status.NO_DIFFERENCE
        rule = (f"T_free z {fz} clears {bar} pooled but in only {up_words} "
                f"(>= {SEEDS_REQUIRED} required)")
    elif free_down:
        status = Status.NO_DIFFERENCE
        rule = (f"T_free z {fz} clears -{bar} pooled but in only {down_words} "
                f"(>= {SEEDS_REQUIRED} required)")
    else:
        status = Status.NO_DIFFERENCE
        rule = f"T_free z {fz} does not clear +-{bar}"
    return ArmReading(
        arm=arm, status=status, rule=rule,
        free_clears_up=free_up, free_clears_down=free_down,
        probe_clears_up=probe_up, probe_clears_down=probe_down,
        seeds_up=seeds_up, seeds_down=seeds_down, seeds_total=seeds_total,
    )


def reading_timing(inputs: TimingInputs) -> TimingReading:
    """One `ArmReading` per arm, in the caller's order; arms never read each
    other (spec 4: no ranking)."""
    return TimingReading(
        arms={arm: _arm_reading(arm, a, inputs.z_fam) for arm, a in inputs.arms.items()},
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_timing(reading: TimingReading, inputs: TimingInputs) -> str:
    """The contrast table, the primary rungs, and the verdict lines, in
    `split_gap.txt`'s style."""
    lines = [
        f"--- Reading T: the timing hypothesis at h={reading.h} (primary rung - "
        f"{REFERENCE_RUNG}, paired on the val windows); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'channel':<9}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for channel, c in (("free", a.t_free), ("probe", a.t_probe)):
            verdict = "yes" if clears(abs(c.z), reading.z_fam) else "no"
            lines.append(
                f"  {arm:<12}{channel:<9}{fmt_z(c.estimate, '+.4f'):>10}{fmt_z(c.se, '.4f'):>9}"
                f"{fmt_z(c.z):>8}  {verdict}"
            )
    lines.append("  primary rung per seed (the rung nearest the reference run's smoothed embedding minimum):")
    for arm, a in inputs.arms.items():
        rungs = " / ".join(f"s{seed} {rung}" for seed, rung in sorted(a.primary_rungs.items()))
        lines.append(f"    {arm:<12}{rungs}")
    for arm, r in reading.arms.items():
        lines.append(
            f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}"
        )
    return "\n".join(lines) + "\n"
