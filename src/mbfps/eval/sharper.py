"""The sharper latent's pure rules -- milestone M3h.

Two readings, neither of which touches torch or a file:

  * spec 3.2 -- Reading N: over the nine shipped cells, does sampling the
    prior more sharply at rollout time keep MORE of the moved draws ahead of
    embedding-space persistence at h = 15? The status gates the retrain and,
    when it passes, names the temperature the retrain runs at.
  * spec 3.3 -- Reading M: does a model TRAINED at that temperature roll out
    better than the one it replaces?

The contrasts arrive as `split_gap.StratumContrast` (estimate, se, z,
clusters) -- the same reduction Readings G, T and S are decided on -- built
by `scripts/sharper_latent.py` from `pooling.paired_contrast`.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from mbfps.eval.split_gap import DECISION_H, StratumContrast, clears, fmt_z

TAU_GRID: tuple[float, ...] = (1.0, 0.7, 0.5, 0.3, 0.0)
"""Every rollout temperature the sweep evaluates, reference first."""
TAU_CANDIDATES: tuple[float, ...] = (0.7, 0.5, 0.3)
"""The temperatures the retrain may run at. 0.0 is an ENDPOINT, reported and
never a candidate: M3a measured the argmax collapsing the imagined trajectory
to 3 distinct latents out of 45 at roughly four times the position error, and
a model cannot be trained toward a sampler that draws nothing."""
REFERENCE_TAU: float = 1.0
"""What every M3b-M3g artefact was trained and evaluated at; the contrast's
control and the pass the self-check is taken on."""
STEPS: int = 20000
"""The retrain's length -- M3c's, so the only difference is the sampler."""
IDENTITY_STEPS: int = 500
"""The tau = 1.0 retrain that must reproduce the M3c loss prefix exactly."""
# DECISION_H is M3e's, imported: one decision horizon (15) across the study.
REPORTED_H: tuple[int, ...] = (5, 15, 45)
SWEEP_FAMILY: int = 9
"""Reading N's family: three arms x three candidate temperatures (spec 3.2)."""
RETRAIN_FAMILY: int = 6
"""Reading M's family: three arms x two channels (spec 3.3)."""
SEEDS_REQUIRED: int = 2
ARMS_REQUIRED: int = 2
"""Spec 3.2: a temperature counts when it clears in at least two ARMS. One arm
alone is a result about that arm, not about the sampler, and retraining nine
cells on it would be spending fifteen hours on a single cell's evidence."""


# ---------------------------------------------------------------------------
# Reading N -- is the rollout noise-limited? (spec 3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TauInputs:
    """One (arm, temperature): the pooled free-channel contrast against
    tau = 1.0 and the same contrast within each seed alone."""

    free: StratumContrast
    per_seed: dict[int, StratumContrast]


@dataclass(frozen=True)
class SweepArm:
    """One arm's inputs, keyed by temperature. The reference temperature is
    not a key: it is what every contrast is against."""

    taus: dict[float, TauInputs]


@dataclass(frozen=True)
class SweepInputs:
    arms: dict[str, SweepArm]
    z_fam: float
    h: int


class SweepStatus(str, Enum):
    """Spec 3.2's three outcomes."""

    NOISE_LIMITED = "noise limited"
    SHARPER_WORSE = "sharper worse"
    NOT_NOISE_LIMITED = "not noise limited"


@dataclass(frozen=True)
class ArmTau:
    """One cell of the sweep table: whether it cleared, which way, and in how
    many seeds. `counts` is the conjunction the arm tally is taken over --
    cleared pooled AND replicated -- so the table and the rule cannot
    disagree about what a cell contributed."""

    arm: str
    tau: float
    clears_up: bool
    clears_down: bool
    seeds_up: int
    seeds_down: int
    seeds_total: int

    @property
    def counts(self) -> bool:
        return self.clears_up and self.seeds_up >= SEEDS_REQUIRED

    @property
    def counts_against(self) -> bool:
        return self.clears_down and self.seeds_down >= SEEDS_REQUIRED


@dataclass(frozen=True)
class SweepReading:
    """`arms_clearing` maps every temperature to the arms that counted for it,
    in the caller's arm order; `cells` is the whole table, endpoint included."""

    cells: dict[tuple[str, float], ArmTau]
    arms_clearing: dict[float, tuple[str, ...]]
    arms_against: dict[float, tuple[str, ...]]
    tau_star: float | None
    status: SweepStatus
    rule: str
    h: int
    z_fam: float


def _cell(arm: str, tau: float, inputs: TauInputs, z_fam: float) -> ArmTau:
    leaves = list(inputs.per_seed.values())
    return ArmTau(
        arm=arm, tau=tau,
        clears_up=clears(inputs.free.z, z_fam),
        clears_down=clears(-inputs.free.z, z_fam),
        seeds_up=sum(1 for leaf in leaves if clears(leaf.z, z_fam)),
        seeds_down=sum(1 for leaf in leaves if clears(-leaf.z, z_fam)),
        seeds_total=len(leaves),
    )


def reading_noise(inputs: SweepInputs) -> SweepReading:
    """Spec 3.2, in one pass over the table: build every cell, tally the arms
    per candidate temperature, and read the status. `tau_star` is the
    candidate with the largest POOLED estimate among those counting in at
    least `ARMS_REQUIRED` arms, ties to the larger temperature."""
    cells = {
        (arm, tau): _cell(arm, tau, tau_inputs, inputs.z_fam)
        for arm, sweep_arm in inputs.arms.items()
        for tau, tau_inputs in sweep_arm.taus.items()
    }
    arms_clearing: dict[float, tuple[str, ...]] = {}
    arms_against: dict[float, tuple[str, ...]] = {}
    for tau in TAU_GRID:
        if tau == REFERENCE_TAU:
            continue
        arms_clearing[tau] = tuple(
            arm for arm in inputs.arms
            if (arm, tau) in cells and cells[(arm, tau)].counts
        )
        arms_against[tau] = tuple(
            arm for arm in inputs.arms
            if (arm, tau) in cells and cells[(arm, tau)].counts_against
        )
    qualifying = [t for t in TAU_CANDIDATES if len(arms_clearing.get(t, ())) >= ARMS_REQUIRED]
    against = [t for t in TAU_CANDIDATES if len(arms_against.get(t, ())) >= ARMS_REQUIRED]
    bar = f"{inputs.z_fam:.2f}"
    if qualifying:
        # Ties to the LARGER temperature: the milder intervention.
        def pooled(tau: float) -> float:
            estimates = [
                inputs.arms[arm].taus[tau].free.estimate for arm in arms_clearing[tau]
            ]
            return float(np.mean(estimates))

        best = max(pooled(t) for t in qualifying)
        tau_star = max(t for t in qualifying if pooled(t) == best)
        names = ", ".join(arms_clearing[tau_star])
        rule = (
            f"tau={tau_star:.1f} clears +{bar} pooled and in >= {SEEDS_REQUIRED} seeds in "
            f"{len(arms_clearing[tau_star])} of {len(inputs.arms)} arms ({names}); "
            f"pooled estimate {fmt_z(best, '+.4f')}"
        )
        return SweepReading(
            cells=cells, arms_clearing=arms_clearing, arms_against=arms_against,
            tau_star=tau_star, status=SweepStatus.NOISE_LIMITED, rule=rule,
            h=inputs.h, z_fam=inputs.z_fam,
        )
    if against:
        tau = against[0]
        names = ", ".join(arms_against[tau])
        rule = (
            f"no candidate temperature clears +{bar} in {ARMS_REQUIRED} arms, and tau={tau:.1f} "
            f"clears -{bar} in {len(arms_against[tau])} of {len(inputs.arms)} arms ({names})"
        )
        return SweepReading(
            cells=cells, arms_clearing=arms_clearing, arms_against=arms_against,
            tau_star=None, status=SweepStatus.SHARPER_WORSE, rule=rule,
            h=inputs.h, z_fam=inputs.z_fam,
        )
    best_tau = max(TAU_CANDIDATES, key=lambda t: len(arms_clearing.get(t, ())))
    counted = arms_clearing.get(best_tau, ())
    detail = (
        f"the most any candidate reaches is tau={best_tau:.1f} in {len(counted)} of "
        f"{len(inputs.arms)} arms ({', '.join(counted)})" if counted
        else f"no candidate temperature clears +{bar} pooled and in >= {SEEDS_REQUIRED} seeds in any arm"
    )
    rule = f"{ARMS_REQUIRED} arms required; {detail}"
    return SweepReading(
        cells=cells, arms_clearing=arms_clearing, arms_against=arms_against,
        tau_star=None, status=SweepStatus.NOT_NOISE_LIMITED, rule=rule,
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_noise(reading: SweepReading, inputs: SweepInputs) -> str:
    """The contrast table over the grid and the verdict, in `ladder.txt`'s style."""
    lines = [
        f"--- Reading N: is the rollout noise-limited at h={reading.h} (S_free(h) at tau minus "
        f"at tau={REFERENCE_TAU}, paired on the val windows); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'tau':>5}{'estimate':>11}{'se':>9}{'z':>8}{'seeds':>7}  clears  counts",
    ]
    for arm, sweep_arm in inputs.arms.items():
        for tau in TAU_GRID:
            if tau == REFERENCE_TAU or tau not in sweep_arm.taus:
                continue
            cell = reading.cells[(arm, tau)]
            contrast = sweep_arm.taus[tau].free
            direction = "up" if cell.clears_up else ("down" if cell.clears_down else "no")
            endpoint = "" if tau in TAU_CANDIDATES else "  (endpoint, never a candidate)"
            lines.append(
                f"  {arm:<12}{tau:>5.1f}{fmt_z(contrast.estimate, '+.4f'):>11}"
                f"{fmt_z(contrast.se, '.4f'):>9}{fmt_z(contrast.z):>8}"
                f"{f'{cell.seeds_up}/{cell.seeds_total}':>7}  {direction:<6}  "
                f"{'yes' if cell.counts else 'no'}{endpoint}"
            )
    lines.append(
        f"  verdict: {reading.status.name.replace('_', ' ')} -- decided by: {reading.rule}"
    )
    if reading.tau_star is not None:
        lines.append(f"  the retrain runs at SAMPLE_TEMPERATURE = {reading.tau_star:.1f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Reading M -- does training it sharper help? (spec 3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrainArm:
    """One arm's paired contrasts, `M3h at tau* minus M3c at 1.0`, on both
    channels, and the same within each seed alone (leaves carry
    `per_seed=None`)."""

    free: StratumContrast
    probe: StratumContrast
    per_seed: "dict[int, RetrainArm] | None"


@dataclass(frozen=True)
class RetrainInputs:
    arms: dict[str, RetrainArm]
    z_fam: float
    h: int
    tau: float


class RetrainStatus(str, Enum):
    """Spec 3.3's four outcomes, in the table's order of precedence."""

    UNRESOLVED_PROBE = "unresolved through the probe"
    SHARPER_BETTER = "sharper better"
    SHARPER_WORSE = "sharper worse"
    NO_DIFFERENCE = "no difference"


@dataclass(frozen=True)
class RetrainArmReading:
    arm: str
    status: RetrainStatus
    rule: str
    free_clears_up: bool
    free_clears_down: bool
    probe_clears_up: bool
    probe_clears_down: bool
    seeds_up: int
    seeds_down: int
    seeds_total: int


@dataclass(frozen=True)
class RetrainReading:
    arms: dict[str, RetrainArmReading]
    h: int
    z_fam: float
    tau: float


def _retrain_arm_reading(arm: str, a: RetrainArm, z_fam: float) -> RetrainArmReading:
    """Spec 3.3's rules in the table's precedence, each status carrying the
    sentence that decided it."""
    free_up, free_down = clears(a.free.z, z_fam), clears(-a.free.z, z_fam)
    probe_up, probe_down = clears(a.probe.z, z_fam), clears(-a.probe.z, z_fam)
    if a.per_seed is None:
        # A single-seed leaf, read alone: there is nothing to replicate across.
        seeds_up, seeds_down, seeds_total = int(free_up), int(free_down), 1
        replicated_up = replicated_down = True
        up_words = down_words = "this seed alone"
    else:
        seeds_up = sum(1 for leaf in a.per_seed.values() if clears(leaf.free.z, z_fam))
        seeds_down = sum(1 for leaf in a.per_seed.values() if clears(-leaf.free.z, z_fam))
        seeds_total = len(a.per_seed)
        replicated_up = seeds_up >= SEEDS_REQUIRED
        replicated_down = seeds_down >= SEEDS_REQUIRED
        up_words = f"{seeds_up} of {seeds_total} seeds"
        down_words = f"{seeds_down} of {seeds_total} seeds"
    fz, pz, bar = fmt_z(a.free.z), fmt_z(a.probe.z), f"{z_fam:.2f}"
    if (free_up and probe_down) or (free_down and probe_up):
        status = RetrainStatus.UNRESOLVED_PROBE
        rule = f"M_free z {fz} and M_probe z {pz} both clear +-{bar} with opposite signs"
    elif free_up and replicated_up:
        status = RetrainStatus.SHARPER_BETTER
        rule = f"M_free z {fz} > {bar} pooled and in {up_words}"
    elif free_down and replicated_down:
        status = RetrainStatus.SHARPER_WORSE
        rule = f"M_free z {fz} < -{bar} pooled and in {down_words}"
    elif free_up or free_down:
        words = up_words if free_up else down_words
        sign = "+" if free_up else "-"
        status = RetrainStatus.NO_DIFFERENCE
        rule = (f"M_free z {fz} clears {sign}{bar} pooled but in only {words} "
                f"(>= {SEEDS_REQUIRED} required)")
    else:
        status = RetrainStatus.NO_DIFFERENCE
        rule = f"M_free z {fz} does not clear +-{bar}"
    return RetrainArmReading(
        arm=arm, status=status, rule=rule,
        free_clears_up=free_up, free_clears_down=free_down,
        probe_clears_up=probe_up, probe_clears_down=probe_down,
        seeds_up=seeds_up, seeds_down=seeds_down, seeds_total=seeds_total,
    )


def reading_sharper(inputs: RetrainInputs) -> RetrainReading:
    """One reading per arm, in the caller's order; arms never read each other
    (spec 4: no ranking)."""
    return RetrainReading(
        arms={
            arm: _retrain_arm_reading(arm, a, inputs.z_fam) for arm, a in inputs.arms.items()
        },
        h=inputs.h, z_fam=inputs.z_fam, tau=inputs.tau,
    )


def format_reading_sharper(reading: RetrainReading, inputs: RetrainInputs) -> str:
    """The contrast table and the verdict lines, in `ladder.txt`'s style."""
    lines = [
        f"--- Reading M: the retrain at tau={reading.tau:.1f} against the M3c cells at "
        f"h={reading.h} (paired on the val windows); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'channel':<9}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for channel, contrast in (("free", a.free), ("probe", a.probe)):
            verdict = "yes" if clears(abs(contrast.z), reading.z_fam) else "no"
            lines.append(
                f"  {arm:<12}{channel:<9}{fmt_z(contrast.estimate, '+.4f'):>10}"
                f"{fmt_z(contrast.se, '.4f'):>9}{fmt_z(contrast.z):>8}  {verdict}"
            )
    for arm, r in reading.arms.items():
        lines.append(
            f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}"
        )
    return "\n".join(lines) + "\n"
