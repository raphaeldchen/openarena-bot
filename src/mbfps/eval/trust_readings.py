"""Reading 1 and Reading 2 of the trust-horizon diagnostic, over pooled inputs.

The M3d spec (section 3) states both readings BEFORE the run. This module is
those rules and nothing else: pure functions over numbers that are already
pooled -- the z of a clustered contrast, the estimate of a ratio of medians,
a boundary flag, a survival curve -- so every rule can be tested on
fabricated inputs with each branch mutated one at a time, and the driver
(`scripts/trust_horizon.py`) only has to put the pooled numbers into
`ReadingOneInputs` and print what comes back.

Reading 1 asks whether the h=45 gate rewards slow drift. It is *supported*
only if three conditions hold pooled AND each holds within at least two of
the three seeds; any other outcome is *not supported*, *not testable* or
*unresolved*, and the verdict line names the rule that decided it. Nothing
here decides on a magnitude the caller did not pass: the bar `z_fam` is
`pooling.cluster_threshold(FAMILY, clusters)` computed by the caller from the
actual cluster count, and every comparison against it is STRICT -- a z equal
to the bar does not clear it, and a NaN or infinite z never clears it (the
ladder's `_responds` policy: an infinite z is a zero standard error, which is
a degenerate ruler, not an infinitely precise one).

Reading 2 attaches no verdict. It turns each arm's survival curve into
`H*_q` -- the largest h with `S(h) >= q`, an integer by construction because
it is read off the curve and never interpolated -- and takes `H*_min` over
the PROBE-FREE channel only, so a dead probe cannot set the horizon M4
designs around.

Every sentence of section 3.2 / 3.3 that is a rule is quoted above the
function that implements it.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np

from mbfps.eval.trust import trust_horizon

ARMS_ORDER: tuple[str, ...] = ("pixel_ae", "frozen_ssl", "random_vit")
"""`mbfps.utils.config.ARMS`, restated so this module imports nothing but
numpy and `trust`; the test pins the two equal."""
TREATMENT = "frozen_ssl"
CONTROL = "random_vit"
FAMILY = 8
"""Spec 3.1: "Reading 1 makes eight clustered contrasts (three pairwise
Δ(45), one cos(45), one held-out c(45) per fold, and the two-channel h×
control); `cluster_threshold(family=8, clusters=24)` is the bar `z_fam`
every one of them is read against." """
Q_PREREGISTERED = 0.75
Q_REPORTED: tuple[float, ...] = (0.5, 0.75, 0.9)
"""Spec 3.3: "q = 0.75 pre-registered, with q = 0.5 and 0.9 printed beside
it (`H*_0.5` is the old median reading)." """

_CHANNELS: tuple[str, ...] = ("probe", "free")
_CONDITION_NAMES: tuple[str, ...] = ("(i)", "(ii)", "(iii)")
_SEEDS_REQUIRED = 2
"""Spec 3.2: "each holds within at least two of the three seeds." """


# ---------------------------------------------------------------------------
# The inputs, reduced to what the rules read.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contrast:
    """What `pooling.paired_contrast` yields, reduced to what the rules read."""

    estimate: float
    se: float
    z: float
    n_windows: int


@dataclass(frozen=True)
class Ratio:
    """What `pooling.pool_ratio` yields, reduced."""

    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class ReadingOneInputs:
    """Every pooled number Reading 1 reads, at h = 45.

    `delta_contrast` is keyed by the three unordered pairs, in whichever
    orientation the caller built them; the estimate of key `(a, b)` is
    `a - b`, and `_pair_z` flips the sign when a pair is looked up the other
    way round. `per_seed` holds the same inputs computed within each seed
    alone (the leaf entries carry `per_seed=None`).
    """

    delta_contrast: dict[tuple[str, str], Contrast]
    ratio_probe: dict[str, Ratio]
    ratio_free: dict[str, Ratio]
    cosine_contrast: Contrast
    corrected_contrast_a: Contrast
    corrected_contrast_b: Contrast
    alpha_boundary: dict[str, bool]
    crossing_contrast_probe: Contrast
    crossing_contrast_free: Contrast
    per_seed: "dict[int, ReadingOneInputs] | None"


class Status(str, Enum):
    """The outcomes spec 3.2 names: "*Supported* only if (i), (ii) and (iii)
    all hold pooled and each holds within at least two of the three seeds;
    any other outcome is *not supported*, *not testable* or *unresolved*,
    written as such." The two unresolved outcomes are kept apart because
    they name different failures: one of the probe, one of the scale fit."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not supported"
    NOT_TESTABLE = "not testable"
    UNRESOLVED_PROBE = "unresolved through the probe"
    UNRESOLVED_ALPHA = "unresolved (alpha on the grid boundary)"


@dataclass(frozen=True)
class ConditionResult:
    """`holds` is True / False / None, where None is undecided, undecidable
    or unreadable -- never folded into False, because the spec's statuses
    tell them apart."""

    name: str
    holds: bool | None
    detail: str


@dataclass(frozen=True)
class ReadingOne:
    status: Status
    conditions: tuple[ConditionResult, ...]
    best_delta_arm: str | None
    least_moving_arm: str | None
    per_seed_agreement: dict[str, int]
    reason: str


# ---------------------------------------------------------------------------
# Small shared pieces.
# ---------------------------------------------------------------------------


def _fmt(value: float, spec: str) -> str:
    """NaN and inf print as themselves, never as a number that looks measured."""
    return format(value, spec) if np.isfinite(value) else str(value)


def _clears(z: float, z_fam: float) -> bool:
    """STRICTLY above the bar. NaN never clears (nothing to read); an
    infinite z never clears either -- `pooling._z` returns ±inf for a zero
    standard error, which is a degenerate ruler, and the ladder's
    `_responds` refuses it the same way."""
    return bool(np.isfinite(z) and z > z_fam)


def _pair_z(delta_contrast: dict[tuple[str, str], Contrast], a: str, b: str) -> float:
    """The z of the Δ(45) contrast oriented as `a - b`, whichever way the
    caller keyed the pair. A missing pair is refused by name rather than
    read as "does not clear"."""
    if (a, b) in delta_contrast:
        return delta_contrast[(a, b)].z
    if (b, a) in delta_contrast:
        return -delta_contrast[(b, a)].z
    raise KeyError(
        f"no Δ(45) contrast for the pair {(a, b)!r}; the table has {sorted(delta_contrast)}"
    )


def _argmin_arm(ratios: dict[str, Ratio]) -> str | None:
    """The arm with the smallest ratio estimate. None when the smallest is
    not defined: a NaN estimate cannot be compared, and an exact tie has no
    single least-moving arm."""
    estimates = {arm: float(ratios[arm].estimate) for arm in ARMS_ORDER}
    if any(not np.isfinite(v) for v in estimates.values()):
        return None
    smallest = min(estimates.values())
    arms = [arm for arm, v in estimates.items() if v == smallest]
    return arms[0] if len(arms) == 1 else None


def _alpha_unreadable(inputs: ReadingOneInputs) -> list[str]:
    """Spec 3.2 (iii): "unreadable ... if either arm's α is on the grid
    boundary on either fold" -- EITHER ARM means the two arms in the
    contrast, the treatment and the control; `pixel_ae`'s boundary flag is
    carried for the record and decides nothing here."""
    return [arm for arm in (TREATMENT, CONTROL) if inputs.alpha_boundary[arm]]


# ---------------------------------------------------------------------------
# Reading 1 -- does the h=45 gate reward slow drift?
# ---------------------------------------------------------------------------


def best_delta_arm(delta_contrast: dict[tuple[str, str], Contrast], z_fam: float) -> str | None:
    """Spec 3.2 (i): "The *best-Δ arm* is the arm whose pooled `Δ(45)`
    contrast against *each* of the other two has `z > z_fam`; if no arm
    clears both, (i) is *undecidable at this precision*."

    Two arms cannot both clear both: `a - b` and `b - a` have opposite
    signs. Two winners can only come from a table keyed in BOTH
    orientations with inconsistent values, and that is refused rather than
    resolved by ARMS_ORDER.
    """
    winners = [
        arm
        for arm in ARMS_ORDER
        if all(_clears(_pair_z(delta_contrast, arm, other), z_fam) for other in ARMS_ORDER if other != arm)
    ]
    if len(winners) > 1:
        raise ValueError(
            f"{winners} each clear the Δ(45) contrast against every other arm; "
            "the contrast table carries both orientations of a pair with inconsistent values"
        )
    return winners[0] if winners else None


def least_moving_arm(ratio_probe: dict[str, Ratio], ratio_free: dict[str, Ratio]) -> str | None:
    """Spec 3.2 (i): "The *least-moving arm* is the arm with the smallest
    pooled `R_probe(45)` -- and it must be the same arm under `R_free(45)`,
    or (i) is *unresolved through the probe*." """
    probe_arm = _argmin_arm(ratio_probe)
    free_arm = _argmin_arm(ratio_free)
    if probe_arm is None or probe_arm != free_arm:
        return None
    return probe_arm


def condition_i(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ConditionResult:
    """Spec 3.2 (i): "The arm that moves least is the arm the gate likes
    best. ... (i) holds iff the least-moving arm is the best-Δ arm." `h`
    only names the step in the detail."""
    pairs = ", ".join(
        f"Δ {a}-{b} z {_fmt(_pair_z(inputs.delta_contrast, a, b), '+.2f')}"
        for a, b in combinations(ARMS_ORDER, 2)
    )
    best = best_delta_arm(inputs.delta_contrast, z_fam)
    if best is None:
        return ConditionResult(
            "(i)",
            None,
            f"undecidable at this precision: no arm's Δ({h}) contrast clears "
            f"z_fam {_fmt(z_fam, '.2f')} against each of the other two ({pairs})",
        )
    ratios = ", ".join(
        f"{arm} R_probe {_fmt(inputs.ratio_probe[arm].estimate, '.3f')} "
        f"R_free {_fmt(inputs.ratio_free[arm].estimate, '.3f')}"
        for arm in ARMS_ORDER
    )
    least = least_moving_arm(inputs.ratio_probe, inputs.ratio_free)
    if least is None:
        probe_arm = _argmin_arm(inputs.ratio_probe) or "undefined"
        free_arm = _argmin_arm(inputs.ratio_free) or "undefined"
        return ConditionResult(
            "(i)",
            None,
            f"unresolved through the probe: the least-moving arm is {probe_arm} under "
            f"R_probe({h}) but {free_arm} under R_free({h}) ({ratios})",
        )
    holds = least == best
    return ConditionResult(
        "(i)",
        holds,
        f"best-Δ arm {best} ({pairs}); least-moving arm {least} under both channels "
        f"({ratios}): {'the same arm' if holds else 'different arms'}",
    )


def condition_ii(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ConditionResult:
    """Spec 3.2 (ii): "The treatment moves the right way.
    `paired_contrast(frozen_ssl − random_vit)` on per-window seed-mean
    `cos(45)` has `z > z_fam`." """
    c = inputs.cosine_contrast
    holds = _clears(c.z, z_fam)
    return ConditionResult(
        "(ii)",
        holds,
        f"cos({h}) {TREATMENT}-{CONTROL}: estimate {_fmt(c.estimate, '+.3f')}, "
        f"se {_fmt(c.se, '.3f')}, z {_fmt(c.z, '+.2f')} {'>' if holds else 'not >'} "
        f"z_fam {_fmt(z_fam, '.2f')} (windows {c.n_windows})",
    )


def condition_iii(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ConditionResult:
    """Spec 3.2 (iii): "Take the magnitude out and the gate ranking goes
    away. On each fold, `paired_contrast(frozen_ssl − random_vit)` on the
    held-out `c_w(45)`: holds iff `z < −z_fam` (the treatment's corrected
    error is smaller); fails iff `z > z_fam`; *undecided* otherwise. (iii)
    holds only if it holds on both folds, and is *unreadable* -- Reading 1
    *unresolved* -- if either arm's α is on the grid boundary on either
    fold." """
    a, b = inputs.corrected_contrast_a, inputs.corrected_contrast_b
    folds = (
        f"held-out c({h}) {TREATMENT}-{CONTROL}: fold A z {_fmt(a.z, '+.2f')}, "
        f"fold B z {_fmt(b.z, '+.2f')}, bar ±{_fmt(z_fam, '.2f')}"
    )
    on_boundary = _alpha_unreadable(inputs)
    if on_boundary:
        return ConditionResult(
            "(iii)",
            None,
            f"unreadable: alpha on the grid boundary at h={h} for {', '.join(on_boundary)} ({folds})",
        )
    holds_a, holds_b = _clears(-a.z, z_fam), _clears(-b.z, z_fam)
    if holds_a and holds_b:
        return ConditionResult("(iii)", True, f"holds on both folds ({folds})")
    if _clears(a.z, z_fam) or _clears(b.z, z_fam):
        return ConditionResult(
            "(iii)", False, f"fails: the treatment's corrected error is larger on a fold ({folds})"
        )
    which = "fold A" if holds_a else "fold B" if holds_b else "neither fold"
    return ConditionResult(
        "(iii)", None, f"undecided: {which} clears -z_fam, both are required ({folds})"
    )


def probe_control(inputs: ReadingOneInputs, z_fam: float) -> bool:
    """Spec 3.2, probe control: "The paired per-draw difference of `h×`
    (frozen_ssl − random_vit) in each channel: if both channels clear
    `z_fam` with *opposite* signs, Reading 1 is *unresolved through the
    probe* whatever (i)–(iii) said. A tie, or an interval covering 0 in
    either channel, contradicts nothing." True means it fires."""
    zp, zf = inputs.crossing_contrast_probe.z, inputs.crossing_contrast_free.z
    both_clear = _clears(abs(zp), z_fam) and _clears(abs(zf), z_fam)
    return bool(both_clear and (zp > 0) != (zf > 0))


def _conditions(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> tuple[ConditionResult, ...]:
    return (condition_i(inputs, z_fam, h), condition_ii(inputs, z_fam, h), condition_iii(inputs, z_fam, h))


def reading_one(inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> ReadingOne:
    """Spec 3.2: "*Supported* only if (i), (ii) and (iii) all hold pooled
    **and** each holds within at least two of the three seeds; any other
    outcome is *not supported*, *not testable* or *unresolved*, written as
    such." And: "(ii) failing alone is the informative failure: the prior
    does not drift slowly, it moves wrong."

    The order of precedence: the probe control fires -> UNRESOLVED_PROBE;
    (iii) unreadable -> UNRESOLVED_ALPHA; (i) undecidable (no best-Δ arm)
    -> NOT_TESTABLE; (i) unresolved through the probe (the least-moving arm
    differs between channels) -> UNRESOLVED_PROBE; then the three conditions
    pooled and per seed -> SUPPORTED or NOT_SUPPORTED. Every condition is
    computed and returned whatever decided the status, so the printout can
    show what (i)-(iii) said. `h` names the step in the details and decides
    nothing; the caller (Task 6) passes the run's horizon.
    """
    if inputs.per_seed is None:
        raise ValueError(
            "reading_one needs the per-seed inputs (ReadingOneInputs.per_seed) to test the "
            "two-of-three rule; per_seed=None is only for a per-seed leaf"
        )
    conditions = _conditions(inputs, z_fam, h)
    best = best_delta_arm(inputs.delta_contrast, z_fam)
    least = least_moving_arm(inputs.ratio_probe, inputs.ratio_free)
    per_seed = {seed: _conditions(sub, z_fam, h) for seed, sub in sorted(inputs.per_seed.items())}
    n_seeds = len(per_seed)
    agreement = {
        name: sum(1 for conds in per_seed.values() if conds[k].holds is True)
        for k, name in enumerate(_CONDITION_NAMES)
    }
    i, ii, iii = conditions
    zp, zf = inputs.crossing_contrast_probe.z, inputs.crossing_contrast_free.z

    if probe_control(inputs, z_fam):
        status = Status.UNRESOLVED_PROBE
        reason = (
            f"probe control: the h× contrasts clear z_fam {_fmt(z_fam, '.2f')} with opposite "
            f"signs (probe z {_fmt(zp, '+.2f')}, free z {_fmt(zf, '+.2f')}), whatever (i)-(iii) said"
        )
    # Each condition's detail already begins with the word that names its
    # outcome ("unreadable: ...", "undecidable at this precision: ...",
    # "unresolved through the probe: ..."), so the reason is the condition's
    # name and its detail, never the word twice.
    elif _alpha_unreadable(inputs):
        status = Status.UNRESOLVED_ALPHA
        reason = f"{iii.name} {iii.detail}"
    elif best is None:
        status = Status.NOT_TESTABLE
        reason = f"{i.name} {i.detail}"
    elif least is None:
        status = Status.UNRESOLVED_PROBE
        reason = f"{i.name} {i.detail}"
    else:
        pooled_failed = [c.name for c in conditions if c.holds is not True]
        seeds_failed = [name for name, count in agreement.items() if count < _SEEDS_REQUIRED]
        if not pooled_failed and not seeds_failed:
            status = Status.SUPPORTED
            counts = ", ".join(f"{name} {agreement[name]}/{n_seeds}" for name in _CONDITION_NAMES)
            reason = (
                f"(i), (ii) and (iii) hold pooled and each in at least {_SEEDS_REQUIRED} of "
                f"{n_seeds} seeds ({counts})"
            )
        else:
            status = Status.NOT_SUPPORTED
            parts = []
            if pooled_failed:
                parts.append(f"pooled: {', '.join(pooled_failed)} not holding")
            if seeds_failed:
                parts.append(
                    "per seed: "
                    + ", ".join(
                        f"{name} holds in {agreement[name]} of {n_seeds} seeds" for name in seeds_failed
                    )
                    + f" ({_SEEDS_REQUIRED} of {n_seeds} required)"
                )
            reason = "; ".join(parts)
            if set(pooled_failed) | set(seeds_failed) == {"(ii)"}:
                reason += (
                    " -- (ii) failing alone is the informative failure: the prior does not "
                    "drift slowly, it moves wrong"
                )
    return ReadingOne(
        status=status,
        conditions=conditions,
        best_delta_arm=best,
        least_moving_arm=least,
        per_seed_agreement=agreement,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Reading 2 -- the horizon M4 designs around.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadingTwo:
    survival: dict[tuple[str, str], np.ndarray]
    horizons: dict[tuple[str, str, float], int]
    h_min: int


def reading_two(
    survival: dict[tuple[str, str], np.ndarray], qs: tuple[float, ...] = Q_REPORTED
) -> ReadingTwo:
    """Spec 3.3: "Per arm and per channel, the **survival curve** `S(h)` =
    the fraction of (window, seed) draws with `h× > h`, h = 0..45, over
    draws that moved. `H*_q` = the largest h with `S(h) ≥ q`; **q = 0.75
    pre-registered**, with q = 0.5 and 0.9 printed beside it. ... One
    derived number: **`H*_min` = min over the three arms of the probe-free
    `H*_0.75`** -- probe-independent, so a dead probe cannot set it. ... No
    verdict is attached."

    `H*_q` is `trust.trust_horizon`, read off the curve: an integer with no
    interpolation, so an even-count multiset of crossings never yields a
    half-step. A curve missing for an arm or channel is refused by name: a
    minimum over two arms would print as the minimum over three.
    """
    keys = [(arm, channel) for arm in ARMS_ORDER for channel in _CHANNELS]
    missing = [key for key in keys if key not in survival]
    if missing:
        raise KeyError(f"reading_two needs a survival curve per arm and channel; missing {missing}")
    if Q_PREREGISTERED not in qs:
        raise ValueError(
            f"the pre-registered q {Q_PREREGISTERED} must be among the reported qs, got {tuple(qs)}"
        )
    curves = {key: np.asarray(survival[key], dtype=float) for key in keys}
    horizons = {
        (arm, channel, float(q)): trust_horizon(curves[(arm, channel)], q)
        for arm, channel in keys
        for q in qs
    }
    h_min = min(horizons[(arm, "free", Q_PREREGISTERED)] for arm in ARMS_ORDER)
    return ReadingTwo(survival=curves, horizons=horizons, h_min=int(h_min))


# ---------------------------------------------------------------------------
# Printing, in the ladder's style: the numbers, then the verdict with the
# rule that decided it.
# ---------------------------------------------------------------------------


def _word(holds: bool | None) -> str:
    return "holds" if holds is True else "fails" if holds is False else "undecided"


def _condition_lines(conditions: tuple[ConditionResult, ...], indent: str) -> list[str]:
    return [f"{indent}{c.name:<6}{_word(c.holds):<10} {c.detail}" for c in conditions]


def format_reading_one(r: ReadingOne, inputs: ReadingOneInputs, z_fam: float, h: int = 45) -> str:
    """The pooled conditions with their details, the probe control, the
    same three conditions within each seed, the agreement counts, and the
    verdict with the rule it was decided by. Every detail is printed even
    when an earlier rule decided the status: the reader must be able to see
    what (i)-(iii) said under an unresolved verdict."""
    zp, zf = inputs.crossing_contrast_probe.z, inputs.crossing_contrast_free.z
    fires = probe_control(inputs, z_fam)
    lines = [
        f"--- Reading 1: does the h={h} gate reward slow drift?  "
        f"(z_fam {_fmt(z_fam, '.2f')}, family {FAMILY}, treatment {TREATMENT}, control {CONTROL})",
        "pooled:",
        *_condition_lines(r.conditions, "  "),
        f"  probe control: {'fires' if fires else 'does not fire'} "
        f"(h× {TREATMENT}-{CONTROL} per draw: probe z {_fmt(zp, '+.2f')}, free z {_fmt(zf, '+.2f')}; "
        f"fires iff both |z| > z_fam with opposite signs)",
    ]
    for seed, sub in sorted((inputs.per_seed or {}).items()):
        lines.append(f"seed {seed}:")
        lines.extend(_condition_lines(_conditions(sub, z_fam, h), "  "))
    n_seeds = len(inputs.per_seed or {})
    counts = ", ".join(f"{name} {r.per_seed_agreement[name]}/{n_seeds}" for name in _CONDITION_NAMES)
    lines.append(f"per-seed agreement: {counts}  ({_SEEDS_REQUIRED} of {n_seeds} required)")
    lines.append(
        f"best-Δ arm: {r.best_delta_arm or 'none'}; least-moving arm: {r.least_moving_arm or 'none'}"
    )
    lines.append(f"verdict: Reading 1 {r.status.value} -- decided by: {r.reason}")
    return "\n".join(lines)


def format_reading_two(r: ReadingTwo) -> str:
    """Per arm and channel: `H*_q` for every reported q, then `S(h)` at every
    h -- the table M4 cites is the curve itself. Then `H*_min` with the
    probe-based `H*_0.75` beside it so M4 can see whether the channels
    agree. No verdict."""
    qs = sorted({q for (_, _, q) in r.horizons})
    header = f"{'arm':<12}{'channel':<9}" + "".join(f"{f'H*_{q:g}':>9}" for q in qs) + "   S(h), h = 0..H"
    lines = [
        "--- Reading 2: the horizon M4 designs around  "
        f"(S(h) = fraction of moved draws with h× > h; H*_q = largest h with S(h) >= q; "
        f"q = {Q_PREREGISTERED:g} pre-registered; no verdict)",
        header,
    ]
    for arm in ARMS_ORDER:
        for channel in _CHANNELS:
            curve = " ".join(_fmt(s, ".2f") for s in r.survival[(arm, channel)])
            horizons = "".join(f"{r.horizons[(arm, channel, q)]:>9d}" for q in qs)
            lines.append(f"{arm:<12}{channel:<9}{horizons}   {curve}")
    probe_side = ", ".join(
        f"{arm} {r.horizons[(arm, 'probe', Q_PREREGISTERED)]}" for arm in ARMS_ORDER
    )
    lines.append(
        f"H*_min = {r.h_min}  (min over arms of the probe-free H*_{Q_PREREGISTERED:g}; "
        f"probe-based H*_{Q_PREREGISTERED:g} beside it: {probe_side})"
    )
    return "\n".join(lines)
