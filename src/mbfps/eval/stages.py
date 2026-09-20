"""The stage decomposition's pure rules -- milestone M3g.

Two halves, neither of which touches torch or a file:

  * spec 2.3 -- the statistics: on the posterior over a window, the
    teacher-forced prior and the open-loop prior (the three logit arrays
    `diagnostics.keep_latents` keeps), the information the frame injects,
    the mode accuracies of the two priors against the posterior's next mode,
    the two baselines every accuracy is read against (latent persistence:
    the class does not change; the marginal: the most frequent class), the
    prior's NLL of the posterior's mode, the per-group entropy, and the
    free-channel margin at one horizon step;
  * spec 3.2 -- the reading: per arm, four stages in a fixed order, each a
    pooled contrast against `z_fam` replicated across seeds, and one status
    with the sentence that decided it (Task 3).

Everything reduces in float64 from whatever it is handed. The window layout
is the pass's: `post_logits[:, context - 1]` is t0, the last context frame,
and `post_logits[:, context - 1 + h]` is horizon step h; the two prior
arrays are indexed by h - 1.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum

from mbfps.eval.split_gap import DECISION_H, StratumContrast, clears, fmt_z
from mbfps.models.rssm import KL_FREE_BITS

FAMILY: int = 5
"""Spec 3.1: the five contrasts one arm's reading may consult -- encode,
predict (two), carry, decode. Each arm is its own fixed-sequence procedure."""
SEEDS_REQUIRED: int = 2
"""Spec 3.2: a stage passes pooled AND within >= 2 seeds."""
REPORTED_K: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 45)
"""The carry k-curve's printed columns, filtered to <= the run's horizon."""
STAGES: tuple[str, ...] = ("encode", "predict", "carry", "decode")
"""The order the stages are read in; the status is the first not passed."""

__all__ = [
    "DECISION_H", "FAMILY", "KL_FREE_BITS", "REPORTED_K", "SEEDS_REQUIRED", "STAGES",
    "categorical_kl", "decode_margin", "entropy_by_group", "information", "log_softmax",
    "marginal_accuracy", "marginal_classes", "mode", "open_accuracy", "open_marginal",
    "open_persistence", "persistence_accuracy", "teacher_accuracy", "teacher_nll",
    "CONTRAST_LABELS", "FAILING_STATUS", "ArmInputs", "ArmReading", "StageResult",
    "StagesInputs", "StagesReading", "Status", "format_reading_stages", "reading_stages",
]


# ---------------------------------------------------------------------------
# Logits.
# ---------------------------------------------------------------------------


def _logits(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"{name} needs (..., groups, classes), got shape {array.shape}")
    return array


def log_softmax(logits) -> np.ndarray:
    """Over the last axis, shifted by the max so a 1000-logit class does not
    overflow: the other classes then underflow to exactly 0 probability,
    which is what makes a sharp posterior's entropy exactly 0."""
    array = _logits(logits, "logits")
    shifted = array - array.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def categorical_kl(logits_q, logits_p) -> np.ndarray:
    """KL(q || p) summed over the groups, per leading index: `(..., G, C)` ->
    `(...)`. The per-element form of `rssm._categorical_kl` before its
    batch-and-time mean, in float64."""
    log_q, log_p = log_softmax(logits_q), log_softmax(logits_p)
    if log_q.shape != log_p.shape:
        raise ValueError(f"logits_q shape {log_q.shape} and logits_p shape {log_p.shape} differ")
    per_group = (np.exp(log_q) * (log_q - log_p)).sum(axis=-1)
    return per_group.sum(axis=-1)


def mode(logits) -> np.ndarray:
    """The argmax over the classes: `(..., G, C)` -> `(..., G)` int; a tie
    goes to the smallest class, as `np.argmax` does."""
    return np.argmax(_logits(logits, "logits"), axis=-1)


def entropy_by_group(logits) -> np.ndarray:
    """Per-group entropy in nats, averaged over every leading axis:
    `(..., G, C)` -> `(G,)`. 0 for a sharp group, log C for a uniform one."""
    log_p = log_softmax(logits)
    p = np.exp(log_p)
    entropy = -(p * np.where(p > 0, log_p, 0.0)).sum(axis=-1)
    return entropy.reshape(-1, entropy.shape[-1]).mean(axis=0)


# ---------------------------------------------------------------------------
# The window layout.
# ---------------------------------------------------------------------------


def _split(post_logits, context: int) -> tuple[np.ndarray, np.ndarray]:
    """`(anchor (n, G, C), horizon (n, H, G, C))`: the posterior at t0 and at
    horizon steps 1..H."""
    post = _logits(post_logits, "post_logits")
    if post.ndim != 4:
        raise ValueError(f"post_logits needs (windows, steps, groups, classes), got shape {post.shape}")
    context = int(context)
    if not 1 <= context < post.shape[1]:
        raise ValueError(
            f"context {context} must be >= 1 and leave at least one horizon step of the "
            f"{post.shape[1]} window steps"
        )
    return post[:, context - 1], post[:, context:]


def _horizon_prior(value, name: str, horizon: np.ndarray) -> np.ndarray:
    prior = _logits(value, name)
    if prior.shape != horizon.shape:
        raise ValueError(f"{name} shape {prior.shape} must match the horizon posterior {horizon.shape}")
    return prior


def _classes(value, horizon: np.ndarray) -> np.ndarray:
    classes = np.asarray(value, dtype=int)
    if classes.shape != (horizon.shape[2],):
        raise ValueError(f"classes must be one class per group, shape ({horizon.shape[2]},), got {classes.shape}")
    return classes


# ---------------------------------------------------------------------------
# Encode.
# ---------------------------------------------------------------------------


def information(post_logits, prior_teacher_logits, context: int) -> np.ndarray:
    """Per window, the mean over horizon steps of KL(posterior || teacher-forced
    prior) summed over groups -- the information the frame injects beyond
    what the prior predicted (spec 2.3, encode). `(n,)`."""
    _, horizon = _split(post_logits, context)
    teacher = _horizon_prior(prior_teacher_logits, "prior_teacher_logits", horizon)
    return categorical_kl(horizon, teacher).mean(axis=1)


# ---------------------------------------------------------------------------
# Predict: teacher-forced, one step.
# ---------------------------------------------------------------------------


def teacher_accuracy(post_logits, prior_teacher_logits, context: int) -> np.ndarray:
    """Per window, over (horizon step, group): does the teacher-forced prior's
    mode equal the posterior's mode at the same step? `(n,)`."""
    _, horizon = _split(post_logits, context)
    teacher = _horizon_prior(prior_teacher_logits, "prior_teacher_logits", horizon)
    return (mode(teacher) == mode(horizon)).mean(axis=(1, 2))


def persistence_accuracy(post_logits, context: int) -> np.ndarray:
    """Latent persistence: the posterior's mode at h - 1 predicts h (with t0
    before h = 1) -- what a model that says "nothing changes" scores exactly.
    `(n,)`."""
    post = _logits(post_logits, "post_logits")
    _split(post, context)
    modes = mode(post[:, int(context) - 1:])
    return (modes[:, :-1] == modes[:, 1:]).mean(axis=(1, 2))


def marginal_classes(post_logits, context: int) -> np.ndarray:
    """Per group, the most frequent posterior mode over every (window,
    horizon step) pair; ties to the smallest class. `(G,)`. In-sample by
    design: the stronger baseline."""
    _, horizon = _split(post_logits, context)
    modes = mode(horizon)
    classes = horizon.shape[-1]
    counts = np.stack([
        np.bincount(modes[:, :, g].ravel(), minlength=classes) for g in range(modes.shape[-1])
    ])
    return counts.argmax(axis=1)


def marginal_accuracy(post_logits, classes, context: int) -> np.ndarray:
    """Per window, over (horizon step, group): does the marginal class equal
    the posterior's mode? `(n,)`."""
    _, horizon = _split(post_logits, context)
    marginal = _classes(classes, horizon)
    return (mode(horizon) == marginal[None, None, :]).mean(axis=(1, 2))


# ---------------------------------------------------------------------------
# Carry: open-loop from t0, per step.
# ---------------------------------------------------------------------------


def open_accuracy(post_logits, prior_open_logits, context: int) -> np.ndarray:
    """Per window and step k, over groups: does the open-loop prior's mode at
    k equal the posterior's mode at k? `(n, H)`."""
    _, horizon = _split(post_logits, context)
    opened = _horizon_prior(prior_open_logits, "prior_open_logits", horizon)
    return (mode(opened) == mode(horizon)).mean(axis=2)


def open_persistence(post_logits, context: int) -> np.ndarray:
    """Latent persistence from t0: the posterior's mode at t0 predicts every
    k. `(n, H)`."""
    anchor, horizon = _split(post_logits, context)
    return (mode(anchor)[:, None, :] == mode(horizon)).mean(axis=2)


def open_marginal(post_logits, classes, context: int) -> np.ndarray:
    """The marginal class against the posterior's mode at every k. `(n, H)`."""
    _, horizon = _split(post_logits, context)
    marginal = _classes(classes, horizon)
    return (mode(horizon) == marginal[None, None, :]).mean(axis=2)


# ---------------------------------------------------------------------------
# Companions and decode.
# ---------------------------------------------------------------------------


def teacher_nll(post_logits, prior_teacher_logits, context: int) -> np.ndarray:
    """Per window, the mean over (horizon step, group) of the teacher-forced
    prior's negative log-probability of the posterior's mode. `(n,)`."""
    _, horizon = _split(post_logits, context)
    teacher = _horizon_prior(prior_teacher_logits, "prior_teacher_logits", horizon)
    picked = np.take_along_axis(log_softmax(teacher), mode(horizon)[..., None], axis=-1)[..., 0]
    return -picked.mean(axis=(1, 2))


def decode_margin(persistence_distance, distance_to_truth, h: int) -> np.ndarray:
    """`||e_hat(0) - e(h)|| - ||e_hat(h) - e(h)||` per window at one horizon
    step: positive where the imagination is closer to the truth than the held
    context rendering is. `(n,)`."""
    persistence = np.asarray(persistence_distance, dtype=np.float64)
    model = np.asarray(distance_to_truth, dtype=np.float64)
    if persistence.ndim != 2 or persistence.shape != model.shape:
        raise ValueError(
            f"persistence_distance shape {persistence.shape} and distance_to_truth shape "
            f"{model.shape} must both be (windows, horizon)"
        )
    h = int(h)
    if not 1 <= h <= persistence.shape[1]:
        raise ValueError(f"h must be in 1..{persistence.shape[1]}, got {h}")
    return persistence[:, h - 1] - model[:, h - 1]


# ---------------------------------------------------------------------------
# The reading (spec 3.2), over pooled inputs.
# ---------------------------------------------------------------------------

CONTRAST_LABELS: dict[str, tuple[str, ...]] = {
    "encode": (f"information - {KL_FREE_BITS:.2f}",),
    "predict": ("teacher - persistence", "teacher - marginal"),
    "carry": ("open - persistence",),
    "decode": ("free margin",),
}
"""Each stage's contrasts in the order its sentence names them: predict's
persistence contrast before its marginal one."""


@dataclass(frozen=True)
class ArmInputs:
    """One arm's five pooled contrasts at the decision horizon, and the same
    within each seed alone (leaves carry `per_seed=None`). `contrasts(stage)`
    is the stage's contrasts under `CONTRAST_LABELS`' names and order."""

    encode: StratumContrast
    predict_persistence: StratumContrast
    predict_marginal: StratumContrast
    carry: StratumContrast
    decode: StratumContrast
    per_seed: "dict[int, ArmInputs] | None"

    def contrasts(self, stage: str) -> tuple[tuple[str, StratumContrast], ...]:
        labels = CONTRAST_LABELS[stage]
        values = {
            "encode": (self.encode,),
            "predict": (self.predict_persistence, self.predict_marginal),
            "carry": (self.carry,),
            "decode": (self.decode,),
        }[stage]
        return tuple(zip(labels, values))


@dataclass(frozen=True)
class StagesInputs:
    arms: dict[str, ArmInputs]
    z_fam: float
    h: int


class Status(str, Enum):
    """Spec 3.2's five outcomes: the first stage not passed, or none."""

    ENCODE_FAILS = "encode fails"
    PREDICT_FAILS = "predict fails"
    CARRY_FAILS = "carry fails"
    DECODE_FAILS = "decode fails"
    NO_STAGE_FAILS = "no stage fails"


FAILING_STATUS: dict[str, Status] = {
    "encode": Status.ENCODE_FAILS,
    "predict": Status.PREDICT_FAILS,
    "carry": Status.CARRY_FAILS,
    "decode": Status.DECODE_FAILS,
}


@dataclass(frozen=True)
class StageResult:
    """One stage of one arm: whether it passed, the word the sentence uses
    (`passes` / `failed` / `not shown`), the sentence, the replication."""

    stage: str
    passes: bool
    wording: str
    rule: str
    seeds_holding: int
    seeds_total: int


@dataclass(frozen=True)
class ArmReading:
    """`stages` carries every stage's result, in `STAGES` order, so the
    tables can print the ones after the first not passed; `status` and
    `rule` are the first not passed's, or `NO_STAGE_FAILS`."""

    arm: str
    status: Status
    rule: str
    stages: dict[str, StageResult]


@dataclass(frozen=True)
class StagesReading:
    arms: dict[str, ArmReading]
    h: int
    z_fam: float


def _holds(a: ArmInputs, stage: str, z_fam: float) -> bool:
    """Every contrast of the stage clears within these inputs."""
    return all(clears(contrast.z, z_fam) for _, contrast in a.contrasts(stage))


def _stage_result(stage: str, a: ArmInputs, z_fam: float) -> StageResult:
    """Spec 3.2's rule for one stage: pooled clears on every contrast AND
    replicated in >= SEEDS_REQUIRED seeds. The sentence names the first
    contrast not clearing (`failed` when its estimate is <= 0 or it clears
    the wrong way, `not shown` otherwise), then the contrasts that do clear,
    then the seeds holding."""
    pooled = a.contrasts(stage)
    bar = f"{z_fam:.2f}"
    if a.per_seed is None:
        # A single-seed leaf, read alone: there is nothing to replicate across.
        seeds_holding, seeds_total, replicated = int(_holds(a, stage, z_fam)), 1, True
        seeds_words = "this seed alone"
    else:
        seeds_holding = sum(1 for one in a.per_seed.values() if _holds(one, stage, z_fam))
        seeds_total = len(a.per_seed)
        replicated = seeds_holding >= SEEDS_REQUIRED
        seeds_words = f"seeds holding {seeds_holding} of {seeds_total}"
    clearing = [
        f"{label} z {fmt_z(contrast.z)} clears +{bar}"
        for label, contrast in pooled if clears(contrast.z, z_fam)
    ]
    blocking = [(label, contrast) for label, contrast in pooled if not clears(contrast.z, z_fam)]
    if blocking:
        label, contrast = blocking[0]
        passes = False
        if not np.isfinite(contrast.z):
            wording, verdict = "not shown", f"{label} z {fmt_z(contrast.z)} cannot be read"
        elif clears(-contrast.z, z_fam):
            wording, verdict = "failed", f"{label} z {fmt_z(contrast.z)} < -{bar}"
        elif not contrast.estimate > 0:
            wording = "failed"
            verdict = f"{label} estimate {fmt_z(contrast.estimate, '+.4f')} <= 0 (z {fmt_z(contrast.z)})"
        else:
            wording, verdict = "not shown", f"{label} z {fmt_z(contrast.z)} does not clear +{bar}"
        rest = clearing + [seeds_words]
    elif not replicated:
        passes, wording = False, "not shown"
        verdict = f"{clearing[0]} pooled"
        rest = clearing[1:] + [f"{seeds_words} (>= {SEEDS_REQUIRED} required)"]
    else:
        passes, wording = True, "passes"
        verdict = clearing[0]
        rest = clearing[1:] + [seeds_words]
    return StageResult(
        stage=stage, passes=passes, wording=wording,
        rule=f"{stage} {wording}: {verdict} ({'; '.join(rest)})",
        seeds_holding=seeds_holding, seeds_total=seeds_total,
    )


def _arm_reading(arm: str, a: ArmInputs, z_fam: float, h: int) -> ArmReading:
    """The stages in order; the status is the first not passed."""
    results = {stage: _stage_result(stage, a, z_fam) for stage in STAGES}
    for stage in STAGES:
        if not results[stage].passes:
            return ArmReading(arm=arm, status=FAILING_STATUS[stage], rule=results[stage].rule, stages=results)
    summary = "; ".join(results[stage].rule for stage in STAGES)
    return ArmReading(
        arm=arm, status=Status.NO_STAGE_FAILS,
        rule=f"every stage passes at h={h}: {summary}", stages=results,
    )


def reading_stages(inputs: StagesInputs) -> StagesReading:
    """One `ArmReading` per arm, in the caller's order; arms never read each
    other (spec 4: no ranking)."""
    return StagesReading(
        arms={arm: _arm_reading(arm, a, inputs.z_fam, inputs.h) for arm, a in inputs.arms.items()},
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_stages(reading: StagesReading, inputs: StagesInputs) -> str:
    """The contrast table (every contrast of every arm, pooled), each stage's
    sentence, and the verdict lines, in `ladder.txt`'s style."""
    lines = [
        f"--- Reading S: the first failing stage at h={reading.h} (each contrast pooled over "
        f"the val windows, seeds averaged per window, episode-clustered); z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'stage':<9}{'contrast':<24}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for stage in STAGES:
            for label, contrast in a.contrasts(stage):
                verdict = "yes" if clears(contrast.z, reading.z_fam) else "no"
                lines.append(
                    f"  {arm:<12}{stage:<9}{label:<24}{fmt_z(contrast.estimate, '+.4f'):>10}"
                    f"{fmt_z(contrast.se, '.4f'):>9}{fmt_z(contrast.z):>8}  {verdict}"
                )
    lines.append("  stage by stage:")
    for arm, r in reading.arms.items():
        for stage in STAGES:
            lines.append(f"    {arm:<12}{r.stages[stage].rule}")
    for arm, r in reading.arms.items():
        lines.append(f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}")
    return "\n".join(lines) + "\n"
