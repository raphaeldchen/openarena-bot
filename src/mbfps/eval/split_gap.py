"""The split gap (M3e): does the world model roll out better on the episodes
it trained on?

Every rollout number the M3 study ever recorded is on the 24 held-out
episodes. The 20,000-step budget is ~430 passes over the 98 training episodes,
so "memorised, does not transfer" has been a live explanation for the h=45
failure since M3b and has never been put to the records. This module is the
pure half of the diagnostic that puts it there: the rule for WHICH training
episodes are model-seen but probe-unseen (`strata_partition`, over
`probe.probe_episodes`), the reduction of one stratum's reference pass to the
quantities the reading needs (`stratum_summary`, through `mbfps.eval.trust`),
the pre-registered reading over pooled stratum contrasts (`reading_gap`,
spec 3.3), and the per-term learning-curve summary read only when the reading
says the gap is not the cause (`learning_curve_summary`, spec 2.3).

Pure over numpy arrays and dicts: no torch, no files, no randomness. The
driver (`scripts/split_gap.py`) does the loading, the passes, the pooling and
the printing.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np

from mbfps.eval.diagnostics import Trajectories
from mbfps.eval.summary import METRICS, metric_summary
from mbfps.eval.trust import (
    crossing_step,
    moved_mask,
    persistence_margin,
    survival,
    trust_horizon,
)
from mbfps.eval.trust_readings import Q_REPORTED

STRATA: tuple[str, ...] = ("val", "train_held", "train_probe")
"""Spec 2.1. `val` is the record's own and the self-check anchor; `train_held`
-- model-seen, probe-unseen -- is the decision stratum; `train_probe` is
confounded by construction (the probe saw it) and is information only."""
DECISION: tuple[str, str] = ("train_held", "val")
"""The two strata every contrast is between, in the orientation printed:
`train_held - val`, positive when the model does better on episodes it saw."""
CHANNELS: tuple[str, ...] = ("probe", "free")
"""`free` (embedding space, the training target's own units) decides; `probe`
(position, through the refit probe) is the control twin."""
CURVE_NAMES: tuple[str, ...] = (
    "rssm_position", "persistence_position", "floor_position",
    "rssm_angle", "persistence_angle", "floor_angle",
)
DECISION_H: int = 15
"""Spec 3.2: the governing spec's imagination horizon, pre-registered."""
REPORTED_H: tuple[int, ...] = (5, 15, 45)
"""Printed beside the decision horizon with their z; not decided on."""
FAMILY: int = 6
"""Spec 3.1: three arms x two channels at one horizon."""
SEEDS_REQUIRED: int = 2
"""Spec 3.3: a gap "clears ... pooled and within >= 2 of 3 seeds"."""
TERMS: tuple[str, ...] = ("embedding", "reward", "continue", "kl_dyn", "kl_rep")
"""The keys of every `history.parts[i]` a study record carries."""


def q_key(q: float) -> str:
    """`0.75 -> "q75"`: a record key may not contain '.', since `write_record`
    addresses non-finite fields by dotted path."""
    return f"q{int(round(q * 100))}"


# ---------------------------------------------------------------------------
# The strata.
# ---------------------------------------------------------------------------


class StrataNotAPartition(ValueError):
    """The three strata are not pairwise disjoint, do not cover the buffer,
    `train_probe` is not the leading block of `train`, or `train_held` is
    empty. A CODE (or fixture) defect, never a data one: `scripts/split_gap.py`
    maps it to exit 31 before any probe is refit."""


def strata_partition(all_paths, train, val, used) -> dict[str, list]:
    """`{"val": val, "train_held": train[len(used):], "train_probe": used}`,
    every list in the caller's order, after checking they partition
    `all_paths` (spec 2.4, exit 31).

    `used` must be the leading block of `train` -- what `probe_episodes(train)`
    returns -- so `train_held` is the rest of `train` and nothing else.
    """
    all_names = sorted(p.name for p in all_paths)
    train, val, used = list(train), list(val), list(used)
    if used != train[: len(used)]:
        raise StrataNotAPartition(
            "train_probe is not the leading block of train: probe_episodes and the "
            "split disagree on which episodes the probe was fit on"
        )
    held = train[len(used):]
    strata = {"val": val, "train_held": held, "train_probe": used}
    names = {name: [p.name for p in paths] for name, paths in strata.items()}
    for a, b in combinations(STRATA, 2):
        shared = sorted(set(names[a]) & set(names[b]))
        if shared:
            raise StrataNotAPartition(
                f"{a} and {b} share {len(shared)} episode(s), e.g. {shared[0]}"
            )
    union = sorted(n for stratum in names.values() for n in stratum)
    if union != all_names:
        missing = sorted(set(all_names) - set(union))
        extra = sorted(set(union) - set(all_names))
        raise StrataNotAPartition(
            f"the strata do not cover the buffer: {len(missing)} episode(s) missing"
            f"{f' (e.g. {missing[0]})' if missing else ''}, {len(extra)} not in the buffer"
            f"{f' (e.g. {extra[0]})' if extra else ''}"
        )
    if not held:
        raise StrataNotAPartition(
            f"train_held is empty: train has {len(train)} episode(s) and the probe used "
            f"{len(used)}, so there is no model-seen, probe-unseen stratum to decide on"
        )
    return strata


# ---------------------------------------------------------------------------
# One stratum's pass, reduced.
# ---------------------------------------------------------------------------


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-D Euclidean distance over the last axis -- `position_error`'s norm."""
    return np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=-1)


def stratum_summary(traj: Trajectories, horizon: int) -> dict:
    """Everything the reading and the results section read off one stratum
    of one cell, every quantity through the pure functions of
    `mbfps.eval.trust` or `mbfps.eval.summary` (spec 2.2).

    The moved mask is the TRUTH's (`moved_mask`), so both channels score the
    same rows. The probe channel's errors are the probe's positions against
    the true positions; the free channel's are the pass's two embedding
    series, `D_hat(h)` and `D_0(h)`, the training target's own units. The
    band is the pass's whole `RolloutResult`, summarised by `metric_summary`
    exactly as `run_job` summarises the val band, so `gap_final` here on
    `train_held` is spec 4.1's criterion on the train side.
    """
    if traj.band is None:
        raise ValueError(
            "Trajectories.band is None: the pass was not reference_trajectories, or a "
            "fabricated Trajectories carries no band"
        )
    moved = moved_mask(traj.true_positions, traj.true_at_context)
    model_err = _distance(traj.positions, traj.true_positions)
    persist_err = _distance(traj.positions_at_context[:, None, :], traj.true_positions)
    d_hat = np.asarray(traj.embedding_distance_to_truth, dtype=float)
    d_0 = np.asarray(traj.embedding_persistence_distance, dtype=float)
    expected = (int(traj.windows_total), int(horizon))
    if model_err.shape != expected or d_hat.shape != expected or d_0.shape != expected:
        raise ValueError(
            f"the rows are not (windows_total, horizon) = {expected}: positions give "
            f"{model_err.shape}, the embedding series {d_hat.shape} and {d_0.shape}"
        )
    crossing = {
        "probe": crossing_step(model_err, persist_err, moved),
        "free": crossing_step(d_hat, d_0, moved),
    }
    margin = {
        "probe": persistence_margin(model_err, persist_err),
        "free": persistence_margin(d_hat, d_0),
    }
    surv = {channel: survival(crossing[channel], horizon) for channel in CHANNELS}
    episode = np.asarray(traj.window_episode, dtype=int)
    return {
        "windows": {
            "total": int(traj.windows_total),
            "episode": [int(e) for e in episode],
            "clusters": int(np.unique(episode).size),
        },
        "curves": {name: np.asarray(getattr(traj.band, name), dtype=float) for name in CURVE_NAMES},
        "band": {metric: metric_summary(traj.band, metric) for metric in METRICS},
        "moved": moved,
        # h0 per window, 1-based: `crossing_step`'s own search start. NaN where
        # the window never moves within the horizon.
        "first_moved": np.where(moved.any(axis=1), moved.argmax(axis=1) + 1.0, np.nan),
        "crossing": crossing,
        "margin": margin,
        "survival": surv,
        "trust_horizon": {
            channel: {q_key(q): trust_horizon(surv[channel], q) for q in Q_REPORTED}
            for channel in CHANNELS
        },
        "counts": {
            "not_moved": (~moved).sum(axis=0),
            "never_moved": int((~moved.any(axis=1)).sum()),
        },
    }


# ---------------------------------------------------------------------------
# Reading G -- the generalisation gap (spec 3.3), over pooled inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratumContrast:
    """`pooling.unpaired_contrast` reduced to what the rules read."""

    estimate: float
    se: float
    z: float
    clusters: int


@dataclass(frozen=True)
class ArmInputs:
    """One arm's pooled numbers at the decision horizon: the two stratum
    contrasts (`train_held - val`), spec 4.1's criterion on `train_held` per
    seed, and the same inputs within each seed alone (leaves carry
    `per_seed=None`)."""

    gap_free: StratumContrast
    gap_probe: StratumContrast
    train_held_gap_final: dict[int, float]
    per_seed: "dict[int, ArmInputs] | None"


@dataclass(frozen=True)
class GapInputs:
    arms: dict[str, ArmInputs]
    z_fam: float
    h: int


class Status(str, Enum):
    """Spec 3.3's five outcomes, in the table's order of precedence."""

    UNRESOLVED_PROBE = "unresolved through the probe"
    MEMORISATION = "memorisation"
    PARTIAL_GAP = "partial gap"
    INVERTED_GAP = "inverted gap"
    NO_GAP = "no gap"


@dataclass(frozen=True)
class ArmReading:
    arm: str
    status: Status
    rule: str
    free_clears_up: bool
    free_clears_down: bool
    probe_clears_up: bool
    probe_clears_down: bool
    seeds_clearing: int
    seeds_total: int
    train_held_unanimous: bool


@dataclass(frozen=True)
class GapReading:
    arms: dict[str, ArmReading]
    h: int
    z_fam: float


def clears(z: float, z_fam: float) -> bool:
    """STRICTLY above the bar. NaN never clears (nothing to read); an infinite
    z never clears either -- `pooling._z` returns +-inf for a zero standard
    error, a degenerate ruler. `trust_readings._clears`'s policy, restated."""
    return bool(np.isfinite(z) and np.isfinite(z_fam) and z > z_fam)


def train_held_passes_gate(gap_final: dict[int, float]) -> bool:
    """Spec 3.3 / governing spec 4.1 on the train side: `gap_closed(45)` on
    position > 0 in ALL seeds of the arm, where a NaN (a non-positive
    persistence-to-floor band) is not > 0. No seed at all is not unanimity."""
    values = list(gap_final.values())
    return bool(values) and all(np.isfinite(v) and v > 0.0 for v in values)


def _fmt(value: float, spec: str = "+.2f") -> str:
    return format(value, spec) if np.isfinite(value) else str(value)


def _gate_detail(gap_final: dict[int, float]) -> str:
    return "gap_closed(45) per seed " + " / ".join(
        f"s{seed} {_fmt(gap_final[seed], '+.3f')}" for seed in sorted(gap_final)
    )


def _arm_reading(arm: str, a: ArmInputs, z_fam: float) -> ArmReading:
    """The rules of spec 3.3, in the table's precedence, each status carrying
    the sentence that decided it."""
    free_up, free_down = clears(a.gap_free.z, z_fam), clears(-a.gap_free.z, z_fam)
    probe_up, probe_down = clears(a.gap_probe.z, z_fam), clears(-a.gap_probe.z, z_fam)
    per_seed = a.per_seed or {}
    seeds_clearing = sum(1 for leaf in per_seed.values() if clears(leaf.gap_free.z, z_fam))
    unanimous = train_held_passes_gate(a.train_held_gap_final)
    fz, pz, bar = _fmt(a.gap_free.z), _fmt(a.gap_probe.z), f"{z_fam:.2f}"
    seeds = f"{seeds_clearing} of {len(per_seed)} seeds"
    if (free_up and probe_down) or (free_down and probe_up):
        status = Status.UNRESOLVED_PROBE
        rule = f"G_free z {fz} and G_probe z {pz} both clear +-{bar} with opposite signs"
    elif free_up and seeds_clearing >= SEEDS_REQUIRED and unanimous:
        status = Status.MEMORISATION
        rule = (f"G_free z {fz} > {bar} pooled and in {seeds}; train_held passes spec 4.1 "
                f"({_gate_detail(a.train_held_gap_final)})")
    elif free_up and seeds_clearing >= SEEDS_REQUIRED:
        status = Status.PARTIAL_GAP
        rule = (f"G_free z {fz} > {bar} pooled and in {seeds}; train_held FAILS spec 4.1 "
                f"({_gate_detail(a.train_held_gap_final)})")
    elif free_down:
        status = Status.INVERTED_GAP
        rule = f"G_free z {fz} < -{bar}: train_held is worse than val"
    elif free_up:
        status = Status.NO_GAP
        rule = (f"G_free z {fz} clears {bar} pooled but in only {seeds} "
                f"(>= {SEEDS_REQUIRED} required)")
    else:
        status = Status.NO_GAP
        rule = f"G_free z {fz} does not clear +-{bar}"
    return ArmReading(
        arm=arm, status=status, rule=rule,
        free_clears_up=free_up, free_clears_down=free_down,
        probe_clears_up=probe_up, probe_clears_down=probe_down,
        seeds_clearing=seeds_clearing, seeds_total=len(per_seed),
        train_held_unanimous=unanimous,
    )


def reading_gap(inputs: GapInputs) -> GapReading:
    """One `ArmReading` per arm, in the caller's order; arms never read each
    other (spec 4: no ranking)."""
    return GapReading(
        arms={arm: _arm_reading(arm, a, inputs.z_fam) for arm, a in inputs.arms.items()},
        h=inputs.h, z_fam=inputs.z_fam,
    )


def format_reading_gap(reading: GapReading, inputs: GapInputs) -> str:
    """The contrast table and the verdict lines, in `trust.txt`'s style."""
    lines = [
        f"--- Reading G: the generalisation gap at h={reading.h} (train_held - val); "
        f"z_fam = {reading.z_fam:.2f} ---",
        f"  {'arm':<12}{'channel':<9}{'estimate':>10}{'se':>9}{'z':>8}  clears",
    ]
    for arm, a in inputs.arms.items():
        for channel, c in (("free", a.gap_free), ("probe", a.gap_probe)):
            verdict = "yes" if clears(abs(c.z), reading.z_fam) else "no"
            lines.append(
                f"  {arm:<12}{channel:<9}{_fmt(c.estimate, '+.4f'):>10}{_fmt(c.se, '.4f'):>9}"
                f"{_fmt(c.z):>8}  {verdict}"
            )
    lines.append("  train_held, spec 4.1 (gap_closed(45) position > 0 in every seed):")
    for arm, a in inputs.arms.items():
        word = "passes" if train_held_passes_gate(a.train_held_gap_final) else "FAILS"
        lines.append(f"    {arm:<12}{_gate_detail(a.train_held_gap_final)} -> {word}")
    for arm, r in reading.arms.items():
        lines.append(
            f"  verdict: {arm:<12}{r.status.name.replace('_', ' ')} -- decided by: {r.rule}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Learning curves (spec 2.3): descriptive, no verdict.
# ---------------------------------------------------------------------------


def _smoothed(series: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(series, np.ones(window) / window, mode="valid")


def learning_curve_summary(history: dict, window: int = 100) -> dict:
    """Per term (and the summed loss): the mean over the last quarter of
    training against the mean over the quarter before it -- the statistic the
    M3b write-up used on the summed loss -- as sign and percentage, and the
    1-based step at which the `window`-step moving mean is lowest (the step
    at the END of the minimising window). `window` shrinks to the history's
    length when the history is shorter than it."""
    loss = np.asarray(history["loss"], dtype=float)
    parts = list(history["parts"])
    n = int(loss.size)
    if n < 4:
        raise ValueError(f"a learning curve needs at least four steps for quarters, got {n}")
    if len(parts) != n:
        raise ValueError(
            f"history.loss ({n}) and history.parts ({len(parts)}) disagree on the step count"
        )
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    window = min(int(window), n)
    series = {"loss": loss}
    for term in TERMS:
        series[term] = np.asarray([float(p[term]) for p in parts], dtype=float)
    quarter = n // 4
    terms = {}
    for name, s in series.items():
        last = float(s[n - quarter:].mean())
        previous = float(s[n - 2 * quarter: n - quarter].mean())
        smoothed = _smoothed(s, window)
        terms[name] = {
            "last_quarter_mean": last,
            "preceding_quarter_mean": previous,
            "change_pct": float(100.0 * (last - previous) / previous) if previous != 0.0 else float("nan"),
            "descending": bool(last < previous),
            "smoothed_min_step": int(np.argmin(smoothed)) + window,
            "smoothed_min": float(smoothed.min()),
            "smoothed_final": float(smoothed[-1]),
        }
    return {"steps": n, "window": window, "quarter": int(quarter), "terms": terms}
