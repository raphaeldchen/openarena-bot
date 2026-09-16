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
