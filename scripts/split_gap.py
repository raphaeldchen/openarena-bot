"""The split gap over the M3c cells: load, check, three strata per cell, one record per cell; then pool, read the gap, draw the learning curves.

Every rollout number the M3 study recorded is on the 24 held-out episodes. The
20,000-step budget is ~430 passes over the 98 training episodes, and whether
the h=45 failure is memorisation has never been measured. This script
evaluates each frozen checkpoint on THREE strata of episodes under the ONE
refit probe its record was scored with:

  val          the record's 24 held-out episodes -- the self-check anchor
  train_held   train[PROBE_EPISODE_LIMIT:] -- 78 episodes the model trained on
               and the probe never saw -- THE DECISION STRATUM
  train_probe  train[:PROBE_EPISODE_LIMIT] -- the probe's own 20; confounded
               by construction and labelled so wherever it is printed

and writes one `split_gap_<arm>_seed<n>.json` per cell holding, per stratum,
the band and its gap_closed, the crossing step in both channels, the
persistence margin in both channels, the survival curve and H*_q, and the
counts -- `mbfps.eval.split_gap.stratum_summary`. After the per-cell loop,
`main` pools the LIVE records, decides Reading G (spec 3.3), prints the
sensitivity lines and the learning-curve table, draws `learning_curves.png`
and writes `split_gap.txt`.

LOADING IS `trust_horizon.py`'S, NOT A SECOND COPY. `Cell`, `load_cell`,
`self_check`, `protocol_mismatch` and the checkpoint/diagnostic paths are
imported from that script by path -- the way every script test loads a
script, and the way trust_horizon itself imports diagnose_dynamics -- so the
three tools read one cell through one loader.

THE CHECKS RUN IN A FIXED ORDER and each has its own status:

  EXIT_NO_CHECKPOINTS (11)          a REQUESTED cell lacks its checkpoint,
                                    study record or diagnostic. Every cell is
                                    loaded before anything else.
  EXIT_STRATA_NOT_A_PARTITION (31)  NEW. The three strata are not a partition
                                    of the buffer, `train_probe` is not what
                                    `probe_episodes` hands `fit_probes`, or
                                    `train_held` is empty. Judged ONCE, up
                                    front, before any refit: a code or
                                    fixture defect, never a data one.
  EXIT_SPLIT_MISMATCH (12)          the split by name is not the record's.
  EXIT_RECORD_MISMATCH (14)         `evaluate_rollout` on val no longer
                                    reproduces the record's curves.rssm_position
                                    (an ENVIRONMENT difference: cpu misses by
                                    6-12 map units), or --context/--horizon
                                    disagree with the diagnostic's protocol.
  EXIT_SELF_CHECK_FAILED (30)       the val stratum's mean curves are not
                                    bitwise the diagnostic's, or its windows
                                    are not the diagnostic's. Same windows,
                                    same rollout, same refit probe -- or the
                                    train strata are not read against the
                                    ruler M3d validated.

11 / 12 / 14 / 30 carry `trust_horizon.py`'s meanings on purpose; 31 is in no
other tool's range (run_study 1/3-6/23, report_study 7-10, spike 10, diagnose
11-17, pool 18-22, trust 30, argparse 2, a traceback 1).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

import mbfps.eval.pooling as pooling
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import reference_trajectories
from mbfps.eval.probe import probe_episodes
from mbfps.eval.split_gap import (
    CHANNELS,
    DECISION,
    DECISION_H,
    FAMILY,
    REPORTED_H,
    STRATA,
    TERMS,
    ArmInputs,
    GapInputs,
    StrataNotAPartition,
    StratumContrast,
    cell_series,
    decision_horizon,
    format_reading_gap,
    learning_curve_summary,
    margin_at,
    q_key,
    reading_gap,
    strata_partition,
    stratum_summary,
    survival_indicator,
)
from mbfps.eval.study import SPLIT_SEED, git_sha, write_record
from mbfps.eval.trust import survival, trust_horizon
from mbfps.eval.trust_readings import ARMS_ORDER, Q_REPORTED
from mbfps.models.rssm import KL_FREE_BITS
from mbfps.utils.config import ARMS
from mbfps.utils.device import get_device


def _sibling(name: str):
    """Load `scripts/<name>.py` by path -- the way the tests load every script."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_for_split_gap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_trust = _sibling("trust_horizon")
Cell = _trust.Cell
CellMissing = _trust.CellMissing
load_cell = _trust.load_cell
self_check = _trust.self_check
prepare_cell = _trust.prepare_cell
probe_is_measurable = _trust.probe_is_measurable

EXIT_OK = _trust.EXIT_OK
EXIT_NO_CHECKPOINTS = _trust.EXIT_NO_CHECKPOINTS
EXIT_SPLIT_MISMATCH = _trust.EXIT_SPLIT_MISMATCH
EXIT_RECORD_MISMATCH = _trust.EXIT_RECORD_MISMATCH
EXIT_SELF_CHECK_FAILED = _trust.EXIT_SELF_CHECK_FAILED
EXIT_STRATA_NOT_A_PARTITION = 31
"""0 / 11 / 12 / 14 / 30 are `trust_horizon.py`'s, imported so they cannot
drift; 31 is new and in no other tool's range."""


# ---------------------------------------------------------------------------
# One cell: the record.
# ---------------------------------------------------------------------------


def split_gap_record_path(out_dir: Path, arm: str, seed: int) -> Path:
    """One file per cell, named by BOTH the arm and the seed."""
    return Path(out_dir) / f"split_gap_{arm}_seed{seed}.json"


def split_gap_record(
    cell: Cell, strata_traj: dict, strata_paths: dict, check, *, context, horizon, device
) -> dict:
    """The LIVE record for one cell: numpy arrays and real NaNs; `write_record`
    sanitises it on the way to disk, so no top-level `nonfinite` key here."""
    curves = cell.diagnostic["curves"]
    return {
        "arm": cell.arm,
        "seed": cell.seed,
        "context": int(context),
        "horizon": int(horizon),
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "checkpoint_git_sha": str(cell.record.get("git_sha", "unknown")),
        "probe": {
            "selection_r2": float(cell.diagnostic["probe"]["embedding_selection_r2"]),
            "measurable": bool(probe_is_measurable(
                {
                    "persistence": curves["persistence_position"][-1],
                    "floor": curves["floor_position"][-1],
                },
                widest_se=0.0,
            )),
        },
        "episodes": {name: [p.name for p in strata_paths[name]] for name in STRATA},
        "self_check": check.record(),
        "strata": {name: stratum_summary(strata_traj[name], horizon) for name in STRATA},
    }


def write_split_gap_record(out_dir: Path, record: dict) -> Path:
    path = split_gap_record_path(out_dir, record["arm"], record["seed"])
    write_record(path, record)
    return path


# ---------------------------------------------------------------------------
# Pooling glue: per-cell records -> the inputs Reading G is decided on.
# ---------------------------------------------------------------------------
# Every estimator is `pooling.py`'s. What is decided here is only WHICH series
# goes in under WHICH mask (spec 3.1): the survival indicator over the windows
# that moved within the horizon (M3d's S(h) denominator, so the pooled mean IS
# S(h)); the margin over the windows moved AT h. The stratum contrast is the
# unpaired one -- two strata share no window.

R2_SENSITIVITY = 0.1
"""Cells whose probe selection R^2 is below this leave the probe channel on
the SENSITIVITY line only (spec 3.4); chosen knowing pixel_ae/s1 reads 0.018
and every other shipped cell >= 0.249. It changes no verdict."""
SURVIVAL_STEPS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 45)
"""The columns of the survival table, M3d's, filtered to <= the run's horizon."""


def _series(record: dict, stratum: str, channel: str, values, changed) -> pooling.CellSeries:
    """One per-window series of one cell's stratum, in the shape the pool
    reads. `rung` is the stratum and `channel` the reading ("S/free",
    "margin/probe"), so `require_compatible` pools seeds of one stratum only
    and `unpaired_contrast` refuses two different readings."""
    return cell_series(
        record["strata"][stratum], values, changed,
        arm=record["arm"], seed=record["seed"], rung=stratum, channel=channel,
        val=record["episodes"][stratum], horizon=record["horizon"], context=record["context"],
        device=record["device"], torch_version=record["torch_version"],
    )


def survival_series(record: dict, stratum: str, channel: str, h: int) -> pooling.CellSeries:
    """`1[h_x > h]` per window over the windows that moved within the horizon
    (finite crossing) -- so the pooled mean is S(h) exactly."""
    return _series(record, stratum, f"S/{channel}", *survival_indicator(record["strata"][stratum], channel, h))


def margin_series(record: dict, stratum: str, channel: str, h: int) -> pooling.CellSeries:
    """`Delta(h)` per window over the windows moved AT h."""
    return _series(record, stratum, f"margin/{channel}", *margin_at(record["strata"][stratum], channel, h))


def _pooled(cells) -> pooling.PooledMean | None:
    """`pool_arm`, or None when there is nothing to pool -- no cell, or no
    window surviving every cell's mask -- decided before `np.mean`."""
    if not cells:
        return None
    pooling.require_compatible(cells)
    if not np.logical_and.reduce([c.changed for c in cells]).any():
        return None
    return pooling.pool_arm(cells)


_NO_CONTRAST = StratumContrast(estimate=float("nan"), se=float("nan"), z=float("nan"), clusters=0)


def _contrast(a: pooling.PooledMean | None, b: pooling.PooledMean | None) -> StratumContrast:
    if a is None or b is None:
        return _NO_CONTRAST
    c = pooling.unpaired_contrast(a, b)
    return StratumContrast(estimate=c.mean, se=c.se, z=c.z, clusters=c.clusters)


def _probe_ok(record: dict, min_r2: float | None) -> bool:
    """Spec 3.1: the probe channel pools the measurable cells; the sensitivity
    line additionally drops cells below `min_r2`."""
    if not record["probe"]["measurable"]:
        return False
    return min_r2 is None or float(record["probe"]["selection_r2"]) >= min_r2


def arm_inputs(records: dict, arm: str, seeds, *, h: int, series=survival_series,
               min_r2: float | None = None) -> ArmInputs:
    """One arm's `ArmInputs` at step `h`: the two stratum contrasts pooled over
    `seeds`, spec 4.1's `gap_final` on train_held per seed, and the same
    within each seed alone."""
    def pooled(stratum, channel, seed_list):
        cells = [
            series(records[(arm, s)], stratum, channel, h) for s in seed_list
            if channel == "free" or _probe_ok(records[(arm, s)], min_r2)
        ]
        return _pooled(cells)

    def contrast(channel, seed_list):
        return _contrast(pooled(DECISION[0], channel, seed_list), pooled(DECISION[1], channel, seed_list))

    gate = {
        int(s): float(records[(arm, s)]["strata"]["train_held"]["band"]["position"]["gap_final"])
        for s in seeds
    }
    per_seed = {
        int(s): ArmInputs(
            gap_free=contrast("free", [s]), gap_probe=contrast("probe", [s]),
            train_held_gap_final={int(s): gate[int(s)]}, per_seed=None,
        )
        for s in seeds
    }
    return ArmInputs(
        gap_free=contrast("free", list(seeds)), gap_probe=contrast("probe", list(seeds)),
        train_held_gap_final=gate, per_seed=per_seed,
    )


def _clusters(records: dict, stratum: str) -> int:
    return int(next(iter(records.values()))["strata"][stratum]["windows"]["clusters"])


def gap_inputs(records: dict, *, arms, seeds, h: int, series=survival_series,
               min_r2: float | None = None) -> GapInputs:
    """Every arm's inputs at `h`, and `z_fam` over the SMALLER of the two
    decision strata's cluster counts (spec 3.1)."""
    clusters = min(_clusters(records, DECISION[0]), _clusters(records, DECISION[1]))
    ordered = sorted(arms, key=lambda a: ARMS_ORDER.index(a) if a in ARMS_ORDER else len(ARMS_ORDER))
    return GapInputs(
        arms={arm: arm_inputs(records, arm, seeds, h=h, series=series, min_r2=min_r2) for arm in ordered},
        z_fam=pooling.cluster_threshold(FAMILY, clusters),
        h=h,
    )


# ---------------------------------------------------------------------------
# The printed tables.
# ---------------------------------------------------------------------------


def _num(value, spec: str = ".3f") -> str:
    value = float(value)
    return format(value, spec) if np.isfinite(value) else "n/a"


def _cells_in_order(records: dict) -> list[tuple[str, int]]:
    return sorted(records, key=lambda k: (ARMS_ORDER.index(k[0]) if k[0] in ARMS_ORDER else 9, k[1]))


def _self_check_table(records: dict) -> str:
    lines = [
        "--- split gap: self-check (val stratum against the diagnostic; exact) ---",
        f"  {'arm':<12}{'seed':>5}{'ref max|d|':>12}{'pers max|d|':>13}{'windows':>9}{'probe R2':>10}{'measurable':>12}  ok",
    ]
    for arm, seed in _cells_in_order(records):
        r = records[(arm, seed)]
        c = r["self_check"]
        lines.append(
            f"  {arm:<12}{seed:>5}{c['reference_position_max_delta']:>12.1e}"
            f"{c['persistence_position_max_delta']:>13.1e}"
            f"{str(c['windows_total_match'] and c['windows_episode_match']):>9}"
            f"{r['probe']['selection_r2']:>10.3f}{str(r['probe']['measurable']):>12}  {c['ok']}"
        )
    return "\n".join(lines) + "\n"


def _strata_table(records: dict, h: int) -> str:
    first = next(iter(records.values()))
    lines = [
        "--- strata (identical for every cell: one split, one probe rule) ---",
        f"  {'stratum':<12}{'episodes':>9}{'windows':>9}{'clusters':>9}{'never_moved':>12}"
        f"{'not_moved@1':>12}{f'not_moved@{h}':>13}  role",
    ]
    roles = {
        "val": "the record's; self-check anchor",
        "train_held": "model-seen, probe-unseen; THE DECISION STRATUM",
        "train_probe": "model-seen, probe-seen; CONFOUNDED, information only",
    }
    for stratum in STRATA:
        block = first["strata"][stratum]
        not_moved = block["counts"]["not_moved"]
        lines.append(
            f"  {stratum:<12}{len(first['episodes'][stratum]):>9}{block['windows']['total']:>9}"
            f"{block['windows']['clusters']:>9}{block['counts']['never_moved']:>12}"
            f"{int(not_moved[0]):>12}{int(not_moved[h - 1]):>13}  {roles[stratum]}"
        )
    return "\n".join(lines) + "\n"


def _band_table(records: dict, arms, seeds) -> str:
    horizon = int(next(iter(records.values()))["horizon"])
    lines = [
        f"--- the band per stratum: gap_closed({horizon}) on position (spec 4.1's metric; "
        "NaN = non-positive band) ---",
        f"  {'arm':<12}{'stratum':<12}" + "".join(f"{f's{s}':>10}" for s in seeds)
        + f"{'nanmean':>10}{'unanimous>0':>13}{'degenerate':>11}",
    ]
    for arm in arms:
        for stratum in STRATA:
            finals = [
                float(records[(arm, s)]["strata"][stratum]["band"]["position"]["gap_final"])
                for s in seeds
            ]
            degenerate = max(
                int(records[(arm, s)]["strata"][stratum]["band"]["position"]["steps_degenerate"])
                for s in seeds
            )
            unanimous = all(np.isfinite(f) and f > 0 for f in finals)
            mean = float(np.nanmean(finals)) if np.isfinite(finals).any() else float("nan")
            lines.append(
                f"  {arm:<12}{stratum:<12}" + "".join(f"{_num(f, '+.4f'):>10}" for f in finals)
                + f"{_num(mean, '+.4f'):>10}{str(unanimous):>13}{degenerate:>11}"
            )
    return "\n".join(lines) + "\n"


def survival_by_arm_and_stratum(records: dict, arms, seeds) -> dict:
    """`(arm, stratum, channel) -> S(h)` over the (window, seed) draws stacked,
    every cell in the free channel, the measurable cells in the probe one."""
    horizon = int(next(iter(records.values()))["horizon"])
    out = {}
    for arm in arms:
        for stratum in STRATA:
            for channel in CHANNELS:
                draws = [
                    np.asarray(records[(arm, s)]["strata"][stratum]["crossing"][channel], dtype=float)
                    for s in seeds if channel == "free" or _probe_ok(records[(arm, s)], None)
                ]
                stacked = np.concatenate(draws) if draws else np.zeros(0)
                out[(arm, stratum, channel)] = survival(stacked, horizon) if stacked.size else np.full(horizon + 1, np.nan)
    return out


def _survival_table(curves: dict, horizon: int) -> str:
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- survival per stratum: S(h) = fraction of moved draws with h_x > h "
        "(a window not yet moved at h survives vacuously, as in M3d) ---",
        f"  {'arm':<12}{'stratum':<12}{'channel':<8}" + "".join(f"{f'S({h})':>7}" for h in steps)
        + "".join(f"{'H*' + q_key(q)[1:]:>8}" for q in Q_REPORTED),
    ]
    for (arm, stratum, channel), s in curves.items():
        lines.append(
            f"  {arm:<12}{stratum:<12}{channel:<8}" + "".join(f"{_num(s[h], '.2f'):>7}" for h in steps)
            + "".join(f"{trust_horizon(s, q):>8d}" for q in Q_REPORTED)
        )
    return "\n".join(lines) + "\n"


def _conditional_table(records: dict, arms, seeds, horizon: int) -> str:
    """Beside S(h), M3d's two series (spec 2.2), computed by `trust_horizon.py`'s
    own functions imported by path: `u(h)`, the fraction of the counted draws
    whose window has not yet moved at h (and so survives vacuously), and
    `S_c(h) = (S(h) - u(h)) / (1 - u(h))`, the survival among the draws
    measured at h. Printed, never decided on."""
    steps = [h for h in SURVIVAL_STEPS if h <= horizon]
    lines = [
        "--- conditional survival per stratum: u(h) = unmoved fraction, "
        "S_c(h) = (S(h) - u(h)) / (1 - u(h)) (M3d's series; printed, not decided on) ---",
        f"  {'arm':<12}{'stratum':<12}{'channel':<8}" + "".join(f"{f'u({h})':>7}" for h in steps)
        + "".join(f"{f'Sc({h})':>8}" for h in steps),
    ]
    for arm in arms:
        for stratum in STRATA:
            for channel in CHANNELS:
                cells = [records[(arm, s)] for s in seeds
                         if channel == "free" or _probe_ok(records[(arm, s)], None)]
                if not cells:
                    continue
                crossings = np.concatenate([
                    np.asarray(c["strata"][stratum]["crossing"][channel], dtype=float) for c in cells
                ])
                first = np.concatenate([
                    np.asarray(c["strata"][stratum]["first_moved"], dtype=float) for c in cells
                ])
                s = (survival(crossings, horizon) if np.isfinite(crossings).any()
                     else np.full(horizon + 1, np.nan))
                u = _trust.unmoved_fraction(crossings, first, horizon)
                sc = _trust.conditional_survival(s, u)
                lines.append(
                    f"  {arm:<12}{stratum:<12}{channel:<8}"
                    + "".join(f"{_num(u[h], '.2f'):>7}" for h in steps)
                    + "".join(f"{_num(sc[h], '.2f'):>8}" for h in steps)
                )
    return "\n".join(lines) + "\n"


def _reported_table(records: dict, arms, seeds, horizon: int) -> str:
    steps = sorted({h for h in REPORTED_H if h <= horizon} | {horizon})
    lines = [
        "--- G at the reported horizons (train_held - val; z only; not decided on) ---",
        f"  {'arm':<12}{'channel':<8}" + "".join(f"{f'z@{h}':>9}" for h in steps)
        + f"{'dFree@dec':>11}",
    ]
    dec, _ = decision_horizon(horizon)
    for arm in arms:
        by_h = {h: arm_inputs(records, arm, seeds, h=h) for h in steps}
        margin = arm_inputs(records, arm, seeds, h=dec, series=margin_series).gap_free
        for channel in CHANNELS:
            zs = [getattr(by_h[h], f"gap_{channel}").z for h in steps]
            extra = f"{_num(margin.estimate, '+.4f')} (z {_num(margin.z, '+.2f')})" if channel == "free" else ""
            lines.append(
                f"  {arm:<12}{channel:<8}" + "".join(f"{_num(z, '+.2f'):>9}" for z in zs) + f"  {extra}"
            )
    lines.append("  dFree@dec: the stratum difference of the embedding-space margin Delta_free "
                 "at the decision horizon -- the continuous companion, in the loss's units.")
    return "\n".join(lines) + "\n"


def _sensitivity_text(records: dict, arms, seeds, h: int, z_fam: float) -> str:
    dropped = [f"{a} s{s}" for a, s in _cells_in_order(records)
               if float(records[(a, s)]["probe"]["selection_r2"]) < R2_SENSITIVITY]
    inputs = gap_inputs(records, arms=arms, seeds=seeds, h=h, min_r2=R2_SENSITIVITY)
    lines = [
        f"--- sensitivity (changes no verdict): probe channel with selection R2 < {R2_SENSITIVITY} "
        f"excluded -- dropped: {', '.join(dropped) if dropped else 'none'} ---",
    ]
    for arm, a in inputs.arms.items():
        lines.append(
            f"  {arm:<12}G_probe estimate {_num(a.gap_probe.estimate, '+.4f')} se "
            f"{_num(a.gap_probe.se, '.4f')} z {_num(a.gap_probe.z, '+.2f')} "
            f"(bar {z_fam:.2f}; clusters {a.gap_probe.clusters})"
        )
    return "\n".join(lines) + "\n"


def _per_seed_text(inputs: GapInputs) -> str:
    lines = ["--- per seed (the same contrasts within one seed alone; no seed averaging) ---"]
    for arm, a in inputs.arms.items():
        for seed, leaf in sorted((a.per_seed or {}).items()):
            leaf_reading = reading_gap(GapInputs(arms={arm: leaf}, z_fam=inputs.z_fam, h=inputs.h)).arms[arm]
            lines.append(
                f"  {arm:<12}s{seed}: G_free z {_num(leaf.gap_free.z, '+.2f')}, G_probe z "
                f"{_num(leaf.gap_probe.z, '+.2f')}, train_held gap_closed "
                f"{_num(leaf.train_held_gap_final[seed], '+.4f')} -> "
                f"{leaf_reading.status.name.replace('_', ' ')}"
            )
    return "\n".join(lines) + "\n"


def _learning_curve_table(records: dict, window: int) -> str:
    lines = [
        f"--- learning curves (from history.parts; {window}-step moving mean; descriptive, "
        "no verdict): last quarter vs preceding quarter, and the smoothed minimum's step ---",
        f"  {'arm':<12}{'seed':>5}  {'term':<11}{'last_q':>11}{'prev_q':>11}{'change%':>9}"
        f"{'descending':>11}{'min_step':>9}",
    ]
    for arm, seed in _cells_in_order(records):
        summary = learning_curve_summary(records[(arm, seed)]["history"], window=window)
        for term in ("loss", *TERMS):
            t = summary["terms"][term]
            lines.append(
                f"  {arm:<12}{seed:>5}  {term:<11}{_num(t['last_quarter_mean'], '.4f'):>11}"
                f"{_num(t['preceding_quarter_mean'], '.4f'):>11}{_num(t['change_pct'], '+.2f'):>9}"
                f"{str(t['descending']):>11}{t['smoothed_min_step']:>9}"
            )
    return "\n".join(lines) + "\n"


def write_learning_curves(records: dict, figure: Path, window: int = 100) -> str:
    """One panel per term plus the summed loss; arms coloured, seeds as thin
    lines, KL_FREE_BITS on the KL panels. A missing or broken matplotlib, or
    an unwritable path, costs the FIGURE and nothing else (report_study's
    guard)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    colours = {"pixel_ae": "tab:orange", "frozen_ssl": "tab:blue", "random_vit": "tab:gray"}
    panels = ("loss", *TERMS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    try:
        for ax, term in zip(axes.flat, panels):
            for (arm, seed), record in records.items():
                history = record["history"]
                s = (np.asarray(history["loss"], dtype=float) if term == "loss"
                     else np.asarray([float(p[term]) for p in history["parts"]], dtype=float))
                w = min(window, s.size)
                smoothed = np.convolve(s, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(w, s.size + 1), smoothed, color=colours.get(arm, "black"),
                        linewidth=0.9, alpha=0.85, label=f"{arm} s{seed}")
            if term in ("kl_dyn", "kl_rep"):
                ax.axhline(KL_FREE_BITS, color="black", linestyle=":", linewidth=0.8, label="free bits")
            ax.set_title(term)
            ax.set_xlabel("training step")
            ax.set_ylabel(f"{window}-step mean")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=110)
    except (ValueError, OSError) as error:
        return f"figure NOT written: {error}"
    finally:
        plt.close(fig)
    return f"figure={figure}"


def readings_text(records: dict, *, arms, seeds, horizon: int, window: int, figure_line: str) -> str:
    """Everything below the per-cell lines, in `trust.txt`'s style; written to
    `split_gap.txt` byte-identical."""
    h, clamped = decision_horizon(horizon)
    inputs = gap_inputs(records, arms=arms, seeds=seeds, h=h)
    reading = reading_gap(inputs)
    note = (f"  NOTE: decision horizon clamped to the run's horizon h={h} "
            f"(pre-registered DECISION_H = {DECISION_H}).\n" if clamped else "")
    return "".join([
        _self_check_table(records),
        _strata_table(records, h),
        _band_table(records, arms, seeds),
        _survival_table(survival_by_arm_and_stratum(records, arms, seeds), horizon),
        _conditional_table(records, arms, seeds, horizon),
        note,
        f"  pooling: z_fam = cluster_threshold({FAMILY}, {min(_clusters(records, DECISION[0]), _clusters(records, DECISION[1]))}) = "
        f"{inputs.z_fam:.2f}; the survival indicator pools windows that moved within the horizon; "
        f"the probe channel pools measurable cells only.\n",
        format_reading_gap(reading, inputs),
        _reported_table(records, arms, seeds, horizon),
        _sensitivity_text(records, arms, seeds, h, inputs.z_fam),
        _per_seed_text(inputs),
        _learning_curve_table(records, window),
        f"  {figure_line}\n",
    ])


def write_readings(out_dir: Path, text: str) -> Path:
    path = Path(out_dir) / "split_gap.txt"
    path.write_text(text)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study_v2"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    # None -> each cell's diagnostic says what it was written at; a value that
    # disagrees is refused (`protocol_mismatch`, 14).
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--figure", type=Path, default=None,
                        help="learning-curve figure; default <out>/learning_curves.png")
    parser.add_argument("--window", type=int, default=100,
                        help="moving-mean window (steps) for the learning curves")
    return parser


def _run_cell(args, cell: Cell, device, train, val, strata: dict) -> tuple[int, dict | None]:
    """One cell: trust_horizon's checks and refit (12, 14, 14) through
    `prepare_cell`, then the val pass and 30, the two train passes, the record."""
    arm, seed = cell.arm, cell.seed
    status, prepared = prepare_cell(args, cell, device, train, val)
    if status != EXIT_OK:
        return status, None
    model, embedding_probe, common = prepared.model, prepared.embedding_probe, prepared.common

    # val FIRST: it is the anchor, and a failed self-check costs no train pass.
    traj = {"val": reference_trajectories(model, val, embedding_probe, **common)}
    check = self_check(traj["val"], cell.diagnostic)
    if not check.ok:
        print(
            f"\nSELF-CHECK FAILED for {arm} seed {seed}: " + "; ".join(check.failures())
            + ". Same windows, same rollout, same refit probe -- or the train strata "
            "would be read against a ruler M3d did not validate. No record written."
        )
        return EXIT_SELF_CHECK_FAILED, None
    for name in ("train_held", "train_probe"):
        traj[name] = reference_trajectories(model, strata[name], embedding_probe, **common)

    record = split_gap_record(
        cell, traj, strata, check,
        context=prepared.context, horizon=prepared.horizon, device=device,
    )
    path = write_split_gap_record(args.out, record)
    counts = "; ".join(
        f"{name} {traj[name].windows_total} windows / {len(strata[name])} episodes"
        for name in STRATA
    )
    print(
        f"{arm} seed {seed}: {counts}; self-check max|delta| reference "
        f"{check.reference_position_max_delta:.1e} persistence "
        f"{check.persistence_position_max_delta:.1e}; wrote {path}"
    )
    return EXIT_OK, record


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = get_device(prefer=args.device)
    try:
        cells = [load_cell(args.out, arm, seed) for arm in args.arms for seed in args.seeds]
    except CellMissing as error:
        print(f"NO CELL: {error}")
        return EXIT_NO_CHECKPOINTS

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    train, val = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=SPLIT_SEED
    )
    # The strata are the SAME for every cell (one split, one probe rule), so
    # they are built and judged once, before the first refit.
    used, _ = probe_episodes(train)
    try:
        strata = {"val": val, **strata_partition(buffer.episode_paths(), train, val, used)}
    except StrataNotAPartition as error:
        print(f"STRATA NOT A PARTITION: {error}")
        return EXIT_STRATA_NOT_A_PARTITION

    records: dict[tuple[str, int], dict] = {}
    horizon = 0
    for cell in cells:
        status, record = _run_cell(args, cell, device, train, val, strata)
        if status != EXIT_OK:
            return status
        records[(cell.arm, cell.seed)] = record
        horizon = int(record["horizon"])
    # The learning curves come from the STUDY records' history, which the
    # split-gap record does not copy; carry them beside each live record.
    for cell in cells:
        records[(cell.arm, cell.seed)]["history"] = cell.record["history"]
    figure = args.figure if args.figure is not None else args.out / "learning_curves.png"
    figure_line = write_learning_curves(records, figure, window=args.window)
    text = readings_text(
        records, arms=list(args.arms), seeds=[int(s) for s in args.seeds],
        horizon=horizon, window=args.window, figure_line=figure_line,
    )
    print(text, end="")
    write_readings(args.out, text)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
