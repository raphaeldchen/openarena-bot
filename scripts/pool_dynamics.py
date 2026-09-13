"""Pool the nine diagnostic records into one table. NO MODEL, NO GPU.

`scripts/diagnose_dynamics.py` decides each of the nine cells on its own, and
on the held-action contrast the per-cell verdicts disagree where the per-cell
power is marginal: `random_vit` reads nominal, nominal, null across its seeds
while the sign is positive in 9 of 9. This script reads the PER-WINDOW series
those records now carry and prints, once:

  PER ARM x RUNG -- the three seeds AVERAGED PER WINDOW (229 rows), mean,
    the EPISODE-clustered standard error (episodes nest windows nest seeds,
    so one label set absorbs both the shared-window correlation and the seed
    replication; stacking the 687 rows gives the identical number), the
    naive one over the seed-averaged rows beside it labelled as NOT the
    ruler, and z; and the embedding-space ratio of pooled medians -- each
    cell in units of its own noise median, so the pool is scale-free --
    with its episode-cluster bootstrap interval. The inference is for THESE
    THREE SEEDS, over episodes: the seeds are a fixed factor and the
    effective replication is the 24 clusters, so z is read against t(23).
  BETWEEN ARMS -- treatment minus control (`frozen_ssl` minus `random_vit`
    unless the flags say otherwise), PAIRED per window: the seeds averaged
    within each arm and window first, then differenced, then clustered by
    episode. ON THE POSITION AND ANGLE CHANNELS THIS IS THE PATHWAY AS READ
    THROUGH EACH ARM'S OWN PROBE -- every cell fits its own ridge probe
    (selection R^2 0.36 on frozen_ssl, -0.036 on cnn), so a between-arm
    difference there is partly a probe difference. The PROBE-FREE between-arm
    estimand is the embedding-ratio contrast, which resamples both arms in
    step; and the milestone's question -- does the treatment share the
    pathway, exceed it, or lack it -- is answered by `pathway_reading` from
    the three verdicts (treatment responds, control responds, contrast
    sign), never from the contrast alone, because a null contrast between
    two arms that both read null is "neither", not "shares".
  THE FAMILY is every z comparison the table prints -- arms x rungs x
    channels in the per-arm block plus rungs x channels of contrasts, 24 on
    the default run -- and the threshold is the Bonferroni quantile of
    t(G - 1), G the cluster count. The bootstrap intervals on the ratios are
    uncorrected 95% intervals and are labelled as such.

WHAT IS REFUSED, EACH UNDER ITS OWN STATUS, BEFORE ANY ROW PRINTS. A missing
planned cell (this repo has been burned by tables that dropped one and looked
complete), a record whose `arm`/`seed` disagree with its file name, a record
written before the per-window series existed -- every record produced before
this change; the fix is to re-run the diagnostic -- two cells that do not score
the same windows on the same device, and a cell whose noise reference measured
no spread. The statuses are disjoint from `run_study.py` (0, 1, 3-6),
`report_study.py` (0, 7-10) and `diagnose_dynamics.py` (0, 11-17), and none is
1 (an uncaught traceback) or 2 (argparse's own).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from mbfps.eval.aggregate import SEEDS
from mbfps.eval.diagnostics import LADDER, ladder_order
from mbfps.eval.pooling import (
    CHANNELS,
    DegenerateNoise,
    IncompatibleCells,
    MislabelledRecord,
    MissingCell,
    StaleRecord,
    cluster_threshold,
    load_cells,
    pathway_reading,
    pool_ladder,
)
from mbfps.utils.config import ARMS

EXIT_OK = 0
EXIT_MISSING_CELL = 18
EXIT_INCOMPATIBLE_RECORDS = 19
EXIT_STALE_RECORD = 20
EXIT_MISLABELLED_RECORD = 21
EXIT_DEGENERATE_NOISE = 22
"""Disjoint from the other three scripts' -- see the module docstring."""

STATISTICS: tuple[str, ...] = ("position delta", "angle delta", "embedding ratio")


def _fmt(value: float, spec: str) -> str:
    """NaN and inf print as themselves, never as a number that looks measured."""
    return format(value, spec) if np.isfinite(value) else f"{value:>{spec.split('.')[0].lstrip('+')}}"


def arm_table(pooled: dict, arms, rungs) -> str:
    """Per arm x rung, three rows: the position and angle deltas (mean,
    clustered SE, the naive SE over the seed-averaged windows -- labelled in
    the header as not the ruler -- z, windows, clusters, seeds) and the
    embedding ratio (ratio, 95% cluster-bootstrap interval, bootstrap SE,
    rows, clusters). The rung order is `LADDER`'s, so the table reads as a
    ladder."""
    lines = [
        f"{'arm':<12}{'rung':<11}{'statistic':<16}{'estimate':>12}{'cluster se':>12}"
        f"{'naive se':>10}{'z':>8}{'95% ci (bootstrap)':>24}{'counts':>34}"
    ]
    for arm in arms:
        for rung in rungs:
            for channel, label in zip(CHANNELS, STATISTICS[:2]):
                entry = pooled["arms"][(arm, rung, channel)]
                counts = (
                    f"windows={entry.windows} clusters={entry.clusters} seeds={len(entry.seeds)}"
                )
                lines.append(
                    f"{arm:<12}{rung:<11}{label:<16}{_fmt(entry.mean, '+12.3f')}"
                    f"{_fmt(entry.se, '12.3f')}{_fmt(entry.se_independent, '10.3f')}"
                    f"{_fmt(entry.z, '+8.2f')}{'':>24}{counts:>34}"
                )
            ratio = pooled["ratios"][(arm, rung)]
            interval = f"[{_fmt(ratio.ci_low, '.3f')}, {_fmt(ratio.ci_high, '.3f')}]"
            lines.append(
                f"{arm:<12}{rung:<11}{STATISTICS[2]:<16}{_fmt(ratio.ratio, '12.3f')}"
                f"{_fmt(ratio.bootstrap_se, '12.3f')}{'':>10}{'':>8}{interval:>24}"
                f"{f'rows={ratio.rows} clusters={ratio.clusters}':>34}"
            )
    return "\n".join(lines)


def contrast_table(pooled: dict, rungs) -> str:
    """Treatment minus control per rung: the paired position and angle
    contrasts (mean, clustered SE, z, windows kept) and the ratio contrast
    with its paired-bootstrap interval."""
    lines = [
        f"{'rung':<11}{'statistic':<16}{'estimate':>12}{'cluster se':>12}{'z':>8}"
        f"{'95% ci (paired bootstrap)':>28}{'counts':>36}"
    ]
    for rung in rungs:
        for channel, label in zip(CHANNELS, STATISTICS[:2]):
            entry = pooled["contrasts"][(rung, channel)]
            lines.append(
                f"{rung:<11}{label:<16}{_fmt(entry.mean, '+12.3f')}{_fmt(entry.se, '12.3f')}"
                f"{_fmt(entry.z, '+8.2f')}{'':>28}"
                f"{f'windows={entry.windows} excluded={entry.windows_excluded} clusters={entry.clusters}':>36}"
            )
        ratio = pooled["ratio_contrasts"][rung]
        interval = f"[{_fmt(ratio.ci_low, '+.3f')}, {_fmt(ratio.ci_high, '+.3f')}]"
        lines.append(
            f"{rung:<11}{STATISTICS[2]:<16}{_fmt(ratio.contrast, '+12.3f')}"
            f"{_fmt(ratio.bootstrap_se, '12.3f')}{'':>8}{interval:>28}"
            f"{f'ratios {ratio.treatment_ratio:.3f} - {ratio.control_ratio:.3f}, windows={ratio.windows}':>36}"
        )
    return "\n".join(lines)


def _responds(entry, threshold: float) -> bool:
    """Does a pooled delta respond with the sign conditioning predicts:
    positive -- the intervened sequence imagining WORSE, or held FORWARD
    running further than held NOOP -- and past the cluster-robust family
    threshold. A NaN z (fewer than two clusters) never responds."""
    return bool(np.isfinite(entry.z) and entry.z > threshold)


def _contrast_sign(z: float, threshold: float) -> int:
    if not np.isfinite(z):
        return 0
    return 1 if z > threshold else -1 if z < -threshold else 0


def _interval_sign(low: float, high: float) -> int:
    """A bootstrap interval that excludes 0, as a sign; 0 when it covers 0
    or is undefined."""
    if not (np.isfinite(low) and np.isfinite(high)):
        return 0
    return 1 if low > 0.0 else -1 if high < 0.0 else 0


def pathway_table(pooled: dict, rungs, treatment: str, control: str, threshold: float) -> str:
    """Per rung and channel: the treatment's verdict, the control's, the
    contrast's sign, and the reading `pathway_reading` gives them.

    The position and angle lines are the pathway AS READ THROUGH EACH ARM'S
    OWN PROBE, and say so; the embedding line is the probe-free reading. On
    the delta channels "responds" is the pooled z past the family threshold
    with the predicted sign; on the embedding ratio the per-arm verdict is
    "nonzero" -- the bootstrap interval excludes 0, i.e. not action-blind --
    and the contrast sign is the paired interval's. Both verdicts, the sign
    and the reading are printed, because the reading is a function of all
    three and a reader shown the reading alone cannot tell "shares" from
    "neither".
    """
    lines = [
        f"{'rung':<11}{'statistic':<16}{treatment[:14]:>16}{control[:14]:>16}{'contrast':>10}"
        f"{'estimand':>12}{'reading':>14}   (treatment responds, control responds, contrast "
        "sign; own-probe = through each arm's own probe)"
    ]
    for rung in rungs:
        for channel, label in zip(CHANNELS, STATISTICS[:2]):
            t = _responds(pooled["arms"][(treatment, rung, channel)], threshold)
            c = _responds(pooled["arms"][(control, rung, channel)], threshold)
            sign = _contrast_sign(pooled["contrasts"][(rung, channel)].z, threshold)
            word = lambda flag: "responds" if flag else "null"  # noqa: E731
            lines.append(
                f"{rung:<11}{label:<16}{word(t):>16}{word(c):>16}{sign:>+10d}"
                f"{'own-probe':>12}{pathway_reading(t, c, sign):>14}"
            )
        ratios = pooled["ratios"]
        t = _interval_sign(ratios[(treatment, rung)].ci_low, ratios[(treatment, rung)].ci_high) > 0
        c = _interval_sign(ratios[(control, rung)].ci_low, ratios[(control, rung)].ci_high) > 0
        entry = pooled["ratio_contrasts"][rung]
        sign = _interval_sign(entry.ci_low, entry.ci_high)
        word = lambda flag: "nonzero" if flag else "zero"  # noqa: E731
        lines.append(
            f"{rung:<11}{STATISTICS[2]:<16}{word(t):>16}{word(c):>16}{sign:>+10d}"
            f"{'probe-free':>12}{pathway_reading(t, c, sign):>14}"
        )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/m3_study_v2"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--rungs", nargs="+", default=list(LADDER), choices=list(LADDER))
    parser.add_argument("--treatment", default="frozen_ssl", choices=list(ARMS))
    parser.add_argument("--control", default="random_vit", choices=list(ARMS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.rungs = ladder_order(args.rungs)
    if args.treatment == args.control:
        parser.error(f"--treatment and --control both name {args.treatment!r}")
    for arm in (args.treatment, args.control):
        if arm not in args.arms:
            parser.error(f"{arm!r} is the treatment or control but not among --arms {args.arms}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    # Every refusal is raised BEFORE any row is assembled, and each is
    # printed under its own status: a table that printed and then failed
    # would leave a reader with numbers a later line disowns.
    try:
        records = load_cells(args.out, args.arms, args.seeds)
        pooled = pool_ladder(
            records, args.arms, args.seeds, args.rungs, args.treatment, args.control,
            bootstrap=args.bootstrap, seed=args.bootstrap_seed,
        )
    except MissingCell as error:
        print(f"MISSING CELL: {error}")
        return EXIT_MISSING_CELL
    except MislabelledRecord as error:
        print(f"MISLABELLED RECORD: {error}")
        return EXIT_MISLABELLED_RECORD
    except StaleRecord as error:
        print(f"STALE RECORD: {error}")
        return EXIT_STALE_RECORD
    except IncompatibleCells as error:
        print(f"INCOMPATIBLE RECORDS: {error}")
        return EXIT_INCOMPATIBLE_RECORDS
    except DegenerateNoise as error:
        print(f"DEGENERATE NOISE REFERENCE: {error}")
        return EXIT_DEGENERATE_NOISE

    first = next(iter(records.values()))
    windows = int(first["windows"]["total"])
    episodes = int(np.unique(first["windows"]["episode"]).size)
    # EVERY z comparison the two tables print: the per-arm block and the
    # contrasts. The ratio rows carry uncorrected bootstrap intervals and
    # are not in the family; the header says so.
    family = len(args.arms) * len(args.rungs) * len(CHANNELS) + len(args.rungs) * len(CHANNELS)
    threshold = cluster_threshold(family, episodes)
    print(
        f"--- pooled ladder: n_seeds={len(args.seeds)} seeds per arm, n_windows={windows}, "
        f"n_episodes={episodes}; seeds averaged per window and clustered by episode -- "
        f"for these three seeds, over episodes; the naive se is over the seed-averaged "
        f"windows and is not the ruler; device={first['device']} "
        f"torch={first['torch_version']} ---"
    )
    print(
        f"family: {family} comparisons ({len(args.arms)} arms x {len(args.rungs)} rungs x "
        f"{len(CHANNELS)} channels, plus {len(args.rungs)} x {len(CHANNELS)} contrasts); "
        f"family-wise threshold z={threshold:.2f}, read against t({episodes - 1}) because "
        f"the effective replication is the {episodes} clusters; embedding ratio: median "
        f"numerator / median noise with each cell in units of its own noise median, 0 = "
        f"action-blind, 1 = the two-draw distance; its intervals are uncorrected 95% "
        f"cluster bootstraps, bootstrap={args.bootstrap} bootstrap_seed={args.bootstrap_seed}"
    )
    print(arm_table(pooled, args.arms, args.rungs))
    # The pair is named from the pooled result, not from the flags, so the
    # header cannot claim a contrast the table did not compute.
    contrast = pooled["contrasts"][(args.rungs[0], CHANNELS[0])]
    print(
        f"\n--- {contrast.treatment} - {contrast.control}, paired per window (seeds "
        "averaged within each arm and window first), clustered by episode; position and "
        "angle are the pathway as read through each arm's own probe, the embedding ratio "
        "is the probe-free contrast and resamples both arms in step ---"
    )
    print(contrast_table(pooled, args.rungs))
    print(
        f"\n--- pathway reading: {contrast.treatment} against {contrast.control}, from "
        "(treatment responds, control responds, contrast sign) ---"
    )
    print(pathway_table(pooled, args.rungs, contrast.treatment, contrast.control, threshold))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
