"""Per-policy state-visitation histogram and per-episode coverage gate.

Coverage gaps are invisible at the milestone that creates them and only become
symptomatic three milestones later, as an M4 agent with high imagined return and
near-zero real return. This makes them visible now.

The gate compares *per-episode* reach (occupied cells within a single episode),
not aggregate reach pooled across many episodes. Aggregate cell counts scale
with maze size and episode count: once enough episodes have been collected on
a small, fixed maze like `my_way_home`, both policies fill nearly all of the
reachable grid and the aggregate count saturates -- at that point it measures
how big the maze is and how much data was collected, not how good the policy
is. Per-episode reach does not have this problem: it asks how much ground one
episode of a given policy covers, which is exactly what the scripted policy
exists to improve over random.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbfps.data.buffer import ReplayBuffer  # noqa: E402


def _occupied_mask(xy: np.ndarray, bins: int, x_range, y_range) -> np.ndarray:
    hist = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[x_range, y_range])[0]
    return hist > 0


def _occupied_count(xy: np.ndarray, bins: int, x_range, y_range) -> int:
    return int(_occupied_mask(xy, bins, x_range, y_range).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--gate-ratio",
        type=float,
        default=1.2,
        help=(
            "Pass/fail threshold for the per-episode-reach gate: the mean "
            "occupied-cell count reached per scripted episode must exceed "
            "the mean per random episode by at least this multiplicative "
            "margin, i.e. mean(scripted) > mean(random) * ratio. The gate "
            "does not depend on a frame budget -- episode length is itself "
            "part of what a policy controls, so per-episode frame counts "
            "are not equalised."
        ),
    )
    args = parser.parse_args()

    episodes = ReplayBuffer(args.data, capacity_transitions=10**9).load_all()
    if not episodes:
        raise SystemExit(f"no episodes found in {args.data}")

    keys = episodes[0].privileged_keys
    x_i, y_i = keys.index("pos_x"), keys.index("pos_y")

    by_policy: dict[str, list[np.ndarray]] = defaultdict(list)
    for ep in episodes:
        by_policy[ep.policy_name].append(ep.privileged[:, [x_i, y_i]])

    names = sorted(by_policy)
    stacked = {n: np.concatenate(by_policy[n]) for n in names}

    # Shared bin range across all policies and episodes so that every
    # occupied-cell count below -- per-episode, aggregate, or union -- refers
    # to the same grid and is directly comparable.
    all_xy = np.concatenate(list(stacked.values()))
    x_range = (all_xy[:, 0].min(), all_xy[:, 0].max())
    y_range = (all_xy[:, 1].min(), all_xy[:, 1].max())

    # --- primary gate: per-episode reach -------------------------------
    # Frame counts differ slightly per episode; they are intentionally NOT
    # equalised here -- episode length is itself part of what the policy
    # controls, and all episodes in this dataset run to the same timeout,
    # so a policy that reaches more of the maze before timing out is doing
    # exactly what it is supposed to do.
    per_episode_counts: dict[str, np.ndarray] = {}
    for name in names:
        per_episode_counts[name] = np.array(
            [_occupied_count(xy, args.bins, x_range, y_range) for xy in by_policy[name]],
            dtype=float,
        )

    print("--- per-episode reach (the gate) ---")
    means: dict[str, float] = {}
    for name in names:
        counts = per_episode_counts[name]
        means[name] = float(counts.mean())
        sd = float(counts.std(ddof=0))
        print(
            f"{name}: n_episodes={len(counts)} "
            f"mean_cells_per_episode={counts.mean():.1f} sd={sd:.1f}"
        )

    have_both = "random" in means and "scripted" in means
    ratio = means["scripted"] / means["random"] if have_both else float("nan")
    if have_both:
        print(f"ratio (mean scripted / mean random) = {ratio:.3f}")

    # --- union analysis: does more of the same data buy coverage? ------
    print()
    print("--- union analysis (does more of the same data buy coverage?) ---")
    masks = {n: _occupied_mask(stacked[n], args.bins, x_range, y_range) for n in names}
    if have_both:
        random_mask, scripted_mask = masks["random"], masks["scripted"]
        random_alone = int(random_mask.sum())
        scripted_alone = int(scripted_mask.sum())
        scripted_only = int((scripted_mask & ~random_mask).sum())
        union = int((random_mask | scripted_mask).sum())
        pct_over_random = 100.0 * (union - random_alone) / random_alone
        print(f"random alone            : {random_alone} cells")
        print(f"scripted alone          : {scripted_alone} cells")
        print(f"scripted-only cells     : {scripted_only}   (states random never reaches)")
        print(
            f"union                   : {union}   "
            f"(+{pct_over_random:.1f}% over random alone)"
        )
    else:
        print("union analysis requires both 'random' and 'scripted' policies; skipping")

    # --- budget ladder: DIAGNOSTIC ONLY, not the gate ------------------
    # Occupied-cell counts pooled across many episodes scale with sample
    # size and, on a small fixed maze, saturate once the reachable area is
    # filled -- both policies eventually cover most of the grid and the
    # ladder flattens out regardless of policy quality. Kept here only as
    # a diagnostic; see the per-episode reach above for the gate.
    max_budget = min(len(v) for v in stacked.values())
    ladder = [b for b in (2_000, 5_000, 10_000, 25_000) if b <= max_budget]
    ladder.append(max_budget)
    ladder = sorted(set(ladder))

    rng = np.random.default_rng(0)
    ladder_counts: dict[int, dict[str, int]] = {}
    max_budget_sampled: dict[str, np.ndarray] = {}
    for budget in ladder:
        counts = {}
        for name in names:
            xy = stacked[name]
            sub = xy[rng.choice(len(xy), size=budget, replace=False)]
            counts[name] = _occupied_count(sub, args.bins, x_range, y_range)
            if budget == max_budget:
                max_budget_sampled[name] = sub
        ladder_counts[budget] = counts

    # Figure is drawn at the max equalised budget, as before.
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        xy = max_budget_sampled[name]
        ax.hist2d(xy[:, 0], xy[:, 1], bins=args.bins, range=[x_range, y_range])
        ax.set_title(f"{name}  (n={len(xy)}, budget={max_budget})")
        ax.set_xlabel("pos_x")
        ax.set_ylabel("pos_y")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"coverage_{args.data.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)

    print()
    print(f"figure={out_path}")
    print(f"max_equalised_budget={max_budget}")
    for name in names:
        xy = max_budget_sampled[name]
        print(
            f"{name}: total_frames={len(stacked[name])} "
            f"occupied_cells={ladder_counts[max_budget][name]}/{args.bins ** 2} "
            f"(at {max_budget} frames) "
            f"x_span={np.ptp(xy[:, 0]):.1f} y_span={np.ptp(xy[:, 1]):.1f}"
        )

    print()
    print("--- aggregate coverage budget ladder (DIAGNOSTIC ONLY -- not the gate) ---")
    print(
        "Aggregate coverage saturates once episodes have filled the reachable "
        "area, so it tracks maze size and episode count rather than policy "
        "quality; it is not used for pass/fail."
    )
    print(f"{'budget':>9}  {'random':>8}  {'scripted':>8}  {'ratio':>6}")
    for budget in ladder:
        counts = ladder_counts[budget]
        agg_ratio = counts.get("scripted", 0) / max(counts.get("random", 1), 1)
        print(
            f"{budget:>9}  {counts.get('random', 0):>8}  "
            f"{counts.get('scripted', 0):>8}  {agg_ratio:>6.3f}"
        )

    # --- gate: mean per-episode reach -----------------------------------
    print()
    if not have_both:
        raise SystemExit(
            "coverage gate requires both 'random' and 'scripted' policies in the dataset"
        )
    passed = means["scripted"] > means["random"] * args.gate_ratio
    print(
        f"GATE: {'PASS' if passed else 'FAIL'} "
        f"(mean_scripted={means['scripted']:.1f}, mean_random={means['random']:.1f}, "
        f"ratio={ratio:.3f}, required_ratio={args.gate_ratio:.2f})"
    )
    if not passed:
        raise SystemExit(
            f"coverage gate failed: mean(scripted)={means['scripted']:.1f} does not "
            f"exceed mean(random)={means['random']:.1f} * {args.gate_ratio:.2f} "
            f"(ratio={ratio:.3f} < required {args.gate_ratio:.2f})"
        )


if __name__ == "__main__":
    main()
