"""Per-policy state-visitation histogram.

Coverage gaps are invisible at the milestone that creates them and only become
symptomatic three milestones later, as an M4 agent with high imagined return and
near-zero real return. This makes them visible now.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbfps.data.buffer import ReplayBuffer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--gate-budget",
        type=int,
        default=5000,
        help=(
            "Frame budget the pass/fail gate is evaluated at. Occupied-cell "
            "counts saturate as the budget grows -- both policies eventually "
            "fill most of the reachable grid -- so gating at the largest "
            "budget the dataset allows becomes less informative the more "
            "data is collected. Gate at a mid-range budget instead; the "
            "largest ladder entry <= this value is used (falling back to "
            "the max equalised budget if the dataset is smaller)."
        ),
    )
    args = parser.parse_args()

    episodes = ReplayBuffer(args.data, capacity_transitions=10**9).load_all()
    if not episodes:
        raise SystemExit(f"no episodes found in {args.data}")

    keys = episodes[0].privileged_keys
    x_i, y_i = keys.index("pos_x"), keys.index("pos_y")

    positions: dict[str, list[np.ndarray]] = defaultdict(list)
    for ep in episodes:
        positions[ep.policy_name].append(ep.privileged[:, [x_i, y_i]])

    names = sorted(positions)
    stacked = {n: np.concatenate(positions[n]) for n in names}

    # Occupied-cell counts scale with sample size, so a policy that merely
    # survives longer looks like it explores more. Subsample every policy to a
    # common frame count before comparing reach. But equalising only at the
    # largest budget the data allows lets the metric saturate -- both policies
    # eventually fill most of the reachable grid and the real difference in
    # reach compresses away -- so we measure a ladder of budgets and gate on a
    # mid-range one where the counts still discriminate.
    max_budget = min(len(v) for v in stacked.values())
    ladder = [b for b in (2_000, 5_000, 10_000, 25_000) if b <= max_budget]
    ladder.append(max_budget)
    ladder = sorted(set(ladder))

    all_xy = np.concatenate(list(stacked.values()))
    x_range = (all_xy[:, 0].min(), all_xy[:, 0].max())
    y_range = (all_xy[:, 1].min(), all_xy[:, 1].max())

    rng = np.random.default_rng(0)
    ladder_counts: dict[int, dict[str, int]] = {}
    max_budget_sampled: dict[str, np.ndarray] = {}
    for budget in ladder:
        counts = {}
        for name in names:
            xy = stacked[name]
            sub = xy[rng.choice(len(xy), size=budget, replace=False)]
            occupied = np.histogram2d(
                sub[:, 0], sub[:, 1], bins=args.bins, range=[x_range, y_range]
            )[0]
            counts[name] = int((occupied > 0).sum())
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
    print(f"{'budget':>9}  {'random':>8}  {'scripted':>8}  {'ratio':>6}")
    for budget in ladder:
        counts = ladder_counts[budget]
        ratio = counts.get("scripted", 0) / max(counts.get("random", 1), 1)
        print(f"{budget:>9}  {counts.get('random', 0):>8}  "
              f"{counts.get('scripted', 0):>8}  {ratio:>6.3f}")

    gate_candidates = [b for b in ladder if b <= args.gate_budget]
    gate_budget = max(gate_candidates) if gate_candidates else max_budget
    gate_counts = ladder_counts[gate_budget]
    random_n = gate_counts.get("random", 0)
    scripted_n = gate_counts.get("scripted", 0)
    passed = scripted_n > random_n
    print(
        f"GATE: {'PASS' if passed else 'FAIL'} "
        f"(budget={gate_budget}, random={random_n}, scripted={scripted_n})"
    )
    if not passed:
        raise SystemExit(
            f"coverage gate failed: scripted={scripted_n} <= random={random_n} "
            f"at budget={gate_budget}"
        )


if __name__ == "__main__":
    main()
