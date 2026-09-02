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
    # survives longer looks like it explores more. Subsample every policy to the
    # same frame count before comparing reach.
    rng = np.random.default_rng(0)
    budget = min(len(v) for v in stacked.values())
    sampled = {
        k: v[rng.choice(len(v), size=budget, replace=False)] for k, v in stacked.items()
    }

    all_xy = np.concatenate(list(stacked.values()))
    x_range = (all_xy[:, 0].min(), all_xy[:, 0].max())
    y_range = (all_xy[:, 1].min(), all_xy[:, 1].max())

    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        xy = sampled[name]
        ax.hist2d(xy[:, 0], xy[:, 1], bins=args.bins, range=[x_range, y_range])
        ax.set_title(f"{name}  (n={len(xy)})")
        ax.set_xlabel("pos_x")
        ax.set_ylabel("pos_y")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"coverage_{args.data.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)

    print(f"figure={out_path}")
    print(f"comparison_budget={budget} frames per policy (equalised)")
    for name in names:
        xy = sampled[name]
        occupied = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=args.bins, range=[x_range, y_range]
        )[0]
        print(
            f"{name}: total_frames={len(stacked[name])} "
            f"occupied_cells={int((occupied > 0).sum())}/{args.bins ** 2} "
            f"(at {budget} frames) "
            f"x_span={np.ptp(xy[:, 0]):.1f} y_span={np.ptp(xy[:, 1]):.1f}"
        )


if __name__ == "__main__":
    main()
