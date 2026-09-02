"""Collect a mixed-policy ViZDoom dataset.

Per spec §5.2, a single random policy has poor state coverage. This alternates
`random` and `scripted` collection so the world model sees geometry random play
never reaches.
"""

import argparse
import time
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.collector import Collector
from mbfps.data.features import cache_episode_features
from mbfps.data.policies import RandomPolicy, ScriptedPolicy
from mbfps.envs.registry import make_env
from mbfps.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="my_way_home")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="encode each episode with the frozen DINOv2 backbone as it is written",
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    # A real episode is ~19.8 MB resident once loaded by SequenceLoader (526
    # frames of 112x112x3 uint8). At the default below (60,000 transitions,
    # ~114 episodes) that is ~2.3 GB resident -- comfortably inside a 16 GB
    # unified-memory machine shared with the model. Raise this only knowing
    # what you are buying: 200,000 transitions (~380 episodes) costs ~7.5 GB.
    parser.add_argument("--capacity", type=int, default=60_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    out = args.out or Path("data") / args.scenario
    buffer = ReplayBuffer(out, capacity_transitions=args.capacity)

    def env_factory():
        return make_env(
            "vizdoom", scenario=args.scenario, frame_skip=args.frame_skip
        )

    probe = env_factory()
    n_actions, button_names = probe.action_space.n, probe.button_names
    probe.close()

    collectors = {
        "random": Collector(
            env_factory, RandomPolicy(n_actions, seed=args.seed), args.max_steps
        ),
        "scripted": Collector(
            env_factory, ScriptedPolicy(button_names, seed=args.seed), args.max_steps
        ),
    }

    extractor = None
    if args.cache_features:
        from mbfps.data.features import FeatureExtractor

        extractor = FeatureExtractor()

    start, kept = time.perf_counter(), 0
    for i in range(args.episodes):
        name = "random" if i % 2 == 0 else "scripted"
        episode = collectors[name].collect_episode(seed=args.seed + i)
        if episode is not None:
            path = buffer.add(episode)
            if extractor is not None and path.is_file():
                cache_episode_features(path, extractor)
            kept += 1
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{args.episodes}] kept={kept} transitions={buffer.n_transitions}")

    elapsed = time.perf_counter() - start
    for collector in collectors.values():
        collector.close()

    print(f"episodes_kept={kept} crashes={sum(c.crash_count for c in collectors.values())}")
    print(f"transitions={buffer.n_transitions} elapsed_s={elapsed:.1f}")
    print(f"transitions_per_second={buffer.n_transitions / elapsed:.1f}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
