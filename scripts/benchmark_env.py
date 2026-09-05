"""Measure headless ViZDoom throughput.

Records the M0 throughput number. Run before and after any change to the
environment or preprocessing path.
"""

import argparse
import time

from mbfps.envs.registry import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="my_way_home")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    env = make_env("vizdoom", scenario=args.scenario, frame_skip=args.frame_skip)
    try:
        env.reset(seed=0)
        start = time.perf_counter()
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset(seed=0)
        elapsed = time.perf_counter() - start
    finally:
        env.close()

    print(f"scenario={args.scenario} frame_skip={args.frame_skip}")
    print(f"steps={args.steps} elapsed_s={elapsed:.2f}")
    print(f"steps_per_second={args.steps / elapsed:.1f}")


if __name__ == "__main__":
    main()
