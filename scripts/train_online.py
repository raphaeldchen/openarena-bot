"""One M5a run on ViZDoom `basic` with live DINOv2 features (spec §3).

Resumes from OUT/state.pt if present. After a complete run, verifies the
record and deletes the replay's feature caches and resumable state (spec §9)
unless --keep-features."""

import argparse
import sys
from pathlib import Path

from mbfps.envs.registry import make_env
from mbfps.training.online import (OnlineConfig, delete_replay_features, train_online,
                                   verify_record)
from mbfps.utils.device import get_device


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("dreamer", "dqn"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--budget", type=int, default=100_000)
    p.add_argument("--prefill", type=int, default=5_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--keep-features", action="store_true")
    args = p.parse_args(argv)

    from mbfps.data.features import FeatureExtractor
    device = get_device(prefer=args.device)
    cfg = OnlineConfig(budget=args.budget, prefill=args.prefill, eval_every=args.eval_every)
    encode = FeatureExtractor(backbone="dinov2", device=device.type).encode
    env_factory = lambda: make_env("vizdoom", scenario="basic", frame_skip=4)
    record = train_online(args.arm, args.seed, args.out, env_factory, encode, device, cfg)
    problems = verify_record(record, cfg)
    if problems:
        print("INCOMPLETE: " + "; ".join(problems))
        return 1
    print(f"complete: {args.arm} seed {args.seed}, {record['update_rounds']} update rounds, "
          f"{record['seconds']:.0f} s")
    if not args.keep_features:
        n = delete_replay_features(args.out, cfg)
        print(f"deleted {n} feature/state files after verification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
