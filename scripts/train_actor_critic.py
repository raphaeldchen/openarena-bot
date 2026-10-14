"""Train the actor-critic in a frozen world model's imagination (M4 Stage B)."""

import argparse
import sys
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.eval.study import git_sha, write_record
from mbfps.training.ac_trainer import ACConfig, train_actor_critic
from mbfps.training.world_model import load_world_model
from mbfps.utils.device import get_device


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--world-model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--updates", type=int, default=10_000)
    p.add_argument("--checkpoint-every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    args = p.parse_args(argv)

    device = get_device(prefer=args.device)
    model = load_world_model(args.world_model, device)
    cfg = ACConfig(updates=args.updates, checkpoint_every=args.checkpoint_every,
                   seed=args.seed, device=args.device)
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    history = train_actor_critic(model, buffer, cfg, out_dir=args.out)
    record = {k: v for k, v in history.items()}
    record.update({"world_model": str(args.world_model), "data": str(args.data),
                   "config": cfg.__dict__, "git_sha": git_sha()})
    path = args.out / f"train_{history['arm']}_seed{args.seed}.json"
    write_record(path, record)
    print(f"seconds_per_update={history['seconds_per_update']:.4f} isolation_ok={history['isolation_ok']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
