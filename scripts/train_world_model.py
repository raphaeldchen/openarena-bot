"""Train one arm's world model (milestone M3)."""

import argparse
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.training.world_model import train_world_model
from mbfps.utils.config import ARMS, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs/m3"))
    args = parser.parse_args()

    cfg = get_config(
        args.arm,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    history = train_world_model(cfg, buffer, out_dir=args.out)
    n = min(20, len(history["loss"]))
    print(f"arm={cfg.arm} seed={cfg.train.seed} steps={history['steps']}")
    print(f"seconds={history['seconds']:.0f}")
    print(f"steps_per_second={history['steps'] / history['seconds']:.2f}")
    print(f"loss_first{n}={sum(history['loss'][:n]) / n:.5f}")
    print(f"loss_last{n}={sum(history['loss'][-n:]) / n:.5f}")


if __name__ == "__main__":
    main()
