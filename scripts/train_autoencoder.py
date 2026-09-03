"""Train one arm's standalone autoencoder (milestone M2)."""

import argparse
import json
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.training.autoencoder import train_autoencoder
from mbfps.utils.config import ARMS, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs/m2"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cfg = get_config(
        args.arm,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        device=args.device,
    )
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    history = train_autoencoder(cfg, buffer, out_dir=args.out)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"history_{args.arm}.json").write_text(json.dumps(history))
    first = sum(history["loss"][:20]) / min(20, len(history["loss"]))
    last = sum(history["loss"][-20:]) / min(20, len(history["loss"]))
    print(f"arm={args.arm} steps={history['steps']} elapsed_s={history['seconds']:.0f}")
    print(f"steps_per_second={history['steps'] / history['seconds']:.2f}")
    print(f"loss_first20={first:.5f} loss_last20={last:.5f}")


if __name__ == "__main__":
    main()
