# scripts/reconstruction_grid.py
"""Render original-vs-reconstruction pairs for a trained arm (milestone M2)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mbfps.data.buffer import ReplayBuffer  # noqa: E402
from mbfps.data.loader import SequenceLoader  # noqa: E402
from mbfps.training.autoencoder import AutoencoderModel, to_device  # noqa: E402
from mbfps.utils.config import ARMS, get_config  # noqa: E402
from mbfps.utils.device import get_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--run", type=Path, default=Path("runs/m2"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    cfg = get_config(args.arm, batch_size=args.samples, seq_len=1, device=args.device)
    device = get_device(prefer=args.device)
    model = AutoencoderModel(cfg).to(device)
    checkpoint = torch.load(
        args.run / f"autoencoder_{args.arm}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)
    loader = SequenceLoader(
        buffer,
        batch_size=args.samples,
        seq_len=1,
        seed=0,
        load_obs=True,
        load_features=model.input_kind == "features",
    )
    with torch.no_grad():
        reconstruction, target = model(to_device(loader.sample(), device))

    recon = reconstruction.cpu().numpy()
    orig = target.cpu().numpy().astype(np.float32) / 255.0
    mse = float(((recon - orig) ** 2).mean())

    n = args.samples
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8), squeeze=False)
    for i in range(n):
        axes[0][i].imshow(np.clip(orig[i], 0, 1))
        axes[1][i].imshow(np.clip(recon[i], 0, 1))
        for row in (0, 1):
            axes[row][i].set_xticks([])
            axes[row][i].set_yticks([])
    axes[0][0].set_ylabel("original", fontsize=11)
    axes[1][0].set_ylabel("reconstruction", fontsize=11)
    fig.suptitle(f"arm={args.arm}   pixel MSE={mse:.5f}", fontsize=13)
    fig.tight_layout()

    args.run.mkdir(parents=True, exist_ok=True)
    out_path = args.run / f"reconstruction_{args.arm}.png"
    fig.savefig(out_path, dpi=110)
    print(f"figure={out_path}")
    print(f"arm={args.arm} pixel_mse={mse:.5f} samples={n}")


if __name__ == "__main__":
    main()
