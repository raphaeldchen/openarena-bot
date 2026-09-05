# scripts/eval_reconstruction.py
"""Paired pixel-MSE evaluation across arms (milestone M2).

`reconstruction_grid.py` reports an MSE over one batch only -- 12 frames at the
default `--samples 6`, since a seq_len=1 window carries two frames. That is far
too small a sample to separate the arms. This script averages over many draws and,
because every arm replays the same seed sequence, compares them pairwise, so the
between-arm difference is measured on identical frames and per-draw sampling
noise cancels.

Two limits on how far the output can be read. These are training-set
reconstructions -- M2 has no held-out split. And each arm contributes a single
checkpoint, so the paired spread measures frame-sampling noise around two fixed
models, not between-arm variability: `sem` shrinks as `1/sqrt(draws)` without
bound, so it supports "these two checkpoints differ on this dataset" and not
"this arm is better". Ranking arms needs several training seeds each.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.models.encoders import encoder_backbone
from mbfps.training.autoencoder import AutoencoderModel, to_device
from mbfps.utils.config import ARMS, get_config
from mbfps.utils.device import get_device


def evaluate(arm: str, run: Path, buffer: ReplayBuffer, draws: int,
             batch_size: int, device: torch.device) -> np.ndarray:
    """Per-draw MSE for `arm`, one entry per seed in `range(draws)`."""
    cfg = get_config(arm, batch_size=batch_size, seq_len=1, device=str(device))
    model = AutoencoderModel(cfg).to(device)
    checkpoint = torch.load(
        run / f"autoencoder_{arm}.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    backbone = encoder_backbone(cfg.encoder)
    per_draw = []
    for seed in range(draws):
        loader = SequenceLoader(
            buffer,
            batch_size=batch_size,
            seq_len=1,
            seed=seed,
            load_obs=True,
            load_features=model.input_kind == "features",
            feature_backbone=backbone or "dinov2",
        )
        with torch.no_grad():
            reconstruction, target = model(to_device(loader.sample(), device))
        recon = reconstruction.cpu().numpy().astype(np.float32)
        orig = target.cpu().numpy().astype(np.float32) / 255.0
        per_draw.append(float(((recon - orig) ** 2).mean()))
    return np.array(per_draw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/m2_long"))
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = get_device(prefer=args.device)
    buffer = ReplayBuffer(args.data, capacity_transitions=10**9)

    results: dict[str, np.ndarray] = {}
    for arm in ARMS:
        draws = evaluate(arm, args.run, buffer, args.draws, args.batch_size, device)
        results[arm] = draws
        sem = draws.std(ddof=1) / np.sqrt(draws.size)
        # seq_len=1 yields two frames per sample.
        frames = draws.size * args.batch_size * 2
        print(
            f"{arm:<11} mse={draws.mean():.5f}  sd={draws.std(ddof=1):.5f}  "
            f"sem={sem:.6f}  frames={frames}"
        )

    print("\npaired differences (identical frames per draw):")
    for left, right in itertools.combinations(results, 2):
        diff = results[left] - results[right]
        sem = diff.std(ddof=1) / np.sqrt(diff.size)
        verdict = "SEPARATED" if abs(diff.mean()) > 3 * sem else "not separated"
        print(
            f"  {left} - {right}: {diff.mean():+.5f}  sem {sem:.6f}  "
            f"t={diff.mean() / sem:+.1f}  {verdict}"
        )


if __name__ == "__main__":
    main()
