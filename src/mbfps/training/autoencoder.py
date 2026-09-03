"""Standalone autoencoder training -- milestone M2.

No recurrence. Most world-model bugs are representation bugs, and a broken
autoencoder is obvious in a reconstruction grid but hard to infer from a
degraded rollout, so this stage is validated before dynamics are added.

All three arms reconstruct the same pixel target, which is what makes their
grids comparable. The pixel arm trains its CNN; the feature arms train only
their bottleneck, because the backbone is frozen and already cached.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.data.prefetch import Prefetcher
from mbfps.models.decoders import PixelDecoder, reconstruction_loss
from mbfps.models.encoders import build_encoder, encoder_input_kind
from mbfps.utils.config import Config
from mbfps.utils.device import get_device
from mbfps.utils.seeding import seed_everything


class AutoencoderModel(nn.Module):
    """Encoder plus pixel decoder, with no recurrent state."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_kind = encoder_input_kind(cfg.encoder)
        self.encoder = build_encoder(cfg.encoder)
        self.decoder = PixelDecoder(
            in_dim=cfg.encoder.embed_dim, depth=cfg.encoder.cnn_depth
        )

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(reconstruction, pixel_target)`, both `(N, 112, 112, 3)`.

        The pixel target is always obs. The encoder *input* is obs only for the
        pixel arm; the feature arms read cached features and never see pixels.
        """
        target = batch["obs"]
        target = target.reshape(-1, *target.shape[-3:])
        if self.input_kind == "obs":
            source = target
        else:
            features = batch["features"]
            source = features.reshape(-1, *features.shape[-2:])
        return self.decoder(self.encoder(source)), target


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray) and key in ("obs", "features"):
            out[key] = torch.from_numpy(value).to(device)
    return out


def train_autoencoder(
    cfg: Config,
    buffer: ReplayBuffer,
    out_dir: Path | None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Train one arm's autoencoder and return its loss history.

    Args:
        cfg: the arm's configuration.
        buffer: the frozen M1 dataset.
        out_dir: where to write a checkpoint, or None to skip writing.
        log_every: print a progress line this often.

    Returns:
        A history dict with `"arm"`, `"steps"`, `"loss"` (per-step floats),
        and `"seconds"`.
    """
    seed_everything(cfg.train.seed)
    device = get_device(prefer=cfg.train.device)
    model = AutoencoderModel(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    needs_features = model.input_kind == "features"
    loader = SequenceLoader(
        buffer,
        batch_size=cfg.train.batch_size,
        seq_len=cfg.train.seq_len,
        seed=cfg.train.seed,
        load_obs=True,  # always: obs is the reconstruction target for every arm
        load_features=needs_features,
    )

    losses: list[float] = []
    start = time.perf_counter()
    with Prefetcher(loader, depth=2) as prefetcher:
        stream = iter(prefetcher)
        for step in range(cfg.train.steps):
            batch = to_device(next(stream), device)
            optimiser.zero_grad()
            reconstruction, target = model(batch)
            loss = reconstruction_loss(reconstruction, target)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
            if log_every and (step + 1) % log_every == 0:
                recent = float(np.mean(losses[-log_every:]))
                print(f"[{cfg.arm}] step {step + 1}/{cfg.train.steps} loss={recent:.5f}")

    elapsed = time.perf_counter() - start
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"arm": cfg.arm, "state_dict": model.state_dict()},
            out_dir / f"autoencoder_{cfg.arm}.pt",
        )
    return {
        "arm": cfg.arm,
        "steps": cfg.train.steps,
        "loss": losses,
        "seconds": elapsed,
    }
