"""Encoders -- the ONLY module that differs between the study's three arms.

Arm 1 learns a CNN over pixels. Arms 2 and 3 apply a small learned bottleneck
to features from a frozen backbone, cached at collection time so no vision
transformer runs during training.

All three emit a 2048-dimensional embedding from byte-identical 112x112 frames,
so resolution and embedding width are controlled and only the representation
differs. If any other module ever needs to know which arm is running, the
comparison has stopped being controlled.
"""

import torch
import torch.nn as nn

from mbfps.utils.config import EncoderConfig

_SPATIAL = 7
"""112 / 2^4 = 7, the spatial size after four stride-2 convolutions."""

_N_PATCHES = 64
"""112 / 14 = 8, so a patch-14 backbone yields an 8x8 = 64 patch grid."""


class CNNEncoder(nn.Module):
    """Learned convolutional encoder over raw pixels (Arm 1).

    Takes uint8 straight from the loader and owns its own normalisation, so no
    caller has to remember to scale. Four stride-2 convolutions reduce
    112x112 to 7x7x256 = 12544, then a linear projection gives the shared
    2048-dimensional embedding.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        channels, layers = 3, []
        for multiple in (1, 2, 4, 8):
            out = cfg.cnn_depth * multiple
            layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1), nn.SiLU()]
            channels = out
        self.conv = nn.Sequential(*layers)
        self.project = nn.Linear(channels * _SPATIAL * _SPATIAL, cfg.embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map `(N, 112, 112, 3)` uint8 to `(N, embed_dim)` float32."""
        x = obs.to(torch.float32).div(255.0).sub(0.5).permute(0, 3, 1, 2)
        return self.project(self.conv(x).flatten(1))


class BottleneckEncoder(nn.Module):
    """Learned per-patch bottleneck over frozen-backbone features (Arms 2, 3).

    The backbone never updates, so its features are cached once at collection
    time and this module is the only trained part of the encoder path. A ViT's
    patch grid is far wider than the shared embedding (64 x 384 = 24576), so a
    per-patch linear reduces each patch to `bottleneck_dim` and the grid is
    flattened to exactly `embed_dim`.

    Arms 2 and 3 build the identical module; only the cached inputs differ.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        if _N_PATCHES * cfg.bottleneck_dim != cfg.embed_dim:
            raise ValueError(
                f"{_N_PATCHES} patches x bottleneck_dim {cfg.bottleneck_dim} "
                f"must equal embed_dim {cfg.embed_dim}"
            )
        self.bottleneck = nn.Linear(cfg.patch_dim, cfg.bottleneck_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map `(N, 64, 384)` to `(N, embed_dim)` float32."""
        return self.bottleneck(features.to(torch.float32)).flatten(1)


def encoder_input_kind(cfg: EncoderConfig) -> str:
    """Whether this arm's encoder consumes `"obs"` or cached `"features"`."""
    return "obs" if cfg.kind == "cnn" else "features"


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    """Construct the encoder for `cfg.kind`.

    Raises:
        KeyError: if `cfg.kind` is not a registered arm.
    """
    if cfg.kind == "cnn":
        return CNNEncoder(cfg)
    if cfg.kind in ("frozen_ssl", "random_vit"):
        return BottleneckEncoder(cfg)
    raise KeyError(f"unknown encoder kind {cfg.kind!r}")
