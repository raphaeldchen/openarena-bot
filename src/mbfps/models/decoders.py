"""Pixel decoder.

For the pixel arm this is part of the trained model: reconstruction is the
objective that shapes the latent. For the feature arms it is a visualisation
decoder that makes a feature-space embedding viewable, so the arms can be
compared side by side at all.

`in_dim` is a parameter rather than a constant because a later milestone feeds
this a concatenated recurrent state instead of a bare embedding.
"""

import torch
import torch.nn as nn

_SPATIAL = 7
"""Matches the encoder: four stride-2 steps between 7x7 and 112x112."""


class PixelDecoder(nn.Module):
    """Maps a latent vector back to a 112x112x3 image in [0, 1]."""

    def __init__(self, in_dim: int, depth: int = 32) -> None:
        super().__init__()
        channels = depth * 8
        self.project = nn.Linear(in_dim, channels * _SPATIAL * _SPATIAL)
        self._channels = channels
        layers = []
        for multiple in (4, 2, 1):
            out = depth * multiple
            layers += [
                nn.ConvTranspose2d(channels, out, 4, stride=2, padding=1),
                nn.SiLU(),
            ]
            channels = out
        layers.append(nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Map `(N, in_dim)` to `(N, 112, 112, 3)` float32 in [0, 1]."""
        x = self.project(latent).view(-1, self._channels, _SPATIAL, _SPATIAL)
        image = torch.sigmoid(self.deconv(x))
        return image.permute(0, 2, 3, 1)


def reconstruction_loss(pred: torch.Tensor, target_obs: torch.Tensor) -> torch.Tensor:
    """Mean squared error between a prediction in [0, 1] and an image target.

    A `uint8` target is normalised here so no caller has to remember to scale.
    A float target must already be in [0, 1]; one still in [0, 255] is rejected
    rather than silently used, because that produces a ~255^2 loss inflation
    that looks like a diverging model rather than a units bug. Checking the
    dtype alone missed exactly that case.

    Raises:
        ValueError: if a float target contains values above 1.
    """
    if target_obs.dtype == torch.uint8:
        target = target_obs.to(pred.dtype) / 255.0
    else:
        target = target_obs.to(pred.dtype)
        if target.numel() and float(target.max()) > 1.0 + 1e-4:
            raise ValueError(
                f"float target has max {float(target.max()):.4f}, expected [0, 1]. "
                "Pass a uint8 tensor to have it normalised, or normalise before "
                "calling."
            )
    diff = (pred - target).square()
    # `.mean()` on a zero-element tensor is NaN (0 / 0), not 0 -- guard it the
    # same way the units check above guards `.max()`, so an empty batch stays
    # a valid, finite loss rather than silently poisoning a running average.
    return diff.mean() if diff.numel() else diff.sum()
