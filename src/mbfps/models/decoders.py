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

from mbfps.envs.protocol import OBS_SHAPE

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
    """Mean squared error between a prediction in [0, 1] and a uint8 target.

    The loss owns the uint8 conversion so no caller has to remember to scale;
    a forgotten division by 255 would train against a 255x-larger target and
    look like a diverging model rather than a units bug.
    """
    target = target_obs.to(pred.dtype)
    if target_obs.dtype == torch.uint8:
        target = target / 255.0
    return (pred - target).square().mean()
