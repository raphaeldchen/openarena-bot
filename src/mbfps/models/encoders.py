"""Encoders -- the ONLY module that differs between the study's three arms.

Every study arm applies the same small learned bottleneck to rows cached from
a frozen backbone: `pixel_ae` from the M2 pixel autoencoder's encoder,
`frozen_ssl` from DINOv2, `random_vit` from an untrained ViT. The arms differ
in exactly one thing -- what the backbone was pretrained on -- and the row
geometry each reads comes from `BACKBONE_GEOMETRY`.

`CNNEncoder`, the end-to-end pixel encoder, is M3b's retired `cnn` arm and
M2's autoencoder encoder. It stays constructible under `kind="cnn"` (M2's
scripts and tests build it, and the shipped M3b checkpoints load through it)
but `cnn` is not in `ARMS`, so no M3 tool can select it.

All four emit a 2048-dimensional embedding from byte-identical 112x112 frames,
so resolution and embedding width are controlled and only the representation
differs. If any other module ever needs to know which arm is running, the
comparison has stopped being controlled.
"""

import torch
import torch.nn as nn

from mbfps.data.geometry import BACKBONE_GEOMETRY, geometry_mismatch
from mbfps.utils.config import EncoderConfig

_SPATIAL = 7
"""112 / 2^4 = 7, the spatial size after four stride-2 convolutions."""


class CNNEncoder(nn.Module):
    """Learned convolutional encoder over raw pixels (M2's encoder; M3b's retired `cnn` arm).

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


def encoder_input_kind(cfg: EncoderConfig) -> str:
    """Whether this encoder consumes `"obs"` or cached `"features"`.

    `"obs"` iff `cfg.kind == "cnn"`. Every study arm -- `pixel_ae` included,
    its M2 encoder having been run over the frames at cache time -- reads
    features; the one pixel-consuming encoder is the retired kind.
    """
    return "obs" if cfg.kind == "cnn" else "features"


_KIND_BACKBONE: dict[str, str | None] = {
    "cnn": None,
    "pixel_ae": "pixel_ae",
    "frozen_ssl": "dinov2",
    "random_vit": "random_vit",
}
"""Which cached feature set each kind reads. None means the encoder reads pixels.

The arm name and the backbone name are deliberately not assumed equal:
`frozen_ssl` reads the `dinov2` cache. Deriving one from the other by string
identity would silently send the treatment arm to a cache that does not exist.
`pixel_ae` and `random_vit` happen to share their backbone's name; that is a
coincidence the tests name, not a rule this table relies on.
"""


def encoder_backbone(cfg: EncoderConfig) -> str | None:
    """Backbone whose cached features this kind consumes, or None for pixels.

    Raises:
        KeyError: if `cfg.kind` is not a registered kind.
    """
    if cfg.kind not in _KIND_BACKBONE:
        raise KeyError(f"unknown encoder kind {cfg.kind!r}")
    return _KIND_BACKBONE[cfg.kind]


class BottleneckEncoder(nn.Module):
    """Learned per-patch bottleneck over frozen-backbone features (every study arm).

    The backbone never updates, so its features are cached once at collection
    time and this module is the only trained part of the encoder path. A
    backbone's grid is far wider than the shared embedding (a ViT's is
    64 x 384 = 24576), so a per-row linear reduces each row to `bottleneck_dim`
    and the grid is flattened to exactly `embed_dim`.

    The grid's `(n_patches, patch_dim)` is the BACKBONE's property, read from
    `BACKBONE_GEOMETRY` for the backbone `cfg.kind` maps to. `bottleneck_dim`
    and `embed_dim` stay shared in `EncoderConfig`, so the identity
    `n_patches * bottleneck_dim == embed_dim` is checked per backbone.

    Every feature arm builds this same class; only the cached inputs differ.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        backbone = encoder_backbone(cfg)
        if backbone is None:
            raise ValueError(
                f"encoder kind {cfg.kind!r} reads pixels; BottleneckEncoder needs "
                "a kind that reads a frozen backbone's cached features"
            )
        n_patches, patch_dim = BACKBONE_GEOMETRY[backbone]
        if n_patches * cfg.bottleneck_dim != cfg.embed_dim:
            raise ValueError(
                f"{n_patches} patches x bottleneck_dim {cfg.bottleneck_dim} "
                f"must equal embed_dim {cfg.embed_dim} (backbone {backbone!r})"
            )
        # Plain attributes, not buffers: they must not enter the state_dict,
        # which the arm-parity tests compare key-for-key across feature arms.
        self.backbone = backbone
        self.n_patches = n_patches
        self.patch_dim = patch_dim
        self.bottleneck = nn.Linear(patch_dim, cfg.bottleneck_dim)
        # Non-learnable on purpose: it must add no parameters and no arm-varying
        # behaviour, only equalise the input scale the caches arrive at.
        # DINOv2's cache has std 2.3559 and random_vit's 1.0000, so without this
        # the treatment and control arms train at effectively different learning
        # rates and the contrast is confounded with optimisation conditioning.
        self.norm: nn.Module = (
            nn.LayerNorm(patch_dim, elementwise_affine=False)
            if cfg.standardise_features
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map `(N, n_patches, patch_dim)` to `(N, embed_dim)` float32.

        Raises:
            ValueError: if the rows are not the registered geometry. The
                dangerous case is the wrong ROW COUNT with the right width:
                `Linear(384, 32)` consumes `(N, 32, 384)` without complaint
                and emits `(N, 1024)`, the wrong embedding width, which only
                surfaces as a shape error somewhere inside the RSSM. So the
                check is at this boundary and names the backbone it is for.
        """
        expected = (self.n_patches, self.patch_dim)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                geometry_mismatch(self.backbone, expected, tuple(features.shape[1:]))
                + f" (input shape {tuple(features.shape)})"
            )
        return self.bottleneck(self.norm(features.to(torch.float32))).flatten(1)


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    """Construct the encoder for `cfg.kind`.

    `cnn` -> `CNNEncoder`; every study arm -> the one `BottleneckEncoder`
    class, which reads its row geometry from `BACKBONE_GEOMETRY` for the
    kind's backbone. The routing is by explicit name, not `kind != "cnn"`,
    so an unregistered kind is a KeyError here rather than a KeyError from
    inside the registry lookup.

    Raises:
        KeyError: if `cfg.kind` is not a registered kind.
    """
    if cfg.kind == "cnn":
        return CNNEncoder(cfg)
    if cfg.kind in ("pixel_ae", "frozen_ssl", "random_vit"):
        return BottleneckEncoder(cfg)
    raise KeyError(f"unknown encoder kind {cfg.kind!r}")
