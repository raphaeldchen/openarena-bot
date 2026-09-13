"""The backbone registry: what each frozen backbone emits, and where M2 left its encoder.

A LEAF MODULE, ON PURPOSE. `mbfps.data.features` imports `transformers` at
module top (measured 1.5-2.2 s), and the two modules that consult the
registry most -- `mbfps.models.encoders`, which sizes every `BottleneckEncoder`
from it, and `mbfps.data.loader`, which refuses a cache whose rows are not its
backbone's -- need these names and nothing else from `features`. Holding them
here means a model build or a loader construction never pays for transformers,
and `features` can import the models layer at module top without a cycle.
`features` re-exports all of them (the SAME objects, not copies), so
`mbfps.data.features.BACKBONE_GEOMETRY` remains a valid spelling and a test
that rebinds an entry through either name rebinds the one dict every reader
consults.
"""

from pathlib import Path

BACKBONES: tuple[str, ...] = ("dinov2", "random_vit", "pixel_ae")
"""Frozen backbones. `dinov2` is the treatment arm's pretrained encoder;
`random_vit` is the control -- identical architecture, random weights, which is
what separates "pretraining helps" from "a stationary target helps"; `pixel_ae`
is the encoder of the M2 pixel autoencoder trained on this very data, which
makes the pixel arm a frozen-target arm like the other two instead of the
end-to-end arm that M3b measured never leaving its collapsed initial state."""

BACKBONE_GEOMETRY: dict[str, tuple[int, int]] = {
    "dinov2": (64, 384),
    "random_vit": (64, 384),
    "pixel_ae": (64, 32),
}
"""`(n_patches, patch_dim)` of each backbone's cached rows.

112 / 14 = 8, so a patch-14 ViT yields an 8x8 = 64 grid of hidden-size-384
tokens. The pixel autoencoder's encoder ends in `Linear(12544 -> 2048)`, one
vector with no spatial meaning; it is reshaped row-major into 64 rows of 32 so
that `64 * bottleneck_dim(32) == embed_dim(2048)` holds for it exactly as for
the ViTs. Every consumer of a cache's shape -- the loader's validation, the
bottleneck's `Linear` width, the cache-size estimate -- reads this dict; there
is deliberately no module constant for "the" patch count or width any more."""

PIXEL_AE_CHECKPOINT: Path = Path("runs/m2_fixed/autoencoder_cnn.pt")
"""Where M2 left the pixel autoencoder (`scripts/train_autoencoder.py --arm cnn`).
Gitignored, so a fresh checkout has to retrain it or copy it in."""


def geometry_mismatch(
    backbone: str,
    expected: tuple[int, ...],
    got: tuple[int, ...],
    path: Path | str | None = None,
) -> str:
    """The one sentence every guard prints when rows are not `backbone`'s geometry.

    Four guards refuse rows of the wrong shape -- the loader from a cache's
    `.npy` header, `FeatureExtractor.encode` on what a ViT emitted,
    `BottleneckEncoder.forward` on what it was handed, and the spike's probe
    loader -- and they used to disagree on the order of `expected`/`got` and
    on whether a re-cache hint was offered. One message, expected before got,
    naming the backbone the caller believed it was serving. The re-cache hint
    is appended only when a `path` is given: a cache file on disk is fixed by
    re-caching that backbone, while a mismatch inside a forward pass is not.
    """
    subject = "the input" if path is None else f"feature cache {path}"
    text = (
        f"{subject} does not match backbone {backbone!r}: expected rows of shape "
        f"{tuple(expected)}, got {tuple(got)}."
    )
    if path is not None:
        text += f" Re-run scripts/cache_features.py --backbone {backbone}."
    return text
