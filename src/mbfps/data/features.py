"""Frozen-backbone patch-feature cache.

A backbone never updates, so each frame is encoded once at collection time and
the features are reused for every training step. The learned bottleneck lives
downstream, so this cache holds the raw `(n_patches, patch_dim)` grid rather
than a bottlenecked vector. The grid's geometry is a property of the backbone,
recorded in `BACKBONE_GEOMETRY`: the two ViTs emit an 8x8 grid of 384-d patch
tokens; the M2 pixel autoencoder emits one 2048-d vector, partitioned into 64
rows of 32 so the same per-row bottleneck downstream can consume it.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.utils.device import get_device

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

_MODEL_NAME = "facebook/dinov2-small"


def _load_pixel_ae(path: Path) -> torch.nn.Module:
    """Load the encoder half of an M2 `cnn` autoencoder checkpoint, frozen."""
    # Local import: `mbfps.models.encoders` reads `BACKBONE_GEOMETRY` from this
    # module at import time, so a top-level import here would be circular.
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    if not path.is_file():
        raise FileNotFoundError(
            f"pixel_ae checkpoint not found at {path}; M2 writes it with "
            "scripts/train_autoencoder.py --arm cnn (runs/ is gitignored)"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arm = checkpoint.get("arm")
    if arm != "cnn":
        # The tag is checked, not just the tensors: the feature arms' M2
        # checkpoints hold a BottleneckEncoder whose keys would fail the strict
        # load below anyway, but with a message about tensor names rather than
        # about which file was handed in.
        raise ValueError(
            f"pixel_ae needs the M2 'cnn' autoencoder, but {path} has arm={arm!r}"
        )
    prefix = "encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    encoder.load_state_dict(encoder_state)  # strict: a missing tensor raises
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def build_backbone(
    kind: str, seed: int = 0, checkpoint: Path | None = None
) -> "torch.nn.Module":
    """Construct a frozen backbone.

    Args:
        kind: one of `BACKBONES`.
        seed: RNG seed for `random_vit`. Ignored for `dinov2` and `pixel_ae`,
            whose weights are fixed. Verified: the same seed reproduces
            bit-identical weights.
        checkpoint: for `pixel_ae`, the M2 autoencoder file; None means
            `PIXEL_AE_CHECKPOINT`. Ignored for the ViT backbones.

    Raises:
        KeyError: if `kind` is not registered.
        FileNotFoundError: `pixel_ae` with no checkpoint at the path.
        ValueError: `pixel_ae` with a checkpoint whose `arm` is not `"cnn"`.
    """
    if kind not in BACKBONES:
        raise KeyError(f"unknown backbone {kind!r}; available: {list(BACKBONES)}")
    if kind == "dinov2":
        return AutoModel.from_pretrained(_MODEL_NAME)
    if kind == "pixel_ae":
        return _load_pixel_ae(
            PIXEL_AE_CHECKPOINT if checkpoint is None else Path(checkpoint)
        )
    torch.manual_seed(seed)
    return AutoModel.from_config(AutoConfig.from_pretrained(_MODEL_NAME))


def require_free_bytes(path: Path, needed: int) -> None:
    """Fail before writing if `path`'s filesystem cannot hold `needed` bytes.

    The DINOv2 cache for the M1 dataset is 2.93 GB and this machine has had as
    little as 2.3 GB free, so a caching run can plausibly fill the disk. Failing
    up front beats dying halfway and leaving a partial cache that later loads
    as a FileNotFoundError on some episodes but not others.

    Raises:
        OSError: if free space is less than `needed`.
    """
    free = shutil.disk_usage(Path(path)).free
    if free < needed:
        raise OSError(
            f"caching needs {needed / 1e9:.2f} GB free at {path}, "
            f"but only {free / 1e9:.2f} GB is available"
        )


class FeatureExtractor:
    """Encodes frames with a frozen backbone."""

    def __init__(
        self,
        backbone: str = "dinov2",
        device: str = "mps",
        seed: int = 0,
        checkpoint: Path | None = None,
    ) -> None:
        self.backbone = backbone
        self.device = get_device(prefer=device)
        self.model = (
            build_backbone(backbone, seed=seed, checkpoint=checkpoint)
            .to(self.device)
            .eval()
        )
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode `(N, 112, 112, 3)` uint8 frames to `(N, n_patches, patch_dim)`
        float16, with the geometry from `BACKBONE_GEOMETRY[self.backbone]`.

        Raises:
            ValueError: if `frames` does not match the expected shape, or the
                backbone emits something other than its registered geometry.
        """
        if frames.ndim != 4 or frames.shape[1:] != OBS_SHAPE:
            raise ValueError(
                f"expected frames of shape (N, {OBS_SHAPE}), got {frames.shape}"
            )
        n_patches, patch_dim = BACKBONE_GEOMETRY[self.backbone]
        if self.backbone == "pixel_ae":
            # Raw uint8 straight in, NO ImageNet statistics: `CNNEncoder.forward`
            # does its own `/255 - 0.5` and NHWC->NCHW, and M2 trained it on
            # exactly that. Normalising here would hand the frozen weights
            # inputs from a distribution they were never fit on.
            out = self.model(torch.from_numpy(frames).to(self.device))  # (N, 2048)
            if out.shape[1] != n_patches * patch_dim:
                raise ValueError(
                    f"pixel_ae encoder emitted {out.shape[1]} dims per frame, "
                    f"expected {n_patches} x {patch_dim} = {n_patches * patch_dim}"
                )
            patches = out.reshape(len(frames), n_patches, patch_dim)
        else:
            x = frames.astype(np.float32) / 255.0
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
            out = self.model(pixel_values=tensor).last_hidden_state
            patches = out[:, 1:, :]  # drop the CLS token
        if tuple(patches.shape[1:]) != (n_patches, patch_dim):
            raise ValueError(
                f"backbone {self.backbone!r} is registered as {(n_patches, patch_dim)} per frame "
                f"but emitted {tuple(patches.shape[1:])}; check that the input is "
                f"112x112 and the patch size is 14"
            )  # Task 1's wording, which its `test_encode_checks_its_output_against_the_registry` pins (registered first, emitted second)
        return patches.to(torch.float16).cpu().numpy()


def cache_episode_features(
    ep_path: Path, extractor: FeatureExtractor, batch_size: int = 32
) -> Path:
    """Encode an episode's frames and write a sibling feature file.

    The output filename is namespaced by `extractor.backbone` (see
    `mbfps.data.loader.feature_suffix`), so caches for different backbones
    coexist instead of one overwriting the other.

    Called by `scripts/collect.py --cache-features` as each episode is written.
    Frames are encoded in batches: an episode can run to a thousand frames, and
    a single forward pass over all of them would exhaust unified memory.

    Returns:
        Path to the written feature file.
    """
    ep_path = Path(ep_path)
    # allow_pickle is left at its default of False, matching load_episode's
    # rationale in episode.py: obs is a plain uint8 array and round-trips
    # through npz without pickle, so there is no need to enable arbitrary
    # code execution on load.
    with np.load(ep_path) as data:
        obs = data["obs"]
    chunks = [
        extractor.encode(obs[i : i + batch_size]) for i in range(0, len(obs), batch_size)
    ]
    features = np.concatenate(chunks, axis=0)
    from mbfps.data.loader import feature_suffix

    out_path = ep_path.with_suffix(feature_suffix(extractor.backbone))
    np.save(out_path, features)
    return out_path
