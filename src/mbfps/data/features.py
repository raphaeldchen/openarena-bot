"""Frozen-backbone patch-feature cache.

A backbone never updates, so each frame is encoded once at collection time and
the features are reused for every training step. The learned bottleneck lives
downstream, so this cache holds the raw `(n_patches, patch_dim)` grid rather
than a bottlenecked vector. The grid's geometry is a property of the backbone,
recorded in `BACKBONE_GEOMETRY`: the two ViTs emit an 8x8 grid of 384-d patch
tokens; the M2 pixel autoencoder emits one 2048-d vector, partitioned into 64
rows of 32 so the same per-row bottleneck downstream can consume it.

The registry itself -- `BACKBONES`, `BACKBONE_GEOMETRY`, `PIXEL_AE_CHECKPOINT`
and the `geometry_mismatch` message -- lives in the leaf module
`mbfps.data.geometry` and is re-exported here unchanged, so that the encoder
and the loader can read it without paying for the `transformers` import below.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel

# Re-exported, NOT copied: `mbfps.data.geometry` is the registry's home and
# these are the same objects, so a test that rebinds an entry through this
# module rebinds the dict the encoder and the loader read.
from mbfps.data.geometry import (  # noqa: F401 (re-exports)
    BACKBONE_GEOMETRY,
    BACKBONES,
    PIXEL_AE_CHECKPOINT,
    geometry_mismatch,
)
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.encoders import CNNEncoder
from mbfps.utils.config import EncoderConfig
from mbfps.utils.device import get_device

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL_NAME = "facebook/dinov2-small"


def _load_pixel_ae(path: Path) -> torch.nn.Module:
    """Load the encoder half of an M2 `cnn` autoencoder checkpoint, frozen."""
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
            # The reshape is the whole guarantee: having passed the width
            # check it can only produce (n_patches, patch_dim) rows, so no
            # post-branch guard is needed on this side.
            patches = out.reshape(len(frames), n_patches, patch_dim)
        else:
            x = frames.astype(np.float32) / 255.0
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
            out = self.model(pixel_values=tensor).last_hidden_state
            patches = out[:, 1:, :]  # drop the CLS token
            if tuple(patches.shape[1:]) != (n_patches, patch_dim):
                # The patch-size hint is a ViT fact and lives in the ViT
                # branch: 112 / 14 = 8 is how the 64 comes about, and it
                # means nothing for the pixel autoencoder's one flat vector.
                raise ValueError(
                    geometry_mismatch(
                        self.backbone, (n_patches, patch_dim), tuple(patches.shape[1:])
                    )
                    + " Check that the input is 112x112 and the patch size is 14."
                )
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
