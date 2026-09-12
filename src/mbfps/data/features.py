"""Frozen-backbone patch-feature cache.

The backbone never updates, so each frame is encoded once at collection time and
the features are reused for every training step. The learned bottleneck lives
downstream, so this cache holds the raw `(n_patches, patch_dim)` grid rather
than a bottlenecked vector. The grid's shape is the BACKBONE's property, read
from `BACKBONE_GEOMETRY`, never a constant a reader copies.
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

BACKBONES: tuple[str, ...] = ("dinov2", "random_vit")
"""Frozen backbones. `dinov2` is the treatment arm's pretrained encoder;
`random_vit` is the control -- identical architecture, random weights, which is
what separates "pretraining helps" from "a stationary target helps"."""

BACKBONE_GEOMETRY: dict[str, tuple[int, int]] = {
    "dinov2": (64, 384),
    "random_vit": (64, 384),
}
"""`(n_patches, patch_dim)` of each backbone's cached rows, per frame.

Both ViT backbones tile 112x112 at patch 14 into an 8x8 = 64 grid and emit
DINOv2-small's 384-wide hidden state. That is a fact about the backbone, not
a choice the study makes per arm, so it lives here beside `BACKBONES` and
every reader -- `FeatureExtractor.encode`'s output check, the encoder's
bottleneck width and its `n_patches * bottleneck_dim == embed_dim` guard, the
cache-size estimate in scripts/cache_features.py -- takes it from this dict.
Before this registry the same two numbers were a module constant here, a
module constant in encoders.py and a field on `EncoderConfig`; a backbone
whose rows are not 384 wide would have been built against 384 with no error
until the first matmul. Every key of this dict is in `BACKBONES` and vice
versa; tests/data/test_features.py pins both the literal and that identity.
"""

_MODEL_NAME = "facebook/dinov2-small"


def build_backbone(kind: str, seed: int = 0) -> "torch.nn.Module":
    """Construct a frozen backbone.

    Args:
        kind: one of `BACKBONES`.
        seed: RNG seed for `random_vit`. Ignored for `dinov2`, whose weights
            are fixed. Verified: the same seed reproduces bit-identical weights.

    Raises:
        KeyError: if `kind` is not registered.
    """
    if kind not in BACKBONES:
        raise KeyError(f"unknown backbone {kind!r}; available: {list(BACKBONES)}")
    if kind == "dinov2":
        return AutoModel.from_pretrained(_MODEL_NAME)
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
    """Encodes frames with a frozen self-supervised backbone."""

    def __init__(
        self, backbone: str = "dinov2", device: str = "mps", seed: int = 0
    ) -> None:
        self.backbone = backbone
        self.device = get_device(prefer=device)
        self.model = build_backbone(backbone, seed=seed).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode `(N, 112, 112, 3)` uint8 frames to `(N, n_patches, patch_dim)`
        float16, with the geometry taken from `BACKBONE_GEOMETRY[self.backbone]`.

        Raises:
            ValueError: if `frames` does not match the expected shape, or the
                backbone emits a grid other than the one registered for it.
        """
        if frames.ndim != 4 or frames.shape[1:] != OBS_SHAPE:
            raise ValueError(
                f"expected frames of shape (N, {OBS_SHAPE}), got {frames.shape}"
            )
        x = frames.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        out = self.model(pixel_values=tensor).last_hidden_state
        patches = out[:, 1:, :]  # drop the CLS token
        # The registry is the contract every downstream reader builds against,
        # so a backbone that emits anything else is refused HERE, before a
        # single cache file is written with the wrong rows in it.
        expected = BACKBONE_GEOMETRY[self.backbone]
        if tuple(patches.shape[1:]) != expected:
            raise ValueError(
                f"backbone {self.backbone!r} is registered as {expected} per frame "
                f"but emitted {tuple(patches.shape[1:])}; check that the input is "
                f"112x112 and the patch size is 14"
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
