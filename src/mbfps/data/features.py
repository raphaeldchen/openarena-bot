"""Frozen DINOv2 patch-feature cache.

The backbone never updates, so each frame is encoded once at collection time and
the features are reused for every training step. The learned bottleneck lives
downstream, so this cache holds the raw 64x384 patch grid rather than a
bottlenecked vector.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.utils.device import get_device

N_PATCHES = 64
"""112 / 14 = 8, so DINOv2's patch grid is 8x8."""

FEATURE_DIM = 384
"""Hidden size of DINOv2-small."""

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FeatureExtractor:
    """Encodes frames with a frozen self-supervised backbone."""

    def __init__(
        self, model_name: str = "facebook/dinov2-small", device: str = "mps"
    ) -> None:
        self.device = get_device(prefer=device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode `(N, 112, 112, 3)` uint8 frames to `(N, 64, 384)` float16.

        Raises:
            ValueError: if `frames` does not match the expected shape.
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
        if patches.shape[1] != N_PATCHES:
            raise ValueError(
                f"expected {N_PATCHES} patch tokens, got {patches.shape[1]}; "
                f"check that the input is 112x112 and the patch size is 14"
            )
        return patches.to(torch.float16).cpu().numpy()


def cache_episode_features(
    ep_path: Path, extractor: FeatureExtractor, batch_size: int = 32
) -> Path:
    """Encode an episode's frames and write a sibling `.features.npy`.

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
    out_path = ep_path.with_suffix(".features.npy")
    np.save(out_path, features)
    return out_path
