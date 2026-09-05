"""Frame preprocessing.

ViZDoom's screen buffer layout depends on the configured `ScreenFormat`, and the
layout has differed across releases. Rather than hard-coding an assumption, this
module detects CHW and transposes it, so a ViZDoom upgrade cannot silently feed
transposed frames into training.
"""

import cv2
import numpy as np

from mbfps.envs.protocol import OBS_SHAPE

_TARGET_WH = (OBS_SHAPE[1], OBS_SHAPE[0])  # cv2.resize takes (width, height)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Normalise a raw engine frame to `(112, 112, 3)` uint8 HWC RGB.

    Args:
        frame: HWC or CHW uint8 array with 3 colour channels.

    Raises:
        ValueError: if `frame` is not 3-dimensional with 3 channels.
    """
    if frame.ndim != 3:
        raise ValueError(f"expected 3 channels, got array with shape {frame.shape}")
    if frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] != 3:
        raise ValueError(f"expected 3 channels, got shape {frame.shape}")
    if frame.shape[:2] == OBS_SHAPE[:2]:
        return np.ascontiguousarray(frame, dtype=np.uint8)
    resized = cv2.resize(frame, _TARGET_WH, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)
