"""Device selection.

MPS is the default target. `PYTORCH_ENABLE_MPS_FALLBACK` is set unconditionally
so that any operator without an MPS kernel silently falls back to CPU instead of
raising. Models in this project are small enough that `prefer="cpu"` is a viable
debugging path, not just a formality.

The env var is set BEFORE `import torch`. PyTorch reads it while initialising the
MPS backend, so setting it afterwards has no effect -- the variable would be
present in `os.environ` while the fallback stayed disabled, which looks correct
and is not.
"""

import os
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402  -- must follow the env var above
import torch  # noqa: E402  -- must follow the env var above


def get_device(prefer: str = "mps") -> torch.device:
    """Return the best available device, honouring `prefer`.

    Args:
        prefer: "mps", "cuda", or "cpu". Falls back to CPU when unavailable.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_TRAINING_KEYS = ("obs", "features", "actions", "rewards", "terminated", "truncated")
"""Keys allowed onto the compute device.

A whitelist, deliberately. `privileged` is absent and must stay absent: it is
evaluation-only, and keeping it off the device makes that a structural property
rather than a rule someone has to remember. `episode_index` and `window_start`
are bookkeeping and stay on the host.
"""


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move the whitelisted arrays of a loader batch onto `device`."""
    return {
        key: torch.from_numpy(value).to(device)
        for key, value in batch.items()
        if isinstance(value, np.ndarray) and key in _TRAINING_KEYS
    }
