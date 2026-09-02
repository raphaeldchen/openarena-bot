"""The modular engine hook.

`EnvProtocol` is the single boundary between the learning code and any game
engine. ViZDoom implements it at M0; OpenArena implements it at M7. If adding a
new engine requires editing anything outside its own module, this abstraction is
wrong and should be fixed rather than worked around.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from gymnasium import spaces

OBS_SHAPE: tuple[int, int, int] = (112, 112, 3)
"""Observation shape, HWC uint8 RGB.

112 = 8 * 14, so DINOv2's patch-14 backbone yields exactly an 8x8 patch grid.
Every arm of the study consumes byte-identical frames at this resolution.
"""


@runtime_checkable
class EnvProtocol(Protocol):
    """A seedable, closable, Gymnasium-shaped environment."""

    observation_space: spaces.Box
    action_space: spaces.Discrete

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode. Returns (observation, info)."""
        ...

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one step. Returns (obs, reward, terminated, truncated, info)."""
        ...

    def close(self) -> None:
        """Release engine resources."""
        ...

    @property
    def privileged_state(self) -> dict[str, float] | None:
        """Ground-truth engine state.

        EVALUATION PROBES ONLY -- never a training input. Leaking these values
        into an observation would silently invalidate the entire study, so
        `tests/test_privileged_isolation.py` asserts they never appear in a
        training tensor.
        """
        ...
