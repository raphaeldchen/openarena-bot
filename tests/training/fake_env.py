"""A deterministic stand-in for ViZDoom `basic` (M5a loop tests).

reset(seed) fixes a target position in 0..7; action 1 moves left, 2 moves
right, 3 shoots. Shooting on target terminates with +97; otherwise -4 a step
(-9 for a miss); after `limit` steps the episode truncates. Frames encode the
position and step, so features differ step to step."""

import numpy as np
from gymnasium import spaces

from mbfps.envs.protocol import OBS_SHAPE


class FakeEnv:
    def __init__(self, limit: int = 12) -> None:
        self.scenario = "fake_basic"
        self.action_space = spaces.Discrete(4)
        self.limit = limit
        self._pos = self._target = self._t = 0

    def reset(self, *, seed=None):
        rng = np.random.default_rng(seed)
        self._target, self._pos, self._t = int(rng.integers(0, 8)), 4, 0
        return self._obs(), {}

    def step(self, action: int):
        self._t += 1
        reward, terminated = -4.0, False
        if action == 1:
            self._pos = max(0, self._pos - 1)
        elif action == 2:
            self._pos = min(7, self._pos + 1)
        elif action == 3:
            if self._pos == self._target:
                reward, terminated = 97.0, True
            else:
                reward = -9.0
        truncated = (not terminated) and self._t >= self.limit
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        obs = np.zeros(OBS_SHAPE, dtype=np.uint8)
        obs[0, 0, 0], obs[0, 0, 1], obs[0, 0, 2] = self._pos, self._target, self._t
        return obs

    @property
    def privileged_state(self):
        return {"pos_x": float(self._pos), "pos_y": 0.0}

    def close(self) -> None:
        pass


def fake_encode(frames: np.ndarray) -> np.ndarray:
    out = np.zeros((len(frames), 64, 384), np.float16)
    out[:, 0, :3] = frames[:, 0, 0, :3]
    return out
