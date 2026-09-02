# tests/test_privileged_isolation.py
"""Privileged state must never reach a training tensor.

The spec makes `privileged_state` evaluation-only. A leak would let the world
model or policy read ground truth the agent could not observe, silently
invalidating every result. This is asserted rather than documented.
"""

import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")
SENTINEL = 12345.0


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(
        Episode(
            obs=np.zeros((81, *OBS_SHAPE), dtype=np.uint8),
            actions=np.zeros(80, dtype=np.int32),
            rewards=np.zeros(80, dtype=np.float32),
            terminated=np.zeros(80, dtype=bool),
            truncated=np.zeros(80, dtype=bool),
            privileged=np.full((81, len(KEYS)), SENTINEL, dtype=np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )
    )
    return buf


def test_default_batch_has_no_privileged_key(buffer):
    assert "privileged" not in SequenceLoader(buffer, 4, 16, seed=0).sample()


def test_sentinel_absent_from_every_default_batch_array(buffer):
    """No array in a training batch may contain the privileged sentinel."""
    batch = SequenceLoader(buffer, 4, 16, seed=0).sample()
    for name, array in batch.items():
        assert not np.any(
            np.asarray(array, dtype=np.float64) == SENTINEL
        ), f"privileged sentinel leaked into training array {name!r}"


def test_privileged_reachable_only_via_explicit_flag(buffer):
    batch = SequenceLoader(buffer, 4, 16, seed=0).sample(include_privileged=True)
    assert np.all(batch["privileged"] == SENTINEL)


def test_env_privileged_values_are_not_in_the_observation():
    """The engine's own observation must not encode privileged values."""
    from mbfps.envs.vizdoom_env import ViZDoomEnv

    env = ViZDoomEnv(scenario="my_way_home", frame_skip=4, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        state = env.privileged_state
        assert obs.dtype == np.uint8, "observations are pixels, not state vectors"
        assert obs.shape == OBS_SHAPE
        assert state is not None and set(state) == set(KEYS)
    finally:
        env.close()
