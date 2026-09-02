import numpy as np
import pytest
from gymnasium import spaces

from mbfps.envs.protocol import OBS_SHAPE, EnvProtocol
from mbfps.envs.registry import make_env


class _FakeEnv:
    """Minimal conforming implementation, used to test the protocol itself."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.closed = False

    def reset(self, *, seed=None):
        return np.zeros(OBS_SHAPE, dtype=np.uint8), {}

    def step(self, action):
        return np.zeros(OBS_SHAPE, dtype=np.uint8), 0.0, False, False, {}

    def close(self):
        self.closed = True

    @property
    def privileged_state(self):
        return {"health": 100.0}


def test_obs_shape_is_112x112x3():
    assert OBS_SHAPE == (112, 112, 3)


def test_conforming_class_satisfies_protocol():
    assert isinstance(_FakeEnv(), EnvProtocol)


def test_non_conforming_class_fails_protocol():
    class Incomplete:
        def reset(self, *, seed=None):
            return None, {}

    assert not isinstance(Incomplete(), EnvProtocol)


def test_make_env_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown environment 'nope'"):
        make_env("nope")


@pytest.mark.xfail(reason="ViZDoomEnv lands in Task 5", strict=True)
def test_make_env_lists_available_names_in_error():
    with pytest.raises(KeyError, match="vizdoom"):
        make_env("nope")
