import numpy as np
import pytest
from gymnasium import spaces

from mbfps.envs import registry
from mbfps.envs.protocol import OBS_SHAPE, EnvProtocol
from mbfps.envs.registry import make_env


class _FakeEnv:
    """Minimal conforming implementation, used to test the protocol itself."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.scenario = "my_way_home"
        self.button_names = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
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


def test_env_missing_scenario_fails_protocol():
    """`scenario` is episode provenance; a leaky boundary let it be optional."""

    class _NoScenario(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            del self.scenario

    assert not isinstance(_NoScenario(), EnvProtocol)


def test_make_env_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown environment 'nope'"):
        make_env("nope")


def test_make_env_lists_available_names_in_error():
    with pytest.raises(KeyError, match="vizdoom"):
        make_env("nope")


def test_try_register_builtins_swallows_missing_vizdoom_env(monkeypatch):
    def _raise():
        raise ModuleNotFoundError(
            "No module named 'mbfps.envs.vizdoom_env'",
            name="mbfps.envs.vizdoom_env",
        )

    monkeypatch.setattr(registry, "_register_builtins", _raise)
    registry._try_register_builtins()  # must not raise


def test_try_register_builtins_reraises_other_missing_module(monkeypatch):
    def _raise():
        raise ModuleNotFoundError(
            "No module named 'some_native_lib'", name="some_native_lib"
        )

    monkeypatch.setattr(registry, "_register_builtins", _raise)
    with pytest.raises(ModuleNotFoundError, match="some_native_lib"):
        registry._try_register_builtins()


def test_try_register_builtins_reraises_plain_import_error(monkeypatch):
    def _raise():
        raise ImportError("cannot import name 'VizdoomEnv'")

    monkeypatch.setattr(registry, "_register_builtins", _raise)
    with pytest.raises(ImportError, match="VizdoomEnv"):
        registry._try_register_builtins()
