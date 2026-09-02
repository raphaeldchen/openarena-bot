# tests/data/test_collector.py
import numpy as np
import pytest
from gymnasium import spaces

from mbfps.data.collector import Collector, DataIntegrityError
from mbfps.data.policies import RandomPolicy
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


class _StubEnv:
    """Stand-in that mirrors ViZDoom's real terminal behaviour.

    Critically, `privileged_state` returns None once the episode is finished,
    exactly as ViZDoom's get_state() does. A more forgiving stub would let the
    ragged-array bug through.
    """

    instances = 0

    def __init__(self, episode_len=6, crash_at=None):
        type(self).instances += 1
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.button_names = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
        self.scenario = "stub"
        self._episode_len = episode_len
        self._crash_at = crash_at
        self._t = 0
        self._done = False

    def reset(self, *, seed=None):
        self._t = 0
        self._done = False
        return np.full(OBS_SHAPE, 1, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        if self._crash_at is not None and self._t == self._crash_at:
            raise RuntimeError("simulated engine crash")
        self._done = self._t >= self._episode_len
        obs = np.full(OBS_SHAPE, self._t % 256, dtype=np.uint8)
        return obs, 1.0, self._done, False, {}

    def close(self):
        pass

    @property
    def privileged_state(self):
        if self._done:
            return None
        return {k: float(self._t) for k in KEYS}


def _collector(episode_len=6, crash_at=None, max_steps=1000):
    return Collector(
        lambda: _StubEnv(episode_len, crash_at), RandomPolicy(4, seed=0), max_steps
    )


def test_collect_episode_returns_episode():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep is not None and ep.length == 6
    c.close()


def test_terminal_episode_is_not_treated_as_a_crash():
    """The bug this guards: a None privileged_state on the terminal frame
    produced a ragged array, so every real episode was discarded."""
    c = _collector(episode_len=6)
    assert c.collect_episode(seed=0) is not None
    assert c.crash_count == 0
    c.close()


def test_privileged_keys_survive_termination():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep.privileged_keys == KEYS, "keys must be captured at reset, not after"
    c.close()


def test_privileged_has_one_row_per_frame():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep.privileged.shape == (ep.length + 1, len(KEYS))
    c.close()


def test_terminal_privileged_row_carries_the_last_value_forward():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert np.array_equal(ep.privileged[-1], ep.privileged[-2])
    c.close()


def test_obs_has_one_more_frame_than_actions():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep.obs.shape[0] == ep.actions.shape[0] + 1
    c.close()


def test_episode_records_policy_name_and_seed():
    c = _collector(episode_len=4)
    ep = c.collect_episode(seed=77)
    assert ep.policy_name == "random" and ep.seed == 77
    c.close()


def test_each_episode_gets_a_different_action_sequence():
    """The bug this guards: resetting the policy to a fixed seed made every
    collected episode replay one identical action sequence."""
    c = _collector(episode_len=40, max_steps=40)
    a = c.collect_episode(seed=1)
    b = c.collect_episode(seed=2)
    assert not np.array_equal(a.actions, b.actions)
    c.close()


def test_same_seed_reproduces_the_action_sequence():
    a = _collector(episode_len=40, max_steps=40).collect_episode(seed=5)
    b = _collector(episode_len=40, max_steps=40).collect_episode(seed=5)
    assert np.array_equal(a.actions, b.actions)


def test_max_steps_truncates():
    c = _collector(episode_len=999, max_steps=5)
    assert c.collect_episode(seed=0).length == 5
    c.close()


def test_crash_returns_none_and_is_counted():
    c = _collector(episode_len=20, crash_at=3)
    assert c.collect_episode(seed=0) is None
    assert c.crash_count == 1
    c.close()


def test_crash_rebuilds_the_environment():
    _StubEnv.instances = 0
    c = _collector(episode_len=20, crash_at=3)
    c.collect_episode(seed=0)
    assert _StubEnv.instances >= 2, "collector must rebuild the env after a crash"
    c.close()


def test_collector_recovers_and_keeps_collecting():
    envs = iter([_StubEnv(20, crash_at=3), _StubEnv(6), _StubEnv(6)])
    c = Collector(lambda: next(envs), RandomPolicy(4, seed=0))
    assert c.collect_episode(seed=0) is None
    assert c.collect_episode(seed=1) is not None
    c.close()


def test_env_with_no_privileged_state_still_collects():
    """A zero-width privileged array must not be mistaken for a crash.

    `np.stack` of zero-length rows gives shape (T+1, 0), which is correct.
    Reshaping it would raise `cannot reshape array of size 0` (measured on
    numpy 2.5.2), and the crash handler would swallow that as a phantom
    engine fault -- discarding every episode from such an env.
    """

    class _Bare(_StubEnv):
        @property
        def privileged_state(self):
            return None

    c = Collector(lambda: _Bare(6), RandomPolicy(4, seed=0))
    ep = c.collect_episode(seed=0)
    assert ep is not None
    assert c.crash_count == 0
    assert ep.privileged_keys == ()
    assert ep.privileged.shape == (ep.length + 1, 0)
    c.close()


def test_malformed_data_is_not_reported_as_an_engine_crash():
    """A key vanishing mid-episode is our bug, not the engine's.

    Without a distinct exception type this raises inside `_collect`, is caught
    by the blanket crash handler, and is silently counted as an engine fault --
    the exact failure mode this collector was rewritten to prevent.
    """

    class _Shifting(_StubEnv):
        @property
        def privileged_state(self):
            if self._done:
                return None
            if self._t > 2:
                return {k: 1.0 for k in KEYS[:-1]}  # drops "angle"
            return {k: float(self._t) for k in KEYS}

    c = Collector(lambda: _Shifting(6), RandomPolicy(4, seed=0))
    with pytest.raises(DataIntegrityError, match="lost keys mid-episode"):
        c.collect_episode(seed=0)
    assert c.crash_count == 0
    c.close()
