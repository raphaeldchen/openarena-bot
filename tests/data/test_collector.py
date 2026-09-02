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

    def __init__(self, episode_len=6, crash_at=None, truncate: bool = False):
        type(self).instances += 1
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.button_names = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
        self.scenario = "stub"
        self._episode_len = episode_len
        self._crash_at = crash_at
        self._truncate = truncate
        self._t = 0
        self._done = False
        self.seeds_seen: list = []

    def reset(self, *, seed=None):
        self._t = 0
        self._done = False
        self.seeds_seen.append(seed)
        return np.full(OBS_SHAPE, 1, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        if self._crash_at is not None and self._t == self._crash_at:
            raise RuntimeError("simulated engine crash")
        self._done = self._t >= self._episode_len
        obs = np.full(OBS_SHAPE, self._t % 256, dtype=np.uint8)
        if self._done and self._truncate:
            return obs, 1.0, False, True, {}
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


def test_same_seed_reproduces_actions_on_a_reused_collector():
    """Guards the per-episode policy reseed.

    A fresh collector per episode cannot detect a missing reset(): two new
    RandomPolicy(seed=0) objects start from the same RNG state regardless.
    Reusing ONE collector is what exposes it -- without the reseed the second
    episode simply continues the first episode's RNG stream.

    This is the failure a pre-flight audit rated critical: every collected
    episode replaying one identical action sequence.
    """
    c = _collector(episode_len=40, max_steps=40)
    try:
        first = c.collect_episode(seed=5)
        second = c.collect_episode(seed=5)
        assert first is not None and second is not None
        assert np.array_equal(first.actions, second.actions), (
            "same seed on a reused collector produced different actions -- "
            "the per-episode policy reseed is missing"
        )
    finally:
        c.close()


def test_different_seeds_differ_on_a_reused_collector():
    """The complement: reseeding must actually track the seed, not ignore it."""
    c = _collector(episode_len=40, max_steps=40)
    try:
        a = c.collect_episode(seed=5)
        b = c.collect_episode(seed=6)
        assert not np.array_equal(a.actions, b.actions), (
            "different seeds on a reused collector produced identical actions"
        )
    finally:
        c.close()


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


def test_truncated_episode_ends_the_loop():
    """A time-limit cutoff must end collection just like a terminal state."""
    c = Collector(lambda: _StubEnv(6, truncate=True), RandomPolicy(4, seed=0), 1000)
    ep = c.collect_episode(seed=0)
    assert ep is not None and ep.length == 6
    c.close()


def test_truncation_is_recorded_separately_from_termination():
    """Collapsing the two is the time-limit bootstrapping bug."""
    c = Collector(lambda: _StubEnv(6, truncate=True), RandomPolicy(4, seed=0), 1000)
    ep = c.collect_episode(seed=0)
    assert ep.truncated[-1], "final step should be flagged truncated"
    assert not ep.terminated[-1], "a time limit is not a true terminal state"
    assert not ep.terminated.any()
    c.close()


def test_collector_passes_the_episode_seed_to_the_env():
    """Without this, a dropped env seed is only caught at the ViZDoomEnv level."""
    env = _StubEnv(6)
    c = Collector(lambda: env, RandomPolicy(4, seed=0), 1000)
    c.collect_episode(seed=11)
    c.collect_episode(seed=12)
    assert env.seeds_seen == [11, 12]
    c.close()


def test_malformed_obs_is_not_reported_as_an_engine_crash():
    """Episode validation failures are our bug, not the engine's."""

    class _BadObs(_StubEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            return obs.astype(np.float32), reward, term, trunc, info

    c = Collector(lambda: _BadObs(6), RandomPolicy(4, seed=0), 1000)
    with pytest.raises(DataIntegrityError, match="failed validation"):
        c.collect_episode(seed=0)
    assert c.crash_count == 0
    c.close()
