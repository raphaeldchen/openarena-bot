import numpy as np
import pytest
from gymnasium import spaces

from mbfps.envs.protocol import OBS_SHAPE, EnvProtocol
from mbfps.envs.vizdoom_env import PRIVILEGED_KEYS, ViZDoomEnv


@pytest.fixture
def env():
    e = ViZDoomEnv(scenario="my_way_home", frame_skip=4, seed=0)
    yield e
    e.close()


def test_satisfies_env_protocol(env):
    assert isinstance(env, EnvProtocol)


def test_observation_space_matches_obs_shape(env):
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == OBS_SHAPE
    assert env.observation_space.dtype == np.uint8


def test_action_space_is_discrete_and_nonempty(env):
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n >= 2


def test_scenario_exposes_movement_and_turning(env):
    """ScriptedPolicy's coverage behaviour depends on these existing."""
    names = env.button_names
    assert "MOVE_FORWARD" in names
    assert "TURN_LEFT" in names and "TURN_RIGHT" in names


def test_episodes_are_long_enough_for_the_training_window(env):
    """seq_len=64 needs 65 frames; a shorter episode is silently discarded."""
    env.reset(seed=0)
    steps = 0
    for _ in range(3000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps >= 65, f"episode was only {steps} transitions; seq_len=64 needs 65"


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)


def test_step_returns_five_tuple(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == OBS_SHAPE
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_eventually_ends(env):
    env.reset(seed=0)
    for _ in range(3000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            return
    pytest.fail("episode did not end within 3000 steps")


def test_timeout_sets_truncated_not_terminated(env):
    """The time-limit bootstrapping guard.

    A time-limit cutoff is not a true terminal state. Asserting only that the
    two flags are never both true does not catch the bug: swapping them still
    satisfies it. This asserts which flag a timeout actually sets.
    """
    env.reset(seed=0)
    for step in range(1, 3000):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            assert truncated, f"timeout at step {step} must set truncated"
            assert not terminated, "a time limit is not a true terminal state"
            return
    pytest.fail("episode did not end within 3000 steps")


def test_goal_reached_sets_terminated_not_truncated():
    """The other direction: a real terminal must not be recorded as a timeout.

    seed=5 with this exact RNG reaches the goal at step 96 (reward ~1.0),
    well before the 525-step timeout.
    """
    env = ViZDoomEnv(scenario="my_way_home", frame_skip=4, seed=5)
    try:
        env.reset(seed=5)
        rng = np.random.default_rng(5)
        for step in range(1, 600):
            action = int(rng.integers(0, env.action_space.n))
            _, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                assert terminated, f"goal reached at step {step} must set terminated"
                assert not truncated, "a goal is not a time-limit cutoff"
                assert reward > 0.5, f"expected goal reward, got {reward}"
                return
        pytest.fail("episode did not end within 600 steps")
    finally:
        env.close()


def test_observation_after_end_is_still_valid(env):
    env.reset(seed=0)
    for _ in range(3000):
        obs, _, terminated, truncated, _ = env.step(0)
        assert obs.shape == OBS_SHAPE, "terminal observation must stay well-formed"
        if terminated or truncated:
            break


def test_privileged_state_has_expected_keys(env):
    env.reset(seed=0)
    state = env.privileged_state
    assert set(state) == set(PRIVILEGED_KEYS)
    assert all(isinstance(v, float) for v in state.values())


def test_privileged_state_is_none_after_the_episode_ends(env):
    """ViZDoom's get_state() returns None once finished; callers must handle it."""
    env.reset(seed=0)
    for _ in range(3000):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert env.privileged_state is None


def test_privileged_keys_are_indexed_by_name_not_position(env):
    """health is the scenario's own variable; the position vars are ours."""
    env.reset(seed=0)
    state = env.privileged_state
    assert state["health"] > 0.0, "health should be positive at episode start"


def test_close_is_idempotent(env):
    env.close()
    env.close()
