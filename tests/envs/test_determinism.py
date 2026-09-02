import json

import numpy as np
import pytest

from mbfps.envs.vizdoom_env import ViZDoomEnv

SCENARIO = "my_way_home"


def _rollout(seed, actions):
    """Replay a fixed action list and return (frames, rewards)."""
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
        frames, rewards = [obs.copy()], []
        for a in actions:
            obs, reward, terminated, truncated, _ = env.step(a)
            frames.append(obs.copy())
            rewards.append(reward)
            if terminated or truncated:
                break
        return frames, rewards
    finally:
        env.close()


@pytest.fixture
def actions():
    # Bound the random action stream by the scenario's actual action count
    # rather than a hardcoded literal, so this fixture cannot silently drift
    # out of sync with the scenario's button set.
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=0)
    try:
        n = env.action_space.n
    finally:
        env.close()
    rng = np.random.default_rng(0)
    return [int(a) for a in rng.integers(0, n, size=40)]


def test_same_seed_produces_identical_frames(actions):
    a_frames, _ = _rollout(seed=7, actions=actions)
    b_frames, _ = _rollout(seed=7, actions=actions)
    assert len(a_frames) == len(b_frames)
    for i, (a, b) in enumerate(zip(a_frames, b_frames)):
        assert np.array_equal(a, b), f"frame {i} differs between identical seeds"


def test_same_seed_produces_identical_rewards(actions):
    _, a_rewards = _rollout(seed=7, actions=actions)
    _, b_rewards = _rollout(seed=7, actions=actions)
    assert a_rewards == b_rewards


def test_different_seeds_diverge(actions):
    """Fresh instances constructed with different seeds produce different episodes.

    `_rollout()` constructs a brand-new `ViZDoomEnv(seed=seed)` per call, so
    this proves construction-time seeding (the `set_seed` call in `__init__`)
    differentiates the two engines. It does NOT exercise `reset()`'s reseed
    path -- a `reset()` that silently no-ops would not be caught here. See
    `test_reset_to_a_different_seed_diverges_within_one_instance` for the test
    that actually guards `reset()`'s reseed.
    """
    a_frames, _ = _rollout(seed=7, actions=actions)
    b_frames, _ = _rollout(seed=8, actions=actions)
    pairs = list(zip(a_frames, b_frames))
    assert any(not np.array_equal(a, b) for a, b in pairs), (
        "two different seeds produced identical episodes -- set_seed is not taking effect"
    )


def test_reset_to_a_different_seed_diverges_within_one_instance():
    """The operation M1's collector actually performs.

    A long-lived engine is reseeded once per episode. Constructing a fresh env
    per seed (as the other divergence test does) proves only that the
    constructor seeds; it passes even when reset()'s reseed is a no-op.
    """
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=0)
    try:
        first, _ = env.reset(seed=31)
        second, _ = env.reset(seed=32)
        assert not np.array_equal(first, second), (
            "reset() to a different seed produced an identical first frame -- "
            "reset() is not reseeding the engine"
        )
    finally:
        env.close()


def test_recorded_episode_replays_from_saved_actions_alone(tmp_path):
    """The M0 exit criterion.

    The recording policy is observation-dependent -- it reads `obs` to choose
    each action -- so the action stream is a function of the rollout's own
    pixel history, not a fixed list chosen independently of what the engine
    renders. Replaying the saved actions into a fresh engine and recovering
    identical pixels therefore proves the *pixels* were deterministic, not
    merely that a fixed action list replays the same regardless of what it's
    fed.
    """
    seed = 11
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
        original_frames, chosen = [obs.copy()], []
        for _ in range(40):
            action = int(obs.sum() % env.action_space.n)
            obs, _, terminated, truncated, _ = env.step(action)
            chosen.append(action)
            original_frames.append(obs.copy())
            if terminated or truncated:
                break
    finally:
        env.close()

    record = tmp_path / "episode_actions.json"
    record.write_text(json.dumps({"seed": seed, "actions": chosen}))

    loaded = json.loads(record.read_text())
    replay_frames, _ = _rollout(seed=loaded["seed"], actions=loaded["actions"])

    assert len(replay_frames) == len(original_frames)
    for i, (a, b) in enumerate(zip(original_frames, replay_frames)):
        assert np.array_equal(a, b), f"replayed frame {i} differs from the recording"


def test_reset_reseeds_within_one_instance():
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=0)
    try:
        first, _ = env.reset(seed=21)
        env.step(1)
        second, _ = env.reset(seed=21)
        assert np.array_equal(first, second)
    finally:
        env.close()
