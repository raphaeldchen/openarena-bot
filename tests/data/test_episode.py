import numpy as np
import pytest

from mbfps.data.episode import Episode, load_episode, save_episode
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int = 5) -> Episode:
    rng = np.random.default_rng(0)
    return Episode(
        obs=rng.integers(0, 256, (t + 1, *OBS_SHAPE), dtype=np.uint8),
        actions=rng.integers(0, 4, t).astype(np.int32),
        rewards=rng.standard_normal(t).astype(np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=rng.standard_normal((t + 1, len(KEYS))).astype(np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=42,
        scenario="my_way_home",
    )


def test_length_is_number_of_transitions():
    assert make_episode(t=5).length == 5


def test_round_trip_preserves_arrays(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert np.array_equal(loaded.obs, ep.obs)
    assert np.array_equal(loaded.actions, ep.actions)
    assert np.array_equal(loaded.rewards, ep.rewards)
    assert np.array_equal(loaded.terminated, ep.terminated)
    assert np.array_equal(loaded.truncated, ep.truncated)
    assert np.array_equal(loaded.privileged, ep.privileged)


def test_round_trip_preserves_metadata(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert loaded.policy_name == "random"
    assert loaded.seed == 42
    assert loaded.scenario == "my_way_home"
    assert loaded.privileged_keys == KEYS


def test_round_trip_preserves_dtypes(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert loaded.obs.dtype == np.uint8
    assert loaded.actions.dtype == np.int32
    assert loaded.rewards.dtype == np.float32
    assert loaded.privileged.dtype == np.float32
    assert loaded.terminated.dtype == bool
    assert loaded.truncated.dtype == bool


def test_truncated_and_terminated_survive_a_round_trip_independently(tmp_path):
    """A time-limit cutoff must stay distinguishable from a true terminal."""
    ep = make_episode(t=3)
    ep.terminated[:] = False
    ep.truncated[:] = False
    ep.truncated[-1] = True

    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)

    assert not loaded.terminated.any(), "no step should be marked terminal"
    assert loaded.truncated[-1], "the time-limit flag must survive the round trip"
    assert not loaded.truncated[:-1].any()


def test_mismatched_lengths_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="obs must have one more"):
        Episode(
            obs=rng.integers(0, 256, (5, *OBS_SHAPE), dtype=np.uint8),
            actions=rng.integers(0, 4, 5).astype(np.int32),
            rewards=rng.standard_normal(5).astype(np.float32),
            terminated=np.zeros(5, dtype=bool),
            truncated=np.zeros(5, dtype=bool),
            privileged=rng.standard_normal((6, len(KEYS))).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )


def test_mismatched_privileged_width_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="privileged must have shape"):
        Episode(
            obs=rng.integers(0, 256, (6, *OBS_SHAPE), dtype=np.uint8),
            actions=rng.integers(0, 4, 5).astype(np.int32),
            rewards=rng.standard_normal(5).astype(np.float32),
            terminated=np.zeros(5, dtype=bool),
            truncated=np.zeros(5, dtype=bool),
            privileged=rng.standard_normal((6, len(KEYS) - 2)).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )


def test_wrong_obs_dtype_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="obs must have dtype uint8"):
        Episode(
            obs=rng.standard_normal((6, *OBS_SHAPE)).astype(np.float32),
            actions=rng.integers(0, 4, 5).astype(np.int32),
            rewards=rng.standard_normal(5).astype(np.float32),
            terminated=np.zeros(5, dtype=bool),
            truncated=np.zeros(5, dtype=bool),
            privileged=rng.standard_normal((6, len(KEYS))).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )
