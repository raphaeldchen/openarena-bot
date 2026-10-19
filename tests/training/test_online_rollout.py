import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode
from mbfps.data.loader import feature_suffix
from mbfps.data.policies import RandomPolicy
from mbfps.training.online_rollout import OnlineRollout, act_with, train_episode_seed
from tests.training.fake_env import FakeEnv, fake_encode


class _NoShoot:
    """Never shoots, so FakeEnv cannot terminate; 12-step limit is the only end."""
    name = "noshoot"

    def __init__(self):
        self._t = 0

    def reset(self, seed=None):
        self._t = 0

    def act(self, obs):
        self._t += 1
        return (1, 2, 0)[self._t % 3]


def _rollout(tmp_path, seen):
    replay = ReplayBuffer(tmp_path / "replay", capacity_transitions=10**9)
    return OnlineRollout(FakeEnv(), fake_encode, replay, run_seed=1,
                         on_episode=seen.append, policy_name="random")


def test_train_episode_seeds_are_disjoint_from_eval_seeds():
    assert train_episode_seed(0, 0) == 2_000_000
    assert train_episode_seed(4, 99_999) == 2_499_999
    with pytest.raises(AssertionError):
        train_episode_seed(0, 100_000)


def test_steps_count_the_budget_and_write_whole_episodes(tmp_path):
    seen = []
    r = _rollout(tmp_path, seen)
    policy = RandomPolicy(4, seed=0)
    for _ in range(50):
        r.step(policy)
    assert r.budget_step == 50
    total = sum(load_episode(p).length for p in seen)
    assert total + (r.current_length if r.in_progress else 0) == 50
    for p in seen:
        ep = load_episode(p)
        feats = np.load(p.with_suffix(feature_suffix("dinov2")))
        assert feats.shape == (ep.length + 1, 64, 384)
        assert ep.terminated[-1] or ep.truncated[-1]
        np.testing.assert_array_equal(feats, fake_encode(ep.obs))


def test_episode_seeds_follow_the_run_seed(tmp_path):
    seen = []
    r = _rollout(tmp_path, seen)
    policy = RandomPolicy(4, seed=0)
    while len(seen) < 3:
        r.step(policy)
    assert [load_episode(p).seed for p in seen] == [train_episode_seed(1, k) for k in range(3)]


def test_close_budget_stores_the_tail_truncated(tmp_path):
    seen = []
    r = _rollout(tmp_path, seen)
    policy = _NoShoot()
    for _ in range(5):
        r.step(policy)
    assert r.in_progress
    r.close_episode_truncated()
    ep = load_episode(seen[-1])
    assert ep.length == 5 and ep.truncated[-1] and not ep.terminated.any()
    assert not r.in_progress and r.budget_step == 5


def _assert_same_episode(a, b):
    """The restored in-progress episode must equal the uninterrupted one."""
    assert a.budget_step == b.budget_step
    assert a.episodes_done == b.episodes_done
    assert a.current_length == b.current_length > 0
    for key in ("obs", "features", "privileged"):
        assert len(a._ep[key]) == len(b._ep[key]), key
        assert all(np.array_equal(x, y) for x, y in zip(a._ep[key], b._ep[key])), key
    assert a._ep["actions"] == b._ep["actions"]
    assert a._ep["rewards"] == b._ep["rewards"]


def test_restore_replays_the_engine_to_the_same_point(tmp_path):
    seen_a, seen_b = [], []
    a = _rollout(tmp_path / "a", seen_a)
    pa = RandomPolicy(4, seed=0)
    for _ in range(7):
        a.step(pa)
    state = a.state()
    b = _rollout(tmp_path / "b", seen_b)
    pb = RandomPolicy(4, seed=0)
    b.restore(state, pb)
    assert a.in_progress and b.in_progress
    _assert_same_episode(a, b)
    pb._rng.bit_generator.state = pa._rng.bit_generator.state
    for _ in range(20):
        a.step(pa)
        b.step(pb)
    assert seen_b, "no episode finished after the restore"
    assert load_episode(seen_a[-1]).actions.tolist() == load_episode(seen_b[-1]).actions.tolist()
    assert a.budget_step == b.budget_step


def test_act_with_prefers_features():
    class P:
        def act_from_features(self, f): return 3
        def act(self, obs): return 1
    class Q:
        def act(self, obs): return 1
    assert act_with(P(), None, None) == 3 and act_with(Q(), None, None) == 1


def test_equality_check_fails_for_a_restore_that_skips_the_replay(tmp_path):
    a = _rollout(tmp_path / "a", [])
    pa = RandomPolicy(4, seed=0)
    for _ in range(7):
        a.step(pa)
    assert a.in_progress
    state = a.state()
    b = _rollout(tmp_path / "b", [])
    b.budget_step, b.episodes_done = state["budget_step"], state["episodes_done"]
    b._start(RandomPolicy(4, seed=0))  # a restore that never replays the actions
    with pytest.raises(AssertionError):
        _assert_same_episode(a, b)


def test_restore_raises_when_the_replay_ends_the_episode(tmp_path):
    class Diverging(FakeEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, True, False, info

    a = _rollout(tmp_path / "a", [])
    pa = _NoShoot()
    for _ in range(3):
        a.step(pa)
    assert a.in_progress
    state = a.state()
    replay = ReplayBuffer(tmp_path / "b" / "replay", capacity_transitions=10**9)
    b = OnlineRollout(Diverging(), fake_encode, replay, run_seed=1, on_episode=lambda p: None)
    with pytest.raises(RuntimeError, match=str(state["episode"]["seed"])):
        b.restore(state, _NoShoot())
