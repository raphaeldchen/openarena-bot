import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mbfps.training.dqn import (
    DQNConfig, EpsilonGreedyQPolicy, QNetwork, TransitionSampler, dqn_step, epsilon,
    stack_indices,
)
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config
from mbfps.utils.seeding import seed_everything


def test_epsilon_schedule():
    cfg = DQNConfig()
    assert epsilon(0, cfg) == 1.0
    assert epsilon(25_000, cfg) == pytest.approx(0.525)
    assert epsilon(50_000, cfg) == pytest.approx(0.05)
    assert epsilon(90_000, cfg) == pytest.approx(0.05)


def test_stack_indices_repeat_the_first_frame():
    assert stack_indices(0, 4) == [0, 0, 0, 0]
    assert stack_indices(2, 4) == [0, 0, 1, 2]
    assert stack_indices(9, 4) == [6, 7, 8, 9]


def test_q_encoder_matches_dreamers_encoder_init():
    """Same class, same width, same seeded initialisation (M5a §5.1)."""
    seed_everything(3)
    wm = WorldModel(get_config("frozen_ssl", seed=3, device="cpu", n_actions=4))
    q = QNetwork(4, seed=3)
    for (ka, a), (kb, b) in zip(wm.encoder.state_dict().items(), q.encoder.state_dict().items()):
        assert ka == kb and torch.equal(a, b)


def test_q_network_shapes():
    q = QNetwork(4, seed=0)
    out = q(torch.zeros(5, 4, 64, 384, dtype=torch.float16))
    assert out.shape == (5, 4)


def _episode(rewards, terminated, truncated=None, width=3):
    t = len(rewards)
    feats = np.arange(t + 1, dtype=np.float32)[:, None, None] * np.ones((1, 1, width), np.float32)
    return (feats, np.zeros(t, np.int64), np.asarray(rewards, np.float32),
            np.asarray(terminated, bool),
            np.zeros(t, bool) if truncated is None else np.asarray(truncated, bool))


def _all_rows(sampler, n=2000):
    b = sampler.sample(n)
    return {(int(b["obs"][i, -1, 0, 0]), ): (float(b["ret"][i]), float(b["discount"][i]),
                                            int(b["next_obs"][i, -1, 0, 0])) for i in range(n)}


def test_n_step_targets_stop_at_termination():
    cfg = DQNConfig(gamma=0.5, n_step=3, reward_scale=1.0)
    s = TransitionSampler(seed=0, cfg=cfg)
    s.add(*_episode([1, 2, 4, 8], [0, 0, 0, 1]))
    rows = _all_rows(s)
    # t=0: 1 + .5*2 + .25*4 = 3, bootstraps from obs 3 with .5**3
    assert rows[(0,)] == (3.0, 0.125, 3)
    # t=2: 4 + .5*8, terminated at t=3 -> no bootstrap
    assert rows[(2,)] == (8.0, 0.0, 4)
    # t=3: 8, terminated
    assert rows[(3,)] == (8.0, 0.0, 4)


def test_n_step_truncation_bootstraps_from_the_last_stored_obs():
    cfg = DQNConfig(gamma=0.5, n_step=3, reward_scale=1.0)
    s = TransitionSampler(seed=0, cfg=cfg)
    s.add(*_episode([1, 1], [0, 0], truncated=[0, 1]))
    rows = _all_rows(s)
    assert rows[(0,)] == (1.5, 0.25, 2)   # 2 steps to the end, bootstrap obs 2
    assert rows[(1,)] == (1.0, 0.5, 2)


def test_reward_scale_applies_to_targets():
    s = TransitionSampler(seed=0, cfg=DQNConfig(n_step=1))
    s.add(*_episode([97.0], [1]))
    assert float(s.sample(1)["ret"][0]) == pytest.approx(0.97)


def test_double_dqn_learns_a_chain():
    """Known answer (M5a §5.4). States 0,1,2 encoded one-hot; action 1 moves
    right, action 0 stays; moving right from 2 pays 1 and terminates. With
    gamma .9: Q*(s,1) = .9**(2-s), Q*(s,0) = .9 * Q*(s,1)."""
    torch.manual_seed(0)
    cfg = DQNConfig(gamma=0.9, n_step=1, reward_scale=1.0, target_every=50, lr=1e-2,
                    batch_size=64, frame_stack=1)
    rng = np.random.default_rng(0)
    s = TransitionSampler(seed=0, cfg=cfg)
    for _ in range(200):
        state, feats, acts, rews, term = 0, [np.eye(3)[0]], [], [], []
        while True:
            a = int(rng.integers(0, 2))
            done = state == 2 and a == 1
            nxt = state if a == 0 else min(state + 1, 2)
            acts.append(a); rews.append(1.0 if done else 0.0); term.append(done)
            feats.append(np.eye(3)[nxt]); state = nxt
            if done or len(acts) >= 12:
                break
        s.add(np.asarray(feats, np.float32)[:, None, :], np.asarray(acts), np.asarray(rews, np.float32),
              np.asarray(term), np.asarray([False] * (len(acts) - 1) + [not term[-1]]))
    q = nn.Sequential(nn.Flatten(), nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 2))
    q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    for i in range(1, 3001):
        dqn_step(q, q_target, opt, s.sample(cfg.batch_size), i, cfg, torch.device("cpu"))
    with torch.no_grad():
        got = q(torch.eye(3)[:, None, None, :].float()).numpy()
    want = np.array([[0.9 * 0.81, 0.81], [0.9 * 0.9, 0.9], [0.9 * 1.0, 1.0]])
    np.testing.assert_allclose(got, want, atol=0.05)


def test_dqn_step_copies_target_on_schedule():
    cfg = DQNConfig(target_every=2, n_step=1, frame_stack=1)
    s = TransitionSampler(seed=0, cfg=cfg)
    s.add(*_episode([1.0, 0.0], [0, 1]))
    q = nn.Sequential(nn.Flatten(), nn.Linear(3, 2))
    q_target = copy.deepcopy(q)
    opt = torch.optim.SGD(q.parameters(), lr=0.1)
    dqn_step(q, q_target, opt, s.sample(4), 1, cfg, torch.device("cpu"))
    assert not torch.equal(q[1].weight, q_target[1].weight)
    dqn_step(q, q_target, opt, s.sample(4), 2, cfg, torch.device("cpu"))
    assert torch.equal(q[1].weight, q_target[1].weight)


def test_policy_repeats_first_frame_and_is_seeded():
    q = QNetwork(4, seed=0)
    feats = lambda v: np.full((64, 384), v, np.float16)
    p = EpsilonGreedyQPolicy(q, encode=None, device=torch.device("cpu"), epsilon_fn=lambda: 0.5)
    p.reset(seed=7)
    a = [p.act_from_features(feats(i)) for i in range(6)]
    p.reset(seed=7)
    b = [p.act_from_features(feats(i)) for i in range(6)]
    assert a == b
    p.reset(seed=7)
    p.act_from_features(feats(3))
    stack = p.get_state()["stack"]
    assert len(stack) == 4 and all(np.array_equal(f, feats(3)) for f in stack)
