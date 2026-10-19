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


@pytest.mark.parametrize("kind", ["frozen_ssl", "pixel_ae", "random_vit"])
def test_qnetwork_builds_each_study_encoder(kind):
    q = QNetwork(3, seed=0, encoder_kind=kind)
    assert q.encoder_kind == kind and q.n_actions == 3
    assert q.head[-1].out_features == 3


def test_qnetwork_frozen_ssl_is_the_default_and_matches_it():
    a, b = QNetwork(4, seed=0), QNetwork(4, seed=0, encoder_kind="frozen_ssl")
    for (n, p), (_, r) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, r), n


def test_qnetwork_refuses_a_pixel_encoder():
    with pytest.raises(ValueError, match="cached features"):
        QNetwork(4, seed=0, encoder_kind="cnn")


@pytest.mark.parametrize("n", [3, 7])
def test_random_action_covers_exactly_the_action_space(n):
    p = EpsilonGreedyQPolicy(QNetwork(n, seed=0), encode=None, device=torch.device("cpu"),
                             epsilon_fn=lambda: 1.0)
    p.reset(seed=1)
    feats = np.zeros((64, 384), np.float16)
    assert {p.act_from_features(feats) for _ in range(400)} == set(range(n))


def test_policy_without_an_action_count_refuses():
    q = nn.Sequential(nn.Flatten(), nn.Linear(3, 2))
    with pytest.raises(ValueError, match="n_actions"):
        EpsilonGreedyQPolicy(q, encode=None, device=torch.device("cpu"), epsilon_fn=lambda: 1.0)


def _chain_episodes(rng, n_episodes: int):
    """Spec §5.4 chain: states 0-3 (one-hot over 5; 4 is the terminal frame).
    Action 1 moves right and, from 3, pays 1 and terminates; action 0 stays.
    Each episode starts at a random state with a random first action, then acts
    optimally, so n-step returns are unbiased for Q*. Half are truncated early:
    a truncation must bootstrap, a termination must not."""
    eye = np.eye(5, dtype=np.float32)
    out = []
    for _ in range(n_episodes):
        s = int(rng.integers(0, 4))
        feats, acts, rews, term = [eye[s]], [], [], []
        a = int(rng.integers(0, 2))
        while True:
            done = s == 3 and a == 1
            s = 4 if done else (s + 1 if a == 1 else s)
            acts.append(a); rews.append(1.0 if done else 0.0); term.append(done)
            feats.append(eye[s])
            if done:
                break
            a = 1
        trunc = [False] * len(acts)
        if len(acts) > 1 and rng.random() < 0.5:
            cut = int(rng.integers(1, len(acts)))
            feats, acts, rews, term, trunc = feats[:cut + 1], acts[:cut], rews[:cut], term[:cut], \
                [False] * (cut - 1) + [True]
        out.append((np.asarray(feats)[:, None, :], np.asarray(acts), np.asarray(rews, np.float32),
                    np.asarray(term), np.asarray(trunc)))
    return out


@pytest.mark.parametrize("n_step", [1, 3])
def test_dqn_converges_to_known_q_on_a_chain_with_both_endings(n_step):
    """M5a §5.4: n-step targets, termination vs truncation, Double-DQN selection
    and the target copy, against Q*(s,1) = .9**(3-s) and Q*(s,0) = .9**(4-s)."""
    torch.manual_seed(0)
    cfg = DQNConfig(gamma=0.9, n_step=n_step, reward_scale=1.0, target_every=50, lr=1e-2,
                    batch_size=64, frame_stack=1)
    s = TransitionSampler(seed=0, cfg=cfg)
    episodes = _chain_episodes(np.random.default_rng(0), 400)
    assert any(e[4].any() for e in episodes) and any(e[3].any() for e in episodes)
    for e in episodes:
        s.add(*e)
    q = nn.Sequential(nn.Flatten(), nn.Linear(5, 2))
    q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    for i in range(1, 4001):
        dqn_step(q, q_target, opt, s.sample(cfg.batch_size), i, cfg, torch.device("cpu"))
    with torch.no_grad():
        got = q(torch.eye(5)[:4, None, None, :]).numpy()
    want = np.array([[0.9 ** (4 - st), 0.9 ** (3 - st)] for st in range(4)])
    np.testing.assert_allclose(got, want, atol=0.03)


def test_double_dqn_target_uses_the_online_argmax_valued_by_the_target_net():
    """Online net prefers action 0 at the next state; the target net values
    action 1 at 5. Double DQN's target is q_target(next)[0] = 0, so the loss is
    0. Vanilla DQN (max of the target net) would give 4.5; bootstrapping from
    the online net itself would give 0.5."""
    def table(rows):
        net = nn.Sequential(nn.Flatten(), nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            net[1].weight.copy_(torch.tensor(rows, dtype=torch.float32).T)
        return net
    q = table([[0.0, 0.0], [1.0, 0.0]])          # rows: Q(A,.), Q(B,.)
    q_target = table([[0.0, 0.0], [0.0, 5.0]])
    batch = {"obs": np.array([[[1.0, 0.0]]], np.float32), "next_obs": np.array([[[0.0, 1.0]]], np.float32),
             "action": np.array([0]), "ret": np.array([0.0], np.float32),
             "discount": np.array([1.0], np.float32)}
    out = dqn_step(q, q_target, torch.optim.SGD(q.parameters(), lr=0.0), batch, 1,
                   DQNConfig(target_every=10**9), torch.device("cpu"))
    assert out["loss"] == pytest.approx(0.0)
