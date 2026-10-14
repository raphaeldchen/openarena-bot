import numpy as np
import pytest
import torch

from mbfps.eval.live import EVAL_SEEDS, AgentPolicy, paired_gate, run_episodes
from mbfps.models.heads import Actor, Critic
from mbfps.training.world_model import WorldModel
from mbfps.utils.config import get_config


def test_eval_seeds():
    assert EVAL_SEEDS[0] == 1_000_000 and len(EVAL_SEEDS) == 100


def test_paired_gate_passes_a_clearly_better_policy():
    rng = np.random.default_rng(0)
    base = rng.normal(-170, 186, 100)
    out = paired_gate(base + 120.0, base, resamples=500, seed=0)
    assert out["passed"] and out["ci_low"] > 0


def test_paired_gate_fails_an_identical_policy():
    base = np.random.default_rng(0).normal(-170, 186, 100)
    out = paired_gate(base, base, resamples=500, seed=0)
    assert not out["passed"] and out["point"] == 0.0


def _agent(seed=0):
    cfg = get_config("cnn", device="cpu", n_actions=4)
    model = WorldModel(cfg).eval()
    stub = lambda frame: frame          # cnn kind consumes pixels directly
    return model, AgentPolicy(model, Actor(4, seed=seed), stub, torch.device("cpu"),
                              critic_slow=Critic(seed=seed))


def test_first_frame_is_filtered_with_noop(monkeypatch):
    model, agent = _agent()
    seen = []
    real_observe = model.rssm.observe
    def spy(embeddings, actions, state=None):
        seen.append(actions.clone())
        return real_observe(embeddings, actions, state)
    monkeypatch.setattr(model.rssm, "observe", spy)
    agent.reset(seed=5)
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    first = agent.act(frame)
    agent.act(frame)
    assert seen[0].tolist() == [[0]]            # the no-op
    assert seen[1].tolist() == [[first]]        # then the agent's own last action
    assert agent.imagined_start_return is not None


class _FakeEnv:
    """Ten-step episodes; reward = +1 for action 3, else -1."""
    def __init__(self): self.t = 0
    def reset(self, *, seed=None):
        self.t = 0
        return np.zeros((112, 112, 3), dtype=np.uint8), {}
    def step(self, action):
        self.t += 1
        done = self.t >= 10
        return (np.full((112, 112, 3), self.t, dtype=np.uint8),
                1.0 if action == 3 else -1.0, False, done, {})


def test_agent_episode_is_reproducible():
    _, agent = _agent()
    a = run_episodes(_FakeEnv(), agent, seeds=[7, 8])
    b = run_episodes(_FakeEnv(), agent, seeds=[7, 8])
    assert [e["return"] for e in a] == [e["return"] for e in b]
    assert [e["imagined_return"] for e in a] == [e["imagined_return"] for e in b]
    assert all(e["length"] == 10 and e["truncated"] and not e["killed"] for e in a)


def test_run_episodes_with_random_policy_on_real_basic():
    from mbfps.data.policies import RandomPolicy
    from mbfps.envs.registry import make_env
    env = make_env("vizdoom", scenario="basic", frame_skip=4)
    try:
        eps = run_episodes(env, RandomPolicy(env.action_space.n), seeds=EVAL_SEEDS[:3])
    finally:
        env.close()
    assert len(eps) == 3 and all(1 <= e["length"] <= 75 for e in eps)
    assert all(e["imagined_return"] is None for e in eps)


def test_act_from_features_matches_act():
    model, agent = _agent()
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    agent.reset(seed=3)
    a = [agent.act(frame) for _ in range(4)]
    agent.reset(seed=3)
    b = [agent.act_from_features(frame) for _ in range(4)]   # cnn kind: features are pixels
    assert a == b


def test_agent_state_round_trips():
    model, agent = _agent()
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    agent.reset(seed=1)
    agent.act(frame)
    s = agent.get_state()
    agent.act(frame)
    agent.set_state(s)
    assert agent.get_state()["last_action"] == s["last_action"]
    assert torch.equal(agent.get_state()["h"], s["h"])
