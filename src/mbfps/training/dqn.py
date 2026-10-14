"""The model-free baseline (M5a §5): Double DQN with n-step returns.

Reads the same frozen DINOv2 features as the Dreamer arm, through the same
`BottleneckEncoder` class at the same seeded initialisation, with a 4-frame
stack in place of Dreamer's recurrent state. Settings are standard (Rainbow-
lite) and were fixed before any run (§5.2)."""

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mbfps.models.encoders import build_encoder
from mbfps.utils.config import EncoderConfig
from mbfps.utils.seeding import fork_seed, seeded_init


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    n_step: int = 3
    target_every: int = 2000
    lr: float = 1e-4
    grad_clip: float = 10.0
    batch_size: int = 256
    reward_scale: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    frame_stack: int = 4
    hidden: int = 512


def epsilon(budget_step: int, cfg: DQNConfig) -> float:
    """Linear from `eps_start` at step 0 to `eps_end` at `eps_decay_steps`, then flat."""
    frac = min(1.0, budget_step / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def stack_indices(t: int, k: int) -> list[int]:
    """Frame indices of the k-stack ending at `t`; the first frame repeats at an episode's start."""
    return [max(0, t - k + 1 + i) for i in range(k)]


class QNetwork(nn.Module):
    """Bottleneck per frame, the k embeddings concatenated, a 2 x 512 SiLU MLP."""

    def __init__(self, n_actions: int, seed: int, frame_stack: int = 4, hidden: int = 512) -> None:
        super().__init__()
        enc_cfg = EncoderConfig(kind="frozen_ssl")
        # The world model builds its encoder as the first module after
        # seed_everything(seed); drawing from a fresh generator at `seed`
        # reproduces that initialisation exactly (§5.1).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = build_encoder(enc_cfg)
        self.frame_stack = frame_stack
        with seeded_init(seed, "q_head"):
            self.head = nn.Sequential(
                nn.Linear(frame_stack * enc_cfg.embed_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_actions))

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        b, k = stack.shape[:2]
        emb = self.encoder(stack.reshape(b * k, *stack.shape[2:]))
        return self.head(emb.reshape(b, -1))


class TransitionSampler:
    """Uniform transitions over every stored episode, with n-step targets.

    Slot `t` of an episode is the transition from obs[t] by actions[t]. The
    n-step return stops at a termination (no bootstrap) and otherwise
    bootstraps from obs[t + m], where m = min(n, T - t): truncation and the end
    of stored data both bootstrap from the last stored obs (§5.2)."""

    def __init__(self, seed: int, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._eps: list[tuple] = []
        self._ends: list[int] = []

    @property
    def n_transitions(self) -> int:
        return self._ends[-1] if self._ends else 0

    def add(self, features, actions, rewards, terminated, truncated) -> None:
        self._eps.append((features, np.asarray(actions), np.asarray(rewards, np.float32),
                          np.asarray(terminated, bool), np.asarray(truncated, bool)))
        self._ends.append(self.n_transitions + len(actions))

    def _target(self, e: int, t: int) -> tuple[float, float, int]:
        _, _, rewards, terminated, _ = self._eps[e]
        cfg, T = self.cfg, len(rewards)
        m = min(cfg.n_step, T - t)
        ret, done = 0.0, False
        for k in range(m):
            ret += (cfg.gamma ** k) * float(rewards[t + k]) * cfg.reward_scale
            if terminated[t + k]:
                done, m = True, k + 1
                break
        return ret, (0.0 if done else cfg.gamma ** m), t + m

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not self._ends:
            raise ValueError("no transitions stored")
        k = self.cfg.frame_stack
        idx = self.rng.integers(0, self.n_transitions, batch_size)
        obs, nxt, act, ret, disc = [], [], [], [], []
        for i in idx:
            e = int(np.searchsorted(self._ends, i, side="right"))
            t = int(i - (self._ends[e - 1] if e else 0))
            feats, actions = self._eps[e][0], self._eps[e][1]
            r, d, n = self._target(e, t)
            obs.append(np.asarray(feats[stack_indices(t, k)]))
            nxt.append(np.asarray(feats[stack_indices(n, k)]))
            act.append(int(actions[t])); ret.append(r); disc.append(d)
        return {"obs": np.stack(obs), "next_obs": np.stack(nxt),
                "action": np.asarray(act, np.int64),
                "ret": np.asarray(ret, np.float32), "discount": np.asarray(disc, np.float32)}


def dqn_step(q, q_target, optimiser, batch: dict, update_index: int, cfg: DQNConfig,
             device: torch.device) -> dict[str, float]:
    """One Double-DQN update: the online net picks a*, the target net values it."""
    obs = torch.as_tensor(batch["obs"]).to(device)
    nxt = torch.as_tensor(batch["next_obs"]).to(device)
    act = torch.as_tensor(batch["action"]).to(device)
    ret = torch.as_tensor(batch["ret"]).to(device)
    disc = torch.as_tensor(batch["discount"]).to(device)
    with torch.no_grad():
        a_star = q(nxt).argmax(dim=-1, keepdim=True)
        target = ret + disc * q_target(nxt).gather(1, a_star).squeeze(1)
    q_sa = q(obs).gather(1, act[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(q_sa, target, beta=1.0)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
    optimiser.step()
    if update_index % cfg.target_every == 0:
        q_target.load_state_dict(q.state_dict())
    return {"loss": float(loss.detach()), "q_mean": float(q_sa.detach().mean())}


class EpsilonGreedyQPolicy:
    """ε-greedy over a frame stack. `epsilon_fn` is read at every action, so
    the behaviour policy follows the budget-step schedule; evaluation passes a
    constant. The ε draw uses its own generator, reseeded per episode."""

    def __init__(self, q, encode: Callable | None, device: torch.device,
                 epsilon_fn: Callable[[], float], frame_stack: int = 4, name: str = "dqn") -> None:
        self.q, self.encode, self.device = q, encode, device
        self.epsilon_fn, self.k, self.name = epsilon_fn, frame_stack, name
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(fork_seed(int(seed or 0), f"eps:{self.name}"))
        self._stack: deque = deque(maxlen=self.k)

    def act(self, obs: np.ndarray) -> int:
        return self.act_from_features(self.encode(obs[None])[0])

    @torch.no_grad()
    def act_from_features(self, features: np.ndarray) -> int:
        if not self._stack:
            self._stack.extend([features] * self.k)
        else:
            self._stack.append(features)
        explore = self._rng.random() < self.epsilon_fn()
        random_action = int(self._rng.integers(0, 4))
        if explore:
            return random_action
        stack = torch.as_tensor(np.stack(self._stack)[None]).to(self.device)
        return int(self.q(stack).argmax(-1))

    def get_state(self) -> dict:
        return {"stack": [np.asarray(f) for f in self._stack],
                "rng": self._rng.bit_generator.state}

    def set_state(self, state: dict) -> None:
        self._stack = deque(state["stack"], maxlen=self.k)
        self._rng.bit_generator.state = state["rng"]
