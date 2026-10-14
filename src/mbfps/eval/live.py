"""Live evaluation (M4 spec §5.4-§5.6).

The live game is used ONLY to evaluate: nothing here writes a dataset or takes
a gradient. Each frame goes through the arm's frozen backbone (`encode`), the
world model's encoder and the posterior filter; the actor then SAMPLES an action
from the posterior latent, as it was trained. The first frame of an episode is
filtered as if a no-op (action 0) had produced it, which is how every training
sequence begins: an initial state, then an action and the frame it led to.

Every episode reseeds torch from the episode seed, so a rerun reproduces it
(the posterior's z draw and the actor's action draw both use that stream).
"""

from typing import Callable

import numpy as np
import torch

from mbfps.models.transforms import read_reward
from mbfps.training.ac_trainer import lambda_returns
from mbfps.utils.seeding import fork_seed

EVAL_SEEDS: tuple[int, ...] = tuple(range(1_000_000, 1_000_100))
RESAMPLES: int = 2000
NOOP: int = 0


class AgentPolicy:
    def __init__(self, world_model, actor, encode: Callable[[np.ndarray], np.ndarray],
                 device: torch.device, mode: str = "sample", critic_slow=None,
                 gamma: float = 0.997, lam: float = 0.95, horizon: int = 15) -> None:
        if mode not in ("sample", "argmax"):
            raise ValueError(f"mode must be 'sample' or 'argmax', got {mode!r}")
        self.name = "agent" if mode == "sample" else "agent_argmax"
        self.world_model, self.actor, self.encode = world_model, actor, encode
        self.device, self.mode = device, mode
        self.critic_slow, self.gamma, self.lam, self.horizon = critic_slow, gamma, lam, horizon
        self.imagined_start_return: float | None = None
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> None:
        torch.manual_seed(fork_seed(int(seed or 0), f"live:{self.name}"))
        self._state = self.world_model.rssm.initial_state(1, self.device)
        self._last_action = NOOP
        self._first = True
        self.imagined_start_return = None

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        inputs = torch.as_tensor(self.encode(obs[None])).to(self.device)
        embedding = self.world_model.encoder(inputs)[:, None]           # (1, 1, E)
        action_in = torch.tensor([[self._last_action]], device=self.device)
        out = self.world_model.rssm.observe(embedding, action_in, state=self._state)
        self._state = (out["h"][:, -1], out["z"][:, -1])
        latent = out["latent"][:, -1]
        if self._first and self.critic_slow is not None:
            self.imagined_start_return = self._imagined_return()
        self._first = False
        dist = self.actor.dist(latent)
        action = int(dist.sample() if self.mode == "sample" else dist.probs.argmax(-1))
        self._last_action = action
        return action

    def _imagined_return(self) -> float:
        wm = self.world_model
        traj = wm.rssm.imagine_policy(lambda l: self.actor.dist(l).sample(),
                                      self._state, self.horizon)
        heads = wm.heads(traj["latent"][:, 1:])
        rewards = read_reward(heads["reward"], wm.cfg.train.reward_transform)
        discounts = self.gamma * torch.sigmoid(heads["continue_logit"])
        returns = lambda_returns(rewards, discounts, self.critic_slow(traj["latent"]), self.lam)
        return float(returns[0, 0])


def run_episodes(env, policy, seeds) -> list[dict]:
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        policy.reset(seed=int(seed))
        total, length = 0.0, 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            total += float(reward)
            length += 1
            if terminated or truncated:
                break
        episodes.append({"seed": int(seed), "return": total, "length": length,
                         "killed": bool(terminated), "truncated": bool(truncated),
                         "imagined_return": getattr(policy, "imagined_start_return", None)})
    return episodes


def seed_mismatches(record: dict, steps=None) -> list[str]:
    """Names of the episode lists whose per-episode seeds differ from `record["seeds"]`.

    The paired gate subtracts the random policy's return from the agent's seed by seed;
    a list in a different order (or over different seeds) pairs the wrong episodes.
    Checks `random`, `untrained`, and `sample`/`argmax` of each rung in `steps` (default
    every rung the record holds).
    """
    lists = {"random": record["random"], "untrained": record["untrained"]}
    for step in (record["rungs"] if steps is None else steps):
        for mode, episodes in record["rungs"][step].items():
            lists[f"rung {step} {mode}"] = episodes
    return [name for name, episodes in lists.items()
            if [e["seed"] for e in episodes] != record["seeds"]]


def paired_gate(agent_returns, random_returns, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """§5.5: the mean per-seed difference, bootstrapped over seeds."""
    d = np.asarray(agent_returns, dtype=np.float64) - np.asarray(random_returns, dtype=np.float64)
    rng = np.random.default_rng(seed)
    reps = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(resamples)])
    low, high = np.percentile(reps, [2.5, 97.5])
    return {"point": float(d.mean()), "ci_low": float(low), "ci_high": float(high),
            "passed": bool(low > 0.0), "n": int(d.size)}
