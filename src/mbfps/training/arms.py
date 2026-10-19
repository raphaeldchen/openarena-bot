"""The two arms of M5a behind one interface (spec §4, §5).

Both read one encoder kind's cached frozen-backbone features (`frozen_ssl`,
i.e. DINOv2, by default, as in M5a) and are built from the run seed. `update_round()` is one update round of §3.2; the driver
calls it 20 times per 100 budget steps."""

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader, feature_suffix
from mbfps.eval.live import AgentPolicy
from mbfps.training.ac_trainer import ACConfig, ActorCriticLearner
from mbfps.training.dqn import (DQNConfig, EpsilonGreedyQPolicy, QNetwork,
                                TransitionSampler, dqn_step, epsilon)
from mbfps.training.world_model import WorldModel, world_model_step
from mbfps.models.encoders import encoder_backbone
from mbfps.utils.config import EncoderConfig, get_config
from mbfps.utils.device import to_device
from mbfps.utils.seeding import seed_everything

def _feature_backbone(encoder: str) -> str:
    backbone = encoder_backbone(EncoderConfig(kind=encoder))
    if backbone is None:
        raise ValueError(f"encoder kind {encoder!r} reads pixels; the online loop feeds "
                         "cached features only")
    return backbone


class _Means:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.n = 0

    def add(self, d: dict[str, float]) -> None:
        for k, v in d.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1

    def pop(self) -> dict[str, float]:
        out = {k: v / self.n for k, v in self.sums.items()} if self.n else {}
        self.sums, self.n = {}, 0
        return out


class DreamerArm:
    name, gated_mode = "dreamer", "sample"

    def __init__(self, seed: int, device: torch.device, encode, replay: ReplayBuffer,
                 batch_size: int = 16, seq_len: int = 64, encoder: str = "frozen_ssl",
                 n_actions: int = 4) -> None:
        self.encoder, self.backbone = encoder, _feature_backbone(encoder)
        self.n_actions = n_actions
        seed_everything(seed)   # the world model is built first after this, as in M3/M4
        self.device, self.encode = device, encode
        self.world_model = WorldModel(get_config(
            encoder, seed=seed, device=device.type, n_actions=n_actions,
            pad_short=True, reward_transform="symlog")).to(device)
        self.wm_opt = torch.optim.Adam(self.world_model.parameters(),
                                       lr=self.world_model.cfg.train.lr)
        self.learner = ActorCriticLearner(
            n_actions, ACConfig(seed=seed, device=device.type, batch_size=batch_size,
                                seq_len=seq_len, bound_reward_readout=True,
                                drop_terminal_starts=True), device)
        self.loader = SequenceLoader(replay, batch_size=batch_size, seq_len=seq_len, seed=seed,
                                     load_obs=False, load_features=True,
                                     feature_backbone=self.backbone, paths=[], pad_short=True)
        self.batch_size, self.seq_len = batch_size, seq_len
        self.reward_bounds: tuple[float, float] | None = None
        self.isolation_ok: bool | None = None
        self._means = _Means()
        self._replay_transitions = 0

    def settings(self) -> dict:
        """The settings this arm ran with, defaults included (written to the record)."""
        return {"encoder": self.encoder, "n_actions": self.n_actions,
                "batch_size": self.batch_size, "seq_len": self.seq_len,
                "ac": asdict(self.learner.cfg)}

    def add_episode(self, path: Path) -> None:
        self.loader.add(path)
        with np.load(path) as d:
            rewards = d["rewards"]
        self._replay_transitions += len(rewards)
        lo, hi = float(rewards.min()), float(rewards.max())
        if self.reward_bounds is not None:
            lo, hi = min(lo, self.reward_bounds[0]), max(hi, self.reward_bounds[1])
        self.reward_bounds = (lo, hi)

    def update_round(self) -> dict[str, float]:
        batch = to_device(self.loader.sample(), self.device)
        wm_loss, parts = world_model_step(self.world_model, self.wm_opt, batch)
        ac = self.learner.step(self.world_model, batch, reward_bounds=self.reward_bounds)
        if self.isolation_ok is None:
            self.isolation_ok = all(p.grad is None for p in self.world_model.parameters())
            if not self.isolation_ok:
                raise RuntimeError("an actor-critic step put a gradient on the world model (§3.5)")
        out = {"wm_loss": wm_loss, **{f"wm_{k}": v for k, v in parts.items()}, **ac}
        self._means.add(out)
        return out

    def behaviour_policy(self, budget_step_fn: Callable[[], int]) -> AgentPolicy:
        return AgentPolicy(self.world_model, self.learner.actor, self.encode, self.device,
                           mode="sample")

    def eval_policies(self) -> dict:
        kw = {"critic_slow": self.learner.critic_slow}
        return {"sample": AgentPolicy(self.world_model, self.learner.actor, self.encode,
                                      self.device, mode="sample", **kw),
                "argmax": AgentPolicy(self.world_model, self.learner.actor, self.encode,
                                      self.device, mode="argmax", **kw)}

    def diagnostics(self) -> dict:
        return {**self._means.pop(), "reward_bounds": self.reward_bounds,
                "replay_transitions": self._replay_transitions,
                "replay_episodes": self.loader.n_episodes}

    def state_dict(self) -> dict:
        # A snapshot, not views: optimiser state tensors are otherwise shared with the
        # arm that loads it (`Optimizer.load_state_dict` does not copy same-device
        # tensors), so two live arms would update one set of Adam moments.
        return copy.deepcopy(self._state())

    def _state(self) -> dict:
        return {"world_model": self.world_model.state_dict(), "wm_opt": self.wm_opt.state_dict(),
                "learner": self.learner.state_dict(),
                "loader_rng": self.loader.rng.bit_generator.state,
                "isolation_ok": self.isolation_ok}

    def load_state_dict(self, sd: dict) -> None:
        self.world_model.load_state_dict(sd["world_model"])
        self.wm_opt.load_state_dict(sd["wm_opt"])
        self.learner.load_state_dict(sd["learner"])
        self.loader.rng.bit_generator.state = sd["loader_rng"]
        self.isolation_ok = sd["isolation_ok"]


class DQNArm:
    name, gated_mode = "dqn", "greedy"

    def __init__(self, seed: int, device: torch.device, encode,
                 cfg: DQNConfig = DQNConfig(), encoder: str = "frozen_ssl",
                 n_actions: int = 4) -> None:
        self.encoder, self.backbone = encoder, _feature_backbone(encoder)
        self.n_actions = n_actions
        seed_everything(seed)
        self.device, self.encode, self.cfg = device, encode, cfg
        self.q = QNetwork(n_actions, seed=seed, frame_stack=cfg.frame_stack,
                          hidden=cfg.hidden, encoder_kind=encoder).to(device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.sampler = TransitionSampler(seed=seed, cfg=cfg)
        self.updates = 0
        self.isolation_ok = None   # not applicable to DQN
        self._means = _Means()
        self._budget_step_fn: Callable[[], int] = lambda: 0

    def settings(self) -> dict:
        """The settings this arm ran with, defaults included (written to the record)."""
        return {"encoder": self.encoder, "n_actions": self.n_actions, "cfg": asdict(self.cfg)}

    def add_episode(self, path: Path) -> None:
        with np.load(path) as d:
            arrays = (d["actions"], d["rewards"], d["terminated"], d["truncated"])
        features = np.load(path.with_suffix(feature_suffix(self.backbone)), mmap_mode="r")
        self.sampler.add(features, *arrays)

    def update_round(self) -> dict[str, float]:
        self.updates += 1
        out = dqn_step(self.q, self.q_target, self.opt, self.sampler.sample(self.cfg.batch_size),
                       self.updates, self.cfg, self.device)
        self._means.add(out)
        return out

    def behaviour_policy(self, budget_step_fn: Callable[[], int]) -> EpsilonGreedyQPolicy:
        self._budget_step_fn = budget_step_fn
        return EpsilonGreedyQPolicy(self.q, self.encode, self.device,
                                    epsilon_fn=lambda: epsilon(budget_step_fn(), self.cfg),
                                    frame_stack=self.cfg.frame_stack, name="dqn")

    def eval_policies(self) -> dict:
        k = self.cfg.frame_stack
        return {"greedy": EpsilonGreedyQPolicy(self.q, self.encode, self.device, lambda: 0.0,
                                               frame_stack=k, name="dqn_greedy"),
                "eps0_05": EpsilonGreedyQPolicy(self.q, self.encode, self.device, lambda: 0.05,
                                                frame_stack=k, name="dqn_eps")}

    def diagnostics(self) -> dict:
        return {**self._means.pop(), "updates": self.updates,
                "epsilon": epsilon(self._budget_step_fn(), self.cfg),
                "replay_transitions": self.sampler.n_transitions}

    def state_dict(self) -> dict:
        return copy.deepcopy(self._state())   # a snapshot, as DreamerArm's

    def _state(self) -> dict:
        return {"q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
                "opt": self.opt.state_dict(), "updates": self.updates,
                "sampler_rng": self.sampler.rng.bit_generator.state}

    def load_state_dict(self, sd: dict) -> None:
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q_target"])
        self.opt.load_state_dict(sd["opt"])
        self.updates = int(sd["updates"])
        self.sampler.rng.bit_generator.state = sd["sampler_rng"]
