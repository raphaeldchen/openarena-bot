"""Actor-critic in imagination -- milestone M4, Stage B.

The world model is loaded from Stage A and FROZEN: every parameter has
`requires_grad=False`, and the optimisers hold only the actor's and critic's
parameters. Start states are every REAL posterior state of a replay batch from
the training split (padding excluded). From each, `imagine_policy` rolls the
prior 15 steps with the actor sampling each action. Returns, losses and the
return scale follow DreamerV3 (spec §5.3).

No real-environment transition ever reaches a gradient here: the trainer
never constructs an environment.
"""

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.models.encoders import encoder_backbone
from mbfps.models.heads import Actor, Critic
from mbfps.models.transforms import read_reward, reward_target
from mbfps.utils.device import to_device
from mbfps.utils.seeding import seed_everything


def reward_target_value(reward: float, transform: str) -> float:
    """A reward bound in the head's output space (M5a §4.3)."""
    return float(reward_target(torch.tensor(float(reward)), transform))


@dataclass(frozen=True)
class ACConfig:
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    entropy: float = 3e-4
    critic_ema: float = 0.98
    scale_decay: float = 0.99
    updates: int = 10_000
    checkpoint_every: int = 2_000
    batch_size: int = 16
    seq_len: int = 64
    seed: int = 0
    device: str = "mps"
    grad_clip: float = 100.0
    bound_reward_readout: bool = False
    """M5a §4.3: clip the reward head to the observed reward range before the
    inverse transform. Off reproduces M4."""
    drop_terminal_starts: bool = False
    """M5a §4.3: never start imagination at a terminated transition. Off
    reproduces M4."""

    def __post_init__(self) -> None:
        if self.checkpoint_every < 1 or self.updates % self.checkpoint_every:
            raise ValueError(
                f"checkpoint_every ({self.checkpoint_every}) must divide updates "
                f"({self.updates}): the gate is read at the final update, which must be a rung"
            )


def lambda_returns(rewards, discounts, values, lam: float) -> torch.Tensor:
    """`G_t = r_t + d_t * ((1 - lam) * V_{t+1} + lam * G_{t+1})`, `G_H = V_H`.
    `rewards`, `discounts`: `(N, H)`; `values`: `(N, H + 1)`."""
    horizon = rewards.shape[1]
    out = [None] * horizon
    nxt = values[:, horizon]
    for t in reversed(range(horizon)):
        nxt = rewards[:, t] + discounts[:, t] * ((1 - lam) * values[:, t + 1] + lam * nxt)
        out[t] = nxt
    return torch.stack(out, dim=1)


def continue_weights(cont: torch.Tensor) -> torch.Tensor:
    """`w_0 = 1`, `w_t = prod_{k<t} cont_k`: steps past a predicted end count less."""
    ones = torch.ones_like(cont[:, :1])
    return torch.cumprod(torch.cat([ones, cont[:, :-1]], dim=1), dim=1)


class ReturnScale:
    """DreamerV3's return normalisation: `S = max(1, P95 - P5)`, the
    percentiles tracked as an EMA. The first update sets them directly."""

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.low: float | None = None
        self.high: float | None = None

    def update(self, returns: torch.Tensor) -> float:
        # `torch.quantile` needs float32 or float64 (MPS has no float64), and is
        # supported on MPS in this torch build (checked); the vector is N * horizon.
        flat = returns.detach().flatten().float()
        low = float(torch.quantile(flat, 0.05))
        high = float(torch.quantile(flat, 0.95))
        if self.low is None:
            self.low, self.high = low, high
        else:
            self.low = self.decay * self.low + (1 - self.decay) * low
            self.high = self.decay * self.high + (1 - self.decay) * high
        return max(1.0, self.high - self.low)


def freeze(world_model: nn.Module) -> None:
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)
        p.grad = None


@torch.no_grad()
def start_states(
    world_model, batch: dict, drop_terminal: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior `(h, z)` at every masked-in position of `batch`.

    `drop_terminal` (M5a §4.3) also excludes positions whose transition
    terminated the episode; truncated (time-limit) positions are kept."""
    out = world_model.rssm.observe(world_model.embed(batch), batch["actions"])
    mask = batch["mask"].bool()
    if drop_terminal:
        mask = mask & ~batch["terminated"].bool()
    return out["h"][mask], out["z"][mask]


def _ema(slow: nn.Module, fast: nn.Module, rate: float) -> None:
    with torch.no_grad():
        for s, f in zip(slow.parameters(), fast.parameters()):
            s.mul_(rate).add_(f, alpha=1 - rate)


def _save(out_dir: Path, arm: str, seed: int, step: int, n_actions: int,
          actor, critic, critic_slow) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"actor_critic_{arm}_seed{seed}_step{step}.pt"
    torch.save({"arm": arm, "seed": seed, "step": step, "n_actions": n_actions,
                "actor": actor.state_dict(), "critic": critic.state_dict(),
                "critic_slow": critic_slow.state_dict()}, path)
    return path


def load_actor(path: Path, device: torch.device) -> Actor:
    payload = torch.load(path, map_location=device, weights_only=True)
    actor = Actor(int(payload["n_actions"]), seed=int(payload["seed"])).to(device)
    actor.load_state_dict(payload["actor"])
    return actor.eval()


class ActorCriticLearner:
    """One DreamerV3 actor-critic update on a world model's imagination (M4
    §5.3), reusable by the fixed-dataset trainer (M4) and the online loop (M5a).

    The world model is read under `no_grad` only: no gradient from this step
    can reach it, whether or not its parameters require grad."""

    def __init__(self, n_actions: int, cfg: ACConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.actor = Actor(n_actions, seed=cfg.seed).to(device)
        self.critic = Critic(seed=cfg.seed).to(device)
        self.critic_slow = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.scale = ReturnScale(cfg.scale_decay)

    def step(self, world_model, batch: dict,
             reward_bounds: tuple[float, float] | None = None) -> dict[str, float]:
        cfg = self.cfg
        transform = world_model.cfg.train.reward_transform
        h, z = start_states(world_model, batch, drop_terminal=cfg.drop_terminal_starts)
        n_real = float(batch["mask"].bool().sum())
        bounds = reward_bounds if cfg.bound_reward_readout else None
        sample_actor = lambda latent: self.actor.dist(latent).sample()
        with torch.no_grad():
            traj = world_model.rssm.imagine_policy(sample_actor, (h, z), cfg.horizon)
            heads = world_model.heads(traj["latent"][:, 1:])
            clipped_share = 0.0
            if bounds is not None:
                lo, hi = (reward_target_value(b, transform) for b in bounds)
                raw = heads["reward"]
                clipped_share = float(((raw < lo) | (raw > hi)).float().mean())
            rewards = read_reward(heads["reward"], transform, bounds=bounds)
            cont = torch.sigmoid(heads["continue_logit"])
            values_slow = self.critic_slow(traj["latent"])
            returns = lambda_returns(rewards, cfg.gamma * cont, values_slow, cfg.lam)
            weights = continue_weights(cont)
        latents = traj["latent"][:, :-1]

        values = self.critic(latents)
        critic_loss = (0.5 * weights * (values - returns).pow(2)).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()
        _ema(self.critic_slow, self.critic, cfg.critic_ema)

        s = self.scale.update(returns)
        dist = self.actor.dist(latents)
        advantage = ((returns - values.detach()) / s).detach()
        entropy = dist.entropy()
        actor_loss = (weights * (-advantage * dist.log_prob(traj["action"])
                                 - cfg.entropy * entropy)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()

        out = {"actor_loss": float(actor_loss.detach()),
               "critic_loss": float(critic_loss.detach()),
               "scale": s,
               "imagined_return": float(returns[:, 0].mean()),
               "entropy": float(entropy.mean().detach()),
               "n_starts": float(h.shape[0]),
               "n_dropped": n_real - float(h.shape[0]),
               "clipped_share": clipped_share}
        if h.device.type == "mps":
            # N (start states) differs every update, so every update allocates
            # tensors of new shapes and the MPS caching allocator never reuses
            # its blocks. Driver memory grew ~60 MB/update and hit OOM near
            # update 350 (M4); releasing the cache each update holds it flat.
            torch.mps.empty_cache()
        return out

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_slow": self.critic_slow.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "scale": {"low": self.scale.low, "high": self.scale.high}}

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_slow.load_state_dict(sd["critic_slow"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.scale.low, self.scale.high = sd["scale"]["low"], sd["scale"]["high"]


def train_actor_critic(world_model, buffer: ReplayBuffer, cfg: ACConfig,
                       out_dir: Path | None, log_every: int = 100) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = next(world_model.parameters()).device
    freeze(world_model)
    arm, n_actions = world_model.cfg.arm, world_model.cfg.train.n_actions
    learner = ActorCriticLearner(n_actions, cfg, device)

    train_paths, _ = episode_split(buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=0)
    kind = world_model.input_kind
    loader = SequenceLoader(
        buffer, batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed=cfg.seed,
        load_obs=kind == "obs", load_features=kind == "features",
        feature_backbone=encoder_backbone(world_model.cfg.encoder) or "dinov2",
        paths=train_paths, pad_short=True,
    )
    bounds = None
    if cfg.bound_reward_readout:
        rewards = np.concatenate([np.load(p)["rewards"] for p in train_paths])
        bounds = (float(rewards.min()), float(rewards.max()))

    history: dict[str, Any] = {"actor_loss": [], "critic_loss": [], "scale": [],
                               "imagined_return": [], "entropy": [], "checkpoints": []}
    isolation_ok = True
    start = time.perf_counter()
    for update in range(1, cfg.updates + 1):
        batch = to_device(loader.sample(), device)
        out = learner.step(world_model, batch, reward_bounds=bounds)
        if update == 1:
            isolation_ok = all(p.grad is None for p in world_model.parameters())
            if not isolation_ok:
                raise RuntimeError("a world-model parameter received a gradient (spec §5.2)")
        for key in ("actor_loss", "critic_loss", "scale", "imagined_return", "entropy"):
            history[key].append(out[key])
        if out_dir is not None and update % cfg.checkpoint_every == 0:
            path = _save(Path(out_dir), arm, cfg.seed, update, n_actions,
                         learner.actor, learner.critic, learner.critic_slow)
            history["checkpoints"].append(str(path))
        if log_every and update % log_every == 0:
            print(f"[ac {arm}] update {update}/{cfg.updates} actor={history['actor_loss'][-1]:+.4f} "
                  f"critic={history['critic_loss'][-1]:.4f} G0={history['imagined_return'][-1]:+.2f}")

    seconds = time.perf_counter() - start
    history.update({"arm": arm, "seed": cfg.seed, "updates": cfg.updates,
                    "seconds": seconds, "seconds_per_update": seconds / cfg.updates,
                    "isolation_ok": isolation_ok})
    return history
