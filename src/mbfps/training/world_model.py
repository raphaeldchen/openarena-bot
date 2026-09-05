"""World-model training -- milestone M3.

Encoder, RSSM and heads trained jointly on the frozen M1 dataset. The encoder
is the only arm-varying part; the RSSM and heads consume the 2048-d embedding
M2 equalises, so they are byte-identical across arms.

`privileged_state` is never read here. It is evaluation-only and appears only
in `mbfps.eval`.
"""

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from mbfps.data.split import VAL_FRACTION, episode_split
from mbfps.models.encoders import build_encoder, encoder_backbone, encoder_input_kind
from mbfps.models.heads import WorldModelHeads, continue_target
from mbfps.models.rssm import KL_FREE_BITS, RSSM, RSSMConfig, kl_loss
from mbfps.utils.config import Config
from mbfps.utils.device import get_device
from mbfps.utils.seeding import seed_everything


class WorldModel(nn.Module):
    """Encoder + RSSM + heads."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_kind = encoder_input_kind(cfg.encoder)
        self.encoder = build_encoder(cfg.encoder)
        self.rssm = RSSM(
            RSSMConfig(embed_dim=cfg.encoder.embed_dim), seed=cfg.train.seed
        )
        self.heads = WorldModelHeads(
            embed_dim=cfg.encoder.embed_dim, seed=cfg.train.seed
        )

    def embed(self, batch: dict[str, Any]) -> torch.Tensor:
        """Encode a `(B, T+1, ...)` window down to `(B, T, 2048)`.

        The window carries one more frame than transitions. `RSSM.observe`
        requires `actions[:, i]` to be the action that PRODUCED
        `embeddings[:, i]` -- i.e. embedding `i` must be the frame action `i`
        led to, not the frame it was taken from. `batch["actions"][i]` is
        taken AT `obs[i]` and leads to `obs[i+1]`, so the RSSM consumes the
        LAST T of the T+1 encoded frames -- `obs[1:T+1]` -- paired with the T
        actions `actions[0:T]`. Returning `embeddings[:, :-1]` (the FIRST T)
        instead would pair each action with the frame it was taken FROM,
        making the posterior acausal.
        """
        source = batch["obs"] if self.input_kind == "obs" else batch["features"]
        b, t_plus_one = source.shape[0], source.shape[1]
        flat = source.reshape(b * t_plus_one, *source.shape[2:])
        embeddings = self.encoder(flat).view(b, t_plus_one, -1)
        return embeddings[:, 1:]

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        embeddings = self.embed(batch)
        out = self.rssm.observe(embeddings, batch["actions"])
        predictions = self.heads(out["latent"])

        embedding_loss = F.mse_loss(predictions["embedding"], embeddings.detach())
        reward_loss = F.mse_loss(predictions["reward"], batch["rewards"])
        continue_loss = F.binary_cross_entropy_with_logits(
            predictions["continue_logit"],
            continue_target(batch["terminated"], batch["truncated"]),
        )
        kl, kl_parts = kl_loss(out["post_logits"], out["prior_logits"])

        loss = embedding_loss + reward_loss + continue_loss + kl
        parts = {
            "embedding": float(embedding_loss.detach()),
            "reward": float(reward_loss.detach()),
            "continue": float(continue_loss.detach()),
            "kl_dyn": kl_parts["dyn"],
            "kl_rep": kl_parts["rep"],
        }
        return loss, parts


def _kl_rate(kl_values: list[float]) -> float:
    """Share of steps whose dyn KL cleared the free-bits floor.

    A RATE, not a max. A max reads True after a single transient step above the
    floor: measured, it reported True on a run whose prior trained on 1.1% of
    steps while another arm's trained on 11.2%. Both looked identical through a
    boolean, so a cross-arm comparison would have measured that asymmetry rather
    than the representations.
    """
    if not kl_values:
        return 0.0
    return sum(1 for v in kl_values if v > KL_FREE_BITS) / len(kl_values)


def train_world_model(
    cfg: Config,
    buffer: ReplayBuffer,
    out_dir: Path | None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Train one arm's world model and return its history."""
    from mbfps.utils.device import to_device

    seed_everything(cfg.train.seed)
    device = get_device(prefer=cfg.train.device)
    model = WorldModel(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    backbone = encoder_backbone(cfg.encoder)
    # Train on the TRAINING episodes only. The split seed is fixed at 0 and is
    # deliberately NOT cfg.train.seed: every arm and every seed must be held out
    # on the same episodes, or the comparison measures which episodes each run
    # happened to get rather than which representation is better.
    train_paths, _ = episode_split(
        buffer.episode_paths(), val_fraction=VAL_FRACTION, seed=0
    )
    loader = SequenceLoader(
        buffer,
        batch_size=cfg.train.batch_size,
        seq_len=cfg.train.seq_len,
        seed=cfg.train.seed,
        load_obs=model.input_kind == "obs",
        load_features=model.input_kind == "features",
        feature_backbone=backbone or "dinov2",
        paths=train_paths,
    )

    losses: list[float] = []
    history_parts: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(cfg.train.steps):
        loss, parts = model(to_device(loader.sample(), device))
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimiser.step()
        losses.append(float(loss.detach()))
        history_parts.append(parts)
        if log_every and (step + 1) % log_every == 0:
            print(f"[{cfg.arm}] step {step + 1}/{cfg.train.steps} loss={losses[-1]:.5f}")

    kl_values = [p["kl_dyn"] for p in history_parts]
    kl_dyn_max = max(kl_values, default=0.0)
    kl_rate = _kl_rate(kl_values)
    history = {
        "arm": cfg.arm,
        "steps": cfg.train.steps,
        "loss": losses,
        "parts": history_parts,
        "seconds": time.perf_counter() - start,
        # Free bits clamp the dyn KL below KL_FREE_BITS nats, so prior_net is
        # frozen until the posterior becomes informative enough to clear the
        # floor. Whether that ever happens on this dataset is empirical, and a
        # run where it never happens trained no dynamics prior at all -- every
        # imagined rollout would come from random weights. Record it rather
        # than assume it.
        "kl_dyn_max": kl_dyn_max,
        "kl_rate_above_free_bits": kl_rate,
    }
    if kl_rate < 0.5:
        print(
            f"[{cfg.arm}] WARNING: kl_dyn exceeded the {KL_FREE_BITS} nat floor on only "
            f"{100 * kl_rate:.1f}% of steps (peak {kl_dyn_max:.4f}). The dynamics prior "
            "is largely untrained, so imagine() runs close to its initialisation. Arms "
            "whose rates differ cannot be compared: the comparison would measure how "
            "much each prior trained, not the representations."
        )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"arm": cfg.arm, "seed": cfg.train.seed, "state_dict": model.state_dict()},
            out_dir / f"world_model_{cfg.arm}_seed{cfg.train.seed}.pt",
        )
    return history
