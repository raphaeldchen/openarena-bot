"""Prediction heads over the RSSM latent (milestone M3).

Three heads share the latent: the embedding predictor supplies the world
model's main reconstruction signal, and the reward and continue heads supply
what M4's actor-critic will bootstrap from.

Arm-invariant, like the RSSM: these consume the latent, which derives from the
2048-d embedding M2 equalises across arms.
"""

import torch
import torch.nn as nn

from mbfps.models.rssm import LATENT_DIM
from mbfps.utils.seeding import seeded_init


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def continue_target(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    """1.0 while the episode continues, 0.0 where it genuinely terminated.

    `truncated` is deliberately unused for the value and is accepted only to
    make the asymmetry explicit at every call site. A time-limit cutoff is not
    a terminal state: the episode did not end, the clock ran out. Treating it
    as terminal teaches the critic that running out of time is as bad as dying,
    which is the classic time-limit bootstrapping bug.
    """
    del truncated  # intentionally not part of the target -- see docstring
    return (~terminated.bool()).to(torch.float32)


class WorldModelHeads(nn.Module):
    """Embedding, reward and continue predictors over the latent."""

    def __init__(self, embed_dim: int = 2048, hidden: int = 512, seed: int = 0) -> None:
        super().__init__()
        with seeded_init(seed, "world_model_heads"):
            self.embedding = _mlp(LATENT_DIM, hidden, embed_dim)
            self.reward = _mlp(LATENT_DIM, hidden, 1)
            self.continue_ = _mlp(LATENT_DIM, hidden, 1)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.embedding(latent),
            "reward": self.reward(latent).squeeze(-1),
            "continue_logit": self.continue_(latent).squeeze(-1),
        }
