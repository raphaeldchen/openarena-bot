"""Pins imagined latents against silent regression during refactors.

Not a correctness test -- a change-detector, deliberately. When it fails,
either the change was intended (update the hash, say why in the commit) or a
refactor silently altered the dynamics.
"""

import hashlib

import numpy as np
import torch

from mbfps.models.rssm import RSSM, RSSMConfig
from mbfps.utils.seeding import seed_everything


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def test_imagined_latents_are_stable():
    seed_everything(0)
    rssm = RSSM(RSSMConfig(), seed=0)
    torch.manual_seed(0)
    actions = torch.arange(12).remainder(6).view(1, 12)
    state = rssm.initial_state(1, torch.device("cpu"))
    with torch.no_grad():
        latent = rssm.imagine(actions, state)["latent"].numpy()
    digest = _hash(np.round(latent, 5))
    assert digest == "f62e4acf423654eb", (
        f"imagined latents changed; got {digest}. If the change was intended, "
        "update this constant and explain why in the commit message."
    )
