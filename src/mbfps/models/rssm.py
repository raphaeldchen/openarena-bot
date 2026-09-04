"""Recurrent state-space model -- the latent dynamics core (milestone M3).

State is a pair: a deterministic GRU state `h` carrying history, and a
stochastic state `z` of 32 categorical variables over 32 classes each.
Categoricals rather than a Gaussian follow DreamerV3; a hard one-hot sample has
zero gradient, so a straight-through estimator passes the softmax gradient
through, which is what lets the encoder train at all.

`observe` runs the posterior, which sees the encoder embedding. `imagine` runs
the prior only, driven by actions alone -- that asymmetry is the whole point of
the architecture, and the reason the actor in M4 never touches a real frame.

This module is arm-invariant: it consumes the 2048-d embedding that M2
equalises across arms, so `encoders.py` remains the only module that differs.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbfps.utils.seeding import seeded_init


@dataclass(frozen=True)
class RSSMConfig:
    """Shared byte-for-byte across arms."""

    h_dim: int = 512
    z_cats: int = 32
    z_classes: int = 32
    embed_dim: int = 2048
    n_actions: int = 6
    hidden: int = 512


LATENT_DIM = 512 + 32 * 32
"""What the heads and probes consume: `h` concatenated with flattened `z`."""


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class RSSM(nn.Module):
    """Deterministic GRU path plus a categorical stochastic state."""

    def __init__(self, cfg: RSSMConfig, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        computed_latent_dim = cfg.h_dim + cfg.z_cats * cfg.z_classes
        if computed_latent_dim != LATENT_DIM:
            raise ValueError(
                f"cfg.h_dim + cfg.z_cats * cfg.z_classes = {computed_latent_dim}, "
                f"but LATENT_DIM = {LATENT_DIM} is hardcoded as 512 + 32*32 for "
                "the heads and probes that consume `latent`. A non-default "
                "RSSMConfig that does not match LATENT_DIM would silently "
                "diverge from every downstream shape assumption."
            )
        self.z_dim = cfg.z_cats * cfg.z_classes
        with seeded_init(seed, "rssm"):
            self.cell = nn.GRUCell(self.z_dim + cfg.n_actions, cfg.h_dim)
            self.prior_net = _mlp(cfg.h_dim, cfg.hidden, self.z_dim)
            self.post_net = _mlp(cfg.h_dim + cfg.embed_dim, cfg.hidden, self.z_dim)

    def initial_state(self, batch_size: int, device: torch.device):
        """Zero `(h, z)`."""
        return (
            torch.zeros(batch_size, self.cfg.h_dim, device=device),
            torch.zeros(batch_size, self.z_dim, device=device),
        )

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """One-hot sample with a straight-through gradient.

        `argmax` has zero gradient everywhere, so the backward pass uses the
        softmax probabilities instead: `probs + (onehot - probs).detach()` is
        numerically the one-hot in forward and the softmax in backward.

        **Sampling is always stochastic, including at evaluation.** Taking the
        mode looks like a free way to make rollouts reproducible and is not:
        measured, it collapses the imagined trajectory to 3 distinct latents out
        of 45 and roughly quadruples position error (1406 against 356). It also
        degrades WITH training, as sharpening logits make the mode more
        dominant, so an untrained smoke test will not reveal it. Reproducibility
        comes from seeding the generator -- see `evaluate_rollout`.
        """
        shaped = logits.view(*logits.shape[:-1], self.cfg.z_cats, self.cfg.z_classes)
        probs = F.softmax(shaped, dim=-1)
        index = torch.distributions.Categorical(probs=probs).sample()
        onehot = F.one_hot(index, self.cfg.z_classes).to(probs.dtype)
        return (probs + (onehot - probs).detach()).flatten(-2)

    def _step(self, h: torch.Tensor, z: torch.Tensor, action_onehot: torch.Tensor):
        return self.cell(torch.cat([z, action_onehot], dim=-1), h)

    def _onehot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), self.cfg.n_actions).float()

    def observe(self, embeddings, actions, state=None) -> dict[str, torch.Tensor]:
        """Filter: the posterior sees each embedding.

        Ordering, per step: `h` is advanced with `actions[:, i]` FIRST, and the
        posterior is formed only afterwards from that already-advanced `h`. So
        `post_logits[:, i]` depends on `actions[0..i]` inclusive -- action `i`
        has already moved `h` before the posterior at `i` is conditioned on it.
        Moving the `_step` call to after the posterior is formed keeps every
        shape and loss the same but silently conditions the posterior on the
        PRE-action `h` instead -- a real behaviour change with no shape or
        smoke test able to catch it.

        Action-time convention (binds every caller of `observe`): `actions[:,
        i]` must be the action that PRODUCED `embeddings[:, i]` -- i.e.
        `embeddings[:, i]` is the encoding of the frame action `i` led to, not
        the frame it was taken from. An `Episode` stores `obs` of length T+1,
        where `actions[i]` is taken AT `obs[i]` and leads to `obs[i+1]`; the
        correct pairing is therefore `encoder(obs[1:T+1])` with `actions[0:T]`.
        Pairing `obs[0:T]` with `actions[0:T]` instead makes the model
        acausal: the posterior at step `i` would be conditioned on the frame
        from BEFORE action `i` was taken, rather than the frame it caused.
        """
        b, t, _ = embeddings.shape
        h, z = state if state is not None else self.initial_state(b, embeddings.device)
        actions_onehot = self._onehot_actions(actions)
        hs, zs, priors, posts = [], [], [], []
        for i in range(t):
            h = self._step(h, z, actions_onehot[:, i])
            prior_logits = self.prior_net(h)
            post_logits = self.post_net(torch.cat([h, embeddings[:, i]], dim=-1))
            z = self._sample(post_logits)
            hs.append(h); zs.append(z)
            priors.append(prior_logits); posts.append(post_logits)
        return self._pack(hs, zs, priors, posts)

    def imagine(self, actions, state) -> dict[str, torch.Tensor]:
        """Roll forward on the prior alone -- no embeddings consumed."""
        h, z = state
        actions_onehot = self._onehot_actions(actions)
        hs, zs, priors = [], [], []
        for i in range(actions.shape[1]):
            h = self._step(h, z, actions_onehot[:, i])
            prior_logits = self.prior_net(h)
            z = self._sample(prior_logits)
            hs.append(h); zs.append(z); priors.append(prior_logits)
        return self._pack(hs, zs, priors, None)

    def _pack(self, hs, zs, priors, posts) -> dict[str, torch.Tensor]:
        cats, classes = self.cfg.z_cats, self.cfg.z_classes
        h = torch.stack(hs, dim=1)
        z = torch.stack(zs, dim=1)
        out = {
            "h": h,
            "z": z,
            "latent": torch.cat([h, z], dim=-1),
            "prior_logits": torch.stack(priors, dim=1).unflatten(-1, (cats, classes)),
        }
        if posts is not None:
            out["post_logits"] = torch.stack(posts, dim=1).unflatten(-1, (cats, classes))
        return out


def _categorical_kl(logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) summed over the 32 categorical groups, averaged over B and T."""
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    per_group = (log_q.exp() * (log_q - log_p)).sum(-1)
    return per_group.sum(-1).mean()


KL_FREE_BITS = 0.20
"""Below this the KL is not optimised at all, so the posterior cannot collapse.

**Deliberately NOT the governing spec's 1.0 nat.** That value was inherited from
DreamerV3 and measured wrong for this setup: the dynamics prior trains only when
`dyn` exceeds the floor, and at 1.0 it receives gradient on 1 of 9 sampled steps
because the measured KL rarely clears it. A floor of 0 is equally wrong in the
other direction -- it penalises every nat and collapses the posterior into the
prior. Measured over 2,000 steps on the real dataset:

    free_bits   prior_net gets gradient   kl_dyn at end
    0.00        --                        0.0001  (collapsed)
    0.05        7/9 sampled steps         0.354
    0.20        8/9 sampled steps         0.333   <- chosen
    1.00        1/9 sampled steps         0.717

Init KL is 0.022-0.038 nat depending on arm, so the prior is still clamped for
the first steps; `train_world_model` records how often the floor is actually
cleared rather than assuming it.
"""


def kl_loss(
    post_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    free_bits: float = KL_FREE_BITS,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL with DreamerV3 balancing and free bits.

    Two terms with stop-gradients on opposite sides. `dyn` pulls the PRIOR
    toward the posterior; `rep` pulls the POSTERIOR toward the prior, and is
    weighted five times lower so the representation is not dragged toward an
    uninformative dynamics prediction.

    Free bits clamp each term at `free_bits` nats. Below that floor the KL is
    not optimised at all, which is what prevents posterior collapse -- without
    it the cheapest way to cut the loss is to make the posterior carry nothing.
    """
    dyn = _categorical_kl(post_logits.detach(), prior_logits)
    rep = _categorical_kl(post_logits, prior_logits.detach())
    floor = torch.tensor(free_bits, dtype=dyn.dtype, device=dyn.device)
    loss = dyn_scale * torch.maximum(dyn, floor) + rep_scale * torch.maximum(rep, floor)
    return loss, {"dyn": float(dyn.detach()), "rep": float(rep.detach())}
