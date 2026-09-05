"""Open-loop rollout evaluation -- milestone M3.

A rollout error is meaningless in isolation, so every error is bracketed:

  persistence (upper) -- hold the last context state for the whole horizon.
                         This is what "learned nothing" scores, and it is
                         physically interpretable: the agent never moved, so
                         the error IS the true displacement.
  RSSM                -- the model under test, imagining from actions alone.
  encoder floor (lower) -- probe the encoder embedding of the REAL frame at
                         each step. The best any dynamics model could reach
                         given this encoder, and what separates "the dynamics
                         model is weak" from "the encoder already discarded
                         this information".

The cross-arm number is `gap_closed`, the dimensionless fraction of that band
the model closes. Arms have different encoders and therefore different floors,
so a raw error is not comparable between them and this ratio is.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mbfps.data.episode import load_episode
from mbfps.eval.probe import (
    angle_error_degrees,
    apply_probe,
    position_error,
    probe_targets,
)


def gap_closed(
    persistence: np.ndarray, model: np.ndarray, floor: np.ndarray
) -> np.ndarray:
    """Fraction of the persistence-to-floor band the model closes.

    1.0 means as good as the encoder permits; 0.0 means no better than assuming
    the agent never moved. Negative values are returned, not clipped -- worse
    than persistence is a real result. NaN where the band has zero width, since
    the ratio is genuinely undefined there rather than zero.
    """
    band = persistence - floor
    with np.errstate(divide="ignore", invalid="ignore"):
        # NaN for a non-positive band, not merely a zero one. A negative band
        # means the floor sits ABOVE persistence, which flips the sign of the
        # ratio -- an arm worse than persistence would then score positive, and
        # "gap_closed > 0" is the M3 gate criterion. Measured on real data the
        # band is ~0.6% of the error magnitude and does go negative, so this is
        # the common case rather than a corner.
        result = np.where(band <= 0, np.nan, (persistence - model) / band)
    return result


@dataclass
class RolloutResult:
    """Per-horizon-step errors for all three references."""

    horizon: np.ndarray
    rssm_position: np.ndarray
    persistence_position: np.ndarray
    floor_position: np.ndarray
    rssm_angle: np.ndarray
    persistence_angle: np.ndarray
    floor_angle: np.ndarray

    def position_gap_closed(self) -> np.ndarray:
        return gap_closed(self.persistence_position, self.rssm_position, self.floor_position)

    def angle_gap_closed(self) -> np.ndarray:
        return gap_closed(self.persistence_angle, self.rssm_angle, self.floor_angle)


@torch.no_grad()
def evaluate_rollout(
    model,
    val_paths,
    embedding_probe_weights: dict,
    context: int = 5,
    horizon: int = 45,
    seed: int = 0,
    device: torch.device | None = None,
    feature_backbone: str | None = None,
) -> RolloutResult:
    """Condition on `context` real frames, then imagine `horizon` steps.

    Every window in every validation episode that is long enough contributes.
    """
    model.eval()
    device = device or next(model.parameters()).device
    need = context + horizon
    # Reproducibility WITHOUT taking the mode. Seeding here makes two runs of
    # this function identical; taking the categorical mode would also do that,
    # and would collapse the trajectory to 3 distinct latents out of 45 while
    # roughly quadrupling position error.
    torch.manual_seed(seed)

    rssm_pos, pers_pos, floor_pos = [], [], []
    rssm_ang, pers_ang, floor_ang = [], [], []

    for path in val_paths:
        episode = load_episode(path)
        # A fast path, not a correctness gate: the window loop below is already
        # empty for any episode of length <= need. This skips loading the
        # feature cache for an episode that could contribute nothing. (Mutation
        # testing confirms it: weakening this to `< need` changes no output,
        # because the only length the two disagree on is exactly `need`, where
        # `range(0, 0, need)` yields no windows either way.)
        if episode.length < need + 1:
            continue
        source = source_for(model, path, episode, feature_backbone)
        for start in range(0, episode.length - need, need):
            window = slice(start, start + need + 1)
            # `window` spans need+1 frames; drop the FIRST one so
            # embeddings[k] is the frame actions[k] led to, matching the
            # RSSM.observe convention documented on WorldModel.embed.
            embeddings = model.encoder(
                torch.as_tensor(source[window][1:]).to(device)
            ).unsqueeze(0)
            actions = torch.as_tensor(
                episode.actions[start : start + need].astype(np.int64)
            ).unsqueeze(0).to(device)

            observed = model.rssm.observe(
                embeddings[:, :context], actions[:, :context]
            )
            state = (observed["h"][:, -1], observed["z"][:, -1])
            imagined = model.rssm.imagine(actions[:, context:], state)

            # Floor: the posterior run on the REAL future frames, then through
            # the SAME embedding head as every other reference. Per spec 3.2
            # this brackets the encoder's information content, not the
            # dynamics. An earlier version probed the posterior LATENT here,
            # which is not a lower bound on anything -- it carries per-step
            # sampling noise, and measured on real data it exceeded persistence
            # at 8 of 10 horizon steps, inverting gap_closed's denominator.
            real = model.rssm.observe(
                embeddings[:, context:], actions[:, context:], state=state
            )
            floor_embeddings = model.heads(real["latent"])["embedding"][0].cpu().numpy()

            # Truth for imagined step k is the frame `actions[context + k]`
            # PRODUCED, which is one later than the frame the action was taken
            # from. Tracing the indices: `embeddings[k]` is frame
            # `start + 1 + k`, so the last observed step is frame
            # `start + context`; `imagine` then applies `actions[context + k]`
            # to reach frame `start + context + 1 + k`. The slice therefore
            # starts at `start + context + 1`, not `start + context`.
            #
            # Verified empirically, not by argument: with an oracle model whose
            # encoder tags each frame with its index and whose dynamics advance
            # that tag by exactly one per action, `start + context` gives a
            # PERFECT predictor a constant error of exactly one step of true
            # displacement at every horizon step -- and makes the encoder floor,
            # which sees the real future frame, pay the same penalty, while
            # persistence scores exactly 0.0 one step out, i.e. "the agent never
            # moved" claimed as perfect. See
            # test_a_perfect_predictor_scores_zero_only_on_the_frame_the_action_produced.
            #
            # Off by one here would shift every reported position error without
            # failing any shape or smoke test.
            truth = probe_targets(
                episode.privileged[start + context + 1 : start + need + 1],
                episode.privileged_keys,
            )
            # Persistence = the last context step's PREDICTED embedding, held.
            # Predicted, not raw, so it shares the probe's distribution.
            last_context_embedding = (
                model.heads(observed["latent"][:, -1:])["embedding"][0, 0].cpu().numpy()
            )

            # ALL THREE references go through the IDENTICAL pipeline
            # (encode -> RSSM -> emb_head) and are probed with the SAME probe,
            # which is fit on the model's OWN predicted embeddings. They differ
            # only in what information produced the latent, which is the only
            # thing the band should measure.
            #
            # Fitting on real encoder embeddings and applying to predicted ones
            # is a distribution mismatch, and it was destroying the signal:
            # band below 2 SE at 18 of 45 horizon steps, against 2 of 45 once
            # every reference shares the pipeline. gap_closed at horizon 45
            # moved from -7.26 to -0.78 on the same checkpoint.
            #
            # An earlier version probed the model and persistence in 1536-d
            # latent space and the floor in 2048-d embedding space. The band
            # then spanned two differently-fit probes, so "dynamics helped" was
            # confounded with "one probe fits better" -- the same defect as M2's
            # feature-scale confound, and the one spec 2 exists to prevent.
            #
            # The RSSM's prediction of the future observation IS its embedding
            # head's output, so using it is not a handicap: it is the model's
            # actual claim about what it will see.
            predicted_embeddings = model.heads(imagined["latent"])["embedding"]
            model_pred = apply_probe(
                embedding_probe_weights, predicted_embeddings[0].cpu().numpy()
            )
            floor_pred = apply_probe(embedding_probe_weights, floor_embeddings)
            pers_pred = apply_probe(
                embedding_probe_weights,
                np.repeat(last_context_embedding[None, :], horizon, axis=0),
            )

            rssm_pos.append(position_error(model_pred, truth))
            floor_pos.append(position_error(floor_pred, truth))
            pers_pos.append(position_error(pers_pred, truth))
            rssm_ang.append(angle_error_degrees(model_pred, truth))
            floor_ang.append(angle_error_degrees(floor_pred, truth))
            pers_ang.append(angle_error_degrees(pers_pred, truth))

    if not rssm_pos:
        raise ValueError(
            f"no validation window reached {need + 1} frames; "
            "lower --context/--horizon or check the split"
        )

    stack = lambda xs: np.stack(xs).mean(axis=0)  # noqa: E731
    return RolloutResult(
        horizon=np.arange(1, horizon + 1),
        rssm_position=stack(rssm_pos),
        persistence_position=stack(pers_pos),
        floor_position=stack(floor_pos),
        rssm_angle=stack(rssm_ang),
        persistence_angle=stack(pers_ang),
        floor_angle=stack(floor_ang),
    )


def source_for(model, path, episode, feature_backbone):
    """Pixels for the pixel arm, the arm's own cached features otherwise."""
    if model.input_kind == "obs":
        return episode.obs
    from mbfps.data.loader import feature_suffix

    return np.load(path.with_suffix(feature_suffix(feature_backbone)))
