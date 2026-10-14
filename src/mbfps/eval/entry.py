"""Stage A's entry check (M4 spec §4.5): does imagination predict reward and
episode end over 15 steps?

The start states are the ones the actor-critic will use: EVERY posterior state
of an episode. From state `s_i` (after action `i`, having seen `obs[i+1]`) the
model imagines up to 15 steps driven by the RECORDED actions
`actions[i+1 : i+1+H_i]`; imagined step `j` predicts `rewards[i+1+j]` and the
continue target of that same transition. All starts of an episode go through
ONE `imagine` call of HORIZON steps with the actions zero-padded past the
episode's end; the steps past the end are dropped by `valid`, and imagining
them cannot change an earlier step.

Two readings, each an episode-clustered bootstrap interval:
  return    R^2 of the imagined H_i-step return against the real one, against
            the best constant known from TRAINING data (c_R).
  continue  Brier skill of the continue head against the training-set
            continue rate (c_C).
The same readings on the POSTERIOR states (teacher-forced) attribute a failure:
the head/data, or imagination. They are reported, not gated.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mbfps.eval.pooling import episode_bootstrap, percentile_interval
from mbfps.models.heads import continue_target
from mbfps.models.transforms import read_reward

HORIZON: int = 15
RESAMPLES: int = 2000
VERDICTS: tuple[str, ...] = ("ENTRY_PASS", "HEAD_OR_DATA", "IMAGINATION")


@dataclass
class EpisodeArrays:
    imagined_reward: np.ndarray
    imagined_continue: np.ndarray
    posterior_reward: np.ndarray
    posterior_continue: np.ndarray
    real_reward: np.ndarray
    real_continue: np.ndarray
    valid: np.ndarray


def real_arrays(rewards, terminated, truncated, horizon: int = HORIZON):
    """`(real_reward, real_continue, valid)`, each `(T - 1, horizon)`: slot
    `(i, j)` is transition `i + 1 + j`, valid while that index is `< T`."""
    rewards = np.asarray(rewards, dtype=np.float64)
    cont = continue_target(
        torch.as_tensor(np.asarray(terminated)), torch.as_tensor(np.asarray(truncated))
    ).numpy().astype(np.float64)
    t = rewards.shape[0]
    index = np.arange(t - 1)[:, None] + 1 + np.arange(horizon)[None, :]
    valid = index < t
    safe = np.where(valid, index, 0)
    return (
        np.where(valid, rewards[safe], 0.0),
        np.where(valid, cont[safe], 0.0),
        valid,
    )


@torch.no_grad()
def episode_arrays(
    model, source, actions, rewards, terminated, truncated, device, seed: int,
    horizon: int = HORIZON,
) -> EpisodeArrays:
    """Imagined, posterior and real arrays for one episode.

    `source` is the episode's `(T+1, ...)` model input -- pixels for the `cnn`
    kind, the arm's cached feature rows otherwise (`rollout.source_for`).
    Seeded with `seed` immediately before the filter, so an episode's arrays do
    not depend on which episodes were read before it."""
    t = len(actions)
    torch.manual_seed(int(seed))
    src = torch.as_tensor(np.asarray(source)).to(device)
    embeddings = model.encoder(src).unsqueeze(0)[:, 1:]          # (1, T, E)
    acts = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=device)
    post = model.rssm.observe(embeddings, acts.unsqueeze(0))
    h, z = post["h"][0, : t - 1], post["z"][0, : t - 1]          # starts 0..T-2

    index = torch.arange(t - 1, device=device)[:, None] + 1 + torch.arange(horizon, device=device)
    valid_t = index < t
    plan = torch.where(valid_t, acts[torch.where(valid_t, index, 0)], torch.zeros_like(index))
    imagined = model.rssm.imagine(plan, (h, z))
    transform = model.cfg.train.reward_transform
    heads_im = model.heads(imagined["latent"])
    heads_post = model.heads(post["latent"][0])                   # (T, D) -> per transition
    post_index = torch.where(valid_t, index, 0)

    real_r, real_c, valid = real_arrays(rewards, terminated, truncated, horizon)
    # CPU first, then float64: MPS has no float64 to cast to.
    as_np = lambda x: x.detach().cpu().double().numpy()
    return EpisodeArrays(
        imagined_reward=as_np(read_reward(heads_im["reward"], transform)),
        imagined_continue=as_np(torch.sigmoid(heads_im["continue_logit"])),
        posterior_reward=as_np(read_reward(heads_post["reward"][post_index], transform)),
        posterior_continue=as_np(torch.sigmoid(heads_post["continue_logit"][post_index])),
        real_reward=real_r,
        real_continue=real_c,
        valid=valid,
    )


def constant_baselines(episodes, horizon: int = HORIZON) -> tuple[float, float]:
    """`(c_R, c_C)` from TRAINING episodes' real data only, weighted exactly as
    the readings weight validation rows: one return per start, one continue
    target per valid slot."""
    returns, conts = [], []
    for rewards, terminated, truncated in episodes:
        real_r, real_c, valid = real_arrays(rewards, terminated, truncated, horizon)
        returns.append((real_r * valid).sum(1))
        conts.append(real_c[valid])
    return float(np.concatenate(returns).mean()), float(np.concatenate(conts).mean())


def _interval(stat, n_rows: int, labels: np.ndarray, resamples: int, seed: int) -> dict:
    point = stat(np.arange(n_rows))
    reps = np.array([stat(rows) for rows in episode_bootstrap(labels, resamples, seed)])
    low, high, _ = percentile_interval(reps)
    return {"point": float(point), "ci_low": low, "ci_high": high,
            "n_rows": int(n_rows), "n_episodes": int(np.unique(labels).size)}


def return_reading(pred, real, valid, labels, c_R: float, resamples: int, seed: int) -> dict:
    """R^2 of per-start returns. `labels` is one episode label per start."""
    pred_ret = (np.asarray(pred) * valid).sum(1)
    real_ret = (np.asarray(real) * valid).sum(1)

    def stat(rows):
        sse = np.sum((real_ret[rows] - pred_ret[rows]) ** 2)
        sst = np.sum((real_ret[rows] - c_R) ** 2)
        return 1.0 - sse / sst if sst > 0 else float("nan")

    return _interval(stat, real_ret.size, np.asarray(labels), resamples, seed)


def continue_reading(p, c, valid, labels, c_C: float, resamples: int, seed: int) -> dict:
    """Brier skill over valid slots. `labels` is one episode label per START;
    it is expanded to the slots here."""
    valid = np.asarray(valid, dtype=bool)
    slot_labels = np.repeat(np.asarray(labels), valid.sum(1))
    p_rows, c_rows = np.asarray(p)[valid], np.asarray(c)[valid]

    def stat(rows):
        model = np.mean((p_rows[rows] - c_rows[rows]) ** 2)
        const = np.mean((c_C - c_rows[rows]) ** 2)
        return 1.0 - model / const if const > 0 else float("nan")

    return _interval(stat, c_rows.size, slot_labels, resamples, seed)


def _passes(interval: dict) -> bool:
    return bool(interval.get("ci_low", float("nan")) > 0.0)


def entry_verdict(imagined: dict, posterior: dict) -> str:
    """§4.7. A reading whose interval is NaN fails, by the `>`."""
    failing = [k for k in ("return", "continue") if not _passes(imagined[k])]
    if not failing:
        return "ENTRY_PASS"
    if any(not _passes(posterior[k]) for k in failing):
        return "HEAD_OR_DATA"
    return "IMAGINATION"


def control_broken(control_return: dict) -> bool:
    """§4.6: the M3 model's return reading passing means the check is broken."""
    return _passes(control_return)
