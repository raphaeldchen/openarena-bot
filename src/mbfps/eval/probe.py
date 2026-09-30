"""Linear probe from latent state into privileged space -- EVALUATION ONLY.

This is the only module that reads `privileged_state`. It is fit post-hoc on
frozen latents with no gradient path to the world model, on held-out episodes.

Deliberately linear: a nonlinear probe measures the probe's capacity as much as
the representation's content, and the question here is what the latent already
encodes.

Two properties of `my_way_home`, measured rather than assumed:

- `health` and `pos_z` have exactly one unique value across the dataset, so R^2
  on them is 0/0. They are excluded; see PROBE_KEYS.
- `angle` is in degrees and wraps, with 191 wrap events in 19,463 steps. It is
  represented as (sin, cos) so that 359 deg and 1 deg are near each other, and
  errors are recovered with atan2 -- a linear probe's raw (sin, cos) output does
  not lie on the unit circle, so arcsin would be wrong.

Positions are Doom map units, not metres.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mbfps.data.episode import load_episode
from mbfps.eval.windows import window_starts

PROBE_KEYS: tuple[str, ...] = ("pos_x", "pos_y", "angle")
"""Privileged channels that actually vary in this scenario."""

TARGET_DIM = 4
"""pos_x, pos_y, sin(angle), cos(angle)."""


def probe_targets(privileged: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
    """Build `(N, 4)` targets from a `(N, K)` privileged array."""
    index = {k: i for i, k in enumerate(keys)}
    missing = [k for k in PROBE_KEYS if k not in index]
    if missing:
        raise KeyError(f"privileged_keys lacks {missing}; got {keys}")
    radians = np.deg2rad(privileged[:, index["angle"]].astype(np.float64))
    return np.stack(
        [
            privileged[:, index["pos_x"]].astype(np.float64),
            privileged[:, index["pos_y"]].astype(np.float64),
            np.sin(radians),
            np.cos(radians),
        ],
        axis=1,
    )


RIDGES: tuple[float, ...] = (1e-1, 1e1, 1e3, 1e5, 1e7)
"""Candidate strengths. The measured optimum here is 1e3-1e5, not 1."""


def _solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0  # never regularise the intercept
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def fit_probe(latents, targets, val_latents=None, val_targets=None, ridge=None) -> dict:
    """Standardise the inputs, then SELECT the ridge on held-out data.

    Returns a dict carrying the weights and the standardisation, because both
    are needed to apply it. The intercept is never regularised: penalising it
    biases predictions toward zero, and Doom coordinates are nowhere near
    zero-centred.

    Selection is not optional here. A fixed `ridge=1.0` on unstandardised inputs
    underfits badly -- the targets have std ~240 in map units while the features
    are order 1 -- and measured, that one default cost held-out R^2 0.16 against
    a ceiling of 0.42. Without validation data the fallback is 1e3, the middle
    of the measured optimum, not 1.
    """
    x = np.asarray(latents, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / scale
    if ridge is not None:           # explicit override, for tests that pin a value
        return {"w": _solve(xs, y, ridge), "mean": mean, "scale": scale, "ridge": ridge}
    if val_latents is None:
        return {"w": _solve(xs, y, 1e3), "mean": mean, "scale": scale, "ridge": 1e3}

    xv = (np.asarray(val_latents, dtype=np.float64) - mean) / scale
    best = None
    for ridge in RIDGES:
        w = _solve(xs, y, ridge)
        pred = np.concatenate([xv, np.ones((xv.shape[0], 1))], axis=1) @ w
        score = _mean_r2(pred, val_targets)
        if best is None or score > best[0]:
            best = (score, ridge, w)
    return {"w": best[2], "mean": mean, "scale": scale, "ridge": best[1], "r2": best[0]}


def apply_probe(probe: dict, latents: np.ndarray) -> np.ndarray:
    xs = (np.asarray(latents, dtype=np.float64) - probe["mean"]) / probe["scale"]
    return np.concatenate([xs, np.ones((xs.shape[0], 1))], axis=1) @ probe["w"]


def _mean_r2(predicted: np.ndarray, targets: np.ndarray) -> float:
    """R^2 averaged PER COLUMN.

    Pooling would let `pos_x` (std ~240) drown `sin(angle)` (std ~0.7), hiding a
    completely useless angle prediction behind good position numbers.
    """
    scores = []
    for c in range(targets.shape[1]):
        truth = targets[:, c]
        denom = float(((truth - truth.mean()) ** 2).sum())
        if denom == 0.0:
            continue
        scores.append(1.0 - float(((truth - predicted[:, c]) ** 2).sum()) / denom)
    if not scores:
        raise ValueError("every target column has zero variance; nothing to score")
    return float(np.mean(scores))


def position_error(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Euclidean distance in Doom map units."""
    return np.linalg.norm(predicted[:, :2] - true[:, :2], axis=1)


def angle_error_degrees(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Absolute angular error in [0, 180].

    `atan2` recovers the angle from an arbitrary (sin, cos) pair, which matters
    because a linear probe's output does not lie on the unit circle.
    """
    pred = np.arctan2(predicted[:, 2], predicted[:, 3])
    real = np.arctan2(true[:, 2], true[:, 3])
    difference = np.abs(np.rad2deg(np.arctan2(np.sin(pred - real), np.cos(pred - real))))
    return difference


def probe_r2(probe: dict, latents: np.ndarray, targets: np.ndarray) -> float:
    """Mean R^2 across the four target columns.

    Averaged per column rather than pooled: `pos_x` has std ~253 while
    `sin(angle)` has std ~0.7, so a pooled variance would be dominated by
    position and a completely useless angle prediction would not show up.

    A column with zero variance contributes nothing rather than a NaN -- which
    is why PROBE_KEYS excludes `health` and `pos_z`, but the guard stays in case
    a validation slice happens to be degenerate.
    """
    return _mean_r2(apply_probe(probe, latents), targets)


def _sampling_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """The distribution `RSSM._sample` actually draws from, as probabilities.

    The temperature matters because the information content of the code is a
    property of the distribution `z` came from, not of the raw logits. Written
    as a guarded division rather than an unconditional one so the shipped
    `sample_temperature = 1.0` path is bitwise the plain softmax, exactly as
    `RSSM._sample` does it -- M3h added the temperature and took the same care,
    and an unconditional `logits / 1.0` is not guaranteed bitwise identical.

    Mirrors `_sample` at BOTH of its special cases, not just 1.0: it skips the
    division at 0.0 as well, and at 0.0 it takes the argmax, so the distribution
    it effectively draws from there is a point mass rather than the softmax.
    Unreachable at the shipped 1.0, and written out because a future
    temperature change should be reflected rather than silently mis-reported.
    """
    if temperature < 0.0:
        raise ValueError(f"sampling temperature must be >= 0, got {temperature}")
    if temperature == 0.0:
        # `_sample` draws from the untempered softmax at 0.0 and then DISCARDS
        # the draw for the argmax, so the distribution it effectively samples
        # from is a point mass -- which is what this has to report, not the
        # softmax it happened to compute. Dividing here instead would give NaN,
        # and the estimator's guard would then refuse it with the wrong
        # diagnosis ("logits handed in place of probabilities").
        onehot = torch.zeros_like(logits)
        onehot.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return onehot
    if temperature != 1.0:
        logits = logits / temperature
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def gather_probe_data(
    model,
    paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = 20,
    seed: int = 0,
) -> dict:
    """Collect probe data under the ROLLOUT's own window protocol.

    Every window is `context` real frames filtered from a ZERO state followed
    by `horizon` more, cut from the episode on the same stride
    `evaluate_rollout` uses -- so no latent here ever carries more filtering
    history than a latent the probe is later applied to.

    Why this is not cosmetic. Filtering each episode WHOLE, as this used to,
    gives the deterministic state `h` ~500 steps of history that the rollout
    never has: the rollout hands it `context` frames out of a zero state. The
    probe was then fit on one distribution and applied to another. Measured on
    the shipped 20k checkpoint, same frames and same probe: position error
    222.4 under whole-episode filtering against 247.5 under the rollout's
    5-step context.

    And the bias does not cancel across the band. Persistence is frozen at the
    last context step forever while the floor's own context grows to
    `context + horizon`, and measured probe error by context length is
    317.3 / 223.4 / 213.5 / 215.3 at contexts 1 / 5 / 20 / 50 -- so the floor
    alone gains roughly 8 units from filtering history, against a median band
    width of 55. That is a real bias inside `gap_closed`'s denominator. It is
    the same defect class as the real-versus-predicted embedding confound this
    module already guards, on the context-length axis.

    The future steps are run through `observe` (the posterior on the real
    frames, warm-started from the context state), not `imagine`. Fitting on
    IMAGINED latents would tune the readout to the model's own dynamics error
    and hand the model arm a probe the floor arm never gets -- reintroducing
    the cross-arm asymmetry that the single-shared-probe rule exists to
    prevent. What is matched here is the filtering DEPTH, which is what the
    three references actually share.

    Returns nine row-aligned arrays:

    - `"latent"` `(N, LATENT)` -- the posterior latent.
    - `"embedding"` `(N, EMBED)` -- the model's PREDICTED embedding, i.e.
      `heads(latent)["embedding"]`.
    - `"encoder_embedding"` `(N, ENC)` -- the RAW encoder output for the same
      frames, before the RSSM and before the head.
    - `"targets"` `(N, 4)`.
    - `"window"` `(N,)` int -- a 0-based index identifying which window each
      row came from.
    - `"step"` `(N,)` int -- the 0-based position of the row within its
      window, `0 .. context + horizon - 1`. M3i reconstructs per-window
      trajectories from these two so it can compute displacement WITHIN a
      window; every earlier caller ignores them.
    - `"episode"` `(N,)` int -- the 0-based index of the episode each row came
      from, over the episodes that contributed at least one window. M3j
      resamples on this rather than on `"window"`; see `_block_bootstrap_ci`'s
      `groups`.
    - `"post_probs"`, `"prior_probs"` `(N, z_cats, z_classes)` float32 -- the
      posterior (the distribution `"latent"`'s `z` was drawn from) and the
      prior, each as probabilities under `RSSM._sample`'s own temperature,
      NOT flattened. M3l measures the information content of the code, which
      is a property of the distribution rather than of the one-hot draw in
      `"latent"`. Collected from the very `observe` calls `"latent"` comes
      from, so they add no operation and consume no random draw.

    THE TWO EMBEDDINGS ARE NOT REDUNDANT AND MUST NOT BE COLLAPSED INTO ONE.
    They answer different questions, and each is the wrong array for the
    other's:

    - The rollout band scores model, persistence and floor *after* the
      embedding head, so `fit_probes` fits the band's probe on `"embedding"`.
      Fitting that probe on the raw encoder output instead is a distribution
      mismatch, measured destroying the signal: band below 2 SE at 18 of 45
      horizon steps against 2 of 45, and `gap_closed` at horizon 45 moving
      -7.26 -> -0.78 on one unchanged checkpoint.
    - The filtering comparison (spec section 3.4, gate criterion section 4.4)
      asks whether the posterior latent adds anything over what the CURRENT
      FRAME alone provides -- and the only thing it can add is history. Its
      reference therefore has to be `"encoder_embedding"`, the frame's own
      encoding. Comparing the latent against the model's *predicted* embedding
      would instead ask whether the latent beats its own head's
      reconstruction, which a latent can win while carrying no history at all,
      so the gate criterion would be satisfiable without the property it
      exists to certify. See `filtering_report`.

    Seeds the global RNG, because the posterior samples and reproducibility
    must come from the seed rather than from taking the categorical mode.
    """
    from mbfps.eval.rollout import source_for

    if context < 1:
        raise ValueError(f"context must be at least 1 real frame, got {context}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 step, got {horizon}")

    torch.manual_seed(seed)
    need = context + horizon
    latents, embeddings, encoder_embeddings, targets = [], [], [], []
    post_probs, prior_probs = [], []
    windows: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    window_index = 0
    episode_index = 0

    for path in list(paths)[:limit]:
        episode = load_episode(path)
        # The rollout's own window rule, from the one place that owns it -- the
        # length guard, the stride and the `+ 1` on its stop. What is shared is
        # the per-episode window RULE, not the episode SET: `limit` above caps
        # how many episodes the probe is fit on, and `evaluate_rollout` scores
        # every validation path.
        starts = window_starts(episode.length, context, horizon)
        if not starts:
            continue
        source = source_for(model, path, episode, backbone)
        all_actions = (
            torch.as_tensor(episode.actions.astype(np.int64)).unsqueeze(0).to(device)
        )
        for start in starts:
            frames = torch.as_tensor(source[start : start + need + 1]).to(device)
            # Drop the FIRST frame: `embeddings[k]` must be the frame
            # `actions[k]` led to, per RSSM.observe's action-time convention.
            window = model.encoder(frames).unsqueeze(0)[:, 1:]
            actions = all_actions[:, start : start + need]

            observed = model.rssm.observe(window[:, :context], actions[:, :context])
            state = (observed["h"][:, -1], observed["z"][:, -1])
            future = model.rssm.observe(
                window[:, context:], actions[:, context:], state=state
            )
            latent = torch.cat([observed["latent"], future["latent"]], dim=1)
            # Row-aligned with `latent` by construction: the same two `observe`
            # calls, concatenated on the same axis in the same order. Deriving
            # these from a second pass would reintroduce the row-alignment
            # hazard M3k needed two layered guards for.
            temperature = model.rssm.cfg.sample_temperature
            for key, sink in (("post_logits", post_probs),
                              ("prior_logits", prior_probs)):
                joined = torch.cat([observed[key], future[key]], dim=1)
                sink.append(
                    _sampling_probs(joined, temperature)[0].float().cpu().numpy()
                )

            latents.append(latent[0].float().cpu().numpy())
            embeddings.append(
                model.heads(latent)["embedding"][0].float().cpu().numpy()
            )
            # The same `window` rows the latents were filtered from, so the
            # raw encoder embedding is aligned with the latent frame for frame.
            encoder_embeddings.append(window[0].float().cpu().numpy())
            # `latent[k]` describes frame `start + 1 + k`, so the target starts
            # at `start + 1`. Losing that `+1` shifts every window by one frame
            # and still fits perfectly, because both sides stay linear in the
            # frame index -- only the recovered values can catch it.
            targets.append(
                probe_targets(
                    episode.privileged[start + 1 : start + need + 1],
                    episode.privileged_keys,
                )
            )
            rows = int(latent.shape[1])
            windows.append(np.full(rows, window_index, dtype=np.int64))
            steps.append(np.arange(rows, dtype=np.int64))
            episodes.append(np.full(rows, episode_index, dtype=np.int64))
            window_index += 1
        # Advanced only here, AFTER the window loop, so an episode that reached
        # `continue` above (too short for one window) never consumes a label and
        # the labels stay gap-free.
        episode_index += 1

    if not latents:
        raise ValueError(
            f"no probe window reached {need + 1} frames; lower context/horizon "
            "or check the episode paths"
        )
    return {
        "latent": np.concatenate(latents),
        "embedding": np.concatenate(embeddings),
        "encoder_embedding": np.concatenate(encoder_embeddings),
        "targets": np.concatenate(targets),
        # Row-aligned with the four arrays above: which window each row came
        # from and its order within that window. M3i reconstructs per-window
        # trajectories from these; every earlier caller ignores them.
        "window": np.concatenate(windows),
        "step": np.concatenate(steps),
        # Which EPISODE each row came from, 0-based over the episodes that
        # actually contributed a window. The coarsest correlated unit the scored
        # array contains: windows are cut non-overlapping, but several windows
        # from one trajectory are not independent observations, and every
        # reading from M3e onward clusters on episodes rather than windows.
        "episode": np.concatenate(episodes),
        # M3l. `(N, z_cats, z_classes)`, NOT flattened: the estimator sums an
        # entropy per categorical and a flattened array cannot tell the groups
        # apart. Additive -- `observe` already computed both, so collecting
        # them consumes no randomness, which a test pins.
        "post_probs": np.concatenate(post_probs),
        "prior_probs": np.concatenate(prior_probs),
    }


PROBE_EPISODE_LIMIT: int = 20
"""How many TRAINING episodes the rollout's probe is fit on: the first
`PROBE_EPISODE_LIMIT` of the train split in the order `episode_split` returns
it, `select_episodes` of which are held back to select the ridge.

Named because a second consumer now depends on the SAME rule. The split-gap
diagnostic (`scripts/split_gap.py`) evaluates the shipped checkpoints on the
training episodes the probe never saw -- model-seen, probe-unseen -- and
which those are is exactly `probe_episodes(train)[1]`. A bare `20` here and
a bare `20` there would be two rules that agree today; `fit_probes` and the
diagnostic both call `probe_episodes`, and a test pins `fit_probes`'s
default to this constant, so moving it moves both or fails the suite.
"""


def probe_episodes(paths, limit: int = PROBE_EPISODE_LIMIT) -> tuple[list, list]:
    """`(used, held)`: the episodes the probe is fit on and every other one,
    both in the caller's order. `used` is the LEADING block, so a caller that
    hands the train split in `episode_split`'s order gets a deterministic set.
    `held` is empty when `paths` has at most `limit` entries -- legal for the
    probe, and what the split-gap diagnostic refuses as an empty stratum."""
    if limit < 1:
        raise ValueError(f"the probe needs at least one episode; limit={limit}")
    ordered = list(paths)
    return ordered[:limit], ordered[limit:]


@torch.no_grad()
def fit_probes(
    model,
    paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = PROBE_EPISODE_LIMIT,
    seed: int = 0,
    select_episodes: int = 4,
    ridge: float | None = None,
) -> tuple[dict, dict]:
    """Fit both probes on TRAINING episodes, under the rollout's protocol.

    Returns `(latent_probe, embedding_probe)`.

    Which one the rollout uses, and why there are still two:

    - `embedding_probe` is THE probe `evaluate_rollout` takes. It is fit on the
      model's OWN PREDICTED embeddings -- `heads(latent)["embedding"]` -- not on
      the raw encoder output, because all three rollout references (model,
      persistence, floor) are scored after passing through that same head. A
      probe fit on real encoder embeddings and applied to predicted ones is a
      distribution mismatch, and it measurably destroys the signal: band below
      2 SE at 18 of 45 horizon steps against 2 of 45, and gap_closed at horizon
      45 moving from -7.26 to -0.78 on one unchanged checkpoint. See the module
      docstring of `mbfps.eval.rollout`.
    - `latent_probe` is a DIAGNOSTIC on the 1536-d posterior latent, reported
      alongside. It is deliberately NOT a second space for the band: an earlier
      design probed model and persistence in latent space and the floor in
      embedding space, so the band spanned two differently-fit probes and
      "dynamics helped" was confounded with "one probe fits better".

    `context` and `horizon` must be the rollout's own, and are forwarded to
    `gather_probe_data` -- see its docstring for the measured cost of fitting
    at a different filtering depth from the one the probe is applied at.

    Sampling in `observe` is stochastic, so the gathering seeds the global RNG:
    without it two identical evaluation runs fit two different probes and every
    downstream number moves. `evaluate_rollout` re-seeds independently, so how
    much RNG is drawn here does not perturb the rollout.

    Args:
        model: a `WorldModel` (or anything duck-typed like one).
        paths: TRAINING episode paths. Passing validation paths would leak.
        backbone: cached-feature backbone name, or None for the pixel arm.
        device: where to run the encoder and RSSM.
        context: real frames filtered from a zero state, per window.
        horizon: steps after the context, per window.
        limit: how many episodes to fit on -- `probe_episodes(paths, limit)[0]`, the rule the split-gap strata share.
        seed: fixes the posterior samples, and so the probe.
        select_episodes: how many of the used episodes are held back to SELECT
            the ridge rather than fit it. Selection is not optional -- a fixed
            ridge on unstandardised inputs cost held-out R^2 0.16 against a
            ceiling of 0.42 -- but it needs data the weights did not see. Zero,
            or too few episodes to spare, falls back to `fit_probe`'s default.
        ridge: pin the penalty and skip selection entirely (tests).
    """
    used, _ = probe_episodes(paths, limit)
    if not used:
        raise ValueError("no episodes to fit the probe on; `paths` was empty")

    gather = lambda ps, s: gather_probe_data(  # noqa: E731
        model, ps, backbone, device, context, horizon, limit=len(ps), seed=s
    )

    # Split at EPISODE granularity, like the train/val split itself. Splitting
    # rows would put frames from one episode on both sides, and consecutive
    # frames are near-duplicates, so the selection set would not be held out in
    # any meaningful sense and every ridge would look equally good.
    spare = len(used) - select_episodes
    if ridge is not None or select_episodes <= 0 or spare < 1:
        train = gather(used, seed)
        return (
            fit_probe(train["latent"], train["targets"], ridge=ridge),
            fit_probe(train["embedding"], train["targets"], ridge=ridge),
        )

    train = gather(used[:spare], seed)
    select = gather(used[spare:], seed + 1)
    return (
        fit_probe(train["latent"], train["targets"],
                  select["latent"], select["targets"]),
        fit_probe(train["embedding"], train["targets"],
                  select["embedding"], select["targets"]),
    )


def filtering_comparison(
    latent_train: np.ndarray,
    embedding_train: np.ndarray,
    latent_val: np.ndarray,
    embedding_val: np.ndarray,
    targets_train: np.ndarray,
    targets_val: np.ndarray,
) -> dict:
    """Does the posterior latent beat the raw embedding at the SAME timestep?

    The posterior has already seen frame t, so it cannot add information about t
    over the embedding of t. What it can add is history, carried in the
    deterministic state `h`. If it does not win here, `h` is inert.

    `embedding_train`/`embedding_val` must be the RAW ENCODER embedding of frame
    t -- `gather_probe_data`'s `"encoder_embedding"`, not its `"embedding"`.
    The data-processing argument above is what makes a win here mean "history",
    and it only holds against the frame's own encoding. See `filtering_report`.

    BOTH REPORTED R^2 ARE OPTIMISTICALLY BIASED AND MUST NOT BE QUOTED AS CLEAN
    HELD-OUT NUMBERS. The ridge is selected on the very rows the probe is then
    scored on, so each value is a max over the five `RIDGES` on the scored
    split, not a single held-out score. The bias is symmetric -- both probes get
    the same five-way maximum on the same rows -- so the COMPARISON, which is
    all gate criterion 4 asks for, stays fair. The levels do not. `filtering_gain`
    selects on a third split and is the function to quote a level from.
    """
    # Ridge SELECTED on the validation split for each probe independently --
    # one probe's optimum is not the other's, and forcing a shared value would
    # handicap whichever space suits it worse, which is the comparison itself.
    latent_weights = fit_probe(latent_train, targets_train, latent_val, targets_val)
    embedding_weights = fit_probe(embedding_train, targets_train, embedding_val, targets_val)
    latent_r2 = probe_r2(latent_weights, latent_val, targets_val)
    embedding_r2 = probe_r2(embedding_weights, embedding_val, targets_val)
    return {
        "latent_r2": latent_r2,
        "embedding_r2": embedding_r2,
        "latent_beats_embedding": bool(latent_r2 > embedding_r2),
    }


def filtering_report(
    model,
    train_paths,
    val_paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = 20,
    seed: int = 0,
) -> dict:
    """Spec section 4, gate criterion 4: does the deterministic state carry history?

    Returns `{"latent_r2", "embedding_r2", "latent_beats_embedding"}`. Nothing
    else in the harness produces this criterion, so if this is not called it
    cannot be reported at all.

    The reference is the RAW ENCODER embedding, `gather_probe_data`'s
    `"encoder_embedding"` -- spec section 3.4: "compared against a probe on the
    encoder embedding at t". That choice is the whole argument, not a detail.
    The posterior has already seen frame t, so by the data-processing
    inequality it cannot add information about t over the ENCODING of t; the
    only thing it can add is memory of earlier frames. Against the model's own
    PREDICTED embedding (`"embedding"`, what the rollout band is scored in) the
    question becomes "does the latent beat its own head's reconstruction of
    itself", which a latent can win with `h` completely inert -- the head is
    lossy and the latent is its input. A win there would prove nothing about
    history, so the gate criterion would be satisfiable without the property it
    exists to certify. `gather_probe_data` returns both arrays for exactly this
    reason; see its docstring before merging them.

    Train and validation windows are gathered SEPARATELY, and under the
    rollout's own context protocol, so neither probe is fit on rows it is
    scored on. Measured on the leaking version, the same comparison ran from
    R^2 -0.246 to 0.9997 -- with more features than validation rows a probe
    memorises the split it is scored on and the criterion becomes vacuous.

    The two gathers get different seeds so the posterior samples on the
    validation split are independent draws rather than a replay of the training
    split's; `seed` is still what fixes both, so the report is reproducible.
    """
    # Keyword-passed on purpose: `context` and `horizon` are adjacent ints of
    # the same type, so a positional call would survive them being swapped.
    train = gather_probe_data(model, train_paths, backbone, device,
                              context=context, horizon=horizon,
                              limit=limit, seed=seed)
    val = gather_probe_data(model, val_paths, backbone, device,
                            context=context, horizon=horizon,
                            limit=limit, seed=seed + 1)
    return filtering_comparison(
        train["latent"], train["encoder_embedding"],
        val["latent"], val["encoder_embedding"],
        train["targets"], val["targets"],
    )


def _require_whole_windows(n_rows: int, window: int) -> None:
    """Both `_block_bootstrap_ci`'s positional branch and `_gain_from_splits`
    need `n_rows` to be a whole number of `window`-row blocks -- a labelled
    call reaches neither check, since it blocks by `groups` instead. Shared so
    the two messages cannot drift apart silently; existing tests pin both
    wordings through both call sites, so neither string may change here."""
    if window < 1:
        raise ValueError(f"window must be at least 1 row, got {window}")
    if n_rows % window:
        raise ValueError(
            f"{n_rows} scored rows is not a whole number of {window}-row windows; "
            "the block bootstrap would mix parts of two windows into one block"
        )


def _block_bootstrap_ci(
    joint_predicted: np.ndarray,
    embedding_predicted: np.ndarray,
    targets: np.ndarray,
    window: int | None = None,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile interval for the gain, resampling WHOLE WINDOWS.

    The resampling unit is the window, not the row. A window is `window`
    consecutive frames of one episode -- 50 in the production setting -- and
    consecutive Doom frames are near-duplicates, so a row-level bootstrap counts
    ~50 correlated observations as 50 independent ones and returns an interval
    several times too narrow. The windows are exactly the blocks
    `gather_probe_data` cuts, on a stride of their own length, so they are the
    coarsest unit the scored array actually contains.

    The probes are held FIXED across resamples. This is an interval on the
    scored sample -- how much the gain would move on a different draw of
    evaluation windows -- not on the whole fit/select/score pipeline.

    TWO WAYS TO BLOCK, EXACTLY ONE PER CALL. `window` blocks POSITIONALLY, in
    fixed strides, which is what `filtering_gain` has always reported through
    and is byte-for-byte unchanged. `groups` blocks by LABEL: rows sharing a
    label travel together. Two callers need the label form and the positional
    form cannot express either. Backward displacement at k drops the first k
    rows of every window, so the groups stop being equal-length and the stride
    arithmetic below would raise; and M3j clusters on the 24 EPISODES rather
    than the 229 windows, because several non-overlapping windows cut from one
    trajectory are not independent observations. Given labels that describe the
    same blocks the stride builds, the two paths agree to the last bit -- the
    same `picked` indices in the same order from the same generator -- and a
    test pins that.
    """
    if (window is None) == (groups is None):
        raise ValueError(
            "pass exactly one of `window` (positional blocks) or `groups` "
            "(labelled blocks); passing both or neither leaves the resampling "
            "unit ambiguous"
        )
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n_rows = targets.shape[0]
    if groups is None:
        _require_whole_windows(n_rows, window)
        strides = np.arange(n_rows).reshape(n_rows // window, window)
        n_blocks = strides.shape[0]

        def pick(indices: np.ndarray) -> np.ndarray:
            return strides[indices].reshape(-1)
    else:
        labels = np.asarray(groups)
        if labels.shape != (n_rows,):
            raise ValueError(
                f"groups must be one label per scored row; got {labels.shape} "
                f"for {n_rows} rows"
            )
        # Stable sort, so a group's rows keep their original relative order and
        # the labelled path reproduces the positional one row for row.
        order = np.argsort(labels, kind="stable")
        _, starts = np.unique(labels[order], return_index=True)
        parts = np.split(order, starts[1:])
        n_blocks = len(parts)

        def pick(indices: np.ndarray) -> np.ndarray:
            return np.concatenate([parts[j] for j in indices])

    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        picked = generator.integers(0, n_blocks, size=n_blocks)
        rows = pick(picked)
        draws[i] = _mean_r2(joint_predicted[rows], targets[rows]) - _mean_r2(
            embedding_predicted[rows], targets[rows]
        )
    tail = 100.0 * (1.0 - confidence) / 2.0
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail))


@dataclass(frozen=True)
class GainSplit:
    """One split's three arrays for `gain_from_blocks`, already row-selected.

    `base` goes into BOTH arms and `block` into the joint arm only, which is
    what makes the statistic an increment rather than a level. The caller
    row-selects all three together: a target defined on a subset of the rows
    (backward displacement has none for the first k rows of a window) must be
    handed the SAME subset of features, and `gain_from_blocks` checks the three
    row counts rather than trusting it.
    """

    base: np.ndarray
    block: np.ndarray
    target: np.ndarray

    def joint(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.base, dtype=np.float64),
             np.asarray(self.block, dtype=np.float64)], axis=1,
        )

    def rows(self) -> int:
        counts = {
            np.asarray(self.base).shape[0],
            np.asarray(self.block).shape[0],
            np.asarray(self.target).shape[0],
        }
        if len(counts) != 1:
            raise ValueError(
                f"base, block and target must describe the same number of rows; "
                f"got {sorted(counts)}"
            )
        return counts.pop()


def _fit_and_score(fit_x, fit_y, select_x, select_y, score_x, score_y):
    """Fit one arm on `fit`, select its ridge on `select`, score it on `score`.

    The three-split discipline for a single feature set, factored out because
    `gain_from_blocks` needs it twice (joint and base) and
    `contrast_from_blocks` needs it twice again (two joints). `select_x is
    None` takes `fit_probe`'s default penalty, which is unbiased too, just
    weaker.

    Returns `(predicted, r2, ridge)` on the scored rows.
    """
    probe = (
        fit_probe(fit_x, fit_y)
        if select_x is None
        else fit_probe(fit_x, fit_y, select_x, select_y)
    )
    predicted = apply_probe(probe, score_x)
    return predicted, _mean_r2(predicted, score_y), probe["ridge"]


def _validate_groups(groups, n: int) -> tuple[np.ndarray, int]:
    """One `groups` label per scored row, and enough units to resample.

    Shared by `gain_from_blocks` and `contrast_from_blocks` so the two
    diagnostics refuse identically-shaped inputs with identical wording.
    """
    groups_arr = np.asarray(groups)
    if groups_arr.shape != (n,):
        raise ValueError(
            f"groups must be one label per scored row; got {groups_arr.shape} "
            f"for {n} rows"
        )
    n_groups = int(np.unique(groups_arr).size)
    if n_groups < 2:
        raise ValueError(
            "a bootstrap interval needs at least two resampling units (distinct "
            f"`groups` labels); got {n_groups}"
        )
    return groups_arr, n_groups


def gain_from_blocks(
    fit: GainSplit,
    select: GainSplit | None,
    score: GainSplit,
    *,
    groups: np.ndarray,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """`R2([base (+) block] -> target) - R2([base] -> target)` on three splits.

    The general form of `_gain_from_splits`, which is now a wrapper. Three
    things are the caller's here rather than hardcoded: the second feature
    block, the target, and the resampling groups. M3j needs all three -- a
    ladder of blocks (`enc(t-k)`, `h`, `z`, `h (+) z`), two motion targets, and
    episode-level rather than window-level clustering.

    The split discipline is unchanged and is the reason this is not
    `filtering_comparison`: weights from `fit`, ridge selected on `select`,
    reported R^2 and interval from `score`. A gain is a DIFFERENCE OF LEVELS
    between feature sets of different widths, so a ridge maximum taken on the
    scored rows favours the wider one and manufactures a positive gain out of
    the selection alone. `select=None` skips selection and takes `fit_probe`'s
    default penalty for both arms -- unbiased too, just weaker, and
    `ridge_selected` says which happened.

    `n_scored_windows` is the number of distinct `groups` labels. Under the
    positional wrapper that equals `rows // window`, which is what it always
    meant; under a filtered row set it is the number of surviving groups, which
    is what the interval actually resamples.

    `groups` must carry at least two distinct labels. With one, or with zero
    scored rows, every bootstrap resample draws the same single block, so the
    percentile interval collapses to zero width -- reading as maximal
    confidence rather than "one resampling unit, no information".
    """
    n_scored = score.rows()
    groups_arr, n_groups = _validate_groups(groups, n_scored)
    fit.rows()
    joint_fit, base_fit = fit.joint(), np.asarray(fit.base, dtype=np.float64)
    joint_score, base_score = score.joint(), np.asarray(score.base, dtype=np.float64)
    target_fit = np.asarray(fit.target, dtype=np.float64)
    target_score = np.asarray(score.target, dtype=np.float64)

    select_target = (
        None if select is None else np.asarray(select.target, dtype=np.float64)
    )
    if select is not None:
        select.rows()
    joint_predicted, joint_r2, joint_ridge = _fit_and_score(
        joint_fit, target_fit,
        None if select is None else select.joint(), select_target,
        joint_score, target_score,
    )
    base_predicted, base_r2, base_ridge = _fit_and_score(
        base_fit, target_fit,
        None if select is None else np.asarray(select.base, dtype=np.float64),
        select_target,
        base_score, target_score,
    )
    low, high = _block_bootstrap_ci(
        joint_predicted, base_predicted, target_score,
        groups=groups_arr, resamples=resamples, confidence=confidence, seed=seed,
    )
    return {
        "gain": joint_r2 - base_r2,
        "joint_r2": joint_r2,
        "embedding_r2": base_r2,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "n_scored_windows": n_groups,
        "ridge_selected": select is not None,
        "joint_ridge": joint_ridge,
        "embedding_ridge": base_ridge,
    }


def contrast_from_blocks(
    a, b, *, groups: np.ndarray, resamples: int = 1000,
    confidence: float = 0.95, seed: int = 0,
) -> dict:
    """`R2([base (+) A]) - R2([base (+) B])` -- the base cancels outright.

    `a` and `b` are each a `(fit, select, score)` triple of `GainSplit`s. Both
    arms MUST carry the same base on the same rows: that is what makes this a
    contrast between the two BLOCKS rather than two unrelated levels
    subtracted, and it is why the difference of gains equals the difference of
    joint R^2 with no base term surviving.

    `_block_bootstrap_ci` already computes `r2(A) - r2(B)` per resample, so the
    interval needs no new statistic -- only the two arms' scored predictions.

    SIGNED, and the sign is the reading. Reading F is two-sided: a negative
    contrast means B's block beats A's, which is a finding about B rather than
    an absence. Swapping the arms negates the result, pinned by test.
    """
    a_fit, a_select, a_score = a
    b_fit, b_select, b_score = b
    n = a_score.rows()
    target_score = np.asarray(a_score.target, dtype=np.float64)
    b_target = np.asarray(b_score.target, dtype=np.float64)
    # `GainSplit` carries no original row indices, only already-selected
    # arrays -- so row alignment can only be verified by content. `target` is
    # the field invariant to the caller's choice of block and base, so content
    # equality is the strongest row-identity signal available given `GainSplit`
    # carries no row indices. A collision is not a realistic concern for
    # continuous position data.
    b_rows = b_score.rows()
    if b_rows != n:
        raise ValueError(
            f"both arms must be scored on the same rows; got {n} and {b_rows}"
        )
    if target_score.shape != b_target.shape or not np.array_equal(target_score, b_target):
        raise ValueError(
            f"both arms must be scored on the same rows; got same-length "
            f"targets with different contents"
        )
    a_base = np.asarray(a_score.base, dtype=np.float64)
    b_base = np.asarray(b_score.base, dtype=np.float64)
    if a_base.shape != b_base.shape or not np.array_equal(a_base, b_base):
        raise ValueError(
            "both arms must carry the same base; the contrast is only a "
            "comparison of the two BLOCKS because the base cancels"
        )
    if (a_select is None) != (b_select is None):
        raise ValueError(
            "both arms must use the same ridge-selection policy because the "
            "contrast compares two blocks under one methodology"
        )
    groups_arr, n_groups = _validate_groups(groups, n)

    def arm(fit, select, score):
        select_target = (
            None if select is None else np.asarray(select.target, dtype=np.float64)
        )
        return _fit_and_score(
            fit.joint(), np.asarray(fit.target, dtype=np.float64),
            None if select is None else select.joint(), select_target,
            score.joint(), target_score,
        )

    a_predicted, a_r2, a_ridge = arm(a_fit, a_select, a_score)
    b_predicted, b_r2, b_ridge = arm(b_fit, b_select, b_score)
    low, high = _block_bootstrap_ci(
        a_predicted, b_predicted, target_score,
        groups=groups_arr, resamples=resamples, confidence=confidence, seed=seed,
    )
    return {
        "contrast": a_r2 - b_r2,
        "a_r2": a_r2,
        "b_r2": b_r2,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "n_scored_windows": n_groups,
        "a_ridge": a_ridge,
        "b_ridge": b_ridge,
        "ridge_selected": a_select is not None,
    }


def _gain_from_splits(
    fit: dict,
    select: dict | None,
    score: dict,
    h_dim: int,
    window: int,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """The array-level half of `filtering_gain`; see that docstring for the why.

    `fit`, `select` and `score` are three DISJOINT `gather_probe_data` dicts.
    The weights come from `fit`, the ridge is selected on `select`, and the
    reported R^2 and the interval come from `score` -- so, unlike
    `filtering_comparison`, no part of the number on the scored rows was tuned
    on those rows. `select=None` skips selection entirely and takes
    `fit_probe`'s default penalty; that is unbiased too, just weaker.

    This is now a thin wrapper over `gain_from_blocks`, which took over the
    split, selection and bootstrap logic so M3j could reuse it with a different
    block and target. What is fixed HERE is the block (`latent[:, :h_dim]`, the
    deterministic head), the target (the privileged state) and the blocking
    (positional, in `window`-row strides). Its output is pinned byte-for-byte by
    a test, because `filtering_gain` reports through it onto the gate's own path.
    """
    def split(data: dict) -> GainSplit:
        embedding = np.asarray(data["encoder_embedding"], dtype=np.float64)
        latent = np.asarray(data["latent"], dtype=np.float64)
        if not 1 <= h_dim <= latent.shape[1]:
            raise ValueError(
                f"h_dim={h_dim} does not index a {latent.shape[1]}-wide latent"
            )
        # `latent` is `cat([h, z])` -- h FIRST, per RSSM.observe. Slicing the
        # tail instead reads the stochastic state and answers a different
        # question with the same shapes.
        return GainSplit(base=embedding, block=latent[:, :h_dim], target=data["targets"])

    n_rows = np.asarray(score["targets"]).shape[0]
    _require_whole_windows(n_rows, window)
    return gain_from_blocks(
        split(fit), None if select is None else split(select), split(score),
        # The same blocks the stride path builds, as labels: Task 2's
        # equivalence pin is what makes this substitution legal.
        groups=np.arange(n_rows) // window,
        resamples=resamples, confidence=confidence, seed=seed,
    )


def filtering_gain(
    model,
    train_paths,
    val_paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = 20,
    seed: int = 0,
    h_dim: int | None = None,
    select_episodes: int = 20,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """The bottleneck-free companion to gate criterion 4.

        gain = R2([e_t (+) h_t] -> s_t)  -  R2([e_t] -> s_t)

    `e_t` is the raw encoder embedding of frame t and `h_t` is the RSSM's
    deterministic state, `latent[:, :h_dim]`. Both arms are handed the same
    `e_t`, so the question is only whether appending `h` buys anything the
    current frame does not already provide. A positive gain says `h` carries
    something about the privileged state that frame t alone does not.

    WHY THIS EXISTS BESIDE criterion 4 RATHER THAN INSTEAD OF IT. Criterion 4
    asks whether a probe on the posterior latent beats a probe on `e_t`. That is
    what the spec defines and it is the criterion the gate reports. But its two
    arms are not matched: the latent is `h` plus a 32x32 categorical `z` -- at
    most 160 bits -- while `e_t` is 2048 continuous floats, so a model can carry
    real history in `h` and still lose on the bottleneck alone. Failing
    criterion 4 is therefore not by itself evidence that `h` is inert. Here the
    raw embedding appears in BOTH arms, so the bottleneck cancels and what is
    left is the incremental contribution of `h`. The two are complementary and
    the gate keeps reporting criterion 4 unchanged.

    THE THREE SPLITS ARE THE POINT. `filtering_comparison` selects its ridge on
    the rows it then scores, which is fair between its two probes but leaves
    both levels optimistically biased. A gain is a DIFFERENCE OF LEVELS, and the
    two feature sets have different widths (2048 against 2048 + h_dim), so a
    five-way maximum taken on the scored rows favours the wider one and would
    manufacture a positive gain out of the selection alone. So: weights from the
    training windows, ridge selected on FURTHER training episodes held out from
    those, and the reported R^2 and interval from the validation windows, which
    neither the weights nor the selection ever saw.

    The selection episodes come from BEYOND `limit`, not out of the fit set, so
    the weights are fit on the same `limit` episodes criterion 4 fits on and the
    two diagnostics differ only in where the ridge came from. The scored split is
    gathered at `seed + 1`, the same draw `filtering_report` scores criterion 4
    on, so both describe the same rows. The selection split takes `seed + 2`.

    `select_episodes` DEFAULTS LARGE BECAUSE SELECTION NOISE DOMINATES THIS
    STATISTIC. `RIDGES` is a decade grid and the scored R^2 moves ~0.10 between
    adjacent decades, which is several times the gain being measured. Measured
    on the 20k checkpoint, fit on 20 episodes and scored on 20:

        joint      1e-1 -0.0807  1e1 +0.0510  1e3 +0.3079  1e5 +0.2583  1e7 +0.0228
        embedding  1e-1 +0.1587  1e1 +0.2099  1e3 +0.3287  1e5 +0.2258  1e7 +0.0072

    Both arms peak at 1e3 and the gain there is -0.0208. A 4-episode selection
    split picks 1e5 for both and reports +0.0325 -- the SIGN FLIPS on a one-step
    selection error. A 20-episode selection split picks 1e3 for both and recovers
    -0.0208. Shrinking this parameter to save a gather does not make the number
    noisier, it makes it wrong.

    Returns `{"gain", "joint_r2", "embedding_r2", "ci_low", "ci_high",
    "confidence", "n_scored_windows", "ridge_selected", "joint_ridge",
    "embedding_ridge"}`. The interval is a window-level block bootstrap; see
    `_block_bootstrap_ci` for why the block is the window.

    Args:
        model: a `WorldModel` (or anything duck-typed like one).
        train_paths: episodes to fit the weights and select the ridge on. The
            first `limit` fit the weights; the next `select_episodes` select
            the ridge.
        val_paths: episodes to score on. Must not overlap `train_paths`.
        backbone: cached-feature backbone name, or None for the pixel arm.
        device: where to run the encoder and RSSM.
        context: real frames filtered from a zero state, per window.
        horizon: steps after the context, per window. `context + horizon` is
            also the bootstrap's block length, because it is the window
            `gather_probe_data` cuts.
        limit: episodes for the fit split, and for the scored split.
        seed: fixes the posterior samples and the bootstrap draws.
        h_dim: width of the deterministic state. Read from
            `model.rssm.cfg.h_dim` when None; pass it explicitly for models
            that do not carry an `RSSMConfig`.
        select_episodes: training episodes AFTER the first `limit` that select
            the ridge. A training pool with nothing beyond `limit` falls back to
            `fit_probe`'s default penalty for both feature sets -- weaker, and
            `ridge_selected` says so, but still never selected on the scored
            rows. See the paragraph above before lowering it.
        resamples: bootstrap draws.
        confidence: interval mass, e.g. 0.95.
    """
    if h_dim is None:
        h_dim = getattr(getattr(getattr(model, "rssm", None), "cfg", None), "h_dim", None)
        if h_dim is None:
            raise ValueError(
                "h_dim could not be read from model.rssm.cfg.h_dim; pass it "
                "explicitly -- guessing the split of `latent` into (h, z) would "
                "silently probe the wrong half"
            )

    used = list(train_paths)
    fit_paths = used[:limit]
    if not fit_paths:
        raise ValueError("no training episodes to fit the gain probes on")
    # Selection comes from the training pool BEYOND `limit`, so the weights are
    # fit on the same episodes criterion 4 fits on. Held out at EPISODE
    # granularity, like `fit_probes` and like the train/val split itself:
    # consecutive frames are near-duplicates, so a row-wise split would put the
    # same scene on both sides and every ridge would look equal.
    select_paths = used[limit:limit + select_episodes] if select_episodes > 0 else []

    gather = lambda ps, s: gather_probe_data(  # noqa: E731
        model, ps, backbone, device,
        context=context, horizon=horizon, limit=len(ps), seed=s,
    )

    fit = gather(fit_paths, seed)
    select = gather(select_paths, seed + 2) if select_paths else None

    score = gather_probe_data(
        model, val_paths, backbone, device,
        context=context, horizon=horizon, limit=limit, seed=seed + 1,
    )
    return _gain_from_splits(
        fit, select, score,
        h_dim=h_dim, window=context + horizon,
        resamples=resamples, confidence=confidence, seed=seed,
    )
