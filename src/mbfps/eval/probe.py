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

import numpy as np
import torch

from mbfps.data.episode import load_episode

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

    Returns four row-aligned arrays:

    - `"latent"` `(N, LATENT)` -- the posterior latent.
    - `"embedding"` `(N, EMBED)` -- the model's PREDICTED embedding, i.e.
      `heads(latent)["embedding"]`.
    - `"encoder_embedding"` `(N, ENC)` -- the RAW encoder output for the same
      frames, before the RSSM and before the head.
    - `"targets"` `(N, 4)`.

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

    for path in list(paths)[:limit]:
        episode = load_episode(path)
        if episode.length < need + 1:
            continue
        source = source_for(model, path, episode, backbone)
        all_actions = (
            torch.as_tensor(episode.actions.astype(np.int64)).unsqueeze(0).to(device)
        )
        # The same stride and the same final window as `evaluate_rollout`:
        # `range(0, length - need, need)` would drop the last window whenever
        # `length % need == 0`, which is a silent change in what is fit on.
        for start in range(0, episode.length - need + 1, need):
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
    }


@torch.no_grad()
def fit_probes(
    model,
    paths,
    backbone,
    device,
    context: int = 5,
    horizon: int = 45,
    limit: int = 20,
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
        limit: how many episodes to fit on.
        seed: fixes the posterior samples, and so the probe.
        select_episodes: how many of the used episodes are held back to SELECT
            the ridge rather than fit it. Selection is not optional -- a fixed
            ridge on unstandardised inputs cost held-out R^2 0.16 against a
            ceiling of 0.42 -- but it needs data the weights did not see. Zero,
            or too few episodes to spare, falls back to `fit_probe`'s default.
        ridge: pin the penalty and skip selection entirely (tests).
    """
    used = list(paths)[:limit]
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
