"""The trust horizon: per-window crossing steps, margins and survival curves.

`scripts/diagnose_dynamics.py` decides the M3 gate from two MEAN curves --
`k45_position` and `persistence_position` -- and a mean curve cannot say on how
many windows the model was still ahead of "you did not move" at step h. These
functions work on the PER-WINDOW error rows the M3d reference pass keeps, and
every one of them is a pure numpy function with no torch, no files and no
randomness, pinned by synthetic priors with hand-built answers:

  * `moved_mask` -- the ground-truth gate on every other quantity. A window
    that has not left its starting point by step h has a persistence error of
    zero there, and any model error at all "loses" to it; the mask is the 2-D
    Euclidean displacement `|p(h) - p(0)| >= MIN_MOVE`, computed once from the
    true positions so the probe channel and the probe-free channel score the
    same rows.
  * `crossing_step` -- the first step, at or after the first moved step, at
    which the model's error STRICTLY exceeds persistence's. Ties never cross:
    a persistence clone is not worse than persistence, so its crossing is
    `horizon + 1` on every row, the same as a perfect predictor's -- the two
    are told apart by the margin (and, in the decomposition, by the ratio),
    not by the crossing.
  * `persistence_margin` -- `persist_err - model_err`, map units, positive
    when the model beats persistence at that step. It is `gap_closed`'s
    numerator PER WINDOW, and it is the decision statistic instead of
    `gap_closed` because a ratio of mean curves has no per-window estimator to
    cluster by episode.
  * `survival` and `trust_horizon` -- `S(h)`, the fraction of finite crossings
    strictly beyond h, and `H*_q`, the largest h with `S(h) >= q`. `H*_q` is an
    integer by construction: it reads the curve, never a quantile of the
    multiset, so an even-count multiset gives no half-step.
"""

from dataclasses import dataclass

import numpy as np

MIN_MOVE: float = 5.0


def moved_mask(
    p_true: np.ndarray, p_true0: np.ndarray, min_move: float = MIN_MOVE
) -> np.ndarray:
    """`(n, H, 2)` true positions and `(n, 2)` anchors -> `(n, H)` bool.

    True where the 2-D Euclidean displacement `|p_true[:, h] - p_true0|` is at
    least `min_move` -- `>=`, so a window that moves exactly the threshold is
    scored. Euclidean, not per-axis: `(3, 4)` has moved 5 units, `(2.5, 2.5)`
    has moved 3.54, and only the norm orders them the way the map does.
    """
    p_true = np.asarray(p_true, dtype=np.float64)
    p_true0 = np.asarray(p_true0, dtype=np.float64)
    if p_true.ndim != 3 or p_true.shape[-1] != 2:
        raise ValueError(f"p_true must be (n, H, 2), got {p_true.shape}")
    if p_true0.shape != (p_true.shape[0], 2):
        raise ValueError(
            f"p_true0 must be (n, 2) = ({p_true.shape[0]}, 2), got {p_true0.shape}"
        )
    displacement = np.linalg.norm(p_true - p_true0[:, None, :], axis=-1)
    return displacement >= min_move


def crossing_step(
    model_err: np.ndarray, persist_err: np.ndarray, moved: np.ndarray
) -> np.ndarray:
    """Per row, the first 1-based step at or after the first moved step where
    `model_err > persist_err` STRICTLY; `H + 1` if never; NaN if the row never
    moves.

    All three arguments are `(n, H)`; the result is `(n,)` float64 (NaN needs
    a float row). Steps BEFORE the first moved step are ignored even when the
    model loses there -- at those steps persistence's error is the probe's
    floor on a frame that has not changed, and nothing about the model is
    being measured.
    """
    model_err = np.asarray(model_err, dtype=np.float64)
    persist_err = np.asarray(persist_err, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    if model_err.ndim != 2:
        raise ValueError(f"model_err must be (n, H), got {model_err.shape}")
    if persist_err.shape != model_err.shape or moved.shape != model_err.shape:
        raise ValueError(
            "model_err, persist_err and moved must share one (n, H) shape, got "
            f"{model_err.shape}, {persist_err.shape}, {moved.shape}"
        )
    if not (np.isfinite(model_err).all() and np.isfinite(persist_err).all()):
        raise ValueError("crossing_step: model_err and persist_err must be finite")
    n, horizon = model_err.shape
    ever_moved = moved.any(axis=1)
    first_moved = moved.argmax(axis=1)  # 0-based index of the first True per row
    at_or_after = np.arange(horizon)[None, :] >= first_moved[:, None]
    crosses = (model_err > persist_err) & at_or_after
    crossing = np.where(
        crosses.any(axis=1), crosses.argmax(axis=1) + 1, horizon + 1
    ).astype(np.float64)
    crossing[~ever_moved] = np.nan
    return crossing


def persistence_margin(model_err: np.ndarray, persist_err: np.ndarray) -> np.ndarray:
    """`persist_err - model_err`, elementwise on `(n, H)`: positive where the
    model beats persistence at that step, zero for a persistence clone."""
    model_err = np.asarray(model_err, dtype=np.float64)
    persist_err = np.asarray(persist_err, dtype=np.float64)
    if model_err.shape != persist_err.shape:
        raise ValueError(
            f"model_err and persist_err must share a shape, got {model_err.shape} "
            f"and {persist_err.shape}"
        )
    return persist_err - model_err


def survival(crossings: np.ndarray, horizon: int) -> np.ndarray:
    """`S(h)` for h = 0..horizon: the fraction of FINITE crossings strictly
    greater than h. `(horizon + 1,)` float; all NaN when no crossing is finite.

    NaN crossings are windows that never moved; they are excluded from the
    denominator, not counted as failures. Every finite crossing must lie in
    `1..horizon + 1` (the value `crossing_step` produces): a 0 would be a
    0-based step and a `horizon + 2` a crossing from a longer horizon, and
    either would move `S` without any test noticing.
    """
    crossings = np.asarray(crossings, dtype=np.float64)
    if crossings.ndim != 1:
        raise ValueError(f"crossings must be (n,), got {crossings.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    finite = crossings[np.isfinite(crossings)]
    if finite.size == 0:
        return np.full(horizon + 1, np.nan)
    if (finite < 1).any() or (finite > horizon + 1).any():
        raise ValueError(
            f"crossings must lie in 1..{horizon + 1}, got min {finite.min()} "
            f"max {finite.max()}"
        )
    steps = np.arange(horizon + 1)
    return (finite[None, :] > steps[:, None]).mean(axis=1)


def trust_horizon(surv: np.ndarray, q: float) -> int:
    """`H*_q`: the largest h with `surv[h] >= q`; 0 when no h qualifies; -1
    when `surv` is all NaN (no finite crossing to read)."""
    surv = np.asarray(surv, dtype=np.float64)
    if surv.ndim != 1:
        raise ValueError(f"surv must be (H + 1,), got {surv.shape}")
    if np.isnan(surv).all():
        return -1
    qualifying = np.flatnonzero(surv >= q)
    if qualifying.size == 0:
        return 0
    return int(qualifying[-1])


# ---------------------------------------------------------------------------
# Displacement decomposition, embedding ratio, scale-corrected error (spec 2.2,
# items 2 and 4). Every quantity is per (row, step) on `moved` cells only; a
# cell that is not moved is NaN in every array, never 0, so a pooled mean
# cannot be diluted by windows where the agent stood still.
# ---------------------------------------------------------------------------

ALPHAS: np.ndarray = np.linspace(0.0, 2.0, 201)
"""The scale grid: 0 (the prior sits still) to 2 (twice the distance), step 0.01.

1.0 (index 100) and 0.5 (index 50) are exact grid points, so a perfect
predictor and an exact 2x overshoot fit exactly; the endpoints are the
`boundary` flag's edges.
"""


@dataclass(frozen=True)
class Decomposition:
    """`displacement_decomposition`'s per-(row, step) channels, all `(n, H)`.

    `ratio_probe` is `|d_hat| / |d_hat_real|` -- the probe's reading of the
    imagined displacement over the SAME probe's reading of the real one, so a
    flatter probe cancels; it is the ratio the readings decide on. `ratio_raw`
    (`|d_hat| / |d|`) is stored beside it as a secondary column: a probe that
    reads every displacement at 0.3 of its length reads 0.3 here and 1.0 in
    `ratio_probe`. `cosine` is the direction agreement of `d_hat` with `d`.
    `zero_displacement` marks moved cells where the prior did not move at all
    (`|d_hat| == 0`): their cosine is NaN and they are counted, not pooled.
    """

    ratio_probe: np.ndarray
    ratio_raw: np.ndarray
    cosine: np.ndarray
    moved: np.ndarray
    zero_displacement: np.ndarray


def displacement_decomposition(p_hat, p_true, p_hat0, p_true0, p_hat_real, moved) -> Decomposition:
    """Magnitude ratios and direction agreement of the imagined displacement.

    `p_hat`, `p_true`, `p_hat_real` are `(n, H, 2)`; `p_hat0`, `p_true0` are
    `(n, 2)`; `moved` is `(n, H)` bool. `d_hat = p_hat - p_hat0`,
    `d = p_true - p_true0`, `d_hat_real = p_hat_real - p_hat0` per row.
    NaN where not moved; `ratio_probe` also NaN where `|d_hat_real| == 0` (a
    probe that reads no real displacement gives no ratio, not an infinite
    one); `cosine` NaN where `|d_hat| == 0`. `ratio_raw` needs no zero guard:
    `moved` means `|d| >= MIN_MOVE > 0`.
    """
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    p_hat0 = np.asarray(p_hat0, dtype=np.float64)
    p_true0 = np.asarray(p_true0, dtype=np.float64)
    p_hat_real = np.asarray(p_hat_real, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    d_hat = p_hat - p_hat0[:, None]
    d = p_true - p_true0[:, None]
    d_hat_real = p_hat_real - p_hat0[:, None]
    norm_hat = np.linalg.norm(d_hat, axis=-1)
    norm_true = np.linalg.norm(d, axis=-1)
    norm_real = np.linalg.norm(d_hat_real, axis=-1)
    # np.where evaluates both branches, so the divisions run on the masked-out
    # cells too; their warnings are silenced here, their values discarded.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_probe = np.where(moved & (norm_real > 0.0), norm_hat / norm_real, np.nan)
        ratio_raw = np.where(moved, norm_hat / norm_true, np.nan)
        # 0 / 0 where |d_hat| = 0 is NaN by IEEE; `zero_displacement` counts those
        cosine = np.where(moved, (d_hat * d).sum(axis=-1) / (norm_hat * norm_true), np.nan)
    return Decomposition(
        ratio_probe=ratio_probe,
        ratio_raw=ratio_raw,
        cosine=cosine,
        moved=moved,
        zero_displacement=moved & (norm_hat == 0.0),
    )


def embedding_ratio(e_hat_disp, e_true_disp, moved) -> np.ndarray:
    """`R_free`: `||e_hat(h) - e_hat(0)|| / ||e(h) - e(0)||`, from the two norms.

    Both inputs are `(n, H)` norms the diagnostics hook already reduced --
    `diagnostics.Trajectories.embedding_displacement` and
    `true_embedding_displacement`, which is why this takes the norms and not
    the four embedding arrays. NaN where not moved or where the true
    embedding did not move (`e_true_disp == 0`).
    """
    e_hat_disp = np.asarray(e_hat_disp, dtype=np.float64)
    e_true_disp = np.asarray(e_true_disp, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(moved & (e_true_disp > 0.0), e_hat_disp / e_true_disp, np.nan)


@dataclass(frozen=True)
class ScaleCorrection:
    """`scale_corrected_error`'s per-step fit and per-row held-out score.

    Fold A is the rows with an even episode label, fold B the odd ones.
    `alpha_a[h]` is fit on A's moved rows at step h and scores B's
    (`score_a[h]` = the median over B's moved rows); `alpha_b` the reverse.
    `held_out[i, h]` is row i scored with the alpha fit on the OTHER fold, NaN
    where not moved. `boundary[h]` is True when either fold's alpha sits on a
    grid endpoint: at 0 the "corrected" error is persistence itself, at 2 the
    grid ran out, and neither is a correction. With fewer than two distinct
    episode labels there is nothing to hold out: every array is NaN (`boundary`
    all False) and `folds_available` is False.
    """

    alpha_a: np.ndarray
    alpha_b: np.ndarray
    score_a: np.ndarray
    score_b: np.ndarray
    held_out: np.ndarray
    boundary: np.ndarray
    folds_available: bool


def _argmin_smallest_alpha(medians: np.ndarray, alphas: np.ndarray) -> int:
    """Index of the smallest median; among ties, the index of the SMALLEST alpha.

    `np.argmin` alone returns the first index, which is the smallest alpha
    only on an ascending grid. A persistence clone ties every alpha and must
    read alpha = 0 whichever way the grid is ordered.
    """
    ties = np.flatnonzero(medians == medians.min())
    return int(ties[np.argmin(alphas[ties])])


def scale_corrected_error(p_hat, p_true, p_hat0, moved, episode, alphas=ALPHAS) -> ScaleCorrection:
    """Cross-fitted scale correction of the imagined displacement (spec 2.2, item 4).

    Per step h and fold, `alpha = argmin over alphas of median over the fold's
    MOVED rows of |p_hat0 + alpha * d_hat - p_true[:, h]|` (2-D Euclidean
    norm), and every moved row of the other fold is scored with it. The median
    is the objective on purpose: one wild row cannot drag the scale. A fold
    with no moved rows at a step cannot be fit there -- its alpha, its score
    and the other fold's held-out errors are NaN at that step, and the step is
    not a boundary. `p_hat`, `p_true` `(n, H, 2)`; `p_hat0` `(n, 2)`; `moved`
    `(n, H)` bool; `episode` `(n,)` int labels.
    """
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    p_hat0 = np.asarray(p_hat0, dtype=np.float64)
    moved = np.asarray(moved, dtype=bool)
    episode = np.asarray(episode)
    alphas = np.asarray(alphas, dtype=np.float64)
    n, horizon, _ = p_hat.shape
    nan_h = np.full(horizon, np.nan)
    if len(np.unique(episode)) < 2:
        return ScaleCorrection(
            alpha_a=nan_h.copy(), alpha_b=nan_h.copy(), score_a=nan_h.copy(), score_b=nan_h.copy(),
            held_out=np.full((n, horizon), np.nan), boundary=np.zeros(horizon, dtype=bool),
            folds_available=False,
        )
    d_hat = p_hat - p_hat0[:, None]
    # candidate error for every (row, step, alpha): |p_hat0 + alpha * d_hat - p_true|, shape (n, H, A)
    corrected = p_hat0[:, None, None, :] + alphas[None, None, :, None] * d_hat[:, :, None, :]
    candidates = np.linalg.norm(corrected - p_true[:, :, None, :], axis=-1)
    fold_a = episode % 2 == 0
    folds = (fold_a, ~fold_a)
    alpha = [nan_h.copy(), nan_h.copy()]
    score = [nan_h.copy(), nan_h.copy()]
    held_out = np.full((n, horizon), np.nan)
    for h in range(horizon):
        for k, fit_rows in enumerate(folds):
            fit = fit_rows & moved[:, h]
            score_rows = folds[1 - k] & moved[:, h]
            if not fit.any():
                continue
            index = _argmin_smallest_alpha(np.median(candidates[fit, h, :], axis=0), alphas)
            alpha[k][h] = alphas[index]
            held_out[score_rows, h] = candidates[score_rows, h, index]
            if score_rows.any():
                score[k][h] = np.median(candidates[score_rows, h, index])
    edges = (alphas[0], alphas[-1])
    boundary = np.isin(alpha[0], edges) | np.isin(alpha[1], edges)
    return ScaleCorrection(
        alpha_a=alpha[0], alpha_b=alpha[1], score_a=score[0], score_b=score[1],
        held_out=held_out, boundary=boundary, folds_available=True,
    )
