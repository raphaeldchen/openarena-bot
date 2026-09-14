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
