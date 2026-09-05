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
