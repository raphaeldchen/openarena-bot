"""M3j: where on the enc(t) -> z -> h path is observed motion lost?

M3i read NO_MOTION -- at the decision horizon no seed of any arm beats
predicting no displacement at all -- and its section 3.3 retired all three of
M3h's remaining prior-side levers. But "upstream" names two levers that need
opposite changes and cost the same 13.5-hour retrain: the ENCODER, if the frozen
features do not resolve motion, or the OBJECTIVE, if they do and the RSSM
discards it. `world_model.py:81` is `embedding + reward + continue + kl`, where
the embedding target is the single frame just seen, so nothing in the loss ever
asks the latent to represent displacement.

Everything here is an INCREMENT over the current frame:

    gain = R2([enc(t) (+) B] -> y) - R2([enc(t)] -> y)

`enc(t)` sits in both arms, so the 32x32 categorical bottleneck cancels (the
argument `probe.filtering_gain` already makes) and so does the
position-constrains-motion confound, which is new here and is the reason M3i's
section 9 levels could not have answered this question: in a corridor your
location predicts your motion, and that credit belongs to the frame.

Targets are BACKWARD, so the model is scored only on motion it has already
observed -- the representation's best case, which is what makes a null decisive.

Everything here is pure: arrays in, arrays and readings out. No torch, no I/O,
no record schema. `scripts/latent_retention.py` owns all three.
"""

from __future__ import annotations

import numpy as np

# Pre-registered (spec 2.2, 2.3, 3.1). Three arms, every gain within a cell
# against that cell's own base -- no arm is ranked against another, so the
# family is the arms and not their pairs.
RETENTION_FAMILY: int = 3
SEEDS_REQUIRED: int = 2
ARMS_REQUIRED: int = 2

# Reported at every one of these; a cleared gain at ANY of them counts as motion
# retained (spec 2.3). The disjunction is deliberate: the lever question is
# binary -- does the RSSM retain observed motion at all -- so being generous
# about WHERE makes a null across all three a stronger null.
K_REPORTED: tuple[int, ...] = (1, 4, 15)

# The horizon the reading is named for. k = 1 is degenerate on both targets
# (translation is 0.6% of the map extent; rotation is exactly zero 62.9% of the
# time), and of the two non-degenerate horizons k = 4 is the shorter, so it
# demands less integration from `h` and a null there is the stronger null.
DECISION_K: int = 4

# The ladder, in path order: what two real frames provide, then what each part
# of the latent adds. `two_frame` is a REFERENCE, not a ceiling -- `h`
# integrates the action sequence, which two frames do not contain, so a latent
# rung can legitimately exceed it.
RUNGS: tuple[str, ...] = ("two_frame", "deterministic", "stochastic", "full")
LATENT_RUNGS: tuple[str, ...] = ("deterministic", "stochastic", "full")
Z_BEARING_RUNGS: tuple[str, ...] = ("stochastic", "full")

# `translation` decides; `rotation` is the positive control. A 20-unit
# translation is ~2% of the map extent through a 112x112 frozen backbone, while
# a 32-degree turn rewrites the frame -- so reading rotation but not translation
# means spatial resolution, and reading neither means the RSSM discards motion
# as such.
TARGETS: tuple[str, ...] = ("translation", "rotation")

CONFIDENCE: float = 0.95
RESAMPLES: int = 1000
"""Interval mass and bootstrap draws. 0.95 per comparison rather than a
Bonferroni over 4 rungs x 3 horizons: that would need the 0.2nd percentile,
which 1000 draws cannot resolve. The multiple-comparison burden is carried by
the SEEDS_REQUIRED x ARMS_REQUIRED agreement instead -- the project's existing
instrument, and one this sample size can actually resolve."""


def shifted_rows(window, step, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For every row with a partner `k` steps earlier IN THE SAME WINDOW, that
    row and its partner. `(rows, source)`, both `(m,)` int64.

    Built from the `window` and `step` LABELS rather than from array order.
    `gather_probe_data` does currently emit each window's rows contiguously in
    step order, and `rows - k` would therefore work today -- but that is a
    regularity of the gather, not a documented guarantee, and a pairing that
    walked off the front of one window into the previous one would stay
    shape-valid while scoring a displacement between two different episodes.

    A `k` no window is long enough for returns two empty arrays rather than
    raising: a horizon with no rows is a fact about the protocol that the
    caller reports beside its row counts, so the refusal lives in one place.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    window = np.asarray(window).astype(np.int64, copy=False)
    step = np.asarray(step).astype(np.int64, copy=False)
    if window.shape != step.shape or window.ndim != 1:
        raise ValueError(
            f"window {window.shape} and step {step.shape} must be the same 1-D shape"
        )
    index = {}
    for i, (w, s) in enumerate(zip(window, step)):
        key = (int(w), int(s))
        if key in index:
            raise ValueError(
                f"duplicate label {key}: the pairing between window and step is ambiguous"
            )
        index[key] = i
    rows, source = [], []
    for i, (w, s) in enumerate(zip(window, step)):
        partner = index.get((int(w), int(s) - k))
        if partner is not None:
            rows.append(i)
            source.append(partner)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(source, dtype=np.int64),
    )


def backward_translation(targets, window, step, k: int) -> tuple[np.ndarray, np.ndarray]:
    """`p(t) - p(t-k)` as a VECTOR in map units, and the rows it is defined on.

    `targets` is the `(N, 4)` array `probe.probe_targets` builds -- pos_x,
    pos_y, sin(angle), cos(angle) -- so position is its first two columns.

    BACKWARD, which is the point: every frame in the pair has already been
    observed by the posterior, so this is the representation's best case. M3i
    scored FORWARD displacement, where the target also depends on the policy;
    a null here says the latent cannot report motion it has already seen, which
    is a sharper statement and a necessary condition for the forward one.
    """
    rows, source = shifted_rows(window, step, k)
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"targets must be (N, >=2), got {values.shape}")
    return values[rows, :2] - values[source, :2], rows


def backward_rotation(targets, window, step, k: int) -> tuple[np.ndarray, np.ndarray]:
    """`(sin, cos)` of the angle change over the last `k` steps, and its rows.

    The POSITIVE CONTROL for the motion question, not a competing reading. A
    20-unit translation is ~2% of the map extent seen through 112x112 frames
    from a frozen single-frame backbone; a 32-degree turn rewrites the frame. So
    a ladder that reads rotation but not translation is telling us about spatial
    RESOLUTION, while a ladder that reads neither is telling us the RSSM
    discards motion as such. Without this the translation null would be
    confounded with "the encoder cannot see a 20-unit move".

    `(sin, cos)` rather than degrees for the reason `probe.probe_targets` uses
    it on absolute angle: 359 and 1 must be near each other. The wrap needs no
    explicit handling HERE because sin and cos are already periodic -- taking
    them of the raw difference is identical to taking them of the wrapped one,
    and cheaper than wrapping first and then discarding the wrap.
    """
    rows, source = shifted_rows(window, step, k)
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"targets must be (N, >=4), got {values.shape}")
    now = np.arctan2(values[rows, 2], values[rows, 3])
    then = np.arctan2(values[source, 2], values[source, 3])
    delta = now - then
    return np.stack([np.sin(delta), np.cos(delta)], axis=1), rows


def rung_block(data: dict, rung: str, *, rows, source, h_dim: int) -> np.ndarray:
    """The second feature block for `rung`, on `rows`.

    Handed the SAME `rows` every other rung gets, so all four are gains against
    one shared base fit on one row set (spec 2.1) and are therefore comparable
    to each other rather than each only to its own fit.

    `two_frame` reads `source` -- `enc(t-k)`, the frame the displacement is
    measured FROM. Not `enc(t-1)`: the two frames that determine
    `p(t) - p(t-k)` are `t` and `t-k`, so a frame one step back would leave the
    reference near-uninformative at every k > 1 and would also hand the latent
    rungs an advantage they did not earn, since `h(t)` sees frame `t-k` and the
    reference would not. At k = 1 the two coincide, which is what makes the
    mistake invisible.
    """
    rows = np.asarray(rows, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    if rung == "two_frame":
        return np.asarray(data["encoder_embedding"], dtype=np.float64)[source]
    if rung not in LATENT_RUNGS:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
    latent = np.asarray(data["latent"], dtype=np.float64)
    h_dim = int(h_dim)
    if not 1 <= h_dim < latent.shape[1]:
        raise ValueError(
            f"h_dim={h_dim} does not index a {latent.shape[1]}-wide latent into a "
            "non-empty (h, z) pair"
        )
    # `latent` is `cat([h, z])` -- h FIRST, per RSSM.observe. Slicing the tail
    # for `deterministic` reads the stochastic state instead and answers a
    # different question with a shape that is often still valid.
    if rung == "deterministic":
        return latent[rows, :h_dim]
    if rung == "stochastic":
        return latent[rows, h_dim:]
    return latent[rows]
