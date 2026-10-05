"""M3n: is the one-step map closer to copying, or to perfect?

M3m established that the 45-step rollout's error is COMPOUNDING: `burden(1)`
is not resolvable from zero at 2 SE in 9 of 9 cells, while `compounding(45)`
is. It could not establish WHY the one-step map is cheap, and it said so: a
map that predicts motion well and a map that predicts almost no motion both
score a small `burden(1)`, because holding position is cheap relative to the
probe's readout error.

This module places the one-step map between those two ends.

WHY M3m's `motion_margin` COULD NOT DO IT. It subtracted the k=1 rung, a
PROBE-SPACE model error of 88-224 map units, from the median true one-step
displacement, a GROUND-TRUTH quantity of 3.97. Readout error dominated the
difference 22.2x-56.5x, the base control failed 9 of 9, and Reading H shipped
UNREADABLE. The error was attractive rather than careless: the statistic needs
a scale for "how much motion was there to predict," and the intuitive answer is
the true displacement. But the numerator lives in probe space, and the only
probe-space answer is what a PERFECT predictor would have won -- which is
measured, not constant, and varies 20.5x across the nine cells at the decision
cell (k=1, h=1), from 0.575649 to 11.824.

THE SAME MISTAKE IS AVAILABLE ONE LEVEL UP. Normalising this module's `skill`
by 3.9694722203504225 produces an apparent split across the nine cells at
(k=1, h=1): three cells at 106-118% (118.3, 112.1, 106.4), five at 6.5-43.2%
(43.2, 21.5, 19.8, 10.1, 6.5), and one at -1.8% (`random_vit_seed0`) that
belongs to neither group. Normalised by `headroom` at the same cell, the same
nine read -2.4% to 72.7%, unimodal, and the split is gone. The pattern was the
constant denominator.

THE THREE DIFFERENCES, all probe-space, all per (k, h):

    headroom(k, h) = hold_k(h) - floor(h)    what a PERFECT predictor wins
    skill(k, h)    = hold_k(h) - rung_k(h)   what the MODEL wins
    deficit(k, h)  = rung_k(h) - floor(h)    M3m's `burden(k, h)`, unchanged

`skill + deficit == headroom` exactly -- the hold term cancels. Measured on the
shipped M3m curves at the k=45 column, the max residual over nine cells and 45
steps is 1.421e-14.

THAT IDENTITY IS WHY THE READING CARRIES NO RATIO. The two-sided test against
the copying end and the perfect end is `skill > 0` and `deficit > 0`: two
differences, no denominator. A denominator would matter, because `headroom`
crosses zero inside the reported grid on a real cell. `pixel_ae_seed1` at k=45,
with the share taken as `skill / headroom`:

    h=1: headroom +0.575649   share  +69.8%
    h=2: headroom -0.641206   share +339.5%
    h=3: headroom +0.798695   share -335.7%

The two absurd shares are at h=2, where the denominator is negative, and h=3,
where it is still under 0.8; h=1 reads plausibly.

BOTH AXES ARE NAMED. `k` is the re-grounding period, `h` the horizon step.
Write `headroom(k, h)`, never `headroom(45)`.
"""

from __future__ import annotations

import numpy as np

from mbfps.eval.burden import checked_pair

REPORTED_H: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45)
"""The horizon steps the record publishes. Inherited from M3m unchanged, so the
two milestones' tables are read on the same grid."""

DECISION_K: int = 1
DECISION_H: int = 1
"""The cell the verdict is read at. `DECISION_K` is a LABEL, not a choice: at
h=1 every k grounds at step 0 and imagines one prior step, so all three
differences are identical across k (`RegroundingSweep.k_invariance_at_h1`).
That makes the one-step verdict k-free by construction, which is a control
rather than an assumption."""

IDENTITY_TOLERANCE: float = 1e-9
"""`triple_residual` above this is a plumbing fault, not rounding. The observed
residual on the shipped curves is 1.421e-14, about five orders below."""

CONFIDENCE: float = 0.95
RESAMPLES: int = 2000
"""The verdict's episode-clustered bootstrap. 229 windows over 24 episodes on
every shipped cell; see `pooling.clustered_interval` for why the unit is the
episode and why the direction of that error matters here specifically."""

SECONDARY_SIGMAS: int = 2
"""Multiplier applied to `RegroundingSweep.standard_error` for the RECORDED secondary
figure, which decides nothing. `RegroundingSweep.standard_error` returns ONE standard error and
`scripts/diagnose_dynamics.py:1430` doubles it; mislabelling one as two in M3m
produced the opposite conclusion from the correct one, and the wrong one was
the more interesting-sounding."""

SEEDS_MINIMUM: int = 3
ARMS_REQUIRED: int = 2
"""An arm with fewer than `SEEDS_MINIMUM` seeds is refused by name rather than
tallied. `ARMS_REQUIRED` is satisfied automatically by the strict-majority bar
over nine cells -- no arm holds more than three -- and is stated anyway so the
protection M3m carried is visibly not dropped."""

DISPLACEMENT_RECORDED_ONLY: float = 3.9694722203504225
"""The median true one-step displacement on the shipped split. RECORDED AND
USED BY NO STATISTIC IN THIS MODULE.

It is the quantity M3m's design mistook for a probe-space scale, kept in the
record so a reader can see that it plays no part in the reading. Nothing below
this line references it."""


def headroom(hold: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """`hold - floor`: what a PERFECT predictor wins over copying, per step.

    The denominator M3m's design needed and did not have. Measured, not
    constant: 0.575649 to 11.824 at h=1 across the nine shipped cells.

    Also the readability gate. If this is not resolvably positive, a perfect
    predictor cannot be told from a copying one, and nothing between them can
    be placed either.
    """
    hold, floor = checked_pair(hold, floor)
    return hold - floor


def skill(hold: np.ndarray, rung: np.ndarray) -> np.ndarray:
    """`hold - rung`: what the MODEL wins over copying, per step.

    Positive means the rung beats holding the floor's own position from the
    rung's last re-grounding step. Both sides pass through the same encoder,
    RSSM and embedding head and are read by the same probe.

    WHAT CANCELS DEPENDS ON `g = ground_step(k, h)`, and it is established only
    where `g == 0`, which is every `h <= k`. There the rung and the hold
    baseline descend from the SAME posterior latent, so the readout error they
    share cancels in the difference and what remains is of the order of the
    model's predicted displacement. That is why this statistic is readable
    where `motion_margin` was not: it subtracted a ground-truth displacement,
    so there was no common latent for anything to cancel against.

    Where `g > 0` that is not exact. The rung's own re-grounding `observe` draws
    fresh categorical samples and the floor's runs elsewhere in the same stream,
    so the two latents are redraws of one another and the difference carries a
    sampling-redraw term. On the test rig at `k=2` their `z` differed in 28 of
    32 groups at step 2 and in all 32 at step 4. Whether the readout error also
    cancels there is NOT claimed; the `g > 0` cells are reported, and a reader
    should expect them to be noisier for this reason as well as for the longer
    horizon (`RegroundingSweep.hold_position` has the measurement).

    The milestone's decision cell is `(k=1, h=1)`, where `g == 0`, so the
    verdict rests on the established case. That is also why the unscoped claim
    was tempting: the cell the verdict reads lies inside the region where the
    claim holds, so the verdict never meets a case that contradicts it.
    """
    hold, rung = checked_pair(hold, rung)
    return hold - rung


def deficit(rung: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """`rung - floor`: how far the rung sits above the floor, per step.

    IDENTICAL to M3m's `burden(k, h)`, which is kept and unchanged. Named again
    here because in M3n's framing it is the distance to the PERFECT end of the
    axis, and the two-sided test reads it as such: `deficit` not resolvable
    from zero means the rung is statistically indistinguishable from a perfect
    one-step predictor.
    """
    rung, floor = checked_pair(rung, floor)
    return rung - floor


def triple_residual(
    hold: np.ndarray, rung: np.ndarray, floor: np.ndarray
) -> float:
    """`max |headroom - (skill + deficit)|` over the horizon.

    `max`, NOT `sum` or `mean`: one bad step is a plumbing fault and summing
    would let 45 tiny roundings hide it while averaging would divide it away.

    Real data barely separates the reductions, so it cannot be what tests this.
    On M3n's own k=45 column, 8 of 9 cells give exactly 0.0 under both `max`
    and `sum`, and the ninth (`pixel_ae_seed0`) separates them by 4x -- 1.421e-14
    against 5.684e-14 -- which is rounding on either reading. A tolerance wide
    enough to pass that column passes the `np.sum` mutant too. So the tests
    assert with exact equality, and build a fixture that separates `max` from
    `sum` (several bad steps, of different sizes and signs) rather than relying
    on real data to do it.
    """
    whole = headroom(hold, floor)
    parts = skill(hold, rung) + deficit(rung, floor)
    return float(np.max(np.abs(whole - parts)))
