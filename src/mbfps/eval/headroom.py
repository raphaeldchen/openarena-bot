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

from dataclasses import dataclass

import numpy as np

from mbfps.eval.burden import checked_pair, strict_majority

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
"""An arm with fewer than `SEEDS_MINIMUM` seeds is refused by name rather than
tallied: `reading_headroom` raises `ValueError`."""

ARMS_REQUIRED: int = 2
"""`reading_headroom` enforces this twice, and neither check covers the other.

  1. Fewer than `ARMS_REQUIRED` arms present: `ValueError`, naming the arms.
     That is a verdict from one arm, the shape `--arms random_vit` built in
     M3j.
  2. A strict majority whose winning cells span fewer than `ARMS_REQUIRED`
     arms: `NO_MAJORITY`, not an error. Arms of 3, 3 and 9 seeds pass the first
     check, and the nine-seed arm alone reaches the bar of 8 at fifteen cells.

The strict-majority bar does not stand in for either: it counts cells, not
arms, and the 3x3 shape that makes five cells span two arms is one shape the
function accepts out of many."""

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


PLACEMENTS: tuple[str, ...] = (
    "BETWEEN", "AT_PERFECT", "AT_COPYING", "AMBIGUOUS", "UNREADABLE",
)
"""Where one cell's one-step map sits on the copying-to-perfect axis.

`UNREADABLE` is about the INSTRUMENT, the other four about the model. It is in
the same tuple because it is a per-cell outcome and a majority of it is a
verdict -- M3n's exit 47 -- rather than a missing value."""

NO_MAJORITY: str = "NO_MAJORITY"
"""No verdict was drawn: either no placement held a strict majority of the
cells, or one did and its cells spanned fewer than `ARMS_REQUIRED` arms. The
`rule` line says which. M3n's exit 48."""


@dataclass(frozen=True)
class Interval:
    """A point estimate and an episode-clustered bootstrap interval."""

    point: float
    ci_low: float
    ci_high: float

    def resolvably_positive(self) -> bool:
        """The whole interval lies above zero.

        `ci_low > 0.0`, strictly. An interval whose lower bound is exactly 0.0
        does not exclude zero, and the whole two-sided reading is a question
        about exclusion.
        """
        return self.ci_low > 0.0


@dataclass(frozen=True)
class HeadroomCell:
    """One cell -- one arm at one seed -- at ONE (k, h)."""

    arm: str
    seed: int
    headroom: Interval
    skill: Interval
    deficit: Interval
    clusters: int
    """Distinct episodes behind the bootstrap. Carried because an interval
    drawn from one cluster is not an interval, and printed beside the intervals
    in the `clusters` column. Nothing in this module refuses on it: `placement`
    and `reading_headroom` never read it, so a one-cluster cell is placed like
    any other, and a caller that wants the refusal has to make it before
    building the cell."""

    def placement(self) -> str:
        """This cell's member of `PLACEMENTS`.

        THE GATE RUNS FIRST, and the order is load-bearing rather than tidy. A
        cell whose `headroom` straddles zero can still have a resolvably
        positive `skill` and an unresolvable `deficit`, which without the gate
        reads `AT_PERFECT` -- a verdict about a model, taken on a cell where a
        perfect predictor is indistinguishable from a copying one.

        The tests construct that case; the shipped record does not show it.
        What the record does show is the premise behind it: `headroom` changes
        sign inside the reported grid on a real cell. `pixel_ae_seed1` at k=45:

            h=1: headroom +0.575649  skill +0.4020  deficit +0.1737
            h=2: headroom -0.641206  skill -2.1768  deficit +1.5356
            h=3: headroom +0.798695  skill -2.6812  deficit +3.4799

        A perfect predictor wins 0.58, then loses 0.64, then wins 0.80, against
        a floor of 259.06 map units at h=1. That is what makes the gate
        load-bearing rather than tidy. It is NOT a measured case of a skill-
        positive cell reading `AT_PERFECT`: the h=2 skill is negative, so
        without the gate that step would read `AT_COPYING` or `AMBIGUOUS`,
        by whether its deficit interval excludes zero. The h=2 point estimate
        is below zero, so on any interval that contains its own point the gate
        refuses it. Whether the h=1 interval excludes zero is not in that
        record, which carries no per-window hold curve; this docstring does
        not claim it.

        `AMBIGUOUS` is reachable WITH a resolvable headroom. The gate
        establishes only that the two ends are separated, not that the ruler is
        fine enough to locate a point between them.
        """
        if not self.headroom.resolvably_positive():
            return "UNREADABLE"
        has_skill = self.skill.resolvably_positive()
        has_deficit = self.deficit.resolvably_positive()
        if has_skill and has_deficit:
            return "BETWEEN"
        if has_skill:
            return "AT_PERFECT"
        if has_deficit:
            return "AT_COPYING"
        return "AMBIGUOUS"


@dataclass(frozen=True)
class HeadroomInputs:
    """The pooled cells, with the (k, h) they were all read at."""

    cells: dict[tuple[str, int], HeadroomCell]
    decision_k: int
    decision_h: int
    ks: tuple[int, ...]


@dataclass(frozen=True)
class HeadroomStatus:
    """The verdict, the rule it was reached by, and the full tally."""

    verdict: str
    rule: str
    tally: dict[str, tuple[tuple[str, int], ...]]
    """Placement -> the (arm, seed) cells voting for it, sorted by (arm,
    seed) whatever order the input dict was built in. Every placement present
    as a key, including those with no votes, so a reader can tell "no cell read
    AT_COPYING" from "AT_COPYING was not considered"."""


def reading_headroom(inputs: HeadroomInputs) -> HeadroomStatus:
    """Place the one-step map, by strict majority of the cells.

    REFUSES rather than falls through on an arm short of `SEEDS_MINIMUM`.
    `strict_majority(1) == 1`, so without that refusal a single lucky cell
    would establish an arm -- the M3j trap, where `--arms random_vit` printed a
    row reading `clears = up` beside a verdict of NO DIFFERENCE.

    THE BAR IS COMPUTED. `strict_majority(len(cells))` returns 5 at nine cells
    and 6 at eleven, so storing 5 would be a majority at one cell count and a
    minority at another; M3l's design was reworked for exactly that.

    `ARMS_REQUIRED` IS ENFORCED, by two checks, because neither covers the
    other. The bar counts cells and not arms, so it cannot stand in for either:
    one arm of three seeds returns `BETWEEN in 3 of 3 cells`, and arms of 3, 3
    and 9 let the nine-seed arm reach the bar of 8 alone.

      1. FEWER THAN `ARMS_REQUIRED` ARMS PRESENT raises `ValueError` naming the
         arms that are. The plan itself is malformed. It is checked before
         `SEEDS_MINIMUM`, so a lone short arm is reported as the missing arm.
      2. A MAJORITY WHOSE CELLS SPAN FEWER THAN `ARMS_REQUIRED` ARMS returns
         `NO_MAJORITY`, and does NOT raise. The inputs were legal -- every arm
         met `SEEDS_MINIMUM` and there were enough arms -- and the data simply
         did not agree across arms, which is a finding and reaches exit 48 as
         one. The `rule` says the placement held a numeric majority and
         spanned only that many arms, so it cannot be read as the other way to
         `NO_MAJORITY`, `no placement reached`. The tally still carries every
         vote. It applies to every placement, `UNREADABLE` included: that
         verdict is also one arm's finding when only one arm reads it.

    Check 2 counts the arms the winning cells come from, not how many cells
    each contributes. Eight cells of a nine-seed arm and one cell of another
    span two arms and are a verdict.
    """
    if not inputs.cells:
        raise ValueError("reading_headroom needs at least one cell")
    by_arm: dict[str, set[int]] = {}
    for arm, seed in inputs.cells:
        by_arm.setdefault(arm, set()).add(seed)
    if len(by_arm) < ARMS_REQUIRED:
        raise ValueError(
            f"a verdict needs ARMS_REQUIRED={ARMS_REQUIRED} arms; "
            f"got {sorted(by_arm)}"
        )
    short = {arm: sorted(s) for arm, s in by_arm.items() if len(s) < SEEDS_MINIMUM}
    if short:
        raise ValueError(
            f"every arm needs SEEDS_MINIMUM={SEEDS_MINIMUM} seeds before a "
            f"placement is tallied; got {short}"
        )

    tally: dict[str, list[tuple[str, int]]] = {name: [] for name in PLACEMENTS}
    for key, cell in sorted(inputs.cells.items()):
        tally[cell.placement()].append(key)
    frozen = {name: tuple(votes) for name, votes in tally.items()}

    needed = strict_majority(len(inputs.cells))
    winners = [name for name in PLACEMENTS if len(frozen[name]) >= needed]
    assert len(winners) <= 1, (
        f"two placements cannot both hold a strict majority of {len(inputs.cells)} "
        f"cells; got {winners} -- the partition is broken"
    )
    at = f"k = {inputs.decision_k}, h = {inputs.decision_h}"
    if not winners:
        counts = ", ".join(f"{n}={len(frozen[n])}" for n in PLACEMENTS)
        return HeadroomStatus(
            verdict=NO_MAJORITY,
            rule=(
                f"no placement reached {needed} of {len(inputs.cells)} cells at "
                f"{at} ({counts})"
            ),
            tally=frozen,
        )
    verdict = winners[0]
    winning_arms = sorted({arm for arm, _ in frozen[verdict]})
    if len(winning_arms) < ARMS_REQUIRED:
        return HeadroomStatus(
            verdict=NO_MAJORITY,
            rule=(
                f"{verdict} held a numeric majority ({len(frozen[verdict])} of "
                f"{len(inputs.cells)} cells at {at}, strict majority {needed}) "
                f"but spanned only {len(winning_arms)} "
                f"{'arm' if len(winning_arms) == 1 else 'arms'} "
                f"({', '.join(winning_arms)}); ARMS_REQUIRED={ARMS_REQUIRED} "
                f"arms must agree"
            ),
            tally=frozen,
        )
    return HeadroomStatus(
        verdict=verdict,
        rule=(
            f"{verdict} in {len(frozen[verdict])} of {len(inputs.cells)} cells at "
            f"{at} (strict majority {needed}, computed from the cells present)"
        ),
        tally=frozen,
    )


READING_COLUMNS: tuple[str, ...] = (
    "cell", "clusters", "headroom", "skill", "deficit", "share", "placement",
)
READING_WIDTHS: tuple[int, ...] = (22, 10, 30, 30, 30, 9, 12)
"""`_interval` prints 28 characters for any value below 1000 in magnitude:
`+8.3f` three times (24), two brackets (2), a comma (1) and one space (1). The
three interval columns are 30 wide so each ends in a space; at 24 every row ran
12 characters past the header and the columns after the first interval were out
of line."""


def _row(values, widths) -> str:
    return "".join(str(v).ljust(w) for v, w in zip(values, widths))


def _interval(i: Interval) -> str:
    return f"{i.point:+8.3f} [{i.ci_low:+8.3f},{i.ci_high:+8.3f}]"


def format_reading_headroom(
    reading: HeadroomStatus, inputs: HeadroomInputs
) -> str:
    """The table, the verdict, and a legend that is conditioned on the gate.

    EVERY DIRECTIONAL SENTENCE IS CONDITIONAL. M3m shipped "both directions are
    sound" unconditionally in three places its own results refuted -- a
    docstring, the legend written into `burden.txt`, and spec 3.3, which the
    next milestone would have inherited. A sign here means something only
    behind the headroom gate, and the legend says so in the same breath as the
    sign, so the sentence cannot be quoted without its condition.

    The `share` column is a POINT ESTIMATE WITH NO INTERVAL, printed only where
    the gate passed. It needs none: the verdict is two-sided on the two
    differences, so the share is presentation. Where the gate failed it prints
    `--`, never a number, because `headroom` crosses zero inside the reported
    grid on a real cell: `pixel_ae_seed1`'s `headroom(k=45, h)` is 0.575649 at
    h=1 and -0.641206 at h=2, and the share reads +339.5% at h=2 then -335.7%
    at h=3.
    """
    at = f"k = {inputs.decision_k}, h = {inputs.decision_h}"
    lines = [
        f"M3n motion headroom -- the one-step map between copying and perfect, at {at}",
        "",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for (arm, seed), cell in sorted(inputs.cells.items()):
        placement = cell.placement()
        share = (
            f"{cell.skill.point / cell.headroom.point * 100:7.1f}%"
            if placement != "UNREADABLE" else "--"
        )
        lines.append(_row(
            (
                f"{arm}_seed{seed}", cell.clusters,
                _interval(cell.headroom), _interval(cell.skill),
                _interval(cell.deficit), share, placement,
            ),
            READING_WIDTHS,
        ))
    lines += [
        "",
        f"verdict: {reading.verdict}",
        f"rule:    {reading.rule}",
        "",
        "legend",
        "  headroom = hold - floor, what a PERFECT predictor wins over copying.",
        "  skill    = hold - rung,  what the model wins over copying.",
        "  deficit  = rung - floor, M3m's burden -- the distance to perfect.",
        "  skill + deficit == headroom at every step, to float rounding; the hold",
        "  term cancels. The interval ends do not add: each is a percentile of",
        "  its own resamples.",
        "  A cell is read ONLY IF headroom is resolvably above zero. Where it is",
        "  not, a perfect predictor is indistinguishable from a copying one, so",
        "  the sign of skill carries no claim about the model and the share is",
        "  printed as `--` rather than as a number.",
        "  Behind that gate: skill resolvably positive means the one-step map",
        "  beats holding; deficit resolvably positive means it is not yet",
        "  indistinguishable from perfect.",
        f"  Intervals are episode-clustered bootstraps at {CONFIDENCE:.2f} over "
        f"{RESAMPLES} resamples.",
        "",
    ]
    return "\n".join(lines)
