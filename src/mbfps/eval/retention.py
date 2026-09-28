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

from dataclasses import dataclass

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


BASE_R2_FLOOR: float = 0.10
"""The r2 `enc(t)` -> absolute position must exceed for the reading to be taken.

A POSITIVE CONTROL promoted to a gate. M3i's equivalent lived only in its final
review and in no record; its own section 9 provenance note asked a successor to
build it into the measure phase, which is what this is.

The value has to fail only when the instrument is broken, never when a cell is
merely weak. The M3c records' `latent_selection_r2` runs 0.18-0.34, with one
outlier at -0.008 -- and that outlier is on the LATENT, a 32x32 bottleneck,
not on the 2048 continuous floats this control probes. M3i measured `enc(t)` ->
position at +0.291 and +0.165 from 132 rows; this milestone fits ~10,500. So
0.10 sits below every recorded figure and far above zero.
"""


@dataclass(frozen=True)
class RungArm:
    """One arm's rung at one target and horizon, summarised over its seeds.

    `gain` is the seed MEAN, so the printed number describes the arm. `ci_low`
    is the LEAST lower bound across the seeds -- the conservative summary, so a
    rung is never credited with an interval only its luckiest seed achieved, and
    `ci_high` is the GREATEST upper bound so the printed pair is a real interval
    rather than a one-sided stub. None of the three decides anything:
    `seeds_clear` does, and it counts seeds whose OWN lower bound excluded zero.
    `ci_high` is reported only -- the rule is one-sided, so nothing reads it.
    """

    gain: float
    ci_low: float
    ci_high: float
    seeds_clear: int
    seeds_total: int

    def clears(self) -> bool:
        """ONE-SIDED, and that is deliberate.

        A negative gain means appending the block made held-out R^2 WORSE, which
        is noise or selection slack rather than a finding about the
        representation. M3i shipped a two-sided control on the argument that a
        result reliably worse than chance is as broken an instrument as one
        reliably better; its own run refuted that (`## Task 8 results` section 5
        of `docs/superpowers/plans/2026-09-26-mb-fps-m3i-latent-motion.md`), and
        the lesson is applied here rather than re-derived.
        """
        return self.seeds_clear >= SEEDS_REQUIRED


@dataclass(frozen=True)
class BaseControl:
    """One arm's base control: `enc(t)` -> absolute position, as an r2 LEVEL.

    Not a gain. There is nothing to take an increment over -- this is the arm
    the increments are measured against, and the question is only whether it
    reads at all.

    `clears()` reads only `seeds_clear`, never `self.r2` -- deliberately, the
    same division of labour as `RungArm.seeds_clear`: a BUILDER applies the
    threshold and this class only reads the tally it was handed. The threshold
    is `BASE_R2_FLOOR`, and it is applied by the caller that builds this class
    (`scripts/latent_retention.py`'s read phase), which computes
    `seeds_clear=sum(1 for r in levels if r > BASE_R2_FLOOR)` per seed before
    this dataclass ever sees the tally. Nothing in this module compares `r2` to
    `BASE_R2_FLOOR` directly.
    """

    r2: float
    seeds_clear: int
    seeds_total: int

    def clears(self) -> bool:
        return self.seeds_clear >= SEEDS_REQUIRED


@dataclass(frozen=True)
class RetentionInputs:
    """Everything Reading E is decided on.

    `ladder` is `target -> k -> rung -> arm -> RungArm`, four levels because
    every one of them is quantified over in the rule: a rung clears a target if
    it clears at ANY k, in ARMS_REQUIRED arms, each in SEEDS_REQUIRED seeds.
    `rows` carries the scored row count per k so the reading prints the
    conditioning it was taken at rather than leaving it to be recomputed.
    """

    ladder: dict[str, dict[int, dict[str, dict[str, RungArm]]]]
    base: dict[str, BaseControl]
    clusters: int
    rows: dict[int, int]


@dataclass(frozen=True)
class RetentionStatus:
    status: str
    rule: str
    surviving: str | None
    translation_rungs: tuple[str, ...]
    rotation_rungs: tuple[str, ...]
    base_failed: tuple[str, ...]


def rung_arm(gains: list[dict]) -> RungArm:
    """One arm's `RungArm` from its per-seed `probe.gain_from_blocks` dicts.

    THE FINITENESS GUARD LIVES HERE, and it raises. M3i's ledger left this as
    the one note for its successor: `clears_up`, `clears_down` and `leaks` all
    evaluate False on NaN, so a single non-finite cell would read "no clear" and
    "no leak" at once -- moving a verdict toward the wrong status while looking
    like a clean null. A non-finite gain or interval bound is an ERROR about the
    measurement, never a statement about the representation, so it is refused
    at the point where the number first becomes a reading.

    A missing key, an inverted interval (`ci_low > ci_high`), a `gain` outside
    `[ci_low, ci_high]`, or fewer than `SEEDS_REQUIRED` seeds are all refused
    the same way and for the same reason: each would otherwise pass silently
    and read as a null -- an arm with one seed, in particular, can never
    satisfy `seeds_clear >= SEEDS_REQUIRED` and so can never clear, which is a
    refusal wearing the shape of a finding.
    """
    if not gains:
        raise ValueError("a rung needs at least one seed to summarise")
    for index, seed in enumerate(gains):
        missing = [k for k in ("gain", "ci_low", "ci_high") if k not in seed]
        if missing:
            raise ValueError(
                f"seed index {index} is missing {', '.join(sorted(missing))}"
            )
        values = {k: float(seed[k]) for k in ("gain", "ci_low", "ci_high")}
        bad = {k: v for k, v in values.items() if not np.isfinite(v)}
        if bad:
            raise ValueError(
                f"non-finite {', '.join(sorted(bad))} in seed index {index}: {bad}. "
                "A non-finite gain is an error about the measurement, not a "
                "non-clear -- see this function's docstring."
            )
        if values["ci_low"] > values["ci_high"]:
            raise ValueError(
                f"seed index {index} has ci_low {values['ci_low']} > ci_high "
                f"{values['ci_high']}: an inverted interval is not a reading"
            )
        if not values["ci_low"] <= values["gain"] <= values["ci_high"]:
            raise ValueError(
                f"seed index {index} has gain {values['gain']} outside its own "
                f"interval [{values['ci_low']}, {values['ci_high']}]"
            )
    if len(gains) < SEEDS_REQUIRED:
        raise ValueError(
            f"a rung needs at least SEEDS_REQUIRED={SEEDS_REQUIRED} seeds to "
            f"summarise, got {len(gains)}: an under-seeded arm can never clear "
            "and would otherwise read as a silent null"
        )
    return RungArm(
        gain=float(np.mean([g["gain"] for g in gains])),
        ci_low=float(min(float(g["ci_low"]) for g in gains)),
        ci_high=float(max(float(g["ci_high"]) for g in gains)),
        seeds_clear=sum(1 for g in gains if float(g["ci_low"]) > 0.0),
        seeds_total=len(gains),
    )


def rung_clears_at(inputs: RetentionInputs, target: str, rung: str) -> tuple[int, ...]:
    """The horizons at which `rung` clears `target`, in `K_REPORTED` order.

    A tuple rather than a boolean because the rule is a DISJUNCTION over
    `K_REPORTED` (spec 2.3) and the reading's rule text names the horizons it
    cleared at. Empty means the rung did not clear anywhere.
    """
    cleared = []
    for k in K_REPORTED:
        arms = inputs.ladder[target][k][rung]
        if sum(1 for arm in arms.values() if arm.clears()) >= ARMS_REQUIRED:
            cleared.append(k)
    return tuple(cleared)


def _cleared_rungs(inputs: RetentionInputs, target: str) -> dict[str, tuple[int, ...]]:
    """Every rung that cleared `target`, mapped to its horizons, in RUNGS order."""
    found = {}
    for rung in RUNGS:
        at = rung_clears_at(inputs, target, rung)
        if at:
            found[rung] = at
    return found


def _validate_family_shape(inputs: RetentionInputs) -> None:
    """Refuse rather than read a status from an under-populated ladder or base.

    A missing (target, k) or (target, k, rung) KEY is already loud -- a
    KeyError -- but a short or empty ARM dict at one is silent: it reads as
    "did not clear" (`rung_clears_at` needs `ARMS_REQUIRED` of the arms
    present to clear, so fewer arms only makes clearing harder, never
    impossible-to-read) and the silence can surface as a pre-registered
    finding manufactured from insufficient data rather than a refusal.
    `scripts/latent_retention.py` guards the plan's arms x seeds shape too,
    but that is defence in depth, not a reason to skip the guard here: this
    function produces a pre-registered finding and must refuse to produce one
    from insufficient data regardless of caller.
    """
    for target in TARGETS:
        for k in K_REPORTED:
            for rung in RUNGS:
                arms = inputs.ladder[target][k][rung]
                if len(arms) != RETENTION_FAMILY:
                    raise ValueError(
                        f"ladder[{target!r}][{k}][{rung!r}] has {len(arms)} arm(s) "
                        f"({sorted(arms)}), expected exactly RETENTION_FAMILY="
                        f"{RETENTION_FAMILY}"
                    )
    if len(inputs.base) != RETENTION_FAMILY:
        raise ValueError(
            f"base has {len(inputs.base)} arm(s) ({sorted(inputs.base)}), expected "
            f"exactly RETENTION_FAMILY={RETENTION_FAMILY}"
        )


def reading_retention(inputs: RetentionInputs) -> RetentionStatus:
    """Reading E: the last rung on the path at which observed motion survives.

    Precedence, and it is the point (spec 3.2). `UNRESOLVED_BASE` is the one
    true control failure and outranks every result -- a null on displacement
    means nothing if the current frame cannot say where it is. Then the ladder,
    highest surviving rung first, so the status IS the lever: a z-bearing rung
    means the bottleneck kept it and M3i is partially overturned; `h` alone
    means the bottleneck destroyed it; `two_frame` alone means the RSSM
    discarded available information; nothing on translation but something on
    rotation means translation is below the encoder's spatial resolution.

    THE LATENT RUNGS ARE CHECKED BEFORE `two_frame`, and `UNRESOLVED_MOTION` IS
    LAST. `h` integrates the action sequence, which two frames do not contain,
    so a latent rung can legitimately clear where `two_frame` does not. An
    earlier draft of the spec put `UNRESOLVED_MOTION` second, refusing whenever
    `two_frame` was silent on both targets -- which would have refused precisely
    that case. `UNRESOLVED_MOTION` therefore requires the WHOLE ladder to be
    silent on BOTH targets, and it is not a control failure but an empty
    measurement, which is why it sorts with the readings rather than ahead of
    them.

    A suppressed reading reports no rungs at all rather than reporting them
    beside a warning nobody reads.
    """
    _validate_family_shape(inputs)
    base_failed = tuple(
        sorted(arm for arm, control in inputs.base.items() if not control.clears())
    )
    holding = len(inputs.base) - len(base_failed)
    if holding < ARMS_REQUIRED:
        return RetentionStatus(
            status="UNRESOLVED_BASE",
            rule=(
                f"enc(t) -> absolute position cleared r2 {BASE_R2_FLOOR:.2f} in only "
                f"{holding} of {len(inputs.base)} arms ({', '.join(base_failed)} failed); "
                f"the current frame cannot linearly say where it is, so the instrument "
                f"is broken and no reading is taken"
            ),
            surviving=None, translation_rungs=(), rotation_rungs=(), base_failed=base_failed,
        )

    translation = _cleared_rungs(inputs, "translation")
    rotation = _cleared_rungs(inputs, "rotation")
    t_rungs = tuple(translation)
    r_rungs = tuple(rotation)

    def at(rung: str) -> str:
        return ", ".join(f"k = {k}" for k in translation[rung])

    for rung in Z_BEARING_RUNGS:
        if rung in translation:
            return RetentionStatus(
                status="MOTION_RETAINED",
                rule=(
                    f"the {rung} rung adds displacement the current frame lacks at "
                    f"{at(rung)}, in at least {ARMS_REQUIRED} arms and {SEEDS_REQUIRED} "
                    f"seeds each; the bottlenecked latent retains motion it has observed, "
                    f"so M3i's NO_MOTION was about forward prediction rather than about "
                    f"the representation's content"
                ),
                surviving=rung, translation_rungs=t_rungs, rotation_rungs=r_rungs,
                base_failed=base_failed,
            )
    if "deterministic" in translation:
        return RetentionStatus(
            status="BOTTLENECK_LOSS",
            rule=(
                f"the deterministic rung adds displacement at {at('deterministic')} but "
                f"no z-bearing rung does; h carries motion and the 32x32 categorical "
                f"bottleneck destroys it"
            ),
            surviving="deterministic", translation_rungs=t_rungs, rotation_rungs=r_rungs,
            base_failed=base_failed,
        )
    if "two_frame" in translation:
        return RetentionStatus(
            status="MOTION_DISCARDED",
            rule=(
                f"two real frames add displacement at {at('two_frame')} and no part of "
                f"the latent does; the information is available and the RSSM discards it, "
                f"which is what a loss whose reconstruction target is the frame just seen "
                f"would predict"
            ),
            surviving="two_frame", translation_rungs=t_rungs, rotation_rungs=r_rungs,
            base_failed=base_failed,
        )
    if rotation:
        return RetentionStatus(
            status="TRANSLATION_UNRESOLVED",
            rule=(
                f"no rung adds translation at any of k = "
                f"{', '.join(str(k) for k in K_REPORTED)}, but "
                f"{', '.join(r_rungs)} adds ROTATION; translation is below what "
                f"112x112 frozen single-frame features resolve, so the lever is the "
                f"encoder's input rather than the loss"
            ),
            surviving=None, translation_rungs=t_rungs, rotation_rungs=r_rungs,
            base_failed=base_failed,
        )
    return RetentionStatus(
        status="UNRESOLVED_MOTION",
        rule=(
            f"no rung adds either translation or rotation at any of k = "
            f"{', '.join(str(k) for k in K_REPORTED)}; a linear read detects no motion "
            f"anywhere on the path, so the measurement is empty and no lever is chosen"
        ),
        surviving=None, translation_rungs=(), rotation_rungs=(), base_failed=base_failed,
    )


# The reading table's columns, in printed order. Declared once so a test can
# assert the header against THIS and the rows against the same offsets -- the
# caption-column pairing that three shipped defects on this project all broke.
READING_COLUMNS: tuple[str, ...] = (
    "rung", "arm", "gain", "ci_low", "ci_high", "seeds", "clears",
)

# Field widths for READING_COLUMNS, in the same order. Kept beside the names so
# the header and the rows -- two separate f-strings -- cannot drift apart, and
# sized so a realistic value cannot EQUAL its width and glue onto the previous
# column with no separator (the arm-name-overflow defect this project shipped
# once already: "random_vit" is 10 characters, so its field is 13).
READING_WIDTHS: tuple[int, ...] = (15, 13, 10, 10, 10, 8, 8)

LADDER_COLUMNS: tuple[str, ...] = (
    "target", "rung", "k", "mean gain", "arms", "seeds", "clears",
)
LADDER_WIDTHS: tuple[int, ...] = (13, 15, 6, 12, 8, 8, 8)


def _row(values, widths) -> str:
    return "  " + "".join(
        f"{value:>{width}}" for value, width in zip(values, widths, strict=True)
    )


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _seed_arms(arms: dict[str, RungArm]) -> tuple[int, int]:
    """`(arms clearing, seeds clearing across all of them)`."""
    return (
        sum(1 for arm in arms.values() if arm.clears()),
        sum(arm.seeds_clear for arm in arms.values()),
    )


def format_reading_retention(reading: RetentionStatus, inputs: RetentionInputs) -> str:
    """Reading E as it is printed and written to `retention.txt`, byte for byte.

    The caption names the statistic, the target, the horizon, the one-sided rule
    and the cluster count, because a table whose header does not say what its
    columns hold is how this project has shipped a wrong number three times.
    The two controls print beside the verdict rather than in a companion table:
    the base control is a GATE, and whether rotation read is a different LEVER
    from whether nothing read, so both belong on the face of the reading.
    """
    k = DECISION_K
    rows = inputs.ladder["translation"][k]
    lines = [
        f"--- Reading E: what each rung adds over enc(t) about translation "
        f"already observed, at k = {k} "
        f"(gain over enc(t) in mean R^2, one-sided: clears when ci_low > 0 in "
        f"{SEEDS_REQUIRED} of 3 seeds and {ARMS_REQUIRED} of 3 arms); "
        f"{inputs.rows.get(k, 0)} rows over {inputs.clusters} clusters ---",
        _row(READING_COLUMNS, READING_WIDTHS),
    ]
    for rung in RUNGS:
        for arm_name in sorted(rows[rung]):
            arm = rows[rung][arm_name]
            lines.append(_row((
                rung, arm_name,
                f"{arm.gain:+.4f}", f"{arm.ci_low:+.4f}", f"{arm.ci_high:+.4f}",
                f"{arm.seeds_clear}/{arm.seeds_total}", _yes(arm.clears()),
            ), READING_WIDTHS))
    base = "  base control (enc(t) -> absolute position, must clear r2 " + (
        f"{BASE_R2_FLOOR:.2f}): "
    ) + ", ".join(
        f"{name} r2={inputs.base[name].r2:+.3f} "
        f"{inputs.base[name].seeds_clear}/{inputs.base[name].seeds_total}"
        for name in sorted(inputs.base)
    )
    rotation = "  rotation control (positive; reads where translation cannot): " + (
        ", ".join(reading.rotation_rungs) if reading.rotation_rungs
        else "no rung cleared rotation at any horizon"
    )
    lines += [
        base,
        rotation,
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}",
    ]
    return "\n".join(lines)


def format_ladder(inputs: RetentionInputs) -> str:
    """Every rung at every horizon for both targets -- the disjunction, in full.

    Reported so a reader can check the rule rather than take it on trust. No row
    decides on its own and the caption says so: M3i's companion table printed
    `decides: no` beside every row because exactly one horizon DID decide there,
    and a reader carrying that habit across would misread this one.
    """
    lines = [
        f"--- The ladder: gain over enc(t) at every horizon, both targets "
        f"(translation decides, rotation controls; any horizon counts -- the "
        f"rule is a disjunction over k = "
        f"{', '.join(str(k) for k in K_REPORTED)}) ---",
        _row(LADDER_COLUMNS, LADDER_WIDTHS),
    ]
    for target in TARGETS:
        for rung in RUNGS:
            for k in K_REPORTED:
                arms = inputs.ladder[target][k][rung]
                clearing, seeds = _seed_arms(arms)
                lines.append(_row((
                    target, rung, str(k),
                    f"{np.mean([a.gain for a in arms.values()]):+.4f}",
                    f"{clearing}/{len(arms)}",
                    f"{seeds}/{sum(a.seeds_total for a in arms.values())}",
                    _yes(clearing >= ARMS_REQUIRED),
                ), LADDER_WIDTHS))
    return "\n".join(lines)
