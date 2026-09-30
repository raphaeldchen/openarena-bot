"""M3k: is the past frame's advantage over the recurrent state information, or
feature count?

M3j left two candidate levers and one confounded comparison. Its objective-lever
argument rests on `two_frame` (+0.03489) beating `deterministic` (+0.02071) at
k = 15 in 8 of 9 cells -- but those blocks are 2048 features against 512.
Measured during M3k's design, a random lift of `h` from 512 to 2048 columns,
adding ZERO information and leaving the rank at 512, buys +0.01355 and +0.01377
of gain on two cells. That is larger than the +0.01418 pooled gap the argument
depends on.

Three passes of one ladder settle it. `down` projects every block to 512,
matching count AND rank, and decides. `up` lifts every block to 2048, matching
count only, and calibrates how much count alone was worth. Neither suffices
alone: a random projection DESTROYS information, so a rung losing in `down`
might have lost to the projection; the lift preserves information exactly but
leaves `enc(t-k)`'s richer rank intact.

Everything here is pure: arrays in, arrays and readings out. No torch, no I/O,
no record schema. `scripts/latent_width.py` owns all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mbfps.eval.retention import ARMS_REQUIRED, BASE_R2_FLOOR, BaseControl, SEEDS_REQUIRED
from mbfps.models.rssm import RSSMConfig

PASSES: tuple[str, ...] = ("shipped", "down", "up")
DOWN_WIDTH: int = 512
UP_WIDTH: int = 2048

TARGET_WIDTH: dict[str, int | None] = {
    "shipped": None, "down": DOWN_WIDTH, "up": UP_WIDTH,
}

CONTRAST_K: int = 15
"""The horizon Reading F decides at.

DELIBERATELY NOT NAMED `DECISION_K`, which is 4 in `retention` and means the
horizon Reading E decides at. Both modules are imported by the same script, and
one name carrying two meanings is a misreading waiting to happen. 15 is where
M3j's objective-lever claim lives and nowhere else -- `two_frame` loses at k = 1
and k = 4 even before matching -- so testing the claim at its own horizon is the
honest test, and it avoids the incoherence a disjunction would create
(PAST_FRAME_AHEAD at one horizon and RECURRENT_AHEAD at another are
contradictory, not complementary).
"""

PROJECTION_SEED: int = 0

_Z_DIM = RSSMConfig.z_cats * RSSMConfig.z_classes
RUNG_WIDTH: dict[str, int] = {
    "two_frame": RSSMConfig.embed_dim,
    "deterministic": RSSMConfig.h_dim,
    "stochastic": _Z_DIM,
    "full": RSSMConfig.h_dim + _Z_DIM,
}
"""Each rung's native block width, DERIVED from the model's own config rather
than re-spelled. A hardcoded 512 here would silently disagree with the model if
`RSSMConfig` ever changed, and every pass would match the wrong width."""

ANCHOR: dict[str, str] = {"down": "deterministic", "up": "two_frame"}
"""Each pass's known-answer rung: the one whose native width already equals the
pass's target, so the pass leaves it untouched and its gain must reproduce the
shipped one EXACTLY. `shipped` projects nothing and so has no anchor. Pinned
against `RUNG_WIDTH` by test, so this cannot drift away from the widths."""


@lru_cache(maxsize=None)
def _draw_projection(native: int, target: int, seed: int):
    """The cached draw, keyed POSITIONALLY so the cache cannot split.

    `lru_cache` keys on the literal call signature, not on resolved defaults,
    so `projection(2048, 512)` and `projection(2048, 512, seed=0)` would land
    on two entries holding two equal-valued copies of the same matrix. Keeping
    the cache on a positional-only helper makes the "drawn exactly once"
    guarantee hold however the public function is called.
    """
    if native == target:
        return None
    rng = np.random.default_rng([seed, native, target])
    matrix = rng.normal(size=(native, target)) / np.sqrt(target)
    matrix.flags.writeable = False
    return matrix


def projection(native: int, target: int, *, seed: int = PROJECTION_SEED):
    """The fixed matrix taking a `native`-wide block to `target`, or `None`.

    `None` means no projection is needed -- NOT an identity matrix. The anchor
    path must do no matmul at all, so it cannot drift by a floating-point ulp
    and the anchor test can demand exact equality rather than approximate.

    Gaussian scaled `1/sqrt(target)`, the standard Johnson-Lindenstrauss form,
    which approximately preserves inner products. Drawn from a fixed seed keyed
    on `(native, target)` and therefore SHARED ACROSS ALL NINE CELLS: a matrix
    redrawn per call would make no two cells comparable and would let an arm win
    on a lucky draw.

    Memoized with `lru_cache`, so the matrix is drawn exactly once per
    `(native, target, seed)` and the SAME array object is returned to every
    caller thereafter -- Task 5's nested loop over rungs x passes x horizons x
    targets would otherwise reallocate matrices up to 1536x2048 float64 (24 MB,
    the largest shape actually reachable: 2048x2048 would be the `up` anchor,
    which returns `None` before allocating) on every call. The returned array is
    marked read-only (`flags.writeable = False`) so one caller cannot mutate the
    shared cached array out from under another.
    """
    return _draw_projection(native, target, seed)


def pass_block(block: np.ndarray, pass_name: str, *, seed: int = PROJECTION_SEED):
    """`block` as `pass_name` sees it.

    `shipped` casts to float64 and returns it -- the values are unchanged, but
    the dtype may not be, so every pass hands the probe the same dtype (the
    cast is lossless for float32 input, so the anchors' bit-identity is
    unaffected). `down` projects to 512, matching count AND rank. `up` lifts
    to 2048, matching count while LEAVING THE RANK at the block's native width
    -- that is what makes `up` a pure width control rather than a capacity
    change, and it is pinned by a rank test.
    """
    if pass_name not in TARGET_WIDTH:
        raise ValueError(f"unknown pass {pass_name!r}; expected one of {PASSES}")
    target = TARGET_WIDTH[pass_name]
    block = np.asarray(block, dtype=np.float64)
    if target is None:
        return block
    matrix = projection(block.shape[1], target, seed=seed)
    return block if matrix is None else block @ matrix


@dataclass(frozen=True)
class ContrastArm:
    """One arm's k = 15 contrast, summarised over its seeds, BOTH ways.

    `seeds_down` sits beside `seeds_up` because M3h shipped a Reading N table
    that printed only the up tally while its verdict was read from the down
    one: every row said `0/3` under a verdict asserting "3 of 3 arms", and the
    natural misreading was the opposite of the truth.

    `contrast` is the seed mean; `ci_low` the least lower bound and `ci_high`
    the greatest upper bound across seeds -- the conservative summary each way,
    so an arm is never credited with an interval only its luckiest seed reached.
    Neither decides: the tallies do.
    """

    contrast: float
    ci_low: float
    ci_high: float
    seeds_up: int
    seeds_down: int
    seeds_total: int

    def clears_up(self) -> bool:
        return self.seeds_up >= SEEDS_REQUIRED

    def clears_down(self) -> bool:
        return self.seeds_down >= SEEDS_REQUIRED


@dataclass(frozen=True)
class ContrastInputs:
    """Everything Reading F is decided on. `anchors` maps each projecting pass
    to whether its known-answer rung reproduced its shipped gain; its keys are
    drawn from `PASSES` and must include `down`, which `reading_contrast`
    enforces."""

    arms: dict[str, ContrastArm]
    base: dict[str, BaseControl]
    anchors: dict[str, bool]
    clusters: int
    rows: int


@dataclass(frozen=True)
class ContrastStatus:
    status: str
    rule: str
    arms_up: tuple[str, ...]
    arms_down: tuple[str, ...]
    base_failed: tuple[str, ...]
    anchors_broken: tuple[str, ...]


_SEED_KEYS: tuple[str, ...] = ("contrast", "ci_low", "ci_high")


def contrast_arm(contrasts: list[dict]) -> ContrastArm:
    """One arm's `ContrastArm` from its per-seed `probe.contrast_from_blocks`
    dicts.

    THE FINITENESS GUARD RAISES, and this is M3i's ledger note still binding:
    every comparison against NaN is False, so a non-finite contrast would read
    "not up" AND "not down" at once. That lands on INDISTINGUISHABLE -- a
    pre-registered finding asserted from a broken number, and the direction of
    the silence is toward a status the project will act on.

    A missing key, an inverted interval (`ci_low > ci_high`), a `contrast`
    outside its own `[ci_low, ci_high]`, or fewer than `SEEDS_REQUIRED` seeds
    are refused the same way and for the same reason, as `retention.rung_arm`
    refuses them one reading over (there for `gain`). A `contrast` its own
    interval does not contain would print a row whose number contradicts its
    interval and its `clears` value. The inverted interval matters more here
    than there: Reading F is two-sided, and an inverted interval (`ci_low > 0`
    AND `ci_high < 0` at once) would count the SAME seed into `seeds_up` and
    `seeds_down`, letting one arm clear both ways from fewer than
    `2 * SEEDS_REQUIRED` seeds. Refusing it here makes that state unreachable,
    not merely untested. It does NOT stop two DIFFERENT seeds from disagreeing,
    so an arm can still land up in some seeds and down in others;
    `reading_contrast` refuses that arm itself, since it sees every arm however
    it was built. Each field is read and coerced with `float()` exactly once,
    and every number below is taken from that one read.
    """
    if len(contrasts) < SEEDS_REQUIRED:
        raise ValueError(
            f"an arm needs at least SEEDS_REQUIRED={SEEDS_REQUIRED} seeds to "
            f"summarise, got {len(contrasts)}"
        )
    seeds = []
    for index, seed in enumerate(contrasts):
        missing = [k for k in _SEED_KEYS if k not in seed]
        if missing:
            raise ValueError(
                f"seed index {index} is missing {', '.join(sorted(missing))}"
            )
        values = {k: float(seed[k]) for k in _SEED_KEYS}
        bad = {k: v for k, v in values.items() if not np.isfinite(v)}
        if bad:
            raise ValueError(
                f"non-finite {', '.join(sorted(bad))} in seed index {index}: "
                f"{bad}. A non-finite contrast is an error about the "
                "measurement, not an INDISTINGUISHABLE reading."
            )
        if values["ci_low"] > values["ci_high"]:
            raise ValueError(
                f"seed index {index} has ci_low {values['ci_low']} > ci_high "
                f"{values['ci_high']}: an inverted interval is not a reading, "
                "and would count this seed as clearing up AND down"
            )
        if not values["ci_low"] <= values["contrast"] <= values["ci_high"]:
            raise ValueError(
                f"seed index {index} has contrast {values['contrast']} outside "
                f"its own interval [{values['ci_low']}, {values['ci_high']}]: "
                "the number would contradict the interval its tally is read from"
            )
        seeds.append(values)
    return ContrastArm(
        contrast=float(np.mean([v["contrast"] for v in seeds])),
        ci_low=min(v["ci_low"] for v in seeds),
        ci_high=max(v["ci_high"] for v in seeds),
        seeds_up=sum(1 for v in seeds if v["ci_low"] > 0.0),
        seeds_down=sum(1 for v in seeds if v["ci_high"] < 0.0),
        seeds_total=len(seeds),
    )


def _common_seeds_total(arms: dict) -> int:
    """The arms' one shared `seeds_total`, or a refusal.

    Arms that disagree on it are refused rather than reconciled: the caption's
    "N of M seeds" would be true of some rows and false of others, and picking
    one silently would print a rule the table does not follow.

    A shared `seeds_total` BELOW `SEEDS_REQUIRED` is refused too. No arm could
    then clear in `SEEDS_REQUIRED` seeds, so every tally would be empty and the
    reading would fall through to INDISTINGUISHABLE -- a pre-registered finding
    asserted from too few seeds, the same silence `reading_contrast` refuses one
    level up for too few arms. (`contrast_arm` already refuses it one level down,
    so only a hand-built `ContrastArm` reaches here.) There is deliberately NO
    upper bound: a run with more seeds is a legitimate run, and the one thing a
    high seed count makes possible -- an arm clearing both ways -- is refused
    where it actually arises, in `reading_contrast`, not through this proxy.
    """
    if not arms:
        raise ValueError(
            "there are no arms, so there is no common seeds_total to read a "
            "verdict or print a caption from"
        )
    counts = {arm.seeds_total for arm in arms.values()}
    if len(counts) != 1:
        per_arm = {name: arm.seeds_total for name, arm in sorted(arms.items())}
        raise ValueError(
            "the arms must share one seeds_total for the "
            f"'{SEEDS_REQUIRED} of N seeds' rule to be true of every row, "
            f"got {per_arm}"
        )
    (total,) = counts
    if total < SEEDS_REQUIRED:
        raise ValueError(
            f"seeds_total={total} is below SEEDS_REQUIRED={SEEDS_REQUIRED}: no arm "
            "could clear in that many seeds, so INDISTINGUISHABLE would be a "
            "pre-registered finding asserted from too few seeds"
        )
    return total


def reading_contrast(inputs: ContrastInputs) -> ContrastStatus:
    """Reading F: at equal block width, does the past frame still beat the
    recurrent state?

    Precedence (spec 3.2). `UNRESOLVED_BASE` outranks every result -- a
    comparison means nothing if the current frame cannot say where it is. Then
    `UNRESOLVED_ANCHOR`, but ONLY for the `down` pass: Reading F is taken on
    `down` alone, so a broken `up` anchor suppresses the width-bias companion
    and is reported, without suppressing the reading. Letting a companion that
    decides nothing veto a verdict that does would be the wrong trade.

    TWO-SIDED, and that departs from Reading E with a reason. M3j's gain was
    one-sided because a negative gain meant noise. Here `deterministic` beating
    `two_frame` at equal width is a positive finding about what `h` retains, not
    an absence.

    The two directions COULD both clear, and then PAST_FRAME_AHEAD would win
    only because its `if` is written first. Five statuses in this precedence is
    a constraint on this reading, so the ambiguity cannot be a sixth status, and
    an incoherent arm cannot be given a `"both"` label that lets it still vote:
    it is REFUSED, and refused where it actually arises. There are two such
    places, from two different causes, and neither implies the other:

      * PER ARM. With `2 * SEEDS_REQUIRED` seeds one arm can clear up in two
        seeds and down in two others, and so sit in BOTH lists. `contrast_arm`'s
        inverted-interval guard stops one seed counting both ways; it says
        nothing about two different seeds disagreeing. An arm saying "in two
        seeds the past frame won, in two others the recurrent state won" is not
        a coherent per-arm reading -- it is an error about the measurement, not
        a finding, the same discipline `contrast_arm` applies to a non-finite
        bound, an inverted interval and under-seeding. It is refused HERE rather
        than in `contrast_arm` because this function sees every arm however it
        was built, so the two refusals cannot drift apart.
      * IN THE AGGREGATE. With `2 * ARMS_REQUIRED` arms, two DISJOINT sets of
        arms can each reach the bar with every arm perfectly one-sided and no
        arm incoherent. After `up` and `down` are tallied, if both hold
        `ARMS_REQUIRED` arms, it raises.

    Both are the exact condition and need no assumption about the study's shape
    -- not its 3 arms, not its 3 seeds -- so a run with more of either is read
    when it is unambiguous and refused only when it is not. Neither bounds
    `seeds_total`: `--seeds 0 1 2 3` is a legitimate run. The study's own 3 x 3
    shape cannot reach either (2 + 2 > 3 arms, and 2 * 2 > 3 seeds), which
    `test_up_and_down_cannot_both_clear_with_three_arms_of_three_seeds` pins.

    BOTH sit AFTER the base and anchor gates, and that ordering is deliberate: a
    failed base or a broken `down` anchor takes no reading at all, so there is
    no verdict for the if-order to decide, and refusing there would turn a
    legitimate UNRESOLVED_BASE or UNRESOLVED_ANCHOR record into a crash.

    `base_failed` is REPORTED in every branch, including the ones that go on to
    take a reading. `ARMS_REQUIRED` of the arms holding is enough to proceed, so
    one arm can fail its base control while its contrast still votes in the
    tallies below; the record must say so rather than read "nothing failed".
    Gating the verdict is not hiding the fact -- the same reason `anchors_broken`
    is reported in all five branches.

    Refuses (raises) an `inputs` that cannot support a reading at all, rather
    than reading it: fewer than `ARMS_REQUIRED` arms would return
    INDISTINGUISHABLE from zero or one measurement; an `anchors` with no `down`
    entry would let the one gate on this verdict fail open; an `anchors` key
    with no anchor -- `shipped`, which projects nothing, or a key outside
    `PASSES` entirely -- would be reported in `anchors_broken` and printed
    beside the real anchors as though it had been checked; a `base` that does
    not name exactly the arms would let an arm vote with no base control at all,
    and leave `base_failed` reading "nothing failed" for it; arms that disagree
    on `seeds_total`, or share one below `SEEDS_REQUIRED`, have no true
    "N of M seeds" rule; and (after the base and anchor gates) an arm clearing
    both ways, or `up` and `down` both clearing in `ARMS_REQUIRED` arms, are the
    two ambiguities described above.
    """
    if len(inputs.arms) < ARMS_REQUIRED:
        raise ValueError(
            f"Reading F needs at least ARMS_REQUIRED={ARMS_REQUIRED} arms to "
            f"read, got {len(inputs.arms)}: INDISTINGUISHABLE would be a "
            "pre-registered finding asserted from too few measurements"
        )
    if "down" not in inputs.anchors:
        raise ValueError(
            "inputs.anchors has no 'down' entry; the down anchor is the only "
            "thing that gates Reading F, so a missing one is an error about the "
            f"measurement, not a pass (got {sorted(inputs.anchors)})"
        )
    unanchored = sorted(set(inputs.anchors) - set(ANCHOR))
    if unanchored:
        raise ValueError(
            f"inputs.anchors names {unanchored}, which have no anchor "
            f"(expected keys drawn from {sorted(ANCHOR)}): `shipped` projects "
            "nothing and so has no rung to reproduce, and a key outside PASSES "
            "names no pass at all -- either would be reported in "
            "anchors_broken and printed beside the real anchors as though it "
            "had been checked, so it is an error about the measurement, not an "
            "extra entry"
        )
    if set(inputs.base) != set(inputs.arms):
        no_base = sorted(set(inputs.arms) - set(inputs.base))
        no_arm = sorted(set(inputs.base) - set(inputs.arms))
        raise ValueError(
            "inputs.base must name exactly the arms in inputs.arms, or an arm "
            "votes with no base control and the base gate counts the wrong arms; "
            f"arms with no base control: {no_base}; base controls with no arm: "
            f"{no_arm}"
        )
    _common_seeds_total(inputs.arms)  # for its refusals only; the total is unused here
    base_failed = tuple(
        sorted(a for a, control in inputs.base.items() if not control.clears())
    )
    holding = len(inputs.base) - len(base_failed)
    anchors_broken = tuple(sorted(p for p, ok in inputs.anchors.items() if not ok))

    if holding < ARMS_REQUIRED:
        return ContrastStatus(
            status="UNRESOLVED_BASE",
            rule=(
                f"enc(t) -> position cleared r2 {BASE_R2_FLOOR:.2f} in only {holding} "
                f"of {len(inputs.base)} arms ({', '.join(base_failed)} failed); the "
                f"current frame cannot linearly say where it is, so no comparison "
                f"between blocks means anything and no reading is taken"
            ),
            arms_up=(), arms_down=(), base_failed=base_failed,
            anchors_broken=anchors_broken,
        )
    if "down" in anchors_broken:
        return ContrastStatus(
            status="UNRESOLVED_ANCHOR",
            rule=(
                f"the down pass's anchor rung ({ANCHOR['down']}) did not reproduce its "
                f"shipped gain, so the projection plumbing is wrong and the pass "
                f"Reading F is taken on is unreadable; no reading is taken"
            ),
            arms_up=(), arms_down=(), base_failed=base_failed,
            anchors_broken=anchors_broken,
        )

    # Past both gates a reading IS taken, so an ambiguity is now a verdict the
    # if-order would decide -- and both kinds are refused, per arm first.
    incoherent = sorted(
        a for a, arm in inputs.arms.items() if arm.clears_up() and arm.clears_down()
    )
    if incoherent:
        detail = ", ".join(
            f"{a} (up {inputs.arms[a].seeds_up}/{inputs.arms[a].seeds_total}, "
            f"down {inputs.arms[a].seeds_down}/{inputs.arms[a].seeds_total})"
            for a in incoherent
        )
        raise ValueError(
            f"incoherent arm(s) clearing BOTH ways, in at least "
            f"SEEDS_REQUIRED={SEEDS_REQUIRED} seeds each: {detail}. An arm "
            "reading 'in some seeds the past frame won, in others the recurrent "
            "state won' is not a "
            "per-arm reading; it is an error about the measurement, not a finding "
            "-- as a non-finite bound, an inverted interval and under-seeding are "
            "in contrast_arm. It would otherwise be counted in the direction "
            "tested first and printed under that one label. Refused rather than "
            "read, and not given a sixth status or a 'both' label that lets it "
            "still vote"
        )

    up = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_up()))
    down = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_down()))

    if len(up) >= ARMS_REQUIRED and len(down) >= ARMS_REQUIRED:
        raise ValueError(
            f"up clears in {len(up)} arms ({', '.join(up)}) AND down clears in "
            f"{len(down)} arms ({', '.join(down)}), each at the bar of "
            f"ARMS_REQUIRED={ARMS_REQUIRED}: the reading is ambiguous, and the "
            "precedence would settle it by the order of its checks -- "
            "PAST_FRAME_AHEAD because it is written first -- not by the data. "
            "Refused rather than read; a study with this many arms or seeds "
            "must re-register the rule rather than inherit an ambiguous one"
        )
    if len(up) >= ARMS_REQUIRED:
        return ContrastStatus(
            status="PAST_FRAME_AHEAD",
            rule=(
                f"at equal block width the past frame still beats the recurrent state "
                f"at k = {CONTRAST_K} in {len(up)} of {len(inputs.arms)} arms "
                f"({', '.join(up)}), each in at least {SEEDS_REQUIRED} seeds; M3j's "
                f"objective-lever argument survives the width matching"
            ),
            arms_up=up, arms_down=down, base_failed=base_failed,
            anchors_broken=anchors_broken,
        )
    if len(down) >= ARMS_REQUIRED:
        return ContrastStatus(
            status="RECURRENT_AHEAD",
            rule=(
                f"at equal block width the recurrent state beats the past frame at "
                f"k = {CONTRAST_K} in {len(down)} of {len(inputs.arms)} arms "
                f"({', '.join(down)}); M3j's objective-lever argument is refuted and "
                f"its k = 15 result was width"
            ),
            arms_up=up, arms_down=down, base_failed=base_failed,
            anchors_broken=anchors_broken,
        )
    # This gloss says only what INDISTINGUISHABLE licenses. It must NOT say
    # "was width" as RECURRENT_AHEAD's does: that claims width EXPLAINS the whole
    # gap, which contradicts the clause that follows ("could not tell ... not the
    # same as ruling one out") and which no result of this shape supports. The
    # spec pre-registered the overclaiming wording at 3.3 while 4 said
    # `INDISTINGUISHABLE` licenses no positive claim; 4 governs. Only this
    # English changed -- not the rule, the thresholds or the precedence.
    return ContrastStatus(
        status="INDISTINGUISHABLE",
        rule=(
            f"neither direction clears in {ARMS_REQUIRED} arms at k = {CONTRAST_K}; "
            f"most of M3j's k = 15 gap is attributable to width, with a residual that "
            f"is not zero, and the bottleneck lever stands alone BY DEFAULT rather "
            f"than by evidence -- we could not tell the two blocks apart, which is "
            f"not the same as ruling one out"
        ),
        arms_up=up, arms_down=down, base_failed=base_failed,
        anchors_broken=anchors_broken,
    )


READING_COLUMNS: tuple[str, ...] = (
    "arm", "contrast", "ci_low", "ci_high", "up", "dn", "clears",
)
READING_WIDTHS: tuple[int, ...] = (13, 12, 11, 11, 7, 7, 9)


def format_reading_contrast(reading: ContrastStatus, inputs: ContrastInputs) -> str:
    """Reading F as it is printed and written to `width.txt`, byte for byte.

    Every number in the caption comes from a constant or from `inputs`, never a
    literal -- this project has shipped a caption that disagreed with its own
    columns three times. The seed count is the arms' COMMON `seeds_total`; arms
    that disagree on it are refused, because the caption's "N of M seeds" would
    then be true of some rows and false of others, and picking one silently
    would print a rule the table does not follow.
    """
    seeds_total = _common_seeds_total(inputs.arms)
    lines = [
        f"--- Reading F: at EQUAL block width ({DOWN_WIDTH}), does two_frame still beat "
        f"deterministic on translation at k = {CONTRAST_K}? "
        f"(difference of joint R^2; the shared base cancels; two-sided, clears when "
        f"an interval excludes 0 in {SEEDS_REQUIRED} of {seeds_total} seeds and "
        f"{ARMS_REQUIRED} of {len(inputs.arms)} arms); "
        f"{inputs.rows} rows over {inputs.clusters} clusters ---",
        "  " + "".join(
            f"{name:>{width}}"
            for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True)
        ),
    ]
    for name in sorted(inputs.arms):
        arm = inputs.arms[name]
        if arm.clears_up() and arm.clears_down():
            # Only reachable inside an UNRESOLVED_* record: `reading_contrast`
            # refuses this arm outright whenever a reading is actually taken.
            # There the table is printed for diagnosis, and labelling an arm
            # "up" beside two EQUAL tallies is the misreading this whole module
            # is built to refuse -- so the label says the arm is incoherent
            # rather than picking whichever direction is tested first. This is
            # not a `clears` VALUE an arm could vote with; nothing counts it.
            clears = "both?"
        else:
            clears = (
                "up" if arm.clears_up()
                else ("down" if arm.clears_down() else "no")
            )
        lines.append("  " + "".join(
            f"{value:>{width}}"
            for value, width in zip((
                name, f"{arm.contrast:+.4f}", f"{arm.ci_low:+.4f}",
                f"{arm.ci_high:+.4f}", f"{arm.seeds_up}/{arm.seeds_total}",
                f"{arm.seeds_down}/{arm.seeds_total}", clears,
            ), READING_WIDTHS, strict=True)
        ))
    lines += [
        "  base control (enc(t) -> position, must clear r2 "
        f"{BASE_R2_FLOOR:.2f}): " + ", ".join(
            f"{a} r2={inputs.base[a].r2:+.3f} "
            f"{inputs.base[a].seeds_clear}/{inputs.base[a].seeds_total}"
            for a in sorted(inputs.base)
        ),
        "  anchors (a pass's untouched rung must reproduce its shipped gain): "
        + ", ".join(
            f"{p}={'ok' if ok else 'BROKEN'}" for p, ok in sorted(inputs.anchors.items())
        ),
        f"  verdict: {reading.status.replace('_', ' ')} -- decided by: {reading.rule}",
    ]
    return "\n".join(lines)
