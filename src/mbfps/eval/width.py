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
    to whether its known-answer rung reproduced its shipped gain."""

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


def contrast_arm(contrasts: list[dict]) -> ContrastArm:
    """One arm's `ContrastArm` from its per-seed `probe.contrast_from_blocks`
    dicts.

    THE FINITENESS GUARD RAISES, and this is M3i's ledger note still binding:
    every comparison against NaN is False, so a non-finite contrast would read
    "not up" AND "not down" at once. That lands on INDISTINGUISHABLE -- a
    pre-registered finding asserted from a broken number, and the direction of
    the silence is toward a status the project will act on.
    """
    if len(contrasts) < SEEDS_REQUIRED:
        raise ValueError(
            f"an arm needs at least SEEDS_REQUIRED={SEEDS_REQUIRED} seeds to "
            f"summarise, got {len(contrasts)}"
        )
    for index, seed in enumerate(contrasts):
        bad = {
            key: float(seed[key])
            for key in ("contrast", "ci_low", "ci_high")
            if not np.isfinite(float(seed[key]))
        }
        if bad:
            raise ValueError(
                f"non-finite {', '.join(sorted(bad))} in seed index {index}: "
                f"{bad}. A non-finite contrast is an error about the "
                "measurement, not an INDISTINGUISHABLE reading."
            )
    return ContrastArm(
        contrast=float(np.mean([c["contrast"] for c in contrasts])),
        ci_low=float(min(float(c["ci_low"]) for c in contrasts)),
        ci_high=float(max(float(c["ci_high"]) for c in contrasts)),
        seeds_up=sum(1 for c in contrasts if float(c["ci_low"]) > 0.0),
        seeds_down=sum(1 for c in contrasts if float(c["ci_high"]) < 0.0),
        seeds_total=len(contrasts),
    )


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
    an absence. The two directions cannot both clear: 2 of 3 arms each way needs
    4 of 3 arms.
    """
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
            arms_up=(), arms_down=(), base_failed=(),
            anchors_broken=anchors_broken,
        )

    up = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_up()))
    down = tuple(sorted(a for a, arm in inputs.arms.items() if arm.clears_down()))

    if len(up) >= ARMS_REQUIRED:
        return ContrastStatus(
            status="PAST_FRAME_AHEAD",
            rule=(
                f"at equal block width the past frame still beats the recurrent state "
                f"at k = {CONTRAST_K} in {len(up)} of {len(inputs.arms)} arms "
                f"({', '.join(up)}), each in at least {SEEDS_REQUIRED} seeds; M3j's "
                f"objective-lever argument survives the width matching"
            ),
            arms_up=up, arms_down=down, base_failed=(), anchors_broken=anchors_broken,
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
            arms_up=up, arms_down=down, base_failed=(), anchors_broken=anchors_broken,
        )
    return ContrastStatus(
        status="INDISTINGUISHABLE",
        rule=(
            f"neither direction clears in {ARMS_REQUIRED} arms at k = {CONTRAST_K}; "
            f"M3j's k = 15 result was width, and the bottleneck lever stands alone BY "
            f"DEFAULT rather than by evidence -- we could not tell the two blocks "
            f"apart, which is not the same as ruling one out"
        ),
        arms_up=up, arms_down=down, base_failed=(), anchors_broken=anchors_broken,
    )


READING_COLUMNS: tuple[str, ...] = (
    "arm", "contrast", "ci_low", "ci_high", "up", "dn", "clears",
)
READING_WIDTHS: tuple[int, ...] = (13, 12, 11, 11, 7, 7, 9)


def format_reading_contrast(reading: ContrastStatus, inputs: ContrastInputs) -> str:
    """Reading F as it is printed and written to `width.txt`, byte for byte."""
    lines = [
        f"--- Reading F: at EQUAL block width (512), does two_frame still beat "
        f"deterministic on translation at k = {CONTRAST_K}? "
        f"(difference of joint R^2; the shared base cancels; two-sided, clears when "
        f"an interval excludes 0 in {SEEDS_REQUIRED} of 3 seeds and {ARMS_REQUIRED} "
        f"of 3 arms); {inputs.rows} rows over {inputs.clusters} clusters ---",
        "  " + "".join(
            f"{name:>{width}}"
            for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True)
        ),
    ]
    for name in sorted(inputs.arms):
        arm = inputs.arms[name]
        clears = "up" if arm.clears_up() else ("down" if arm.clears_down() else "no")
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
