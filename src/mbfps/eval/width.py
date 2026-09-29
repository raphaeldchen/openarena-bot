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

import numpy as np

from mbfps.eval.retention import RUNGS
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
    """
    if native == target:
        return None
    rng = np.random.default_rng([seed, native, target])
    return rng.normal(size=(native, target)) / np.sqrt(target)


def pass_block(block: np.ndarray, pass_name: str, *, seed: int = PROJECTION_SEED):
    """`block` as `pass_name` sees it.

    `shipped` returns it unchanged. `down` projects to 512, matching count AND
    rank. `up` lifts to 2048, matching count while LEAVING THE RANK at the
    block's native width -- that is what makes `up` a pure width control rather
    than a capacity change, and it is pinned by a rank test.
    """
    if pass_name not in TARGET_WIDTH:
        raise ValueError(f"unknown pass {pass_name!r}; expected one of {PASSES}")
    target = TARGET_WIDTH[pass_name]
    block = np.asarray(block, dtype=np.float64)
    if target is None:
        return block
    matrix = projection(block.shape[1], target, seed=seed)
    return block if matrix is None else block @ matrix
