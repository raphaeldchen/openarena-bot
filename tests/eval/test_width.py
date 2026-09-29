"""M3k: is the past frame's advantage information, or feature count?"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mbfps.eval.retention import ARMS_REQUIRED, BASE_R2_FLOOR, BaseControl, RUNGS, SEEDS_REQUIRED
from mbfps.eval.width import (
    ANCHOR, CONTRAST_K, ContrastArm, ContrastInputs, DOWN_WIDTH, PASSES,
    PROJECTION_SEED, READING_COLUMNS, READING_WIDTHS, RUNG_WIDTH, TARGET_WIDTH,
    UP_WIDTH, contrast_arm, format_reading_contrast, pass_block, projection,
    reading_contrast,
)

_WIDTH_PATH = Path(__file__).resolve().parents[2] / "src" / "mbfps" / "eval" / "width.py"


def test_constants_are_the_pre_registered_values():
    assert PASSES == ("shipped", "down", "up")
    assert DOWN_WIDTH == 512 and UP_WIDTH == 2048
    assert CONTRAST_K == 15
    assert TARGET_WIDTH == {"shipped": None, "down": 512, "up": 2048}


def test_contrast_k_is_not_named_decision_k():
    """`retention.DECISION_K` is 4 and means the horizon Reading E decides at;
    this module's is 15. Both are imported by the same script, so one name with
    two meanings is a misreading waiting to happen. Pinned so a later 'tidy-up'
    cannot unify them."""
    import mbfps.eval.width as width
    from mbfps.eval.retention import DECISION_K
    assert DECISION_K == 4 and CONTRAST_K == 15
    assert not hasattr(width, "DECISION_K")


def test_rung_widths_are_derived_from_the_rssm_config_not_re_spelled():
    """A hardcoded 512 here would silently disagree with the model if
    `RSSMConfig` ever changed, and every pass would be matching the wrong
    width."""
    from mbfps.models.rssm import LATENT_DIM, RSSMConfig
    z = RSSMConfig.z_cats * RSSMConfig.z_classes
    assert RUNG_WIDTH == {
        "two_frame": RSSMConfig.embed_dim,
        "deterministic": RSSMConfig.h_dim,
        "stochastic": z,
        "full": RSSMConfig.h_dim + z,
    }
    assert RUNG_WIDTH["full"] == LATENT_DIM
    assert set(RUNG_WIDTH) == set(RUNGS)


def test_rung_width_moves_when_the_rssm_config_it_derives_from_moves():
    """Value-agreement alone doesn't prove sourcing-by-reference: a hardcoded
    `RUNG_WIDTH = {"two_frame": 2048, "deterministic": 512, "stochastic": 1024,
    "full": 1536}` would pass the test above identically, since today's numbers
    happen to match. Patch `RSSMConfig.h_dim` to a distinctive value, load a
    FRESH copy of `width.py` under that patch, and check the derived values
    move with it -- proof by dependency, not by coincidence of numbers.

    A separate throwaway module is loaded via `importlib.util`, under a name
    that is not `mbfps.eval.width`, rather than `importlib.reload`-ing the
    canonical module: `reload` would mutate the module object every other test
    (and `width` itself, imported above) holds a reference to, out from under
    them.

    Registered in `sys.modules` under its throwaway name for the duration of
    `exec_module` (and removed immediately after, in a `finally`) because
    Task 4's `@dataclass` classes need it: under `from __future__ import
    annotations`, `dataclasses` resolves each field's string annotation via
    `sys.modules[cls.__module__]` to rule out `ClassVar`/`InitVar`, and a
    module the loader never registered makes that lookup return `None` and
    crash with `AttributeError` -- a Python stdlib quirk of this loading
    technique, unrelated to the values being tested here, that only surfaced
    once `width.py` gained its first dataclass.
    """
    from mbfps.models.rssm import RSSMConfig

    with patch.object(RSSMConfig, "h_dim", 777):
        spec = importlib.util.spec_from_file_location(
            "mbfps.eval._width_derivation_probe", _WIDTH_PATH
        )
        fresh_width = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fresh_width
        try:
            spec.loader.exec_module(fresh_width)
        finally:
            del sys.modules[spec.name]

    assert fresh_width.RUNG_WIDTH["deterministic"] == 777, (
        "RUNG_WIDTH['deterministic'] did not follow RSSMConfig.h_dim=777; "
        "it is re-spelled rather than derived"
    )
    assert fresh_width.RUNG_WIDTH["full"] == 777 + RSSMConfig.z_cats * RSSMConfig.z_classes, (
        "RUNG_WIDTH['full'] did not follow RSSMConfig.h_dim=777; "
        "it is re-spelled rather than derived"
    )
    # The canonical module, imported at collection time, must be untouched.
    assert RUNG_WIDTH["deterministic"] == RSSMConfig.h_dim == 512


def test_each_anchor_is_the_rung_its_pass_leaves_untouched():
    """THE structural property the anchors rest on: a pass's anchor must be the
    rung whose native width already equals that pass's target, so the pass does
    not touch it and its gain must reproduce the shipped one exactly. Derived
    here rather than trusted, so ANCHOR cannot drift away from the widths."""
    for pass_name, rung in ANCHOR.items():
        target = TARGET_WIDTH[pass_name]
        assert RUNG_WIDTH[rung] == target, (
            f"{pass_name}'s anchor {rung!r} is {RUNG_WIDTH[rung]} wide but the "
            f"pass targets {target}; it would be projected, not untouched"
        )
        untouched = [r for r in RUNGS if RUNG_WIDTH[r] == target]
        assert untouched == [rung], f"{pass_name}: expected exactly one anchor"
    assert set(ANCHOR) == {"down", "up"}, "`shipped` projects nothing, so it has no anchor"


def test_projection_is_the_identity_when_no_projection_is_needed():
    """None, not an identity matrix: the anchor path must do no matmul at all,
    so it cannot drift by a floating-point ulp and the anchor test can demand
    exact equality."""
    assert projection(512, 512) is None
    assert projection(2048, 2048) is None
    assert projection(2048, 512) is not None


def test_projection_has_the_johnson_lindenstrauss_shape_and_scale():
    p = projection(2048, 512)
    assert p.shape == (2048, 512)
    # Gaussian / sqrt(target): column norms ~1, so inner products survive.
    assert np.allclose(p.std(), 1.0 / np.sqrt(512), rtol=0.05)


def test_projection_is_fixed_across_calls_and_shared_across_cells():
    """One matrix per (native, target), drawn once. If it were redrawn per call
    no two cells would be comparable, and an arm could win on a lucky draw."""
    a, b = projection(1024, 512), projection(1024, 512)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(projection(1024, 512), projection(2048, 512)[:1024])


def test_pass_block_leaves_the_shipped_pass_untouched():
    block = np.arange(40, dtype=np.float64).reshape(10, 4)
    np.testing.assert_array_equal(pass_block(block, "shipped"), block)


def test_the_down_pass_matches_count_and_rank():
    """Projected to 512 every block has 512 columns AND rank 512, so `down` is
    the pass that can ask who wins at equal capacity."""
    rng = np.random.default_rng(0)
    for native in (2048, 1536, 1024):
        block = rng.normal(size=(3000, native))
        out = pass_block(block, "down")
        assert out.shape == (3000, DOWN_WIDTH)
        assert np.linalg.matrix_rank(out) == DOWN_WIDTH
    already = rng.normal(size=(3000, 512))
    np.testing.assert_array_equal(pass_block(already, "down"), already)


def test_the_up_pass_matches_count_but_preserves_rank():
    """THE property that makes `up` a pure width control: lifting adds columns
    and no information, so the rank must stay at the block's native width. If
    the rank rose, the lift would be adding capacity and the calibration would
    measure the wrong thing."""
    rng = np.random.default_rng(1)
    block = rng.normal(size=(3000, 512))
    out = pass_block(block, "up")
    assert out.shape == (3000, UP_WIDTH)
    assert np.linalg.matrix_rank(out) == 512, (
        "the lift changed the rank; it is meant to add columns, not information"
    )
    already = rng.normal(size=(3000, 2048))
    np.testing.assert_array_equal(pass_block(already, "up"), already)


def test_the_down_pass_routes_through_the_real_projection_matrix():
    """Shape and rank alone don't prove `pass_block` actually multiplies by
    `projection()`: a fake `down` that just truncates to the first 512 columns
    gives the same shape (3000, 512) and the same rank 512 on a generic
    Gaussian block, so it would pass the test above unnoticed. Pin exact
    equality (not `allclose`) against `block @ projection(native, DOWN_WIDTH)`
    for every non-anchor native width, so any substitution -- truncation,
    zero-padding, a different seed, a transposed matrix -- fails here."""
    rng = np.random.default_rng(0)
    for native in (2048, 1536, 1024):
        block = rng.normal(size=(30, native))
        expected = block.astype(np.float64) @ projection(native, DOWN_WIDTH)
        np.testing.assert_array_equal(pass_block(block, "down"), expected)


def test_the_up_pass_routes_through_the_real_projection_matrix():
    """The `up` analogue of the routing pin above: a fake `up` that zero-pads
    the native columns out to 2048 also matches shape and preserves rank (the
    padded columns are all zero), so it would pass the rank test unnoticed.
    Pin exact equality against `block @ projection(native, UP_WIDTH)` for
    every non-anchor native width."""
    rng = np.random.default_rng(1)
    for native in (512, 1024, 1536):
        block = rng.normal(size=(30, native))
        expected = block.astype(np.float64) @ projection(native, UP_WIDTH)
        np.testing.assert_array_equal(pass_block(block, "up"), expected)


def test_the_down_pass_approximately_preserves_squared_norms():
    """Johnson-Lindenstrauss: a random projection should roughly preserve each
    row's squared norm in expectation (mean ratio ~1.0). A fake `down` that
    truncates to the first 512 columns instead of projecting also matches
    shape and rank (see the routing test above) but discards 3/4 of the
    signal's energy, collapsing the mean ratio to ~0.25 -- this is the check
    that would catch that substitution even if the routing pin above were
    somehow dodged."""
    rng = np.random.default_rng(2)
    native = 2048
    block = rng.normal(size=(400, native))
    projected = pass_block(block, "down")
    before = np.sum(block.astype(np.float64) ** 2, axis=1)
    after = np.sum(projected ** 2, axis=1)
    ratio = float(np.mean(after / before))
    assert abs(ratio - 1.0) < 0.1, (
        f"mean squared-norm ratio after the `down` projection was {ratio:.4f}, "
        "expected ~1.0 (Johnson-Lindenstrauss norm preservation); a ratio near "
        "0.25 means the pass is truncating columns instead of projecting"
    )


def test_the_up_pass_leaves_no_dead_column():
    """Norm preservation (the check above) cannot catch zero-padding: padding
    with zeros preserves each row's squared norm exactly. A fake `up` that
    zero-pads the native 512 columns out to 2048 leaves the 1536 padded
    columns at std 0 -- dead columns that, for a ridge probe reading the `up`
    pass, mean the lift adds no features at all. Assert every output column
    has real spread."""
    rng = np.random.default_rng(3)
    block = rng.normal(size=(400, 512))
    out = pass_block(block, "up")
    col_std = out.std(axis=0)
    assert np.all(col_std > 1e-8), (
        "at least one column of the `up` pass output is constant (std ~0); "
        "zero-padding the lift leaves those columns dead instead of spreading "
        "the block's information across all 2048 output columns"
    )


def test_pass_block_rejects_an_unknown_pass():
    with pytest.raises(ValueError, match="unknown pass"):
        pass_block(np.zeros((4, 8)), "sideways")


def test_the_projection_cache_does_not_split_on_calling_convention():
    """`lru_cache` keys on the literal call signature, not on resolved
    defaults, so caching the public function directly would put
    `projection(2048, 512)` and `projection(2048, 512, seed=0)` in two entries
    holding two equal-valued copies of the same matrix -- the "drawn exactly
    once and shared" guarantee would quietly hold only per calling style, and
    `pass_block` (which always passes `seed=` explicitly) would never share
    with a direct positional caller. The cache lives on a positional-only
    helper for exactly this reason."""
    implicit = projection(2048, DOWN_WIDTH)
    explicit = projection(2048, DOWN_WIDTH, seed=PROJECTION_SEED)
    assert implicit is explicit, (
        "the two calling conventions returned different objects; the cache has "
        "split on the call signature and the matrix is materialised twice"
    )
    block = np.random.default_rng(7).normal(size=(64, 2048))
    assert pass_block(block, "down") is not None
    assert projection(2048, DOWN_WIDTH) is implicit, (
        "pass_block's internal call populated a third cache entry"
    )


ARMS = ("frozen_ssl", "pixel_ae", "random_vit")


def _arm(ci_low: float, ci_high: float) -> ContrastArm:
    up = 3 if ci_low > 0 else 0
    down = 3 if ci_high < 0 else 0
    return ContrastArm(contrast=(ci_low + ci_high) / 2, ci_low=ci_low,
                       ci_high=ci_high, seeds_up=up, seeds_down=down, seeds_total=3)


def _inputs(arms=None, *, base_r2=0.60, anchors=None) -> ContrastInputs:
    return ContrastInputs(
        arms=arms or {a: _arm(-0.01, 0.01) for a in ARMS},
        base={a: BaseControl(r2=base_r2, seeds_clear=3 if base_r2 > BASE_R2_FLOOR else 0,
                             seeds_total=3) for a in ARMS},
        anchors=anchors or {"down": True, "up": True},
        clusters=24, rows=8015,
    )


def test_contrast_arm_refuses_a_non_finite_value():
    """M3i's ledger note, still binding: every comparison against NaN is False,
    so a non-finite contrast would read 'not up' AND 'not down' at once, which
    lands on INDISTINGUISHABLE -- a pre-registered finding asserted from a
    broken number."""
    for field in ("contrast", "ci_low", "ci_high"):
        for bad in (float("nan"), float("inf"), float("-inf")):
            seed = {"contrast": 0.1, "ci_low": 0.05, "ci_high": 0.2}
            seed[field] = bad
            with pytest.raises(ValueError, match="non-finite"):
                contrast_arm([seed] * SEEDS_REQUIRED)


def test_contrast_arm_tallies_both_directions_separately():
    """Reading F is two-sided, so an arm needs BOTH tallies: M3h shipped a table
    that printed only the up tally beside a verdict read from the down one."""
    arm = contrast_arm([
        {"contrast": +0.30, "ci_low": +0.10, "ci_high": +0.50},
        {"contrast": -0.20, "ci_low": -0.40, "ci_high": -0.05},
        {"contrast": +0.05, "ci_low": -0.02, "ci_high": +0.12},
    ])
    assert arm.seeds_up == 1 and arm.seeds_down == 1 and arm.seeds_total == 3
    assert arm.contrast == pytest.approx(0.05)
    assert arm.ci_low == pytest.approx(-0.40) and arm.ci_high == pytest.approx(+0.50)
    assert arm.clears_up() is False and arm.clears_down() is False


def test_contrast_arm_needs_seeds_required_seeds():
    with pytest.raises(ValueError, match="at least"):
        contrast_arm([{"contrast": 0.1, "ci_low": 0.05, "ci_high": 0.2}])


def test_reading_is_unresolved_base_when_the_frame_cannot_locate_itself():
    reading = reading_contrast(_inputs({a: _arm(0.1, 0.3) for a in ARMS},
                                       base_r2=BASE_R2_FLOOR - 0.05))
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.base_failed == ARMS
    assert reading.arms_up == () and reading.arms_down == ()


def test_reading_is_unresolved_anchor_when_the_down_pass_anchor_breaks():
    """`down` is the pass Reading F is taken on, so a broken anchor there means
    the projection plumbing is wrong and the reading is unreadable."""
    reading = reading_contrast(_inputs({a: _arm(0.1, 0.3) for a in ARMS},
                                       anchors={"down": False, "up": True}))
    assert reading.status == "UNRESOLVED_ANCHOR"
    assert reading.anchors_broken == ("down",)


def test_a_broken_up_anchor_does_not_suppress_the_reading():
    """Reading F is taken on `down` alone. `up` only calibrates the width-bias
    companion, so letting its failure block a verdict would let a companion that
    decides nothing veto one that does."""
    reading = reading_contrast(_inputs({a: _arm(0.1, 0.3) for a in ARMS},
                                       anchors={"down": True, "up": False}))
    assert reading.status == "PAST_FRAME_AHEAD"
    assert reading.anchors_broken == ("up",), "still reported, just not fatal"


def test_reading_is_past_frame_ahead_when_the_contrast_clears_positive():
    reading = reading_contrast(_inputs({
        "frozen_ssl": _arm(0.02, 0.06), "pixel_ae": _arm(0.03, 0.07),
        "random_vit": _arm(-0.01, 0.01),
    }))
    assert reading.status == "PAST_FRAME_AHEAD"
    assert reading.arms_up == ("frozen_ssl", "pixel_ae")


def test_reading_is_recurrent_ahead_when_the_contrast_clears_negative():
    """The direction M3j could not have reported: h retaining MORE than the past
    frame at equal width is a positive finding about h, not an absence."""
    reading = reading_contrast(_inputs({
        "frozen_ssl": _arm(-0.06, -0.02), "pixel_ae": _arm(-0.07, -0.03),
        "random_vit": _arm(-0.01, 0.01),
    }))
    assert reading.status == "RECURRENT_AHEAD"
    assert reading.arms_down == ("frozen_ssl", "pixel_ae")


def test_reading_is_indistinguishable_when_neither_direction_clears():
    reading = reading_contrast(_inputs())
    assert reading.status == "INDISTINGUISHABLE"
    assert "could not" in reading.rule or "by default" in reading.rule


def test_up_and_down_cannot_both_clear_with_three_arms():
    """ARMS_REQUIRED of 2 out of 3 arms means the two directions cannot both
    reach the bar -- 2 + 2 > 3. Pinned so a later change to ARMS_REQUIRED or the
    arm count surfaces the contradiction here rather than in a verdict."""
    assert 2 * ARMS_REQUIRED > len(ARMS)


def test_reading_columns_and_widths_stay_the_same_length():
    assert len(READING_COLUMNS) == len(READING_WIDTHS)


def test_reading_table_puts_each_value_under_its_own_caption():
    """Header and rows sliced at the same offsets, and the VALUE asserted --
    asserting non-emptiness alone is what let three wrong captions ship here."""
    inputs = _inputs({"frozen_ssl": _arm(0.02, 0.06), "pixel_ae": _arm(-0.07, -0.03),
                      "random_vit": _arm(-0.01, 0.01)})
    text = format_reading_contrast(reading_contrast(inputs), inputs)
    lines = [line for line in text.splitlines() if line.startswith("  ")]
    header = lines[0]
    offset = 2
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        assert header[offset:offset + width].strip() == name
        offset += width
    row = next(line for line in lines[1:] if "frozen_ssl" in line)
    offset = 2
    parsed = {}
    for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
        parsed[name] = row[offset:offset + width].strip()
        offset += width
    arm = inputs.arms["frozen_ssl"]
    assert parsed["arm"] == "frozen_ssl"
    assert float(parsed["contrast"]) == pytest.approx(arm.contrast, abs=5e-5)
    assert float(parsed["ci_low"]) == pytest.approx(arm.ci_low, abs=5e-5)
    assert float(parsed["ci_high"]) == pytest.approx(arm.ci_high, abs=5e-5)
    assert parsed["up"] == f"{arm.seeds_up}/3" and parsed["dn"] == f"{arm.seeds_down}/3"


def test_reading_caption_names_the_horizon_the_rule_and_the_clusters():
    inputs = _inputs()
    caption = format_reading_contrast(reading_contrast(inputs), inputs).splitlines()[0]
    assert f"k = {CONTRAST_K}" in caption
    assert "two_frame" in caption and "deterministic" in caption
    assert "24 clusters" in caption
    assert "two-sided" in caption


# --- Supplementary tests, added beyond the brief's Step 1 block -----------
#
# The task's standing-hazard note calls out two failure modes this project has
# shipped repeatedly: precedence inversions that survive a whole suite because
# every fixture clears exactly one rung, and table captions that disagree with
# their columns because a test asserted non-emptiness rather than a value. The
# brief's own fixtures above close most of the precedence lattice already (see
# each test below for which pair it targets), but leave three gaps and one
# textbook "fixed value in every fixture" trap. These tests close them.


def test_unresolved_base_outranks_a_broken_down_anchor():
    """Precedence pair (UNRESOLVED_BASE, UNRESOLVED_ANCHOR), not exercised
    above: every existing UNRESOLVED_BASE fixture uses the default anchors
    (both true), and every UNRESOLVED_ANCHOR fixture uses a clearing base. A
    precedence inversion that checked the anchor before the base would pass
    every test above -- each fixture only ever makes ONE of the two
    conditions true -- and would only be caught by a fixture where both are
    true at once, like this one (base fails AND the down anchor is broken).
    `anchors_broken` must still name the broken anchor: it is reported
    alongside the base failure, not suppressed by it.
    """
    reading = reading_contrast(_inputs(
        {a: _arm(0.1, 0.3) for a in ARMS},
        base_r2=BASE_R2_FLOOR - 0.05,
        anchors={"down": False, "up": True},
    ))
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.base_failed == ARMS
    assert reading.anchors_broken == ("down",), "still reported, just not the verdict"
    assert reading.arms_up == () and reading.arms_down == ()


def test_unresolved_base_outranks_a_clearing_recurrent_ahead_direction():
    """Precedence pair (UNRESOLVED_BASE, RECURRENT_AHEAD). The brief's own
    UNRESOLVED_BASE fixture already overlaps with PAST_FRAME_AHEAD (its arms
    clear up), but Reading F is two-sided and a base check wired to look only
    at the `up` tally (a plausible copy-paste from Reading E, which is
    one-sided) would still win against that fixture while missing a `down`-
    clearing one. This fixture makes both conditions true in the other
    direction."""
    reading = reading_contrast(_inputs(
        {"frozen_ssl": _arm(-0.06, -0.02), "pixel_ae": _arm(-0.07, -0.03),
         "random_vit": _arm(-0.01, 0.01)},
        base_r2=BASE_R2_FLOOR - 0.05,
    ))
    assert reading.status == "UNRESOLVED_BASE"
    assert reading.arms_up == () and reading.arms_down == ()


def test_unresolved_anchor_outranks_a_clearing_recurrent_ahead_direction():
    """Precedence pair (UNRESOLVED_ANCHOR, RECURRENT_AHEAD), the down-direction
    analogue of `test_reading_is_unresolved_anchor_when_the_down_pass_anchor_breaks`
    (which only overlaps UNRESOLVED_ANCHOR with the `up` direction). An anchor
    check applied asymmetrically -- gating `up` results but not `down` ones --
    would pass that test and fail this one."""
    reading = reading_contrast(_inputs(
        {"frozen_ssl": _arm(-0.06, -0.02), "pixel_ae": _arm(-0.07, -0.03),
         "random_vit": _arm(-0.01, 0.01)},
        anchors={"down": False, "up": True},
    ))
    assert reading.status == "UNRESOLVED_ANCHOR"
    assert reading.anchors_broken == ("down",)
    assert reading.arms_up == () and reading.arms_down == ()


def test_reading_table_pins_a_negative_row_and_the_clears_column():
    """The brief's `test_reading_table_puts_each_value_under_its_own_caption`
    pins only `frozen_ssl` (an up-clearing, positive row) and never asserts
    the `clears` column's value at all. M3h's shipped defect was exactly a
    column read from the wrong tally, and that defect is invisible on a
    table with only one clearing direction present -- both directions need a
    row, and `clears` needs its value checked, not just its column slot. Uses
    the same three-arm, two-direction fixture as that test so this is a real
    extension of it, not a different scenario."""
    inputs = _inputs({"frozen_ssl": _arm(0.02, 0.06), "pixel_ae": _arm(-0.07, -0.03),
                      "random_vit": _arm(-0.01, 0.01)})
    text = format_reading_contrast(reading_contrast(inputs), inputs)
    lines = [line for line in text.splitlines() if line.startswith("  ")]

    def parse(row: str) -> dict:
        offset = 2
        parsed = {}
        for name, width in zip(READING_COLUMNS, READING_WIDTHS, strict=True):
            parsed[name] = row[offset:offset + width].strip()
            offset += width
        return parsed

    positive = parse(next(line for line in lines[1:] if "frozen_ssl" in line))
    negative = parse(next(line for line in lines[1:] if "pixel_ae" in line))
    neither = parse(next(line for line in lines[1:] if "random_vit" in line))

    pos_arm, neg_arm, neu_arm = (
        inputs.arms["frozen_ssl"], inputs.arms["pixel_ae"], inputs.arms["random_vit"],
    )
    assert float(negative["contrast"]) == pytest.approx(neg_arm.contrast, abs=5e-5)
    assert float(negative["ci_low"]) == pytest.approx(neg_arm.ci_low, abs=5e-5)
    assert float(negative["ci_high"]) == pytest.approx(neg_arm.ci_high, abs=5e-5)
    assert negative["up"] == f"{neg_arm.seeds_up}/3" and negative["dn"] == f"{neg_arm.seeds_down}/3"

    assert positive["clears"] == "up", "the up-clearing row must read 'up', not 'down' or 'no'"
    assert negative["clears"] == "down", "the down-clearing row must read 'down', not 'up' or 'no'"
    assert neither["clears"] == "no"
    assert pos_arm.seeds_up != neg_arm.seeds_up or pos_arm.seeds_down != neg_arm.seeds_down, (
        "fixture sanity: the two rows must differ, or a transposed pair of rows "
        "would be undetectable"
    )


def test_reading_caption_uses_the_inputs_clusters_and_rows_not_a_hardcoded_value():
    """`_inputs()` always builds `clusters=24, rows=8015` -- every test above
    that reads the caption's cluster/row count uses that same fixture, so a
    caption that printed the literal string "24 clusters" / "8015 rows"
    instead of `inputs.clusters` / `inputs.rows` would pass every one of
    them. This is the exact "one value fixed in every fixture" trap the task
    warns about. Built directly (not through `_inputs`) so the numbers differ
    from every other test's fixture and a hardcoded caption is exposed."""
    inputs = ContrastInputs(
        arms={a: _arm(-0.01, 0.01) for a in ARMS},
        base={a: BaseControl(r2=0.60, seeds_clear=3, seeds_total=3) for a in ARMS},
        anchors={"down": True, "up": True},
        clusters=7, rows=123,
    )
    caption = format_reading_contrast(reading_contrast(inputs), inputs).splitlines()[0]
    assert "7 clusters" in caption and "123 rows" in caption
    assert "24 clusters" not in caption and "8015 rows" not in caption
