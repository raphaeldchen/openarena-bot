"""M3k: is the past frame's advantage information, or feature count?"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mbfps.eval.retention import RUNGS
from mbfps.eval.width import (
    ANCHOR, CONTRAST_K, DOWN_WIDTH, PASSES, PROJECTION_SEED, RUNG_WIDTH,
    TARGET_WIDTH, UP_WIDTH, pass_block, projection,
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
    """
    from mbfps.models.rssm import RSSMConfig

    with patch.object(RSSMConfig, "h_dim", 777):
        spec = importlib.util.spec_from_file_location(
            "mbfps.eval._width_derivation_probe", _WIDTH_PATH
        )
        fresh_width = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh_width)

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
