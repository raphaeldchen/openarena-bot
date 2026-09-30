"""M3k's script: three passes over one gather, the anchors that gate them, and the
read that pools nine records into Reading F."""

import copy
import importlib.util
import inspect
import json
import re
import types
import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.retention import (
    ARMS_REQUIRED, BASE_R2_FLOOR, K_REPORTED, RESAMPLES, RETENTION_FAMILY, RUNGS,
    SEEDS_REQUIRED, TARGETS, backward_rotation, backward_translation, rung_block,
    shifted_rows,
)
from mbfps.eval.retention import reading_retention as _reading_retention
from mbfps.eval.width import (
    ANCHOR, CONTRAST_K, DOWN_WIDTH, PASSES, PROJECTION_SEED, RUNG_WIDTH, UP_WIDTH,
    pass_block, reading_contrast,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latent_width.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_width_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


def _gathered(n_windows=8, steps=20, enc=2048, h_dim=512, z=1024, seed=0, factors=8):
    """A `gather_probe_data`-shaped dict at the REAL widths, so the projections
    exercise their production shapes and the anchors (native 2048 and 512) are
    genuinely native. The deterministic half of the latent holds the previous
    frame's position, so `deterministic` must gain and a noise rung must not.

    THIS FIXTURE DIFFERS FROM THE BRIEF'S IN TWO WAYS, EACH MEASURED.

    (1) THE WIDE BLOCKS ARE LOW-RANK MIXTURES, not 2 informative columns among
    2046 independent noise ones. With n ~ 150 rows and p ~ 4100 columns the
    brief's version leaves the signal's kernel eigenvalue (2 columns x n) below
    the noise's (~2000), so ridge shrinks it away: measured, `enc(t)` -> position
    read r^2 0.246 (the brief's own `> 0.5` assertion FAILS on it) and every gain
    at k = 1 was ~1e-5, so every real-width numeric test compared near-zero
    numbers. Real embeddings are low effective rank; here each block is
    `factors @ M` for a fixed random `M` per width (the same `M` in the fit,
    select and score gathers, as one model would give) plus 1e-3 of independent
    noise.

    (2) THE ANGLE CARRIES NOISE. With the angle a pure function of
    `(step, window)`, `backward_rotation` is the constant 7k degrees on every
    row -- measured std 3e-16, float jitter only -- and `cell_passes` on that
    fixture RAISES "every target column has zero variance" inside a bootstrap
    resample. `latent_retention`'s own fixture adds the same term for the same
    reason. Rotation is then a genuine NULL here (no block carries it): its
    gains are ~0, real numbers rather than rounding error.

    `embedding` (the model's PREDICTED embedding) is a decoy of the same scale
    with no position in it: nothing in this script may read it, a wrong read
    would be decisively wrong rather than marginally so, and a test below drops
    it from the dict to prove the script never touches it.
    """
    rng = np.random.default_rng(seed)
    rows = n_windows * steps
    window = np.repeat(np.arange(n_windows), steps)
    step = np.tile(np.arange(steps), n_windows)
    episode = window // 2
    pos = np.column_stack([
        window * 100.0 + step * 3.0 + rng.normal(scale=5.0, size=rows),
        window * 50.0 - step * 2.0 + rng.normal(scale=5.0, size=rows),
    ])
    radians = (
        np.deg2rad((step * 7.0 + window * 11.0) % 360.0)
        + rng.normal(scale=0.2, size=rows)
    )
    targets = np.column_stack([pos, np.sin(radians), np.cos(radians)])

    def block(informative, width):
        drivers = np.column_stack([
            informative, rng.normal(size=(rows, factors - informative.shape[1])),
        ])
        mixing = np.random.default_rng(1000 + width).normal(size=(factors, width))
        return drivers @ mixing + 1e-3 * rng.normal(size=(rows, width))

    encoder = block(pos, enc)
    latent = np.column_stack([
        block(np.roll(pos, 1, axis=0), h_dim), block(np.empty((rows, 0)), z),
    ])
    return {
        "latent": latent, "encoder_embedding": encoder,
        "embedding": rng.normal(size=encoder.shape) * encoder.std(),
        "targets": targets, "window": window, "step": step, "episode": episode,
    }


def _splits(seeds=(0, 1, 2), **kwargs):
    return [_gathered(seed=s, **kwargs) for s in seeds]


KEY = f"k{CONTRAST_K}"

FEW_RESAMPLES = 20
"""Bootstrap draws for a test that runs the REAL probe at production widths and
asserts on no interval's width. `cell_passes` defaults to the protocol's
`RESAMPLES`; a test may pay for fewer only if nothing it asserts depends on the
draws -- never one that pins an interval bound to a side of zero by a small
margin. Measured, and worth knowing before reaching for it: on this fixture a
gain takes 1.43 s at 10 draws and 1.46 s at 1000, because ~96% of `cell_passes`
is the ridge solve on 2560- to 4096-column designs. Fewer draws are honest but
they are ~3% of the runtime, not the bulk of it."""


@pytest.fixture(scope="module")
def _real_run_and_widths():
    """The one real `cell_passes` run, and the `(block width, base width)` the
    real probe was handed on every gain, in call order.

    The recorder wraps `gain_from_blocks` for the duration of this run only (a
    module-scoped fixture cannot use the function-scoped `monkeypatch`) and
    passes every call straight through. Recording here rather than in a second
    run of its own is what lets `test_the_down_pass_projects_every_rung_to_the_
    same_width` keep the REAL ridge without paying for a second ~24 s of it.
    """
    seen = []
    real = script.gain_from_blocks

    def recording(fit, select, score, **kwargs):
        seen.append((np.asarray(fit.block).shape[1], np.asarray(fit.base).shape[1]))
        return real(fit, select, score, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(script, "gain_from_blocks", recording)
        out = script.cell_passes(
            *_splits(), h_dim=512, ks=(1, CONTRAST_K), resamples=FEW_RESAMPLES,
        )
    return out, seen


@pytest.fixture(scope="module")
def real_run(_real_run_and_widths):
    """One real `cell_passes` run at the production widths, shared by every test
    that asserts on its NUMBERS. Module-scoped, so it is built before any
    test's own `monkeypatch` can reach `gain_from_blocks`, and it is never
    mutated: a test that needs to break it deep-copies first.

    Run at `FEW_RESAMPLES`: every assertion on it is on a gain, a joint R^2, a
    window count, or an interval that is compared for EQUALITY against another
    interval from the same draws (the anchors) or for sitting above zero by ~0.8
    (`ci_low` of the contrast). None depends on the interval's width.
    """
    return _real_run_and_widths[0]


@pytest.fixture(scope="module")
def real_run_widths(_real_run_and_widths):
    """What the real probe was handed, per gain, during `real_run`'s run."""
    return _real_run_and_widths[1]


def _stub_blocks(monkeypatch):
    """Replace both probe entry points with recorders that hand their arguments
    straight back, so the result of `cell_passes` CARRIES the labels
    `(pass, target, k, rung)` each call was made under. Order-independent by
    construction, and it costs no ridge solve.

    The recorders' signatures are the real ones (keyword-only `groups`,
    `resamples`, `confidence`, `seed`), so a call the real function would reject
    is rejected here too."""
    def gain(fit, select, score, *, groups, resamples, confidence, seed):
        return {
            "gain": 0.0, "splits": (fit, select, score),
            "groups": np.asarray(groups), "seed": seed,
            "resamples": resamples, "confidence": confidence,
        }

    def contrast(a, b, *, groups, resamples, confidence, seed):
        return {
            "contrast": 0.0, "a": a, "b": b, "groups": np.asarray(groups),
            "seed": seed, "resamples": resamples, "confidence": confidence,
        }

    monkeypatch.setattr(script, "gain_from_blocks", gain)
    monkeypatch.setattr(script, "contrast_from_blocks", contrast)


def test_exit_codes_are_41_and_42_and_do_not_collide():
    """M3i holds 38 and M3j holds 39-40. A reused number makes two different
    refusals indistinguishable to a caller reading `$?`."""
    assert script.EXIT_BASE_UNRESOLVED == 41
    assert script.EXIT_ANCHOR_BROKEN == 42
    inherited = {
        script.EXIT_OK, script.EXIT_NO_CHECKPOINTS, script.EXIT_SPLIT_MISMATCH,
        script.EXIT_RECORD_MISMATCH, script.EXIT_SELF_CHECK_FAILED,
    }
    assert inherited == {0, 11, 12, 14, 30}
    assert not inherited & {script.EXIT_BASE_UNRESOLVED, script.EXIT_ANCHOR_BROKEN}


def test_phases_are_measure_read_all():
    assert script.PHASES == ("measure", "read", "all")


def test_episode_budget_is_the_filtering_gain_defaults():
    """`select_episodes` at 20, not 4. `probe.filtering_gain`'s docstring records
    that a 4-episode selection split picks the wrong ridge and FLIPS THE SIGN of
    the gain. Read off the function's own signature, not re-typed: a constant
    that merely agrees with today's default would not follow it."""
    from mbfps.eval.probe import filtering_gain
    parameters = inspect.signature(filtering_gain).parameters
    assert script.FIT_EPISODES == parameters["limit"].default == 20
    assert script.SELECT_EPISODES == parameters["select_episodes"].default == 20


def test_record_path_names_both_the_arm_and_the_seed(tmp_path):
    """A record named by only one of (arm, seed) collides three ways or nine."""
    assert script.width_record_path(tmp_path, "pixel_ae", 2).name == "width_pixel_ae_seed2.json"
    cells = [(a, s) for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)]
    assert len({script.width_record_path(tmp_path, a, s) for a, s in cells}) == 9
    assert script.width_record_path(tmp_path, "pixel_ae", 2).parent == tmp_path


def test_k_key_carries_no_dot():
    """`write_record` addresses non-finite fields by dotted path, so a key with
    a '.' in it would be unaddressable."""
    assert script.k_key(15) == "k15"
    assert "." not in script.k_key(4)


# ---------------------------------------------------------------------------
# cell_passes: the numbers, at production widths.
# ---------------------------------------------------------------------------


def test_cell_passes_covers_every_pass_target_rung_and_horizon(real_run):
    out = real_run
    assert set(out["passes"]) == set(PASSES)
    for pass_name in PASSES:
        assert set(out["passes"][pass_name]) == set(TARGETS)
        for target in TARGETS:
            assert set(out["passes"][pass_name][target]) == {"k1", KEY}
            for k in out["passes"][pass_name][target]:
                assert set(out["passes"][pass_name][target][k]) == set(RUNGS)


def test_each_pass_anchor_reproduces_its_shipped_gain_exactly(real_run):
    """THE known-answer property. `projection` returns the identity when the
    native width already equals the target, so the anchor rung is not touched
    and its gain must be bit-identical to the shipped pass -- not approximately
    equal. Anything else means the projection machinery is running where it
    should not.

    Asserted on the WHOLE gain dict (all ten keys, bootstrap interval and ridge
    included), at EVERY target and horizon: identical inputs and an identical
    bootstrap draw give an identical dict, so a pass that quietly used another
    seed, another split or another ridge for its anchor is caught here too."""
    out = real_run
    for pass_name, rung in ANCHOR.items():
        for target in TARGETS:
            for k in ("k1", KEY):
                shipped = out["passes"]["shipped"][target][k][rung]
                projected = out["passes"][pass_name][target][k][rung]
                assert projected == shipped, (
                    f"{pass_name}'s anchor {rung} moved at {target}/{k}: "
                    f"{projected['gain']} vs shipped {shipped['gain']}"
                )
    assert script.anchor_check(out["passes"]) == {"down": True, "up": True}


def test_every_projected_rung_actually_moved(real_run):
    """The anchor test alone is VACUOUS against an implementation that never
    projects anything -- `pass_block(block, "shipped")` for every pass makes every
    anchor bit-identical, trivially. So the rungs that MUST change are pinned to
    change: in `down` everything but `deterministic`, in `up` everything but
    `two_frame`, at both targets and every horizon."""
    out = real_run
    for pass_name, anchor in ANCHOR.items():
        for target in TARGETS:
            for k in ("k1", KEY):
                for rung in RUNGS:
                    if rung == anchor:
                        continue
                    moved = out["passes"][pass_name][target][k][rung]["gain"]
                    native = out["passes"]["shipped"][target][k][rung]["gain"]
                    # A real change, not the last bit: measured, the smallest
                    # move on this fixture is ~4e-7.
                    assert abs(moved - native) > 1e-9, (
                        f"{pass_name}/{rung} at {target}/{k} equals its shipped gain "
                        f"{native}: the block was not projected"
                    )


def test_the_base_is_one_array_across_every_pass_and_rung(real_run):
    """Matching applies to the rung's BLOCK only, so the base-only R^2 at one
    `(target, k)` is ONE number across all twelve `(pass, rung)` fits. A base
    that was projected in any pass, or row-selected differently for any rung,
    would split it."""
    out = real_run
    for target in TARGETS:
        for k in ("k1", KEY):
            levels = {
                out["passes"][p][target][k][r]["embedding_r2"]
                for p in PASSES for r in RUNGS
            }
            assert len(levels) == 1, f"{target}/{k}: base R^2 differs across fits: {levels}"


def test_the_bootstrap_resamples_episodes_and_the_rows_are_counted_per_horizon(real_run):
    """Hand-typed: 8 windows of 20 steps in 4 episodes. At k there are `20 - k`
    rows per window, so 8 * (20 - k) scored rows -- and the interval resamples
    the 4 EPISODES, not the 8 windows and not the rows. `n_scored_windows` is
    the number of distinct group labels the bootstrap drew from."""
    out = real_run
    assert out["rows"] == {"k1": 152, KEY: 40}
    for pass_name in PASSES:
        for target in TARGETS:
            for k in ("k1", KEY):
                for rung in RUNGS:
                    got = out["passes"][pass_name][target][k][rung]
                    assert got["n_scored_windows"] == 4, (pass_name, target, k, rung)
    assert out["contrast"][KEY]["n_scored_windows"] == 4


def test_the_contrast_is_taken_on_the_down_pass_at_the_contrast_horizon(real_run):
    """Reading F is decided on `down` at k = CONTRAST_K. A contrast taken on the
    shipped pass would reproduce exactly the confound this milestone exists to
    remove; one taken on `up`, on the rotation target, with the arms swapped, or
    at another horizon answers a different question.

    Every one of those is pinned against a number that DIFFERS on this fixture:
    the down pass's own two_frame and deterministic joint R^2s (which the shipped
    and up passes do not reproduce, per `test_every_projected_rung_actually_moved`)."""
    out = real_run
    assert set(out["contrast"]) == {KEY}
    c = out["contrast"][KEY]
    down = out["passes"]["down"]["translation"][KEY]
    assert c["a_r2"] == pytest.approx(down["two_frame"]["joint_r2"], abs=1e-12)
    assert c["b_r2"] == pytest.approx(down["deterministic"]["joint_r2"], abs=1e-12)
    assert c["contrast"] == pytest.approx(c["a_r2"] - c["b_r2"], abs=1e-12)
    for other in ("shipped", "up"):
        wrong = out["passes"][other]["translation"][KEY]
        assert c["a_r2"] != pytest.approx(wrong["two_frame"]["joint_r2"], abs=1e-9)
    rotation = out["passes"]["down"]["rotation"][KEY]
    assert c["a_r2"] != pytest.approx(rotation["two_frame"]["joint_r2"], abs=1e-9)
    assert c["a_r2"] != pytest.approx(c["b_r2"], abs=1e-9), (
        "the two arms must differ on this fixture or a swap could not be seen"
    )
    # By construction `two_frame` holds `enc(t-15)` and `deterministic` holds
    # only `p(t-1)`, so at k = 15 the past frame is far ahead and a positive
    # contrast IS that finding. A contrast with its arms swapped is the same
    # magnitude and the OPPOSITE verdict.
    assert c["contrast"] > 0.5 and c["ci_low"] > 0.0, c


def test_the_shipped_pass_finds_the_gain_a_lagged_latent_carries(real_run):
    """A positive control at production widths, so a pipeline that returns
    plausible-looking garbage cannot pass on structure alone. At k = 1 the
    target is `p(t) - p(t-1)`; `enc(t)` holds only `p(t)`, while `deterministic`
    holds `p(t-1)` and `two_frame` holds `enc(t-1)`. The stochastic half of the
    latent is pure noise and must add nothing."""
    shipped = real_run["passes"]["shipped"]["translation"]["k1"]
    assert shipped["deterministic"]["gain"] > 0.3, shipped["deterministic"]["gain"]
    assert shipped["two_frame"]["gain"] > 0.3, shipped["two_frame"]["gain"]
    assert shipped["stochastic"]["gain"] < 0.1, shipped["stochastic"]["gain"]


# ---------------------------------------------------------------------------
# cell_passes: the wiring, label by label, without a single ridge solve.
# ---------------------------------------------------------------------------


def _expected(data, target, k, rung, pass_name, h_dim=512):
    """What `cell_passes` should have handed the probe for one split, built from
    the library's own pieces WITH the arguments each one is documented to take:
    `two_frame` reads `source` (`enc(t-k)`), the latent rungs read `rows`."""
    builder = backward_translation if target == "translation" else backward_rotation
    values, rows = builder(data["targets"], data["window"], data["step"], k)
    _, source = shifted_rows(data["window"], data["step"], k)
    native = rung_block(data, rung, rows=rows, source=source, h_dim=h_dim)
    return data["encoder_embedding"][rows], pass_block(native, pass_name, seed=PROJECTION_SEED), values, rows


def test_every_gain_gets_the_untouched_base_and_its_passs_projection_of_the_native_block(
    monkeypatch,
):
    """Label by label -- `(k, target, pass, rung)` -- what each fit was handed.
    The base must be `enc(t)` on the target's rows, UNTOUCHED, in every pass; the
    block must be that rung's native block under THAT pass's projection.

    Label-aware on purpose: a Counter over block widths cannot see `down` and
    `up` swapped (512 x 4 and 2048 x 4 just trade places), nor which RUNG got
    which width. Here the results carry the labels they were computed under, so
    a swap, a base that was projected, a `two_frame` block read from `rows`
    instead of `source`, and a fit that never received its `select` split all
    fail by name."""
    _stub_blocks(monkeypatch)
    gathers = dict(zip(("fit", "select", "score"), _splits()))
    ks = (4, CONTRAST_K, 1)
    out = script.cell_passes(*gathers.values(), h_dim=512, ks=ks)
    checked = 0
    for k in ks:
        for target in TARGETS:
            for pass_name in PASSES:
                for rung in RUNGS:
                    got = out["passes"][pass_name][target][f"k{k}"][rung]
                    for name, split in zip(gathers, got["splits"]):
                        base, block, values, _ = _expected(
                            gathers[name], target, k, rung, pass_name,
                        )
                        where = (k, target, pass_name, rung, name)
                        assert np.array_equal(split.base, base), where
                        assert np.array_equal(split.block, block), where
                        assert np.array_equal(split.target, values), where
                        checked += 1
    assert checked == len(ks) * len(TARGETS) * len(PASSES) * len(RUNGS) * 3


def test_the_down_pass_projects_every_rung_to_the_same_width(real_run_widths):
    """Matching means every rung's BLOCK is the same width; the base is left
    alone so gains stay on M3j's scale. Only a recorded width can catch a rung
    that silently kept its native size.

    Order-independent, because the loop nests k > target > pass > rung and a
    positional slice would silently assume otherwise. Per target and horizon:
    `shipped` contributes each native width once, `down` contributes 512 four
    times and `up` contributes 2048 four times. This counts widths over the
    whole run and cannot see WHICH pass or rung got which -- the label-aware test
    above does that -- but it is recorded from the fixture's REAL ridge run, so
    it also proves the probe accepts every block this script builds.

    The run covers two horizons (`1` and `CONTRAST_K`), so every count is twice
    what one horizon contributes."""
    seen = real_run_widths
    per_horizon = {2048: 10, 512: 10, 1024: 2, 1536: 2}
    assert Counter(width for width, _ in seen) == {w: 2 * n for w, n in per_horizon.items()}
    assert {base for _, base in seen} == {RUNG_WIDTH["two_frame"]}, (
        "the base enc(t) is 2048 wide and is never projected, in any pass"
    )
    assert UP_WIDTH == 2048 and DOWN_WIDTH == 512


def test_the_bootstrap_groups_are_episodes_not_windows(monkeypatch):
    """The bootstrap groups on `episode`, never `window`. The fixture's episodes
    are `window // 2`, so the two labellings differ and the wrong one cannot be
    handed in unnoticed."""
    _stub_blocks(monkeypatch)
    gathers = dict(zip(("fit", "select", "score"), _splits()))
    out = script.cell_passes(*gathers.values(), h_dim=512, ks=(4, CONTRAST_K))
    score = gathers["score"]
    for k in (4, CONTRAST_K):
        _, _, _, rows = _expected(score, "translation", k, "two_frame", "shipped")
        assert not np.array_equal(score["episode"][rows], score["window"][rows])
        for pass_name in PASSES:
            for target in TARGETS:
                for rung in RUNGS:
                    got = out["passes"][pass_name][target][f"k{k}"][rung]
                    assert np.array_equal(got["groups"], score["episode"][rows])
    assert np.array_equal(out["contrast"][KEY]["groups"], score["episode"][
        _expected(score, "translation", CONTRAST_K, "two_frame", "down")[3]
    ])


def test_the_contrast_pairs_the_down_passs_two_frame_against_its_deterministic(monkeypatch):
    """`contrast_from_blocks(a, b)` is `a - b`, and Reading F reads a positive
    contrast as PAST_FRAME_AHEAD -- so `a` MUST be `two_frame`. Pinned by content
    on all three splits, against the down pass's own blocks."""
    _stub_blocks(monkeypatch)
    gathers = dict(zip(("fit", "select", "score"), _splits()))
    out = script.cell_passes(*gathers.values(), h_dim=512, ks=(4, CONTRAST_K, 1))
    assert set(out["contrast"]) == {KEY}, "one horizon, and it is CONTRAST_K -- not ks[0], not ks[-1]"
    got = out["contrast"][KEY]
    for role, rung in (("a", "two_frame"), ("b", "deterministic")):
        for name, split in zip(gathers, got[role]):
            base, block, values, _ = _expected(
                gathers[name], "translation", CONTRAST_K, rung, "down",
            )
            assert split.block.shape[1] == DOWN_WIDTH
            assert np.array_equal(split.base, base), (role, name)
            assert np.array_equal(split.block, block), (role, name)
            assert np.array_equal(split.target, values), (role, name)
    # The two arms differ on this fixture, so the check above could fail.
    assert not np.array_equal(got["a"][2].block, got["b"][2].block)


def test_no_contrast_is_taken_when_the_contrast_horizon_is_not_measured(monkeypatch):
    """A smoke run at other horizons has no Reading F input and says so with an
    empty dict, rather than contrasting at whichever horizon it did measure."""
    _stub_blocks(monkeypatch)
    out = script.cell_passes(*_splits(), h_dim=512, ks=(1, 4))
    assert out["contrast"] == {}
    assert set(out["rows"]) == {"k1", "k4"}


def test_every_probe_call_gets_the_cells_own_bootstrap_seed(monkeypatch):
    """`gain_from_blocks` and `contrast_from_blocks` both default their `seed`
    to 0, so a caller that never passes one hands every cell, pass, rung, target
    and horizon the SAME bootstrap draw -- correlating interval noise across
    CELLS that the seeds x arms agreement rule treats as independent, and that
    rule carries this milestone's whole multiple-comparison burden. Both entry
    points, every call."""
    _stub_blocks(monkeypatch)
    out = script.cell_passes(*_splits(), h_dim=512, seed=17, ks=(1, CONTRAST_K))
    seeds = Counter(
        got["seed"]
        for by_target in out["passes"].values()
        for by_k in by_target.values()
        for by_rung in by_k.values()
        for got in by_rung.values()
    )
    assert seeds == {17: len(PASSES) * len(TARGETS) * 2 * len(RUNGS)}, seeds
    assert out["contrast"][KEY]["seed"] == 17


def test_the_cells_seed_moves_the_bootstrap_and_never_the_projection(monkeypatch):
    """`seed` is the CELL's, and is for the bootstrap only. The projection is one
    fixed matrix shared by all nine cells: were it drawn from the cell seed, no
    two cells' gains would be comparable and an arm could win on a lucky matrix.
    Two cells on identical data with different seeds must therefore be handed
    BYTE-IDENTICAL blocks and differ only in the seed they resample with."""
    _stub_blocks(monkeypatch)
    splits = _splits()
    first = script.cell_passes(*splits, h_dim=512, seed=1, ks=(CONTRAST_K,))
    second = script.cell_passes(*splits, h_dim=512, seed=2, ks=(CONTRAST_K,))
    for pass_name in PASSES:
        for target in TARGETS:
            for rung in RUNGS:
                a = first["passes"][pass_name][target][KEY][rung]
                b = second["passes"][pass_name][target][KEY][rung]
                assert (a["seed"], b["seed"]) == (1, 2)
                for x, y in zip(a["splits"], b["splits"]):
                    assert np.array_equal(x.block, y.block), (pass_name, target, rung)
    assert first["contrast"][KEY]["seed"] != second["contrast"][KEY]["seed"]


def test_a_missing_select_split_reaches_every_fit_as_none(monkeypatch):
    """A training pool with nothing beyond `FIT_EPISODES` has no selection split.
    `gain_from_blocks` takes `select=None` and falls back to a fixed ridge, so
    `cell_passes` must hand `None` through -- for the contrast as well -- rather
    than crash or fabricate a split."""
    _stub_blocks(monkeypatch)
    fit, _, score = _splits()
    out = script.cell_passes(fit, None, score, h_dim=512, ks=(CONTRAST_K,))
    for pass_name in PASSES:
        for rung in RUNGS:
            got = out["passes"][pass_name]["translation"][KEY][rung]
            assert got["splits"][1] is None
            assert got["splits"][0] is not None and got["splits"][2] is not None
    assert out["contrast"][KEY]["a"][1] is None and out["contrast"][KEY]["b"][1] is None


def test_neither_cell_passes_nor_base_control_reads_the_predicted_embedding():
    """`encoder_embedding` is the base; `embedding` is the model's PREDICTED
    embedding, and comparing against it asks whether the latent beats its own
    head's reconstruction -- which a latent can win while carrying no history.
    Byte-identical arrays in a fixture once made this swap invisible, so the
    proof here is stronger than a numeric difference: the key is REMOVED from
    every gather, and both functions must still run."""
    splits = _splits(enc=64, h_dim=16, z=32)
    for data in splits:
        del data["embedding"]
    script.cell_passes(*splits, h_dim=16, ks=(1,))
    script.base_control(*splits)


def test_cell_passes_refuses_a_horizon_no_window_is_long_enough_for():
    """A k with no rows is a protocol refusal, not an empty table."""
    with pytest.raises(ValueError, match="no window is long enough"):
        script.cell_passes(
            *_splits(steps=6, enc=64, h_dim=16, z=32), h_dim=16, ks=(15,),
        )


def test_resamples_defaults_to_the_protocols_and_reaches_every_probe_call(monkeypatch):
    """`resamples` is the draw count of every interval this script reports, and
    the protocol's figure is `retention.RESAMPLES`. The default must BE that
    figure, and an explicit value must reach the contrast as well as the gains --
    the contrast is the reading itself, and a parameter threaded only into
    `gain_from_blocks` would leave its interval on the default while a test
    believed it had lowered it."""
    _stub_blocks(monkeypatch)

    def draws(out):
        return {
            got["resamples"]
            for by_target in out["passes"].values()
            for by_k in by_target.values()
            for by_rung in by_k.values()
            for got in by_rung.values()
        } | {got["resamples"] for got in out["contrast"].values()}

    default = script.cell_passes(*_splits(), h_dim=512, ks=(CONTRAST_K,))
    assert draws(default) == {RESAMPLES} and KEY in default["contrast"]
    few = script.cell_passes(*_splits(), h_dim=512, ks=(CONTRAST_K,), resamples=7)
    assert draws(few) == {7} and KEY in few["contrast"]


def test_every_gain_at_one_target_and_horizon_shares_one_read_only_base(monkeypatch):
    """The base is `enc(t)` on the target's rows and NO pass or rung touches it,
    so it is materialised ONCE per `(target, k, split)` and every fit is handed
    that one object -- not a copy per rung and per pass, which at the production
    sizes was a fresh ~500 MB float64 array each time. Content equality cannot
    see the difference (`test_every_gain_gets_the_untouched_base...` passes on
    copies just as well), so this asks for IDENTITY, across all twelve gains and
    the contrast.

    Read-only, because twelve consumers hold it: one that wrote to it would
    corrupt the other eleven, and would do so silently."""
    _stub_blocks(monkeypatch)
    out = script.cell_passes(*_splits(), h_dim=512, ks=(1, CONTRAST_K))
    for target in TARGETS:
        for k in ("k1", KEY):
            fits = [
                out["passes"][p][target][k][r]["splits"] for p in PASSES for r in RUNGS
            ]
            if k == KEY and target == "translation":
                fits += [out["contrast"][KEY]["a"], out["contrast"][KEY]["b"]]
            for i, name in enumerate(("fit", "select", "score")):
                first = fits[0][i].base
                assert all(f[i].base is first for f in fits), (target, k, name)
                assert not first.flags.writeable, (target, k, name)


def test_only_one_rungs_blocks_exist_at_a_time_when_no_contrast_is_pending(monkeypatch):
    """A rung's splits are built, fit and let go before the next rung's are
    built -- building all four first held ~1 GB of blocks the fits never needed
    at once. The stub carries a weak reference to every block the probe has been
    handed, and each new call asserts that ALL the earlier ones have been freed.

    `ks=(1,)` so no contrast is pending: at `CONTRAST_K` the down pass's
    `two_frame` and `deterministic` splits are deliberately kept for the
    contrast, and `test_the_contrast_pairs...` pins that they are the right
    ones."""
    seen = []

    def gain(fit, select, score, *, groups, resamples, confidence, seed):
        alive = [ref for ref in seen if ref() is not None]
        assert not alive, f"{len(alive)} block(s) from earlier fits are still alive"
        seen.extend(weakref.ref(split.block) for split in (fit, select, score))
        return {"gain": 0.0}

    monkeypatch.setattr(script, "gain_from_blocks", gain)
    monkeypatch.setattr(script, "contrast_from_blocks", lambda *a, **k: {"contrast": 0.0})
    script.cell_passes(*_splits(enc=64, h_dim=16, z=32), h_dim=16, ks=(1,))
    assert len(seen) == len(PASSES) * len(TARGETS) * len(RUNGS) * 3
    # The in-call assertion runs at the START of each fit, so it never checks
    # the LAST rung's blocks -- an implementation that held the final rung's
    # splits on the returned structure would pass every one of those and leave
    # 250 MB resident per pass at production width. Checked here, after the
    # return, where the only live reference could be one `cell_passes` kept.
    import gc
    gc.collect()
    alive = [ref for ref in seen if ref() is not None]
    assert not alive, (
        f"{len(alive)} block(s) are still alive after cell_passes returned; the "
        "last rung's splits were retained past their fit"
    )


def _dropping_builder(real, *, at_k: int):
    """A `TARGET_BUILDERS` entry with a row rule of its own: at `at_k` it drops
    the first row `real` returns, and elsewhere it is `real`. `values` and `rows`
    are shortened together, so what it hands back is internally consistent."""
    def builder(targets, window, step, k):
        values, rows = real(targets, window, step, k)
        return (values[1:], rows[1:]) if k == at_k else (values, rows)
    return builder


def _permuting_builder(real, *, at_k: int):
    """A `TARGET_BUILDERS` entry that keeps every row and reorders them: at
    `at_k` it swaps the first two rows `real` returns, and elsewhere it is
    `real`. `values` and `rows` move TOGETHER, so what it hands back is
    internally consistent AND has the same count -- which is exactly why no
    count can see it."""
    def builder(targets, window, step, k):
        values, rows = real(targets, window, step, k)
        if k != at_k:
            return values, rows
        order = np.arange(rows.size)
        order[0], order[1] = 1, 0
        return values[order], rows[order]
    return builder


@pytest.mark.parametrize("permuting", TARGETS)
def test_cell_passes_refuses_a_builder_whose_rows_are_reordered(monkeypatch, permuting):
    """The finer half of the row-selection contract, and the one no count can
    see. `source` -- each row's partner k steps earlier, which is what
    `two_frame` reads -- is derived in `_target_rows` INDEPENDENTLY of the
    builder. Pairing row i of the target with row i of `source` is only valid
    while the two are the same selection in the SAME ORDER.

    A builder that returns the same COUNT of rows permuted therefore clears the
    row-count agreement check, leaves `rows` in the record untouched, and
    silently misaligns `two_frame`'s block against its own target: a wrong
    number in the arm this milestone decides on, with no refusal anywhere. It is
    the same shape as the dropping case one column over, and the count guard
    that catches the dropping case is blind to it.

    The probe is stubbed: the refusal is made from the row selections, before
    the first fit it would have wasted. k = 1 agrees and is fully measured."""
    probed = []
    monkeypatch.setattr(script, "gain_from_blocks", lambda *a, **k: probed.append("gain"))
    monkeypatch.setattr(
        script, "contrast_from_blocks", lambda *a, **k: probed.append("contrast"),
    )
    monkeypatch.setitem(
        script.TARGET_BUILDERS, permuting,
        _permuting_builder(script.TARGET_BUILDERS[permuting], at_k=4),
    )
    with pytest.raises(ValueError, match=r"k=4, target '" + permuting) as raised:
        script.cell_passes(*_splits(enc=64, h_dim=16, z=32), h_dim=16, ks=(1, 4))
    assert "not `shifted_rows`' rows" in str(raised.value)
    assert "disagree on scored row count" not in str(raised.value), (
        "the count guard cannot see a permutation; this must be the order guard"
    )
    assert probed == ["gain"] * (len(PASSES) * len(TARGETS) * len(RUNGS)), (
        "k = 1 agrees and is fully measured; nothing at k = 4 may be fit first"
    )


@pytest.mark.parametrize("diverging", TARGETS)
def test_cell_passes_refuses_targets_that_disagree_on_the_scored_row_count(
    monkeypatch, diverging,
):
    """`rows` is the figure the record reports and Task 7 cites as its acceptance
    number, and every target at one k is supposed to share one row set. A target
    whose builder has a row rule of its own must be REFUSED, not have its count
    quietly reported under the key the other target also writes -- which is what
    happened while the assignment sat inside the target loop and the last write
    won.

    The probe is stubbed on purpose. `GainSplit.rows()` would incidentally
    refuse a row-DROPPING builder, because `two_frame`'s block is read from
    `shifted_rows`' `source` and no longer matches the shortened base -- but
    with a message that names no target, only when that fit is reached, and
    only while `rung_block` keeps that coupling. This guard must not lean
    on it.

    The divergence is at k = 4 only, and either target may be the odd one out:
    the check is per horizon and symmetric in the targets. Every gain of the
    agreeing k = 1 is taken, and none of k = 4's -- the refusal is made from the
    row selections, before the first fit it would have wasted."""
    probed = []
    monkeypatch.setattr(script, "gain_from_blocks", lambda *a, **k: probed.append("gain"))
    monkeypatch.setattr(
        script, "contrast_from_blocks", lambda *a, **k: probed.append("contrast"),
    )
    monkeypatch.setitem(
        script.TARGET_BUILDERS, diverging,
        _dropping_builder(script.TARGET_BUILDERS[diverging], at_k=4),
    )
    with pytest.raises(ValueError, match=r"k=4: targets disagree on scored row count") as raised:
        script.cell_passes(*_splits(enc=64, h_dim=16, z=32), h_dim=16, ks=(1, 4))
    # 8 windows of 20 steps: 8 * (20 - 4) = 128 rows, and 127 for the one that dropped.
    assert "128" in str(raised.value) and "127" in str(raised.value)
    assert probed == ["gain"] * (len(PASSES) * len(TARGETS) * len(RUNGS)), (
        "k = 1 agrees and is fully measured; nothing at k = 4 may be fit first"
    )


# ---------------------------------------------------------------------------
# anchor_check
# ---------------------------------------------------------------------------


def _synthetic_passes(ks=(1, 4, CONTRAST_K)):
    """A `passes` dict of the right shape whose anchors reproduce, built with no
    ridge solve so the parametrised tests below cost nothing. Every gain is
    distinct so a comparison against the wrong entry cannot pass by equality."""
    passes = {}
    for p, pass_name in enumerate(PASSES):
        passes[pass_name] = {}
        for t, target in enumerate(TARGETS):
            passes[pass_name][target] = {}
            for k in ks:
                passes[pass_name][target][f"k{k}"] = {
                    rung: {"gain": 0.001 * (1 + r) + 0.01 * t + 0.0001 * k + 0.5 * p}
                    for r, rung in enumerate(RUNGS)
                }
    for pass_name, rung in ANCHOR.items():
        for target in TARGETS:
            for k in ks:
                passes[pass_name][target][f"k{k}"][rung] = copy.deepcopy(
                    passes["shipped"][target][f"k{k}"][rung]
                )
    return passes


def test_anchor_check_accepts_reproducing_anchors():
    assert script.anchor_check(_synthetic_passes()) == {"down": True, "up": True}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("key", ["k1", "k4", KEY])
@pytest.mark.parametrize("broken", sorted(ANCHOR))
def test_anchor_check_reports_a_break_at_any_target_and_horizon_and_only_that_pass(
    broken, key, target,
):
    """A one-part-in-a-billion move in ONE anchor gain, at ONE target and ONE
    horizon, must read as that pass broken and the other pass fine. Every
    horizon and both targets are checked because a projection that leaks at one
    and not another must not hide -- an implementation that inspected only
    translation, or only `CONTRAST_K`, fails the rows it skips."""
    passes = _synthetic_passes()
    passes[broken][target][key][ANCHOR[broken]]["gain"] += 1e-9
    expected = {name: name != broken for name in ANCHOR}
    assert script.anchor_check(passes) == expected


def test_anchor_check_is_exact_not_approximate():
    """`isclose` would accept a matmul's last-ulp drift, which is exactly the
    evidence that a projection ran where it should not."""
    passes = _synthetic_passes()
    gain = passes["down"]["translation"][KEY][ANCHOR["down"]]["gain"]
    passes["down"]["translation"][KEY][ANCHOR["down"]]["gain"] = np.nextafter(gain, 1.0)
    assert script.anchor_check(passes) == {"down": False, "up": True}


def test_anchor_check_works_when_the_contrast_horizon_is_not_measured():
    """A smoke run whose `ks` omit `CONTRAST_K` still checks its anchors, at the
    horizons it did measure, instead of raising a KeyError."""
    assert script.anchor_check(_synthetic_passes(ks=(1,))) == {"down": True, "up": True}


def test_anchor_check_refuses_to_pass_on_nothing():
    """Zero measured horizons is no evidence that the anchors reproduce, and
    `all([])` is True."""
    with pytest.raises(ValueError, match="no horizon"):
        script.anchor_check(_synthetic_passes(ks=()))


def test_anchor_check_reads_the_shipped_pass_it_is_given():
    """`shipped` is the reference, not `down` or `up` against each other."""
    passes = _synthetic_passes()
    passes["shipped"]["rotation"]["k4"][ANCHOR["up"]]["gain"] += 1e-9
    assert script.anchor_check(passes) == {"down": True, "up": False}


# ---------------------------------------------------------------------------
# base_control
# ---------------------------------------------------------------------------


def test_base_control_reports_the_level_not_a_gain():
    """`enc(t)` -> absolute position. There is nothing to take an increment
    over: this IS the arm the increments are measured against."""
    control = script.base_control(*_splits())
    assert control["position_r2"] > 0.5, (
        "enc(t) carries position exactly in this fixture; a low position_r2 "
        "means the control is reading the wrong array"
    )
    assert "gain" not in control, "the base control is a level, not a gain"
    assert "r2" in control and "per_column_r2" in control


def test_base_control_probes_the_raw_encoder_embedding_not_the_predicted_one():
    """Recomputed independently from `encoder_embedding` through the same
    `fit_probe` / `probe_r2` calls, and pinned exactly -- while the same
    computation on `embedding` is shown to give a DIFFERENT number, so the
    swap is catchable at all."""
    from mbfps.eval.probe import fit_probe, probe_r2

    fit, select, score = _splits(enc=64, h_dim=16, z=32)
    control = script.base_control(fit, select, score)
    probe = fit_probe(
        fit["encoder_embedding"], fit["targets"],
        select["encoder_embedding"], select["targets"],
    )
    expected = probe_r2(probe, score["encoder_embedding"], score["targets"])
    assert control["r2"] == pytest.approx(expected)
    assert control["position_r2"] == pytest.approx(
        probe_r2(probe, score["encoder_embedding"], score["targets"][:, :2])
    )
    wrong_probe = fit_probe(
        fit["embedding"], fit["targets"], select["embedding"], select["targets"],
    )
    wrong = probe_r2(wrong_probe, score["embedding"], score["targets"])
    assert expected != pytest.approx(wrong)


def test_base_control_records_position_alone_beside_the_four_column_mean():
    """M3k gates on position alone (spec 2.5), but the 4-column mean is kept so
    the record bridges to M3j's. Position reads exactly here and heading is
    noisy, so the two differ -- a `position_r2` that was really the mean, or a
    mean that was really position, fails."""
    control = script.base_control(*_splits(enc=64, h_dim=16, z=32))
    assert len(control["per_column_r2"]) == 4
    assert control["position_r2"] == pytest.approx(sum(control["per_column_r2"][:2]) / 2)
    assert control["r2"] == pytest.approx(sum(control["per_column_r2"]) / 4)
    assert control["position_r2"] > control["r2"] + 0.05
    assert control["rows"] == 160


def test_base_control_without_a_select_split_takes_the_default_ridge():
    """A training pool with nothing beyond `FIT_EPISODES` has no selection split;
    `fit_probe` then takes its default penalty, and the record says the ridge was
    not selected."""
    fit, _, score = _splits(enc=64, h_dim=16, z=32)
    control = script.base_control(fit, None, score)
    assert control["ridge_selected"] is False and control["ridge"] == 1e3
    assert control["position_r2"] > 0.5
    selected = script.base_control(*_splits(enc=64, h_dim=16, z=32))
    assert selected["ridge_selected"] is True


def test_base_control_reports_a_constant_column_as_nan_not_as_a_missing_one():
    """`probe._mean_r2` silently drops a zero-variance column from its average;
    `per_column_r2` keeps the slot and says NaN, so the position/heading split
    can be read column by column and a vanished column is visible."""
    fit, select, score = _splits(enc=64, h_dim=16, z=32)
    score["targets"] = score["targets"].copy()
    score["targets"][:, 3] = 1.0
    control = script.base_control(fit, select, score)
    assert len(control["per_column_r2"]) == 4
    assert np.isnan(control["per_column_r2"][3])
    assert all(np.isfinite(control["per_column_r2"][:3]))
    assert np.isfinite(control["r2"]) and np.isfinite(control["position_r2"])


# ---------------------------------------------------------------------------
# gather_three_splits
# ---------------------------------------------------------------------------


def _prepared():
    return types.SimpleNamespace(
        model="the-model",
        common={"feature_backbone": "the-backbone", "device": "cpu"},
        context=5, horizon=45,
    )


def test_gather_three_splits_takes_the_filtering_gain_draws(monkeypatch):
    """The three gathers carry the exact `(paths, seed)` pairs
    `probe.filtering_gain` uses: fit on `train[:FIT_EPISODES]` at `seed`, select
    on the NEXT `SELECT_EPISODES` at `seed + 2`, score on `val` at `seed + 1`.
    A transposed seed would score a different draw from criterion 4's while
    producing entirely plausible numbers."""
    calls = []

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        calls.append({"paths": list(paths), "seed": seed, "limit": limit,
                      "model": model, "backbone": backbone, "context": context,
                      "horizon": horizon})
        return _gathered(enc=4, h_dim=3, z=2)

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    train = [f"t{i}" for i in range(50)]
    val = [f"v{i}" for i in range(10)]
    script.gather_three_splits(_prepared(), train, val, seed=7)
    fit, select, score = calls
    assert fit["paths"] == train[:script.FIT_EPISODES] and fit["seed"] == 7
    assert select["paths"] == train[
        script.FIT_EPISODES:script.FIT_EPISODES + script.SELECT_EPISODES
    ]
    assert select["seed"] == 9
    assert score["paths"] == val and score["seed"] == 8
    assert fit["limit"] == script.FIT_EPISODES and select["limit"] == script.SELECT_EPISODES
    # The scored split's cap is ITS OWN, the number of validation episodes -- and
    # 10 is not `FIT_EPISODES`, so a scored split that reused the fit split's cap
    # reads 20 here and fails. (`test_gather_three_splits_scores_every_validation_episode`
    # asserts the same property on what the gather CONTAINS, with 24 episodes.)
    assert score["limit"] == len(val) != script.FIT_EPISODES
    assert {c["model"] for c in calls} == {"the-model"}
    assert {c["backbone"] for c in calls} == {"the-backbone"}
    assert {(c["context"], c["horizon"]) for c in calls} == {(5, 45)}


@pytest.mark.parametrize("n_val", [10, 24])
def test_gather_three_splits_scores_every_validation_episode(monkeypatch, n_val):
    """M3j's smoke caught this: `filtering_gain`'s `limit` caps the fit split AND
    the scored split at the same number, so mirroring it discarded four of the
    24 validation episodes -- 17% of the evaluation data.

    Two fixtures, because ONE cannot fail. With 10 validation episodes a cap of
    `FIT_EPISODES` (20) never binds inside `gather_probe_data` itself, and a
    `min(FIT_EPISODES, len(val))` regression reads identically to the fix. With
    24 it binds. So the recorder also emulates the real function's cap
    (`paths[:limit]`) and the assertion is on the EPISODES THE SCORED GATHER
    ACTUALLY CONTAINS, which is what the reading is taken on. The bug this
    catches is the 24 case; the 10 case catches the plain shared cap."""
    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        kept = list(paths)[:limit]          # `gather_probe_data`'s own cap
        return _gathered(n_windows=2 * len(kept), steps=6, enc=4, h_dim=3, z=2)

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    train = [f"t{i}" for i in range(50)]
    val = [f"v{i}" for i in range(n_val)]
    _, _, score = script.gather_three_splits(_prepared(), train, val, seed=0)
    assert np.unique(score["episode"]).size == n_val, (
        f"the scored split holds {np.unique(score['episode']).size} of {n_val} "
        "validation episodes; the reading must be taken on all of them"
    )


def test_gather_three_splits_has_no_select_split_when_the_pool_ends_at_the_fit_set(monkeypatch):
    """`select` is None when there is nothing beyond `FIT_EPISODES`, and the two
    gathers that remain are still the fit and the score."""
    calls = []
    monkeypatch.setattr(
        script, "gather_probe_data",
        lambda model, paths, *a, **k: calls.append(list(paths)) or _gathered(enc=4, h_dim=3, z=2),
    )
    train = [f"t{i}" for i in range(script.FIT_EPISODES)]
    fit, select, score = script.gather_three_splits(_prepared(), train, ["v0", "v1"], seed=0)
    assert select is None and fit is not None and score is not None
    assert calls == [train, ["v0", "v1"]]


def test_gather_three_splits_refuses_an_empty_training_pool():
    with pytest.raises(ValueError, match="no training episodes"):
        script.gather_three_splits(_prepared(), [], ["v0"], seed=0)


# ---------------------------------------------------------------------------
# measure_cell: the wiring `cell_passes` cannot see.
# ---------------------------------------------------------------------------


def _measure_args(source="runs/m3_study_v2", out="runs/m3k_retention", **extra):
    """The attributes `measure_cell` reads off its args. `source` is the STUDY
    directory: `prepare_cell` loads the checkpoint from the `out` of the args it
    is handed, so passing this script's own `--out` through would hunt for the
    nine M3c checkpoints among the width records."""
    return types.SimpleNamespace(
        source=source, out=out, device="cpu", context=None, horizon=None, **extra,
    )


def _cell(seed=3):
    return types.SimpleNamespace(
        arm="pixel_ae", seed=seed, record={"steps": 20000, "git_sha": "ca3e140"},
        diagnostic={"whatever": True},
    )


class _Check:
    def __init__(self, ok):
        self.ok = ok

    def failures(self):
        return ["windows do not match the diagnostic"]

    def record(self):
        return {"ok": self.ok}


def _prepared_model(h_dim=512):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            rssm=types.SimpleNamespace(cfg=types.SimpleNamespace(h_dim=h_dim))
        ),
        embedding_probe=object(),
        common={"feature_backbone": "the-backbone", "device": "cpu"},
        context=5, horizon=45,
    )


def test_measure_cell_loads_the_checkpoint_from_source_not_out(monkeypatch):
    """`prepare_cell` loads the checkpoint from the `out` of the args IT is
    handed (`trust_horizon.load_checkpoint_model(args.out, ...)`), which is the
    STUDY directory -- while this script's own `--out` holds the width records.
    Handing `args` through unchanged hunts for the nine M3c checkpoints among
    the records and refuses every cell, so a `measure` run fails on its first
    cell with a refusal that names the wrong directory.

    Nothing else catches this: every other test stubs `prepare_cell` or feeds
    `cell_passes` directly, so the one line that routes the directories is
    exercised only here. `context`, `horizon` and `device` are read off the same
    shim by `prepare_cell`'s protocol check, so they are pinned with it: a shim
    that dropped `--context` would silently ignore the override."""
    seen = {}

    def recording_prepare_cell(cell_args, cell, device, train, val):
        seen["out"] = str(cell_args.out)
        seen["device"] = cell_args.device
        seen["context"] = cell_args.context
        seen["horizon"] = cell_args.horizon
        return script.EXIT_SELF_CHECK_FAILED, None

    monkeypatch.setattr(script, "prepare_cell", recording_prepare_cell)
    args = _measure_args(source="runs/m3_study_v2", out="runs/m3k_retention")
    args.context, args.horizon = 7, 9
    status, record = script.measure_cell(args, _cell(), "cpu", [], [])

    assert status == script.EXIT_SELF_CHECK_FAILED and record is None
    assert seen["out"] == "runs/m3_study_v2", (
        f"prepare_cell was handed out={seen['out']!r}; it must be the STUDY "
        "directory (--source), never this script's record directory (--out)"
    )
    assert seen["out"] != args.out
    assert (seen["device"], seen["context"], seen["horizon"]) == ("cpu", 7, 9)


def _stub_measure(monkeypatch, *, self_check_ok=True, h_dim=512, calls=None):
    """`prepare_cell`, the reference pass and `self_check` replaced; the gathers
    are `_gathered` at production widths, counted in `calls`, and everything
    downstream of them is the REAL code."""
    calls = [] if calls is None else calls
    prepared = _prepared_model(h_dim)

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        calls.append(seed)
        return _gathered(seed=seed)

    monkeypatch.setattr(script, "prepare_cell", lambda *a, **k: (script.EXIT_OK, prepared))
    monkeypatch.setattr(script, "reference_trajectories", lambda *a, **k: "a-trajectory")
    monkeypatch.setattr(script, "self_check", lambda traj, diagnostic: _Check(self_check_ok))
    monkeypatch.setattr(script, "gather_probe_data", recorder)
    return calls


def test_measure_cell_refuses_a_failed_self_check_before_any_gather(monkeypatch):
    """The 30 check is NOT free the way 12/14 are: it reduces per-window rows a
    regression could get wrong while `prepare_cell`'s own 14 still passes on the
    mean curve. It must refuse BEFORE the gathers -- the expensive part -- and
    write no record."""
    calls = _stub_measure(monkeypatch, self_check_ok=False)
    status, record = script.measure_cell(
        _measure_args(), _cell(), "cpu", [f"t{i}" for i in range(25)],
        [f"v{i}" for i in range(24)], ks=(CONTRAST_K,),
    )
    assert status == script.EXIT_SELF_CHECK_FAILED and record is None
    assert calls == [], "a cell that failed its self-check still paid for a gather"


def test_measure_cell_gathers_once_and_records_all_three_passes(monkeypatch, tmp_path):
    """End to end at production widths, with the real `cell_passes`,
    `anchor_check` and `base_control`.

    ONE GATHER SERVES ALL THREE PASSES -- the reason this is not three times
    M3j's cost. `gather_three_splits` is pass-agnostic, so counting ITS gathers
    proves nothing; the property lives in `measure_cell`, which is what could
    call it once per pass. Three gathers per CELL, not nine.

    The record schema is what Task 6 reads, so its key set is pinned exactly,
    and the two anchors must come back True on real numbers."""
    calls = []
    _stub_measure(monkeypatch, calls=calls)
    seen = {}
    real = script.cell_passes

    def spy(*args, **kwargs):
        seen.update(kwargs)
        # What `measure_cell` passed is recorded ABOVE; the draw count below is
        # this test's own, layered on afterwards, so the assertion that
        # `measure_cell` passes none is about production and not about the test.
        return real(*args, **dict(kwargs, resamples=FEW_RESAMPLES))

    monkeypatch.setattr(script, "cell_passes", spy)
    train = [Path(f"t{i}.npz") for i in range(25)]
    val = [Path(f"v{i}.npz") for i in range(24)]
    status, record = script.measure_cell(
        _measure_args(), _cell(seed=3), "cpu", train, val, ks=(CONTRAST_K,),
    )
    assert status == script.EXIT_OK
    assert sorted(calls) == [3, 4, 5], "fit, score and select draws -- once each, per cell"
    assert seen["seed"] == 3 and seen["h_dim"] == 512
    assert "resamples" not in seen, (
        "measure_cell must leave the draw count to cell_passes' default, which "
        "is the protocol's RESAMPLES -- a production run cannot depart from it"
    )
    assert set(record) == {
        "arm", "seed", "step", "record_git_sha", "device", "context", "horizon",
        "h_dim", "ks", "split_seed", "torch_version", "passes", "contrast", "rows",
        "anchors", "base_control", "projection_seed", "rung_width", "clusters",
        "windows", "self_check", "episodes",
    }
    assert (record["arm"], record["seed"], record["step"]) == ("pixel_ae", 3, 20000)
    assert record["record_git_sha"] == "ca3e140"
    assert record["ks"] == [CONTRAST_K] and record["h_dim"] == 512
    assert set(record["passes"]) == set(PASSES) and set(record["contrast"]) == {KEY}
    assert record["anchors"] == {"down": True, "up": True}
    assert record["projection_seed"] == PROJECTION_SEED
    assert record["rung_width"] == RUNG_WIDTH
    assert record["clusters"] == 4 and record["rows"] == {KEY: 40}
    assert len(record["windows"]["episode"]) == len(record["windows"]["window"]) == 160
    assert record["self_check"] == {"ok": True}
    assert record["episodes"] == {
        "fit": [p.name for p in train[:script.FIT_EPISODES]],
        "select": [p.name for p in train[
            script.FIT_EPISODES:script.FIT_EPISODES + script.SELECT_EPISODES
        ]],
        "val": [p.name for p in val],
    }
    assert record["base_control"]["position_r2"] > 0.5

    # THE RECORD MUST SURVIVE `write_record`. It is written only after every
    # gather and fit of the cell, so a value `write_record` cannot serialise
    # (a numpy scalar, a non-string key) would surface as a crash at the END of
    # the first ~3-minute cell of Task 7's smoke rather than here.
    from mbfps.eval.study import load_record, write_record
    path = tmp_path / "record.json"
    write_record(path, dict(record, git_sha="deadbeef"))
    loaded = load_record(path)
    assert loaded["anchors"] == {"down": True, "up": True}
    assert loaded["rung_width"] == RUNG_WIDTH and loaded["projection_seed"] == PROJECTION_SEED
    for pass_name in PASSES:
        assert loaded["passes"][pass_name]["translation"][KEY]["two_frame"]["gain"] == (
            record["passes"][pass_name]["translation"][KEY]["two_frame"]["gain"]
        )
    assert loaded["contrast"][KEY]["contrast"] == record["contrast"][KEY]["contrast"]


def test_measure_cell_records_a_broken_anchor_instead_of_hiding_or_raising(monkeypatch):
    """A cell whose anchor fails is still a record: `read` decides what a broken
    `down` anchor means (42), and a `measure` that raised would take the
    evidence of WHICH pass broke with it. `cell_passes` is replaced by a
    hand-built `passes` dict with one anchor moved, so this costs no ridge solve
    and the recorded `anchors` can only have come from `anchor_check`."""
    _stub_measure(monkeypatch)
    passes = _synthetic_passes(ks=(CONTRAST_K,))
    passes["down"]["translation"][KEY][ANCHOR["down"]]["gain"] += 1e-9
    monkeypatch.setattr(
        script, "cell_passes",
        lambda *a, **k: {"passes": passes, "contrast": {}, "rows": {KEY: 40}},
    )
    status, record = script.measure_cell(
        _measure_args(), _cell(), "cpu", [Path(f"t{i}.npz") for i in range(25)],
        [Path(f"v{i}.npz") for i in range(24)], ks=(CONTRAST_K,),
    )
    assert status == script.EXIT_OK
    assert record["anchors"] == {"down": False, "up": True}
    assert record["passes"] is passes


# ---------------------------------------------------------------------------
# measure_phase
# ---------------------------------------------------------------------------


def _phase_record(cell, anchors=None):
    return {
        "arm": cell.arm, "seed": cell.seed,
        "base_control": {"r2": 0.33, "position_r2": 0.65},
        "contrast": {KEY: {"contrast": 0.0123, "ci_low": 0.001, "ci_high": 0.023}},
        "anchors": {"down": True, "up": True} if anchors is None else anchors,
    }


def _stub_phase(monkeypatch, *, measured=None, written=None, loaded=None, anchors=None):
    def fake_load_cell(directory, arm, seed):
        if loaded is not None:
            loaded.append((Path(directory), arm, seed))
        return types.SimpleNamespace(arm=arm, seed=seed, record={}, diagnostic={})

    def fake_measure_cell(args, cell, device, train, val, ks=None):
        if measured is not None:
            measured.append((cell.arm, cell.seed))
        return script.EXIT_OK, _phase_record(cell, anchors)

    def fake_write_record(path, record):
        if written is not None:
            written.append((Path(path), record))
        return record

    monkeypatch.setattr(script, "load_cell", fake_load_cell)
    monkeypatch.setattr(script, "measure_cell", fake_measure_cell)
    monkeypatch.setattr(script, "write_record", fake_write_record)
    monkeypatch.setattr(script, "git_sha", lambda: "deadbeef")


def test_measure_phase_loads_cells_from_source_and_writes_records_to_out(monkeypatch, tmp_path):
    """Pinned by watching which directory each call actually used, not by
    asserting on a path string an unrelated implementation could satisfy by
    coincidence. `--out` is this script's OWN record directory -- empty until
    `measure` writes to it -- and the nine M3c checkpoints, records and
    diagnostics live in `--source`. Every record picks up the run's `git_sha`."""
    source_dir, out_dir = tmp_path / "source", tmp_path / "out"
    source_dir.mkdir()
    loaded, written = [], []
    _stub_phase(monkeypatch, loaded=loaded, written=written)
    args = types.SimpleNamespace(out=out_dir, source=source_dir, phase="measure")
    status = script.measure_phase(args, [("pixel_ae", 0)], "cpu", [], [], ks=(1,))
    assert status == script.EXIT_OK
    assert loaded == [(source_dir, "pixel_ae", 0)]
    assert [path for path, _ in written] == [script.width_record_path(out_dir, "pixel_ae", 0)]
    assert written[0][1]["git_sha"] == "deadbeef"
    assert out_dir.exists(), "the output directory must still be created"


def test_measure_phase_names_a_missing_cell_and_measures_nothing(monkeypatch, tmp_path, capsys):
    """Refuse the WHOLE run before any pass when a cell is missing (11): every
    cell is loaded first, so the 26-minute measure never starts on a grid that
    cannot finish. Caught with this script's OWN `CellMissing` -- the class
    `_sibling` loaded -- not another importer's copy of it."""
    measured, written = [], []
    _stub_phase(monkeypatch, measured=measured, written=written)

    def fake_load_cell(directory, arm, seed):
        if seed == 1:
            raise script.CellMissing("pixel_ae seed 1: no checkpoint")
        return types.SimpleNamespace(arm=arm, seed=seed, record={}, diagnostic={})

    monkeypatch.setattr(script, "load_cell", fake_load_cell)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    status = script.measure_phase(
        args, [("pixel_ae", 0), ("pixel_ae", 1)], "cpu", [], [], ks=(1,),
    )
    assert status == script.EXIT_NO_CHECKPOINTS
    assert measured == [] and written == []
    assert "pixel_ae seed 1" in capsys.readouterr().out


def test_measure_phase_stops_at_the_first_refusal_and_writes_nothing_for_it(
    monkeypatch, tmp_path,
):
    measured, written = [], []
    _stub_phase(monkeypatch, measured=measured, written=written)

    def refusing(args, cell, device, train, val, ks=None):
        measured.append((cell.arm, cell.seed))
        return script.EXIT_SELF_CHECK_FAILED, None

    monkeypatch.setattr(script, "measure_cell", refusing)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    status = script.measure_phase(
        args, [("pixel_ae", 0), ("pixel_ae", 1)], "cpu", [], [], ks=(1,),
    )
    assert status == script.EXIT_SELF_CHECK_FAILED
    assert measured == [("pixel_ae", 0)] and written == []


def test_measure_phase_prints_a_broken_anchor_loudly_and_only_a_broken_one(
    monkeypatch, tmp_path, capsys,
):
    """`up=BROKEN` is the same spelling `width.txt` will use. The record is still
    written; the line must be unmissable in a 26-minute log, and must not cry
    wolf when both anchors held."""
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    _stub_phase(monkeypatch, anchors={"down": True, "up": False})
    script.measure_phase(args, [("pixel_ae", 0)], "cpu", [], [], ks=(CONTRAST_K,))
    broken = capsys.readouterr().out
    assert "up=BROKEN" in broken and "down=ok" in broken and "down=BROKEN" not in broken
    _stub_phase(monkeypatch, anchors={"down": True, "up": True})
    script.measure_phase(args, [("pixel_ae", 0)], "cpu", [], [], ks=(CONTRAST_K,))
    fine = capsys.readouterr().out
    assert "BROKEN" not in fine and "down=ok up=ok" in fine


# ---------------------------------------------------------------------------
# The plan check, BEFORE the probe.
# ---------------------------------------------------------------------------


def test_require_readable_plan_refuses_a_plan_too_narrow_to_reach_a_verdict():
    """Reading F needs ARMS_REQUIRED arms of SEEDS_REQUIRED seeds, and
    `reading_contrast` raises when it has fewer -- AFTER all the probe work."""
    with pytest.raises(SystemExit):
        script.require_readable_plan(("pixel_ae",), (0, 1, 2))
    with pytest.raises(SystemExit):
        script.require_readable_plan(("pixel_ae", "frozen_ssl"), (0,))
    script.require_readable_plan(("pixel_ae", "frozen_ssl"), (0, 1))


@pytest.mark.parametrize("cells", [
    [("frozen_ssl", s) for s in (0, 1, 2)],                       # one arm
    [(a, 0) for a in ("frozen_ssl", "pixel_ae", "random_vit")],   # one seed
])
def test_measure_phase_all_refuses_a_narrow_plan_before_any_cell_work(
    monkeypatch, tmp_path, cells,
):
    """`--arms frozen_ssl --phase all` would otherwise do ~25 minutes of GPU work
    and then crash inside `reading_contrast`. The refusal must come before a
    single cell is loaded, let alone measured."""
    loaded, measured, written = [], [], []
    _stub_phase(monkeypatch, loaded=loaded, measured=measured, written=written)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="all")
    with pytest.raises(SystemExit):
        script.measure_phase(args, cells, "cpu", [], [], ks=(1,))
    assert loaded == [] and measured == [] and written == []


def test_measure_phase_measure_only_still_takes_a_one_cell_smoke(monkeypatch, tmp_path):
    """The refusal is for a plan `read` will follow. Task 7's smoke is
    `--phase measure --arms pixel_ae --seeds 0`, one cell, and a plan check that
    fired on every measure would make the mandated smoke impossible."""
    measured = []
    _stub_phase(monkeypatch, measured=measured)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="measure")
    assert script.measure_phase(args, [("pixel_ae", 0)], "cpu", [], [], ks=(1,)) == script.EXIT_OK
    assert measured == [("pixel_ae", 0)]


def test_measure_phase_all_takes_the_full_grid(monkeypatch, tmp_path):
    """The other side of the refusal: it must not fire on a readable plan."""
    measured = []
    _stub_phase(monkeypatch, measured=measured)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path, phase="all")
    cells = [(a, s) for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)]
    assert script.measure_phase(args, cells, "cpu", [], [], ks=(1,)) == script.EXIT_OK
    assert sorted(measured) == sorted(cells)


_ABSENT = object()


@pytest.mark.parametrize("phase", [_ABSENT, None, "", "al", "ALL", "measures", "bogus"])
def test_measure_phase_refuses_a_phase_that_is_not_one_of_the_known_ones(
    monkeypatch, tmp_path, phase,
):
    """The plan check is made only for `--phase all`, so `args.phase` is what
    decides whether a 1-arm plan is refused or run for ~25 minutes and lost to a
    traceback. Read with a default, a MISSING phase looks exactly like `measure`
    and the check is skipped without a word -- which is how it would regress if
    a caller ever built its args without one, and every test above sets `phase`
    explicitly, so none would notice.

    The plan here is the full, readable 3 x 3 grid, so `require_readable_plan`
    cannot be what refuses it: the only thing left to refuse is the phase. A
    typo of `all` (`"al"`) is the case that hurts, since it looks intentional."""
    loaded, measured, written = [], [], []
    _stub_phase(monkeypatch, loaded=loaded, measured=measured, written=written)
    args = types.SimpleNamespace(out=tmp_path / "out", source=tmp_path)
    if phase is not _ABSENT:
        args.phase = phase
    cells = [(a, s) for a in ("frozen_ssl", "pixel_ae", "random_vit") for s in (0, 1, 2)]
    with pytest.raises(SystemExit, match="unknown phase"):
        script.measure_phase(args, cells, "cpu", [], [], ks=(1,))
    assert loaded == [] and measured == [] and written == []


# ---------------------------------------------------------------------------
# require_one_protocol
# ---------------------------------------------------------------------------

_ARMS = ("frozen_ssl", "pixel_ae", "random_vit")


def _protocol_records():
    """Nine records carrying only what `require_one_protocol` reads, each with
    its OWN copies of every list and dict, so a mutation to one victim cell
    cannot alias into the rest."""
    return {
        (arm, seed): {
            "windows": {"episode": list(range(24))},
            "episodes": {"val": [f"e{i}.npz" for i in range(24)]},
            "context": 5, "horizon": 45, "device": "mps",
            "ks": [1, 4, CONTRAST_K], "torch_version": "2.13.0", "git_sha": "abc123",
            "projection_seed": PROJECTION_SEED, "rung_width": dict(RUNG_WIDTH),
            "h_dim": RUNG_WIDTH["deterministic"],
        }
        for arm in _ARMS for seed in (0, 1, 2)
    }


# Every field `require_one_protocol` compares, mapped to a mutation that changes
# it on one victim cell. Eight are `latent_retention`'s own, `git_sha` included
# (an earlier version of that function compared five and would have pooled
# records from different torch builds and code versions with no refusal); the
# last two are this milestone's: nine cells measured through different random
# matrices, or at different native widths, are not one measurement and are
# exactly what "one fixed Gaussian shared across all nine cells" forbids.
_PROTOCOL_FIELD_MUTATIONS = {
    "windows.episode": lambda r: r["windows"].update(episode=list(range(5))),
    "episodes.val": lambda r: r["episodes"].update(val=["different.npz"]),
    "context": lambda r: r.update(context=999),
    "horizon": lambda r: r.update(horizon=999),
    "device": lambda r: r.update(device="cpu"),
    "ks": lambda r: r.update(ks=[1, 4]),
    "torch_version": lambda r: r.update(torch_version="1.9.0"),
    "git_sha": lambda r: r.update(git_sha="def456"),
    "projection_seed": lambda r: r.update(projection_seed=PROJECTION_SEED + 1),
    "rung_width": lambda r: r["rung_width"].update(deterministic=RUNG_WIDTH["deterministic"] + 1),
    "h_dim": lambda r: r.update(h_dim=int(r["h_dim"]) + 1),
}


_REQUIRED_PROTOCOL_FIELDS = {
    "windows.episode", "episodes.val", "context", "horizon", "device", "ks",
    "torch_version", "git_sha", "projection_seed", "rung_width", "h_dim",
}


def test_the_protocol_table_names_every_field_the_check_compares():
    """The fields the check compares are `script._PROTOCOL_FIELDS`, the table
    `require_one_protocol` iterates -- so a field added to the comparison is a
    row there, and this fails until it has a disagreement case above. (This
    used to compare the mutation table to a literal copy of itself, which no
    edit to the function could ever move.)

    The second assertion is the other direction: a comparison REMOVED from both
    the function and the table would leave them equal, so the ten this milestone
    promised stay named here."""
    compared = [field for field, _ in script._PROTOCOL_FIELDS]
    assert len(compared) == len(set(compared)), f"a field is listed twice: {compared}"
    assert set(compared) == set(_PROTOCOL_FIELD_MUTATIONS), (
        f"compared but with no disagreement case: "
        f"{sorted(set(compared) - set(_PROTOCOL_FIELD_MUTATIONS))}; "
        f"has a case but is not compared: "
        f"{sorted(set(_PROTOCOL_FIELD_MUTATIONS) - set(compared))}"
    )
    assert _REQUIRED_PROTOCOL_FIELDS <= set(compared), (
        f"no longer compared: {sorted(_REQUIRED_PROTOCOL_FIELDS - set(compared))}"
    )


def test_require_one_protocol_compares_exactly_the_fields_it_publishes(monkeypatch):
    """The table test above is only as good as the function's use of the table.
    A function that kept a private copy of the tuple would leave
    `_PROTOCOL_FIELDS` decorative -- the table test would pass while the real
    comparison went unwatched. So append a field to the PUBLISHED table and
    require the function to refuse a disagreement on it."""
    monkeypatch.setattr(
        script, "_PROTOCOL_FIELDS",
        script._PROTOCOL_FIELDS + (("appended", lambda r: r["appended"]),),
    )
    records = _protocol_records()
    for record in records.values():
        record["appended"] = 0
    script.require_one_protocol(records)
    records[("pixel_ae", 1)]["appended"] = 1
    with pytest.raises(SystemExit, match="disagree on appended"):
        script.require_one_protocol(records)


@pytest.mark.parametrize("field", sorted(_PROTOCOL_FIELD_MUTATIONS))
def test_require_one_protocol_refuses_a_disagreement_on_every_compared_field(field):
    records = _protocol_records()
    _PROTOCOL_FIELD_MUTATIONS[field](records[("pixel_ae", 1)])
    with pytest.raises(SystemExit) as raised:
        script.require_one_protocol(records)
    message = str(raised.value)
    assert f"disagree on {field}" in message
    assert "pixel_ae seed 1" in message


def test_require_one_protocol_accepts_nine_records_that_agree():
    """The other side: it must not refuse agreement."""
    script.require_one_protocol(_protocol_records())


def test_require_one_protocol_refuses_an_empty_pool():
    with pytest.raises(SystemExit, match="no width record"):
        script.require_one_protocol({})


# ---------------------------------------------------------------------------
# read: the nine records pooled into Reading F
# ---------------------------------------------------------------------------
#
# THE FIXTURE CARRIES NINE DISTINCT PAYLOADS. A record built from one template
# with only the labels changed would let an implementation pair `frozen_ssl`
# with `pixel_ae`'s seeds, collapse every arm onto seed 0, or read the wrong
# horizon, and every test would still pass -- so every per-cell number below
# (contrast, gain at every pass/target/horizon/rung, both base-control r2s)
# is a function of the cell's `(arm, seed)`. What is DELIBERATELY equal across
# cells is exactly what a real study holds fixed: the protocol, the windows,
# the row counts and the cluster count.

_ARM_ORDER = ("frozen_ssl", "pixel_ae", "random_vit", "fourth_arm")

_QUIET = 0.002
_HALF = 0.04
_CLEAR = 0.08
"""A gain clears when `ci_low = gain - _HALF` exceeds 0. `_QUIET` gains plus every
offset and every bias below stay UNDER `_HALF` (checked by `test_the_quiet_...`),
so a rung clears if and only if a test put it in `clearing` -- in particular the
`up` pass, whose bias would otherwise make it clear on its own."""

_BIAS = {
    "translation": {"two_frame": 0.0, "deterministic": 0.0137,
                    "stochastic": 0.0071, "full": 0.0093},
    "rotation": {"two_frame": 0.0, "deterministic": 0.0041,
                 "stochastic": 0.0023, "full": 0.0032},
}
"""`up - shipped` at `CONTRAST_K`, hand-typed. `two_frame` is `up`'s anchor, so its
bias is exactly zero: the untouched rung."""
_OFF_K = 0.01
_ARM_STEP = 0.001
_SEED_STEP = 0.0006
_BROKEN_SHIFT = 0.0009


def _rank(arm, seed):
    return 10 * _ARM_ORDER.index(arm) + seed


def _bias(target, k, rung, arm, seed):
    if rung == ANCHOR["up"]:
        return 0.0
    return (
        _BIAS[target][rung] + (0.0 if k == CONTRAST_K else _OFF_K)
        + _ARM_STEP * _ARM_ORDER.index(arm) + _SEED_STEP * seed
    )


def _shipped_gain(target, rung, arm, seed, clearing):
    value = _QUIET + 0.0001 * _rank(arm, seed) + 0.0002 * RUNGS.index(rung)
    return value + (_CLEAR if ("shipped", target, rung) in clearing else 0.0)


def _pass_gain(pass_name, target, k, rung, arm, seed, clearing, anchors):
    """One cell's gain for one `(pass, target, k, rung)`.

    An ANCHOR rung is bit-identical to the shipped pass while its anchor holds --
    which is what a real record looks like -- so `clearing` cannot make a `down`
    `deterministic` clear on its own: put it in `("shipped", ...)` and the anchor
    carries it. A broken anchor shifts that rung instead, so a record that says
    `up=BROKEN` also carries gains that show it.
    """
    shipped = _shipped_gain(target, rung, arm, seed, clearing)
    if pass_name == "shipped":
        return shipped
    holds = anchors[0] if pass_name == "down" else anchors[1]
    if rung == ANCHOR[pass_name]:
        return shipped if holds else shipped + _BROKEN_SHIFT
    gain = shipped + (
        0.001 if pass_name == "down" else _bias(target, k, rung, arm, seed)
    )
    return gain + (_CLEAR if (pass_name, target, rung) in clearing else 0.0)


def _gain_dict(gain):
    return {"gain": gain, "ci_low": gain - _HALF, "ci_high": gain + _HALF,
            "joint_r2": 0.05, "embedding_r2": 0.04, "confidence": 0.95,
            "n_scored_windows": 48, "ridge_selected": True, "joint_ridge": 1e5,
            "embedding_ridge": 1e5}


def _record(arm, seed, *, clears=0.0, base_r2=0.65, anchors=(True, True),
            clearing=frozenset(), base_spread=0.0004, r2=None):
    """One cell's record, in the schema `measure_cell` writes.

    `clears` shifts the k = 15 contrast (`+0.05` puts every seed's interval
    above zero, `-0.05` below, `0.0` straddling it). `base_r2` is the POSITION
    r2 the gate reads and `r2` the 4-column mean it does not (default 0.33 plus a
    per-cell offset) -- deliberately different numbers so a read of the wrong one
    is a different verdict, not the same one. `anchors` is `(down, up)`.
    """
    rank = _rank(arm, seed)
    contrast = clears + 0.0003 * rank
    passes = {
        p: {t: {f"k{k}": {r: _gain_dict(_pass_gain(p, t, k, r, arm, seed, clearing, anchors))
                          for r in RUNGS}
                for k in K_REPORTED}
            for t in TARGETS}
        for p in PASSES
    }
    return {
        "arm": arm, "seed": seed, "step": 20000, "device": "cpu",
        "context": 5, "horizon": 45, "h_dim": RUNG_WIDTH["deterministic"],
        "passes": passes,
        "contrast": {f"k{CONTRAST_K}": {
            "contrast": contrast, "ci_low": contrast - 0.01, "ci_high": contrast + 0.01,
            "a_r2": 0.05, "b_r2": 0.04, "confidence": 0.95, "n_scored_windows": 48,
            "a_ridge": 1e5, "b_ridge": 1e5, "ridge_selected": True,
        }},
        "rows": {"k1": 11221, "k4": 10534, "k15": 8015}, "clusters": 24,
        "anchors": {"down": anchors[0], "up": anchors[1]},
        "base_control": {
            "r2": (0.33 + 0.001 * rank) if r2 is None else r2,
            "position_r2": base_r2 + base_spread * rank,
            "per_column_r2": [0.7, 0.7, 0.0, -0.05],
            "ridge": 1e3, "ridge_selected": True, "rows": 11450,
        },
        "projection_seed": PROJECTION_SEED, "rung_width": dict(RUNG_WIDTH),
        # PER GATHERED ROW, as a real record's lists are: 96 rows over 48 windows
        # over 24 episodes, three different numbers a caption could confuse.
        "windows": {"episode": [i // 4 for i in range(96)], "window": [i // 2 for i in range(96)]},
        "self_check": {"ok": True}, "git_sha": "abc", "ks": list(K_REPORTED),
        "torch_version": "2.13.0", "split_seed": 0, "record_git_sha": "ca3e140",
        "episodes": {"fit": [], "select": [], "val": [f"e{i}" for i in range(24)]},
    }


def _records(arms=_ARM_ORDER[:3], seeds=(0, 1, 2), **kw):
    return {(a, s): _record(a, s, **kw) for a in arms for s in seeds}


def _write(directory, records):
    for (arm, seed), record in records.items():
        script.width_record_path(directory, arm, seed).write_text(json.dumps(record))


def _main_read(*argv):
    """`main(["--phase", "read", ...])` with the MEASURE half made unreachable.

    THIS GUARD EXISTS BECAUSE ITS ABSENCE COST A REAL MEASURE. A mutation of
    `main` that let `--phase read` fall into the measure branch did not fail a
    test -- the read tests handed `main` a real `--source` default, so it loaded
    the real checkpoints and ran a real cell on the GPU, ~7 minutes of it,
    writing a record into pytest's tmp dir before it was killed. A regression of
    that shape must be a loud, instant AssertionError, never a measurement."""
    def forbidden(*args, **kwargs):
        raise AssertionError("--phase read reached the measure phase")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(script, "measure_phase", forbidden)
        patch.setattr(script, "ReplayBuffer", forbidden)
        return script.main(list(argv))


def _read(tmp_path, capsys, records, *extra):
    """Write `records`, run the real `main --phase read`, return `(status, stdout)`."""
    _write(tmp_path, records)
    status = _main_read("--phase", "read", "--out", str(tmp_path), *extra)
    return status, capsys.readouterr().out


def _verdict_lines(text):
    return [line.strip() for line in text.splitlines() if "verdict:" in line]


def _set_contrast(record, contrast, half=0.01):
    record["contrast"][KEY].update(
        contrast=contrast, ci_low=contrast - half, ci_high=contrast + half,
    )


def test_the_fixture_carries_nine_distinct_payloads():
    """Guards the FIXTURE, which is what every read test below stands on: nine
    records that differed only in their labels would let a cross-arm pairing, a
    seed collapse or a first-record-stands-for-all pass every test in this
    section. Labels are dropped before comparing, so this is about payloads."""
    records = _records()
    payloads = {
        json.dumps({k: v for k, v in r.items() if k not in ("arm", "seed")}, sort_keys=True)
        for r in records.values()
    }
    assert len(payloads) == 9
    for pick in (
        lambda r: r["contrast"][KEY]["contrast"],
        lambda r: r["base_control"]["position_r2"],
        lambda r: r["base_control"]["r2"],
        lambda r: r["passes"]["down"]["translation"]["k4"]["full"]["gain"],
        lambda r: r["passes"]["up"]["rotation"][KEY]["stochastic"]["gain"],
    ):
        values = [pick(r) for r in records.values()]
        assert len(set(values)) == 9, values


def test_the_quiet_gains_never_clear_so_only_a_test_can_make_a_rung_clear():
    """Every `up` bias, arm step and seed step stays under the interval's
    half-width, at every horizon: a rung clears iff a test asked for it."""
    for record in _records(arms=_ARM_ORDER, seeds=(0, 1, 2, 3)).values():
        for pass_name in PASSES:
            for target in TARGETS:
                for key in (f"k{k}" for k in K_REPORTED):
                    for rung in RUNGS:
                        assert record["passes"][pass_name][target][key][rung]["ci_low"] < 0.0


# ---- Reading F, end to end ------------------------------------------------


@pytest.mark.parametrize("clears,expected", [
    (+0.05, "PAST FRAME AHEAD"),
    (-0.05, "RECURRENT AHEAD"),
    (0.0, "INDISTINGUISHABLE"),
])
def test_read_reaches_every_decided_status(tmp_path, capsys, clears, expected):
    status, printed = _read(tmp_path, capsys, _records(clears=clears))
    assert f"verdict: {expected}" in printed
    assert _verdict_lines(printed)[-1].startswith(f"verdict: {expected} -- decided by:"), (
        "Reading F's verdict is the LAST line: evidence before the conclusion"
    )
    assert status == script.EXIT_OK
    assert (tmp_path / "width.txt").read_bytes() == printed.encode()


def test_read_exits_41_when_the_position_control_fails(tmp_path, capsys):
    """The 4-column `r2` sits at 0.33 in every cell here -- above the floor -- so a
    gate that read it would clear and exit 0. Only the POSITION r2 is below."""
    status, printed = _read(tmp_path, capsys, _records(base_r2=BASE_R2_FLOOR - 0.05))
    assert status == 41 == script.EXIT_BASE_UNRESOLVED
    assert _verdict_lines(printed)[-1].startswith("verdict: UNRESOLVED BASE")


def test_the_gate_reads_position_r2_and_ignores_the_four_column_mean(tmp_path, capsys):
    """The converse: the 4-column `r2` is BELOW the floor everywhere and the
    position r2 well above it. A gate on `r2` would exit 41 here."""
    status, printed = _read(
        tmp_path, capsys, _records(clears=+0.05, r2=BASE_R2_FLOOR - 0.05),
    )
    assert status == script.EXIT_OK
    assert _verdict_lines(printed)[-1].startswith("verdict: PAST FRAME AHEAD")


def test_a_position_r2_exactly_on_the_floor_does_not_clear_it(tmp_path, capsys):
    """Strict `>`, as `retention`'s own read applies it. `base_spread=0.0` puts
    EVERY cell exactly on the line; a `>=` would clear all nine and exit 0."""
    status, _ = _read(
        tmp_path, capsys, _records(base_r2=BASE_R2_FLOOR, base_spread=0.0),
    )
    assert status == script.EXIT_BASE_UNRESOLVED


def _tally_pattern_records():
    """Per-seed position r2 chosen so the ARM MEAN and the SEED TALLY disagree:
    frozen_ssl (0.5, 0.05, 0.05) mean 0.20 tally 1; pixel_ae (0.05, 0.5, 0.5)
    mean 0.35 tally 2; random_vit (0.05, 0.05, 0.11) mean 0.07 tally 1. Every
    arm differs from the others in mean AND in which seed carries the clearing,
    so a base-control loop that read seed 0 (or seed 2) three times, or one arm's
    seeds under another's name, lands on a different tally for at least one arm.
    """
    records = _records(clears=+0.05)
    pattern = {
        "frozen_ssl": (0.5, 0.05, 0.05),
        "pixel_ae": (0.05, 0.5, 0.5),
        "random_vit": (0.05, 0.05, 0.11),
    }
    for arm, levels in pattern.items():
        for seed, level in enumerate(levels):
            records[(arm, seed)]["base_control"]["position_r2"] = level
    return records


def test_the_base_tally_is_per_seed_within_each_arm_not_a_mean_and_not_one_seed_repeated():
    base = script.contrast_inputs(_tally_pattern_records()).base
    assert {arm: control.seeds_clear for arm, control in base.items()} == {
        "frozen_ssl": 1, "pixel_ae": 2, "random_vit": 1,
    }
    assert {arm: control.seeds_total for arm, control in base.items()} == {
        "frozen_ssl": 3, "pixel_ae": 3, "random_vit": 3,
    }
    assert base["frozen_ssl"].r2 == pytest.approx(0.20)
    assert base["pixel_ae"].r2 == pytest.approx(0.35)
    assert base["random_vit"].r2 == pytest.approx(0.07)


def test_the_base_gate_counts_seeds_not_an_arm_mean(tmp_path, capsys):
    """Only pixel_ae holds 2 of 3 seeds, so 1 of 3 arms holds: UNRESOLVED_BASE. A
    gate on the ARM MEAN would see frozen_ssl (0.20) and pixel_ae (0.35) clear and
    read the comparison."""
    status, printed = _read(tmp_path, capsys, _tally_pattern_records())
    assert status == script.EXIT_BASE_UNRESOLVED
    assert _verdict_lines(printed)[-1].startswith("verdict: UNRESOLVED BASE")


def test_one_failing_arm_does_not_stop_the_reading_but_is_printed(tmp_path, capsys):
    """Two of three arms hold, which is `ARMS_REQUIRED`: the reading goes on, and
    the failing arm's tally is on the page rather than silently dropped."""
    records = _records(clears=+0.05)
    for seed, level in enumerate((0.04, 0.05, 0.06)):
        records[("random_vit", seed)]["base_control"]["position_r2"] = level
    status, printed = _read(tmp_path, capsys, records)
    assert status == script.EXIT_OK
    assert _verdict_lines(printed)[-1].startswith("verdict: PAST FRAME AHEAD")
    assert "random_vit r2=+0.050 0/3" in printed
    assert "frozen_ssl r2=+0.650 3/3" in printed


def test_the_base_line_prints_the_position_r2_the_tally_was_read_from(tmp_path, capsys):
    """THE M3h TRAP: a tally printed beside a number it was not read from. The
    gate is `position_r2`, so the `r2=` figure on the base-control line must be
    the arm's mean POSITION r2 -- 0.650 / 0.654 / 0.658 here -- and the
    4-column mean (0.331 / 0.341 / 0.351) must appear NOWHERE. Reading E's own
    base line is the same object, so each figure is printed twice, never with a
    different number the second time."""
    _, printed = _read(tmp_path, capsys, _records(clears=+0.05))
    for arm, position, four_column in (
        ("frozen_ssl", "+0.650", "+0.331"),
        ("pixel_ae", "+0.654", "+0.341"),
        ("random_vit", "+0.658", "+0.351"),
    ):
        assert printed.count(f"{arm} r2={position} 3/3") == 2, arm
        assert four_column not in printed, (
            f"the 4-column r2 {four_column} is printed; it is not what the gate reads"
        )


def test_read_exits_42_when_the_down_anchor_is_broken(tmp_path, capsys):
    status, printed = _read(tmp_path, capsys, _records(anchors=(False, True)))
    assert status == 42 == script.EXIT_ANCHOR_BROKEN
    assert "down=BROKEN" in printed
    assert _verdict_lines(printed)[-1].startswith("verdict: UNRESOLVED ANCHOR")


def test_an_unresolved_base_outranks_a_broken_down_anchor(tmp_path, capsys):
    status, printed = _read(
        tmp_path, capsys,
        _records(base_r2=BASE_R2_FLOOR - 0.05, anchors=(False, True)),
    )
    assert status == script.EXIT_BASE_UNRESOLVED
    assert _verdict_lines(printed)[-1].startswith("verdict: UNRESOLVED BASE")


@pytest.mark.parametrize("cell", sorted(_records()))
@pytest.mark.parametrize("which", ["down", "up"])
def test_one_broken_anchor_in_any_single_cell_breaks_the_pooled_anchor(cell, which):
    """`all` over the nine cells, not the first, the last, or `any`. The broken cell
    is each of the nine in turn, so a read that let one record stand for the set
    would miss it for eight of them."""
    records = _records()
    records[cell]["anchors"][which] = False
    anchors = script.contrast_inputs(records).anchors
    assert anchors == {"down": which != "down", "up": which != "up"}


def test_a_broken_up_anchor_is_reported_but_does_not_block_the_reading(tmp_path, capsys):
    status, printed = _read(tmp_path, capsys, _records(clears=+0.05, anchors=(True, False)))
    assert status == script.EXIT_OK
    assert "verdict: PAST FRAME AHEAD" in printed
    assert "up=BROKEN" in printed
    assert "width bias" in printed and "unreadable" in printed


# ---- the loader and the plan ------------------------------------------------


def test_read_names_a_missing_record_and_exits_11(tmp_path, capsys):
    records = _records()
    del records[("pixel_ae", 1)]
    status, printed = _read(tmp_path, capsys, records)
    assert status == script.EXIT_NO_CHECKPOINTS == 11
    assert "NO CELL:" in printed and "pixel_ae seed 1" in printed
    assert not (tmp_path / "width.txt").exists()


def test_a_record_that_disagrees_with_its_own_filename_is_refused_by_name(tmp_path):
    """A swapped pair of files would pool one cell under the other's name with
    every count still right."""
    records = _records()
    _write(tmp_path, records)
    swapped = records[("pixel_ae", 1)]
    script.width_record_path(tmp_path, "frozen_ssl", 2).write_text(json.dumps(swapped))
    with pytest.raises(SystemExit, match=r"width_frozen_ssl_seed2\.json.*frozen_ssl seed 2.*pixel_ae"):
        script.load_width(tmp_path, ["frozen_ssl", "pixel_ae", "random_vit"], [0, 1, 2])


@pytest.mark.parametrize("extra", [
    ["--arms", "frozen_ssl", "frozen_ssl"],
    ["--arms", "frozen_ssl"],
    ["--seeds", "0", "0", "0"],
    ["--seeds", "1"],
])
def test_read_refuses_a_plan_too_narrow_before_touching_the_records(tmp_path, extra):
    """`--arms frozen_ssl frozen_ssl` is TWO names and ONE arm, and `--seeds 0 0 0`
    is three names and one seed: counting the argument lists would pass both and
    the reading would raise deep inside `contrast_arm`. No record exists here, so
    a refusal is provably made from the plan alone."""
    with pytest.raises(SystemExit, match="cannot be read"):
        _main_read("--phase", "read", "--out", str(tmp_path), *extra)


def test_read_refuses_records_that_are_not_one_measurement_before_reading_anything(tmp_path, capsys):
    """Nine records pooled into one reading must share a protocol. The comparison
    itself is pinned field by field in `require_one_protocol`'s own tests; this is
    the test that `read` CALLS it -- a `git_sha` that differs in one cell (two code
    versions in one finding) must stop the read with the cell named, print
    nothing, and write no artefact."""
    records = _records(clears=+0.05)
    records[("random_vit", 2)]["git_sha"] = "a-different-commit"
    _write(tmp_path, records)
    with pytest.raises(SystemExit, match=r"random_vit seed 2.*disagree on git_sha"):
        _main_read("--phase", "read", "--out", str(tmp_path))
    assert not (tmp_path / "width.txt").exists()
    assert capsys.readouterr().out == ""


def test_read_takes_a_plan_of_duplicated_names_as_the_distinct_plan(tmp_path, capsys):
    """Duplicates collapse rather than refuse when the distinct plan is readable:
    the records dict is keyed by cell, so a name given twice is one cell."""
    status, printed = _read(
        tmp_path, capsys, _records(arms=_ARM_ORDER[:2], clears=+0.05),
        "--arms", "pixel_ae", "frozen_ssl", "pixel_ae", "--seeds", "0", "1", "2", "1",
    )
    assert status == script.EXIT_OK
    assert "verdict: PAST FRAME AHEAD" in printed


# ---- what the builder hands the reading ----------------------------------------


def test_contrast_inputs_pools_each_arm_from_its_own_seeds():
    """Hand-typed. Per arm the contrast is the seed MEAN, `ci_low` the LEAST lower
    bound and `ci_high` the GREATEST upper bound, and the tallies count seeds
    whose OWN interval excludes zero -- three arms with three different
    patterns, so pairing one arm with another's seeds changes a number."""
    records = _records()
    contrasts = {
        "frozen_ssl": (0.030, 0.040, 0.050),      # up 3, down 0
        "pixel_ae": (0.011, 0.020, -0.002),       # up 2, down 0
        "random_vit": (-0.030, -0.040, 0.001),    # up 0, down 2
    }
    for arm, values in contrasts.items():
        for seed, value in enumerate(values):
            _set_contrast(records[(arm, seed)], value)
    arms = script.contrast_inputs(records).arms
    expected = {
        #             contrast   ci_low  ci_high  up dn
        "frozen_ssl": (0.040, 0.020, 0.060, 3, 0),
        "pixel_ae": (0.029 / 3, -0.012, 0.030, 2, 0),
        "random_vit": (-0.023, -0.050, 0.011, 0, 2),
    }
    assert set(arms) == set(expected)
    for name, (contrast, low, high, up, down) in expected.items():
        arm = arms[name]
        assert arm.contrast == pytest.approx(contrast), name
        assert arm.ci_low == pytest.approx(low), name
        assert arm.ci_high == pytest.approx(high), name
        assert (arm.seeds_up, arm.seeds_down, arm.seeds_total) == (up, down, 3), name
    assert reading_contrast(script.contrast_inputs(records)).status == "PAST_FRAME_AHEAD"


def test_contrast_inputs_reads_the_contrast_horizon_and_its_row_count():
    """`rows` is the row count AT `CONTRAST_K` -- 8015 -- not `k4`'s 10534 (the
    other reading's horizon) or `k1`'s 11221, which is the first key. And
    `clusters` is the distinct-episode count, 24, not the 48 windows."""
    inputs = script.contrast_inputs(_records())
    assert inputs.rows == 8015
    assert inputs.clusters == 24


def test_the_script_never_reaches_for_the_other_readings_horizon():
    """`retention.DECISION_K` is 4 and means a different horizon. Nothing here
    imports it, so no line can read `k4` where `CONTRAST_K` was meant."""
    assert not hasattr(script, "DECISION_K")
    assert script.CONTRAST_K == 15


def test_the_builder_satisfies_every_precondition_the_reading_enforces():
    """`reading_contrast` raises on eight kinds of malformed input; the builder's
    output for a well-formed study must clear all of them."""
    inputs = script.contrast_inputs(_records())
    assert len(inputs.arms) >= ARMS_REQUIRED
    assert set(inputs.anchors) == set(ANCHOR)
    assert set(inputs.base) == set(inputs.arms)
    assert len({arm.seeds_total for arm in inputs.arms.values()}) == 1
    assert inputs.arms["frozen_ssl"].seeds_total >= SEEDS_REQUIRED
    reading_contrast(inputs)


@pytest.mark.parametrize("field,value", [("clusters", 23), ("rows", {"k1": 11221, "k4": 10534, "k15": 8000})])
def test_records_that_disagree_on_clusters_or_rows_are_refused_not_sampled(field, value):
    """`ContrastInputs.clusters` and `.rows` are ONE number in a caption that
    speaks for nine cells. Taking the first record's would print `8015 rows` over
    a set whose members do not agree -- one member standing for the set. The
    victim sorts LAST, so a builder that read the first record sees nothing."""
    records = _records()
    records[("random_vit", 2)][field] = value
    with pytest.raises(SystemExit) as raised:
        script.contrast_inputs(records)
    message = str(raised.value)
    assert field in message and "random_vit seed 2" in message


@pytest.mark.parametrize("mutate,names", [
    (lambda r: r["contrast"].pop(KEY), ("pixel_ae seed 1", f"contrast.{KEY}")),
    (lambda r: r["anchors"].pop("down"), ("pixel_ae seed 1", "anchors.down")),
    (lambda r: r["base_control"].pop("position_r2"), ("pixel_ae seed 1", "base_control.position_r2")),
    (lambda r: r.pop("clusters"), ("pixel_ae seed 1", "clusters")),
])
def test_a_record_missing_a_field_the_reading_needs_is_a_named_refusal(mutate, names):
    """SystemExit naming the cell and the path -- never a bare KeyError, which
    would surface as a traceback after the reader had paid for the measure."""
    records = _records()
    mutate(records[("pixel_ae", 1)])
    with pytest.raises(SystemExit) as raised:
        script.contrast_inputs(records)
    for name in names:
        assert name in str(raised.value)


def test_records_that_all_lack_the_contrast_horizons_rows_are_a_named_refusal():
    """Every record without `rows.k15` -- as if `k15` had not been measured -- is a
    refusal that names the path, distinct from the disagreement refusal above
    (one record lacking it would DISAGREE with the rest and be refused as that)."""
    records = _records()
    for record in records.values():
        del record["rows"][KEY]
    with pytest.raises(SystemExit, match=rf"rows\.{KEY}"):
        script.contrast_inputs(records)


def test_a_non_boolean_anchor_is_refused_rather_than_read_as_truthy():
    """`"false"` is truthy: passed through `all`, a hand-edited or damaged record
    would read as an anchor that HELD."""
    records = _records()
    records[("random_vit", 0)]["anchors"]["down"] = "false"
    with pytest.raises(SystemExit, match=r"random_vit seed 0.*anchors\.down"):
        script.contrast_inputs(records)


@pytest.mark.parametrize("key", ["contrast", "ci_low", "ci_high"])
def test_a_non_finite_contrast_is_refused_by_arm_and_not_read_as_indistinguishable(key):
    """NaN compares False both ways, so it would land on INDISTINGUISHABLE -- a
    pre-registered finding asserted from a broken number."""
    records = _records()
    records[("pixel_ae", 2)]["contrast"][KEY][key] = float("nan")
    with pytest.raises(SystemExit, match=r"pixel_ae.*non-finite"):
        script.contrast_inputs(records)


def test_a_non_finite_position_r2_is_refused_rather_than_counted_as_a_failed_control():
    """`nan > BASE_R2_FLOOR` is False, so a NaN would silently count as a seed that
    did NOT clear -- UNRESOLVED_BASE, from a measurement that never happened."""
    records = _records()
    records[("frozen_ssl", 1)]["base_control"]["position_r2"] = float("nan")
    with pytest.raises(SystemExit, match=r"frozen_ssl seed 1.*position_r2"):
        script.contrast_inputs(records)


@pytest.mark.parametrize("build,needle", [
    (lambda: {c: r for c, r in _records().items() if c[0] == "pixel_ae"}, r"cover 1 arm"),
    (lambda: {c: r for c, r in _records().items() if c[1] == 0}, r"SEEDS_REQUIRED=2 seeds"),
    (lambda: {c: r for c, r in _records().items() if c != ("random_vit", 2)},
     r"do not share one seed count"),
])
def test_records_too_narrow_or_uneven_for_the_reading_are_a_named_refusal(build, needle):
    """One arm; one seed per arm; and an arm one seed short of the others -- the
    three shapes `reading_contrast` would raise a bare ValueError on. Each is
    matched on the message of the check that owns it (the middle one is
    `contrast_arm`'s, named by arm), so one refusal cannot stand in for another."""
    with pytest.raises(SystemExit, match=needle):
        script.contrast_inputs(build())


# ---- refusals the READING owns, and where they sit relative to the gates -----


def _both_ways_records(**kw):
    """frozen_ssl clears UP in seeds 0-1 and DOWN in seeds 2-3: with four seeds
    one arm clears both ways. The other two arms straddle zero."""
    records = _records(seeds=(0, 1, 2, 3), **kw)
    for seed, value in enumerate((+0.05, +0.05, -0.05, -0.05)):
        _set_contrast(records[("frozen_ssl", seed)], value)
    return records


def test_an_arm_clearing_both_ways_is_a_named_refusal_and_writes_nothing(tmp_path, capsys):
    """Past both gates a reading IS taken, and `reading_contrast` refuses this arm
    rather than let if-order decide it. From `read` that is a SystemExit naming
    the arm, not a ValueError traceback -- and no `width.txt`, since a refused
    read has no artefact to leave."""
    _write(tmp_path, _both_ways_records())
    with pytest.raises(SystemExit, match=r"BOTH ways.*frozen_ssl"):
        _main_read("--phase", "read", "--out", str(tmp_path), "--seeds", "0", "1", "2", "3")
    assert not (tmp_path / "width.txt").exists()
    assert capsys.readouterr().out == "", "a refused read prints no partial report"


def test_the_both_ways_refusal_does_not_pre_empt_the_gates(tmp_path, capsys):
    """The same incoherent arm inside a failed base or a broken down anchor is a
    legitimate UNRESOLVED_* record, read normally -- refusing it in the builder
    would turn each into a crash."""
    for kw, status in (
        ({"base_r2": BASE_R2_FLOOR - 0.05}, script.EXIT_BASE_UNRESOLVED),
        ({"anchors": (False, True)}, script.EXIT_ANCHOR_BROKEN),
    ):
        directory = tmp_path / str(status)
        directory.mkdir()
        _write(directory, _both_ways_records(**kw))
        got = _main_read(
            "--phase", "read", "--out", str(directory), "--seeds", "0", "1", "2", "3",
        )
        assert got == status
        assert "both?" in capsys.readouterr().out, "the table marks the arm, it does not vote it"


def test_both_directions_clearing_at_the_bar_is_a_named_refusal(tmp_path, capsys):
    """Four arms: two clear up in every seed, two clear down in every seed, every
    arm perfectly one-sided. Each direction is at `ARMS_REQUIRED`, so if-order
    would decide it. `read_phase` is called directly, since argparse's `choices`
    admits only the study's three arms."""
    arms = _ARM_ORDER
    records = _records(arms=arms)
    for (arm, seed), record in records.items():
        _set_contrast(record, +0.05 if arm in arms[:2] else -0.05)
    _write(tmp_path, records)
    args = types.SimpleNamespace(out=tmp_path, arms=list(arms), seeds=[0, 1, 2])
    with pytest.raises(SystemExit, match="ambiguous"):
        script.read_phase(args)
    assert not (tmp_path / "width.txt").exists()


# ---- the companions ----------------------------------------------------------------


def test_the_corrected_reading_e_uses_stochastic_only(monkeypatch, tmp_path, capsys):
    """M3j's own results recorded that `full` is h + z and cannot attribute a
    clearance to z. The companion must pass the corrected tuple rather than rely
    on the module constant, which stays as M3j's records were taken under."""
    seen = {}
    real = script.reading_retention

    def recording(inputs, **kwargs):
        seen.update(kwargs)
        return real(inputs, **kwargs)

    monkeypatch.setattr(script, "reading_retention", recording)
    _read(tmp_path, capsys, _records())
    assert seen.get("z_bearing") == ("stochastic",)


def test_the_companions_heading_names_the_pass_and_the_tuple_it_actually_ran(monkeypatch):
    """A caption that says `('stochastic',)` over a call that passed something else
    is the shipped-three-times defect. The heading is built from the same
    constants the reading is taken with, so changing the tuple changes both -- and
    the tuple the reading was CALLED with is watched, not assumed."""
    records = _records()
    heading = script.corrected_reading_e(records).splitlines()[0]
    assert f"`{script.RETENTION_PASS}` pass" in heading
    assert f"{DOWN_WIDTH} columns" in heading
    assert str(("stochastic",)) in heading
    seen = {}
    real = script.reading_retention
    monkeypatch.setattr(
        script, "reading_retention",
        lambda inputs, **kwargs: seen.update(kwargs) or real(inputs, **kwargs),
    )
    monkeypatch.setattr(script, "CORRECTED_Z_BEARING", ("full",))
    heading = script.corrected_reading_e(records).splitlines()[0]
    assert seen["z_bearing"] == ("full",)
    assert str(("full",)) in heading and "stochastic" not in heading


def test_the_corrected_tuple_changes_the_status_where_full_alone_carries_z(tmp_path, capsys):
    """The spy above shows the keyword is PASSED; this shows it is not decorative.
    `full` (h + z) and `deterministic` clear translation on the down pass and
    `stochastic` does not. Under M3j's own rule that is MOTION_RETAINED, via
    `full`; under the corrected one, BOTTLENECK_LOSS. A companion that omitted
    the keyword, or passed `Z_BEARING_RUNGS`, prints the first."""
    clearing = {("shipped", "translation", "deterministic"), ("down", "translation", "full")}
    records = _records(clearing=clearing)
    old_rule = _reading_retention(script.retention_inputs(records))
    assert old_rule.status == "MOTION_RETAINED" and old_rule.surviving == "full"
    _, printed = _read(tmp_path, capsys, records)
    companion = _companion(printed)
    assert "verdict: BOTTLENECK LOSS" in companion
    assert "MOTION RETAINED" not in companion


def _companion(text):
    """The corrected-Reading-E block: from its heading to Reading F's table."""
    start = text.index("Companion, decides nothing: Reading E")
    return text[start:text.index("--- Reading F")]


def test_the_corrected_reading_e_reads_the_down_pass_not_shipped_or_up(tmp_path, capsys):
    """Only `stochastic` on the DOWN pass clears. The shipped pass reads
    UNRESOLVED_MOTION on the same records, and the fixture is checked to
    discriminate before the companion is: a builder over the wrong pass would
    print a different verdict."""
    records = _records(clearing={("down", "translation", "stochastic")})
    statuses = {
        p: _reading_retention(
            script.retention_inputs(records, pass_name=p), z_bearing=("stochastic",),
        ).status
        for p in PASSES
    }
    assert statuses["down"] == "MOTION_RETAINED"
    assert statuses["shipped"] != "MOTION_RETAINED" and statuses["up"] != "MOTION_RETAINED"
    _, printed = _read(tmp_path, capsys, records)
    assert "verdict: MOTION RETAINED" in _companion(printed)


def test_retention_inputs_pairs_every_arm_with_its_own_seeds_on_the_down_pass():
    """Every `(target, k, rung, arm)` cell of the ladder against the arm's own
    three raw records, recomputed from the fixture: an arm-for-arm swap, a seed
    collapse or a read of another horizon lands on a different number because
    every cell's gain is distinct."""
    records = _records()
    inputs = script.retention_inputs(records)
    assert inputs.clusters == 24
    assert inputs.rows == {1: 11221, 4: 10534, 15: 8015}
    for target in TARGETS:
        for k in K_REPORTED:
            for rung in RUNGS:
                arms = inputs.ladder[target][k][rung]
                assert set(arms) == {"frozen_ssl", "pixel_ae", "random_vit"}
                assert len({a.gain for a in arms.values()}) == 3
                for arm, summary in arms.items():
                    raw = [
                        records[(arm, s)]["passes"]["down"][target][f"k{k}"][rung]
                        for s in (0, 1, 2)
                    ]
                    assert summary.gain == pytest.approx(np.mean([r["gain"] for r in raw]))
                    assert summary.ci_low == pytest.approx(min(r["ci_low"] for r in raw))
                    assert summary.ci_high == pytest.approx(max(r["ci_high"] for r in raw))
                    assert summary.seeds_total == 3


def test_the_corrected_reading_e_is_gated_on_the_same_base_control_as_reading_f():
    """Both corrections are adopted (spec 2.5), so Reading E's base is the
    position-gated one Reading F reads: the same `BaseControl` objects, not a
    second gate on the 4-column mean beside the first."""
    records = _tally_pattern_records()
    assert script.retention_inputs(records).base == script.contrast_inputs(records).base


def test_the_companion_says_it_is_unreadable_when_the_plan_has_too_few_arms(tmp_path, capsys):
    """Two arms are a legitimate Reading F (`ARMS_REQUIRED`) and not a Reading E,
    which is defined over `RETENTION_FAMILY`. The companion decides nothing, so it
    says so on the page instead of vetoing the verdict or raising."""
    status, printed = _read(
        tmp_path, capsys, _records(arms=_ARM_ORDER[:2], clears=+0.05),
        "--arms", "frozen_ssl", "pixel_ae",
    )
    assert status == script.EXIT_OK
    assert "verdict: PAST FRAME AHEAD" in printed
    companion = _companion(printed)
    assert "unreadable" in companion and f"exactly {RETENTION_FAMILY} arms" in companion
    assert "verdict:" not in companion and "--- Reading E" not in companion


def test_the_companion_says_it_is_unreadable_when_a_horizon_was_not_measured(tmp_path, capsys):
    """Reading E is a disjunction over `K_REPORTED`. Records that lack k = 1 can
    still carry Reading F, which is taken at k = 15 alone."""
    records = _records(clears=+0.05)
    for record in records.values():
        for target in TARGETS:
            for pass_name in PASSES:
                del record["passes"][pass_name][target]["k1"]
        del record["rows"]["k1"]
        record["ks"] = [4, CONTRAST_K]
    status, printed = _read(tmp_path, capsys, records)
    assert status == script.EXIT_OK
    assert "verdict: PAST FRAME AHEAD" in printed
    companion = _companion(printed)
    assert "unreadable" in companion and "k = 1" in companion
    assert "verdict:" not in companion


def test_a_non_finite_gain_on_the_down_pass_is_a_named_refusal_not_a_traceback():
    records = _records()
    records[("pixel_ae", 1)]["passes"]["down"]["rotation"]["k4"]["full"]["gain"] = float("nan")
    with pytest.raises(SystemExit, match=r"pixel_ae.*down pass.*rotation.*k = 4.*full"):
        script.corrected_reading_e(records)


# ---- the width-bias table ------------------------------------------------------------

_BIAS_ROWS = [
    # target, rung, frozen_ssl, pixel_ae, random_vit, all -- `up - shipped`, by hand:
    # bias = D + 0.001 * arm_index + 0.0006 * seed, so an arm's mean over seeds
    # 0..2 is D + 0.001 * arm_index + 0.0006, and the nine-cell mean D + 0.0016.
    ("translation", "two_frame", "+0.0000", "+0.0000", "+0.0000", "+0.0000"),
    ("translation", "deterministic", "+0.0143", "+0.0153", "+0.0163", "+0.0153"),
    ("translation", "stochastic", "+0.0077", "+0.0087", "+0.0097", "+0.0087"),
    ("translation", "full", "+0.0099", "+0.0109", "+0.0119", "+0.0109"),
    ("rotation", "two_frame", "+0.0000", "+0.0000", "+0.0000", "+0.0000"),
    ("rotation", "deterministic", "+0.0047", "+0.0057", "+0.0067", "+0.0057"),
    ("rotation", "stochastic", "+0.0029", "+0.0039", "+0.0049", "+0.0039"),
    ("rotation", "full", "+0.0038", "+0.0048", "+0.0058", "+0.0048"),
]


def _bias_rows(table):
    """`{(target, rung): [values...]}` parsed off the printed rows."""
    rows = {}
    for line in table.splitlines():
        cells = line.split()
        if len(cells) == 7 and cells[0] in TARGETS:
            rows[(cells[0], cells[1])] = cells[3:]
    return rows


def test_the_width_bias_table_is_up_minus_shipped_per_rung_target_and_arm():
    """Every printed number, hand-typed. `up - shipped` (so a positive number is
    what count alone bought, and swapping the two flips every sign), at
    `CONTRAST_K` (the other horizons carry an extra +0.01 in this fixture), on
    the `up` and `shipped` passes (the `down` pass carries +0.001 over shipped).
    Three arms with three different columns, and seed steps that make seed 0
    alone or seed 2 alone print differently from the mean."""
    table = script.width_bias_table(_records())
    rows = _bias_rows(table)
    assert set(rows) == {(t, r) for t in TARGETS for r in RUNGS}
    for target, rung, *values in _BIAS_ROWS:
        assert rows[(target, rung)] == values, (target, rung)


def test_the_width_bias_columns_are_the_arms_in_order_then_all():
    table = script.width_bias_table(_records())
    header = next(line for line in table.splitlines() if line.split()[:2] == ["target", "rung"])
    assert header.split() == [
        "target", "rung", "width", "frozen_ssl", "pixel_ae", "random_vit", "all",
    ]


def test_the_width_bias_table_prints_the_native_width_each_rung_was_lifted_from():
    table = script.width_bias_table(_records())
    widths = {
        cells[1]: int(cells[2])
        for cells in (line.split() for line in table.splitlines())
        if len(cells) == 7 and cells[0] == "translation"
    }
    assert widths == {rung: RUNG_WIDTH[rung] for rung in RUNGS}


def test_the_width_bias_caption_names_what_it_computes_from_the_constants_and_the_data():
    """The caption's `2048`, `15` and cell count come from `UP_WIDTH`,
    `CONTRAST_K` and the records -- a 12-cell run says 12."""
    nine = script.width_bias_table(_records())
    twelve = script.width_bias_table(_records(seeds=(0, 1, 2, 3)))
    for text, cells in ((nine, 9), (twelve, 12)):
        caption = text.splitlines()[0]
        assert caption.startswith("--- The width bias")
        assert f"lifted to {UP_WIDTH} columns" in caption
        assert f"k = {CONTRAST_K}" in caption
        assert f"over all {cells} cells" in caption
        assert ANCHOR["up"] in caption


def test_the_width_bias_table_follows_the_contrast_horizon_and_the_lift_width(monkeypatch):
    """`CONTRAST_K` and `UP_WIDTH` are read at call time, and the caption and the
    numbers move together: at k = 4 this fixture's bias carries an extra +0.01, so
    a caption that said `k = 4` over k = 15's numbers (or the reverse) is caught
    by the deterministic row alone."""
    monkeypatch.setattr(script, "CONTRAST_K", 4)
    monkeypatch.setattr(script, "UP_WIDTH", 4096)
    table = script.width_bias_table(_records())
    assert "at k = 4," in table.splitlines()[0]
    assert "lifted to 4096 columns" in table.splitlines()[0]
    # D + 0.01 + arm step * arm index + seed step * mean(seeds) = 0.0243 for frozen_ssl.
    assert _bias_rows(table)[("translation", "deterministic")][0] == "+0.0243"


def test_a_broken_up_anchor_replaces_every_number_in_the_table_with_unreadable():
    """`two_frame` is `up`'s known-answer rung; broken in ONE cell, the lift is
    not what it claims and calibrates nothing. Nothing numeric may print -- the
    caption included."""
    records = _records()
    records[("pixel_ae", 1)]["anchors"]["up"] = False
    table = script.width_bias_table(records)
    assert "width bias" in table and "unreadable" in table
    assert re.search(r"[+-]\d\.\d{4}", table) is None, table
    assert "1 of 9 cells" in table


def test_a_broken_down_anchor_does_not_suppress_the_width_bias_table():
    """Only the `up` anchor gates the companion; `down` gates Reading F."""
    records = _records(anchors=(False, True))
    rows = _bias_rows(script.width_bias_table(records))
    assert len(rows) == 8 and rows[("translation", "deterministic")][3] == "+0.0153"


def test_a_non_finite_gain_in_the_width_bias_table_is_a_named_refusal():
    records = _records()
    records[("random_vit", 0)]["passes"]["up"]["translation"][KEY]["stochastic"]["gain"] = float("inf")
    with pytest.raises(SystemExit, match=r"random_vit seed 0.*up.*translation.*stochastic"):
        script.width_bias_table(records)


# ---- width.txt -----------------------------------------------------------------------


def test_width_txt_is_the_text_the_run_composed_and_stdout_is_the_same_bytes(tmp_path, capsys):
    """Byte identity, on the REAL text. Exact equality of bytes, plus the two
    endings a `rstrip()` or an extra `print` newline would change: exactly one
    trailing newline, nothing after it."""
    status, printed = _read(tmp_path, capsys, _records(clears=+0.05))
    written = (tmp_path / "width.txt").read_bytes()
    assert written == printed.encode()
    assert written.endswith(b"\n") and not written.endswith(b"\n\n")
    assert printed.isascii(), "byte identity should not depend on the platform's encoding"
    assert len(printed.splitlines()) > 40


def test_the_file_and_stdout_are_each_exactly_what_width_text_returned(monkeypatch, tmp_path, capsys):
    """The real text ends in one clean newline, so on its own it cannot tell a
    `write_text(text)` from `write_text(text.rstrip() + "\\n")`, nor `print(text,
    end="")` from `print(text.rstrip())`. This text ends in trailing spaces and
    THREE newlines: any strip, any added newline, and any normalisation on either
    channel changes the bytes, and the two channels must still agree."""
    sentinel = "first line  \nsecond line\t\n\n\n"
    monkeypatch.setattr(script, "width_text", lambda *args, **kwargs: sentinel)
    _, printed = _read(tmp_path, capsys, _records())
    assert (tmp_path / "width.txt").read_bytes() == sentinel.encode()
    assert printed == sentinel


def test_a_second_read_prints_exactly_the_bytes_the_first_wrote(tmp_path, capsys):
    """Exit criterion: `width.txt` is byte-identical to a second `--phase read`."""
    _, first_printed = _read(tmp_path, capsys, _records(clears=-0.05))
    first_file = (tmp_path / "width.txt").read_bytes()
    status = _main_read("--phase", "read", "--out", str(tmp_path))
    second_printed = capsys.readouterr().out
    assert status == script.EXIT_OK
    assert second_printed.encode() == first_file == first_printed.encode()
    assert (tmp_path / "width.txt").read_bytes() == first_file


def test_the_artefact_is_written_before_anything_is_printed(monkeypatch, tmp_path, capsys):
    """`width.txt` gates the log: if the write fails, nothing was printed that a
    reader could mistake for a recorded result."""
    _write(tmp_path, _records())

    def failing(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(script, "write_text", failing)
    with pytest.raises(OSError):
        _main_read("--phase", "read", "--out", str(tmp_path))
    assert capsys.readouterr().out == ""


def test_width_text_orders_the_evidence_before_the_conclusion():
    records = _records(clears=+0.05)
    inputs = script.contrast_inputs(records)
    reading = reading_contrast(inputs)
    text = script.width_text(records, inputs, reading)
    order = [
        text.index("--- self-check per record"),
        text.index("--- The width bias"),
        text.index("Companion, decides nothing: Reading E"),
        text.index("--- The ladder"),
        text.index("--- Reading E:"),
        text.index("--- Reading F"),
        text.index("verdict: PAST FRAME AHEAD"),
    ]
    assert order == sorted(order), order
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_the_self_check_table_has_one_row_per_cell_with_that_cells_own_values():
    """Steps are made distinct per cell here (a real study has one), and one cell's
    self-check fails, so a row paired with another cell's record shows up."""
    records = _records()
    for (arm, seed), record in records.items():
        record["step"] = 20000 + _rank(arm, seed)
    records[("pixel_ae", 1)]["self_check"]["ok"] = False
    inputs = script.contrast_inputs(records)
    text = script.width_text(records, inputs, reading_contrast(inputs))
    rows = {}
    for line in text.splitlines():
        cells = line.split()
        if len(cells) == 7 and cells[0] in _ARM_ORDER and cells[2].isdigit():
            rows[(cells[0], int(cells[1]))] = cells[2:]
    assert set(rows) == set(records)
    for (arm, seed), cells in rows.items():
        ok = "NO" if (arm, seed) == ("pixel_ae", 1) else "yes"
        assert cells == [str(20000 + _rank(arm, seed)), "48", "96", "24", ok], (arm, seed)


def test_the_self_check_table_counts_windows_and_gathered_rows_under_their_own_headers():
    """A real record's `windows.episode` is 11,450 long over 229 windows. Printing
    that length under `windows` (as `latent_retention`'s table does) is a caption
    disagreeing with its column; here `windows` is distinct windows and the row
    count is `gathered`. The fixture's three counts are all different -- 48, 96
    and 24 -- so any two of the columns swapped, or `windows` taken as a length,
    reads a different number."""
    records = _records()
    inputs = script.contrast_inputs(records)
    text = script.width_text(records, inputs, reading_contrast(inputs))
    header = next(line for line in text.splitlines() if line.split()[:2] == ["arm", "seed"])
    assert header.split() == ["arm", "seed", "step", "windows", "gathered", "clusters", "ok"]
    first = next(line for line in text.splitlines() if line.split()[:2] == ["frozen_ssl", "0"])
    assert first.split()[3:6] == ["48", "96", "24"]


# ---- main ------------------------------------------------------------------------------


def _stub_main(monkeypatch, *, measure=script.EXIT_OK, read=script.EXIT_OK):
    calls = []

    def fake_buffer(*args, **kwargs):
        calls.append(("buffer",))
        return types.SimpleNamespace(episode_paths=lambda: ["p"])

    monkeypatch.setattr(script, "ReplayBuffer", fake_buffer)
    monkeypatch.setattr(script, "episode_split", lambda paths, val_fraction, seed: (["t"], ["v"]))
    monkeypatch.setattr(script, "get_device", lambda prefer="mps": "cpu")
    monkeypatch.setattr(
        script, "measure_phase",
        lambda args, cells, device, train, val, ks=None: (
            calls.append(("measure", list(cells), ks)) or measure
        ),
    )
    monkeypatch.setattr(
        script, "read_phase", lambda args: (calls.append(("read",)) or read),
    )
    return calls


@pytest.mark.parametrize("phase,expected", [
    ("measure", ["buffer", "measure"]),
    ("read", ["read"]),
    ("all", ["buffer", "measure", "read"]),
])
def test_main_runs_the_phases_it_was_asked_for_and_read_needs_no_data(monkeypatch, phase, expected):
    """`read` opens no replay buffer and splits no episodes: it pools records that
    are already on disk, so it must run where the data directory does not exist."""
    calls = _stub_main(monkeypatch)
    assert script.main(["--phase", phase]) == script.EXIT_OK
    assert [call[0] for call in calls] == expected


def test_main_measures_each_distinct_cell_once_and_hands_over_the_ks(monkeypatch):
    calls = _stub_main(monkeypatch)
    script.main(["--phase", "measure", "--arms", "pixel_ae", "frozen_ssl", "pixel_ae",
                 "--seeds", "1", "0", "1"])
    (measure,) = [call for call in calls if call[0] == "measure"]
    assert measure[1] == [("pixel_ae", 1), ("pixel_ae", 0), ("frozen_ssl", 1), ("frozen_ssl", 0)]
    assert measure[2] == K_REPORTED
    calls.clear()
    script.main(["--phase", "measure"], ks=(CONTRAST_K,))
    assert [c for c in calls if c[0] == "measure"][0][2] == (CONTRAST_K,)


@pytest.mark.parametrize("failing", ["measure", "read"])
def test_main_returns_the_first_phase_status_that_is_not_ok(monkeypatch, failing):
    """A refused measure stops `all` before the read; a refused read is the run's
    status. (`measure_phase` returns OK on a broken anchor -- 42 lives in `read`.)"""
    kwargs = {failing: script.EXIT_ANCHOR_BROKEN if failing == "read" else script.EXIT_NO_CHECKPOINTS}
    calls = _stub_main(monkeypatch, **kwargs)
    status = script.main(["--phase", "all"])
    assert status == kwargs[failing]
    assert ("read",) in calls if failing == "read" else ("read",) not in calls


def test_read_exits_map_exactly_the_two_refusals():
    assert script.READ_EXITS == {"UNRESOLVED_BASE": 41, "UNRESOLVED_ANCHOR": 42}


# ---------------------------------------------------------------------------
# Reading E's caption is suppressed at a seed count it cannot describe
# ---------------------------------------------------------------------------


def test_reading_e_caption_seeds_mirrors_the_closed_modules_own_literal():
    """`retention.format_reading_retention` hardcodes its caption's denominator
    ("in N of 3 seeds and M of 3 arms") as a bare literal, and `retention` is
    closed, so `READING_E_CAPTION_SEEDS` is a MIRROR. A mirror that can drift is
    worse than no mirror: this renders the real caption and reads the denominator
    back out of it, so changing the constant here without the module -- or the
    module without the constant -- fails."""
    from mbfps.eval.retention import format_reading_retention, reading_retention

    records = _records()
    inputs = script.retention_inputs(records)
    caption = format_reading_retention(
        reading_retention(inputs, z_bearing=script.CORRECTED_Z_BEARING), inputs,
    ).splitlines()[0]
    match = re.search(r"of (\d+) seeds and \d+ of (\d+) arms", caption)
    assert match, f"the caption's shape moved; it now reads {caption!r}"
    assert int(match.group(1)) == script.READING_E_CAPTION_SEEDS, (
        "the caption's seed denominator and READING_E_CAPTION_SEEDS disagree, so "
        "the suppression is keyed on a number the caption no longer prints"
    )


def test_reading_e_is_unreadable_at_a_seed_count_its_caption_cannot_describe(
    tmp_path, capsys,
):
    """The THIRD hardcode in Reading E's caption, and the one that was missed.
    Its "3 arms" half is enforced by `_validate_family_shape`; its "3 seeds" half
    is enforced nowhere, and `--seeds` accepts any list -- so a four-seed run
    printed "in 2 of 3 seeds" above twelve rows each reading `/4`, into
    `width.txt`, as the record. That is the caption-disagreeing-with-its-columns
    defect this project has shipped three times.

    Reading F is unaffected and must still be taken: its caption is built from
    its own `inputs`, so it is correct at any seed count. The companion says it
    cannot be read; it does not veto a verdict that can.

    `--seeds` must be passed: the read phase loads exactly `args.seeds`, so
    writing a fourth record without asking for it would leave nine records read
    and prove nothing. That is also why this is reachable ONLY from the CLI."""
    seeds = (0, 1, 2, 3)
    status, out = _read(
        tmp_path, capsys, _records(seeds=seeds),
        "--seeds", *[str(s) for s in seeds],
    )
    assert status == 0, "the companion must not veto Reading F"
    assert "unreadable:" in out
    assert f"carry {len(seeds)}" in out, "the reason names the seed count it found"
    assert "of 3 seeds and" not in out, (
        "the caption printed its hardcoded denominator over rows that read "
        f"/{len(seeds)}"
    )
    assert _verdict_lines(out), "Reading F's own verdict is still printed"


def test_the_pooled_row_count_refuses_a_missing_horizon_by_name(tmp_path, capsys):
    """The one index on the read path that bypassed this file's `_get` discipline.
    Records whose `passes` carry every horizon but whose `rows` lack one used to
    give a bare `KeyError` out of `corrected_reading_e` -- after
    `_reading_e_unreadable` had already reported the records readable, which is
    the worst possible moment for a traceback."""
    records = _records()
    for record in records.values():
        record["rows"].pop(script.k_key(K_REPORTED[0]))
    assert script._reading_e_unreadable(records) is None, (
        "the premise: these records look readable, which is what makes a bare "
        "KeyError here the worst possible moment for one"
    )
    with pytest.raises(SystemExit) as raised:
        script.retention_inputs(records)
    assert f"k = {K_REPORTED[0]}" in str(raised.value)
    assert "cannot be read as zero" in str(raised.value)


def test_every_phase_read_test_routes_through_the_measure_guard():
    """`_main_read` is the guard whose absence cost a real 7-minute measure, and
    until now nothing made using it MECHANICAL -- a future test calling
    `script.main(["--phase", "read", ...])` directly would bypass it and could
    run the GPU again. This reads this module's own source and refuses a bare
    `main` call carrying `--phase` outside the two helpers that patch the measure
    half."""
    # Built at runtime so this scan's own source lines do not match it -- the
    # first version of this test flagged its own filter expression, and the one
    # before that flagged its own docstring.
    call = "script" + ".main("
    phase = '"' + "read" + '"'
    offenders = [
        line.strip() for line in Path(__file__).read_text().splitlines()
        if call in line and phase in line
        and not line.lstrip().startswith(("`", "#", '"'))
    ]
    assert not offenders, (
        "call `_main_read(...)` instead of `script.main([...])` so the measure "
        f"half stays unreachable: {offenders}"
    )


def test_the_width_bias_table_refuses_records_that_disagree_on_rung_width(tmp_path):
    """`width_bias_table` is a public entry point, so it does not assume
    `require_one_protocol` ran -- and that guard was unreachable by any test,
    which is how a guard becomes decoration. Driven directly here."""
    records = _records()
    victim = sorted(records)[-1]
    records[victim]["rung_width"] = dict(
        records[victim]["rung_width"], deterministic=RUNG_WIDTH["deterministic"] + 1,
    )
    with pytest.raises(SystemExit) as raised:
        script.width_bias_table(records)
    assert "rung_width" in str(raised.value)
    assert script._cell_name(victim) in str(raised.value), "names the odd cell out"
