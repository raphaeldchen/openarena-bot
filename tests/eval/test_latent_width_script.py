"""M3k's script: three passes over one gather, and the anchors that gate them."""

import copy
import importlib.util
import inspect
import types
import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.retention import (
    RESAMPLES, RUNGS, TARGETS, backward_rotation, backward_translation, rung_block,
    shifted_rows,
)
from mbfps.eval.width import (
    ANCHOR, CONTRAST_K, DOWN_WIDTH, PASSES, PROJECTION_SEED, RUNG_WIDTH, UP_WIDTH,
    pass_block,
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
    return data["encoder_embedding"][rows], pass_block(native, pass_name), values, rows


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


def _dropping_builder(real, *, at_k: int):
    """A `TARGET_BUILDERS` entry with a row rule of its own: at `at_k` it drops
    the first row `real` returns, and elsewhere it is `real`. `values` and `rows`
    are shortened together, so what it hands back is internally consistent."""
    def builder(targets, window, step, k):
        values, rows = real(targets, window, step, k)
        return (values[1:], rows[1:]) if k == at_k else (values, rows)
    return builder


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
}


_REQUIRED_PROTOCOL_FIELDS = {
    "windows.episode", "episodes.val", "context", "horizon", "device", "ks",
    "torch_version", "git_sha", "projection_seed", "rung_width",
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
