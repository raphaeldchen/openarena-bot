"""M3j's script: the three gathers, the shared bases, and the record schema."""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

from mbfps.eval.retention import K_REPORTED, RUNGS, TARGETS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latent_retention.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_retention_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


def _gathered(n_windows: int = 8, steps: int = 20, width: int = 6, h_dim: int = 3,
              seed: int = 0) -> dict:
    """A `gather_probe_data`-shaped dict with REAL structure: the deterministic
    half of the latent remembers the previous frame's position, so the
    deterministic rung must show a gain and a flat rung must not."""
    rng = np.random.default_rng(seed)
    rows = n_windows * steps
    window = np.repeat(np.arange(n_windows), steps)
    step = np.tile(np.arange(steps), n_windows)
    episode = window // 2
    # A per-row noise term on top of the linear trend. WITHOUT it, `p(t) -
    # p(t-k)` is a pure function of k alone (the trend's slope is fixed across
    # window and step), so every backward displacement is a CONSTANT vector --
    # zero variance, and `probe.fit_probe`'s own ridge selection raises
    # "every target column has zero variance" before any rung is scored. The
    # noise is exactly what only `h`'s lagged position (this row's OWN noisy
    # `p(t-1)`, not a resampled draw) can resolve: `enc(t)` alone sees this
    # row's noise but has no way to know an earlier row's independent draw,
    # while `enc(t) (+) h(t)` has both `p(t)` and `p(t-1)` as separate inputs
    # and recovers `p(t) - p(t-1)` by exact subtraction regardless of it.
    pos = np.column_stack([
        window * 100.0 + step * 3.0, window * 50.0 - step * 2.0,
    ]) + rng.normal(scale=5.0, size=(rows, 2))
    radians = (
        np.deg2rad((step * 7.0 + window * 11.0) % 360.0)
        + rng.normal(scale=0.2, size=rows)
    )
    targets = np.column_stack([pos, np.sin(radians), np.cos(radians)])
    enc = np.column_stack([pos, rng.normal(size=(rows, width - 2))])
    # `embedding` (the model's PREDICTED embedding) and `encoder_embedding`
    # (the raw encoder output) must be genuinely DIFFERENT arrays here, even
    # though nothing in this task reads `embedding` -- see
    # `gather_probe_data`'s own docstring: comparing the filtering gain against
    # the predicted embedding "would instead ask whether the latent beats its
    # own head's reconstruction, which a latent can win while carrying no
    # history at all". Making them byte-identical (as this fixture used to)
    # means a bug that read `"embedding"` where `"encoder_embedding"` belongs
    # -- in `_split_for`'s `base=` or `base_control`'s `probe_for` -- would
    # still pass every test in this file while answering a different question
    # in the real artefact. The offset is large (scale=20 against `enc`'s
    # position columns of std ~sqrt(5**2+100**2/12)-ish and noise columns of
    # std 1) so a swap changes the numbers a great deal, not marginally.
    embedding = enc + rng.normal(scale=20.0, size=enc.shape)
    lagged = np.roll(pos, 1, axis=0)
    latent = np.column_stack([
        lagged, rng.normal(size=(rows, h_dim - 2)), rng.normal(size=(rows, 4)),
    ])
    return {
        "latent": latent, "embedding": embedding, "encoder_embedding": enc,
        "targets": targets, "window": window, "step": step, "episode": episode,
    }


def test_exit_codes_are_39_and_40_and_do_not_collide_with_m3i():
    """M3i holds 38 (`EXIT_CONTROL_LEAKED`). A reused number makes two different
    refusals indistinguishable to a caller reading `$?`."""
    assert script.EXIT_BASE_UNRESOLVED == 39
    assert script.EXIT_MOTION_UNRESOLVED == 40
    assert script.EXIT_OK == 0


def test_phases_are_measure_read_all():
    assert script.PHASES == ("measure", "read", "all")


def test_episode_budget_is_the_filtering_gain_defaults():
    """`select_episodes` at 20, not 4. `probe.filtering_gain`'s docstring records
    that a 4-episode selection split picks the wrong ridge and FLIPS THE SIGN of
    the gain (+0.0325 against -0.0208 on the same checkpoint). Shrinking this to
    save a gather does not make the number noisier, it makes it wrong."""
    assert script.FIT_EPISODES == 20
    assert script.SELECT_EPISODES == 20


def test_record_path_names_both_the_arm_and_the_seed(tmp_path):
    """M3i's task reports collided because two milestones wrote the same
    filename; a record named by only one of (arm, seed) collides nine ways."""
    path = script.retention_record_path(tmp_path, "pixel_ae", 2)
    assert path.name == "retention_pixel_ae_seed2.json"


def test_k_key_carries_no_dot():
    """`write_record` addresses non-finite fields by dotted path, so a key with
    a '.' in it would be unaddressable."""
    assert script.k_key(15) == "k15"
    assert "." not in script.k_key(4)


def test_cell_ladder_covers_every_target_rung_and_horizon():
    """The rule is a disjunction over k across four rungs and two targets. A
    ladder missing a cell would silently make the disjunction narrower."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1, 2))
    assert set(ladder["ladder"]) == set(TARGETS)
    for target in TARGETS:
        assert set(ladder["ladder"][target]) == {"k1", "k2"}
        for k in ("k1", "k2"):
            assert set(ladder["ladder"][target][k]) == set(RUNGS)


def test_cell_ladder_hands_every_rung_a_byte_identical_base(monkeypatch):
    """Spec 2.1: the four gains at one `(target, k)` must be differences against
    the SAME base, or they are not comparable to each other -- only each to its
    own fit. `fit_probe` is deterministic, so identical base arrays and an
    identical target give an identical base level, which is therefore the
    property to pin is therefore that the ARRAYS are identical, which four
    independent row selections would silently break while still producing four
    plausible gains.

    Both targets share a row set at a given k (Task 4 pins that), so the base is
    byte-identical across targets too and one distinct array is the correct
    count, not two."""
    calls = []
    real = script.gain_from_blocks

    def counting(fit, select, score, **kwargs):
        calls.append(np.asarray(fit.base).tobytes())
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", counting)
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    assert len(calls) == len(TARGETS) * len(RUNGS), "one call per (target, rung)"
    assert len(set(calls)) == 1, (
        "the rungs were not handed one identical base array; their gains are "
        "against different bases and cannot be compared to each other"
    )


def test_cell_ladder_groups_the_bootstrap_on_episodes_not_windows(monkeypatch):
    """Spec 3.1. 229 non-overlapping windows cut from 24 trajectories are not 229
    independent observations, and every reading from M3e onward clusters on the
    episodes. Passing `window` here would silently narrow every interval."""
    seen = []
    real = script.gain_from_blocks

    def recording(fit, select, score, **kwargs):
        seen.append(np.asarray(kwargs["groups"]).copy())
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", recording)
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    for groups in seen:
        assert np.unique(groups).size == np.unique(score["episode"]).size, (
            "the bootstrap is grouped on windows, not episodes"
        )


def test_cell_ladder_scores_every_rung_on_the_same_rows():
    """The row count per k is recorded on the artefact so the shared-row-set
    claim (spec 2.1) is checkable after the fact; this pins the count itself.

    (An earlier version of this test also asserted that `n_scored_windows` was
    equal across all four rungs at k4 -- but `groups` is computed once per
    `(k, target)` and handed to all four rungs by `cell_ladder`'s own
    structure, so those four values are equal BY CONSTRUCTION for any
    implementation, correct or not. That half caught nothing and claimed to
    catch "rungs scored on different groups", which was never at risk here.)
    """
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1, 4))
    # 8 windows of 20 steps: k=1 drops 8 rows, k=4 drops 32.
    assert ladder["rows"] == {"k1": 152, "k4": 128}


def test_cell_ladder_finds_the_gain_a_lagged_latent_carries():
    """A known-answer case. The deterministic half of `latent` is the PREVIOUS
    frame's position, so `enc(t) (+) h(t)` determines `p(t) - p(t-1)` exactly
    while `enc(t)` alone cannot. If the deterministic rung shows no gain here,
    the rows, the block and the target have come apart and no null this script
    produces would mean anything.

    The bounds are 0.4 / 0.2, not a bare `> 0`, and that is load-bearing: the
    fixture's `"embedding"` (the model's predicted embedding) and
    `"encoder_embedding"` (the raw encoder output) are deliberately DIFFERENT
    arrays (see `_gathered`'s docstring), and `_split_for`'s `base=` must read
    `"encoder_embedding"`. Measured on this fixture: reading the right array
    gives gain ~0.776 / ci_low ~0.764; swapping in `"embedding"` for `base=`
    still clears a bare `> 0.1` / `> 0.0` (it measures ~0.102 / ~0.016, right
    on the line) while answering a different research question in the real
    artefact -- the base array is noisier there and drags down the joint fit's
    ability to recover `p(t)`, which is exactly the confound this bound is
    here to catch."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    deterministic = ladder["ladder"]["translation"]["k1"]["deterministic"]
    assert deterministic["gain"] > 0.4, deterministic
    assert deterministic["ci_low"] > 0.2, deterministic


def test_cell_ladder_shows_no_gain_for_a_rung_that_adds_only_noise():
    """The other half of the known-answer pair. `z` here is pure noise, so its
    rung must not clear -- otherwise the statistic is rewarding width and the
    base is not really in both arms.

    The bare `ci_low <= 0.0` sign check alone passes on a knife-edge margin
    here (measured `ci_low` ~ -7.1e-07 against a CI width of ~7.5e-07 -- the
    sign is real but the margin is a rounding error's width), which a
    numpy/BLAS/platform change could flip with no change in what is actually
    being measured -- and a spurious red on this exact test is how a
    known-answer pair gets "fixed" into uselessness. The magnitude claim the
    docstring above actually makes (stochastic adds nothing LIKE what
    deterministic adds) has a much wider margin: measured deterministic gain
    ~+0.78 against stochastic ~-2.8e-07, six orders of magnitude apart."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    ladder = script.cell_ladder(fit, select, score, h_dim=3, ks=(1,))
    stochastic = ladder["ladder"]["translation"]["k1"]["stochastic"]
    deterministic = ladder["ladder"]["translation"]["k1"]["deterministic"]
    assert stochastic["ci_low"] <= 0.0, stochastic
    assert stochastic["gain"] < deterministic["gain"] / 10, (stochastic, deterministic)


def test_base_control_reports_the_level_not_a_gain():
    """`enc(t)` -> absolute position. There is nothing to take an increment
    over: this IS the arm the increments are measured against."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    control = script.base_control(fit, select, score)
    assert control["r2"] > 0.5, (
        "enc(t) carries position exactly in this fixture; a low r2 means the "
        "control is reading the wrong array"
    )
    assert "gain" not in control, "the base control is a level, not a gain"


def test_base_control_records_position_r2_beside_the_gated_mean():
    """`r2` stays the GATED 4-column mean (position and heading mixed); it must
    not move. `position_r2` is a companion that isolates position alone, and in
    this fixture heading (sin, cos of a noisy angle) is harder to read than
    position (a noisy linear trend) -- per-column r2 is approximately
    [1.0, 1.0, 0.126, 0.859], so `position_r2` (~1.0) sits well above the
    4-column `r2` (~0.746). A cell shaped like this clears the 0.10 gate on
    `r2` alone while telling a different story about position specifically,
    which is exactly the case `position_r2` exists to surface."""
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    control = script.base_control(fit, select, score)
    assert {"r2", "position_r2", "per_column_r2"} <= set(control)
    assert len(control["per_column_r2"]) == 4
    assert control["position_r2"] == pytest.approx(sum(control["per_column_r2"][:2]) / 2)
    assert control["position_r2"] != pytest.approx(control["r2"]), (
        "position_r2 must not equal the mixed 4-column r2 on a fixture where "
        "heading and position read differently"
    )
    assert control["position_r2"] > control["r2"], (
        "position reads easier than heading in this fixture, so position_r2 "
        "should sit above the mean that heading drags down"
    )


def test_cell_ladder_refuses_a_horizon_no_window_is_long_enough_for():
    """A k with no rows is a protocol refusal, not an empty table. Reporting a
    horizon whose gain was computed on zero rows would put a number in the
    record that describes nothing."""
    with pytest.raises(ValueError, match="no window is long enough"):
        script.cell_ladder(
            _gathered(steps=6), _gathered(steps=6, seed=1), _gathered(steps=6, seed=2),
            h_dim=3, ks=(15,),
        )


def test_cell_ladder_threads_the_seed_into_gain_from_blocks(monkeypatch):
    """`gain_from_blocks` defaults its own `seed` to 0, so a caller that never
    passes one hands every cell, rung, target and horizon the SAME bootstrap
    draw -- correlating the interval noise across CELLS that the seeds x arms
    agreement rule in `retention.py` treats as independent, and that rule
    carries this milestone's entire multiple-comparison burden.
    `probe.filtering_gain` threads its own `seed` straight through to
    `gain_from_blocks`; `cell_ladder` must do the same rather than leaving it
    defaulted."""
    seen = []
    real = script.gain_from_blocks

    def recording(fit, select, score, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(fit, select, score, **kwargs)

    monkeypatch.setattr(script, "gain_from_blocks", recording)
    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    script.cell_ladder(fit, select, score, h_dim=3, seed=17, ks=(1, 4))
    assert seen, "gain_from_blocks was never called"
    assert all(s == 17 for s in seen), (
        f"cell_ladder must thread its own `seed` into every gain_from_blocks "
        f"call rather than leaving it defaulted; got {set(seen)}"
    )


# ---------------------------------------------------------------------------
# The impure half: gather_three_splits, measure_cell, measure_phase.
#
# Before this, none of it had coverage -- the self_check gate, the
# cell.record["steps"] key, the seed triple on the three gathers, and the
# record schema Task 8 reads. A transposed seed (scoring at seed+2 instead of
# seed+1) would silently score a different draw from the one criterion 4
# scores, defeating the stated reason the convention exists, while producing
# entirely plausible numbers.
# ---------------------------------------------------------------------------


def test_gather_three_splits_takes_the_filtering_gain_draws(monkeypatch):
    """The three gathers must carry the exact `(paths, seed)` pairs
    `probe.filtering_gain` uses: fit on `train[:FIT_EPISODES]` at `seed`,
    select on the NEXT `SELECT_EPISODES` episodes at `seed + 2`, and score on
    `val` at `seed + 1` with `limit=FIT_EPISODES` -- NOT `len(val)`, which is
    `gather_probe_data`'s fit-episode rule and is what keeps this diagnostic
    describing the same rows `filtering_gain` does. No model needed:
    `gather_probe_data` itself is replaced with a recorder."""
    calls = []

    def recorder(model, paths, backbone, device, *, context, horizon, limit, seed):
        calls.append({"paths": list(paths), "seed": seed, "limit": limit})
        return _gathered()

    monkeypatch.setattr(script, "gather_probe_data", recorder)
    prepared = types.SimpleNamespace(
        model="the-model",
        common={"feature_backbone": "the-backbone", "device": "cpu"},
        context=5, horizon=45,
    )
    train = [f"train_ep{i}" for i in range(50)]
    val = [f"val_ep{i}" for i in range(10)]

    script.gather_three_splits(prepared, train, val, seed=7)

    assert len(calls) == 3, "expected exactly three gathers: fit, select, score"
    fit_call, select_call, score_call = calls
    assert fit_call["paths"] == train[:script.FIT_EPISODES]
    assert fit_call["seed"] == 7
    assert select_call["paths"] == train[
        script.FIT_EPISODES:script.FIT_EPISODES + script.SELECT_EPISODES
    ]
    assert select_call["seed"] == 9
    assert score_call["paths"] == val
    assert score_call["seed"] == 8
    assert score_call["limit"] == script.FIT_EPISODES


def test_measure_cell_refuses_a_failed_self_check_and_records_steps(monkeypatch):
    """Two claims that had zero coverage before this. First: a failed
    self-check refuses (`EXIT_SELF_CHECK_FAILED`, no record) before any ladder
    is built -- the 30 check is NOT free the way 12/14 are, because it reduces
    the per-window rows a regression could get wrong while `prepare_cell`'s own
    14 check still passes on the mean curve. Second: on success, the record's
    `"step"` comes from `cell.record["steps"]` (not the singular `"step"` the
    brief's own sketch read, which does not exist on a real study record), and
    the record's top-level key set is exactly the schema `measure_phase` writes
    to disk (`measure_phase` itself adds the run-level `"git_sha"` afterward,
    so it is not part of what `measure_cell` alone returns)."""
    prepared = types.SimpleNamespace(
        model=types.SimpleNamespace(
            rssm=types.SimpleNamespace(cfg=types.SimpleNamespace(h_dim=3))
        ),
        embedding_probe=object(),
        common={},
        context=5,
        horizon=20,
    )

    class _Check:
        def __init__(self, ok):
            self.ok = ok

        def failures(self):
            return ["windows do not match the diagnostic"]

        def record(self):
            return {"ok": self.ok}

    def fake_prepare_cell(args, cell, device, train, val):
        return script.EXIT_OK, prepared

    def fake_reference_trajectories(*args, **kwargs):
        return "a-trajectory"

    def fake_gather_three_splits(prepared_arg, train_arg, val_arg, *, seed):
        assert prepared_arg is prepared
        return (_gathered(seed=0), _gathered(seed=1), _gathered(seed=2))

    monkeypatch.setattr(script, "prepare_cell", fake_prepare_cell)
    monkeypatch.setattr(script, "reference_trajectories", fake_reference_trajectories)
    monkeypatch.setattr(script, "gather_three_splits", fake_gather_three_splits)

    cell = types.SimpleNamespace(
        arm="pixel_ae", seed=3, record={"steps": 20000}, diagnostic={"whatever": True},
    )
    train = [Path(f"train_ep{i}") for i in range(25)]
    val = [Path(f"val_ep{i}") for i in range(10)]

    monkeypatch.setattr(script, "self_check", lambda traj, diagnostic: _Check(False))
    status, record = script.measure_cell(object(), cell, "cpu", train, val, ks=(1,))
    assert status == script.EXIT_SELF_CHECK_FAILED
    assert record is None

    monkeypatch.setattr(script, "self_check", lambda traj, diagnostic: _Check(True))
    status, record = script.measure_cell(object(), cell, "cpu", train, val, ks=(1,))
    assert status == script.EXIT_OK
    assert record["step"] == 20000
    assert set(record) == {
        "arm", "seed", "step", "record_git_sha", "device", "context", "horizon",
        "h_dim", "ks", "split_seed", "torch_version", "ladder", "rows",
        "base_control", "clusters", "windows", "self_check", "episodes",
    }
