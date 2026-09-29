"""M3j's script: the three gathers, the shared bases, and the record schema."""

import importlib.util
import json
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
    identical target give an identical base level, so the property to pin is
    that the ARRAYS are identical, which four independent row selections would
    silently break while still producing four plausible gains.

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


def test_base_control_probes_the_raw_encoder_embedding_not_the_predicted_one():
    """`base_control`'s `probe_for` must read `data["encoder_embedding"]`, not
    `data["embedding"]` (the model's PREDICTED embedding) -- `_gathered`'s own
    docstring names this exact call site as one of the two the asymmetric
    fixture exists to protect, and `test_base_control_reports_the_level_not_a_gain`
    only asserts `r2 > 0.5`, which this fixture's `embedding` array also clears
    (it is `encoder_embedding` plus extra noise, still linearly related to
    position) -- so that test alone cannot catch the swap.

    This test recomputes the expected r2 independently, straight from
    `encoder_embedding` through the same `fit_probe` / `probe_r2` calls
    `base_control` makes internally, and pins `control["r2"]` to that exact
    value. It also proves the mutation would be CAUGHT: probing `embedding`
    instead gives a measurably different r2 on this fixture."""
    from mbfps.eval.probe import fit_probe, probe_r2

    fit, select, score = (_gathered(seed=s) for s in (0, 1, 2))
    control = script.base_control(fit, select, score)

    probe = fit_probe(
        fit["encoder_embedding"], fit["targets"],
        select["encoder_embedding"], select["targets"],
    )
    expected = probe_r2(probe, score["encoder_embedding"], score["targets"])
    assert control["r2"] == pytest.approx(expected), (
        "base_control's r2 must come from encoder_embedding, computed exactly "
        "as this test computes it independently"
    )

    wrong_probe = fit_probe(
        fit["embedding"], fit["targets"], select["embedding"], select["targets"],
    )
    wrong = probe_r2(wrong_probe, score["embedding"], score["targets"])
    assert expected != pytest.approx(wrong), (
        "encoder_embedding and embedding must give distinguishable r2 on this "
        "fixture, or a probe_for swap would not be catchable at all"
    )


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
    `val` at `seed + 1`.

    The scored split takes EVERY validation episode, which is the one place
    this departs from `filtering_gain`: that function's `limit` caps the fit
    split and the scored split at the same number, so mirroring it literally
    scored 20 of the 24 validation episodes and silently discarded four. The
    smoke run caught it -- 20 clusters and 9,261 rows at k = 1 against the 24
    and 11,221 the spec accepts on. The protocol every milestone since M3d
    reads is 229 windows over 24 validation episodes.

    No model needed: `gather_probe_data` itself is replaced with a recorder."""
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
    assert score_call["limit"] == len(val), (
        f"the scored split was capped at {score_call['limit']} of {len(val)} "
        "validation episodes; the reading must be taken on all of them"
    )
    assert score_call["limit"] != script.FIT_EPISODES, (
        "the fixture must make the two numbers differ, or this cannot catch "
        "a regression back to filtering_gain's shared cap"
    )


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
    status, record = script.measure_cell(_measure_args(), cell, "cpu", train, val, ks=(1,))
    assert status == script.EXIT_SELF_CHECK_FAILED
    assert record is None

    monkeypatch.setattr(script, "self_check", lambda traj, diagnostic: _Check(True))
    status, record = script.measure_cell(_measure_args(), cell, "cpu", train, val, ks=(1,))
    assert status == script.EXIT_OK
    assert record["step"] == 20000
    assert set(record) == {
        "arm", "seed", "step", "record_git_sha", "device", "context", "horizon",
        "h_dim", "ks", "split_seed", "torch_version", "ladder", "rows",
        "base_control", "clusters", "windows", "self_check", "episodes",
    }


def test_measure_phase_loads_cells_from_source_and_writes_records_to_out(monkeypatch, tmp_path):
    """Correction 2, pinned by watching which directory each call actually
    used, not by asserting on a path string an unrelated implementation could
    satisfy by coincidence.

    `--out` is this script's OWN record directory -- empty until `measure`
    writes to it -- and the nine M3c checkpoints, study records and
    diagnostics live in the study directory instead. `measure_phase` used to
    call `load_cell(args.out, ...)`, which would look for those three files in
    the empty output directory and fail before a single cell was measured;
    the fix reads `args.source`. `measure_cell` itself is replaced here, so
    this test is only about which directory each of the two real calls this
    function makes -- `load_cell` and `write_record` -- actually reads from
    and writes to.
    """
    source_dir = tmp_path / "source"
    out_dir = tmp_path / "out"
    source_dir.mkdir()

    seen_load = []

    def fake_load_cell(directory, arm, seed):
        seen_load.append(Path(directory))
        return types.SimpleNamespace(arm=arm, seed=seed, record={}, diagnostic={})

    def fake_measure_cell(args, cell, device, train, val, ks=None):
        return script.EXIT_OK, {
            "arm": cell.arm, "seed": cell.seed, "ladder": {},
            "base_control": {"r2": 1.0},
        }

    written = []

    def fake_write_record(path, record):
        written.append(Path(path))
        return record

    monkeypatch.setattr(script, "load_cell", fake_load_cell)
    monkeypatch.setattr(script, "measure_cell", fake_measure_cell)
    monkeypatch.setattr(script, "write_record", fake_write_record)
    monkeypatch.setattr(script, "git_sha", lambda: "deadbeef")

    args = types.SimpleNamespace(out=out_dir, source=source_dir)
    status = script.measure_phase(args, [("pixel_ae", 0)], "cpu", [], [], ks=(1,))

    assert status == script.EXIT_OK
    assert seen_load == [source_dir], (
        f"the cell was loaded from {seen_load}, not from --source ({source_dir})"
    )
    assert written == [script.retention_record_path(out_dir, "pixel_ae", 0)], (
        "the record was not written under --out"
    )
    assert out_dir.exists(), "the output directory must still be created"


# ---------------------------------------------------------------------------
# read: the nine records pooled into Reading E.
# ---------------------------------------------------------------------------


def _measure_args(source="runs/m3_study_v2"):
    """The attributes `measure_cell` reads off its args. `source` is the STUDY
    directory: `prepare_cell` loads the checkpoint from the `out` of the args it
    is handed, so passing this script's own `--out` through would hunt for the
    nine M3c checkpoints among the retention records.
    """
    import types as _types
    return _types.SimpleNamespace(
        source=source, out="runs/m3j_retention", device="cpu",
        context=None, horizon=None,
    )


def test_measure_cell_loads_the_checkpoint_from_source_not_out(monkeypatch):
    """`prepare_cell` loads the checkpoint from the `out` of the args IT is
    handed (`trust_horizon.load_checkpoint_model(args.out, ...)`), which is the
    STUDY directory -- while this script's own `--out` holds the retention
    records. Handing `args` through unchanged hunts for the nine M3c
    checkpoints among the records and refuses every cell, so a `measure` run
    fails on its first cell with a refusal that names the wrong directory.

    Nothing else catches this: every other test in this file stubs
    `prepare_cell` or feeds `cell_ladder` directly, so the one line that routes
    the directories is exercised only here.
    """
    seen = {}

    def recording_prepare_cell(cell_args, cell, device, train, val):
        seen["out"] = str(cell_args.out)
        seen["device"] = cell_args.device
        return script.EXIT_SELF_CHECK_FAILED, None

    monkeypatch.setattr(script, "prepare_cell", recording_prepare_cell)
    import types as _types
    cell = _types.SimpleNamespace(
        arm="pixel_ae", seed=0, record={"steps": 20000}, diagnostic={},
    )
    args = _measure_args(source="runs/m3_study_v2")
    status, record = script.measure_cell(args, cell, "cpu", [], [])

    assert status == script.EXIT_SELF_CHECK_FAILED and record is None
    assert seen["out"] == "runs/m3_study_v2", (
        f"prepare_cell was handed out={seen['out']!r}; it must be the STUDY "
        "directory (--source), never this script's record directory (--out)"
    )
    assert seen["out"] != args.out


_RECORD_ARMS = ("frozen_ssl", "pixel_ae", "random_vit")


def _cell_offset(arm: str, seed: int) -> float:
    """A small, deterministic, per-`(arm, seed)` offset -- distinguishable
    across BOTH arms and seeds -- so a record that gets pooled under the
    wrong arm, or fed to every arm at once, changes a printed number rather
    than reproducing a byte-identical payload.

    Before this, `_record`'s ladder and base-control numbers depended only on
    `clearing` and `base_r2` -- never on `arm` or `seed` -- so with the
    `arm`/`seed` labels dropped, all nine records serialised to ONE distinct
    payload: an implementation that paired `frozen_ssl` with `pixel_ae`'s
    seeds, or fed all nine records to every arm, produced identical
    `RungArm`s / `BaseControl`s and no test noticed.

    Kept well inside the margins every existing test already relies on
    (`BASE_R2_FLOOR` +/- 0.05, the hit/non-hit gain gap of ~0.26) so it can
    never flip a clearing boundary on its own -- callers that need the exact
    boundary (`BASE_R2_FLOOR` itself) pass `distinguish=False` instead.
    """
    return 0.005 * _RECORD_ARMS.index(arm) + 0.001 * seed


def _record(arm: str, seed: int, *, clearing=(), base_r2: float = 0.30,
            position_r2: float | None = None, episodes: int = 24,
            distinguish: bool = True) -> dict:
    """A minimal record with the schema `read` addresses. `clearing` is a set
    of `(target, k, rung)` triples whose interval excludes zero. `position_r2`
    defaults to `base_r2` (no position/heading divergence) -- Correction 3's
    own two tests below override a single cell's to build one.

    `distinguish=True` (the default) offsets this cell's gains and base r2 by
    `_cell_offset(arm, seed)` -- see that function's docstring for why. Pass
    `distinguish=False` for a test that needs every cell byte-identical
    except for the field under test, such as pinning the exact
    `BASE_R2_FLOOR` boundary.
    """
    offset = _cell_offset(arm, seed) if distinguish else 0.0
    ladder = {}
    for target in TARGETS:
        ladder[target] = {}
        for k in K_REPORTED:
            ladder[target][f"k{k}"] = {}
            for rung in RUNGS:
                hit = (target, k, rung) in clearing
                ladder[target][f"k{k}"][rung] = {
                    "gain": (0.25 if hit else -0.01) + (offset if hit else -offset),
                    "ci_low": (0.10 if hit else -0.08) + (offset if hit else -offset),
                    "ci_high": (0.40 if hit else 0.03) + (offset if hit else -offset),
                    "joint_r2": 0.5, "embedding_r2": 0.25,
                    "confidence": 0.95, "n_scored_windows": episodes,
                    "ridge_selected": True, "joint_ridge": 1e3,
                    "embedding_ridge": 1e3,
                }
    r2 = base_r2 + offset
    return {
        "arm": arm, "seed": seed, "step": 20000, "device": "cpu",
        "context": 5, "horizon": 45, "h_dim": 512,
        "ladder": ladder,
        "rows": {f"k{k}": 11221 - k * 229 for k in K_REPORTED},
        "base_control": {
            "r2": r2,
            "position_r2": r2 if position_r2 is None else position_r2,
            "ridge": 1e3, "ridge_selected": True, "rows": 11450,
        },
        "clusters": episodes,
        "windows": {"episode": list(range(episodes)), "window": list(range(episodes))},
        "self_check": {"ok": True},
        "episodes": {"fit": [], "select": [], "val": [f"e{i}" for i in range(episodes)]},
    }


def _records(clearing=(), base_r2: float = 0.30, position_r2: float | None = None,
             distinguish: bool = True) -> dict:
    return {
        (arm, seed): _record(
            arm, seed, clearing=clearing, base_r2=base_r2, position_r2=position_r2,
            distinguish=distinguish,
        )
        for arm in _RECORD_ARMS for seed in (0, 1, 2)
    }


def test_require_readable_plan_refuses_a_plan_too_narrow_to_reach_a_verdict():
    """The rule needs ARMS_REQUIRED arms of SEEDS_REQUIRED seeds. A narrower
    plan cannot reach any status, so reading it would print a verdict the data
    could not have supported -- M3i shipped exactly that defect and caught it in
    review, where a narrowed plan printed `z +996.13, 3/3 up` beside NO
    DIFFERENCE."""
    with pytest.raises(SystemExit):
        script.require_readable_plan(("pixel_ae",), (0, 1, 2))
    with pytest.raises(SystemExit):
        script.require_readable_plan(("pixel_ae", "frozen_ssl"), (0,))
    script.require_readable_plan(("pixel_ae", "frozen_ssl"), (0, 1))


# Every field `require_one_protocol` compares, mapped to a mutation that
# changes it on one victim cell. `windows.episode` through `device` are the
# five the module shipped with (before this fix, only `windows.episode` --
# via the one hand-written test this table replaces -- was ever driven to a
# refusal). `ks` and `torch_version` mirror `scripts/latent_motion.py`'s own
# `require_one_protocol`, which already compares them; `git_sha` is new even
# there, added because Task 9's acceptance step requires nine records at one
# `git_sha`, and pooling across two code versions is exactly what this
# refusal exists to prevent.
_PROTOCOL_FIELD_MUTATIONS = {
    "windows.episode": lambda r: r["windows"].update(episode=list(range(5))),
    "episodes.val": lambda r: r["episodes"].update(val=["different.npz"]),
    "context": lambda r: r.update(context=999),
    "horizon": lambda r: r.update(horizon=999),
    "device": lambda r: r.update(device="mps"),
    "ks": lambda r: r.update(ks=[1, 4]),
    "torch_version": lambda r: r.update(torch_version="1.9.0"),
    "git_sha": lambda r: r.update(git_sha="def456"),
}


@pytest.mark.parametrize("field", sorted(_PROTOCOL_FIELD_MUTATIONS))
def test_require_one_protocol_refuses_a_disagreement_on_every_compared_field(field):
    """Table-driven over every field `require_one_protocol` compares. `ks`,
    `torch_version` and `git_sha` are given a matching BASELINE value on every
    record first (a real record always carries them) before the victim cell's
    is changed, so this test is about the comparison catching a mismatch, not
    about the `.get(...)` default."""
    records = _records()
    for record in records.values():
        record["ks"] = [1, 4, 15]
        record["torch_version"] = "2.1.0"
        record["git_sha"] = "abc123"
    _PROTOCOL_FIELD_MUTATIONS[field](records[("pixel_ae", 1)])
    with pytest.raises(SystemExit, match="disagree on"):
        script.require_one_protocol(records)


def test_require_one_protocol_does_not_raise_when_ks_and_provenance_are_absent():
    """A record written before `ks`/`torch_version`/`git_sha` were compared --
    or before they existed at all -- must not be refused by their absence:
    `.get(...)` defaults each of the three identically across every record
    rather than raising a bare `KeyError` or reading a spurious disagreement.
    `_records()`'s fixture carries none of the three, so this is exactly that
    case."""
    script.require_one_protocol(_records())


def test_retention_inputs_tallies_seeds_and_arms_from_the_records():
    """The tally IS the rule, so it has to come from the per-seed intervals in
    the records rather than from a pooled number -- and each arm's OWN seeds,
    not any other cell's. Before this fix, every one of the nine records in
    `_records()` serialised to the same payload once its `arm`/`seed` labels
    were dropped (`_cell_offset`'s docstring), so an implementation that
    pooled the wrong three seeds under an arm -- or all nine under every arm
    -- produced identical tallies and this test could not tell the
    difference. `_record`'s `distinguish=True` default gives every `(arm,
    seed)` its own gain, so the exact per-arm MEAN is now pinned, not just its
    seed count."""
    clearing = {("translation", 4, "full")}
    records = _records(clearing)
    inputs = script.retention_inputs(records)
    arm = inputs.ladder["translation"][4]["full"]["pixel_ae"]
    assert arm.seeds_clear == 3 and arm.seeds_total == 3
    expected_gain = sum(
        records[("pixel_ae", seed)]["ladder"]["translation"]["k4"]["full"]["gain"]
        for seed in (0, 1, 2)
    ) / 3
    assert arm.gain == pytest.approx(expected_gain), (
        "pixel_ae's RungArm must be the mean of pixel_ae's OWN three seeds' "
        "gains, not another arm's or a pool of all nine"
    )
    other_arm = inputs.ladder["translation"][4]["full"]["frozen_ssl"]
    assert other_arm.gain != pytest.approx(arm.gain), (
        "frozen_ssl and pixel_ae must be built from their own distinguishable "
        "seeds; equal means here would mean the arm filter did nothing"
    )
    other = inputs.ladder["translation"][1]["full"]["pixel_ae"]
    assert other.seeds_clear == 0


def test_retention_inputs_refuses_a_non_finite_gain_in_any_record():
    """M3i's ledger note, closed. A NaN must stop the read, not read as a
    non-clear."""
    records = _records()
    records[("pixel_ae", 1)]["ladder"]["translation"]["k4"]["full"]["gain"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        script.retention_inputs(records)


def test_retention_inputs_reads_the_base_control_against_the_floor():
    """`seeds_clear=sum(1 for r in levels if r > BASE_R2_FLOOR)` is a STRICT
    `>`, pinned exactly at the boundary here (`distinguish=False`, so every
    cell sits on EXACTLY `BASE_R2_FLOOR` with nothing to round it either way)
    rather than +/- 0.05 away from it, where a `>=` bug would pass unnoticed.

    The holding/failing cases use the default `distinguish=True`, which gives
    every arm's base r2 a distinguishable value (`_cell_offset`) well clear of
    the boundary -- so a read that pooled one arm's base control from
    another arm's seeds, or from all nine records at once, lands on a
    different number for at least one arm instead of reproducing the same
    pass/fail by coincidence.

    Both cases also give `position_r2` the OPPOSITE pass/fail from `r2`
    (`_record` defaults `position_r2` to `r2`, which is why a gate that read
    `["position_r2"]` instead of `["r2"]` used to pass this test unnoticed --
    `retention_inputs`'s and `base_control`'s own docstrings insist the gate is
    the 4-column mean, not `position_r2`). If the read ever switches keys, the
    clears()/does-not-clear() assertions below flip and this test fails."""
    from mbfps.eval.retention import BASE_R2_FLOOR

    holding = script.retention_inputs(_records(
        base_r2=BASE_R2_FLOOR + 0.05, position_r2=BASE_R2_FLOOR - 0.05,
    ))
    assert all(c.clears() for c in holding.base.values()), (
        "the gate reads r2 (clears here); position_r2 does not clear in this "
        "fixture, so this would fail if the gate read position_r2 instead"
    )
    levels = [control.r2 for control in holding.base.values()]
    assert len(set(round(level, 6) for level in levels)) == len(levels), (
        "every arm's base r2 must be distinguishable, or a cross-arm swap "
        "would reproduce the same numbers"
    )

    # WITHIN-arm too, not only across arms. The cross-arm check above cannot
    # see a base-control loop that read records[(arm, 0)] for all three seed
    # slots: the arms still differ from each other, so the distinctness
    # assertion holds while every arm silently reports seed 0 three times. The
    # base control is a GATE, so a seed-collapse here suppresses or licenses a
    # reading without changing anything a reader would notice. Recomputed from
    # the fixture rather than hardcoded, so it tracks _cell_offset.
    raw = _records(base_r2=BASE_R2_FLOOR + 0.05)
    for arm in ("frozen_ssl", "pixel_ae", "random_vit"):
        seeds = [raw[(arm, seed)]["base_control"]["r2"] for seed in (0, 1, 2)]
        assert len(set(seeds)) == 3, f"{arm}: the fixture must vary r2 by seed"
        assert holding.base[arm].r2 == pytest.approx(sum(seeds) / 3), (
            f"{arm}: base r2 is not the mean of ITS OWN three seeds; the read "
            "collapsed the seeds or gathered them from another arm"
        )

    failing = script.retention_inputs(_records(
        base_r2=BASE_R2_FLOOR - 0.05, position_r2=BASE_R2_FLOOR + 0.05,
    ))
    assert not any(c.clears() for c in failing.base.values()), (
        "the gate reads r2 (fails here); position_r2 clears in this fixture, "
        "so this would fail if the gate read position_r2 instead"
    )

    on_the_line = script.retention_inputs(
        _records(base_r2=BASE_R2_FLOOR, distinguish=False)
    )
    assert not any(c.clears() for c in on_the_line.base.values()), (
        "a cell sitting exactly on BASE_R2_FLOOR must not clear it -- the "
        "rule is a strict '>', not '>='"
    )


def test_retention_text_contains_the_ladder_before_reading_e():
    """`retention_text`'s pure return value: Reading E's verdict and the
    ladder are both present, with the ladder printed BEFORE the verdict so a
    reader meets the evidence before the conclusion -- the order M3i's
    motion.txt uses. This does NOT check that `retention.txt` matches stdout
    byte for byte; that exact-equality guarantee is
    `test_read_reaches_every_status_end_to_end`'s job, via
    `read_bytes() == printed.encode()`."""
    from mbfps.eval.retention import reading_retention
    records = _records({("translation", 4, "two_frame")})
    inputs = script.retention_inputs(records)
    text = script.retention_text(records, inputs, reading_retention(inputs))
    assert "Reading E" in text and "verdict: MOTION DISCARDED" in text
    assert "The ladder" in text
    assert text.index("The ladder") < text.index("Reading E")


def test_retention_text_prints_position_r2_beside_the_base_control():
    """Correction 3: `base_control`'s `r2` stays the gated 4-column mean
    (position and heading mixed) and the gate stays on it -- but the read
    phase must also print the per-arm `position_r2` companion, the same
    fitted probe's r2 against absolute position alone, because
    `EXIT_BASE_UNRESOLVED`'s own rule text describes the gate as being about
    absolute position while the gated number mixes in heading."""
    from mbfps.eval.retention import reading_retention
    clearing = {("translation", 4, "stochastic")}
    records = _records(clearing)
    inputs = script.retention_inputs(records)
    text = script.retention_text(records, inputs, reading_retention(inputs))
    assert "position control" in text
    for arm in ("frozen_ssl", "pixel_ae", "random_vit"):
        assert f"{arm} position_r2=" in text
    assert "WARNING" not in text, "no cell here diverges, so no warning should print"


def test_retention_text_warns_when_r2_clears_but_position_r2_does_not():
    """The gate stays on the mixed 4-column `r2`, so a cell shaped like a
    frozen single-frame backbone that reads heading far more easily than map
    position -- clearing on `r2` while `position_r2` sits under the same
    floor -- would otherwise read as a clean pass nothing flags. `pixel_ae`
    seed 1 is built into exactly that shape: `r2` = ~0.30 (clears
    `BASE_R2_FLOOR` = 0.10) while `position_r2` is pulled to 0.05 (does not)."""
    from mbfps.eval.retention import BASE_R2_FLOOR, reading_retention
    clearing = {("translation", 4, "stochastic")}
    records = _records(clearing)
    records[("pixel_ae", 1)]["base_control"]["position_r2"] = BASE_R2_FLOOR - 0.05
    inputs = script.retention_inputs(records)
    text = script.retention_text(records, inputs, reading_retention(inputs))
    assert "WARNING" in text
    assert "pixel_ae seed 1" in text
    # No OTHER cell diverges, so the warning names this one and not another.
    warning_lines = [line for line in text.splitlines() if "WARNING" in line]
    assert len(warning_lines) == 1, warning_lines


def test_position_companion_refuses_a_record_missing_position_r2():
    """A record written before `position_r2` shipped (Correction 3) must
    raise a NAMED refusal -- naming the missing key and the cell -- rather
    than dying with a bare `KeyError` the way this function's named-refusal
    neighbours (`load_retention`, `require_one_protocol`) never do."""
    records = _records()
    del records[("pixel_ae", 1)]["base_control"]["position_r2"]
    with pytest.raises(ValueError, match=r"pixel_ae seed 1.*position_r2"):
        script._position_companion(records)


@pytest.mark.parametrize("clearing,expected", [
    ({("translation", 4, "stochastic")}, "MOTION RETAINED"),
    ({("translation", 15, "full")}, "MOTION RETAINED"),
    ({("translation", 4, "deterministic")}, "BOTTLENECK LOSS"),
    ({("translation", 4, "two_frame")}, "MOTION DISCARDED"),
    ({("rotation", 4, "two_frame")}, "TRANSLATION UNRESOLVED"),
    (set(), "UNRESOLVED MOTION"),
])
def test_read_reaches_every_status_end_to_end(tmp_path, capsys, clearing, expected):
    """Every branch reachable, driven through the real `read` phase to a printed
    verdict. M3i's spec promised a MOTION_ENCODED path and nothing tested it;
    the final whole-branch review found it, and a fixture per status is the
    cheap version of that check.

    UNRESOLVED_MOTION is the one row here that does NOT exit 0: it is one of
    the two statuses `READ_EXITS` maps to a refusal code (40, see
    `test_read_exits_40_when_nothing_reads_either_target`), because "no rung
    read either target at any horizon" is an empty measurement rather than a
    finding a lever gets chosen from. Every other status here is a finding
    and exits 0 -- a milestone that exited non-zero on a finding would make
    "the run worked" and "the news was good" the same signal.
    """
    for (arm, seed), record in _records(clearing).items():
        path = script.retention_record_path(tmp_path, arm, seed)
        path.write_text(json.dumps(record))
    status = script.main(["--phase", "read", "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert f"verdict: {expected}" in printed
    # EXACT byte equality, not containment: `print(text)` (an extra trailing
    # newline) or `write_text(text.rstrip())` would both still satisfy a
    # containment check, and the artefact and the log would quietly disagree
    # about what the run said.
    assert (tmp_path / "retention.txt").read_bytes() == printed.encode(), (
        "retention.txt and stdout must be byte-identical"
    )
    if expected == "UNRESOLVED MOTION":
        assert status == script.EXIT_MOTION_UNRESOLVED
    else:
        assert status == script.EXIT_OK


def test_read_exits_39_when_the_base_control_fails(tmp_path, capsys):
    """A broken instrument is a refusal with its own exit code, not a status
    printed beside a reading."""
    from mbfps.eval.retention import BASE_R2_FLOOR
    for (arm, seed), record in _records(base_r2=BASE_R2_FLOOR - 0.05).items():
        script.retention_record_path(tmp_path, arm, seed).write_text(json.dumps(record))
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == 39
    assert "UNRESOLVED BASE" in capsys.readouterr().out


def test_read_exits_40_when_nothing_reads_either_target(tmp_path):
    for (arm, seed), record in _records().items():
        script.retention_record_path(tmp_path, arm, seed).write_text(json.dumps(record))
    assert script.main(["--phase", "read", "--out", str(tmp_path)]) == 40
