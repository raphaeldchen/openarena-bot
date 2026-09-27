"""scripts/latent_motion.py: the displacement probe, its permuted control and
the per-cell record (Task 6), then the reading (Task 7).

The script is loaded by path, the way `tests/eval/test_sharper_latent_script.py`
loads its own. The REFERENCE is a real tiny cell on `wide_buffer` -- thirty
40-step episodes, `run_job(..., steps=5, seq_len=4, context=5, horizon=15,
device="cpu")` -- with the diagnostic `diagnose_dynamics.main` writes, exactly
as the sharper-latent, ladder and stage tests build theirs.

THE FIXTURE'S FACTS: six validation episodes; `window_starts(40, 5, 15)` cuts
TWO windows per episode (`range(0, 40 - 20 + 1, 20)`), so 12 val windows over 6
clusters, and each window is 20 rows long. The probe anchors at the rollout's
own t0 -- window row `context - 1` -- so `displacement` is handed the
`horizon + 1 = 16` rows from t0 onward and a k fits exactly when `k <= horizon`.
`K_REPORTED` reaches 45, which this fixture's horizon of 15 cannot carry, so the
measure phase takes `ks` as a parameter with `K_REPORTED` as its default -- a
test never patches the pre-registration -- and the fixture runs `(1, 5, 15)`,
which includes the horizon M3i decides at and sits exactly on the boundary the
shipped protocol's k=45 sits on.
"""

import importlib.util
import inspect
import types
from pathlib import Path

import numpy as np
import pytest

import mbfps.eval.pooling as pooling
from mbfps.eval.motion import K_REPORTED, contrast_series, displacement
from mbfps.eval.probe import fit_probes
from mbfps.eval.study import StudyJob, load_record, run_job, write_record

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script("latent_motion")
diagnose = _load_script("diagnose_dynamics")
trust = _load_script("trust_horizon")

JOB = StudyJob("random_vit", 1)
JOB_KW = dict(steps=5, seq_len=4, context=5, horizon=15, device="cpu")
CONTEXT, HORIZON = JOB_KW["context"], JOB_KW["horizon"]
KS = (1, 5, 15)
WINDOWS, CLUSTERS = 12, 6
MOTION = "motion_random_vit_seed1.json"

MOTION_KEYS = {
    "arm", "seed", "source", "step", "record_git_sha", "context", "horizon", "ks",
    "split_seed", "device", "torch_version", "git_sha", "episodes", "windows",
    "self_check", "k", "description", "information", "nonfinite",
}
K_ENTRY_KEYS = {"k", "contrast", "control", "ridge", "train_windows"}


@pytest.fixture
def reference(tmp_path, curved_buffer, capsys):
    """One real cell and the diagnostic the ladder writes for it.

    `curved_buffer`, never `wide_buffer`: on the straight path every window's
    `p(t + k) - p(t)` is the same vector, so the permuted control would be
    bitwise the treatment and no test of it could fail.
    """
    out = tmp_path / "reference"
    run_job(JOB, curved_buffer, out, **JOB_KW)
    status = diagnose.main([
        "--out", str(out), "--data", str(curved_buffer.root), "--device", "cpu",
        "--arms", JOB.arm, "--seeds", str(JOB.seed),
        "--context", str(CONTEXT), "--horizon", str(HORIZON), "--ks", "1", str(HORIZON),
    ])
    assert status == diagnose.EXIT_OK, "the fixture's diagnostic was not written cleanly"
    capsys.readouterr()
    return types.SimpleNamespace(
        out=out, data=curved_buffer.root, motion=tmp_path / "motion",
    )


def _argv(ref, *extra: str) -> list[str]:
    return [
        "--out", str(ref.motion), "--reference", str(ref.out), "--data", str(ref.data),
        "--device", "cpu", "--arms", JOB.arm, "--seeds", str(JOB.seed),
        *extra,
    ]


def _run(ref, *extra: str) -> int:
    return script.main(_argv(ref, *extra), ks=KS)


@pytest.fixture
def measured(reference, capsys):
    assert _run(reference, "--phase", "measure") == script.EXIT_OK
    capsys.readouterr()
    return reference


def _never_pass(*args, **kwargs):
    raise AssertionError("a pass ran; this refusal must come before any pass")


# ---------------------------------------------------------------------------
# The control permutation.
# ---------------------------------------------------------------------------


def _plain_leaks(n: int, seed: int) -> bool:
    """Would a plain `rng.permutation(n)` at this seed keep a row on itself?"""
    return bool((np.random.default_rng(seed).permutation(n) == np.arange(n)).any())


def test_the_control_permutation_pairs_no_window_with_its_own_displacement():
    """The control must destroy the latent->displacement pairing. A
    permutation that leaves any window paired with its own row is a control
    that still carries that window's signal.

    THIS CASE ALONE CANNOT CATCH A PLAIN SHUFFLE, and that is worth saying
    where the reader is: `default_rng(0).permutation(200)` happens to have no
    fixed point, so deleting the derangement entirely leaves this assertion
    green. It is kept because it pins the draw the real control actually
    takes; the test below is the one that fails when the derangement goes.
    """
    order = script.permute_pairing(200, seed=script.CONTROL_SEED)
    assert sorted(order.tolist()) == list(range(200)), "must be a permutation, not a resample"
    assert not (order == np.arange(200)).any(), "no window may keep its own displacement"


def test_the_derangement_holds_on_every_case_a_plain_shuffle_leaks():
    """The mutation-sensitive half. Every (n, seed) below is one where a plain
    `rng.permutation` DOES keep a row on itself, so a control that skipped the
    derangement would carry that window's own displacement -- real signal in a
    control whose whole claim is that it has none."""
    sizes = (2, 3, 4, 5, 8, 13, 64, 200)
    leaky = [(n, s) for n in sizes for s in range(64) if _plain_leaks(n, s)]
    assert len(leaky) > 100, (
        f"only {len(leaky)} cases where a plain shuffle leaks; this test would prove little"
    )
    for n, seed in leaky:
        order = script.permute_pairing(n, seed=seed)
        assert sorted(order.tolist()) == list(range(n)), f"n={n} seed={seed} is not a permutation"
        assert not (order == np.arange(n)).any(), f"n={n} seed={seed} kept a window's own row"


def test_the_derangement_holds_where_the_briefs_neighbour_swap_did_not():
    """Hand-typed regressions on the repair this function does NOT use: the
    brief swapped each fixed point with its right-hand neighbour and wrapped
    the last onto index 0, which returns the identity to itself at n = 2 and
    leaves `[0, 2, 1]` at n = 3. Both are deranged here."""
    for n in (2, 3):
        for seed in range(64):
            order = script.permute_pairing(n, seed=seed)
            assert not (order == np.arange(n)).any(), f"n={n} seed={seed} kept a fixed point"
    assert script.permute_pairing(2, seed=0).tolist() == [1, 0], "the only derangement of 2"


def test_the_control_permutation_is_reproducible():
    """A control that draws a fresh permutation each run is a control whose
    refusal cannot be repeated."""
    a = script.permute_pairing(64, seed=script.CONTROL_SEED)
    b = script.permute_pairing(64, seed=script.CONTROL_SEED)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, script.permute_pairing(64, seed=script.CONTROL_SEED + 1))


def test_the_permutation_refuses_a_window_count_it_cannot_derange():
    """A single window cannot be paired with anything but itself."""
    with pytest.raises(ValueError, match="at least 2"):
        script.permute_pairing(1, seed=0)


# ---------------------------------------------------------------------------
# The probe, bounded from both sides.
# ---------------------------------------------------------------------------


def _exact_case():
    """Latents that encode the displacement EXACTLY: 64 rows of 8 features,
    the first two scaled by 3. 48 rows fit the weights, 16 select the ridge."""
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(64, 8))
    return latents, latents[:, :2] * 3.0


def test_a_probe_fit_on_exact_displacement_recovers_it():
    """Known answer: latents that linearly encode the displacement must be
    read back at essentially zero error, so a positive contrast is possible at
    all. Without this the reading could only ever fail.

    Hand-typed: the selected ridge is 1e-1, `RIDGES`' smallest, and it reads
    this case back to max |error| 0.0169 against a signal of magnitude ~3 --
    a mean contrast of +3.27 where staying put scores 0.
    """
    latents, true = _exact_case()
    probe = script.fit_displacement_probe(
        {"latent": latents[:48], "displacement": true[:48]}, k=1,
        select={"latent": latents[48:], "displacement": true[48:]},
    )
    predicted = script.apply_displacement_probe(probe, latents)
    assert probe["ridge"] == 0.1
    assert np.abs(predicted - true).max() < 0.02
    assert contrast_series(predicted, true).mean() > 3.0


def test_the_unselected_ridge_recovers_nothing_which_is_why_select_is_passed():
    """`fit_probe`'s no-selection fallback is `ridge=1e3`, chosen for POSITION
    targets whose std is ~240 map units. On the same known answer above it
    reads the displacement back at max |error| 5.87 against a signal of
    magnitude ~3 -- it recovers nothing. Taking that fallback would underfit
    every displacement probe M3i fits and reach NO_MOTION through the probe
    rather than through the latent, so `measure_cell` always passes `select`
    and this pins what the difference is worth."""
    latents, true = _exact_case()
    fallback = script.fit_displacement_probe(
        {"latent": latents[:48], "displacement": true[:48]}, k=1,
    )
    assert fallback["ridge"] == 1e3
    assert np.abs(script.apply_displacement_probe(fallback, latents) - true).max() > 5.0


def test_a_probe_fit_on_noise_does_not_beat_staying_put():
    """The other known answer. Latents independent of the displacement must
    not produce a positive contrast on held-out rows. Hand-typed: the mean
    contrast on the held-out half is -0.0057."""
    rng = np.random.default_rng(1)
    latents = rng.normal(size=(256, 8))
    true = rng.normal(size=(256, 2)) * 10.0
    probe = script.fit_displacement_probe(
        {"latent": latents[:96], "displacement": true[:96]}, k=1,
        select={"latent": latents[96:128], "displacement": true[96:128]},
    )
    held_out = script.apply_displacement_probe(probe, latents[128:])
    assert contrast_series(held_out, true[128:]).mean() <= 0.0


def test_the_selection_share_is_the_position_probes_own():
    """Both probes are selected on the same share of the same split, or the
    displacement probe is fit under a rule nobody wrote down."""
    assert script.SELECT_EPISODES == (
        inspect.signature(fit_probes).parameters["select_episodes"].default
    )


def test_the_probe_records_the_k_it_was_fit_at():
    """A probe fit at one horizon and applied at another would be read as this
    horizon's number; the record has to say which."""
    rng = np.random.default_rng(2)
    latents = rng.normal(size=(32, 4))
    probe = script.fit_displacement_probe(
        {"latent": latents, "displacement": latents[:, :2]}, k=30
    )
    assert probe["k"] == 30


# ---------------------------------------------------------------------------
# Reshaping the rows -- the alignment this whole measurement rests on.
# ---------------------------------------------------------------------------


def test_window_rows_reshapes_by_window_and_step_in_window_order():
    """Hand-typed: two windows of three steps, handed in row order. Row `i` of
    the result must be window `i`'s steps in step order."""
    data = {
        "window": np.array([0, 0, 0, 1, 1, 1]),
        "step": np.array([0, 1, 2, 0, 1, 2]),
        "latent": np.arange(12, dtype=float).reshape(6, 2),
        "targets": np.arange(24, dtype=float).reshape(6, 4),
    }
    rows = script.window_rows(data)
    assert rows["latent"].shape == (2, 3, 2)
    assert rows["positions"].shape == (2, 3, 2)
    np.testing.assert_array_equal(rows["latent"][0], [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(rows["latent"][1], [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]])
    # positions are targets[..., :2] -- pos_x, pos_y, never sin/cos.
    np.testing.assert_array_equal(rows["positions"][0], [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]])


def test_window_rows_refuses_windows_of_unequal_length():
    """A ragged gather would reshape into a silently wrong (n, steps) grid."""
    data = {
        "window": np.array([0, 0, 1]),
        "step": np.array([0, 1, 0]),
        "latent": np.zeros((3, 2)),
        "targets": np.zeros((3, 4)),
    }
    with pytest.raises(ValueError, match="same number of rows"):
        script.window_rows(data)


def test_window_rows_anchors_the_probe_at_the_rollouts_own_t0():
    """`anchor_at_t0` reads window row `context - 1`, never row 0.

    `gather_probe_data`'s row `j` is frame `start + 1 + j`, so row 0 is the
    posterior after ONE real frame out of a zero RSSM state -- a latent that
    cannot encode a two-frame quantity even in principle, whose only route to
    a positive contrast is a correlation between absolute position and
    displacement. Row `context - 1` is frame `start + context`: the rollout's
    own t0, the frame the persistence baseline freezes at.

    Hand-typed on four rows at context 3: the anchor is row 2, and the
    positions handed to `displacement` start there, so there are
    `horizon + 1 = 2` of them.
    """
    rows = {
        "latent": np.arange(4 * 2, dtype=float).reshape(1, 4, 2),
        "positions": np.arange(4 * 2, dtype=float).reshape(1, 4, 2) + 100.0,
    }
    latent, positions = script.anchor_at_t0(rows, context=3)
    np.testing.assert_array_equal(latent, [[4.0, 5.0]])
    np.testing.assert_array_equal(positions, [[[104.0, 105.0], [106.0, 107.0]]])
    # and the displacement is measured from t0, not from the window's first row
    np.testing.assert_array_equal(displacement(positions, 1), [[2.0, 2.0]])
    # a context of zero has no t0 to anchor at; `row = -1` would silently take
    # the window's LAST step, which is a different horizon entirely.
    with pytest.raises(ValueError, match="at least 1 real frame"):
        script.anchor_at_t0(rows, context=0)


def _aligned(n: int = 3, horizon: int = 2, context: int = 1):
    """Three windows whose gathered rows ARE the pass's own: row `context - 1`
    is the frame at t0, and the rows after it are the horizon's truth."""
    at_context = np.arange(n * 2, dtype=float).reshape(n, 2)
    true_positions = np.arange(n * horizon * 2, dtype=float).reshape(n, horizon, 2) + 100.0
    positions = np.concatenate([at_context[:, None, :], true_positions], axis=1)
    return positions, at_context, true_positions, context


def test_aligned_val_windows_pass_the_check():
    """The positive half: rows that ARE the pass's own must not be refused, or
    the guard below could not distinguish anything."""
    positions, at_context, true_positions, context = _aligned()
    script.require_aligned_windows(
        positions, at_context, true_positions, context=context, total=3, arm="a", seed=0,
    )


def test_a_val_gather_of_a_different_window_COUNT_is_refused_naming_both():
    """THE hazard, first half. `cell_series` pairs every per-window series
    positionally with the record's `windows.episode`; a different count is a
    different protocol, never something to truncate or pad to fit."""
    positions, at_context, true_positions, context = _aligned()
    with pytest.raises(ValueError, match=r"3 val windows.*windows\.total is 4"):
        script.require_aligned_windows(
            positions, at_context, true_positions, context=context, total=4, arm="a", seed=0,
        )


def test_a_val_gather_in_a_different_ORDER_is_refused():
    """THE hazard, second half, and the one nothing downstream would notice:
    the same count in another order pairs every contrast with another window's
    displacement while every shape still agrees."""
    positions, at_context, true_positions, context = _aligned()
    with pytest.raises(ValueError, match="not the windows the record was scored on"):
        script.require_aligned_windows(
            positions[::-1].copy(), at_context, true_positions,
            context=context, total=3, arm="a", seed=0,
        )


# ---------------------------------------------------------------------------
# One reported horizon: the values, and which way round they are recorded.
# ---------------------------------------------------------------------------


def _anchored_rows(n: int, steps: int, context: int, seed: int) -> dict:
    """`window_rows`' output, built so ONLY the latent at the anchor can
    predict the displacement.

    Each window travels in a straight line whose velocity vector is that
    window's own latent at row `context - 1`, scaled; every other row of the
    latent is independent noise, and each window's starting point is its own.
    So a probe that reads the anchor recovers the displacement nearly exactly,
    a probe that reads any other row recovers nothing, and pairing a
    prediction with another window's displacement is plainly worse than
    pairing it with its own -- which is what makes the three numbers below
    move when the read-out point, the sign or the pairing does.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, steps, 6))
    velocity = latent[:, context - 1, :2] * 5.0
    offset = (np.arange(steps, dtype=float) - (context - 1))[None, :, None]
    positions = rng.normal(size=(n, 1, 2)) * 50.0 + offset * velocity[:, None, :]
    return {"latent": latent, "positions": positions, "windows": n, "steps": steps}


def test_the_k_entry_records_the_real_pairing_as_contrast_and_the_permuted_one_as_control():
    """The record's per-window series, pinned BY VALUE and BY ORIENTATION.

    Three separate mutations to this function left every other test in this
    file green, because the only record-level assertions were shapes and
    `contrast != control`: negating the recorded contrast, SWAPPING the
    `contrast` and `control` keys -- which would hand the next task the
    permuted pairing as the result and the real pairing as the control -- and
    reading the validation latents at a row other than the one the probe was
    fit at.

    Hand-typed from the fixture above: window 0's contrast is +19.0998, the
    probe reading its displacement back almost exactly; its control is
    -12.3473, the same prediction read against another window's displacement;
    and anchoring at row 0 instead of `context - 1` would make the first
    contrast -1.82. No two of those can be confused.
    """
    context, steps, k, n_val = 5, 20, 3, 9
    train = _anchored_rows(40, steps, context, 0)
    select = _anchored_rows(12, steps, context, 1)
    val = _anchored_rows(n_val, steps, context, 2)
    entry = script._k_entry(train, select, val, k, context)

    assert entry["k"] == k and entry["train_windows"] == 40
    assert entry["contrast"][0] == pytest.approx(19.09979931796889, rel=1e-6)
    assert entry["control"][0] == pytest.approx(-12.347330772303156, rel=1e-6)

    # And the whole control series is the treatment's own predictions read
    # against `permute_pairing(n, CONTROL_SEED)`'s displacements -- nothing
    # refit, nothing resampled, only the pairing destroyed.
    row = context - 1
    probe = script.fit_displacement_probe(
        {"latent": train["latent"][:, row, :],
         "displacement": displacement(train["positions"][:, row:], k)},
        k,
        select={"latent": select["latent"][:, row, :],
                "displacement": displacement(select["positions"][:, row:], k)},
    )
    predicted = script.apply_displacement_probe(probe, val["latent"][:, row, :])
    true = displacement(val["positions"][:, row:], k)
    order = script.permute_pairing(n_val, script.CONTROL_SEED)
    np.testing.assert_allclose(entry["contrast"], contrast_series(predicted, true), rtol=1e-9)
    np.testing.assert_allclose(
        entry["control"], contrast_series(predicted, true[order]), rtol=1e-9
    )


def test_the_k_grid_fits_exactly_when_k_is_at_most_the_horizon():
    """The anchor is t0, so `displacement` is handed `horizon + 1` rows and
    not `context + horizon`: the grid fits exactly when `k <= horizon`.

    Pinned on the SHIPPED protocol -- context 5, horizon 45 -- where
    `K_REPORTED`'s largest k is 45 and consumes all 46 rows with none to
    spare, and pinned at the boundary in both directions. The `context=1`
    case is the one that fails if the old `context + horizon` arithmetic ever
    comes back: the context steps sit BEFORE the anchor and are gone.
    """
    assert max(K_REPORTED) == 45, "a k above the shipped horizon could not be reported"
    script.require_reportable_ks(K_REPORTED, context=5, horizon=45, arm="a", seed=0)
    script.require_reportable_ks((45,), context=1, horizon=45, arm="a", seed=0)
    with pytest.raises(ValueError, match=r"k=46 needs 47 rows from the rollout's t0"):
        script.require_reportable_ks((46,), context=5, horizon=45, arm="a", seed=0)


# ---------------------------------------------------------------------------
# measure, end to end on a real cell.
# ---------------------------------------------------------------------------


def test_a_missing_cell_is_exit_11_before_any_pass_runs(reference, monkeypatch, capsys):
    monkeypatch.setattr(script, "reference_trajectories", _never_pass)
    assert _run(reference, "--phase", "measure", "--seeds", "0") == script.EXIT_NO_CHECKPOINTS
    assert "NO CELL" in capsys.readouterr().out
    assert not reference.motion.exists()


def test_measure_writes_one_record_per_cell_with_one_row_per_val_window(measured):
    record = load_record(measured.motion / MOTION)
    assert set(record) == MOTION_KEYS
    assert record["arm"] == JOB.arm and record["seed"] == JOB.seed
    assert record["context"] == CONTEXT and record["horizon"] == HORIZON
    assert record["ks"] == list(KS)
    assert record["step"] == JOB_KW["steps"] and record["source"] == str(measured.out)
    assert record["windows"]["total"] == WINDOWS
    assert len(record["windows"]["episode"]) == WINDOWS
    assert len(set(record["windows"]["episode"])) == CLUSTERS
    assert len(record["episodes"]["val"]) == CLUSTERS
    assert record["self_check"]["ok"] is True
    assert set(record["k"]) == {f"k{k}" for k in KS}
    for k in KS:
        entry = record["k"][f"k{k}"]
        assert set(entry) == K_ENTRY_KEYS
        assert entry["k"] == k
        # One row per val window, in the record's own window order -- Task 7
        # pairs these positionally with `windows.episode`.
        assert len(entry["contrast"]) == WINDOWS
        assert len(entry["control"]) == WINDOWS
        assert entry["contrast"] != entry["control"], "the control is not the treatment"


def test_the_record_is_what_cell_series_can_cluster(measured):
    """The pooling reads these six fields off the record; a record missing one,
    or carrying a series of another length, is refused at the pool."""
    record = load_record(measured.motion / MOTION)
    values = np.asarray(record["k"]["k15"]["contrast"], dtype=float)
    series = trust.cell_series(
        JOB.arm, JOB.seed, "displacement", values, np.ones(values.size, dtype=bool), record,
    )
    assert series.windows_total == WINDOWS
    assert series.episode.tolist() == record["windows"]["episode"]
    assert series.horizon == HORIZON and series.context == CONTEXT
    assert series.val == tuple(record["episodes"]["val"])
    assert np.unique(series.episode).size == CLUSTERS


def _reverse_window_blocks(data: dict) -> dict:
    """The same gather with its WINDOWS in reverse order and the `window` /
    `step` index left ascending -- the exact shape of corruption that survives
    every downstream shape check, because the counts and the dtypes are all
    still right and only the pairing is wrong."""
    steps = int(np.asarray(data["step"]).max()) + 1
    n = int(np.asarray(data["window"]).size) // steps
    index = np.concatenate(
        [np.arange(w * steps, (w + 1) * steps) for w in reversed(range(n))]
    )
    out = dict(data)
    for key in ("latent", "embedding", "encoder_embedding", "targets"):
        out[key] = np.asarray(data[key])[index]
    return out


def test_measure_refuses_a_val_gather_whose_windows_are_in_another_order(
    reference, monkeypatch,
):
    """The alignment guard's WIRING, which its own unit tests cannot cover.

    Replacing the `require_aligned_windows(...)` call in `measure_cell` with
    `pass` left every other test in this file green. A count mismatch would
    still be caught downstream by `cell_series`' shape check; an ORDER
    mismatch has no other backstop at all, and every per-window series would
    be clustered onto the wrong episodes with nothing raising.

    So the val gather -- the third of the three -- comes back with its window
    blocks reversed while `window` and `step` stay ascending, and
    `measure_cell` must refuse it and write nothing.
    """
    real = script.gather_probe_data
    calls = []

    def spy(model, paths, backbone, device, **kwargs):
        data = real(model, paths, backbone, device, **kwargs)
        calls.append(1)
        return _reverse_window_blocks(data) if len(calls) == 3 else data

    monkeypatch.setattr(script, "gather_probe_data", spy)
    with pytest.raises(ValueError, match="not the windows the record was scored on"):
        _run(reference, "--phase", "measure")
    assert len(calls) == 3, "the fit, selection and val gathers, in that order"
    assert not (reference.motion / MOTION).exists(), "no record for a refused gather"


def test_the_val_gather_is_not_capped_at_the_probes_twenty_episodes(measured, monkeypatch):
    """`gather_probe_data`'s default `limit=20` is the PROBE's fit rule. The
    shipped val split is 24 episodes, so scoring it at that default would drop
    four episodes -- 189 of the record's 229 windows -- and the fixture's six
    val episodes would never reveal it.

    The three blocks are pinned BY NAME as well as by count, because the
    counts alone cannot see a validation leak: taking the ridge-selection
    episodes straight out of the val split -- selecting the probe on the exact
    windows its contrast is then read from, which would inflate every contrast
    and manufacture MOTION_ENCODED -- leaves `n_fit + n_select == 20` and
    `n_select == SELECT_EPISODES` both true.
    """
    calls = []
    real = script.gather_probe_data

    def spy(model, paths, backbone, device, **kwargs):
        paths = list(paths)
        calls.append(({Path(p).name for p in paths}, kwargs.get("limit")))
        return real(model, paths, backbone, device, **kwargs)

    monkeypatch.setattr(script, "gather_probe_data", spy)
    assert _run(measured, "--phase", "measure") == script.EXIT_OK
    assert len(calls) == 3, "fit block, ridge-selection block, val"
    (fit, _), (select, _), (val, val_limit) = calls
    n_fit, n_select, n_val = len(fit), len(select), len(val)
    assert n_fit + n_select == 20, "PROBE_EPISODE_LIMIT training episodes, split for selection"
    assert n_select == script.SELECT_EPISODES
    assert fit.isdisjoint(val), f"the probe is FIT on episodes it is scored on: {fit & val}"
    assert select.isdisjoint(val), (
        f"the ridge is SELECTED on episodes the probe is scored on: {select & val}"
    )
    assert fit.isdisjoint(select), "the selection block is not held out of the fit"
    assert val_limit is not None and val_limit >= n_val, (
        f"the val gather asked for limit={val_limit} over {n_val} episodes; every val "
        "window the record was scored on must be gathered"
    )


def test_the_description_is_the_posterior_against_the_teacher_forced_prior(measured):
    """Spec 2.3, reported and deciding nothing. Entropy against its own log-C
    ceiling, live groups against the group count, top-1 mass for both."""
    record = load_record(measured.motion / MOTION)
    description = record["description"]
    assert set(description) == {
        "entropy_by_group", "entropy_mean", "entropy_max",
        "live_groups", "top1_posterior", "top1_prior",
    }
    groups = len(description["entropy_by_group"])
    assert 0.0 <= description["entropy_mean"] <= description["entropy_max"]
    assert 0.0 <= description["live_groups"] <= groups
    for key in ("top1_posterior", "top1_prior"):
        assert 0.0 < description[key] <= 1.0
    assert description["top1_posterior"] != description["top1_prior"], (
        "the prior must be read separately from the posterior"
    )
    assert record["information"]["mean"] >= 0.0, "a KL is never negative"


def test_a_k_longer_than_the_window_is_refused_before_anything_is_gathered(
    reference, monkeypatch,
):
    """`K_REPORTED` reaches 45; this fixture's horizon of 15 leaves 16 rows
    from t0 and cannot carry it. Scoring a short window at a smaller k would
    pool a different horizon as if it were this one, so the grid is refused
    rather than truncated.

    The match is on `require_reportable_ks`' OWN sentence, not on the shared
    prefix: `displacement` raises `k=45 needs 46 steps per window, got 16`
    too, so a looser pattern would pass with this guard deleted -- after a
    full rollout and three gathers per cell had already run. `prepare_cell` is
    replaced by a raiser to prove the refusal comes first.
    """
    monkeypatch.setattr(script, "prepare_cell", _never_pass)
    with pytest.raises(
        ValueError, match=r"k=45 needs 46 rows from the rollout's t0, but this protocol"
    ):
        script.main(_argv(reference, "--phase", "measure"), ks=(1, 45))
    assert not reference.motion.exists(), "nothing is written for a grid that was refused"


def test_the_pre_registered_grid_is_the_default(reference, monkeypatch):
    """A test that passes a smaller grid must not be able to move the run's."""
    assert script.K_REPORTED == K_REPORTED == (1, 5, 15, 30, 45)
    seen = {}

    def spy(args, cells, device, train, val, ks):
        seen["ks"] = tuple(ks)
        return script.EXIT_OK

    monkeypatch.setattr(script, "measure_phase", spy)
    assert script.main(_argv(reference, "--phase", "measure")) == script.EXIT_OK
    assert seen["ks"] == K_REPORTED


# ---------------------------------------------------------------------------
# The statuses.
# ---------------------------------------------------------------------------


def test_the_reused_statuses_are_trust_horizons_and_38_is_new():
    assert script.EXIT_OK == trust.EXIT_OK == 0
    assert script.EXIT_NO_CHECKPOINTS == trust.EXIT_NO_CHECKPOINTS == 11
    assert script.EXIT_SPLIT_MISMATCH == trust.EXIT_SPLIT_MISMATCH == 12
    assert script.EXIT_RECORD_MISMATCH == trust.EXIT_RECORD_MISMATCH == 14
    assert script.EXIT_SELF_CHECK_FAILED == trust.EXIT_SELF_CHECK_FAILED == 30
    assert script.EXIT_CONTROL_LEAKED == 38


def test_the_parser_defaults_are_the_specs():
    args = script._parser().parse_args([])
    assert args.out == Path("runs/m3i_motion")
    assert args.reference == Path("runs/m3_study_v2") and args.data == Path("data/my_way_home")
    assert args.device == "mps" and args.context is None and args.horizon is None
    assert args.arms == ["pixel_ae", "frozen_ssl", "random_vit"] and args.seeds == [0, 1, 2]
    assert args.phase == script.PHASES[-1]


def test_the_record_path_names_both_the_arm_and_the_seed(tmp_path):
    """A name carrying only the arm would let seed 2 overwrite seed 0 -- see
    `study.job_record_path` for what that costs."""
    paths = {
        script.motion_record_path(tmp_path, arm, seed)
        for arm in script.ARMS for seed in script.SEEDS
    }
    assert len(paths) == len(script.ARMS) * len(script.SEEDS)
    assert script.motion_record_path(tmp_path, "pixel_ae", 0).name == "motion_pixel_ae_seed0.json"


# ---------------------------------------------------------------------------
# read: pooling the records into Reading D (Task 7).
# ---------------------------------------------------------------------------
#
# THREE OF THE BRIEF'S READ-PHASE ASSERTIONS DO NOT DESCRIBE THIS FIXTURE, and
# are written here against what it actually holds:
#
#   * the records are under `--out` (`ref.motion`), never under `ref.out`,
#     which is the M3c STUDY directory the checkpoints are read from. The
#     brief's `measured.out` would look for `motion_*.json` beside the
#     checkpoints and find none.
#   * the fixture measures ONE cell (`--arms random_vit --seeds 1`), so
#     `load_motion(..., ARMS, SEEDS)` would refuse eight cells that were never
#     measured, and `set(inputs.arms) == set(ARMS)` cannot hold.
#   * the fixture's grid is `KS = (1, 5, 15)`, the largest its horizon of 15
#     can carry, so k = 30 and k = 45 are not in these records and no honest
#     table can print them. The "reported, deciding nothing" horizons here are
#     1 and 5.
#
# And the brief's orientation assertion -- `arms[a].z != control[a].z` -- is
# kept but is NOT the test of the orientation: swapping `arms=` and `control=`
# leaves two unequal numbers unequal. The orientation is pinned BY VALUE below.


def _series(record, key, k=15):
    return np.asarray(record["k"][f"k{k}"][key], dtype=float)


def _rewrite(path, record):
    """A doctored record back to disk. `load_record` restores the `nonfinite`
    map it read, and `write_record` refuses a record that already carries one
    under the name it writes it under -- so it is dropped here and rebuilt."""
    record.pop("nonfinite", None)
    write_record(path, record)


def _pooled_z(record, key, arm=JOB.arm, seed=JOB.seed, k=15):
    """The pooled z of one of the record's own per-window series, recomputed
    in the test by the route the script is supposed to take: `cell_series`
    onto `windows.episode`, then `pool_arm`. Independent of which slot the
    script put it in, which is what makes it a check of the orientation."""
    values = _series(record, key, k)
    cell = trust.cell_series(
        arm, seed, key, values, np.ones(values.size, dtype=bool), record,
    )
    return pooling.pool_arm([cell]).z


def test_the_reading_is_built_from_the_records_treatment_and_control(measured):
    """The orientation, pinned by value. A swapped treatment and control would
    invert the verdict, and no shape check would notice -- nor would an
    inequality between the two, since both stay unequal after a swap."""
    records = script.load_motion(measured.motion, [JOB.arm], [JOB.seed])
    inputs = script.motion_inputs(records, k=15)
    assert set(inputs.arms) == {JOB.arm}
    assert set(inputs.control) == {JOB.arm}
    assert inputs.k == 15
    assert inputs.clusters == CLUSTERS
    assert inputs.z_fam == pytest.approx(script.motion_threshold(inputs.clusters))

    record = records[(JOB.arm, JOB.seed)]
    # One seed and every window kept, so the pooled estimate IS the mean of
    # the record's own series -- the plainest statement of which series went
    # into which slot.
    assert inputs.arms[JOB.arm].estimate == pytest.approx(_series(record, "contrast").mean())
    assert inputs.control[JOB.arm].estimate == pytest.approx(_series(record, "control").mean())
    assert inputs.arms[JOB.arm].z == pytest.approx(_pooled_z(record, "contrast"))
    assert inputs.control[JOB.arm].z == pytest.approx(_pooled_z(record, "control"))
    # The control is the PERMUTED series, so it must differ from the treatment
    # (it is bitwise equal on a straight path -- hence `curved_buffer`).
    for arm in inputs.arms:
        assert inputs.arms[arm].z != inputs.control[arm].z


def test_the_treatment_and_the_control_are_pooled_by_the_same_route(measured):
    """A control computed differently from the treatment is not a control: it
    is a differently-computed number that happens to be called one. So the two
    slots must agree with the SAME recomputation, differing only in which of
    the record's series went in."""
    records = script.load_motion(measured.motion, [JOB.arm], [JOB.seed])
    inputs = script.motion_inputs(records, k=15)
    record = records[(JOB.arm, JOB.seed)]
    for slot, key in ((inputs.arms, "contrast"), (inputs.control, "control")):
        values = _series(record, key)
        cell = trust.cell_series(
            JOB.arm, JOB.seed, key, values, np.ones(values.size, dtype=bool), record,
        )
        pooled = pooling.pool_arm([cell])
        assert slot[JOB.arm].estimate == pytest.approx(pooled.mean)
        assert slot[JOB.arm].se == pytest.approx(pooled.se)
        assert slot[JOB.arm].z == pytest.approx(pooled.z)
        assert slot[JOB.arm].seeds_total == 1


def test_the_seed_tallies_are_counted_per_seed_and_not_off_the_pooled_z(measured):
    """`seeds_up` / `seeds_down` are the arm's cells whose OWN per-seed z
    clears the bar. Read off the pooled z instead, every arm would read 0/n or
    n/n and `MotionArm.clears_up`'s replication clause would stop binding.

    One cell here, so the per-seed z IS the pooled one -- which is exactly why
    the tally is asserted against the per-seed rule and against the bar, not
    against the pooled number it happens to equal.
    """
    records = script.load_motion(measured.motion, [JOB.arm], [JOB.seed])
    inputs = script.motion_inputs(records, k=15)
    arm = inputs.arms[JOB.arm]
    own = _pooled_z(records[(JOB.arm, JOB.seed)], "contrast")
    assert arm.seeds_total == 1
    assert arm.seeds_up == int(own >= inputs.z_fam)
    assert arm.seeds_down == int(own <= -inputs.z_fam)
    assert arm.seeds_up + arm.seeds_down <= 1


def test_a_leaking_control_is_exit_38_and_writes_no_reading(measured, capsys):
    """The gate, in code. Doctoring the control to clear the bar must refuse
    before any verdict is printed."""
    path = script.motion_record_path(measured.motion, JOB.arm, JOB.seed)
    record = load_record(path)
    entry = record["k"]["k15"]
    entry["control"] = [x + 500.0 for x in entry["contrast"]]
    _rewrite(path, record)
    assert script.main(_argv(measured, "--phase", "read")) == script.EXIT_CONTROL_LEAKED
    out = capsys.readouterr().out
    assert "UNRESOLVED CONTROL" in out
    assert "MOTION ENCODED" not in out and "NO MOTION" not in out
    assert not (measured.motion / "motion.txt").exists()


def test_read_writes_motion_txt_byte_identical_to_what_it_printed(measured, capsys):
    assert script.main(_argv(measured, "--phase", "read")) == script.EXIT_OK
    printed = capsys.readouterr().out
    written = (measured.motion / "motion.txt").read_text()
    assert written == printed, "motion.txt must be what the reader saw"


def test_the_reading_is_decided_at_DECISION_H_and_reports_the_others(measured, capsys):
    """Only k = 15 decides. The other horizons are printed and decide nothing,
    exactly as the M3 gate is.

    The binding half is the SECOND run: a control doctored to leak at k = 1 --
    a reported horizon -- must change no verdict and cost no exit code, while
    the same doctoring at k = 15 is exit 38 above. A read that pooled every k
    into the decision would return 38 here.
    """
    assert script.main(_argv(measured, "--phase", "read")) == script.EXIT_OK
    out = capsys.readouterr().out
    assert "Reading D: does the latent encode displacement at k = 15" in out
    assert script.DECISION_H == 15
    # Every reported horizon has its own row in the per-k table, and no k the
    # records do not carry is printed as if it had been measured.
    rows = [line for line in out.splitlines() if line.startswith("  k=")]
    assert {line.split()[0] for line in rows} == {f"k={k}" for k in KS}
    assert sum(line.split()[-1] == "yes" for line in rows) == 1, (
        "exactly one horizon decides, and it is DECISION_H"
    )
    assert [line.split()[0] for line in rows if line.split()[-1] == "yes"] == ["k=15"]

    path = script.motion_record_path(measured.motion, JOB.arm, JOB.seed)
    record = load_record(path)
    record["k"]["k1"]["control"] = [x + 500.0 for x in record["k"]["k1"]["contrast"]]
    _rewrite(path, record)
    assert script.main(_argv(measured, "--phase", "read")) == script.EXIT_OK
    assert "UNRESOLVED CONTROL" not in capsys.readouterr().out


def test_read_names_a_missing_record_and_is_exit_11(reference, capsys):
    """Nothing measured, so the first cell has no record: named, never pooled
    over what happens to be on disk."""
    assert script.main(_argv(reference, "--phase", "read")) == script.EXIT_NO_CHECKPOINTS
    out = capsys.readouterr().out
    assert "NO CELL" in out and JOB.arm in out
    assert not (reference.motion / "motion.txt").exists()


def test_a_grid_without_DECISION_H_is_refused_rather_than_read_at_a_neighbour(
    measured, capsys,
):
    """Reading D is decided at k = 15 and nowhere else. A record measured at a
    grid that does not carry it has no reading to take, and taking the nearest
    horizon instead would print a number under 15's caption."""
    path = script.motion_record_path(measured.motion, JOB.arm, JOB.seed)
    record = load_record(path)
    record["ks"] = [1, 5]
    del record["k"]["k15"]
    _rewrite(path, record)
    with pytest.raises(ValueError, match=r"does not include the pre-registered DECISION_H"):
        script.main(_argv(measured, "--phase", "read"))
    assert not (measured.motion / "motion.txt").exists()


def test_two_records_that_did_not_score_the_same_windows_cannot_share_one_bar():
    """`pool_arm`'s own compatibility check only ever sees ONE arm here, since
    the arms are pooled separately and only their z's meet in the reading. So
    the cross-arm identity has no other backstop, and `clusters` -- the single
    number `z_fam` is read against -- would silently become whichever arm was
    pooled last."""
    base = {
        "ks": [1, 15], "windows": {"total": 2, "episode": [0, 1]},
        "episodes": {"val": ["a.npz", "b.npz"]}, "context": 5, "horizon": 15,
        "device": "cpu", "torch_version": "2.0.0",
    }
    script.require_one_protocol({("pixel_ae", 0): base, ("random_vit", 0): dict(base)})
    for field, value in (
        ("ks", [1, 5, 15]),
        ("windows", {"total": 2, "episode": [0, 0]}),
        ("horizon", 45),
        ("device", "mps"),
    ):
        other = {**base, field: value}
        with pytest.raises(ValueError, match="cannot be read against one bar"):
            script.require_one_protocol({("pixel_ae", 0): base, ("random_vit", 0): other})
