import inspect

import numpy as np
import pytest

from mbfps.eval.probe import (
    PROBE_KEYS,
    RIDGES,
    TARGET_DIM,
    _mean_r2,
    angle_error_degrees,
    apply_probe,
    filtering_comparison,
    fit_probe,
    position_error,
    probe_r2,
    probe_targets,
)

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def _reference_r2(probe: dict, latents: np.ndarray, targets: np.ndarray) -> float:
    """Per-column R^2, averaged -- independent of the module's own _mean_r2 (and
    of the module's own `probe_r2`, which delegates straight to it) so a
    mutation to that private helper doesn't also corrupt the check on it.

    Named apart from the module's `probe_r2` (imported above, and exercised
    directly by the filtering-comparison tests below) so the two are never
    confused: this one exists purely as an independent oracle."""
    pred = apply_probe(probe, latents)
    scores = []
    for c in range(targets.shape[1]):
        truth = targets[:, c]
        denom = float(((truth - truth.mean()) ** 2).sum())
        scores.append(1.0 - float(((truth - pred[:, c]) ** 2).sum()) / denom)
    return float(np.mean(scores))


def test_probe_keys_exclude_the_zero_variance_channels():
    """health and pos_z have exactly one unique value in my_way_home, so R^2 on
    them is 0/0. Including them would emit NaN or inflate an averaged score."""
    assert "health" not in PROBE_KEYS
    assert "pos_z" not in PROBE_KEYS
    assert PROBE_KEYS == ("pos_x", "pos_y", "angle")


def test_targets_encode_angle_as_sin_cos():
    privileged = np.array([[100.0, 5.0, -3.0, 0.0, 90.0]], dtype=np.float32)
    targets = probe_targets(privileged, KEYS)
    assert targets.shape == (1, 4)
    np.testing.assert_allclose(targets[0, :2], [5.0, -3.0], atol=1e-6)
    np.testing.assert_allclose(targets[0, 2:], [1.0, 0.0], atol=1e-6)


def test_targets_are_continuous_across_the_wrap():
    """359 deg and 1 deg are two degrees apart. In raw degrees they are 358
    apart, which is what makes a linear probe on degrees wrong."""
    a = probe_targets(np.array([[0, 0, 0, 0, 359.0]], dtype=np.float32), KEYS)
    b = probe_targets(np.array([[0, 0, 0, 0, 1.0]], dtype=np.float32), KEYS)
    assert np.linalg.norm(a[0, 2:] - b[0, 2:]) < 0.05


def test_probe_recovers_an_exact_linear_map():
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(200, 16))
    true_w = rng.normal(size=(17, 4))
    targets = latents @ true_w[:-1] + true_w[-1]
    probe = fit_probe(latents, targets, ridge=1e-8)
    np.testing.assert_allclose(apply_probe(probe, latents), targets, atol=1e-4)


def test_probe_has_an_intercept():
    """Without a bias term a probe cannot represent a constant offset, and Doom
    coordinates are nowhere near zero-centred."""
    latents = np.zeros((10, 4))
    targets = np.full((10, 4), 7.0)
    np.testing.assert_allclose(
        apply_probe(fit_probe(latents, targets, ridge=1e-8), latents), targets, atol=1e-3
    )


def test_ridge_selection_beats_a_fixed_default():
    """The fix's whole point. A fixed ridge=1.0 on unstandardised inputs cost
    held-out R^2 0.16 against a measured ceiling of 0.42 on real features."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 24))
    w = rng.normal(size=(25, 4)) * 200.0          # targets at Doom-coordinate scale
    y = x @ w[:-1] + w[-1] + rng.normal(size=(300, 4)) * 5.0
    xv = rng.normal(size=(150, 24))
    yv = xv @ w[:-1] + w[-1] + rng.normal(size=(150, 4)) * 5.0

    selected = fit_probe(x, y, xv, yv)
    fixed = fit_probe(x, y, ridge=1.0)
    assert _reference_r2(selected, xv, yv) >= _reference_r2(fixed, xv, yv)
    assert "r2" in selected and selected["ridge"] in RIDGES


def test_probe_standardises_its_inputs():
    """Unstandardised features of wildly different scales make ridge meaningless:
    one penalty strength cannot suit a column of std 1 and a column of std 1000."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3)) * np.array([1.0, 1000.0, 0.001])
    y = rng.normal(size=(200, 4))
    probe = fit_probe(x, y, ridge=1.0)
    # Exact equality, not a loose tolerance: pins the implementation's `+1e-8`
    # stability epsilon on `scale` precisely, rather than merely bounding it.
    np.testing.assert_array_equal(probe["scale"], x.std(0) + 1e-8)
    np.testing.assert_allclose(probe["mean"], x.mean(0), rtol=1e-6)


def test_position_error_is_euclidean():
    pred = np.array([[3.0, 4.0, 0.0, 1.0]])
    true = np.array([[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(position_error(pred, true), [5.0])


def test_angle_error_wraps_the_short_way():
    """1 deg vs 359 deg is an error of 2 deg, not 358."""
    def at(deg):
        r = np.deg2rad(deg)
        return np.array([[0.0, 0.0, np.sin(r), np.cos(r)]])

    np.testing.assert_allclose(angle_error_degrees(at(1.0), at(359.0)), [2.0], atol=1e-4)


def test_angle_error_wraps_even_when_raw_atan2_difference_exceeds_180():
    """1 deg vs 359 deg (the brief's own wrap test) never actually forces the
    wrap: atan2 already maps both angles near 0, so their raw difference never
    leaves [-180, 180] and an unwrapped `abs(pred - real)` would pass it too.
    170 vs 190 does force it: atan2 puts them at +170 and -170, a raw gap of
    340 deg, while the true short-way separation is 20 deg."""
    def at(deg):
        r = np.deg2rad(deg)
        return np.array([[0.0, 0.0, np.sin(r), np.cos(r)]])

    np.testing.assert_allclose(angle_error_degrees(at(170.0), at(190.0)), [20.0], atol=1e-4)


def test_angle_error_is_bounded_at_180():
    def at(deg):
        r = np.deg2rad(deg)
        return np.array([[0.0, 0.0, np.sin(r), np.cos(r)]])

    assert angle_error_degrees(at(0.0), at(180.0))[0] == pytest.approx(180.0, abs=1e-4)
    for deg in (0.0, 90.0, 200.0, 350.0):
        assert 0.0 <= angle_error_degrees(at(deg), at(37.0))[0] <= 180.0


def test_unnormalised_sin_cos_predictions_still_give_a_valid_angle():
    """A linear probe's (sin, cos) output does not lie on the unit circle.
    atan2 must be used, not arcsin, or the error is garbage off-circle."""
    pred = np.array([[0.0, 0.0, 0.6, 0.6]])   # 45 deg, norm 0.85
    true = np.array([[0.0, 0.0, 0.0, 1.0]])   # 0 deg
    np.testing.assert_allclose(angle_error_degrees(pred, true), [45.0], atol=1e-4)


# --- Additional guards -----------------------------------------------------
#
# The standing pattern found by every prior review of this project: tests
# guard structure and routing well but leave values and defaults unpinned.
# These four target exactly the value/default mutations called out for this
# task: a truncated/reordered RIDGES grid, a skipped standardisation, a
# regularised intercept invisible at the low ridge the base test uses, and a
# pooled-instead-of-per-column R^2.


def test_ridges_grid_is_the_measured_candidate_set():
    """A truncated or reordered grid can make selection always land on the
    same ridge without any single fit_probe call revealing it."""
    assert RIDGES == (1e-1, 1e1, 1e3, 1e5, 1e7)


def test_intercept_is_never_regularised_even_at_high_ridge():
    """test_probe_has_an_intercept uses ridge=1e-8, where regularising the
    intercept would be invisible. Pin it at a ridge large enough to matter:
    with constant-zero latents (after standardisation the design matrix's
    only non-trivial column is the intercept), the fitted intercept must
    still equal the target mean exactly, not be shrunk toward zero."""
    latents = np.zeros((20, 4))
    targets = np.full((20, 4), 500.0)
    probe = fit_probe(latents, targets, ridge=1e5)
    np.testing.assert_allclose(apply_probe(probe, latents), targets, atol=1.0)


def test_probe_r2_averages_per_column_not_pooled_variance():
    """A pooled R^2 (summing residuals and totals across all columns before
    dividing) would let a huge-variance, near-perfectly-fit column hide a
    near-zero R^2 on small-variance columns -- exactly the sin(angle)/pos_x
    scale mismatch this module exists to avoid. Per-column averaging keeps a
    bad angle prediction visible even next to a good position prediction."""
    rng = np.random.default_rng(0)
    n_train, n_val = 400, 150
    latents = rng.normal(size=(n_train, 4))
    val_latents = rng.normal(size=(n_val, 4))

    # Column 0: huge scale (like pos_x, std ~240), a clean linear function of
    # latents with tiny noise -- easy to fit almost perfectly.
    def make_targets(z):
        col0 = z[:, 0] * 1000.0
        noise = rng.normal(size=(z.shape[0], 3))  # columns 1-3: unrelated to z
        return np.column_stack([col0, noise])

    targets = make_targets(latents)
    val_targets = make_targets(val_latents)

    probe = fit_probe(latents, targets, val_latents, val_targets)
    # Correct (per-column mean): ~(nearly_1 + ~0 + ~0 + ~0) / 4 ~= 0.25.
    # Pooled: column 0's huge variance dominates the sums, driving the score
    # toward ~1 regardless of columns 1-3 being pure noise.
    assert probe["r2"] < 0.5


# --- Ridge selection must actually select, and on the right data ----------
#
# `test_ridge_selection_beats_a_fixed_default`'s 300x24 standardised data has
# near-zero discriminating power: both the selected and the fixed(1.0) ridge
# land in the effectively-unregularised regime, so "always return RIDGES[0]"
# and "score selection on training data" both pass it. A p >> n case makes
# the correct ridge unambiguous and far from RIDGES[0]: with 20 training rows
# and 100 features, an unregularised fit (ridge=1e-1) reaches train R^2 ~1.0
# by memorising noise, while its held-out R^2 is deeply negative; only heavy
# regularisation (ridge=1e3) generalises.


def _overparameterised_case(seed: int = 47):
    """20 train rows, 100 features, 60 held-out rows -- p >> n so an
    unregularised fit overfits catastrophically and only the grid's high end
    generalises. Returns (x, y, xv, yv)."""
    rng = np.random.default_rng(seed)
    n_train, n_val, p = 20, 60, 100
    w = rng.normal(size=(p + 1, 4)) * 0.3
    x = rng.normal(size=(n_train, p))
    y = x @ w[:-1] + w[-1] + rng.normal(size=(n_train, 4)) * 10.0
    xv = rng.normal(size=(n_val, p))
    yv = xv @ w[:-1] + w[-1] + rng.normal(size=(n_val, 4)) * 10.0
    return x, y, xv, yv


def test_ridge_selection_picks_the_genuine_held_out_maximum():
    """Catches turning selection into a no-op: `if best is None or score >
    best[0]` mutated to `if best is None`, which always keeps the first grid
    entry (RIDGES[0] = 1e-1) and never updates thereafter. The correct pick
    on this data is measured independently below (via `_reference_r2`, not
    the module's own `_mean_r2`) and is far from RIDGES[0]."""
    x, y, xv, yv = _overparameterised_case()

    independently_scored = {
        ridge: _reference_r2(fit_probe(x, y, ridge=ridge), xv, yv) for ridge in RIDGES
    }
    correct_ridge = max(independently_scored, key=independently_scored.get)
    assert correct_ridge != RIDGES[0]  # confirms the case is discriminating

    selected = fit_probe(x, y, xv, yv)
    assert selected["ridge"] == correct_ridge


def test_ridge_selection_scores_on_held_out_data_not_training_data():
    """Catches scoring the selection loop on `(xs, y)` instead of `(xv,
    val_targets)`. On this p >> n data the train-optimal ridge (lowest --
    train R^2 ~1.0 from memorising noise) and the val-optimal ridge (highest
    -- the only one that generalises) are different grid entries entirely.
    `fit_probe` must return the val-optimal one, and its reported "r2" must
    be the held-out score, not the training score."""
    x, y, xv, yv = _overparameterised_case()

    train_scored = {ridge: _reference_r2(fit_probe(x, y, ridge=ridge), x, y) for ridge in RIDGES}
    val_scored = {ridge: _reference_r2(fit_probe(x, y, ridge=ridge), xv, yv) for ridge in RIDGES}
    train_optimal = max(train_scored, key=train_scored.get)
    val_optimal = max(val_scored, key=val_scored.get)
    assert train_optimal != val_optimal  # confirms the case is discriminating

    selected = fit_probe(x, y, xv, yv)
    assert selected["ridge"] == val_optimal
    assert selected["r2"] == pytest.approx(val_scored[val_optimal])
    assert selected["r2"] != pytest.approx(train_scored[train_optimal], abs=0.05)


def test_fit_probe_without_validation_data_falls_back_to_the_measured_ridge():
    """No `val_latents`/`val_targets` and no explicit `ridge`: the fallback
    must be the measured 1e3 (R^2 0.16 at the old default of 1.0 against a
    0.42 ceiling), and there is no held-out score to report."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4))
    y = rng.normal(size=(20, 4))

    probe = fit_probe(x, y)
    assert probe["ridge"] == 1e3
    assert "r2" not in probe


def test_position_error_ignores_the_angle_columns():
    """`test_position_error_is_euclidean` uses identical sin/cos columns in
    predicted and true, so `predicted - true` (all 4 columns) and
    `predicted[:, :2] - true[:, :2]` agree by coincidence. Here the angle
    columns differ (0 deg vs 90 deg): distance must come from position only."""
    pred = np.array([[3.0, 4.0, 0.0, 1.0]])  # angle 0 deg
    true = np.array([[0.0, 0.0, 1.0, 0.0]])  # angle 90 deg
    np.testing.assert_allclose(position_error(pred, true), [5.0])


def test_mean_r2_skips_a_zero_variance_target_column():
    """health and pos_z have exactly zero variance in the real dataset; this
    guard is what keeps their 0/0 R^2 from poisoning the average with nan."""
    predicted = np.array([[1.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    targets = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])  # column 1: zero variance
    result = _mean_r2(predicted, targets)
    assert np.isfinite(result)
    # Column 1 is skipped entirely, so the result is column 0's R^2 alone.
    denom = float(((targets[:, 0] - targets[:, 0].mean()) ** 2).sum())
    numer = float(((targets[:, 0] - predicted[:, 0]) ** 2).sum())
    assert result == pytest.approx(1.0 - numer / denom)


def test_mean_r2_raises_when_every_column_has_zero_variance():
    predicted = np.array([[5.0], [5.0], [5.0]])
    targets = np.array([[5.0], [5.0], [5.0]])
    with pytest.raises(ValueError, match="zero variance"):
        _mean_r2(predicted, targets)


def test_probe_targets_names_the_missing_keys():
    """A bare `KeyError` from the internal index lookup (e.g. `index['angle']`)
    would only ever name one missing key, and not necessarily the most useful
    one. The guard must report all of them."""
    privileged = np.zeros((1, 2))
    with pytest.raises(KeyError) as excinfo:
        probe_targets(privileged, ("foo", "bar"))
    message = str(excinfo.value)
    assert "pos_x" in message
    assert "pos_y" in message
    assert "angle" in message


# --- Filtering probe: does the posterior latent carry history? ------------
#
# The posterior has already seen frame t, so by the data-processing
# inequality it cannot add information about t over the encoder embedding of
# t. What it can add is history, carried in the deterministic state `h`. If
# a probe on the posterior latent does not beat a probe on the raw embedding
# at the same timestep, `h` is inert -- a specific, actionable bug.


def test_r2_is_one_for_a_perfect_linear_fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 8))
    w = rng.normal(size=(9, 4))
    y = x @ w[:-1] + w[-1]
    assert probe_r2(fit_probe(x, y, ridge=1e-10), x, y) == pytest.approx(1.0, abs=1e-4)


def test_r2_is_near_zero_for_unrelated_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 8))
    y = rng.normal(size=(400, 4))
    assert probe_r2(fit_probe(x, y, ridge=1.0), rng.normal(size=(400, 8)), y) < 0.2


def test_r2_uses_per_column_variance_not_pooled():
    """pos_x has std ~253 and sin(angle) ~0.7. Pooling the variance would let
    the position columns dominate and hide a useless angle prediction."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    y = np.column_stack([
        x[:, 0] * 250.0,            # large scale, perfectly predictable
        x[:, 1] * 250.0,
        rng.normal(size=200) * 0.7,  # small scale, pure noise
        rng.normal(size=200) * 0.7,
    ])
    r2 = probe_r2(fit_probe(x, y, ridge=1e-8), x, y)
    assert r2 < 0.9, f"pooled variance hid two unpredictable columns (r2={r2:.3f})"


def test_filtering_comparison_detects_history_in_the_latent():
    """Synthetic: the latent carries a lagged signal the embedding lacks, so a
    probe on it must score higher."""
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    lagged = np.roll(signal, 1)
    embed = signal[:, None] * np.ones((1, 4))
    latent = np.column_stack([embed, lagged[:, None] * np.ones((1, 4))])
    targets = np.column_stack([lagged, lagged, lagged, lagged]) + 0.01 * rng.normal(size=(n, 4))

    half = n // 2
    result = filtering_comparison(
        latent[:half], embed[:half], latent[half:], embed[half:],
        targets[:half], targets[half:],
    )
    assert result["latent_r2"] > result["embedding_r2"]
    assert result["latent_beats_embedding"] is True


def test_filtering_comparison_reports_failure_when_the_latent_adds_nothing():
    """If h is dead the latent is just the embedding, and this must say so
    rather than passing quietly."""
    rng = np.random.default_rng(0)
    n = 400
    embed = rng.normal(size=(n, 4))
    latent = np.concatenate([embed, np.zeros((n, 4))], axis=1)  # dead h
    targets = embed + 0.01 * rng.normal(size=(n, 4))
    half = n // 2
    result = filtering_comparison(
        latent[:half], embed[:half], latent[half:], embed[half:],
        targets[:half], targets[half:],
    )
    assert result["latent_beats_embedding"] is False


def test_filtering_comparison_fits_on_the_train_split_not_the_validation_split():
    """A probe fit on the *validation* set (a leak: passing val data as the
    fit arguments instead of train) would memorise a small val split when it
    has more features than rows, reporting a near-perfect R^2 that has
    nothing to do with genuine held-out generalisation. This is caught
    empirically here, not by inspection: `test_r2_is_near_zero_for_unrelated_
    features` -- the test the task brief names as the guard against exactly
    this mutation -- never calls `filtering_comparison` at all, so it cannot
    catch anything wrong inside it (confirmed: the mutation survives the
    entire pre-existing suite, 27/27 green, before this test was added).

    On fully unrelated features and targets, a probe correctly fit only on a
    large TRAIN split, then scored on a small held-out split, must not reach
    a high R^2 -- unlike a leaked fit-on-val, which (measured) reaches
    ~0.9999 on this exact data by memorising 5 points with 20 features.
    """
    rng = np.random.default_rng(0)
    n_train, n_val, p = 200, 5, 20
    latent_train = rng.normal(size=(n_train, p))
    embed_train = rng.normal(size=(n_train, p))
    latent_val = rng.normal(size=(n_val, p))
    embed_val = rng.normal(size=(n_val, p))
    targets_train = rng.normal(size=(n_train, 4))
    targets_val = rng.normal(size=(n_val, 4))

    result = filtering_comparison(
        latent_train, embed_train, latent_val, embed_val, targets_train, targets_val
    )
    assert result["latent_r2"] < 0.5
    assert result["embedding_r2"] < 0.5


def test_filtering_comparison_selects_each_probes_ridge_independently():
    """Sharing one ridge across both probes (e.g. reusing whichever ridge the
    latent probe selected to also fit the embedding probe) would handicap
    whichever space needs different regularisation -- exactly what passing
    validation data to each `fit_probe` call independently exists to avoid.
    This mutation is not cosmetic: measured, it survives the entire
    pre-existing suite (27/27 green) before this test was added.

    Engineered gap: latent is low-dimensional and well-determined (a small
    ridge already generalises well); embedding is overparameterised relative
    to n_train (only heavy regularisation generalises). A correct, per-probe
    independent selection must score the embedding at least as well as
    forcing the latent probe's selected ridge onto it would."""
    rng = np.random.default_rng(0)
    n_train, n_val = 50, 60
    p_lat, p_emb = 3, 150

    w_lat = rng.normal(size=(p_lat + 1, 4)) * 3.0
    latent_train = rng.normal(size=(n_train, p_lat))
    latent_val = rng.normal(size=(n_val, p_lat))
    targets_train = latent_train @ w_lat[:-1] + w_lat[-1] + rng.normal(size=(n_train, 4)) * 0.2
    targets_val = latent_val @ w_lat[:-1] + w_lat[-1] + rng.normal(size=(n_val, 4)) * 0.2

    # Overparameterised relative to n_train=50, unrelated to targets: a small
    # ridge overfits noise, only heavy regularisation generalises.
    embed_train = rng.normal(size=(n_train, p_emb))
    embed_val = rng.normal(size=(n_val, p_emb))

    result = filtering_comparison(
        latent_train, embed_train, latent_val, embed_val, targets_train, targets_val
    )

    latent_only = fit_probe(latent_train, targets_train, latent_val, targets_val)
    forced = fit_probe(embed_train, targets_train, ridge=latent_only["ridge"])
    forced_r2 = probe_r2(forced, embed_val, targets_val)

    assert result["embedding_r2"] > forced_r2


def test_filtering_comparison_genuinely_selects_a_ridge_for_each_probe():
    """Sharing a ridge and eliminating selection entirely are different
    mutations. `test_filtering_comparison_selects_each_probes_ridge_
    independently` above catches the latent probe's selected ridge being
    reused for the embedding probe -- but it does not catch BOTH probes
    skipping selection and silently using a hardcoded `ridge=1e3` instead,
    because on that test's data 1e3 already beats the "forced" comparison it
    makes (measured: the hardcode-both-to-1e3 mutation survives the entire
    suite, including that test, before this one was added).

    Both spaces here are built from the same underlying `cause` and the same
    `targets`, so `filtering_comparison`'s shared `targets_train`/
    `targets_val` contract is respected. Each space's genuinely-best ridge
    (found independently below by sweeping RIDGES and scoring with
    `_reference_r2`, not the module's own `_mean_r2`) is engineered to differ
    from 1e3 AND from the other space's optimum, by a wide held-out R^2
    margin in both spaces:
    - `latent`: a clean, low-noise copy of `cause` -- light regularisation
      (RIDGES[0] = 0.1) wins, and 1e3 already oversmooths it badly.
    - `embedding`: a noisier, higher-dimensional linear encoding of the same
      `cause` -- RIDGES[1] = 10 wins, and 1e3 oversmooths it just as badly,
      by a different margin than the latent space's.

    A genuine, independent per-probe selection reports each space's true
    optimum. Hardcoding 1e3 for both collapses BOTH reported values toward
    the (measurably worse) fixed-1e3 score; sharing the latent probe's ridge
    with the embedding probe collapses only the embedding value, since the
    two optima are different grid entries -- either mutation is caught here.
    """
    rng = np.random.default_rng(11)
    n_train, n_val, p_lat, p_emb = 30, 100, 3, 60

    cause_train = rng.normal(size=(n_train, p_lat))
    cause_val = rng.normal(size=(n_val, p_lat))
    w = rng.normal(size=(p_lat + 1, 4)) * 5.0
    targets_train = cause_train @ w[:-1] + w[-1] + rng.normal(size=(n_train, 4)) * 0.01
    targets_val = cause_val @ w[:-1] + w[-1] + rng.normal(size=(n_val, 4)) * 0.01

    latent_train, latent_val = cause_train, cause_val

    mix = rng.normal(size=(p_lat, p_emb))
    embedding_train = cause_train @ mix + rng.normal(size=(n_train, p_emb))
    embedding_val = cause_val @ mix + rng.normal(size=(n_val, p_emb))

    def ridge_scores(x, y, xv, yv):
        return {r: _reference_r2(fit_probe(x, y, ridge=r), xv, yv) for r in RIDGES}

    latent_scores = ridge_scores(latent_train, targets_train, latent_val, targets_val)
    embedding_scores = ridge_scores(embedding_train, targets_train, embedding_val, targets_val)
    latent_best = max(latent_scores, key=latent_scores.get)
    embedding_best = max(embedding_scores, key=embedding_scores.get)

    # Confirm the case is discriminating before trusting it as a guard: the
    # two optima must differ from each other and from 1e3, by a wide margin.
    assert latent_best != 1e3 and embedding_best != 1e3
    assert latent_best != embedding_best
    assert latent_scores[latent_best] - latent_scores[1e3] > 0.1
    assert embedding_scores[embedding_best] - embedding_scores[1e3] > 0.1

    result = filtering_comparison(
        latent_train, embedding_train, latent_val, embedding_val, targets_train, targets_val
    )
    assert result["latent_r2"] == pytest.approx(latent_scores[latent_best], abs=1e-6)
    assert result["embedding_r2"] == pytest.approx(embedding_scores[embedding_best], abs=1e-6)


# ---------------------------------------------------------------------------
# fit_probes -- the whole-episode driver. Task 12's CLI and its untrained
# control both call it, so what it returns IS the measurement.
# ---------------------------------------------------------------------------

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mbfps.data.episode import Episode, save_episode  # noqa: E402
from mbfps.envs.protocol import OBS_SHAPE  # noqa: E402
from mbfps.eval.probe import fit_probes  # noqa: E402

DX, DY = 10.0, -3.0


def _tagged_episode(length: int) -> Episode:
    """`obs[t]` carries `t` verbatim and `pos_x = 10*t`, `pos_y = -3*t`."""
    obs = np.zeros((length + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(length + 1, dtype=np.uint8)
    t = np.arange(length + 1, dtype=np.float32)
    privileged = np.zeros((length + 1, len(KEYS)), dtype=np.float32)
    privileged[:, 0] = 100.0
    privileged[:, 1] = DX * t
    privileged[:, 2] = DY * t
    return Episode(
        obs=obs,
        actions=np.arange(length, dtype=np.int32) % 6,
        rewards=np.zeros(length, dtype=np.float32),
        terminated=np.zeros(length, dtype=bool),
        truncated=np.zeros(length, dtype=bool),
        privileged=privileged,
        privileged_keys=KEYS,
        policy_name="synthetic",
        seed=0,
        scenario="probe",
    )


def _write_episodes(tmp_path, lengths) -> list:
    paths = []
    for i, length in enumerate(lengths):
        path = tmp_path / f"ep_{i:06d}_len{length:05d}.npz"
        save_episode(_tagged_episode(length), path)
        paths.append(path)
    return paths


class _TagEncoder(nn.Module):
    """`emb[n] = [index of the frame obs[n]]` -- an exact, invertible tag."""

    def forward(self, obs):
        return obs[:, 0, 0, 0].to(torch.float32).unsqueeze(-1)


def _packed(latent: torch.Tensor) -> dict:
    """The real RSSM returns `latent = cat([h, z])` plus `h` and `z` separately,
    and callers hand `(h[:, -1], z[:, -1])` back as the next window's state.
    These stubs are one feature wide, so there is nothing to split -- but the
    keys have to be there or the state handoff cannot be exercised at all."""
    return {"latent": latent, "h": latent, "z": latent}


class _PassThroughRSSM(nn.Module):
    def observe(self, embeddings, actions, state=None):
        return _packed(embeddings)


class _NoisyRSSM(nn.Module):
    """A posterior that genuinely SAMPLES, like the real categorical one."""

    def observe(self, embeddings, actions, state=None):
        return _packed(embeddings + torch.randn_like(embeddings))


class _WideningHeads(nn.Module):
    """Predicted embedding is a different WIDTH from the encoder's output, so
    which of the two the probe was fit on is visible in its shape alone."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, latent):
        return {"embedding": latent.repeat_interleave(self.width, dim=-1)}


class _FakeModel(nn.Module):
    input_kind = "obs"

    def __init__(self, rssm=None, head_width: int = 1) -> None:
        super().__init__()
        self.encoder = _TagEncoder()
        self.rssm = rssm or _PassThroughRSSM()
        self.heads = _WideningHeads(head_width)


def _fit(paths, **kwargs):
    # context=2, horizon=3 -> a 5-frame window. The production window is
    # 5+45=50, longer than these fixture episodes; 5 keeps the same frames in
    # play while still exercising the real windowing. Lengths below are
    # multiples of 5 so every frame lands inside a complete window and the
    # "which episodes were fit on" assertions stay exact.
    kwargs.setdefault("context", 2)
    kwargs.setdefault("horizon", 3)
    return fit_probes(_FakeModel(**kwargs.pop("model", {})), paths, None,
                      torch.device("cpu"), **kwargs)


def test_fit_probes_fits_the_embedding_probe_on_predicted_not_encoder_embeddings(
    tmp_path,
):
    """THE pipeline test. `evaluate_rollout` scores all three references after
    the embedding HEAD, so the probe must be fit on the head's output. Fitting
    it on the raw encoder embedding instead is a distribution mismatch that was
    measured destroying the signal -- band below 2 SE at 18 of 45 horizon steps
    against 2 of 45, gap_closed at 45 moving -7.26 -> -0.78 on one checkpoint.

    Made visible by giving the head a different width from the encoder: the
    encoder emits 1 feature, the head 5, so the fitted weight matrix's shape
    says unambiguously which one was regressed."""
    paths = _write_episodes(tmp_path, [20])
    latent_probe, embedding_probe = _fit(paths, model={"head_width": 5}, ridge=1e-8)

    assert latent_probe["w"].shape == (1 + 1, 4), "latent probe is the 1-d tag"
    assert embedding_probe["w"].shape == (5 + 1, 4), (
        "embedding probe must be fit on the 5-wide PREDICTED embedding, not on "
        "the 1-wide encoder output"
    )


def test_fit_probes_aligns_each_latent_with_the_frame_its_action_produced(tmp_path):
    """The regression must pair `latent[i]` with the privileged state of the
    frame action `i` LED TO, matching RSSM.observe's convention. Dropping the
    last frame instead of the first shifts every pair by one step and still
    fits perfectly -- both sides stay linear in the frame index -- so only the
    recovered VALUES can catch it. Here a tag of 5 must decode to frame 5's
    position; under the off-by-one it decodes to frame 6's."""
    paths = _write_episodes(tmp_path, [20])
    latent_probe, _ = _fit(paths, ridge=1e-8)

    decoded = apply_probe(latent_probe, np.array([[5.0]]))
    expected = probe_targets(_tagged_episode(20).privileged[5:6], KEYS)
    np.testing.assert_allclose(decoded, expected, atol=1e-4)
    assert not np.allclose(
        decoded, probe_targets(_tagged_episode(20).privileged[6:7], KEYS), atol=1.0
    ), "a tag of 5 decoded to frame 6 -- the pairing is off by one"


def test_fit_probes_is_reproducible_under_a_fixed_seed(tmp_path):
    """The posterior samples, so two evaluation runs fit two different probes
    unless the seed pins them -- and then every downstream error, band and
    gap_closed moves for no reason anyone could trace. Reproducibility comes
    from the seed, never from taking the categorical mode."""
    paths = _write_episodes(tmp_path, [20, 25])
    first, _ = _fit(paths, model={"rssm": _NoisyRSSM()}, seed=0, ridge=1e-8)
    second, _ = _fit(paths, model={"rssm": _NoisyRSSM()}, seed=0, ridge=1e-8)
    np.testing.assert_array_equal(first["w"], second["w"])


def test_fit_probes_seed_actually_drives_the_sampling(tmp_path):
    """Guards the test above from passing vacuously: if the seed were ignored
    (or the model were secretly deterministic) two seeds would agree too, and
    the reproducibility check would prove nothing."""
    paths = _write_episodes(tmp_path, [20, 25])
    first, _ = _fit(paths, model={"rssm": _NoisyRSSM()}, seed=0, ridge=1e-8)
    other, _ = _fit(paths, model={"rssm": _NoisyRSSM()}, seed=1, ridge=1e-8)
    assert not np.array_equal(first["w"], other["w"])


def test_fit_probes_selects_the_ridge_on_episodes_it_did_not_fit_on(tmp_path):
    """Selection needs data the weights never saw. Held out at EPISODE
    granularity, because consecutive frames are near-duplicates: a row-wise
    split would put the same scene on both sides and every ridge would score
    the same."""
    lengths = [20, 25, 30, 35, 40]
    paths = _write_episodes(tmp_path, lengths)
    latent_probe, _ = _fit(paths, select_episodes=1)

    assert latent_probe["ridge"] in RIDGES
    assert "r2" in latent_probe, "a selected probe must report what it scored"

    # The standardisation is computed from the FIT episodes only. Lengths
    # differ, so the mean tag over the first four episodes is not the mean over
    # all five -- fitting on everything is therefore visible here.
    fit_tags = np.concatenate([np.arange(1, n + 1) for n in lengths[:4]])
    all_tags = np.concatenate([np.arange(1, n + 1) for n in lengths])
    assert latent_probe["mean"][0] == pytest.approx(fit_tags.mean())
    assert latent_probe["mean"][0] != pytest.approx(all_tags.mean())


def test_fit_probes_falls_back_when_there_is_nothing_to_spare(tmp_path):
    """One episode cannot be both fit on and selected on. The fallback is
    fit_probe's default penalty, not a crash and not a leaked selection."""
    paths = _write_episodes(tmp_path, [20])
    latent_probe, _ = _fit(paths, select_episodes=4)

    assert latent_probe["ridge"] == 1e3
    assert "r2" not in latent_probe


def test_fit_probes_rejects_an_empty_episode_list():
    with pytest.raises(ValueError, match="no episodes"):
        _fit([])


# ---------------------------------------------------------------------------
# gather_probe_data -- the probe must be fit under the ROLLOUT's own protocol.
#
# `evaluate_rollout` gives the RSSM `context` real frames out of a ZERO state
# and then imagines `horizon` more. Fitting the probe on whole-episode
# filtering instead hands `h` ~500 steps of history it never has at evaluation
# time. Measured on the shipped checkpoint, same frames and same probe:
# position error 222.4 with the long context against 247.5 with the short one.
#
# The bias is NOT common-mode across the three references, which is what makes
# it fatal rather than merely untidy: persistence is frozen at the context step
# forever while the floor's own context grows to context+horizon, and measured
# probe error by context length is 317.3 / 223.4 / 213.5 / 215.3 at contexts
# 1 / 5 / 20 / 50. The floor therefore gains ~8 units from filtering history
# alone, against a median band width of 55 -- a real bias inside the quantity
# the study's headline ratio divides by.
# ---------------------------------------------------------------------------

import mbfps.eval.probe as probe_module  # noqa: E402
import mbfps.eval.rollout as rollout_module  # noqa: E402
from mbfps.eval.probe import gather_probe_data  # noqa: E402
from mbfps.eval.rollout import evaluate_rollout  # noqa: E402


class _DepthRSSM(nn.Module):
    """Latent == how many frames have been filtered since the last ZERO state.

    That count is the thing at issue. Under the rollout's protocol it can never
    exceed `context + horizon` and it restarts at 1 for every window; under
    whole-episode filtering it runs to the episode length.
    """

    def observe(self, embeddings, actions, state=None):
        b, t, _ = embeddings.shape
        already = 0.0 if state is None else float(state[0][0, 0])
        depth = already + torch.arange(1, t + 1, dtype=torch.float32).view(1, t, 1)
        return _packed(depth.expand(b, t, 1).contiguous())


class _ObserveImagineRSSM(nn.Module):
    """A pass-through posterior plus a trivial one-step-per-action dynamics
    model -- just enough structure for BOTH `gather_probe_data` (`observe`
    only) and `evaluate_rollout` (`observe` and `imagine`) to run on the same
    fixture, so the two can be compared directly on identical episodes."""

    def observe(self, embeddings, actions, state=None):
        return _packed(embeddings)

    def imagine(self, actions, state):
        h0 = state[0]
        steps = torch.arange(1, actions.shape[1] + 1, dtype=h0.dtype).view(1, -1, 1)
        return _packed(h0.unsqueeze(1) + steps)


class _RecordingRSSM(nn.Module):
    """Records `(length, warm_started)` for every `observe` call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, bool]] = []

    def observe(self, embeddings, actions, state=None):
        self.calls.append((int(embeddings.shape[1]), state is not None))
        b, t, _ = embeddings.shape
        return _packed(torch.zeros(b, t, 1))


def _gather(paths, rssm=None, head_width: int = 1, **kwargs):
    kwargs.setdefault("context", 2)
    kwargs.setdefault("horizon", 3)
    model = _FakeModel(rssm=rssm, head_width=head_width)
    return gather_probe_data(model, paths, None, torch.device("cpu"), **kwargs)


def test_gather_probe_data_never_filters_more_history_than_the_rollout_does(tmp_path):
    """THE guard. Every latent the probe is fit on must carry exactly as many
    steps of filtering history as the latent the probe is later APPLIED to.

    Whole-episode filtering gives `h` ~500 steps; the rollout gives it
    `context` real frames out of a zero state. Measured on the shipped
    checkpoint that is a 222.4-vs-247.5 position error on identical frames with
    an identical probe, and it does not cancel across the three references --
    persistence stays at the context step while the floor's context grows to
    `context + horizon`, so the band's own denominator moves.
    """
    paths = _write_episodes(tmp_path, [20])
    data = _gather(paths, rssm=_DepthRSSM(), context=2, horizon=3)

    depths = data["latent"][:, 0]
    assert depths.max() == 5.0, (
        f"a latent carried {depths.max():.0f} steps of filtering history, but the "
        "rollout only ever gives it context+horizon=5 -- the probe is being fit "
        "on a distribution it is never applied to"
    )
    # Four windows over a 20-transition episode, each restarting from zero.
    np.testing.assert_array_equal(depths, np.tile(np.arange(1, 6.0), 4))


def test_gather_probe_data_observes_with_the_rollouts_own_call_protocol(tmp_path):
    """`context` frames from a ZERO state, then the rest warm-started -- the
    exact call sequence `evaluate_rollout` makes.

    For a plain recurrence this is equivalent to one call over the window, so
    this test pins the STRUCTURE rather than a number: the gathering must go on
    reading as the rollout reads, so that a future change to `observe`'s
    sequence-boundary behaviour cannot silently separate the two.
    """
    paths = _write_episodes(tmp_path, [20])
    rssm = _RecordingRSSM()
    _gather(paths, rssm=rssm, context=2, horizon=3)

    assert rssm.calls == [(2, False), (3, True)] * 4, (
        f"observe was called as {rssm.calls}, not as the rollout calls it: "
        "context frames from a zero state, then horizon frames warm-started"
    )


def test_gather_probe_data_pairs_each_latent_with_the_frame_its_action_produced(
    tmp_path,
):
    """The `start + 1` in the target slice.

    `embeddings[k]` is frame `start + 1 + k`, so the target must be
    `privileged[start + 1 + k]`. Losing the `+1` shifts every window by one
    frame and still fits perfectly -- both sides stay linear in the frame index
    -- so only the recovered VALUES catch it. The tagged fixture makes the
    frame index readable straight off the latent, and `pos_x == 10 * index`.
    """
    paths = _write_episodes(tmp_path, [20])
    data = _gather(paths, context=2, horizon=3)

    frame = data["latent"][:, 0]          # _TagEncoder: the latent IS the index
    np.testing.assert_array_equal(frame, np.arange(1, 21.0))
    np.testing.assert_allclose(data["targets"][:, 0], DX * frame, atol=1e-4)
    np.testing.assert_allclose(data["targets"][:, 1], DY * frame, atol=1e-4)


def test_gather_probe_data_windows_an_episode_exactly_as_the_rollout_does(tmp_path):
    """Same stride, same final window. `range(0, length - need + 1, need)`:
    dropping the `+1` loses the last window whenever `length % need == 0`, and
    a 20-transition episode at need=5 is exactly that case."""
    paths = _write_episodes(tmp_path, [10, 13])
    data = _gather(paths, context=2, horizon=3)

    # ep0 (10 transitions): windows at 0 and 5   -> frames 1..10
    # ep1 (13 transitions): windows at 0 and 5   -> frames 1..10, 11..13 unused
    expected = np.concatenate([np.arange(1, 11.0), np.arange(1, 11.0)])
    np.testing.assert_array_equal(data["latent"][:, 0], expected)


def test_gather_probe_data_returns_aligned_rows_under_the_four_documented_keys(
    tmp_path,
):
    """Two consumers read this dict by key and pair their arrays at the SAME
    timestep, so misaligned rows would compare two different frames and no
    shape check would notice.

    There are FOUR keys, and the two embeddings are not interchangeable:
    `"embedding"` is the head's PREDICTED embedding (the space the rollout band
    is scored in, what `fit_probes` fits on) and `"encoder_embedding"` is the
    RAW encoder output for the same frame (the reference the filtering gate
    criterion needs, per spec section 3.4). The widening head makes the two
    different WIDTHS here, so any collapse of one into the other is visible in
    the shapes alone."""
    paths = _write_episodes(tmp_path, [20])
    data = _gather(paths, head_width=7, context=2, horizon=3)

    assert set(data) == {"latent", "embedding", "encoder_embedding", "targets"}
    assert data["latent"].shape == (20, 1)
    assert data["embedding"].shape == (20, 7), "embedding is the HEAD's output"
    assert data["encoder_embedding"].shape == (20, 1), (
        "encoder_embedding is the ENCODER's output, not the head's"
    )
    assert data["targets"].shape == (20, TARGET_DIM)
    # Rows are aligned: the widening head repeats the tag, so column 0 of the
    # embedding must still be the frame the target describes.
    np.testing.assert_allclose(data["targets"][:, 0], DX * data["embedding"][:, 0],
                               atol=1e-4)
    # ...and the raw encoder embedding is the tag itself, on the same rows.
    np.testing.assert_allclose(data["encoder_embedding"][:, 0], np.arange(1, 21.0),
                               atol=1e-4)
    np.testing.assert_allclose(data["targets"][:, 0],
                               DX * data["encoder_embedding"][:, 0], atol=1e-4)


def test_gather_probe_data_skips_episodes_too_short_for_one_window(tmp_path):
    """Short episodes are skipped, not truncated into a shorter-context window
    that would reintroduce the very mismatch this function exists to remove."""
    paths = _write_episodes(tmp_path, [3, 20])
    data = _gather(paths, context=2, horizon=3)
    assert data["latent"].shape[0] == 20, "the 3-transition episode contributed rows"


def test_gather_probe_data_rejects_a_dataset_with_no_window_long_enough(tmp_path):
    paths = _write_episodes(tmp_path, [3])
    with pytest.raises(ValueError, match="no probe window reached"):
        _gather(paths, context=2, horizon=3)


def test_gather_probe_data_rejects_a_degenerate_context_or_horizon(tmp_path):
    paths = _write_episodes(tmp_path, [20])
    with pytest.raises(ValueError, match="context"):
        _gather(paths, context=0, horizon=3)
    with pytest.raises(ValueError, match="horizon"):
        _gather(paths, context=2, horizon=0)


def test_fit_probes_forwards_the_rollout_context_to_the_gathering(tmp_path):
    """`fit_probes` must not gather at its own default while the caller asked
    for a different context -- that is the original defect one level up."""
    paths = _write_episodes(tmp_path, [20])
    probe, _ = fit_probes(_FakeModel(rssm=_DepthRSSM()), paths, None,
                          torch.device("cpu"), context=2, horizon=3, ridge=1e-8)
    # _DepthRSSM's latents run 1..need per window, so the standardisation mean
    # reads back the window length the gathering actually used.
    assert probe["mean"][0] == pytest.approx(3.0), "need was not 2+3=5"

    probe, _ = fit_probes(_FakeModel(rssm=_DepthRSSM()), paths, None,
                          torch.device("cpu"), context=1, horizon=1, ridge=1e-8)
    assert probe["mean"][0] == pytest.approx(1.5), "need was not 1+1=2"


# ---------------------------------------------------------------------------
# Review follow-up: five mutations that survived the full 444-test suite
# because nothing pinned the new functions' defaults, matched their window
# guard against `evaluate_rollout`'s, or checked that the fit/selection split
# is genuinely disjoint. See .superpowers/sdd/task-2-report.md.
# ---------------------------------------------------------------------------


def test_fit_probes_defaults_are_the_spec_values():
    """context=5, horizon=45, select_episodes=4 are the production defaults.
    `scripts/eval_rollout.py` passes `context`/`horizon` explicitly but NEVER
    passes `select_episodes` -- it relies entirely on the bare default to
    enable ridge selection. A silently changed default here changes probe
    behaviour (and therefore every downstream rollout number) for every real
    caller without any caller noticing."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(fit_probes).parameters.items()
    }
    assert defaults["context"] == 5
    assert defaults["horizon"] == 45
    assert defaults["select_episodes"] == 4


def test_gather_probe_data_defaults_are_the_spec_values():
    """context=5, horizon=45 must match `evaluate_rollout`'s own defaults.
    `fit_probes` always forwards its caller's context/horizon here, but a
    bare `gather_probe_data(...)` call -- from a future diagnostic script, or
    from a test that forgets to pass them -- must land on the SAME window the
    rollout evaluates, not on whatever this function's own default happens to
    be."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(gather_probe_data).parameters.items()
    }
    assert defaults["context"] == 5
    assert defaults["horizon"] == 45


def test_fit_probes_bare_call_runs_ridge_selection_under_the_default_select_episodes(
    tmp_path,
):
    """`scripts/eval_rollout.py` never passes `select_episodes`; it relies on
    the bare default to enable selection. If that default silently became 0,
    `spare = len(used) - select_episodes` would make `select_episodes <= 0`
    true for every real call, and every production rollout would silently
    fall back to the untuned ridge=1e3 (measured held-out R^2 0.16 against a
    ceiling of 0.42) instead of a genuinely selected ridge -- with the full
    suite still green, because no existing test drives `fit_probes` through a
    bare call under the production defaults.

    `"r2"` is present in the returned dict ONLY when selection actually ran
    (see `fit_probe`), so its presence is the direct behavioural signal --
    checking the numeric default alone (above) would not distinguish "0" from
    some other wrong-but-nonzero default that happens to still select.
    """
    # >= need + 1 = 5 + 45 + 1 = 51 transitions, under the DEFAULT
    # context/horizon -- this exercises the real bare-call path, not a
    # shortened fixture. Five episodes so spare = 5 - default(4) = 1.
    lengths = [51, 52, 53, 54, 55]
    paths = _write_episodes(tmp_path, lengths)

    latent_probe, embedding_probe = fit_probes(
        _FakeModel(), paths, None, torch.device("cpu")
    )
    assert "r2" in latent_probe, "ridge selection did not run under the bare defaults"
    assert "r2" in embedding_probe


def test_gather_probe_data_windows_match_the_rollouts_length_guard_exactly(
    tmp_path, monkeypatch
):
    """`episode.length == need` is the ONE length where `evaluate_rollout`'s
    guard (`< need + 1`) and a weakened `< need` guard disagree: the weaker
    guard admits a window the rollout skips, reintroducing exactly the
    protocol divergence this task exists to close. Rather than hardcoding
    which side is "right", this compares the two functions directly on the
    identical episode and requires them to agree on how many windows it
    contributes -- whatever that number is.
    """
    context, horizon = 2, 3
    need = context + horizon
    paths = _write_episodes(tmp_path, [need])  # exactly `need` transitions

    model = _FakeModel(rssm=_ObserveImagineRSSM())
    probe = fit_probe(np.zeros((2, 1)), np.zeros((2, 4)), ridge=1.0)

    seen = []
    real_probe_targets = rollout_module.probe_targets
    monkeypatch.setattr(
        rollout_module,
        "probe_targets",
        lambda privileged, keys: seen.append(privileged) or real_probe_targets(privileged, keys),
    )
    try:
        evaluate_rollout(
            model, paths, probe, context=context, horizon=horizon,
            device=torch.device("cpu"),
        )
    except ValueError as error:
        assert "no validation window" in str(error)
    rollout_windows = len(seen)

    try:
        data = gather_probe_data(
            model, paths, None, torch.device("cpu"), context=context, horizon=horizon
        )
        gather_windows = data["latent"].shape[0] // need
    except ValueError as error:
        assert "no probe window" in str(error)
        gather_windows = 0

    assert gather_windows == rollout_windows, (
        f"gather_probe_data produced {gather_windows} windows at "
        f"episode.length == need, but evaluate_rollout produced {rollout_windows} "
        "for the identical episode -- the two window guards have diverged"
    )


def test_fit_probes_selection_and_fit_paths_are_disjoint(tmp_path, monkeypatch):
    """A recording wrapper around `gather_probe_data` pins the actual PATH
    LISTS `fit_probes` hands to each half of the split. `used[spare:]` is the
    exact complement of `used[:spare]`; an off-by-one (`used[spare - 1:]`)
    would put one episode on BOTH sides, so the ridge would be selected
    partly on data it was also fit on. The existing coverage
    (`test_fit_probes_selects_the_ridge_on_episodes_it_did_not_fit_on`) only
    checks the fit side's standardisation mean and does not notice an extra
    episode leaking into the selection side, so this checks path identity
    directly instead."""
    lengths = [20, 25, 30, 35, 40]
    paths = _write_episodes(tmp_path, lengths)

    calls = []
    real_gather = probe_module.gather_probe_data

    def recording_gather(model, ps, backbone, device, *args, **kwargs):
        calls.append(list(ps))
        return real_gather(model, ps, backbone, device, *args, **kwargs)

    monkeypatch.setattr(probe_module, "gather_probe_data", recording_gather)

    _fit(paths, select_episodes=1)

    assert len(calls) == 2, "expected exactly one fit-side and one selection-side gather"
    fit_paths, select_paths = calls
    assert set(fit_paths).isdisjoint(select_paths), (
        f"fit paths {fit_paths} and selection paths {select_paths} overlap -- "
        "the ridge would be selected on data it was also fit on"
    )


# ---------------------------------------------------------------------------
# filtering_report -- spec section 4, gate criterion 4.
#
# `filtering_comparison` was implemented and unit-tested but called from
# NOTHING in src/ or scripts/, so the criterion could not be produced at all.
# This is the driver that produces it.
#
# The reference it compares the latent against is the RAW ENCODER embedding
# (`gather_probe_data`'s "encoder_embedding"), per spec section 3.4:
# "compared against a probe on the encoder embedding at t". That is the whole
# argument, not a detail. The posterior has already seen frame t, so by the
# data-processing inequality it cannot add information about t over the
# ENCODING of t -- the only thing it can add is memory of earlier frames.
# Against the model's own PREDICTED embedding the question would instead be
# "does the latent beat its own head's reconstruction of itself", which a
# latent can win with `h` completely inert, since the head is lossy and the
# latent is its input. The gate would then be passable without the property it
# certifies. `gather_probe_data` returns both arrays for exactly this reason.
# ---------------------------------------------------------------------------

from mbfps.eval.probe import filtering_report  # noqa: E402


class _DeadHeads(nn.Module):
    """A head whose predicted embedding carries NO information at all.

    Stands in for the general fact that the head is lossy: comparing the latent
    against the head's own output can only flatter the latent.
    """

    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.width = width

    def forward(self, latent):
        zeros = torch.zeros_like(latent).repeat_interleave(self.width, dim=-1)
        return {"embedding": zeros}


def test_filtering_report_returns_the_gate_criterion_keys(monkeypatch):
    """Spec section 4 criterion 4 cannot be reported unless something calls this."""
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=n)
    lagged = np.roll(signal, 1)
    embedding = signal[:, None] * np.ones((1, 4))
    latent = np.column_stack([embedding, lagged[:, None] * np.ones((1, 4))])
    targets = np.column_stack([lagged] * 4) + 0.01 * rng.normal(size=(n, 4))

    calls = []

    def fake_gather(model, paths, backbone, device, context=5, horizon=45,
                    limit=20, seed=0):
        calls.append(list(paths))
        half = n // 2
        sl = slice(0, half) if len(calls) == 1 else slice(half, n)
        return {"latent": latent[sl], "embedding": np.zeros((sl.stop - sl.start, 2)),
                "encoder_embedding": embedding[sl], "targets": targets[sl]}

    monkeypatch.setattr(probe_module, "gather_probe_data", fake_gather)
    out = filtering_report(object(), ["train"], ["val"], None, torch.device("cpu"))

    assert set(out) == {"latent_r2", "embedding_r2", "latent_beats_embedding"}
    assert out["latent_beats_embedding"] is True
    assert calls == [["train"], ["val"]], "train and val must be gathered separately"


def test_filtering_report_does_not_gather_train_and_val_together(monkeypatch):
    """Fitting and scoring on the same rows would make the criterion vacuous.

    Measured on the leaking version, this comparison ran from R^2 -0.246 to
    0.9997: with more features than validation rows the probe simply memorises
    the split it is then scored on."""
    seen = []

    def fake_gather(model, paths, backbone, device, **kwargs):
        seen.append(tuple(paths))
        k = 200
        rng = np.random.default_rng(len(seen))
        return {"latent": rng.normal(size=(k, 6)),
                "embedding": rng.normal(size=(k, 4)),
                "encoder_embedding": rng.normal(size=(k, 4)),
                "targets": rng.normal(size=(k, 4))}

    monkeypatch.setattr(probe_module, "gather_probe_data", fake_gather)
    filtering_report(object(), ["a", "b"], ["c"], None, torch.device("cpu"))
    assert seen == [("a", "b"), ("c",)]


def test_filtering_report_probes_the_raw_encoder_embedding_not_the_predicted_one(
    tmp_path, monkeypatch
):
    """Spec section 3.4's reference is the ENCODER embedding at t.

    Made unambiguous by giving the head a different width from the encoder: the
    encoder emits 1 feature and the head 7, so the arrays actually handed to
    `filtering_comparison` say which of the two was compared. The companion
    test below shows why the distinction changes the VERDICT and not just the
    shapes."""
    paths = _write_episodes(tmp_path, [20, 25])
    captured = {}

    real_comparison = probe_module.filtering_comparison

    def recording_comparison(lat_tr, emb_tr, lat_va, emb_va, tgt_tr, tgt_va):
        captured.update(embedding_train=emb_tr, embedding_val=emb_va,
                        latent_train=lat_tr)
        return real_comparison(lat_tr, emb_tr, lat_va, emb_va, tgt_tr, tgt_va)

    monkeypatch.setattr(probe_module, "filtering_comparison", recording_comparison)
    filtering_report(_FakeModel(head_width=7), paths[:1], paths[1:], None,
                     torch.device("cpu"), context=2, horizon=3)

    assert captured["embedding_train"].shape[1] == 1, (
        "the filtering comparison was handed a 7-wide array -- that is the "
        "head's PREDICTED embedding, not the raw encoder embedding the gate "
        "criterion is defined against"
    )
    assert captured["embedding_val"].shape[1] == 1
    # And it really is the encoder's output: the tag encoder emits the frame
    # index verbatim, so the first training window reads 1, 2, 3, 4, 5.
    np.testing.assert_allclose(captured["embedding_train"][:5, 0],
                               np.arange(1, 6.0), atol=1e-4)
    # The two episodes have different lengths, so the row counts say which
    # split each argument came from: a train/val swap is visible right here.
    assert captured["embedding_train"].shape[0] == 20, "train got the val rows"
    assert captured["embedding_val"].shape[0] == 25, "val got the train rows"
    assert captured["latent_train"].shape[0] == 20


def test_filtering_report_verdict_is_false_when_the_latent_is_just_the_frame(
    tmp_path,
):
    """The behavioural half of the raw-vs-predicted question.

    `_PassThroughRSSM` makes the latent EXACTLY the current frame's encoder
    embedding -- `h` inert, no history whatsoever. The honest verdict is then
    `False`: the latent adds nothing over the frame's own encoding, which is
    precisely what this diagnostic exists to detect.

    Compared against the head's PREDICTED embedding instead, the same inert
    model reports `True`, because `_DeadHeads` throws the information away and
    the latent trivially beats it. That mutation flips the gate criterion from
    a correct failure to a false pass, and no shape assertion is involved --
    which is why the reference has to be the encoder embedding.
    """
    paths = _write_episodes(tmp_path, [20, 25])
    model = _FakeModel(rssm=_PassThroughRSSM())
    model.heads = _DeadHeads(width=3)

    out = filtering_report(model, paths[:1], paths[1:], None, torch.device("cpu"),
                           context=2, horizon=3)

    assert out["latent_r2"] == pytest.approx(out["embedding_r2"], abs=1e-9), (
        "the latent IS the encoder embedding here, so the two probes must be "
        "the same probe -- a different number means a different array was probed"
    )
    assert out["latent_beats_embedding"] is False, (
        "an inert `h` was reported as carrying history"
    )


def test_filtering_report_gathers_at_the_callers_context_and_horizon(monkeypatch):
    """The filtering probe must be measured at the window the rollout evaluates
    at, for the same reason `fit_probes` forwards them: a latent filtered from
    50 steps of context is a different distribution from one filtered from 5,
    and the whole claim is about how much history `h` accumulates."""
    seen = []

    def fake_gather(model, paths, backbone, device, context=5, horizon=45,
                    limit=20, seed=0):
        seen.append({"context": context, "horizon": horizon, "limit": limit,
                     "seed": seed})
        rng = np.random.default_rng(len(seen))
        return {"latent": rng.normal(size=(80, 3)),
                "embedding": rng.normal(size=(80, 3)),
                "encoder_embedding": rng.normal(size=(80, 3)),
                "targets": rng.normal(size=(80, 4))}

    monkeypatch.setattr(probe_module, "gather_probe_data", fake_gather)
    filtering_report(object(), ["a"], ["b"], None, torch.device("cpu"),
                     context=7, horizon=11, limit=3, seed=5)

    assert [c["context"] for c in seen] == [7, 7], "context was not forwarded"
    assert [c["horizon"] for c in seen] == [11, 11], "horizon was not forwarded"
    assert [c["limit"] for c in seen] == [3, 3], "limit was not forwarded"
    assert seen[0]["seed"] == 5, "the caller's seed must fix the training gather"
    assert seen[1]["seed"] != seen[0]["seed"], (
        "the validation gather must draw its own posterior samples, not replay "
        "the training split's"
    )


def test_filtering_report_defaults_are_the_spec_values():
    """`scripts/eval_rollout.py` passes context/horizon/seed but relies on the
    bare default for `limit`, and a future caller may rely on all of them. The
    window must be the rollout's own (5 + 45), or the criterion is reported at
    a filtering depth the study never evaluates at."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(filtering_report).parameters.items()
    }
    assert defaults["context"] == 5
    assert defaults["horizon"] == 45
    assert defaults["limit"] == 20
    assert defaults["seed"] == 0


def test_filtering_report_fits_on_the_train_rows_and_scores_on_the_val_rows(
    monkeypatch,
):
    """Gathering the two splits separately is not enough -- the TRAIN arrays
    have to be the ones handed to the fit.

    `filtering_comparison` takes six arrays and the train/val pairs are
    interchangeable at the type level, so passing `val[...]` in the train
    position gathers correctly and still fits each probe on the very rows it is
    then scored on. Measured, that mutation survived every other test in this
    file: `test_filtering_report_does_not_gather_train_and_val_together` only
    inspects the gather CALLS, and the fake data the key/verdict tests use is
    predictable enough that a leaked fit reaches the same verdict.

    Caught empirically. On fully unrelated features and targets, with more
    features (20) than validation rows (5), an honest fit-on-train probe cannot
    score well on the held-out rows, while a leaked fit-on-val probe memorises
    them: measured here at R^2 0.9997 against -0.246 -- the exact leak the
    equivalent guard on `filtering_comparison` itself pins one level down.
    """
    rng = np.random.default_rng(0)
    n_train, n_val, p = 200, 5, 20

    def fake_gather(model, paths, backbone, device, **kwargs):
        k = n_train if paths == ["train"] else n_val
        return {"latent": rng.normal(size=(k, p)),
                "embedding": rng.normal(size=(k, p)),
                "encoder_embedding": rng.normal(size=(k, p)),
                "targets": rng.normal(size=(k, 4))}

    monkeypatch.setattr(probe_module, "gather_probe_data", fake_gather)
    out = filtering_report(object(), ["train"], ["val"], None, torch.device("cpu"))

    assert out["latent_r2"] < 0.5, (
        f"latent_r2={out['latent_r2']:.4f} on pure noise -- the probe was fit "
        "on the rows it is scored on"
    )
    assert out["embedding_r2"] < 0.5, (
        f"embedding_r2={out['embedding_r2']:.4f} on pure noise -- the probe was "
        "fit on the rows it is scored on"
    )
