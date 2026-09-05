import numpy as np
import pytest

from mbfps.eval.probe import (
    PROBE_KEYS,
    RIDGES,
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
