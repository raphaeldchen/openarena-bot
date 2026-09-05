import numpy as np
import pytest

from mbfps.eval.probe import (
    PROBE_KEYS,
    RIDGES,
    _mean_r2,
    angle_error_degrees,
    apply_probe,
    fit_probe,
    position_error,
    probe_targets,
)

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def probe_r2(probe: dict, latents: np.ndarray, targets: np.ndarray) -> float:
    """Per-column R^2, averaged -- independent of the module's own _mean_r2 so
    a mutation to that private helper doesn't also corrupt the check on it."""
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
    assert probe_r2(selected, xv, yv) >= probe_r2(fixed, xv, yv)
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
    on this data is measured independently below (via `probe_r2`, not the
    module's own `_mean_r2`) and is far from RIDGES[0]."""
    x, y, xv, yv = _overparameterised_case()

    independently_scored = {
        ridge: probe_r2(fit_probe(x, y, ridge=ridge), xv, yv) for ridge in RIDGES
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

    train_scored = {ridge: probe_r2(fit_probe(x, y, ridge=ridge), x, y) for ridge in RIDGES}
    val_scored = {ridge: probe_r2(fit_probe(x, y, ridge=ridge), xv, yv) for ridge in RIDGES}
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
