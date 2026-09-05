import numpy as np
import pytest

from mbfps.eval.probe import (
    PROBE_KEYS,
    RIDGES,
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
    # atol=1e-7 bounds the implementation's known +1e-8 stability epsilon on
    # `scale` (std alone, without it, would make rtol=1e-6 fail deterministically
    # on this seed's small-scale column: 1e-8 is a ~1e-5 relative perturbation
    # of a std of ~9.7e-4).
    np.testing.assert_allclose(probe["scale"], x.std(0), rtol=1e-6, atol=1e-7)
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
