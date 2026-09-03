import pytest
import torch

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.decoders import PixelDecoder, reconstruction_loss


def test_output_shape_matches_observation_shape():
    dec = PixelDecoder(in_dim=2048)
    assert dec(torch.randn(4, 2048)).shape == (4, *OBS_SHAPE)


def test_output_is_float32_in_unit_range():
    dec = PixelDecoder(in_dim=2048)
    out = dec(torch.randn(4, 2048) * 10.0)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_in_dim_is_configurable_for_later_recurrent_use():
    """M3 feeds a concatenated recurrent state, not a bare embedding."""
    dec = PixelDecoder(in_dim=1536)
    assert dec(torch.randn(2, 1536)).shape == (2, *OBS_SHAPE)


def test_wrong_input_width_raises():
    dec = PixelDecoder(in_dim=2048)
    with pytest.raises(RuntimeError, match="2048"):
        dec(torch.randn(2, 1024))


def test_gradients_flow_to_every_parameter():
    dec = PixelDecoder(in_dim=2048)
    dec(torch.randn(2, 2048)).square().mean().backward()
    missing = [n for n, p in dec.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_reconstruction_loss_is_zero_for_a_perfect_match():
    target = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    perfect = target.to(torch.float32) / 255.0
    assert reconstruction_loss(perfect, target).item() == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_is_positive_for_a_mismatch():
    target = torch.zeros((2, *OBS_SHAPE), dtype=torch.uint8)
    pred = torch.ones((2, *OBS_SHAPE), dtype=torch.float32)
    assert reconstruction_loss(pred, target).item() == pytest.approx(1.0, abs=1e-6)


def test_reconstruction_loss_accepts_uint8_targets_directly():
    """The loader hands out uint8; the loss owns the conversion."""
    target = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    pred = torch.rand((2, *OBS_SHAPE))
    assert torch.isfinite(reconstruction_loss(pred, target))


def test_reconstruction_loss_rejects_an_unnormalised_float_target():
    """A float target in [0, 255] is a units bug, not a valid input.

    Guarding on dtype alone let this through silently, inflating the loss by
    ~255^2 -- which reads as a diverging model rather than a scaling mistake.
    """
    pred = torch.rand((2, *OBS_SHAPE))
    target = torch.rand((2, *OBS_SHAPE)) * 255.0
    with pytest.raises(ValueError, match="expected \\[0, 1\\]"):
        reconstruction_loss(pred, target)


def test_reconstruction_loss_accepts_a_normalised_float_target():
    """The complement: a correctly-scaled float target must still work."""
    target = torch.rand((2, *OBS_SHAPE))
    assert reconstruction_loss(target, target).item() == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_accepts_an_empty_target():
    """The max() guard must not crash on a zero-element tensor."""
    empty = torch.zeros((0, *OBS_SHAPE))
    assert torch.isfinite(reconstruction_loss(empty, empty))


def test_recorded_parameter_count():
    """Pins the measured count so an architecture drift is visible."""
    n = sum(p.numel() for p in PixelDecoder(in_dim=2048).parameters())
    assert n == 26_392_547


# --- Loss value tests with a discriminating error magnitude ------------------
# Every existing value assertion uses a per-pixel error of exactly 0 or exactly
# 1, and 0^2 == 0, 1^2 == 1 -- so MSE, MAE and any other p-norm agree on those
# inputs. Swapping .square() for .abs() passed the whole suite. These use an
# error where the norms disagree.


def test_reconstruction_loss_is_mean_squared_not_mean_absolute():
    """0.5^2 = 0.25 != 0.5, so this separates MSE from MAE."""
    pred = torch.full((2, 4, 4, 3), 0.5)
    target = torch.zeros((2, 4, 4, 3))
    loss = reconstruction_loss(pred, target)
    assert loss.item() == pytest.approx(0.25, abs=1e-6)


def test_reconstruction_loss_scales_quadratically_with_error():
    """Doubling the error must quadruple the loss; a linear norm would not."""
    target = torch.zeros((2, 4, 4, 3))
    small = reconstruction_loss(torch.full((2, 4, 4, 3), 0.2), target).item()
    large = reconstruction_loss(torch.full((2, 4, 4, 3), 0.4), target).item()
    assert large == pytest.approx(4.0 * small, rel=1e-5)


def test_reconstruction_loss_normalises_uint8_targets_by_255():
    """Pins the divisor: /256 or /128 would shift this away from the exact value."""
    pred = torch.zeros((1, 2, 2, 3))
    target = torch.full((1, 2, 2, 3), 51, dtype=torch.uint8)  # 51/255 == 0.2
    loss = reconstruction_loss(pred, target)
    assert loss.item() == pytest.approx(0.04, abs=1e-6)
