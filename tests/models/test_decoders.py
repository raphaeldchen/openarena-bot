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
    with pytest.raises(RuntimeError):
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


def test_recorded_parameter_count():
    """Pins the measured count so an architecture drift is visible."""
    n = sum(p.numel() for p in PixelDecoder(in_dim=2048).parameters())
    assert n == 26_392_547
