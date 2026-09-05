from dataclasses import replace

import pytest
import torch

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.utils.config import get_config
from mbfps.models.encoders import (
    BottleneckEncoder,
    CNNEncoder,
    build_encoder,
    encoder_backbone,
    encoder_input_kind,
)
from mbfps.utils.config import ARMS, EncoderConfig


def cfg(kind: str) -> EncoderConfig:
    return EncoderConfig(kind=kind)


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_cnn_encoder_output_shape():
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.randint(0, 256, (4, *OBS_SHAPE), dtype=torch.uint8)
    assert enc(obs).shape == (4, 2048)


def test_cnn_encoder_output_is_float32():
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    assert enc(obs).dtype == torch.float32


def test_cnn_encoder_accepts_uint8_without_manual_conversion():
    """The loader hands out uint8; the encoder owns normalisation."""
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.full((2, *OBS_SHAPE), 255, dtype=torch.uint8)
    assert torch.isfinite(enc(obs)).all()


def test_cnn_encoder_normalises_and_reorders_its_input(monkeypatch):
    """Pins the uint8 -> centred-unit-range conversion and the NCHW reorder.

    Shape/dtype/finiteness assertions cannot catch a missing /255: SiLU and
    Linear stay finite at any input scale, so a 255x units bug would surface
    only as a model that refuses to converge. Verified by mutation -- removing
    the normalisation left all other encoder tests green.
    """
    enc = CNNEncoder(cfg("cnn"))
    seen: dict[str, torch.Tensor] = {}
    original = enc.conv.forward

    def spy(x):
        seen["x"] = x.detach().clone()
        return original(x)

    monkeypatch.setattr(enc.conv, "forward", spy)

    enc(torch.full((2, *OBS_SHAPE), 255, dtype=torch.uint8))
    assert seen["x"].shape == (2, 3, 112, 112), "input must be NCHW float"
    assert seen["x"].dtype == torch.float32
    assert seen["x"].max().item() == pytest.approx(0.5, abs=1e-6), (
        "uint8 255 must map to +0.5"
    )

    enc(torch.zeros((2, *OBS_SHAPE), dtype=torch.uint8))
    assert seen["x"].min().item() == pytest.approx(-0.5, abs=1e-6), (
        "uint8 0 must map to -0.5"
    )

    enc(torch.full((2, *OBS_SHAPE), 128, dtype=torch.uint8))
    assert seen["x"].mean().item() == pytest.approx(128 / 255 - 0.5, abs=1e-6)


def test_bottleneck_encoder_output_shape():
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    feats = torch.randn(4, 64, 384)
    assert enc(feats).shape == (4, 2048)


def test_bottleneck_flattens_patch_grid_to_embed_dim():
    """64 patches x 32 bottleneck dims = 2048, matching the CNN arm exactly."""
    c = cfg("frozen_ssl")
    assert 64 * c.bottleneck_dim == c.embed_dim


@pytest.mark.parametrize("arm", ARMS)
def test_build_encoder_returns_something_for_every_arm(arm):
    assert build_encoder(cfg(arm)) is not None


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_emits_the_same_embedding_width(arm):
    c = cfg(arm)
    enc = build_encoder(c)
    if encoder_input_kind(c) == "obs":
        x = torch.randint(0, 256, (3, *OBS_SHAPE), dtype=torch.uint8)
    else:
        x = torch.randn(3, 64, 384)
    assert enc(x).shape == (3, 2048)


def test_input_kind_is_obs_for_cnn_and_features_for_ssl_arms():
    assert encoder_input_kind(cfg("cnn")) == "obs"
    assert encoder_input_kind(cfg("frozen_ssl")) == "features"
    assert encoder_input_kind(cfg("random_vit")) == "features"


def test_unknown_arm_rejected():
    with pytest.raises(KeyError, match="unknown encoder kind 'nope'"):
        build_encoder(EncoderConfig(kind="nope"))


def test_each_arm_maps_to_its_own_backbone():
    """The control arm must not read the treatment arm's cache.

    Arm name and backbone name are not the same string for `frozen_ssl`, so a
    naive identity mapping would send it to a cache that does not exist.
    """
    assert encoder_backbone(cfg("cnn")) is None
    assert encoder_backbone(cfg("frozen_ssl")) == "dinov2"
    assert encoder_backbone(cfg("random_vit")) == "random_vit"


def test_feature_arms_map_to_distinct_backbones():
    """If these collided, arms 2 and 3 would be the same experiment."""
    assert encoder_backbone(cfg("frozen_ssl")) != encoder_backbone(cfg("random_vit"))


@pytest.mark.parametrize("arm", ["frozen_ssl", "random_vit"])
def test_ssl_arms_build_the_identical_module(arm):
    """Arm 3 is Arm 2 with different cached inputs -- nothing else.

    Driven through `build_encoder` rather than the class constructor, so a
    broken routing table is caught here rather than incidentally by a shape
    assertion elsewhere. Compares module type and state-dict keys as well as
    shapes, because a per-arm weight-init branch would leave shapes identical
    while changing behaviour.
    """
    reference = build_encoder(cfg("frozen_ssl"))
    built = build_encoder(cfg(arm))

    assert type(built) is type(reference), "arms got different module types"
    assert n_params(built) == n_params(reference)
    assert [tuple(p.shape) for p in built.parameters()] == [
        tuple(p.shape) for p in reference.parameters()
    ]
    assert built.state_dict().keys() == reference.state_dict().keys()


def test_ssl_arms_build_value_identical_modules():
    """Structural checks cannot see a per-arm weight-init branch.

    Under the same seed both arms must produce bit-identical parameters. A
    branch like `if cfg.kind == "random_vit": nn.init.zeros_(...)` leaves every
    type, shape and state-dict key identical while changing behaviour -- and
    would confound the very comparison the study is built on. Verified by
    mutation: the structural parity test passes under exactly that change.
    """
    torch.manual_seed(0)
    reference = build_encoder(cfg("frozen_ssl"))
    torch.manual_seed(0)
    control = build_encoder(cfg("random_vit"))

    ref_params = dict(reference.named_parameters())
    ctl_params = dict(control.named_parameters())
    assert ref_params.keys() == ctl_params.keys()
    for name, ref in ref_params.items():
        assert torch.equal(ref, ctl_params[name]), (
            f"parameter {name!r} differs between the treatment and control arms; "
            "their encoders must be identical apart from their cached inputs"
        )


def test_recorded_parameter_counts():
    """Pins the measured counts so a silent architecture change is visible.

    The spec's own estimates (~4M encoder+decoder, ~0.2M bottleneck) are wrong;
    these are the measured values for the architecture it describes.
    """
    assert n_params(CNNEncoder(cfg("cnn"))) == 26_382_304
    assert n_params(BottleneckEncoder(cfg("frozen_ssl"))) == 12_320


def test_cnn_gradients_flow_to_every_parameter():
    enc = CNNEncoder(cfg("cnn"))
    obs = torch.randint(0, 256, (2, *OBS_SHAPE), dtype=torch.uint8)
    enc(obs).square().mean().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_bottleneck_gradients_flow_to_every_parameter():
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    enc(torch.randn(2, 64, 384)).square().mean().backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


# --- Feature standardisation ------------------------------------------------
# The two caches arrive at very different scales (measured on disk: DINOv2 std
# 2.3559, random_vit std 1.0000). Feeding both into a bare nn.Linear at one
# learning rate confounds the treatment/control contrast with optimisation
# conditioning, so the bottleneck normalises its input. It must do so without
# adding parameters and without differing between the two feature arms.


def _scaled_features(scale: float, shift: float = 0.0) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 64, 384) * scale + shift


def test_standardisation_equalises_inputs_arriving_at_different_scales():
    """The whole point: two caches at 2.36x different scale must arrive alike."""
    encoder = build_encoder(get_config("frozen_ssl", device="cpu").encoder)
    wide = encoder.norm(_scaled_features(scale=2.3559, shift=0.0566))
    narrow = encoder.norm(_scaled_features(scale=1.0))
    assert wide.std().item() == pytest.approx(1.0, abs=0.02)
    assert narrow.std().item() == pytest.approx(1.0, abs=0.02)
    assert wide.mean().item() == pytest.approx(0.0, abs=0.02)


def test_standardisation_adds_no_parameters():
    """It must not change the trainable budget the arms are pinned to."""
    encoder = build_encoder(get_config("frozen_ssl", device="cpu").encoder)
    assert sum(p.numel() for p in encoder.parameters()) == 12_320
    assert sum(p.numel() for p in encoder.norm.parameters()) == 0


def test_both_feature_arms_standardise_identically():
    """Arm parity: normalisation must not be one of the things that differs."""
    features = _scaled_features(scale=2.0, shift=0.3)
    outputs = {}
    for arm in ("frozen_ssl", "random_vit"):
        encoder = build_encoder(get_config(arm, device="cpu").encoder)
        outputs[arm] = encoder.norm(features)
    assert torch.equal(outputs["frozen_ssl"], outputs["random_vit"])


def test_standardisation_actually_changes_the_encoder_output():
    """Guards against a normalisation that is silently a no-op."""
    cfg_on = get_config("frozen_ssl", device="cpu").encoder
    cfg_off = replace(cfg_on, standardise_features=False)
    features = _scaled_features(scale=2.3559, shift=0.0566)
    torch.manual_seed(0)
    on = build_encoder(cfg_on)
    torch.manual_seed(0)
    off = build_encoder(cfg_off)
    # Same weights, different input conditioning -> different output.
    assert torch.equal(on.bottleneck.weight, off.bottleneck.weight)
    assert not torch.allclose(on(features), off(features))


def test_standardisation_can_be_disabled_to_reproduce_the_earlier_runs():
    cfg = replace(get_config("frozen_ssl", device="cpu").encoder, standardise_features=False)
    encoder = build_encoder(cfg)
    features = _scaled_features(scale=2.3559)
    assert torch.equal(encoder.norm(features), features)
