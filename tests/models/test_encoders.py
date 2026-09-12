from dataclasses import replace

import pytest
import torch

import mbfps.models.encoders as encoders  # Task 1's `test_no_second_source_of_truth_for_the_patch_count` reads it
from mbfps.data.features import BACKBONE_GEOMETRY
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.utils.config import get_config
from mbfps.models.encoders import (
    BottleneckEncoder,
    CNNEncoder,
    build_encoder,
    encoder_backbone,
    encoder_input_kind,
)
from mbfps.utils.config import ARMS, KINDS, EncoderConfig


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
    """64 patches x 32 bottleneck dims = 2048, matching the CNN arm exactly.

    The patch count is the BACKBONE's, read from the registry, not a number
    this test knows: a backbone with a different grid must satisfy the same
    identity with its own row count.
    """
    c = cfg("frozen_ssl")
    n_patches, _ = BACKBONE_GEOMETRY[encoder_backbone(c)]
    assert n_patches == 64
    assert n_patches * c.bottleneck_dim == c.embed_dim


def test_the_kinds_under_test_are_the_four_the_registry_knows():
    """L7 guard for every test below that parametrises over KINDS or ARMS: a
    tuple that silently lost a member would shrink those tests rather than
    fail them. The names are spelled out once, here."""
    assert KINDS == ("cnn", "pixel_ae", "frozen_ssl", "random_vit")
    assert ARMS == ("pixel_ae", "frozen_ssl", "random_vit")


@pytest.mark.parametrize("kind", KINDS)
def test_build_encoder_routes_every_kind_to_its_class(kind):
    """Over KINDS, not ARMS: `cnn` must stay constructible for M2 and for the
    shipped M3b checkpoints, and `pixel_ae` must route to the SAME bottleneck
    class the ViT arms use -- the study's arms are one pipeline."""
    built = build_encoder(cfg(kind))
    expected = CNNEncoder if kind == "cnn" else BottleneckEncoder
    assert type(built) is expected, (kind, type(built))


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_emits_the_same_embedding_width(kind):
    """The feature input is shaped from the backbone registry, not a literal
    (64, 384): `pixel_ae`'s rows are 32 wide, and a hardcoded width would
    make this test the one place the study's own geometry is not read."""
    c = cfg(kind)
    enc = build_encoder(c)
    if encoder_input_kind(c) == "obs":
        x = torch.randint(0, 256, (3, *OBS_SHAPE), dtype=torch.uint8)
    else:
        n_patches, patch_dim = BACKBONE_GEOMETRY[encoder_backbone(c)]
        x = torch.randn(3, n_patches, patch_dim)
    assert enc(x).shape == (3, 2048)


def test_input_kind_is_obs_for_cnn_and_features_for_every_study_arm():
    """Every study arm reads cached features. Written out by name AND over
    the tuple: the tuple form is what the study relies on (`run_job` picks
    the cache from this), the named form is what stops a shrunken ARMS from
    passing it vacuously."""
    assert encoder_input_kind(cfg("cnn")) == "obs"
    assert encoder_input_kind(cfg("pixel_ae")) == "features"
    assert encoder_input_kind(cfg("frozen_ssl")) == "features"
    assert encoder_input_kind(cfg("random_vit")) == "features"
    assert {encoder_input_kind(cfg(arm)) for arm in ARMS} == {"features"}
    assert "cnn" not in ARMS, "the one obs-kind encoder is not a study arm"


def test_unknown_arm_rejected():
    with pytest.raises(KeyError, match="unknown encoder kind 'nope'"):
        build_encoder(EncoderConfig(kind="nope"))


def test_each_arm_maps_to_its_own_backbone():
    """The control arm must not read the treatment arm's cache.

    Arm name and backbone name are not the same string for `frozen_ssl`, so a
    naive identity mapping would send it to a cache that does not exist.
    """
    assert encoder_backbone(cfg("cnn")) is None
    assert encoder_backbone(cfg("pixel_ae")) == "pixel_ae"
    assert encoder_backbone(cfg("frozen_ssl")) == "dinov2"
    assert encoder_backbone(cfg("random_vit")) == "random_vit"


def test_study_arms_map_to_three_distinct_backbones():
    """If any two collided, two arms would be the same experiment."""
    backbones = [encoder_backbone(cfg(arm)) for arm in ARMS]
    assert len(set(backbones)) == 3 == len(ARMS), backbones
    assert None not in backbones, "a study arm that reads pixels has no cache"


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
    assert n_params(BottleneckEncoder(cfg("pixel_ae"))) == 1_056


def test_pixel_ae_builds_the_same_bottleneck_class_over_32_wide_rows():
    """Spec section 2.2's one stated asymmetry: same class, same per-row
    treatment, same state-dict keys as the ViT arms, but `Linear(32 -> 32)`
    (1,056 parameters) against `Linear(384 -> 32)` (12,320), because the M2
    autoencoder's `project` layer already reduced each row to 32. Pinned so
    the asymmetry stays the one the design recorded and not a second, silent
    one -- and driven through `build_encoder`, so a routing table that sent
    `pixel_ae` to the ViT geometry fails here rather than at step 0."""
    built = build_encoder(cfg("pixel_ae"))
    reference = build_encoder(cfg("frozen_ssl"))
    assert type(built) is BottleneckEncoder
    assert built.state_dict().keys() == reference.state_dict().keys()
    assert isinstance(built.bottleneck, torch.nn.Linear)
    assert (built.bottleneck.in_features, built.bottleneck.out_features) == (32, 32)
    assert (reference.bottleneck.in_features, reference.bottleneck.out_features) == (384, 32)
    assert isinstance(built.norm, torch.nn.LayerNorm)
    assert tuple(built.norm.normalized_shape) == (32,)
    assert n_params(built.norm) == 0
    assert built(torch.randn(3, 64, 32)).shape == (3, 2048)


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


# --- Backbone geometry ------------------------------------------------------
# `(n_patches, patch_dim)` is a property of the frozen backbone whose cache the
# arm reads, not a choice the study makes per arm. Before this block the
# encoder carried a module constant `_N_PATCHES = 64` and `EncoderConfig` a
# shared `patch_dim = 384`, both describing the ViT backbones' output. A third
# backbone with a different row width would have been built against 384 with
# no error until the first matmul. The geometry now lives in ONE registry,
# `mbfps.data.features.BACKBONE_GEOMETRY`, and the encoder reads it.


def test_bottleneck_takes_its_geometry_from_the_backbone_registry(monkeypatch):
    """Both numbers must come from the registry -- neither may be hardcoded.

    Both registered backbones today are (64, 384), so building the real arms
    cannot tell a registry read from a literal 64 or 384 (fixture coincidence).
    Rebinding one backbone's entry to a geometry that matches NEITHER constant
    is what makes a hardcoded value fail: 128 rows of 16 still multiply out to
    2048 with `bottleneck_dim=16`, so the only way this build and forward pass
    succeed is if the encoder read the registry for both values.
    """
    monkeypatch.setitem(BACKBONE_GEOMETRY, "random_vit", (128, 16))
    enc = BottleneckEncoder(EncoderConfig(kind="random_vit", bottleneck_dim=16))
    assert enc.bottleneck.in_features == 16, "patch_dim must come from the registry"
    assert enc.bottleneck.out_features == 16
    assert tuple(enc.norm.normalized_shape) == (16,), (
        "the LayerNorm must be sized by the registry's patch_dim too"
    )
    assert enc(torch.randn(2, 128, 16)).shape == (2, 2048), (
        "n_patches must come from the registry"
    )


def test_bottleneck_guard_uses_the_registry_patch_count(monkeypatch):
    """`n_patches * bottleneck_dim == embed_dim` runs per backbone.

    With 128 rows the default `bottleneck_dim=32` gives 4096, not 2048, and
    the guard must say so. A guard still multiplying a literal 64 would accept
    this config and build an encoder that emits the wrong width.
    """
    monkeypatch.setitem(BACKBONE_GEOMETRY, "random_vit", (128, 16))
    with pytest.raises(ValueError, match=r"128 patches x bottleneck_dim 32"):
        BottleneckEncoder(cfg("random_vit"))


def test_bottleneck_guard_names_the_backbone_geometry():
    """The unchanged real-arm case: 64 x 33 != 2048, and the message says so."""
    with pytest.raises(ValueError, match=r"64 patches x bottleneck_dim 33.*embed_dim 2048"):
        BottleneckEncoder(EncoderConfig(kind="frozen_ssl", bottleneck_dim=33))


@pytest.mark.parametrize("arm", ["frozen_ssl", "random_vit"])
def test_bottleneck_forward_rejects_the_wrong_row_count(arm):
    """A cache with the wrong number of rows must not silently reshape.

    This is the dangerous case: `Linear(384, 32)` happily consumes
    `(N, 32, 384)` and emits `(N, 1024)`, which is the wrong embedding width
    and would only surface as a shape error deep inside the RSSM. The
    encoder must refuse at its own boundary and name what it expected.
    """
    enc = build_encoder(cfg(arm))
    backbone = encoder_backbone(cfg(arm))
    with pytest.raises(ValueError) as excinfo:
        enc(torch.randn(2, 32, 384))
    message = str(excinfo.value)
    assert backbone in message, "the error must name the backbone"
    assert "(64, 384)" in message, "the error must name the expected geometry"
    assert "(32, 384)" in message, "the error must name what it got"


def test_bottleneck_forward_rejects_the_wrong_row_width():
    """Wrong width would be a matmul error anyway; it must be OUR error."""
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    with pytest.raises(ValueError, match=r"dinov2.*\(64, 384\).*\(64, 32\)"):
        enc(torch.randn(2, 64, 32))


def test_bottleneck_forward_rejects_a_flat_input():
    """`(N, 24576)` is the right number of values in the wrong shape."""
    enc = BottleneckEncoder(cfg("frozen_ssl"))
    with pytest.raises(ValueError, match=r"dinov2"):
        enc(torch.randn(2, 64 * 384))


def test_bottleneck_refuses_a_kind_that_reads_pixels():
    """`cnn` has no backbone and therefore no geometry to read."""
    with pytest.raises(ValueError, match=r"'cnn'.*pixels"):
        BottleneckEncoder(cfg("cnn"))


def test_no_second_source_of_truth_for_the_patch_count():
    """Every reader goes through the registry, so the module constant is gone.

    A leftover `_N_PATCHES` is how a future reader bypasses the dict. (The
    matching removal of `EncoderConfig.patch_dim` is guarded in
    tests/utils/test_config.py.)
    """
    assert not hasattr(encoders, "_N_PATCHES")


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
