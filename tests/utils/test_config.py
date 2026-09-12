import dataclasses

import pytest

from mbfps.utils.config import (
    ARMS,
    KINDS,
    Config,
    EncoderConfig,
    TrainConfig,
    get_config,
)


def test_the_studys_three_arms_are_registered_and_cnn_is_not_one_of_them():
    """M3c retired the end-to-end pixel arm from the study (its embedding
    target was born collapsed, `embedding_loss` 0.0008 at step 0, and its
    dynamics prior never trained) and put the frozen M2 autoencoder in its
    place. Asserted as the whole tuple, in order: the arm order is the row
    order of every report table."""
    assert ARMS == ("pixel_ae", "frozen_ssl", "random_vit")
    assert "cnn" not in ARMS


def test_kinds_is_the_arms_plus_the_cnn_encoder_m2_still_builds():
    """Two tuples because two milestones read them: M2's autoencoder scripts
    still construct `CNNEncoder` under the name `cnn`, and M3's tools must not
    be able to select it. `cnn` first, so a reader sees at once which member
    is the odd one out."""
    assert KINDS == ("cnn",) + ARMS
    assert KINDS == ("cnn", "pixel_ae", "frozen_ssl", "random_vit")


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_builds_a_config(kind):
    """Over KINDS, not ARMS: `get_config("cnn")` must keep working for M2 and
    for the M3b artefacts that pin it, and `get_config("pixel_ae")` must work
    for the new study. Parametrised so the name of the kind that stops
    building is in the failure, not lost in a loop."""
    cfg = get_config(kind)
    assert isinstance(cfg, Config)
    assert cfg.arm == kind
    assert cfg.encoder.kind == kind


def test_get_config_still_builds_cnn_even_though_the_study_does_not_run_it():
    """The retired arm is buildable, not selectable. This is the property the
    shipped M3b checkpoints (`runs/m3_study/world_model_cnn_seed0.pt`) and
    M2's autoencoder tests depend on. Named explicitly rather than left to
    the KINDS parametrisation above: a KINDS that lost `cnn` would shrink
    that test, not fail it."""
    assert "cnn" not in ARMS
    assert get_config("cnn").encoder.kind == "cnn"


def test_unknown_arm_rejected():
    with pytest.raises(KeyError, match="unknown arm 'nope'"):
        get_config("nope")


def test_unknown_arm_error_lists_every_kind():
    """The message lists KINDS -- all four -- so an operator who typed the
    retired name sees it is still a kind, and one who typed a study arm
    wrong sees the study's three."""
    with pytest.raises(KeyError) as excinfo:
        get_config("nope")
    message = str(excinfo.value)
    for kind in KINDS:
        assert kind in message, f"{kind!r} missing from {message}"
    assert str(list(KINDS)) in message


def test_all_arms_share_identical_training_settings():
    """The study's validity depends on this. Arms may differ ONLY in encoder."""
    trains = {arm: get_config(arm).train for arm in ARMS}
    first = trains["pixel_ae"]          # was trains["cnn"]
    for arm, train in trains.items():
        assert train == first, f"arm {arm!r} has different training settings"


def test_all_arms_share_identical_embed_dim():
    """Embedding width is controlled; only the representation differs."""
    dims = {get_config(arm).encoder.embed_dim for arm in ARMS}
    assert dims == {2048}


def test_arms_differ_only_in_encoder_kind():
    configs = {arm: get_config(arm) for arm in ARMS}
    encoders = {arm: dataclasses.asdict(c.encoder) for arm, c in configs.items()}
    baseline = dict(encoders["pixel_ae"])   # was encoders["cnn"]
    for arm, enc in encoders.items():
        differing = {k for k in enc if enc[k] != baseline[k]}
        assert differing <= {"kind"}, f"arm {arm!r} differs beyond kind: {differing}"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.train, "batch_size", 32),
        lambda c: setattr(c.encoder, "embed_dim", 1024),
        lambda c: setattr(c, "arm", "other"),
    ],
    ids=["train", "encoder", "config"],
)
def test_every_config_layer_is_frozen(mutate):
    """All three layers must be immutable.

    Checking only TrainConfig leaves a real hole: a mutable EncoderConfig
    would let one arm's embedding width drift at runtime, silently breaking
    the comparison the whole study rests on. Verified by mutation -- removing
    frozen=True from EncoderConfig passed the single-layer version of this
    test.
    """
    with pytest.raises(dataclasses.FrozenInstanceError):
        mutate(get_config("cnn"))


def test_overrides_apply_to_train_settings():
    cfg = get_config("cnn", steps=5, batch_size=2)
    assert cfg.train.steps == 5
    assert cfg.train.batch_size == 2


def test_override_of_unknown_field_rejected():
    with pytest.raises(TypeError):
        get_config("cnn", not_a_field=1)


def test_seq_len_default_matches_the_dataset():
    """119 of 122 episodes support a 64-step window; a larger default would
    silently discard usable episodes."""
    assert get_config("cnn").train.seq_len == 64


# --- patch_dim is not an arm-level setting ---------------------------------
# It described the ViT backbones' 384-wide rows and was consumed in exactly two
# lines of BottleneckEncoder. A third backbone with a different width would
# have been built against 384 with no error. The width now lives with the
# backbone (`mbfps.data.features.BACKBONE_GEOMETRY`), and a dead field left
# here is how a future arm gets built against the wrong number.


def test_encoder_config_has_no_patch_dim_field():
    names = {f.name for f in dataclasses.fields(EncoderConfig)}
    assert "patch_dim" not in names, names


def test_encoder_config_rejects_patch_dim_at_construction():
    """The field must be GONE, not merely unused: a config that silently
    accepts `patch_dim=512` is the exact failure the removal exists to stop."""
    with pytest.raises(TypeError, match="patch_dim"):
        EncoderConfig(kind="frozen_ssl", patch_dim=512)


def test_encoder_config_keeps_the_shared_bottleneck_fields():
    """The removal must take ONLY patch_dim; the two shared numbers the
    per-backbone guard `n_patches * bottleneck_dim == embed_dim` reads stay."""
    c = EncoderConfig(kind="frozen_ssl")
    assert c.embed_dim == 2048
    assert c.bottleneck_dim == 32
    assert c.cnn_depth == 32
    assert c.standardise_features is True
