import re
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.data.episode import Episode, save_episode
import mbfps.data.features as features  # Task 1's `test_the_old_module_constants_are_gone` reads it
from mbfps.data.features import (
    BACKBONE_GEOMETRY,
    BACKBONES,
    PIXEL_AE_CHECKPOINT,
    FeatureExtractor,
    build_backbone,
    cache_episode_features,
)
from mbfps.envs.protocol import OBS_SHAPE

pytestmark = pytest.mark.slow  # downloads ~88MB on first run


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(device="cpu")


# --- Backbone geometry ------------------------------------------------------
# `(n_patches, patch_dim)` belongs to the backbone, not to the study. The two
# ViT backbones share 112 / 14 = 8, an 8x8 = 64 patch grid, at DINOv2-small's
# hidden size of 384. Both numbers used to be module constants that every
# reader -- the encoder, the cache-size estimate, these tests -- copied; a
# third backbone with a different width would have been built against 384
# with no error until the first matmul. There is now one registry.


def test_backbone_geometry_is_the_literal_registry():
    """Literal, not derived: geometry is what the loader and the bottleneck
    size themselves by, so a typo here is a matmul error at step 0 (or, worse,
    a cache that loads and trains on the wrong width)."""
    assert BACKBONE_GEOMETRY == {
        "dinov2": (64, 384),
        "random_vit": (64, 384),
        "pixel_ae": (64, 32),
    }


def test_every_backbone_has_a_geometry_and_nothing_else_does():
    assert set(BACKBONE_GEOMETRY) == set(BACKBONES)


def test_the_old_module_constants_are_gone():
    """Every reader goes through the dict. A leftover `N_PATCHES` or
    `FEATURE_DIM` is a second source of truth that a future reader will copy."""
    assert not hasattr(features, "N_PATCHES")
    assert not hasattr(features, "FEATURE_DIM")


def test_encode_checks_its_output_against_the_registry(monkeypatch):
    """`encode`'s patch-count check must read the registry, not a literal 64.

    DINOv2 emits 64 patches, so on the real registry a hardcoded 64 and a
    registry read are indistinguishable. Rebinding the entry to a count the
    backbone cannot produce forces the difference: only a registry read
    raises. Built fresh rather than via the module fixture so the patched
    registry is what the extractor sees at encode time.
    """
    extractor = FeatureExtractor(device="cpu")
    frames = np.zeros((1, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).shape == (1, 64, 384)  # sanity: unpatched passes
    monkeypatch.setitem(BACKBONE_GEOMETRY, "dinov2", (63, 384))
    with pytest.raises(ValueError, match=r"dinov2.*\(63, 384\).*\(64, 384\)"):
        extractor.encode(frames)


def test_encode_output_shape(extractor):
    frames = np.random.randint(0, 256, (3, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).shape == (3, *BACKBONE_GEOMETRY[extractor.backbone])


def test_encode_output_dtype_is_float16(extractor):
    frames = np.random.randint(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).dtype == np.float16


def test_encode_is_byte_identical_on_repeat(extractor):
    """The cache is only sound if the frozen backbone is deterministic."""
    frames = np.random.randint(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    a = extractor.encode(frames)
    b = extractor.encode(frames)
    assert np.array_equal(a, b)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_encode_is_byte_identical_on_repeat_on_mps():
    """Collection runs on MPS by default, so CPU repeatability is not enough."""
    mps_extractor = FeatureExtractor(device="mps")
    frames = np.random.default_rng(1).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert np.array_equal(mps_extractor.encode(frames), mps_extractor.encode(frames))


def test_encode_distinguishes_different_frames(extractor):
    frames = np.stack(
        [
            np.zeros(OBS_SHAPE, dtype=np.uint8),
            np.full(OBS_SHAPE, 255, dtype=np.uint8),
        ]
    )
    out = extractor.encode(frames)
    assert not np.allclose(out[0].astype(np.float32), out[1].astype(np.float32))


def test_encode_rejects_wrong_shape(extractor):
    with pytest.raises(ValueError, match="expected frames of shape"):
        extractor.encode(np.zeros((2, 64, 64, 3), dtype=np.uint8))


def _encode_preprocessed(extractor, x):
    """Run the frozen backbone on already-preprocessed NHWC float32 input."""
    import torch

    tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(extractor.device)
    with torch.no_grad():
        out = extractor.model(pixel_values=tensor).last_hidden_state
    return out[:, 1:, :].cpu().numpy().astype(np.float32)


def test_encode_applies_imagenet_normalization(extractor):
    """Normalisation must be applied -- but any equivalent formulation is fine.

    An exact-equality check against the implementation's own expression is a
    change-detector: a mathematically-equivalent refactor shifts results by
    ~1e-6, which survives the float16 cast and fails the comparison. So this
    asserts closeness to the reference AND clear separation from the
    un-normalised alternative.
    """
    frames = np.random.default_rng(0).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    actual = extractor.encode(frames).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    scaled = frames.astype(np.float32) / 255.0

    normalised = _encode_preprocessed(extractor, (scaled - mean) / std)
    unnormalised = _encode_preprocessed(extractor, scaled)

    assert np.allclose(actual, normalised, atol=2e-2), (
        "encode() does not match ImageNet-normalised input"
    )
    separation = np.abs(normalised - unnormalised).mean()
    assert separation > 1e-2, (
        f"normalised and un-normalised encodings differ by only {separation:.2e}; "
        "this test cannot detect whether normalisation is applied"
    )


def test_cache_episode_features_writes_sibling_file(tmp_path, extractor):
    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    ep = Episode(
        obs=np.zeros((4, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(3, dtype=np.int32),
        rewards=np.zeros(3, dtype=np.float32),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
        privileged=np.zeros((4, len(keys)), dtype=np.float32),
        privileged_keys=keys,
        policy_name="random",
        seed=0,
        scenario="my_way_home",
    )
    path = tmp_path / "ep_000000.npz"
    save_episode(ep, path)
    out = cache_episode_features(path, extractor)
    assert out.is_file()
    saved = np.load(out)
    assert saved.shape == (4, *BACKBONE_GEOMETRY[extractor.backbone])
    # Mutation-testing gap-fill: shape alone doesn't catch writing float32
    # instead of float16, which would blow the ~49KB/frame storage budget
    # per spec §3.3.
    assert saved.dtype == np.float16


def test_cache_writes_to_the_backbones_own_suffix(tmp_path):
    """Two caches must never collide.

    A hardcoded suffix here would make the random_vit run overwrite dinov2's
    cache, leaving arms 2 and 3 training on identical inputs with no error
    anywhere -- the control silently becomes a copy of the treatment. Verified
    by mutation: with the suffix hardcoded, the rest of the suite stays green.
    """

    class _StubExtractor:
        backbone = "random_vit"

        def encode(self, frames):
            return np.zeros((len(frames), *BACKBONE_GEOMETRY[self.backbone]), dtype=np.float16)

    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    episode = Episode(
        obs=np.zeros((4, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(3, dtype=np.int32),
        rewards=np.zeros(3, dtype=np.float32),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
        privileged=np.zeros((4, len(keys)), dtype=np.float32),
        privileged_keys=keys,
        policy_name="random",
        seed=0,
        scenario="my_way_home",
    )
    path = tmp_path / "ep_000000_len00003.npz"
    save_episode(episode, path)

    # A pre-existing dinov2 cache that must survive untouched.
    dinov2_cache = path.with_suffix(".features.npy")
    np.save(dinov2_cache, np.full((4, *BACKBONE_GEOMETRY["dinov2"]), 7, dtype=np.float16))

    out_path = cache_episode_features(path, _StubExtractor())

    assert out_path.name.endswith(".features_random_vit.npy"), out_path.name
    assert out_path.is_file()
    assert dinov2_cache.is_file(), "the dinov2 cache was deleted"
    assert np.load(dinov2_cache)[0, 0, 0] == 7, "the dinov2 cache was overwritten"
    assert np.load(out_path).shape == (4, *BACKBONE_GEOMETRY["random_vit"])


def test_backbones_registered():
    assert BACKBONES == ("dinov2", "random_vit", "pixel_ae")


def test_unknown_backbone_rejected():
    with pytest.raises(KeyError, match="unknown backbone 'nope'"):
        build_backbone("nope")


@pytest.mark.slow
def test_random_vit_has_the_same_output_shape_as_dinov2():
    """Arm 3 must be Arm 2 with different weights, not a different shape."""
    ext = FeatureExtractor(backbone="random_vit", device="cpu", seed=0)
    frames = np.random.default_rng(0).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert ext.encode(frames).shape == (2, *BACKBONE_GEOMETRY["random_vit"])


@pytest.mark.slow
def test_random_vit_is_reproducible_from_its_seed():
    a = FeatureExtractor(backbone="random_vit", device="cpu", seed=3)
    b = FeatureExtractor(backbone="random_vit", device="cpu", seed=3)
    frames = np.random.default_rng(1).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert np.array_equal(a.encode(frames), b.encode(frames))


@pytest.mark.slow
def test_random_vit_differs_from_dinov2():
    """If these matched, Arm 3 would not be a control at all."""
    rnd = FeatureExtractor(backbone="random_vit", device="cpu", seed=0)
    pre = FeatureExtractor(backbone="dinov2", device="cpu")
    frames = np.random.default_rng(2).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert not np.allclose(
        rnd.encode(frames).astype(np.float32),
        pre.encode(frames).astype(np.float32),
        atol=1e-2,
    )


@pytest.mark.slow
def test_different_seeds_give_different_random_backbones():
    a = FeatureExtractor(backbone="random_vit", device="cpu", seed=0)
    b = FeatureExtractor(backbone="random_vit", device="cpu", seed=1)
    frames = np.random.default_rng(3).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert not np.array_equal(a.encode(frames), b.encode(frames))


def test_require_free_bytes_passes_when_space_available(tmp_path):
    from mbfps.data.features import require_free_bytes

    require_free_bytes(tmp_path, 1)


def test_require_free_bytes_raises_when_space_insufficient(tmp_path):
    from mbfps.data.features import require_free_bytes

    with pytest.raises(OSError, match="needs .* free"):
        require_free_bytes(tmp_path, 10**15)


def test_require_free_bytes_message_names_both_numbers(tmp_path):
    from mbfps.data.features import require_free_bytes

    with pytest.raises(OSError) as excinfo:
        require_free_bytes(tmp_path, 10**15)
    message = str(excinfo.value)
    assert "GB" in message and "available" in message


# --- Multi-chunk batching ---------------------------------------------------
# Both existing tests of cache_episode_features use a 4-frame episode against a
# default batch_size of 32, so the comprehension ran exactly once and
# np.concatenate never joined more than one chunk. Real episodes are hundreds
# of frames, so the loop that production actually exercises had no coverage:
# an off-by-one in the range step, or a chunk written in the wrong order, would
# not have been caught.


class _CountingExtractor:
    """Returns each frame's identity marker, and records the chunk sizes seen."""

    backbone = "random_vit"

    def __init__(self):
        self.chunk_sizes = []

    def encode(self, frames):
        self.chunk_sizes.append(len(frames))
        markers = frames[:, 0, 0, 0].astype(np.float16)
        return np.broadcast_to(
            markers[:, None, None], (len(frames), *BACKBONE_GEOMETRY[self.backbone])
        ).copy()


def test_cache_episode_features_batches_and_preserves_frame_order(tmp_path):
    """A 70-frame episode at batch_size=32 must run three chunks, in order."""
    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    n = 70
    obs = np.zeros((n, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(n, dtype=np.uint8)  # frame identity
    episode = Episode(
        obs=obs,
        actions=np.zeros(n - 1, dtype=np.int32),
        rewards=np.zeros(n - 1, dtype=np.float32),
        terminated=np.zeros(n - 1, dtype=bool),
        truncated=np.zeros(n - 1, dtype=bool),
        privileged=np.zeros((n, len(keys)), dtype=np.float32),
        privileged_keys=keys,
        policy_name="random",
        seed=0,
        scenario="my_way_home",
    )
    path = tmp_path / "ep_000000_len00069.npz"
    save_episode(episode, path)

    extractor = _CountingExtractor()
    out_path = cache_episode_features(path, extractor, batch_size=32)

    assert extractor.chunk_sizes == [32, 32, 6], (
        f"expected three chunks 32/32/6, got {extractor.chunk_sizes} -- the "
        "batching loop is not covering the episode as claimed"
    )
    written = np.load(out_path)
    assert written.shape == (n, *BACKBONE_GEOMETRY["random_vit"])
    # Frame order must survive concatenation across chunk boundaries.
    np.testing.assert_array_equal(
        written[:, 0, 0].astype(np.int64), np.arange(n)
    )


# --- pixel_ae: the M2 autoencoder's encoder as a frozen backbone --------------
# None of these tests touch `runs/m2_fixed/autoencoder_cnn.pt`: it is gitignored
# and absent on a fresh checkout. The checkpoint under test is built here from a
# real `CNNEncoder` in the exact `{"arm", "state_dict"}` layout that
# `mbfps.training.autoencoder.train_autoencoder` writes, with `encoder.*` keys
# next to `decoder.*` keys so the loader has to select the subset. It is a full
# 26.4M-parameter encoder (the contract fixes the architecture to
# `CNNEncoder(EncoderConfig(kind="cnn"))`, whose `project` layer is 12544 x
# 2048), so it is written ONCE per module rather than per test.


def _frames(seed: int, n: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, *OBS_SHAPE), dtype=np.uint8)


@pytest.fixture(scope="module")
def fake_m2(tmp_path_factory):
    """A fake M2 `cnn` checkpoint and the state_dict it was written from."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    torch.manual_seed(1234)
    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    state = {f"encoder.{k}": v.clone() for k, v in encoder.state_dict().items()}
    # Decoder keys, as in the real file. The loader must ignore them rather
    # than fail a strict load or, worse, try to fit them into the encoder.
    state["decoder.project.weight"] = torch.zeros(3, 2048)
    state["decoder.project.bias"] = torch.zeros(3)
    path = tmp_path_factory.mktemp("m2") / "autoencoder_cnn.pt"
    torch.save({"arm": "cnn", "state_dict": state}, path)
    return path, state


@pytest.fixture(scope="module")
def pixel_ae(fake_m2):
    path, _ = fake_m2
    return FeatureExtractor(backbone="pixel_ae", device="cpu", checkpoint=path)


def test_pixel_ae_checkpoint_default_is_the_m2_cnn_autoencoder():
    assert PIXEL_AE_CHECKPOINT == Path("runs/m2_fixed/autoencoder_cnn.pt")


def test_build_backbone_pixel_ae_loads_the_checkpoint_weights_not_a_fresh_init(fake_m2):
    """Every encoder tensor must equal the checkpoint's, so the features are
    the TRAINED autoencoder's and not a random CNN's -- which is the exact
    collapsed target the design retires."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    path, state = fake_m2
    loaded = build_backbone("pixel_ae", checkpoint=path).state_dict()
    expected = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    assert set(loaded) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(loaded[key], tensor), key
    # Self-check: a fresh init must NOT match, or the equality above is vacuous.
    fresh = CNNEncoder(EncoderConfig(kind="cnn")).state_dict()
    assert not torch.equal(fresh["project.weight"], expected["project.weight"])


def test_build_backbone_pixel_ae_is_frozen(fake_m2):
    path, _ = fake_m2
    model = build_backbone("pixel_ae", checkpoint=path)
    params = list(model.parameters())
    assert len(params) == 10  # 4 convs + project, weight and bias each
    assert all(not p.requires_grad for p in params)


def test_build_backbone_pixel_ae_is_in_eval_mode(fake_m2):
    path, _ = fake_m2
    assert build_backbone("pixel_ae", checkpoint=path).training is False


def test_build_backbone_pixel_ae_missing_checkpoint_names_the_path(tmp_path):
    """Matched on the guard's own wording, not just the exception type:
    `torch.load` raises its own FileNotFoundError with the path in it, so a
    type-only assertion passes with the guard deleted. The guard exists to
    say WHICH backbone wanted the file and how M2 produces it."""
    missing = tmp_path / "nowhere" / "autoencoder_cnn.pt"
    with pytest.raises(
        FileNotFoundError,
        match="pixel_ae checkpoint not found at " + re.escape(str(missing)),
    ):
        build_backbone("pixel_ae", checkpoint=missing)


def test_build_backbone_pixel_ae_rejects_a_non_cnn_checkpoint(fake_m2, tmp_path):
    """Only the arm tag differs from a loadable checkpoint. A loader that
    skipped the tag and relied on `load_state_dict` failing would pass a
    mismatched-weights test and still accept this file."""
    _, state = fake_m2
    wrong = tmp_path / "autoencoder_frozen_ssl.pt"
    torch.save({"arm": "frozen_ssl", "state_dict": state}, wrong)
    with pytest.raises(ValueError, match="frozen_ssl"):
        build_backbone("pixel_ae", checkpoint=wrong)


def test_build_backbone_pixel_ae_reads_the_default_path_when_none_is_given(
    fake_m2, monkeypatch
):
    """`checkpoint=None` means the module default, not "no checkpoint"."""
    path, _ = fake_m2
    monkeypatch.setattr(features, "PIXEL_AE_CHECKPOINT", path)
    assert build_backbone("pixel_ae").training is False
    monkeypatch.setattr(features, "PIXEL_AE_CHECKPOINT", path.with_name("absent.pt"))
    with pytest.raises(FileNotFoundError, match="absent.pt"):
        build_backbone("pixel_ae")


def test_pixel_ae_encode_output_shape_and_dtype(pixel_ae):
    out = pixel_ae.encode(_frames(0, n=3))
    assert out.shape == (3, 64, 32)
    assert out.dtype == np.float16


def test_pixel_ae_encode_is_byte_identical_on_repeat(pixel_ae):
    frames = _frames(1)
    assert np.array_equal(pixel_ae.encode(frames), pixel_ae.encode(frames))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_pixel_ae_encode_is_byte_identical_on_repeat_on_mps(fake_m2):
    """`cache_features.py` runs on MPS by default; CPU repeatability alone
    would not license the cache the study trains on."""
    path, _ = fake_m2
    ext = FeatureExtractor(backbone="pixel_ae", device="mps", checkpoint=path)
    frames = _frames(6)
    assert np.array_equal(ext.encode(frames), ext.encode(frames))


def test_pixel_ae_encode_is_the_2048_vector_partitioned_row_major(pixel_ae, fake_m2):
    """Row r of the (64, 32) grid is dims [32r, 32r+32) of the encoder's
    output. Any other partition -- transposed, or a (32, 64) grid swapped
    into place -- is a permutation of the same numbers, so shape cannot catch
    it; flattening back must reproduce the encoder's output bit-for-bit."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    _, state = fake_m2
    reference = CNNEncoder(EncoderConfig(kind="cnn")).eval()
    reference.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    frames = _frames(2)
    with torch.no_grad():
        expected = reference(torch.from_numpy(frames)).to(torch.float16).numpy()
    assert expected.shape == (2, 2048)
    # Self-check: the values must be distinct enough that a permutation is
    # detectable, or this test could not tell row-major from any other order.
    assert np.unique(expected[0]).size > 1500

    flat = pixel_ae.encode(frames).reshape(2, 2048)
    assert np.array_equal(flat, expected)


def test_pixel_ae_encode_feeds_raw_uint8_with_no_imagenet_normalisation(pixel_ae, fake_m2):
    """M2 trained the CNN on `uint8 / 255 - 0.5`, which `CNNEncoder.forward`
    applies itself. ImageNet mean/std on top would shift every input off the
    distribution the weights were fit on. Asserts closeness to the raw path
    AND clear separation from the normalised one, so the check is not vacuous."""
    from mbfps.models.encoders import CNNEncoder
    from mbfps.utils.config import EncoderConfig

    _, state = fake_m2
    reference = CNNEncoder(EncoderConfig(kind="cnn")).eval()
    reference.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    frames = _frames(3)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # What a copy-pasted ViT preprocessing branch would hand the CNN: floats
    # already scaled, which the CNN then divides by 255 again.
    normalised_input = (frames.astype(np.float32) / 255.0 - mean) / std
    with torch.no_grad():
        raw = reference(torch.from_numpy(frames)).numpy()
        normalised = reference(torch.from_numpy(normalised_input)).numpy()

    actual = pixel_ae.encode(frames).reshape(2, 2048).astype(np.float32)
    # Tolerances are RELATIVE to the output scale. A random-init CNN emits
    # values of order 5e-3 (the near-constant embedding of design §1), so an
    # absolute atol=1e-2 would accept the normalised path too and the test
    # would pass against either implementation. float16 keeps ~3 significant
    # digits; 5e-3 covers its round-trip and nothing more.
    scale = float(np.abs(raw).max())
    tol = dict(rtol=5e-3, atol=5e-3 * scale)
    assert np.allclose(actual, raw, **tol)
    assert not np.allclose(normalised, raw, **tol), (
        "raw and normalised encodings agree within tolerance; this test "
        "cannot detect whether normalisation is applied"
    )


def test_pixel_ae_encode_rejects_wrong_frame_shape(pixel_ae):
    with pytest.raises(ValueError, match="expected frames of shape"):
        pixel_ae.encode(np.zeros((2, 64, 64, 3), dtype=np.uint8))


def test_pixel_ae_encode_refuses_an_encoder_of_the_wrong_width(pixel_ae, monkeypatch):
    """A checkpoint with a different `embed_dim` would otherwise die inside
    `reshape` with a message about tensor sizes, not about the backbone."""
    monkeypatch.setattr(pixel_ae, "model", lambda x: torch.zeros(len(x), 2047))
    with pytest.raises(ValueError, match=r"pixel_ae.*2047.*64 x 32"):
        pixel_ae.encode(_frames(4))


def test_vit_encode_refuses_the_wrong_patch_count(extractor, monkeypatch):
    """The shared post-branch guard, exercised on the ViT path: 1 + 63 tokens
    (a wrong patch size) must be named, not silently cached as 63 rows."""

    class _Out:
        last_hidden_state = torch.zeros(2, 1 + 63, 384)

    monkeypatch.setattr(extractor, "model", lambda pixel_values: _Out())
    # registered (64, 384) first, emitted (63, 384) second -- Task 1's field order
    with pytest.raises(ValueError, match=r"dinov2.*\(64, 384\).*\(63, 384\)"):
        extractor.encode(_frames(5))


def test_pixel_ae_cache_writes_64_by_32_rows_to_its_own_suffix(tmp_path, pixel_ae):
    """End to end through `cache_episode_features`: the pixel_ae cache must
    land in `.features_pixel_ae.npy` with the registry's geometry, so the
    loader's shape validation and `feature_suffix` agree with what is on disk."""
    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    episode = Episode(
        obs=np.zeros((4, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(3, dtype=np.int32),
        rewards=np.zeros(3, dtype=np.float32),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
        privileged=np.zeros((4, len(keys)), dtype=np.float32),
        privileged_keys=keys,
        policy_name="random",
        seed=0,
        scenario="my_way_home",
    )
    path = tmp_path / "ep_000000_len00003.npz"
    save_episode(episode, path)
    out = cache_episode_features(path, pixel_ae)
    assert out.name == "ep_000000_len00003.features_pixel_ae.npy"
    saved = np.load(out)
    assert saved.shape == (4, 64, 32)  # literal: this is the on-disk contract
    assert saved.dtype == np.float16
