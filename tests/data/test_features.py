import numpy as np
import pytest
import torch

from mbfps.data.features import (
    FEATURE_DIM,
    N_PATCHES,
    FeatureExtractor,
    cache_episode_features,
)
from mbfps.envs.protocol import OBS_SHAPE

pytestmark = pytest.mark.slow  # downloads ~88MB on first run


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(device="cpu")


def test_patch_grid_is_8x8():
    assert N_PATCHES == 64
    assert FEATURE_DIM == 384


def test_encode_output_shape(extractor):
    frames = np.random.randint(0, 256, (3, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).shape == (3, N_PATCHES, FEATURE_DIM)


def test_encode_output_dtype_is_float16(extractor):
    frames = np.random.randint(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).dtype == np.float16


def test_encode_is_byte_identical_on_repeat(extractor):
    """The cache is only sound if the frozen backbone is deterministic."""
    frames = np.random.randint(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    a = extractor.encode(frames)
    b = extractor.encode(frames)
    assert np.array_equal(a, b)


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


def test_encode_applies_imagenet_normalization(extractor):
    """Mutation-testing gap-fill: none of the tests above pin down the actual
    preprocessing values, so silently feeding raw [0, 1] pixels instead of
    ImageNet-normalised ones passes every test above undetected. Recompute the
    reference preprocessing independently and compare bit-for-bit against
    `encode`'s output -- a frozen cache built on the wrong preprocessing would
    be internally consistent (deterministic, distinguishes frames) but wrong.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames = np.random.randint(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    x = (frames.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(extractor.device)
    with torch.no_grad():
        expected = (
            extractor.model(pixel_values=tensor)
            .last_hidden_state[:, 1:, :]
            .to(torch.float16)
            .cpu()
            .numpy()
        )
    assert np.array_equal(extractor.encode(frames), expected)


def test_cache_episode_features_writes_sibling_file(tmp_path, extractor):
    from mbfps.data.episode import Episode, save_episode

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
    assert saved.shape == (4, N_PATCHES, FEATURE_DIM)
    # Mutation-testing gap-fill: shape alone doesn't catch writing float32
    # instead of float16, which would blow the ~49KB/frame storage budget
    # per spec §3.3.
    assert saved.dtype == np.float16
