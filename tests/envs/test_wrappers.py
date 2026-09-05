import cv2
import numpy as np
import pytest

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.envs.wrappers import preprocess_frame


def test_hwc_input_resized_to_obs_shape():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    assert preprocess_frame(frame).shape == OBS_SHAPE


def test_chw_input_is_transposed_then_resized():
    frame = np.random.randint(0, 256, (3, 120, 160), dtype=np.uint8)
    assert preprocess_frame(frame).shape == OBS_SHAPE


def test_output_is_uint8():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    assert preprocess_frame(frame).dtype == np.uint8


def test_hwc_output_matches_area_interpolation():
    """Pins the interpolation mode: INTER_NEAREST/INTER_LINEAR give different pixels."""
    frame = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    expected = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)
    assert np.array_equal(preprocess_frame(frame), expected)


def test_chw_input_transposes_on_the_correct_axes():
    """A non-square CHW input catches a (2,1,0) transpose that (1,2,0) shape checks miss."""
    frame_chw = np.random.default_rng(1).integers(0, 256, (3, 100, 160), dtype=np.uint8)
    expected = cv2.resize(
        np.transpose(frame_chw, (1, 2, 0)), (112, 112), interpolation=cv2.INTER_AREA
    )
    assert np.array_equal(preprocess_frame(frame_chw), expected)


def test_vertical_split_stays_vertical():
    """Independent of the implementation: a left/right split must not become top/bottom."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, 80:, :] = 255
    out = preprocess_frame(frame)
    assert out[:, :50, :].mean() < 10, "left half should stay dark"
    assert out[:, 62:, :].mean() > 245, "right half should stay bright"
    assert 100 < out[:50, :, :].mean() < 155, "top half should be mixed, not uniform"


def test_output_is_c_contiguous():
    """Downstream code stacks these into batches; a non-contiguous view copies silently."""
    frame = np.random.default_rng(2).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    assert preprocess_frame(frame).flags["C_CONTIGUOUS"]
    already_sized = np.random.default_rng(3).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    assert preprocess_frame(already_sized[::-1]).flags["C_CONTIGUOUS"]


def test_already_correct_size_is_passed_through_unchanged():
    frame = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
    assert np.array_equal(preprocess_frame(frame), frame)


def test_deterministic_for_same_input():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    assert np.array_equal(preprocess_frame(frame), preprocess_frame(frame))


def test_grayscale_input_rejected():
    with pytest.raises(ValueError, match="expected 3 channels"):
        preprocess_frame(np.zeros((120, 160), dtype=np.uint8))
