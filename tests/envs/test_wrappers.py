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


def test_output_values_stay_in_uint8_range():
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)
    out = preprocess_frame(frame)
    assert out.min() >= 0 and out.max() <= 255


def test_already_correct_size_is_passed_through_unchanged():
    frame = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
    assert np.array_equal(preprocess_frame(frame), frame)


def test_deterministic_for_same_input():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    assert np.array_equal(preprocess_frame(frame), preprocess_frame(frame))


def test_grayscale_input_rejected():
    with pytest.raises(ValueError, match="expected 3 channels"):
        preprocess_frame(np.zeros((120, 160), dtype=np.uint8))
