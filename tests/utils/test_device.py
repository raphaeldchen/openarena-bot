import numpy as np
import torch

from mbfps.utils.device import to_device


def test_privileged_is_never_moved_to_the_device():
    """Invariant #2 held structurally: privileged cannot reach a training
    tensor because it never leaves the host dict."""
    batch = {
        "obs": np.zeros((1, 2), dtype=np.uint8),
        "privileged": np.zeros((1, 5), dtype=np.float32),
    }
    moved = to_device(batch, torch.device("cpu"))
    assert "privileged" not in moved
    assert "obs" in moved


def test_world_model_inputs_survive():
    batch = {
        k: np.zeros((1, 2), dtype=np.float32)
        for k in ("obs", "features", "actions", "rewards", "terminated", "truncated")
    }
    assert set(to_device(batch, torch.device("cpu"))) == set(batch)


def test_the_whitelist_is_a_whitelist_and_not_a_dtype_test():
    """The barrier is the key list, not "is it an array".

    Replacing the membership test with something permissive (`hasattr(value,
    "dtype")`, `isinstance(value, np.ndarray)` alone) looks like a tidy
    generalisation and quietly dissolves invariant #2: `privileged` is a
    perfectly ordinary float32 array and would sail through. So an unknown key
    carrying an entirely well-formed array must still be dropped.
    """
    batch = {
        "obs": np.zeros((1, 2), dtype=np.uint8),
        "privileged": np.zeros((1, 5), dtype=np.float32),
        "episode_index": np.zeros(1, dtype=np.int32),
        "window_start": np.zeros(1, dtype=np.int32),
        "something_new": np.zeros((1, 3), dtype=np.float32),
    }
    assert set(to_device(batch, torch.device("cpu"))) == {"obs"}


def test_moved_arrays_become_tensors_on_the_requested_device():
    device = torch.device("cpu")
    moved = to_device({"actions": np.arange(4, dtype=np.int32)}, device)
    assert isinstance(moved["actions"], torch.Tensor)
    assert moved["actions"].device == device
    torch.testing.assert_close(moved["actions"], torch.arange(4, dtype=torch.int32))


def test_non_array_values_are_dropped_even_on_whitelisted_keys():
    """A batch key can carry a list or a scalar; only arrays are converted."""
    assert to_device({"obs": [1, 2, 3], "rewards": 0.5}, torch.device("cpu")) == {}
