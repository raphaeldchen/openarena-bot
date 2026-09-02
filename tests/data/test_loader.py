import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    """Every frame is filled with `fill`, so windows carry episode identity."""
    return Episode(
        obs=np.full((t + 1, *OBS_SHAPE), fill, dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.full((t + 1, len(KEYS)), float(fill), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4):
        buf.add(make_episode(t=80, fill=fill))
    return buf


def test_batch_shapes(buffer):
    batch = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0).sample()
    assert batch["obs"].shape == (4, 17, *OBS_SHAPE)
    assert batch["actions"].shape == (4, 16)
    assert batch["rewards"].shape == (4, 16)
    assert batch["terminated"].shape == (4, 16)
    assert batch["truncated"].shape == (4, 16)
    assert batch["episode_index"].shape == (4,)


def test_batch_dtypes(buffer):
    batch = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0).sample()
    assert batch["obs"].dtype == np.uint8
    assert batch["actions"].dtype == np.int32
    assert batch["rewards"].dtype == np.float32
    assert batch["terminated"].dtype == bool
    assert batch["truncated"].dtype == bool


def test_windows_never_cross_episode_boundaries(buffer):
    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    for _ in range(20):
        for window in loader.sample()["obs"]:
            assert len(np.unique(window)) == 1, "window spans two episodes"


def test_privileged_absent_by_default(buffer):
    """The safe path is the default path."""
    assert "privileged" not in SequenceLoader(buffer, 4, 16, seed=0).sample()


def test_privileged_present_only_when_requested(buffer):
    batch = SequenceLoader(buffer, 4, 16, seed=0).sample(include_privileged=True)
    assert batch["privileged"].shape == (4, 17, len(KEYS))


def test_sampling_is_reproducible(buffer):
    a = SequenceLoader(buffer, 4, 16, seed=99).sample()
    b = SequenceLoader(buffer, 4, 16, seed=99).sample()
    assert np.array_equal(a["obs"], b["obs"])
    assert np.array_equal(a["episode_index"], b["episode_index"])


def test_episodes_shorter_than_seq_len_are_skipped(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=4, fill=9))
    buf.add(make_episode(t=80, fill=7))
    batch = SequenceLoader(buf, batch_size=8, seq_len=16, seed=0).sample()
    assert np.all(batch["obs"] == 7)


def test_no_usable_episodes_raises(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=4, fill=9))
    with pytest.raises(ValueError, match="no episodes long enough"):
        SequenceLoader(buf, batch_size=4, seq_len=64, seed=0).sample()
