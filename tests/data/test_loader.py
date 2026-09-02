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


def test_batch_draws_from_multiple_episodes(buffer):
    """A loader stuck on one episode would train on a fraction of the data.

    The buffer fixture holds four usable episodes, so eight draws landing on a
    single one has probability ~6e-5 per batch; over five batches it is
    negligible, and the loader is seeded, so this is deterministic.
    """
    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    seen = set()
    for _ in range(5):
        seen.update(loader.sample()["episode_index"].tolist())
    assert len(seen) > 1, f"every sample came from episode(s) {seen}"


def test_window_offsets_vary(buffer):
    """A loader fixed at offset 0 would only ever show each episode's opening.

    Each episode here is filled with a constant, so the offset is invisible in
    the pixels. Detect it through the frames themselves: write a per-timestep
    marker into one episode and check that sampled windows start at different
    points within it.
    """
    from mbfps.data.episode import load_episode, save_episode

    path = buffer.episode_paths()[0]
    episode = load_episode(path)
    for t in range(episode.obs.shape[0]):
        episode.obs[t, 0, 0, 0] = t % 251  # a per-timestep marker
    save_episode(episode, path)

    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    starts = set()
    for _ in range(10):
        batch = loader.sample()
        for row, idx in zip(batch["obs"], batch["episode_index"]):
            if idx == 0:
                starts.add(int(row[0, 0, 0, 0]))
    assert len(starts) > 1, f"all sampled windows began at the same offset: {starts}"


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


def test_large_buffer_logs_a_memory_warning(tmp_path, caplog, monkeypatch):
    """The eager load is a real constraint on a 16 GB machine; make it visible."""
    import mbfps.data.loader as loader_module

    monkeypatch.setattr(loader_module, "_WARN_BYTES", 1)
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    with caplog.at_level("WARNING"):
        SequenceLoader(buf, batch_size=2, seq_len=16, seed=0)
    assert "resident in RAM" in caplog.text


def test_small_buffer_logs_no_warning(buffer, caplog):
    with caplog.at_level("WARNING"):
        SequenceLoader(buffer, batch_size=2, seq_len=16, seed=0)
    assert "resident in RAM" not in caplog.text
