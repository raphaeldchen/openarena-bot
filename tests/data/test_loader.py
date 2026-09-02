import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode, load_episode
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


def test_window_start_is_present_and_valid(buffer):
    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    for _ in range(10):
        batch = loader.sample()
        starts = batch["window_start"]
        assert starts.shape == (8,)
        assert starts.dtype == np.int32
        for start, idx in zip(starts, batch["episode_index"]):
            ep = loader._episodes[idx]
            assert 0 <= start <= ep.length - loader.seq_len


def test_episode_path_matches_the_episode_the_window_came_from(buffer):
    from mbfps.data.episode import load_episode

    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    batch = loader.sample()
    for idx in batch["episode_index"]:
        path = loader.episode_path(int(idx))
        assert path.is_file()
        on_disk = load_episode(path)
        in_memory = loader._episodes[idx]
        assert on_disk.policy_name == in_memory.policy_name
        assert on_disk.seed == in_memory.seed


def test_episode_path_and_window_start_slice_back_to_the_sampled_window(buffer):
    """The cache is only usable if a consumer can reconstruct a batch's window
    from `episode_path()` and `window_start` alone -- this proves it can."""
    from mbfps.data.episode import load_episode

    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    batch = loader.sample()
    for i in range(loader.batch_size):
        idx = int(batch["episode_index"][i])
        start = int(batch["window_start"][i])
        episode = load_episode(loader.episode_path(idx))
        reconstructed = episode.obs[start : start + loader.seq_len + 1]
        assert np.array_equal(reconstructed, batch["obs"][i])


def test_load_obs_false_omits_obs(buffer):
    loader = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0, load_obs=False)
    batch = loader.sample()
    assert "obs" not in batch
    assert batch["actions"].shape == (4, 16)
    assert batch["episode_index"].shape == (4,)
    assert batch["window_start"].shape == (4,)


def test_load_obs_false_still_supports_privileged(buffer):
    loader = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0, load_obs=False)
    batch = loader.sample(include_privileged=True)
    assert batch["privileged"].shape[0] == 4


def test_load_features_requires_cached_files(tmp_path):
    """A missing cache must fail loudly, not silently train on nothing."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    with pytest.raises(FileNotFoundError, match="no cached features"):
        SequenceLoader(buf, batch_size=2, seq_len=16, seed=0, load_features=True)


def test_features_window_matches_manual_slice(tmp_path):
    """The batch must slice the cache at exactly the reported window_start."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2):
        buf.add(make_episode(t=80, fill=fill))
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        feats = rng.random((81, 4, 8)).astype(np.float16)
        np.save(path.with_suffix(".features.npy"), feats)

    loader = SequenceLoader(
        buf, batch_size=4, seq_len=16, seed=0, load_obs=False, load_features=True
    )
    batch = loader.sample()
    assert batch["features"].shape == (4, 17, 4, 8)
    assert batch["features"].dtype == np.float16
    for i in range(4):
        idx, start = int(batch["episode_index"][i]), int(batch["window_start"][i])
        on_disk = np.load(loader.episode_path(idx).with_suffix(".features.npy"))
        assert np.array_equal(on_disk[start : start + 17], batch["features"][i])


def test_features_and_obs_windows_are_aligned(tmp_path):
    """Both must come from the same episode and the same offset."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=3))
    feats = np.arange(81 * 4 * 8, dtype=np.float16).reshape(81, 4, 8)
    np.save(buf.episode_paths()[0].with_suffix(".features.npy"), feats)

    loader = SequenceLoader(
        buf, batch_size=2, seq_len=16, seed=0, load_obs=True, load_features=True
    )
    batch = loader.sample()
    for i in range(2):
        start = int(batch["window_start"][i])
        assert np.array_equal(batch["features"][i], feats[start : start + 17])
        assert batch["obs"][i].shape == (17, *OBS_SHAPE)


def test_feature_suffix_namespaces_non_default_backbones():
    from mbfps.data.loader import feature_suffix

    assert feature_suffix("dinov2") == ".features.npy"
    assert feature_suffix("random_vit") == ".features_random_vit.npy"


def test_loader_reads_the_requested_backbones_cache(tmp_path):
    """Two caches coexist; picking the wrong one would silently destroy the
    control by training arms 2 and 3 on identical inputs."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    path = buf.episode_paths()[0]
    np.save(path.with_suffix(".features.npy"), np.zeros((81, 4, 8), np.float16))
    np.save(path.with_suffix(".features_random_vit.npy"), np.ones((81, 4, 8), np.float16))

    for backbone, expected in (("dinov2", 0.0), ("random_vit", 1.0)):
        loader = SequenceLoader(
            buf, 2, 16, seed=0, load_obs=False,
            load_features=True, feature_backbone=backbone,
        )
        assert (loader.sample()["features"] == expected).all(), backbone


def test_obs_free_loader_does_not_read_pixels(tmp_path, monkeypatch):
    """Guards the 2.24 GB regression at the point pixels would actually be read.

    Spying on `load_episode` is not enough: the obs-free path never calls it,
    so that version of this guard passed even when `from_npz` was changed to
    read `data["obs"]` directly. Intercepting the npz key access catches an
    obs read from any code path.

    The patch is class-level because Python resolves `__getitem__` on the type,
    so patching the NpzFile instance would not intercept `data["obs"]`.
    """
    from numpy.lib.npyio import NpzFile

    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))

    original = NpzFile.__getitem__

    def guarded(self, key):
        assert key != "obs", "obs must never be read when load_obs=False"
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", guarded)
    loader = SequenceLoader(buf, batch_size=2, seq_len=16, seed=0, load_obs=False)
    assert loader.sample()["actions"].shape == (2, 16)


def test_obs_loading_loader_does_read_pixels(tmp_path, monkeypatch):
    """The complement: proves the guard above can actually fire.

    Without this, a guard that never triggers under any condition would look
    identical to one that correctly never triggers.
    """
    from numpy.lib.npyio import NpzFile

    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))

    original = NpzFile.__getitem__
    seen: list[str] = []

    def spy(self, key):
        seen.append(key)
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", spy)
    SequenceLoader(buf, batch_size=2, seq_len=16, seed=0, load_obs=True)
    assert "obs" in seen, "load_obs=True should read obs; the spy is not wired up"
