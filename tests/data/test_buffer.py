import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, policy: str = "random", seed: int = 0) -> Episode:
    return Episode(
        obs=np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name=policy,
        seed=seed,
        scenario="my_way_home",
    )


def test_add_writes_a_file(tmp_path):
    assert ReplayBuffer(tmp_path, capacity_transitions=100).add(make_episode(10)).is_file()


def test_counts_track_contents(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    buf.add(make_episode(10))
    buf.add(make_episode(15))
    assert buf.n_episodes == 2
    assert buf.n_transitions == 25


def _poison_load_episode(monkeypatch):
    """Make any load_episode call fail, so decompression becomes detectable."""
    import mbfps.data.buffer as buffer_module

    def _boom(*args, **kwargs):
        raise AssertionError("capacity accounting must not load episode files")

    monkeypatch.setattr(buffer_module, "load_episode", _boom)


def test_n_transitions_does_not_decompress_episodes(tmp_path, monkeypatch):
    buf = ReplayBuffer(tmp_path, capacity_transitions=1000)
    for _ in range(3):
        buf.add(make_episode(10))
    _poison_load_episode(monkeypatch)
    assert buf.n_transitions == 30


def test_eviction_does_not_decompress_episodes(tmp_path, monkeypatch):
    """`_evict` is the hotter path -- `add()` calls it on every episode.

    Poison BEFORE the adds and use a capacity that forces eviction, so this
    actually covers `_evict`. Poisoning afterwards leaves the O(n^2)
    regression restorable with the whole suite still green.
    """
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    _poison_load_episode(monkeypatch)
    for _ in range(4):
        buf.add(make_episode(10))
    assert buf.n_episodes == 2, "eviction must have run without decompressing"
    assert buf.n_transitions <= 25


def test_eviction_respects_capacity(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    for _ in range(5):
        buf.add(make_episode(10))
    assert buf.n_transitions <= 25


def test_eviction_removes_oldest_first(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    for seed in (1, 2, 3):
        buf.add(make_episode(10, seed=seed))
    seeds = {ep.seed for ep in buf.load_all()}
    assert 1 not in seeds, "oldest episode should have been evicted"
    assert {2, 3} <= seeds


def test_eviction_also_removes_cached_features(tmp_path):
    """Orphaned .features.npy files would defeat the disk-growth bound."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    first = buf.add(make_episode(10, seed=1))
    features = first.with_suffix(".features.npy")
    np.save(features, np.zeros((11, 4, 8), dtype=np.float16))
    assert features.is_file()
    buf.add(make_episode(10, seed=2))
    buf.add(make_episode(10, seed=3))
    assert not first.is_file()
    assert not features.is_file(), "feature cache outlived its episode"


def test_buffer_reopens_existing_directory(tmp_path):
    ReplayBuffer(tmp_path, capacity_transitions=100).add(make_episode(10))
    assert ReplayBuffer(tmp_path, capacity_transitions=100).n_episodes == 1


def test_indices_continue_after_reopen(tmp_path):
    ReplayBuffer(tmp_path, capacity_transitions=1000).add(make_episode(4, seed=1))
    reopened = ReplayBuffer(tmp_path, capacity_transitions=1000)
    reopened.add(make_episode(4, seed=2))
    assert reopened.n_episodes == 2, "second buffer must not overwrite the first file"


def test_load_all_returns_episodes(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    buf.add(make_episode(4, policy="scripted"))
    episodes = buf.load_all()
    assert len(episodes) == 1
    assert episodes[0].policy_name == "scripted"


def test_single_oversized_episode_is_kept(tmp_path):
    """Evicting to empty would make the buffer useless; keep the newest."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=5)
    buf.add(make_episode(50))
    assert buf.n_episodes == 1


def test_stray_files_are_reported(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    buf.add(make_episode(10))
    stray_a = tmp_path / "ep_.npz"
    stray_b = tmp_path / "ep_1_len.npz"
    stray_a.touch()
    stray_b.touch()
    assert set(buf.stray_paths()) == {stray_a, stray_b}
    assert buf.n_episodes == 1, "strays must not be counted as episodes"


def test_stray_files_are_not_deleted(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=5)
    stray_a = tmp_path / "ep_.npz"
    stray_b = tmp_path / "ep_1_len.npz"
    stray_a.touch()
    stray_b.touch()
    for seed in (1, 2, 3):
        buf.add(make_episode(10, seed=seed))  # forces eviction, capacity=5
    assert stray_a.is_file(), "eviction must never delete unrecognised files"
    assert stray_b.is_file(), "eviction must never delete unrecognised files"


def test_load_all_detects_filename_length_divergence(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    path = buf.add(make_episode(10))
    bad_path = path.with_name(path.name.replace("_len00010.npz", "_len00099.npz"))
    path.rename(bad_path)
    with pytest.raises(ValueError, match=r"99.*10"):
        buf.load_all()


def test_two_buffers_over_one_directory_do_not_overwrite(tmp_path):
    buf_a = ReplayBuffer(tmp_path, capacity_transitions=1000)
    buf_b = ReplayBuffer(tmp_path, capacity_transitions=1000)
    buf_a.add(make_episode(10, seed=101))
    buf_b.add(make_episode(10, seed=202))
    assert buf_a.n_episodes == 2
    seeds = {ep.seed for ep in buf_a.load_all()}
    assert seeds == {101, 202}
