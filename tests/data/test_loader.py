import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode, load_episode
from mbfps.data.loader import SequenceLoader, feature_suffix
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
        # (64, 384) is dinov2's registered geometry; the loader now refuses
        # anything else under this suffix (see the geometry tests below).
        feats = rng.random((81, 64, 384)).astype(np.float16)
        np.save(path.with_suffix(".features.npy"), feats)

    loader = SequenceLoader(
        buf, batch_size=4, seq_len=16, seed=0, load_obs=False, load_features=True
    )
    batch = loader.sample()
    assert batch["features"].shape == (4, 17, 64, 384)
    assert batch["features"].dtype == np.float16
    for i in range(4):
        idx, start = int(batch["episode_index"][i]), int(batch["window_start"][i])
        on_disk = np.load(loader.episode_path(idx).with_suffix(".features.npy"))
        assert np.array_equal(on_disk[start : start + 17], batch["features"][i])


def test_features_and_obs_windows_are_aligned(tmp_path):
    """Both must come from the same episode and the same offset."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=3))
    feats = np.arange(81 * 64 * 384, dtype=np.float32).reshape(81, 64, 384)
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
    assert feature_suffix("pixel_ae") == ".features_pixel_ae.npy"


def test_loader_reads_the_requested_backbones_cache(tmp_path):
    """Two caches coexist; picking the wrong one would silently destroy the
    control by training arms 2 and 3 on identical inputs."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    path = buf.episode_paths()[0]
    np.save(path.with_suffix(".features.npy"), np.zeros((81, 64, 384), np.float16))
    np.save(path.with_suffix(".features_random_vit.npy"), np.ones((81, 64, 384), np.float16))

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


# --- Feature geometry ---------------------------------------------------------
# A cache is a bare `.npy`; nothing in it records which backbone wrote it. The
# suffix says which backbone the loader THINKS it is reading, and the row
# geometry is the only property of the bytes that can contradict it. With
# three backbones and two geometries -- (64, 384) for the ViTs, (64, 32) for
# the M2 pixel autoencoder -- a cache written under the wrong suffix used to
# surface as a matmul shape error inside `BottleneckEncoder` at step 0, after
# the episode loading and model construction time was paid. It is refused at
# construction now, naming what was found and what the suffix promised.

# Literal, deliberately NOT `BACKBONE_GEOMETRY.items()`: parametrising over
# the registry would make these tests shrink silently if an entry were
# dropped, and would agree with whatever the registry said even if a width
# were edited to the wrong number.
_GEOMETRY_CASES = [
    ("dinov2", (64, 384)),
    ("random_vit", (64, 384)),
    ("pixel_ae", (64, 32)),
]


def _buffer_with_cache(tmp_path, backbone, rows, n_episodes=1, t=16):
    """`n_episodes` episodes, each with a `(t + 1, *rows)` cache for `backbone`.

    `t = 16` is exactly `_feature_loader`'s `seq_len`, the shortest episode
    it can sample from, so a ViT-geometry cache is 17 x 64 x 384 x 2 bytes
    (0.8 MB) rather than the 4 MB an 80-step one costs; these tests write a
    dozen of them.
    """
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for i in range(n_episodes):
        buf.add(make_episode(t=t, fill=i + 1))
    for path in buf.episode_paths():
        np.save(path.with_suffix(feature_suffix(backbone)),
                np.zeros((t + 1, *rows), dtype=np.float16))
    return buf


def _feature_loader(buf, backbone):
    return SequenceLoader(
        buf, batch_size=2, seq_len=16, seed=0, load_obs=False,
        load_features=True, feature_backbone=backbone,
    )


def test_a_vit_geometry_cache_under_the_pixel_ae_suffix_is_refused(tmp_path):
    """The rows are 64 in both, so this is the width mismatch alone."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 384))
    with pytest.raises(ValueError, match="does not match backbone"):
        _feature_loader(buf, "pixel_ae")


def test_a_pixel_ae_geometry_cache_under_the_dinov2_suffix_is_refused(tmp_path):
    """The reverse direction: same row count, narrower rows than promised."""
    buf = _buffer_with_cache(tmp_path, "dinov2", (64, 32))
    with pytest.raises(ValueError, match="does not match backbone"):
        _feature_loader(buf, "dinov2")


def test_a_cache_with_the_right_width_but_wrong_row_count_is_refused(tmp_path):
    """The two tests above both carry 64 rows, so a guard comparing only the
    width passes them. This one has the right width and 16 rows, so a guard
    comparing only the row count is the one it catches -- together they pin
    the comparison to the whole `(n_patches, patch_dim)` pair."""
    buf = _buffer_with_cache(tmp_path, "dinov2", (16, 384))
    with pytest.raises(ValueError, match="does not match backbone"):
        _feature_loader(buf, "dinov2")


def test_an_unpartitioned_pixel_ae_cache_is_refused(tmp_path):
    """The M2 encoder emits a flat 2048-vector; the cache must hold it as
    64 rows of 32. A `(T, 2048)` file is exactly the mistake a future writer
    makes by forgetting the reshape, and its `shape[1:]` is `(2048,)`."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (2048,))
    with pytest.raises(ValueError, match=r"got \(2048,\)"):
        _feature_loader(buf, "pixel_ae")


def test_the_refusal_names_backbone_expected_got_and_path(tmp_path):
    """Each of the four is asserted on its own, in labelled form: a message
    that carried both shapes but swapped `expected` and `got` would still
    contain both substrings, and would send the user to re-cache the wrong
    backbone."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 384))
    feature_path = buf.episode_paths()[0].with_suffix(".features_pixel_ae.npy")
    with pytest.raises(ValueError) as excinfo:
        _feature_loader(buf, "pixel_ae")
    message = str(excinfo.value)
    assert "'pixel_ae'" in message
    assert "expected rows of shape (64, 32)" in message
    assert "got (64, 384)" in message
    assert str(feature_path) in message


@pytest.mark.parametrize("backbone,rows", _GEOMETRY_CASES)
def test_every_registered_backbone_loads_a_cache_of_its_own_geometry(
    tmp_path, backbone, rows
):
    """The complement of the refusals: the guard must admit the shape the
    backbone actually writes, for EVERY backbone, or `pixel_ae` -- the one
    with the unusual width -- is the one a hardcoded `(64, 384)` rejects."""
    buf = _buffer_with_cache(tmp_path, backbone, rows)
    batch = _feature_loader(buf, backbone).sample()
    assert batch["features"].shape == (2, 17, *rows)


def test_the_guard_fires_on_the_first_cache_it_reads(tmp_path):
    """Every fixture above holds one episode, so "first", "last" and "any"
    are indistinguishable there. Two episodes, of which only the first is
    malformed: the guard must open the first file, not only the last."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 32), n_episodes=2)
    first = buf.episode_paths()[0].with_suffix(".features_pixel_ae.npy")
    np.save(first, np.zeros((17, 64, 384), dtype=np.float16))
    with pytest.raises(ValueError, match="does not match backbone") as excinfo:
        _feature_loader(buf, "pixel_ae")
    assert str(first) in str(excinfo.value)


def test_the_guard_fires_on_a_later_cache_too(tmp_path):
    """THE OTHER HALF: every file, not the first. The guard used to check
    file 0 only, on the reasoning that one `cache_features.py` run writes one
    geometry for every episode -- true of a completed run, and false of an
    interrupted `--clear` or a partial re-cache, which leave a directory that
    is right at file 0 and wrong at file 60. That directory used to fail
    minutes into a cell inside `BottleneckEncoder.forward`. Two episodes, of
    which only the SECOND is malformed, and the message names that file."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 32), n_episodes=2)
    first, second = (
        path.with_suffix(".features_pixel_ae.npy") for path in buf.episode_paths()
    )
    np.save(second, np.zeros((17, 64, 384), dtype=np.float16))
    assert np.load(first, mmap_mode="r").shape == (17, 64, 32), (
        "fixture: the first cache must be the good one, or this is the test above")
    with pytest.raises(ValueError, match="does not match backbone") as excinfo:
        _feature_loader(buf, "pixel_ae")
    assert str(second) in str(excinfo.value)
    assert str(first) not in str(excinfo.value)


def test_the_geometry_check_reads_the_header_not_the_frames(tmp_path):
    """The cache is memmapped, not loaded: the module docstring promises it
    (0.04 GB to map all 122 against 2.93 GB to read them) and this is what
    pins `mmap_mode="r"` on the `np.load`. It is all this test can see -- a
    guard that materialised the array internally and then kept the memmap
    would still pass it."""
    buf = _buffer_with_cache(tmp_path, "pixel_ae", (64, 32))
    loader = _feature_loader(buf, "pixel_ae")
    assert isinstance(loader._features[0], np.memmap)


def test_an_unregistered_backbone_is_a_keyerror_listing_the_registry(tmp_path):
    """No cache exists for a name that is not a backbone, so this also pins
    the ORDER of the checks: the name is validated before the file lookup.
    Otherwise a typo raises FileNotFoundError and tells the user to run
    `cache_features.py --backbone <typo>`, which rejects the same name one
    step later."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    with pytest.raises(KeyError) as excinfo:
        _feature_loader(buf, "vit_huge")
    message = str(excinfo.value)
    assert "'vit_huge'" in message
    for registered in ("dinov2", "random_vit", "pixel_ae"):
        assert registered in message


# --- Value-level window alignment -------------------------------------------
# The fixtures above fill actions/rewards/terminated/truncated with zeros and
# assert only shape and dtype, so four separate single-line mutations in
# SequenceLoader.sample() -- swapping truncated for terminated, slicing actions
# or rewards from a fixed offset 0, or zeroing terminated -- all passed the
# whole suite. Nothing tied those arrays to the sampled window. The episodes
# below encode (episode, timestep) in every field, so any misalignment or
# cross-wiring shows up as a value mismatch.


def make_varied_episode(t: int, ep_id: int) -> Episode:
    """Every field encodes its own index, so misalignment is visible."""
    steps = np.arange(t, dtype=np.int64)
    obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
    obs[:, 0, 0, 0] = np.arange(t + 1, dtype=np.uint8)  # timestep
    obs[:, 0, 0, 1] = ep_id  # episode identity
    return Episode(
        obs=obs,
        actions=(steps % 8).astype(np.int32),
        rewards=(steps.astype(np.float32) + 0.5 * ep_id),
        # Deliberately different periods: a terminated/truncated swap changes
        # the values, not just their meaning.
        terminated=(steps % 7 == 0),
        truncated=(steps % 5 == 0),
        privileged=np.full((t + 1, len(KEYS)), float(ep_id), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=ep_id,
        scenario="my_way_home",
    )


@pytest.fixture
def varied_buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for ep_id in (1, 2, 3, 4):
        buf.add(make_varied_episode(t=80, ep_id=ep_id))
    return buf


def _expected(path, start, seq_len):
    ep = load_episode(path)
    end = start + seq_len
    return ep.actions[start:end], ep.rewards[start:end], ep.terminated[start:end], ep.truncated[start:end]


def test_actions_and_rewards_come_from_the_sampled_window(varied_buffer):
    """Slicing from a fixed offset instead of window_start must fail here."""
    loader = SequenceLoader(varied_buffer, batch_size=6, seq_len=12, seed=0)
    for _ in range(5):
        batch = loader.sample()
        for i in range(6):
            idx = int(batch["episode_index"][i])
            start = int(batch["window_start"][i])
            actions, rewards, _, _ = _expected(loader.episode_path(idx), start, 12)
            np.testing.assert_array_equal(batch["actions"][i], actions)
            np.testing.assert_allclose(batch["rewards"][i], rewards)


def test_terminated_and_truncated_are_distinct_and_window_aligned(varied_buffer):
    """Invariant 4: the two flags must never be conflated.

    Time-limit bootstrapping depends on the distinction -- a truncated episode
    still bootstraps from its final value, a terminated one does not.
    """
    loader = SequenceLoader(varied_buffer, batch_size=6, seq_len=12, seed=0)
    saw_disagreement = False
    for _ in range(5):
        batch = loader.sample()
        for i in range(6):
            idx = int(batch["episode_index"][i])
            start = int(batch["window_start"][i])
            _, _, terminated, truncated = _expected(loader.episode_path(idx), start, 12)
            np.testing.assert_array_equal(batch["terminated"][i], terminated)
            np.testing.assert_array_equal(batch["truncated"][i], truncated)
            if (batch["terminated"][i] != batch["truncated"][i]).any():
                saw_disagreement = True
    assert saw_disagreement, (
        "fixture is degenerate: terminated and truncated never differ, so a "
        "swap between them would still pass"
    )


def test_obs_window_values_match_the_sampled_offset(varied_buffer):
    """Pins obs to window_start by value, not just by shape."""
    loader = SequenceLoader(varied_buffer, batch_size=6, seq_len=12, seed=0)
    batch = loader.sample()
    for i in range(6):
        idx = int(batch["episode_index"][i])
        start = int(batch["window_start"][i])
        expected = load_episode(loader.episode_path(idx)).obs[start : start + 13]
        np.testing.assert_array_equal(batch["obs"][i], expected)
        # The timestep marker must actually advance, or the check is vacuous.
        assert len(np.unique(batch["obs"][i][:, 0, 0, 0])) > 1
