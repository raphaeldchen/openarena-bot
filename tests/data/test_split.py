from pathlib import Path

import pytest

from mbfps.data.split import episode_split

PATHS = [Path(f"ep_{i:06d}_len00100.npz") for i in range(20)]


def test_split_is_deterministic():
    assert episode_split(PATHS, seed=0) == episode_split(PATHS, seed=0)


def test_split_differs_by_seed():
    assert episode_split(PATHS, seed=0) != episode_split(PATHS, seed=1)


def test_split_is_a_partition():
    train, val = episode_split(PATHS, val_fraction=0.25)
    assert sorted(train + val) == sorted(PATHS)
    assert set(train).isdisjoint(val)


def test_val_fraction_is_respected():
    train, val = episode_split(PATHS, val_fraction=0.25)
    assert len(val) == 5


def test_split_does_not_depend_on_input_order():
    """Episode ordering must not silently change which episodes are held out."""
    a = episode_split(PATHS, seed=0)
    b = episode_split(list(reversed(PATHS)), seed=0)
    assert sorted(a[1]) == sorted(b[1])


def test_split_is_identical_across_arms():
    """Arms must be compared on the same held-out episodes, or the comparison
    measures which episodes each arm happened to get."""
    splits = {arm: episode_split(PATHS, seed=0) for arm in ("cnn", "frozen_ssl", "random_vit")}
    assert len({tuple(sorted(v)) for _, v in splits.values()}) == 1


def test_empty_val_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        episode_split(PATHS[:2], val_fraction=0.01)


from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
from tests.data.test_loader import make_episode


def test_loader_restricted_to_paths_never_samples_outside_them(tmp_path):
    """If this fails the split is decorative: the world model would train on
    the very episodes the probe is later evaluated on."""
    buffer = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4):
        buffer.add(make_episode(t=80, fill=fill))
    train, _ = episode_split(buffer.episode_paths(), val_fraction=0.5, seed=0)

    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0, paths=train)
    allowed = {p.name for p in train}
    for _ in range(20):
        batch = loader.sample()
        for i in range(8):
            name = loader.episode_path(int(batch["episode_index"][i])).name
            assert name in allowed, f"sampled {name}, which is held out"


def test_loader_without_paths_uses_the_whole_buffer(tmp_path):
    buffer = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4):
        buffer.add(make_episode(t=80, fill=fill))
    assert len(SequenceLoader(buffer, seq_len=16)._paths) == 4


def test_loader_paths_subset_keeps_features_index_aligned_with_episodes(tmp_path):
    """A SUBSET passed via paths= must keep self._features aligned with
    self._episodes/self._paths by index, and episode_path(i) must resolve to
    the same episode whose features are stored at self._features[i].

    Regression target: if _features were built from buffer.episode_paths()
    (the full buffer) while _episodes/_paths were built from the restricted
    subset, index i would pull episode i's data but arm's feature vector
    would belong to a different, unrelated episode.
    """
    import numpy as np

    buffer = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    # t = 16 is the loader's seq_len below, the shortest usable episode, so
    # each (17, 64, 384) cache is 0.8 MB rather than the 4 MB of an 80-step one.
    for fill in (1, 2, 3, 4):
        buffer.add(make_episode(t=16, fill=fill))
    all_paths = buffer.episode_paths()
    for path in all_paths:
        # Encode each episode's identity into its own feature cache so a
        # misalignment between _paths and _features is visible by value.
        marker = float(path.stem.split("_len")[0].split("_")[1])
        # (64, 384) is dinov2's registered geometry; the loader refuses any
        # other row shape under the bare `.features.npy` suffix.
        feats = np.full((17, 64, 384), marker, dtype=np.float16)
        np.save(path.with_suffix(".features.npy"), feats)

    train, _ = episode_split(all_paths, val_fraction=0.5, seed=0)
    assert 1 <= len(train) < len(all_paths), "test needs a genuine subset"

    loader = SequenceLoader(
        buffer, batch_size=8, seq_len=16, seed=0, load_features=True, paths=train
    )
    assert len(loader._features) == len(train)

    for _ in range(20):
        batch = loader.sample()
        for i in range(8):
            idx = int(batch["episode_index"][i])
            path = loader.episode_path(idx)
            expected_marker = float(path.stem.split("_len")[0].split("_")[1])
            assert (batch["features"][i] == expected_marker).all(), (
                f"features for episode_index {idx} (path {path.name}) do not "
                f"match that episode's own cache -- _features is misaligned "
                f"with _paths/_episodes"
            )
