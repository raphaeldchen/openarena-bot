# src/mbfps/data/loader.py
"""Sequence sampling for world-model training.

Windows never span an episode boundary: a window that stitched the end of one
episode to the start of another would teach the RSSM a transition the engine
can never produce.

The study's arms never read pixels at all -- they train on cached features
from a frozen backbone; only M2's end-to-end `cnn` encoder reads obs. Loading
obs for a feature arm costs 2.24 GB of resident memory for nothing (measured),
so `load_obs=False` skips it. `.npz` decompresses lazily per key, so not
reading `obs` genuinely avoids the cost.

Feature files are memmapped rather than loaded: mapping all 122 costs 0.04 s
and 0.04 GB, versus 2.93 GB to read them. The first file's row geometry is
checked against the backbone registry at construction, from the header alone:
a cache is a bare `.npy` that does not record which backbone wrote it, and
with two geometries in play -- (64, 384) for the ViTs, (64, 32) for the M2
pixel autoencoder -- a wrong-suffix cache used to surface as a matmul error
in the encoder at step 0, after the loading time was paid.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import load_episode

logger = logging.getLogger(__name__)

_WARN_BYTES = 4_000_000_000
"""Warn above this resident footprint.

One episode's obs is ~19.8 MB, so the default capacity of 60,000 transitions
(~122 episodes) sits near 2.24 GB. The threshold is set above that so the
warning fires only when a run is genuinely at risk on the 16 GB shared budget,
rather than on every ordinary run.
"""

_DEFAULT_BACKBONE = "dinov2"


def feature_suffix(backbone: str) -> str:
    """Sibling filename suffix for `backbone`'s cached features.

    `dinov2` keeps the bare `.features.npy` that M1 already wrote, so the
    existing 122 files stay valid. Every other backbone is namespaced, because
    two caches now coexist and a shared name would let one silently overwrite
    the other -- leaving arms 2 and 3 training on identical inputs, with no
    error to notice.
    """
    if backbone == _DEFAULT_BACKBONE:
        return ".features.npy"
    return f".features_{backbone}.npy"


def _check_feature_geometry(
    features: np.ndarray, backbone: str, expected: tuple[int, int], path: Path
) -> None:
    """Refuse a cache whose rows are not `backbone`'s registered geometry.

    `features` is the memmap `np.load(..., mmap_mode="r")` returned, so
    `.shape` comes from the `.npy` header and no frame is read. The message
    names all four things a reader needs -- which backbone the suffix
    promised, what that backbone writes, what the file holds, and which file
    -- because the fix is always "re-cache this backbone" and the user must
    not have to reopen the file to know that.

    Raises:
        ValueError: if `features.shape[1:] != expected`.
    """
    got = tuple(features.shape[1:])
    if got != tuple(expected):
        raise ValueError(
            f"feature cache {path} does not match backbone {backbone!r}: "
            f"expected rows of shape {tuple(expected)}, got {got}. "
            f"Re-run scripts/cache_features.py --backbone {backbone}."
        )


class SequenceLoader:
    """Samples `(B, T)` windows from episodes held in a `ReplayBuffer`."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 16,
        seq_len: int = 64,
        seed: int = 0,
        load_obs: bool = True,
        load_features: bool = False,
        feature_backbone: str = _DEFAULT_BACKBONE,
        paths: list[Path] | None = None,
    ) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.load_obs = load_obs
        self.load_features = load_features
        self.feature_backbone = feature_backbone
        self._rng = np.random.default_rng(seed)

        # Keep paths and episodes index-aligned by construction rather than by
        # relying on load_all() happening to preserve order.
        self._paths: list[Path] = (
            list(paths) if paths is not None else buffer.episode_paths()
        )
        self._episodes = [self._read(p) for p in self._paths]

        self._features: list[np.ndarray | None] = []
        if load_features:
            # Imported here rather than at module top: `mbfps.data.features`
            # pulls in `transformers` (measured 1.53 s against this module's
            # 0.09 s), and the pixel `cnn` kind and M2's autoencoder tools
            # construct loaders that never read a cache. The registry is the
            # one source of truth for a backbone's row geometry --
            # `BottleneckEncoder` reads the same dict -- so what the loader
            # admits is exactly what the encoder's `Linear` can consume.
            from mbfps.data.features import BACKBONE_GEOMETRY

            # Before the file loop: an unregistered name has no cache either,
            # and the FileNotFoundError below would send the user to
            # `cache_features.py --backbone <typo>`, which refuses the same
            # name one step later.
            if feature_backbone not in BACKBONE_GEOMETRY:
                raise KeyError(
                    f"unknown feature backbone {feature_backbone!r}; "
                    f"available: {list(BACKBONE_GEOMETRY)}"
                )
            expected = BACKBONE_GEOMETRY[feature_backbone]
            suffix = feature_suffix(feature_backbone)
            for index, path in enumerate(self._paths):
                feature_path = path.with_suffix(suffix)
                if not feature_path.is_file():
                    raise FileNotFoundError(
                        f"no cached features for {path.name}; expected "
                        f"{feature_path.name}. Run scripts/cache_features.py "
                        f"--backbone {feature_backbone} first."
                    )
                features = np.load(feature_path, mmap_mode="r")
                # The first file only. One `cache_features.py` invocation
                # writes one backbone's cache for every episode, so the
                # geometry is a property of the cache, not of a file, and
                # one error at one place is what a reader wants.
                if index == 0:
                    _check_feature_geometry(
                        features, feature_backbone, expected, feature_path
                    )
                self._features.append(features)

        if load_obs:
            total = sum(ep.obs.nbytes for ep in self._episodes)
            if total > _WARN_BYTES:
                logger.warning(
                    "SequenceLoader holds %d episodes (%.2f GB) resident in RAM. "
                    "Episodes are loaded eagerly at construction. Pass "
                    "load_obs=False for arms that train on cached features.",
                    len(self._episodes),
                    total / 1e9,
                )

    def _read(self, path: Path):
        """Load one episode, skipping its obs array when pixels are not needed."""
        if self.load_obs:
            return load_episode(path)
        return _EpisodeMeta.from_npz(path)

    def episode_path(self, index: int) -> Path:
        """File backing the episode a batch's `episode_index` refers to.

        Use it to locate the sibling feature cache for a window.
        """
        return self._paths[index]

    def _usable(self) -> list[int]:
        return [i for i, ep in enumerate(self._episodes) if ep.length >= self.seq_len]

    def sample(self, include_privileged: bool = False) -> dict[str, Any]:
        """Sample one batch.

        Args:
            include_privileged: include ground-truth engine state. EVALUATION
                PROBES ONLY -- passing True in a training loop invalidates the
                study. Defaults to False so the safe path needs no thought.

        Raises:
            ValueError: if no episode is at least `seq_len` transitions long.
        """
        usable = self._usable()
        if not usable:
            raise ValueError(
                f"no episodes long enough for seq_len={self.seq_len}; "
                f"buffer holds {len(self._episodes)} episodes"
            )

        obs, actions, rewards = [], [], []
        terminated, truncated, privileged, features = [], [], [], []
        indices, starts = [], []

        for _ in range(self.batch_size):
            idx = int(self._rng.choice(usable))
            ep = self._episodes[idx]
            start = int(self._rng.integers(0, ep.length - self.seq_len + 1))
            end = start + self.seq_len
            actions.append(ep.actions[start:end])
            rewards.append(ep.rewards[start:end])
            terminated.append(ep.terminated[start:end])
            truncated.append(ep.truncated[start:end])
            indices.append(idx)
            starts.append(start)
            if self.load_obs:
                obs.append(ep.obs[start : end + 1])
            if self.load_features:
                features.append(np.asarray(self._features[idx][start : end + 1]))
            if include_privileged:
                privileged.append(ep.privileged[start : end + 1])

        batch: dict[str, Any] = {
            "actions": np.stack(actions).astype(np.int32),
            "rewards": np.stack(rewards).astype(np.float32),
            "terminated": np.stack(terminated).astype(bool),
            "truncated": np.stack(truncated).astype(bool),
            "episode_index": np.asarray(indices, dtype=np.int32),
            "window_start": np.asarray(starts, dtype=np.int32),
        }
        if self.load_obs:
            batch["obs"] = np.stack(obs).astype(np.uint8)
        if self.load_features:
            batch["features"] = np.stack(features)
        if include_privileged:
            batch["privileged"] = np.stack(privileged).astype(np.float32)
        return batch


class _EpisodeMeta:
    """An episode with everything except its obs array.

    `np.load` on an npz returns a lazy handle, so the pixel data is never
    decompressed when only these fields are read.
    """

    __slots__ = (
        "actions", "rewards", "terminated", "truncated",
        "privileged", "privileged_keys", "policy_name", "seed", "scenario",
    )

    @classmethod
    def from_npz(cls, path: Path) -> "_EpisodeMeta":
        meta = cls()
        with np.load(path) as data:
            meta.actions = data["actions"]
            meta.rewards = data["rewards"]
            meta.terminated = data["terminated"]
            meta.truncated = data["truncated"]
            meta.privileged = data["privileged"]
            meta.privileged_keys = tuple(data["privileged_keys"].tolist())
            meta.policy_name = str(data["policy_name"])
            meta.seed = int(data["seed"])
            meta.scenario = str(data["scenario"])
        return meta

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])
