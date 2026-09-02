"""Fixed-capacity, disk-backed episode store.

Capacity is measured in transitions rather than episodes because episode length
varies with policy and scenario; transitions are what actually bound disk use
and training-set size.

Filenames carry both the ordering index and the episode length
(`ep_000042_len00318.npz`), so capacity accounting never has to open a file.
"""

import itertools
import logging
import re
from pathlib import Path

from mbfps.data.episode import Episode, load_episode, save_episode

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^ep_(\d+)_len(\d+)\.npz$")


class ReplayBuffer:
    """Episodes on disk, oldest evicted once capacity is exceeded."""

    def __init__(self, root: Path, capacity_transitions: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_transitions = capacity_transitions
        indices = [self._parse(p)[0] for p in self.episode_paths()]
        self._counter = itertools.count(max(indices) + 1 if indices else 0)
        strays = self.stray_paths()
        if strays:
            examples = ", ".join(p.name for p in strays[:3])
            logger.warning(
                "%d file(s) in %s match the episode glob but not the episode "
                "filename pattern; they are excluded from all counts and from "
                "eviction and will accumulate forever (e.g. %s)",
                len(strays),
                self.root,
                examples,
            )

    @staticmethod
    def _parse(path: Path) -> tuple[int, int]:
        """Return (index, length) parsed from an episode filename."""
        match = _NAME_RE.match(path.name)
        if match is None:
            raise ValueError(f"malformed episode filename: {path.name}")
        return int(match.group(1)), int(match.group(2))

    def episode_paths(self) -> list[Path]:
        """Episode files, oldest first."""
        return sorted(
            (p for p in self.root.glob("ep_*.npz") if _NAME_RE.match(p.name)),
            key=lambda p: self._parse(p)[0],
        )

    def stray_paths(self) -> list[Path]:
        """Files matching the episode glob whose names this class cannot parse.

        They are excluded from every count AND from eviction, so they accumulate
        forever. Deleting them automatically would risk destroying something a
        user placed here, so they are surfaced instead.
        """
        return sorted(
            p for p in self.root.glob("ep_*.npz") if not _NAME_RE.match(p.name)
        )

    @property
    def n_episodes(self) -> int:
        return len(self.episode_paths())

    @property
    def n_transitions(self) -> int:
        """Total transitions stored, read from filenames without decompressing."""
        return sum(self._parse(p)[1] for p in self.episode_paths())

    def add(self, ep: Episode) -> Path:
        """Write `ep` and evict oldest episodes until within capacity."""
        path = self._next_path(ep.length)
        save_episode(ep, path)
        self._evict()
        return path

    def _next_path(self, length: int) -> Path:
        """Reserve the next unused index, resyncing if another writer advanced.

        The counter alone is not enough: a second ReplayBuffer over the same
        directory seeds its own counter at construction and can hand out an
        index this one has already used. Length is part of the filename, so a
        bare exists() check would miss the collision.
        """
        while True:
            index = next(self._counter)
            if not any(self.root.glob(f"ep_{index:06d}_len*.npz")):
                return self.root / f"ep_{index:06d}_len{length:05d}.npz"

    def load_all(self) -> list[Episode]:
        """Load every stored episode, oldest first.

        This is the one place a filename/content divergence is detectable: the
        encoded length is otherwise trusted as-is (by `n_transitions` and
        `_evict`) to keep capacity accounting O(1), but `load_all` already pays
        the decompression cost for every episode, so verifying here is free.
        """
        episodes = []
        for path in self.episode_paths():
            ep = load_episode(path)
            _, encoded_length = self._parse(path)
            if ep.length != encoded_length:
                raise ValueError(
                    f"{path.name}: filename encodes length {encoded_length}, "
                    f"but the episode actually has {ep.length} transitions"
                )
            episodes.append(ep)
        return episodes

    def _evict(self) -> None:
        paths = self.episode_paths()
        lengths = [self._parse(p)[1] for p in paths]
        total = sum(lengths)
        # Never evict the newest episode: an empty buffer is worse than an
        # oversized one.
        for path, length in zip(paths[:-1], lengths[:-1]):
            if total <= self.capacity_transitions:
                break
            path.unlink()
            path.with_suffix(".features.npy").unlink(missing_ok=True)
            total -= length
