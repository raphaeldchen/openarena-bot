"""Fixed-capacity, disk-backed episode store.

Capacity is measured in transitions rather than episodes because episode length
varies with policy and scenario; transitions are what actually bound disk use
and training-set size.

Filenames carry both the ordering index and the episode length
(`ep_000042_len00318.npz`), so capacity accounting never has to open a file.
"""

import itertools
import re
from pathlib import Path

from mbfps.data.episode import Episode, load_episode, save_episode

_NAME_RE = re.compile(r"^ep_(\d+)_len(\d+)\.npz$")


class ReplayBuffer:
    """Episodes on disk, oldest evicted once capacity is exceeded."""

    def __init__(self, root: Path, capacity_transitions: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_transitions = capacity_transitions
        indices = [self._parse(p)[0] for p in self.episode_paths()]
        self._counter = itertools.count(max(indices) + 1 if indices else 0)

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

    @property
    def n_episodes(self) -> int:
        return len(self.episode_paths())

    @property
    def n_transitions(self) -> int:
        """Total transitions stored, read from filenames without decompressing."""
        return sum(self._parse(p)[1] for p in self.episode_paths())

    def add(self, ep: Episode) -> Path:
        """Write `ep` and evict oldest episodes until within capacity."""
        path = self.root / f"ep_{next(self._counter):06d}_len{ep.length:05d}.npz"
        save_episode(ep, path)
        self._evict()
        return path

    def load_all(self) -> list[Episode]:
        """Load every stored episode, oldest first."""
        return [load_episode(p) for p in self.episode_paths()]

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
