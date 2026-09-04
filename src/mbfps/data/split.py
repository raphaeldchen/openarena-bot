"""Deterministic held-out split, at episode granularity.

Splitting at window granularity would leak: windows drawn from one episode
overlap in frames, so a window in validation can share pixels with a window in
training and the held-out error would be optimistic.

The split is a pure function of the episode *names* and the seed, not of the
order `episode_paths()` happens to return -- an ordering change would otherwise
silently move episodes across the boundary and invalidate comparisons against
previously recorded numbers.
"""

from pathlib import Path

import numpy as np


def episode_split(
    paths: list[Path], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Partition `paths` into (train, val).

    Args:
        paths: episode files, in any order.
        val_fraction: share held out, rounded down but forced to at least one.
        seed: fixes the partition. The SAME seed must be used for every arm.

    Raises:
        ValueError: if the split cannot yield at least one episode on each side.
    """
    ordered = sorted(paths, key=lambda p: p.name)
    n_val = int(len(ordered) * val_fraction)
    if n_val < 1 or len(ordered) - n_val < 1:
        raise ValueError(
            f"need at least one episode on each side; {len(ordered)} episodes at "
            f"val_fraction={val_fraction} gives {n_val} validation episodes"
        )
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    val_index = set(permutation[:n_val].tolist())
    train = [p for i, p in enumerate(ordered) if i not in val_index]
    val = [p for i, p in enumerate(ordered) if i in val_index]
    return train, val
