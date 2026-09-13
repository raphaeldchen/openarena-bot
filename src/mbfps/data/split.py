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

VAL_FRACTION = 0.2
"""The held-out share, shared by training and evaluation rather than duplicated.

`train_world_model`, `mbfps.eval.study.run_job` and `scripts/eval_rollout.py`
must all split IDENTICALLY, or the evaluation scores a model on episodes it
trained on. The seed half of that coupling already has a name
(`study.SPLIT_SEED`); this is the other half. It was a bare `0.2` literal at
every call site, and a drift in any one of them is invisible to every test that
only checks the two sides are disjoint and sum to the whole: at 0.4 the
fixture's held-out set grows from ['ep003'] to ['ep002', 'ep003'] and 'ep002' is
an episode training saw.

All three callers are pinned by a test that moves this constant and requires the
call site to follow it -- equality to 0.2 would pass again the moment someone
re-hardcodes the literal.
"""


def episode_split(
    paths: list[Path], val_fraction: float = VAL_FRACTION, seed: int = 0
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
