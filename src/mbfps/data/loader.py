"""Sequence sampling for world-model training.

Windows never span an episode boundary: a window that stitched the end of one
episode to the start of another would teach the RSSM a transition the engine can
never produce.
"""

import logging

import numpy as np

from mbfps.data.buffer import ReplayBuffer

logger = logging.getLogger(__name__)

# One real episode (my_way_home, 526 frames of 112x112x3 uint8) is ~19.8 MB.
# 2 GB is roughly 100 such episodes -- comfortably inside a training run's
# working set on a 16 GB unified-memory machine, but large enough to be worth
# a warning rather than silence.
_WARN_BYTES = 2_000_000_000


class SequenceLoader:
    """Samples `(B, T)` windows from episodes held in a `ReplayBuffer`.

    The entire buffer is loaded eagerly at construction and held resident in
    RAM for the lifetime of this object -- there is no streaming path. Each
    episode costs roughly 19.8 MB (526 frames of 112x112x3 uint8), so a
    `ReplayBuffer` sized for hundreds of thousands of transitions can occupy
    several GB. See `_WARN_BYTES` below.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 16,
        seq_len: int = 64,
        seed: int = 0,
    ) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed)
        self._episodes = buffer.load_all()

        total_bytes = sum(ep.obs.nbytes for ep in self._episodes)
        if total_bytes > _WARN_BYTES:
            logger.warning(
                "SequenceLoader holds %d episodes (%.2f GB) resident in RAM. "
                "Episodes are loaded eagerly at construction; a streaming loader "
                "is deferred to a later plan. Reduce ReplayBuffer capacity if "
                "this competes with model memory.",
                len(self._episodes),
                total_bytes / 1e9,
            )

    def _usable(self) -> list[int]:
        return [i for i, ep in enumerate(self._episodes) if ep.length >= self.seq_len]

    def sample(self, include_privileged: bool = False) -> dict[str, np.ndarray]:
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

        obs, actions, rewards, indices = [], [], [], []
        terminated, truncated, privileged = [], [], []
        for _ in range(self.batch_size):
            idx = int(self._rng.choice(usable))
            ep = self._episodes[idx]
            start = int(self._rng.integers(0, ep.length - self.seq_len + 1))
            end = start + self.seq_len
            obs.append(ep.obs[start : end + 1])
            actions.append(ep.actions[start:end])
            rewards.append(ep.rewards[start:end])
            terminated.append(ep.terminated[start:end])
            truncated.append(ep.truncated[start:end])
            indices.append(idx)
            if include_privileged:
                privileged.append(ep.privileged[start : end + 1])

        batch = {
            "obs": np.stack(obs).astype(np.uint8),
            "actions": np.stack(actions).astype(np.int32),
            "rewards": np.stack(rewards).astype(np.float32),
            "terminated": np.stack(terminated).astype(bool),
            "truncated": np.stack(truncated).astype(bool),
            "episode_index": np.asarray(indices, dtype=np.int32),
        }
        if include_privileged:
            batch["privileged"] = np.stack(privileged).astype(np.float32)
        return batch
