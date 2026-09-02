"""Sequence sampling for world-model training.

Windows never span an episode boundary: a window that stitched the end of one
episode to the start of another would teach the RSSM a transition the engine can
never produce.
"""

import numpy as np

from mbfps.data.buffer import ReplayBuffer


class SequenceLoader:
    """Samples `(B, T)` windows from episodes held in a `ReplayBuffer`."""

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

    def refresh(self) -> None:
        """Reload episodes from disk. Call after new data is collected."""
        self._episodes = self.buffer.load_all()

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
