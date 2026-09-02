"""Episode container and on-disk format.

One compressed `.npz` per episode. `obs` holds T+1 frames for T transitions, so
a window of length T yields T aligned (s_t, a_t, r_t, s_{t+1}) tuples.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    """A single recorded episode."""

    obs: np.ndarray  # (T+1, 112, 112, 3) uint8
    actions: np.ndarray  # (T,) int32
    rewards: np.ndarray  # (T,) float32
    terminated: np.ndarray  # (T,) bool -- a true terminal state
    truncated: np.ndarray  # (T,) bool -- a time-limit cutoff, NOT terminal
    privileged: np.ndarray  # (T+1, K) float32 -- EVALUATION ONLY
    privileged_keys: tuple[str, ...]
    policy_name: str
    seed: int
    scenario: str

    def __post_init__(self) -> None:
        t = self.actions.shape[0]
        if self.obs.shape[0] != t + 1:
            raise ValueError(
                f"obs must have one more entry than actions; "
                f"got {self.obs.shape[0]} obs and {t} actions"
            )
        for name, arr, expected in (
            ("rewards", self.rewards, t),
            ("terminated", self.terminated, t),
            ("truncated", self.truncated, t),
            ("privileged", self.privileged, t + 1),
        ):
            if arr.shape[0] != expected:
                raise ValueError(
                    f"{name} must have length {expected}, got {arr.shape[0]}"
                )

    @property
    def length(self) -> int:
        """Number of transitions (T)."""
        return int(self.actions.shape[0])


def save_episode(ep: Episode, path: Path) -> None:
    """Write `ep` to `path` as a compressed npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=ep.obs,
        actions=ep.actions,
        rewards=ep.rewards,
        terminated=ep.terminated,
        truncated=ep.truncated,
        privileged=ep.privileged,
        privileged_keys=np.array(ep.privileged_keys, dtype=object),
        policy_name=ep.policy_name,
        seed=ep.seed,
        scenario=ep.scenario,
    )


def load_episode(path: Path) -> Episode:
    """Read an episode written by `save_episode`."""
    # allow_pickle=True is required because privileged_keys is stored as an
    # object-dtype array of strings. Episodes are files this pipeline writes
    # to its own data/ directory, not artifacts from an untrusted source.
    with np.load(path, allow_pickle=True) as data:
        return Episode(
            obs=data["obs"],
            actions=data["actions"],
            rewards=data["rewards"],
            terminated=data["terminated"],
            truncated=data["truncated"],
            privileged=data["privileged"],
            privileged_keys=tuple(data["privileged_keys"].tolist()),
            policy_name=str(data["policy_name"]),
            seed=int(data["seed"]),
            scenario=str(data["scenario"]),
        )
