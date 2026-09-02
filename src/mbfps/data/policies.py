"""Collection policies.

Action indices follow `mbfps.envs.actions.build_action_set`: index 0 is the
no-op and index `i + 1` presses button `i`.
"""

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Something that maps an observation to a discrete action."""

    name: str

    def act(self, obs: np.ndarray) -> int:
        """Return an action index for `obs`."""
        ...

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode, reseeding when `seed` is given."""
        ...


class RandomPolicy:
    """Uniform random actions. Unbiased coverage near the start distribution."""

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.name = "random"
        self._n = n_actions
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, self._n))

    def reset(self, seed: int | None = None) -> None:
        """Reseed for a new episode.

        The collector passes its per-episode seed here. Rewinding to a fixed
        seed instead would make every episode replay one identical action
        sequence, collapsing the dataset to a single trajectory.
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)


class ScriptedPolicy:
    """Committed movement with a periodic sweep, adapted to available buttons.

    Random play oscillates near the spawn point. This policy commits to an
    advance action and sweeps the view, so the world model sees corridors,
    corners and distant geometry that random play rarely reaches.

    Button sets differ per scenario, so the roles are resolved from whatever the
    engine exposes rather than hard-coded:

    - advance: MOVE_FORWARD if present, else None
    - sweep:   the first available (left, right) pair -- turning preferred,
               strafing as a fallback
    - attack:  ATTACK if present

    With no advance and no sweep (a degenerate button set) it falls back to
    uniform random, which is still better than emitting a constant.
    """

    _ADVANCE_PROB = 0.5
    _SWEEP_PROB = 0.75
    _SWEEP_LEN = 100

    def __init__(self, button_names: Sequence[str], seed: int = 0) -> None:
        self.name = "scripted"
        self._seed = seed
        self._n = len(button_names) + 1  # +1 for the no-op

        self._advance = self._index_of(button_names, "MOVE_FORWARD")
        self._sweep = self._first_pair(
            button_names, [("TURN_LEFT", "TURN_RIGHT"), ("MOVE_LEFT", "MOVE_RIGHT")]
        )
        self._attack = self._index_of(button_names, "ATTACK")
        self.reset(seed)

    @staticmethod
    def _index_of(names: Sequence[str], target: str) -> int | None:
        """Action index for `target`, or None if the button is unavailable."""
        for i, name in enumerate(names):
            if name == target:
                return i + 1
        return None

    @classmethod
    def _first_pair(
        cls, names: Sequence[str], candidates: Sequence[tuple[str, str]]
    ) -> tuple[int, int] | None:
        """First candidate pair whose buttons are both available."""
        for left, right in candidates:
            li, ri = cls._index_of(names, left), cls._index_of(names, right)
            if li is not None and ri is not None:
                return li, ri
        return None

    def act(self, obs: np.ndarray) -> int:
        self._t += 1
        if self._advance is not None and self._rng.random() < self._ADVANCE_PROB:
            return self._advance
        if self._sweep is not None and self._rng.random() < self._SWEEP_PROB:
            sweeping_left = (self._t // self._SWEEP_LEN) % 2 == 0
            return self._sweep[0] if sweeping_left else self._sweep[1]
        if self._attack is not None and self._rng.random() < 0.5:
            return self._attack
        return int(self._rng.integers(0, self._n))

    def reset(self, seed: int | None = None) -> None:
        """Restart the sweep and reseed for a new episode."""
        if seed is not None:
            self._seed = seed
        self._t = 0
        self._rng = np.random.default_rng(self._seed)
