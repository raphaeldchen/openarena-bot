# src/mbfps/data/collector.py
"""Drive a policy through an environment and record episodes."""

import logging
from typing import Callable

import numpy as np

from mbfps.data.episode import Episode
from mbfps.data.policies import Policy
from mbfps.envs.protocol import EnvProtocol

logger = logging.getLogger(__name__)


class DataIntegrityError(Exception):
    """The collected data is malformed.

    Deliberately NOT caught by `collect_episode`'s crash handler. An engine
    crash is an external fault worth retrying; malformed data is our own bug,
    and recording it as a crash is exactly how an earlier version of this
    collector discarded every episode while reporting a healthy `crash_count`.
    """


class Collector:
    """Collects episodes, restarting the engine when it crashes."""

    def __init__(
        self,
        env_factory: Callable[[], EnvProtocol],
        policy: Policy,
        max_steps: int = 1000,
    ) -> None:
        self._env_factory = env_factory
        self._policy = policy
        self._max_steps = max_steps
        self._env = env_factory()
        self.crash_count = 0

    def collect_episode(self, seed: int) -> Episode | None:
        """Collect one episode, or None if the engine crashed.

        A crash discards the partial trajectory entirely: a truncated episode
        would look terminal to the loader and teach the world model that the
        world ends at an arbitrary point.
        """
        try:
            return self._collect(seed)
        except DataIntegrityError:
            raise
        except Exception:
            self.crash_count += 1
            logger.warning("engine crashed during collection; restarting", exc_info=True)
            self._restart()
            return None

    def _collect(self, seed: int) -> Episode:
        # Reseed the policy per episode. Rewinding it to a fixed seed would make
        # every episode replay one identical action sequence.
        self._policy.reset(seed=seed)
        obs, _ = self._env.reset(seed=seed)

        # Capture the key set now, while the episode is live. Querying it after
        # the loop would read a finished engine, whose state is None.
        state = self._env.privileged_state
        keys = tuple(state) if state else ()
        width = len(keys)

        frames = [obs.copy()]
        privileged = [self._row(state, keys, previous=None)]
        actions: list[int] = []
        rewards: list[float] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []

        for _ in range(self._max_steps):
            action = self._policy.act(obs)
            obs, reward, terminated, truncated, _ = self._env.step(action)
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            frames.append(obs.copy())
            privileged.append(
                self._row(self._env.privileged_state, keys, previous=privileged[-1])
            )
            if terminated or truncated:
                break

        # Fail loudly rather than letting np.stack raise into the crash handler,
        # which would silently misreport a data bug as an engine fault.
        bad = [i for i, row in enumerate(privileged) if row.shape != (width,)]
        if bad:
            raise DataIntegrityError(
                f"ragged privileged rows at indices {bad[:5]}; expected width {width}"
            )

        return Episode(
            obs=np.stack(frames).astype(np.uint8),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated_flags, dtype=bool),
            truncated=np.asarray(truncated_flags, dtype=bool),
            privileged=np.stack(privileged).astype(np.float32),
            privileged_keys=keys,
            policy_name=self._policy.name,
            seed=seed,
            scenario=getattr(self._env, "scenario", "unknown"),
        )

    @staticmethod
    def _row(
        state: dict[str, float] | None,
        keys: tuple[str, ...],
        previous: np.ndarray | None,
    ) -> np.ndarray:
        """One privileged row, carrying the last value forward when unavailable.

        The engine reports no state on the terminal frame, so the final row
        repeats the last live reading. Zeros would be a plausible-looking lie:
        position 0,0,0 is a real coordinate, and the M3 latent probe would fit
        against it.
        """
        if state:
            missing = [k for k in keys if k not in state]
            if missing:
                raise DataIntegrityError(
                    f"privileged state lost keys mid-episode: {missing}"
                )
            return np.asarray([state[k] for k in keys], dtype=np.float32)
        if previous is not None:
            return previous.copy()
        return np.zeros(len(keys), dtype=np.float32)

    def _restart(self) -> None:
        try:
            self._env.close()
        except Exception:
            logger.debug("close() failed on a crashed env; ignoring", exc_info=True)
        self._env = self._env_factory()

    def close(self) -> None:
        """Release the environment."""
        self._env.close()
