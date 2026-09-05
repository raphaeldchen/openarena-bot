"""ViZDoom implementation of `EnvProtocol`.

Rendering happens at ViZDoom's smallest resolution (160x120) and is downsampled
to 112x112 -- the engine is never asked to render pixels that are immediately
thrown away.
"""

from pathlib import Path
from typing import Any

import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from mbfps.envs.actions import build_action_set
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.envs.wrappers import preprocess_frame

# `register` is imported at the bottom of this module, not here, so that
# `ViZDoomEnv` is fully defined before `mbfps.envs.registry` re-enters this
# module. `registry`'s own import-time bootstrap does
# `from mbfps.envs.vizdoom_env import ViZDoomEnv`; if that re-entry happens
# while this module is still executing its own top-of-file imports (i.e.
# something imports `mbfps.envs.vizdoom_env` directly, before anything has
# imported `mbfps.envs.registry`), `ViZDoomEnv` would not exist on the
# partially-initialised module yet and the import would fail with
# `ImportError: cannot import name 'ViZDoomEnv' from partially initialized
# module` -- a plain `ImportError`, not `ModuleNotFoundError`, so registry's
# narrow except clause does not swallow it. Deferring this import until after
# the class statement below guarantees `ViZDoomEnv` is already an attribute
# of this module by the time any reentrant import looks for it, regardless of
# which module a caller imports first.

PRIVILEGED_KEYS: tuple[str, ...] = ("health", "pos_x", "pos_y", "pos_z", "angle")
"""Keys of `privileged_state`. EVALUATION ONLY -- never a training input."""

_PRIVILEGED_VARS: dict[str, "vzd.GameVariable"] = {
    "health": vzd.GameVariable.HEALTH,
    "pos_x": vzd.GameVariable.POSITION_X,
    "pos_y": vzd.GameVariable.POSITION_Y,
    "pos_z": vzd.GameVariable.POSITION_Z,
    "angle": vzd.GameVariable.ANGLE,
}


class ViZDoomEnv:
    """A headless, seedable ViZDoom environment."""

    def __init__(
        self,
        scenario: str = "my_way_home",
        frame_skip: int = 4,
        seed: int = 0,
        doom_skill: int | None = None,
    ) -> None:
        self.scenario = scenario
        self.frame_skip = frame_skip
        self._seed = seed
        self._closed = False

        self._game = vzd.DoomGame()
        config = Path(vzd.scenarios_path) / f"{scenario}.cfg"
        if not config.is_file():
            raise FileNotFoundError(f"no ViZDoom scenario config at {config}")
        self._game.load_config(str(config))
        self._game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self._game.set_screen_format(vzd.ScreenFormat.RGB24)
        self._game.set_window_visible(False)
        self._game.set_mode(vzd.Mode.PLAYER)
        if doom_skill is not None:
            # Difficulty drives episode length, which decides whether the
            # dataset can supply a 64-step training window at all. Keep it
            # explicit rather than inheriting whatever the .cfg happens to set.
            self._game.set_doom_skill(doom_skill)
        for var in _PRIVILEGED_VARS.values():
            self._game.add_available_game_variable(var)
        self._game.set_seed(seed)
        self._game.init()

        # add_available_game_variable de-duplicates against variables the config
        # already declares, so the final list length varies per scenario. Resolve
        # each key to its actual index by name; a fixed slice would silently read
        # the wrong column on a scenario that declares its own HEALTH.
        declared = [v.name for v in self._game.get_available_game_variables()]
        self._var_index = {
            key: declared.index(var.name) for key, var in _PRIVILEGED_VARS.items()
        }

        self._actions = build_action_set(len(self._game.get_available_buttons()))
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._actions))
        self._last_obs = np.zeros(OBS_SHAPE, dtype=np.uint8)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode, optionally reseeding first.

        `set_seed` after `init` is verified to reseed correctly in ViZDoom 1.3.0,
        so no engine restart is needed.
        """
        if seed is not None:
            self._seed = seed
            self._game.set_seed(seed)
        self._game.new_episode()
        self._last_obs = self._observe()
        return self._last_obs, {"scenario": self.scenario}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance `frame_skip` tics with the button vector for `action`.

        ViZDoom reports `is_episode_finished()` for both death and timeout. A
        timeout is a time-limit cutoff, not a true terminal state -- conflating
        them is the classic time-limit bootstrapping bug, and it would teach M3's
        continue predictor that the world ends when the clock runs out.
        """
        reward = float(self._game.make_action(self._actions[action], self.frame_skip))
        finished = bool(self._game.is_episode_finished())
        truncated = finished and self._timed_out()
        terminated = finished and not truncated
        if not finished:
            self._last_obs = self._observe()
        return self._last_obs, reward, terminated, truncated, {}

    def _timed_out(self) -> bool:
        timeout = self._game.get_episode_timeout()
        return timeout > 0 and self._game.get_episode_time() >= timeout

    def close(self) -> None:
        """Release the ViZDoom instance. Safe to call more than once."""
        if not self._closed:
            self._game.close()
            self._closed = True

    @property
    def privileged_state(self) -> dict[str, float] | None:
        """Ground-truth engine state, or None once the episode has finished.

        EVALUATION PROBES ONLY. Returns None on the terminal frame because
        ViZDoom's `get_state()` does; callers must handle that rather than
        assuming a row is always available.
        """
        state = self._game.get_state()
        if state is None:
            return None
        variables = state.game_variables
        return {k: float(variables[i]) for k, i in self._var_index.items()}

    @property
    def button_names(self) -> tuple[str, ...]:
        """Names of the scenario's available buttons, in button-vector order."""
        return tuple(b.name for b in self._game.get_available_buttons())

    def _observe(self) -> np.ndarray:
        state = self._game.get_state()
        if state is None:
            return self._last_obs
        return preprocess_frame(state.screen_buffer)


from mbfps.envs.registry import register  # noqa: E402  (see comment above imports)

register("vizdoom", ViZDoomEnv)
