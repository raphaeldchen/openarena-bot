"""Environment construction by name."""

import logging
from typing import Any, Callable

from mbfps.envs.protocol import EnvProtocol

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., EnvProtocol]] = {}


def register(name: str, factory: Callable[..., EnvProtocol]) -> None:
    """Register an environment factory under `name`."""
    _REGISTRY[name] = factory


def available() -> list[str]:
    """Return the sorted names of registered environments."""
    return sorted(_REGISTRY)


def make_env(name: str, **kwargs: Any) -> EnvProtocol:
    """Construct a registered environment.

    Raises:
        KeyError: if `name` is not registered. The message lists what is.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown environment {name!r}; available: {available()}"
        )
    return _REGISTRY[name](**kwargs)


def _register_builtins() -> None:
    from mbfps.envs.vizdoom_env import ViZDoomEnv

    register("vizdoom", ViZDoomEnv)


def _try_register_builtins() -> None:
    """Register built-in environments, tolerating ones not yet written.

    `vizdoom_env` does not exist until Task 5. Any other import failure -- a
    missing third-party package, a broken native library, a misspelled class
    name -- must propagate: swallowing it would leave an empty registry and
    report a real bug as "unknown environment", sending a debugger to the
    wrong file.
    """
    try:
        _register_builtins()
    except ModuleNotFoundError as exc:
        if exc.name != "mbfps.envs.vizdoom_env":
            raise
        logger.debug("mbfps.envs.vizdoom_env not available yet; registry left empty")


_try_register_builtins()
