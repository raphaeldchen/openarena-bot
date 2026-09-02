"""Environment construction by name."""

from typing import Any, Callable

from mbfps.envs.protocol import EnvProtocol

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


try:  # pragma: no cover - exercised once vizdoom_env exists
    _register_builtins()
except ImportError:
    pass
