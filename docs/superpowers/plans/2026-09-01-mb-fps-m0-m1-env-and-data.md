# MB-FPS M0–M1: Environment Spine and Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, headless ViZDoom environment behind a game-agnostic interface, and a mixed-policy data pipeline that writes verified episode data to disk with cached self-supervised features.

**Architecture:** A `EnvProtocol` interface isolates all game-engine specifics; `ViZDoomEnv` is its first implementation (OpenArena will be its second, at M7). A collector drives configurable policies through the env and writes one compressed `.npz` per episode. A fixed-capacity buffer evicts oldest episodes; a sequence loader samples `(B, T, ...)` windows that never cross episode boundaries. A frozen DINOv2-S/14 backbone caches patch features alongside frames.

**Tech Stack:** Python 3.12, PyTorch 2.x (MPS), ViZDoom 1.3.0, Gymnasium, NumPy, OpenCV, HuggingFace Transformers (DINOv2 only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-mb-fps-design.md`

## Global Constraints

- **Python 3.12** in a project venv at `.venv/`. The system Python 3.14 is left untouched.
- **Observations are `uint8`, shape `(112, 112, 3)`, HWC, RGB.** All arms consume byte-identical frames. 112 = 8 x 14 so DINOv2's patch-14 grid is exactly 8x8.
- **fp32 everywhere.** No fp16/bf16 autocast on MPS.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** must be set; every entry point accepts `--device {mps,cpu}` with `mps` the default and `cpu` the debugging escape hatch.
- **`privileged_state` is evaluation-only.** It must never appear in any tensor fed to a model. Enforced by a test, not by convention.
- **No network access at training time.** The only download is the DINOv2 checkpoint, fetched once in Task 12 and cached locally.
- **Default scenario is `my_way_home`** (buttons: `TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT`; no `ATTACK`). Chosen on measured episode length, not button count. With `seq_len=64`, every episode must exceed 64 transitions or the loader silently discards it. Measured over 12 episodes at `frame_skip=4`:

  | Scenario | random mean | scripted mean | episodes >= 65 |
  |---|---|---|---|
  | `deadly_corridor` (cfg `doom_skill=5`) | 32 | **16** | **1/12, 0/12** |
  | `defend_the_center` | 74 | 74 | 9/12 |
  | `health_gathering` | 111 | 105 | 12/12 |
  | **`my_way_home`** | **489** | **525** | **12/12** |

  `deadly_corridor` has the richest button set but its config sets `doom_skill = 5`; the agent dies in ~16 steps and the entire dataset falls below the training window. `my_way_home` is a maze-navigation task with long episodes and varied geometry.
- **`doom_skill` is an explicit environment parameter**, never left implicit in a scenario config, so episode length is a controlled variable rather than an accident.
- **Package name is `mbfps`** (the directory is `csgo-bot`; this is intentional and recorded in the spec's open questions).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest config |
| `src/mbfps/utils/device.py` | Device selection and MPS fallback env var |
| `src/mbfps/utils/seeding.py` | Seed every RNG from one integer |
| `src/mbfps/envs/protocol.py` | `EnvProtocol` — the modular engine hook |
| `src/mbfps/envs/actions.py` | Button-vector <-> discrete-index mapping |
| `src/mbfps/envs/vizdoom_env.py` | ViZDoom implementation of `EnvProtocol` |
| `src/mbfps/envs/registry.py` | `make_env(name, **kwargs)` |
| `src/mbfps/data/policies.py` | `RandomPolicy`, `ScriptedPolicy` |
| `src/mbfps/data/episode.py` | `Episode` dataclass + npz read/write |
| `src/mbfps/data/collector.py` | Policy -> episodes, with crash recovery |
| `src/mbfps/data/buffer.py` | Fixed-capacity episode store with eviction |
| `src/mbfps/data/loader.py` | `(B, T, ...)` sequence sampling |
| `src/mbfps/data/features.py` | Frozen DINOv2 patch-feature cache |
| `scripts/collect.py` | CLI entry point for data collection |
| `scripts/coverage_report.py` | Per-policy state-visitation histogram |

---

## Task 1: Project scaffold and environment verification

**Files:**
- Create: `pyproject.toml`
- Create: `src/mbfps/__init__.py` (sets the MPS fallback env var package-wide)
- Create: `src/mbfps/utils/__init__.py`
- Create: `src/mbfps/utils/device.py`
- Create: `src/mbfps/utils/seeding.py`
- Create: `tests/test_environment_setup.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `get_device(prefer: str = "mps") -> torch.device`; `seed_everything(seed: int) -> None`.

- [ ] **Step 1: Create the Python 3.12 venv and install dependencies**

```bash
brew install python@3.12
cd /Users/raphaelchen/Desktop/csgo-bot
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch>=2.4" "vizdoom>=1.3.0" "gymnasium>=1.0" "numpy>=1.26" "opencv-python>=4.10" "transformers>=4.44" "pytest>=8.0"
```

Expected: all installs succeed. If `brew install python@3.12` reports it is already installed, continue.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_environment_setup.py
import os
import sys

import numpy as np
import torch

from mbfps.utils.device import get_device
from mbfps.utils.seeding import seed_everything


def test_python_version_is_312():
    assert sys.version_info[:2] == (3, 12)


def test_vizdoom_imports_and_exposes_scenarios():
    import vizdoom as vzd

    assert os.path.isdir(vzd.scenarios_path)
    assert os.path.isfile(os.path.join(vzd.scenarios_path, "basic.cfg"))


def test_get_device_returns_mps_or_cpu():
    device = get_device()
    assert device.type in {"mps", "cpu"}


def test_get_device_honours_cpu_preference():
    assert get_device(prefer="cpu").type == "cpu"


def test_mps_fallback_env_var_is_set():
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_package_init_sets_mps_fallback():
    """The package root must set it, not just `device.py`.

    Python runs a parent package's `__init__` before any submodule body, so
    setting it in `mbfps/__init__.py` guarantees it precedes every `import
    torch` in the project -- including `mbfps.data.features`, which imports
    torch at the top of its own file before importing anything from mbfps.
    """
    import ast
    import inspect

    import mbfps

    tree = ast.parse(inspect.getsource(mbfps))
    assert any(
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "setdefault"
        and any(
            isinstance(a, ast.Constant) and a.value == "PYTORCH_ENABLE_MPS_FALLBACK"
            for a in node.args
        )
        for node in ast.walk(tree)
    ), "mbfps/__init__.py must set PYTORCH_ENABLE_MPS_FALLBACK"


def test_mps_fallback_is_set_before_torch_is_imported():
    """Checks statement order via the AST, not string search.

    A `source.index(...)` comparison is vacuous here: device.py's docstring
    mentions both `PYTORCH_ENABLE_MPS_FALLBACK` and `import torch`, so the
    string assertion holds regardless of where the real statements sit --
    verified to return True even on deliberately wrong-ordered source.
    """
    import ast
    import inspect

    import mbfps.utils.device as device_module

    tree = ast.parse(inspect.getsource(device_module))
    env_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "setdefault"
        and any(
            isinstance(a, ast.Constant) and a.value == "PYTORCH_ENABLE_MPS_FALLBACK"
            for a in node.args
        )
    )
    torch_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name.split(".")[0] == "torch" for alias in node.names)
    )
    assert env_line < torch_line


def test_seed_everything_makes_numpy_and_torch_reproducible():
    seed_everything(123)
    a_np, a_torch = np.random.rand(4), torch.rand(4)
    seed_everything(123)
    b_np, b_torch = np.random.rand(4), torch.rand(4)
    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_torch, b_torch)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_environment_setup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps'`

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mbfps"
version = "0.1.0"
description = "Model-Based First-Person Shooter Agent"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.4",
    "vizdoom>=1.3.0",
    "gymnasium>=1.0",
    "numpy>=1.26",
    "opencv-python>=4.10",
    "transformers>=4.44",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 5: Write the implementation**

```python
# src/mbfps/__init__.py
"""Model-Based First-Person Shooter Agent.

Setting PYTORCH_ENABLE_MPS_FALLBACK at the package root is what makes the
ordering guarantee project-wide. Python runs a parent package's `__init__`
before any submodule body, so this precedes every `import torch` in the
codebase -- including modules like `mbfps.data.features` that import torch at
the top of their own file, before importing anything from mbfps.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

```python
# src/mbfps/utils/__init__.py
```

```python
# src/mbfps/utils/device.py
"""Device selection.

MPS is the default target. `PYTORCH_ENABLE_MPS_FALLBACK` is set unconditionally
so that any operator without an MPS kernel silently falls back to CPU instead of
raising. Models in this project are small enough that `prefer="cpu"` is a viable
debugging path, not just a formality.

The env var is set BEFORE `import torch`. PyTorch reads it while initialising the
MPS backend, so setting it afterwards has no effect -- the variable would be
present in `os.environ` while the fallback stayed disabled, which looks correct
and is not.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  -- must follow the env var above


def get_device(prefer: str = "mps") -> torch.device:
    """Return the best available device, honouring `prefer`.

    Args:
        prefer: "mps", "cuda", or "cpu". Falls back to CPU when unavailable.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

```python
# src/mbfps/utils/seeding.py
"""Seed every RNG this project touches from a single integer."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed `random`, NumPy, and PyTorch (CPU + all accelerators)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 6: Install the package in editable mode and run the tests**

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/test_environment_setup.py -v
```

Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/mbfps tests/test_environment_setup.py
git commit -m "feat: project scaffold with device selection and seeding"
```

---

## Task 2: EnvProtocol and registry

**Files:**
- Create: `src/mbfps/envs/__init__.py`
- Create: `src/mbfps/envs/protocol.py`
- Create: `src/mbfps/envs/registry.py`
- Create: `tests/envs/test_protocol.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EnvProtocol` (runtime-checkable Protocol) with `observation_space`, `action_space`, `reset(*, seed) -> (np.ndarray, dict)`, `step(action: int) -> (np.ndarray, float, bool, bool, dict)`, `close() -> None`, `privileged_state -> dict | None`. Also `OBS_SHAPE: tuple[int, int, int] = (112, 112, 3)` and `make_env(name: str, **kwargs) -> EnvProtocol`.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/test_protocol.py
import numpy as np
import pytest
from gymnasium import spaces

from mbfps.envs.protocol import OBS_SHAPE, EnvProtocol
from mbfps.envs.registry import make_env


class _FakeEnv:
    """Minimal conforming implementation, used to test the protocol itself."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.closed = False

    def reset(self, *, seed=None):
        return np.zeros(OBS_SHAPE, dtype=np.uint8), {}

    def step(self, action):
        return np.zeros(OBS_SHAPE, dtype=np.uint8), 0.0, False, False, {}

    def close(self):
        self.closed = True

    @property
    def privileged_state(self):
        return {"health": 100.0}


def test_obs_shape_is_112x112x3():
    assert OBS_SHAPE == (112, 112, 3)


def test_conforming_class_satisfies_protocol():
    assert isinstance(_FakeEnv(), EnvProtocol)


def test_non_conforming_class_fails_protocol():
    class Incomplete:
        def reset(self, *, seed=None):
            return None, {}

    assert not isinstance(Incomplete(), EnvProtocol)


def test_make_env_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown environment 'nope'"):
        make_env("nope")


def test_make_env_lists_available_names_in_error():
    with pytest.raises(KeyError, match="vizdoom"):
        make_env("nope")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/envs/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.envs'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/envs/__init__.py
```

```python
# src/mbfps/envs/protocol.py
"""The modular engine hook.

`EnvProtocol` is the single boundary between the learning code and any game
engine. ViZDoom implements it at M0; OpenArena implements it at M7. If adding a
new engine requires editing anything outside its own module, this abstraction is
wrong and should be fixed rather than worked around.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from gymnasium import spaces

OBS_SHAPE: tuple[int, int, int] = (112, 112, 3)
"""Observation shape, HWC uint8 RGB.

112 = 8 * 14, so DINOv2's patch-14 backbone yields exactly an 8x8 patch grid.
Every arm of the study consumes byte-identical frames at this resolution.
"""


@runtime_checkable
class EnvProtocol(Protocol):
    """A seedable, closable, Gymnasium-shaped environment."""

    observation_space: spaces.Box
    action_space: spaces.Discrete

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode. Returns (observation, info)."""
        ...

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one step. Returns (obs, reward, terminated, truncated, info)."""
        ...

    def close(self) -> None:
        """Release engine resources."""
        ...

    @property
    def privileged_state(self) -> dict[str, float] | None:
        """Ground-truth engine state.

        EVALUATION PROBES ONLY -- never a training input. Leaking these values
        into an observation would silently invalidate the entire study, so
        `tests/test_privileged_isolation.py` asserts they never appear in a
        training tensor.
        """
        ...
```

```python
# src/mbfps/envs/registry.py
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
```

Also append these tests, which lock in the narrow catch by monkeypatching `_register_builtins` and calling `_try_register_builtins()` directly -- they must fail if the guard is ever widened back to a bare `except ImportError: pass`:

```python
def test_try_register_builtins_swallows_missing_vizdoom_env(monkeypatch):
    def _raise():
        raise ModuleNotFoundError(
            "No module named 'mbfps.envs.vizdoom_env'",
            name="mbfps.envs.vizdoom_env",
        )

    monkeypatch.setattr(registry, "_register_builtins", _raise)
    registry._try_register_builtins()  # must not raise


def test_try_register_builtins_reraises_other_missing_module(monkeypatch):
    def _raise():
        raise ModuleNotFoundError(
            "No module named 'some_native_lib'", name="some_native_lib"
        )

    monkeypatch.setattr(registry, "_register_builtins", _raise)
    with pytest.raises(ModuleNotFoundError, match="some_native_lib"):
        registry._try_register_builtins()


def test_try_register_builtins_reraises_plain_import_error(monkeypatch):
    def _raise():
        raise ImportError("cannot import name 'VizdoomEnv'")

    monkeypatch.setattr(registry, "_register_builtins", _raise)
    with pytest.raises(ImportError, match="VizdoomEnv"):
        registry._try_register_builtins()
```

This requires importing the module itself alongside its public names, at the top of the test file:

```python
from mbfps.envs import registry
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/test_protocol.py -v`
Expected: 7 passed, 1 xfailed.

Note: `test_make_env_lists_available_names_in_error` passes only once `vizdoom_env.py` exists, which is **Task 5** (not Task 3). Until then `_try_register_builtins` swallows the `ModuleNotFoundError` for `mbfps.envs.vizdoom_env` specifically -- any other import failure still propagates -- and the registry stays empty. Mark that one test now:

```python
@pytest.mark.xfail(reason="ViZDoomEnv lands in Task 5", strict=True)
def test_make_env_lists_available_names_in_error():
```

`strict=True` makes the suite fail if it starts passing, so the marker cannot be silently left behind after Task 5 removes the cause.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/envs tests/envs
git commit -m "feat: EnvProtocol interface and environment registry"
```

---

## Task 3: Discrete action space

**Files:**
- Create: `src/mbfps/envs/actions.py`
- Create: `tests/envs/test_actions.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_action_set(n_buttons: int) -> list[list[int]]` returning `n_buttons + 1` button vectors (a no-op followed by one one-hot per button); `n_actions(n_buttons: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/test_actions.py
import pytest

from mbfps.envs.actions import build_action_set, n_actions


def test_action_set_size_is_buttons_plus_noop():
    assert len(build_action_set(5)) == 6
    assert n_actions(5) == 6


def test_first_action_is_noop():
    assert build_action_set(3)[0] == [0, 0, 0]


def test_remaining_actions_are_one_hot():
    actions = build_action_set(3)
    assert actions[1] == [1, 0, 0]
    assert actions[2] == [0, 1, 0]
    assert actions[3] == [0, 0, 1]


def test_every_vector_has_length_n_buttons():
    assert all(len(a) == 4 for a in build_action_set(4))


def test_entries_are_plain_ints_for_vizdoom():
    # ViZDoom's make_action requires a list of ints, not numpy scalars.
    assert all(isinstance(v, int) for a in build_action_set(3) for v in a)


def test_zero_buttons_rejected():
    with pytest.raises(ValueError, match="at least one button"):
        build_action_set(0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/envs/test_actions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.envs.actions'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/envs/actions.py
"""Discrete action set.

The action set is derived from whatever buttons a scenario makes available, so
it is scenario-agnostic: a no-op plus one one-hot vector per button.

This deliberately forbids simultaneous presses (no move-and-shoot). That is a
real limitation, accepted for M0 under YAGNI -- the spec defers action-space
refinement to an open question, to be revisited once M5 results on
`deadly_corridor` show whether it binds.
"""


def build_action_set(n_buttons: int) -> list[list[int]]:
    """Return `n_buttons + 1` button vectors: a no-op, then one-hots.

    Raises:
        ValueError: if `n_buttons` is not positive.
    """
    if n_buttons < 1:
        raise ValueError(f"need at least one button, got {n_buttons}")
    noop = [0] * n_buttons
    one_hots = [[1 if i == j else 0 for j in range(n_buttons)] for i in range(n_buttons)]
    return [noop, *one_hots]


def n_actions(n_buttons: int) -> int:
    """Return the size of the discrete action space for `n_buttons` buttons."""
    return len(build_action_set(n_buttons))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/test_actions.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/envs/actions.py tests/envs/test_actions.py
git commit -m "feat: scenario-agnostic discrete action set"
```

---

## Task 4: Frame preprocessing

**Files:**
- Create: `src/mbfps/envs/wrappers.py`
- Create: `tests/envs/test_wrappers.py`

**Interfaces:**
- Consumes: `OBS_SHAPE` from `mbfps.envs.protocol`.
- Produces: `preprocess_frame(frame: np.ndarray) -> np.ndarray` returning `(112, 112, 3)` `uint8` HWC RGB. Accepts HWC or CHW input and normalises orientation.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/test_wrappers.py
import cv2
import numpy as np
import pytest

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.envs.wrappers import preprocess_frame


def test_hwc_input_resized_to_obs_shape():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    assert preprocess_frame(frame).shape == OBS_SHAPE


def test_chw_input_is_transposed_then_resized():
    frame = np.random.randint(0, 256, (3, 120, 160), dtype=np.uint8)
    assert preprocess_frame(frame).shape == OBS_SHAPE


def test_output_is_uint8():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    assert preprocess_frame(frame).dtype == np.uint8


def test_hwc_output_matches_area_interpolation():
    """Pins the interpolation mode: INTER_NEAREST/INTER_LINEAR give different pixels."""
    frame = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    expected = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)
    assert np.array_equal(preprocess_frame(frame), expected)


def test_chw_input_transposes_on_the_correct_axes():
    """A non-square CHW input catches a (2,1,0) transpose that (1,2,0) shape checks miss."""
    frame_chw = np.random.default_rng(1).integers(0, 256, (3, 100, 160), dtype=np.uint8)
    expected = cv2.resize(
        np.transpose(frame_chw, (1, 2, 0)), (112, 112), interpolation=cv2.INTER_AREA
    )
    assert np.array_equal(preprocess_frame(frame_chw), expected)


def test_vertical_split_stays_vertical():
    """Independent of the implementation: a left/right split must not become top/bottom."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, 80:, :] = 255
    out = preprocess_frame(frame)
    assert out[:, :50, :].mean() < 10, "left half should stay dark"
    assert out[:, 62:, :].mean() > 245, "right half should stay bright"
    assert 100 < out[:50, :, :].mean() < 155, "top half should be mixed, not uniform"


def test_output_is_c_contiguous():
    """Downstream code stacks these into batches; a non-contiguous view copies silently."""
    frame = np.random.default_rng(2).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    assert preprocess_frame(frame).flags["C_CONTIGUOUS"]
    already_sized = np.random.default_rng(3).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    assert preprocess_frame(already_sized[::-1]).flags["C_CONTIGUOUS"]


def test_already_correct_size_is_passed_through_unchanged():
    frame = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
    assert np.array_equal(preprocess_frame(frame), frame)


def test_deterministic_for_same_input():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    assert np.array_equal(preprocess_frame(frame), preprocess_frame(frame))


def test_grayscale_input_rejected():
    with pytest.raises(ValueError, match="expected 3 channels"):
        preprocess_frame(np.zeros((120, 160), dtype=np.uint8))
```

Note: `test_output_values_stay_in_uint8_range` was deliberately dropped — for a `uint8`
array, `min() >= 0 and max() <= 255` holds by construction and no source change can make
it fail. `test_output_is_uint8` already covers dtype. The four tests added above catch
resize-interpolation and transpose-axis regressions that shape/dtype checks cannot: a
square `OBS_SHAPE` means a wrong transpose axis or a wrong interpolation mode still
produces a `(112, 112, 3)` `uint8` array, so those checks must compare actual pixel
values (pinned against `cv2.resize` directly, plus an implementation-independent
vertical-split check) and contiguity flags.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/envs/test_wrappers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.envs.wrappers'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/envs/wrappers.py
"""Frame preprocessing.

ViZDoom's screen buffer layout depends on the configured `ScreenFormat`, and the
layout has differed across releases. Rather than hard-coding an assumption, this
module detects CHW and transposes it, so a ViZDoom upgrade cannot silently feed
transposed frames into training.
"""

import cv2
import numpy as np

from mbfps.envs.protocol import OBS_SHAPE

_TARGET_WH = (OBS_SHAPE[1], OBS_SHAPE[0])  # cv2.resize takes (width, height)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Normalise a raw engine frame to `(112, 112, 3)` uint8 HWC RGB.

    Args:
        frame: HWC or CHW uint8 array with 3 colour channels.

    Raises:
        ValueError: if `frame` is not 3-dimensional with 3 channels.
    """
    if frame.ndim != 3:
        raise ValueError(f"expected 3 channels, got array with shape {frame.shape}")
    if frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] != 3:
        raise ValueError(f"expected 3 channels, got shape {frame.shape}")
    if frame.shape[:2] == OBS_SHAPE[:2]:
        return np.ascontiguousarray(frame, dtype=np.uint8)
    resized = cv2.resize(frame, _TARGET_WH, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/test_wrappers.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/envs/wrappers.py tests/envs/test_wrappers.py
git commit -m "feat: frame preprocessing to 112x112 RGB uint8"
```

---

## Task 5: ViZDoomEnv

**Files:**
- Create: `src/mbfps/envs/vizdoom_env.py`
- Create: `tests/envs/test_vizdoom_env.py`
- Modify: `tests/envs/test_protocol.py` (remove the `xfail` marker added in Task 2)

**Interfaces:**
- Consumes: `OBS_SHAPE`, `EnvProtocol` from `mbfps.envs.protocol`; `build_action_set` from `mbfps.envs.actions`; `preprocess_frame` from `mbfps.envs.wrappers`; `register` from `mbfps.envs.registry`.
- Produces: `ViZDoomEnv(scenario: str = "my_way_home", frame_skip: int = 4, seed: int = 0, doom_skill: int | None = None)` implementing `EnvProtocol`. Exposes `PRIVILEGED_KEYS: tuple[str, ...] = ("health", "pos_x", "pos_y", "pos_z", "angle")`, a `button_names -> tuple[str, ...]` property, and a `scenario` attribute.

**Three verified facts this task depends on** (measured against ViZDoom 1.3.0 on this machine — do not "simplify" them away):

1. `get_state()` returns `None` once the episode is finished, so `privileged_state` is `None` on the terminal frame.
2. There is **no** `is_episode_timeout()` method. Truncation must be derived from `get_episode_time() >= get_episode_timeout()`.
3. `add_available_game_variable` **de-duplicates** against variables the scenario config already declares, so the resulting list length varies per scenario. Index game variables **by name**, never by a fixed slice.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/test_vizdoom_env.py
import numpy as np
import pytest
from gymnasium import spaces

from mbfps.envs.protocol import OBS_SHAPE, EnvProtocol
from mbfps.envs.vizdoom_env import PRIVILEGED_KEYS, ViZDoomEnv


@pytest.fixture
def env():
    e = ViZDoomEnv(scenario="my_way_home", frame_skip=4, seed=0)
    yield e
    e.close()


def test_satisfies_env_protocol(env):
    assert isinstance(env, EnvProtocol)


def test_observation_space_matches_obs_shape(env):
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == OBS_SHAPE
    assert env.observation_space.dtype == np.uint8


def test_action_space_is_discrete_and_nonempty(env):
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n >= 2


def test_scenario_exposes_movement_and_turning(env):
    """ScriptedPolicy's coverage behaviour depends on these existing."""
    names = env.button_names
    assert "MOVE_FORWARD" in names
    assert "TURN_LEFT" in names and "TURN_RIGHT" in names


def test_episodes_are_long_enough_for_the_training_window(env):
    """seq_len=64 needs 65 frames; a shorter episode is silently discarded."""
    env.reset(seed=0)
    steps = 0
    for _ in range(3000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps >= 65, f"episode was only {steps} transitions; seq_len=64 needs 65"


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)


def test_step_returns_five_tuple(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == OBS_SHAPE
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_eventually_ends(env):
    env.reset(seed=0)
    for _ in range(3000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            return
    pytest.fail("episode did not end within 3000 steps")


def test_timeout_sets_truncated_not_terminated(env):
    """The time-limit bootstrapping guard.

    A time-limit cutoff is not a true terminal state. Asserting only that the
    two flags are never both true does not catch the bug: swapping them still
    satisfies it. This asserts which flag a timeout actually sets.
    """
    env.reset(seed=0)
    for step in range(1, 3000):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            assert truncated, f"timeout at step {step} must set truncated"
            assert not terminated, "a time limit is not a true terminal state"
            return
    pytest.fail("episode did not end within 3000 steps")


def test_goal_reached_sets_terminated_not_truncated():
    """The other direction: a real terminal must not be recorded as a timeout.

    seed=5 with this exact RNG reaches the goal at step 96 (reward ~1.0),
    well before the 525-step timeout.
    """
    env = ViZDoomEnv(scenario="my_way_home", frame_skip=4, seed=5)
    try:
        env.reset(seed=5)
        rng = np.random.default_rng(5)
        for step in range(1, 600):
            action = int(rng.integers(0, env.action_space.n))
            _, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                assert terminated, f"goal reached at step {step} must set terminated"
                assert not truncated, "a goal is not a time-limit cutoff"
                assert reward > 0.5, f"expected goal reward, got {reward}"
                return
        pytest.fail("episode did not end within 600 steps")
    finally:
        env.close()


def test_observation_after_end_is_still_valid(env):
    env.reset(seed=0)
    for _ in range(3000):
        obs, _, terminated, truncated, _ = env.step(0)
        assert obs.shape == OBS_SHAPE, "terminal observation must stay well-formed"
        if terminated or truncated:
            break


def test_privileged_state_has_expected_keys(env):
    env.reset(seed=0)
    state = env.privileged_state
    assert set(state) == set(PRIVILEGED_KEYS)
    assert all(isinstance(v, float) for v in state.values())


def test_privileged_state_is_none_after_the_episode_ends(env):
    """ViZDoom's get_state() returns None once finished; callers must handle it."""
    env.reset(seed=0)
    for _ in range(3000):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert env.privileged_state is None


def test_privileged_keys_are_indexed_by_name_not_position(env):
    """health is the scenario's own variable; the position vars are ours."""
    env.reset(seed=0)
    state = env.privileged_state
    assert state["health"] > 0.0, "health should be positive at episode start"


def test_close_is_idempotent(env):
    env.close()
    env.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/envs/test_vizdoom_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.envs.vizdoom_env'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/envs/vizdoom_env.py
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

# NOTE: `register` is imported at the BOTTOM of this file, not here. `registry`
# imports ViZDoomEnv from this module, so importing it at the top creates a
# cycle: importing `mbfps.envs.vizdoom_env` first raises "cannot import name
# 'ViZDoomEnv' from partially initialized module". The test suite hides this,
# because pytest collects alphabetically and always imports registry first.

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


from mbfps.envs.registry import register  # noqa: E402 -- see the note above

register("vizdoom", ViZDoomEnv)
```

**Any future engine module must use this bottom-import pattern.** Copying the top-import form into an OpenArena module at M7 reproduces the same cycle.

- [ ] **Step 4: Remove the xfail marker from Task 2's test**

In `tests/envs/test_protocol.py`, delete the `@pytest.mark.xfail(...)` decorator above `test_make_env_lists_available_names_in_error`. It is `strict=True`, so leaving it in place now fails the suite.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/ -v`
Expected: 39 passed — `test_protocol.py` 8 (the strict xfail is removed in this task, so all 8 pass), `test_actions.py` 6, `test_wrappers.py` 10, `test_vizdoom_env.py` 15 (the mutually-exclusive test is split into two direction-pinning tests). `test_determinism.py` does not exist until Task 6.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/envs/vizdoom_env.py tests/envs/
git commit -m "feat: headless seedable ViZDoom environment with timeout truncation"
```

## Task 6: Determinism, action replay, and throughput

**Files:**
- Create: `tests/envs/test_determinism.py`
- Create: `scripts/benchmark_env.py`

**Interfaces:**
- Consumes: `ViZDoomEnv`, `make_env`.
- Produces: `scripts/benchmark_env.py` printing `steps_per_second=<float>`. No new library API.

This task is the M0 exit gate: **same seed produces a bit-identical episode, and a recorded episode replays from its saved action sequence alone.**

Note: `set_seed` after `init` is verified to reseed correctly in ViZDoom 1.3.0 (same seed reproduces, different seed diverges). No engine-restart fallback is needed, and none should be added -- an earlier draft of this plan proposed one guarded by `seed != self._seed`, which would have made the reseed a no-op in exactly the case the test exercises.

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/test_determinism.py
import json

import numpy as np
import pytest

from mbfps.envs.vizdoom_env import ViZDoomEnv

SCENARIO = "my_way_home"


def _rollout(seed, actions):
    """Replay a fixed action list and return (frames, rewards)."""
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
        frames, rewards = [obs.copy()], []
        for a in actions:
            obs, reward, terminated, truncated, _ = env.step(a)
            frames.append(obs.copy())
            rewards.append(reward)
            if terminated or truncated:
                break
        return frames, rewards
    finally:
        env.close()


@pytest.fixture
def actions():
    # Bound the random action stream by the scenario's actual action count
    # rather than a hardcoded literal, so this fixture cannot silently drift
    # out of sync with the scenario's button set.
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=0)
    try:
        n = env.action_space.n
    finally:
        env.close()
    rng = np.random.default_rng(0)
    return [int(a) for a in rng.integers(0, n, size=40)]


def test_same_seed_produces_identical_frames(actions):
    a_frames, _ = _rollout(seed=7, actions=actions)
    b_frames, _ = _rollout(seed=7, actions=actions)
    assert len(a_frames) == len(b_frames)
    for i, (a, b) in enumerate(zip(a_frames, b_frames)):
        assert np.array_equal(a, b), f"frame {i} differs between identical seeds"


def test_same_seed_produces_identical_rewards(actions):
    _, a_rewards = _rollout(seed=7, actions=actions)
    _, b_rewards = _rollout(seed=7, actions=actions)
    assert a_rewards == b_rewards


def test_different_seeds_diverge(actions):
    """Fresh instances constructed with different seeds produce different episodes.

    `_rollout()` constructs a brand-new `ViZDoomEnv(seed=seed)` per call, so
    this proves construction-time seeding (the `set_seed` call in `__init__`)
    differentiates the two engines. It does NOT exercise `reset()`'s reseed
    path -- a `reset()` that silently no-ops would not be caught here. See
    `test_reset_reseeds_rather_than_letting_the_engine_rng_drift` for the test
    that actually guards `reset()`'s reseed.
    """
    a_frames, _ = _rollout(seed=7, actions=actions)
    b_frames, _ = _rollout(seed=8, actions=actions)
    pairs = list(zip(a_frames, b_frames))
    assert any(not np.array_equal(a, b) for a, b in pairs), (
        "two different seeds produced identical episodes -- set_seed is not taking effect"
    )


def test_reset_reseeds_rather_than_letting_the_engine_rng_drift():
    """A live engine reseeded to N must match a fresh engine constructed at N.

    This is the operation M1's collector performs: one long-lived engine,
    reseeded once per episode. Comparing two different-seed resets on the same
    instance proves nothing -- ViZDoom's RNG advances on every new_episode(), so
    consecutive episodes differ even when reset() ignores the seed entirely.
    The reference must be an engine whose history cannot matter.
    """
    live = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=0)
    fresh = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=99)
    try:
        live.reset(seed=31)
        for _ in range(10):
            live.step(1)
        live.reset(seed=17)
        for _ in range(10):
            live.step(2)
        reseeded, _ = live.reset(seed=99)
        reference, _ = fresh.reset(seed=99)
        assert np.array_equal(reseeded, reference), (
            "a reseeded live engine diverged from a fresh engine at the same "
            "seed -- reset() is advancing the engine RNG instead of reseeding it"
        )
    finally:
        live.close()
        fresh.close()


def test_recorded_episode_replays_from_saved_actions_alone(tmp_path):
    """The M0 exit criterion.

    The recording policy is observation-dependent -- it reads `obs` to choose
    each action -- so the action stream is a function of the rollout's own
    pixel history, not a fixed list chosen independently of what the engine
    renders. Replaying the saved actions into a fresh engine and recovering
    identical pixels therefore proves the *pixels* were deterministic, not
    merely that a fixed action list replays the same regardless of what it's
    fed.
    """
    seed = 11
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
        original_frames, chosen = [obs.copy()], []
        for _ in range(40):
            action = int(obs.sum() % env.action_space.n)
            obs, _, terminated, truncated, _ = env.step(action)
            chosen.append(action)
            original_frames.append(obs.copy())
            if terminated or truncated:
                break
    finally:
        env.close()

    record = tmp_path / "episode_actions.json"
    record.write_text(json.dumps({"seed": seed, "actions": chosen}))

    loaded = json.loads(record.read_text())
    replay_frames, _ = _rollout(seed=loaded["seed"], actions=loaded["actions"])

    assert len(replay_frames) == len(original_frames)
    for i, (a, b) in enumerate(zip(original_frames, replay_frames)):
        assert np.array_equal(a, b), f"replayed frame {i} differs from the recording"


def test_reset_reseeds_within_one_instance():
    env = ViZDoomEnv(scenario=SCENARIO, frame_skip=4, seed=0)
    try:
        first, _ = env.reset(seed=21)
        env.step(1)
        second, _ = env.reset(seed=21)
        assert np.array_equal(first, second)
    finally:
        env.close()
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/test_determinism.py -v`
Expected: 6 passed.

Do not proceed to Task 7 with a failing determinism test -- every downstream arm comparison depends on it. In particular, `test_different_seeds_diverge` failing means `set_seed` is not taking effect; investigate before continuing rather than working around it.

- [ ] **Step 3: Write the throughput benchmark**

```python
# scripts/benchmark_env.py
"""Measure headless ViZDoom throughput.

Records the M0 throughput number. Run before and after any change to the
environment or preprocessing path.
"""

import argparse
import time

from mbfps.envs.registry import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="my_way_home")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    env = make_env("vizdoom", scenario=args.scenario, frame_skip=args.frame_skip)
    try:
        env.reset(seed=0)
        start = time.perf_counter()
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset(seed=0)
        elapsed = time.perf_counter() - start
    finally:
        env.close()

    print(f"scenario={args.scenario} frame_skip={args.frame_skip}")
    print(f"steps={args.steps} elapsed_s={elapsed:.2f}")
    print(f"steps_per_second={args.steps / elapsed:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the benchmark and record the number**

Run: `.venv/bin/python scripts/benchmark_env.py --steps 2000`
Expected: prints `steps_per_second=<N>`. Record N in the commit message -- it is the M0 baseline and sizes every collection run in M1.

- [ ] **Step 5: Commit**

```bash
git add tests/envs/test_determinism.py scripts/benchmark_env.py
git commit -m "test: determinism and action-replay gate; add env throughput benchmark"
```

## Task 7: Collection policies

**Files:**
- Create: `src/mbfps/data/__init__.py`
- Create: `src/mbfps/data/policies.py`
- Create: `tests/data/test_policies.py`

**Interfaces:**
- Consumes: nothing (`ViZDoomEnv.button_names` already exists from Task 5).
- Produces: `Policy` Protocol with `name: str`, `act(obs: np.ndarray) -> int`, and `reset(seed: int | None = None) -> None`; `RandomPolicy(n_actions: int, seed: int = 0)`; `ScriptedPolicy(button_names: Sequence[str], seed: int = 0)`.

**Two requirements that an earlier draft got wrong** (both would have silently defeated M1):

1. **`reset(seed)` must reseed, not rewind.** The collector calls `reset()` once per episode. If `reset()` restores a fixed seed, every episode a policy produces replays one identical action sequence, and the dataset contains one trajectory per policy rather than hundreds.
2. **`ScriptedPolicy` must adapt to the buttons that exist.** Button sets differ per scenario (verified: `deadly_corridor` has all seven; `basic` has only `MOVE_LEFT, MOVE_RIGHT, ATTACK`; `defend_the_center` has only `TURN_LEFT, TURN_RIGHT, ATTACK`). Hard-coding `MOVE_FORWARD` makes the policy degenerate to a constant `ATTACK` wherever that button is absent.

Rationale (spec §5.2): a single random policy has poor state coverage, and the world model hallucinates wherever the collector never went. The M4 actor-critic then optimises straight into those hallucinations, producing high imagined return and near-zero real return. `ScriptedPolicy` exists to reach states `RandomPolicy` does not — it can only do that if it actually moves.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_policies.py
import numpy as np
import pytest

from mbfps.data.policies import Policy, RandomPolicy, ScriptedPolicy
from mbfps.envs.protocol import OBS_SHAPE

# Real button sets, verified against ViZDoom 1.3.0.
CORRIDOR = ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD",
            "TURN_LEFT", "TURN_RIGHT")
BASIC = ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
CENTER = ("TURN_LEFT", "TURN_RIGHT", "ATTACK")
HOME = ("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_RIGHT")
ALL_SETS = [CORRIDOR, BASIC, CENTER, HOME]  # HOME has no ATTACK button


@pytest.fixture
def obs():
    return np.zeros(OBS_SHAPE, dtype=np.uint8)


def _run(policy, obs, n=1000):
    return [policy.act(obs) for _ in range(n)]


def test_random_policy_satisfies_protocol():
    assert isinstance(RandomPolicy(n_actions=5, seed=0), Policy)


def test_scripted_policy_satisfies_protocol():
    assert isinstance(ScriptedPolicy(CORRIDOR, seed=0), Policy)


def test_policies_expose_distinct_names():
    assert RandomPolicy(5, seed=0).name == "random"
    assert ScriptedPolicy(CORRIDOR, seed=0).name == "scripted"


def test_random_policy_stays_in_range(obs):
    assert all(0 <= a < 5 for a in _run(RandomPolicy(5, seed=0), obs, 200))


def test_random_policy_covers_every_action(obs):
    assert set(_run(RandomPolicy(5, seed=0), obs, 500)) == {0, 1, 2, 3, 4}


def test_same_seed_gives_the_same_sequence(obs):
    p, q = RandomPolicy(5, seed=3), RandomPolicy(5, seed=3)
    assert _run(p, obs, 50) == _run(q, obs, 50)


def test_reset_with_a_new_seed_changes_the_sequence(obs):
    """The bug this guards: reset() rewinding to a fixed seed would make every
    collected episode replay one identical action sequence."""
    policy = RandomPolicy(5, seed=0)
    policy.reset(seed=1)
    first = _run(policy, obs, 200)
    policy.reset(seed=2)
    assert _run(policy, obs, 200) != first


def test_reset_with_the_same_seed_reproduces(obs):
    policy = RandomPolicy(5, seed=0)
    policy.reset(seed=7)
    first = _run(policy, obs, 100)
    policy.reset(seed=7)
    assert _run(policy, obs, 100) == first


def test_scripted_reset_with_a_new_seed_changes_the_sequence(obs):
    policy = ScriptedPolicy(CORRIDOR, seed=0)
    policy.reset(seed=1)
    first = _run(policy, obs, 200)
    policy.reset(seed=2)
    assert _run(policy, obs, 200) != first


@pytest.mark.parametrize("buttons", ALL_SETS)
def test_scripted_policy_returns_valid_indices(obs, buttons):
    assert all(0 <= a <= len(buttons) for a in _run(ScriptedPolicy(buttons, seed=0), obs))


@pytest.mark.parametrize("buttons", ALL_SETS)
def test_scripted_policy_never_degenerates_to_one_action(obs, buttons):
    """The bug this guards: hard-coded button names made the policy emit a
    constant ATTACK on any scenario lacking MOVE_FORWARD."""
    actions = _run(ScriptedPolicy(buttons, seed=0), obs)
    assert len(set(actions)) >= 2, f"degenerate policy on {buttons}"


@pytest.mark.parametrize("buttons", [CORRIDOR, BASIC, CENTER])
def test_scripted_policy_is_not_attack_dominated(obs, buttons):
    actions = _run(ScriptedPolicy(buttons, seed=0), obs)
    attack_index = buttons.index("ATTACK") + 1
    assert actions.count(attack_index) / len(actions) < 0.5


def test_scripted_policy_commits_to_forward_when_available(obs):
    actions = _run(ScriptedPolicy(CORRIDOR, seed=0), obs)
    forward = CORRIDOR.index("MOVE_FORWARD") + 1
    assert actions.count(forward) / len(actions) > 0.4


def test_scripted_policy_sweeps_both_directions(obs):
    actions = set(_run(ScriptedPolicy(CORRIDOR, seed=0), obs))
    assert CORRIDOR.index("TURN_LEFT") + 1 in actions
    assert CORRIDOR.index("TURN_RIGHT") + 1 in actions


def test_scripted_policy_strafes_both_ways_without_turn_buttons(obs):
    """On `basic` there is no turning, so the sweep must fall back to strafing."""
    actions = set(_run(ScriptedPolicy(BASIC, seed=0), obs))
    assert BASIC.index("MOVE_LEFT") + 1 in actions
    assert BASIC.index("MOVE_RIGHT") + 1 in actions


@pytest.mark.parametrize("buttons", ALL_SETS)
def test_scripted_policy_commits_more_than_random(obs, buttons):
    """Sustained commitment is what buys coverage over random oscillation.

    Compared against random at the same seed rather than against a fixed
    threshold. A longest-run check does not discriminate: uniform random clears
    a run of 3 about 94% of the time, and on 3-button sets random reaches runs
    of 8 while scripted's minimum is 7. The repeat rate separates cleanly --
    measured worst-case paired ratio across four button sets and 60 seeds is
    1.86, so 1.5 has headroom.
    """

    def repeat_rate(policy):
        actions = np.array([policy.act(obs) for _ in range(400)])
        return float((actions[1:] == actions[:-1]).mean())

    scripted = repeat_rate(ScriptedPolicy(buttons, seed=0))
    uniform = repeat_rate(RandomPolicy(len(buttons) + 1, seed=0))
    assert scripted > uniform * 1.5, (
        f"scripted repeat rate {scripted:.3f} vs random {uniform:.3f} -- "
        "the policy is not committing to sustained actions"
    )


def test_scripted_policy_with_only_attack_still_varies(obs):
    """Degenerate button set: must fall back to random rather than a constant."""
    actions = _run(ScriptedPolicy(("ATTACK",), seed=0), obs, 200)
    assert set(actions) == {0, 1}


def test_scripted_policy_works_without_an_attack_button(obs):
    """my_way_home, the default scenario, has no ATTACK. The policy must still
    advance and sweep rather than falling through to uniform random."""
    actions = _run(ScriptedPolicy(HOME, seed=0), obs)
    forward = HOME.index("MOVE_FORWARD") + 1
    assert actions.count(forward) / len(actions) > 0.4
    assert HOME.index("TURN_LEFT") + 1 in set(actions)
    assert HOME.index("TURN_RIGHT") + 1 in set(actions)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_policies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/data/policies.py
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

    _ADVANCE_PROB = 0.6
    _SWEEP_PROB = 0.75
    _SWEEP_LEN = 8

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_policies.py -v`
Expected: 29 passed. Three tests are parametrised over all four button sets and one over the three sets that have an `ATTACK` button, so the collected count exceeds the number of `def test_` lines.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data tests/data
git commit -m "feat: random and button-set-adaptive scripted collection policies"
```

## Task 8: Episode storage

**Files:**
- Create: `src/mbfps/data/episode.py`
- Create: `tests/data/test_episode.py`

**Interfaces:**
- Consumes: nothing. (`privileged_keys` travels with each episode as data, so this
  module has no dependency on any particular engine's key set.)
- Produces: `Episode` dataclass with fields `obs (T+1, 112, 112, 3) uint8`, `actions (T,) int32`, `rewards (T,) float32`, `terminated (T,) bool`, `truncated (T,) bool`, `privileged (T+1, K) float32`, `privileged_keys tuple[str, ...]`, `policy_name str`, `seed int`, `scenario str`; property `length -> int` (= T). Functions `save_episode(ep: Episode, path: Path) -> None` and `load_episode(path: Path) -> Episode`.

`terminated` and `truncated` are stored separately. A ViZDoom episode that hits its time limit is not a true terminal state, and collapsing the two would teach M3's continue predictor that the world ends when the clock runs out — the classic time-limit bootstrapping bug.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_episode.py
import numpy as np
import pytest

from mbfps.data.episode import Episode, load_episode, save_episode
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int = 5) -> Episode:
    rng = np.random.default_rng(0)
    return Episode(
        obs=rng.integers(0, 256, (t + 1, *OBS_SHAPE), dtype=np.uint8),
        actions=rng.integers(0, 4, t).astype(np.int32),
        rewards=rng.standard_normal(t).astype(np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=rng.standard_normal((t + 1, len(KEYS))).astype(np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=42,
        scenario="my_way_home",
    )


def test_length_is_number_of_transitions():
    assert make_episode(t=5).length == 5


def test_round_trip_preserves_arrays(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert np.array_equal(loaded.obs, ep.obs)
    assert np.array_equal(loaded.actions, ep.actions)
    assert np.array_equal(loaded.rewards, ep.rewards)
    assert np.array_equal(loaded.terminated, ep.terminated)
    assert np.array_equal(loaded.truncated, ep.truncated)
    assert np.array_equal(loaded.privileged, ep.privileged)


def test_round_trip_preserves_metadata(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert loaded.policy_name == "random"
    assert loaded.seed == 42
    assert loaded.scenario == "my_way_home"
    assert loaded.privileged_keys == KEYS


def test_round_trip_preserves_dtypes(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert loaded.obs.dtype == np.uint8
    assert loaded.actions.dtype == np.int32
    assert loaded.rewards.dtype == np.float32
    assert loaded.privileged.dtype == np.float32
    assert loaded.terminated.dtype == bool
    assert loaded.truncated.dtype == bool


def test_truncated_and_terminated_survive_a_round_trip_independently(tmp_path):
    """A time-limit cutoff must stay distinguishable from a true terminal."""
    ep = make_episode(t=3)
    ep.terminated[:] = False
    ep.truncated[:] = False
    ep.truncated[-1] = True

    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)

    assert not loaded.terminated.any(), "no step should be marked terminal"
    assert loaded.truncated[-1], "the time-limit flag must survive the round trip"
    assert not loaded.truncated[:-1].any()


def test_mismatched_lengths_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="obs must have one more"):
        Episode(
            obs=rng.integers(0, 256, (5, *OBS_SHAPE), dtype=np.uint8),
            actions=rng.integers(0, 4, 5).astype(np.int32),
            rewards=rng.standard_normal(5).astype(np.float32),
            terminated=np.zeros(5, dtype=bool),
            truncated=np.zeros(5, dtype=bool),
            privileged=rng.standard_normal((6, len(KEYS))).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )


def test_mismatched_privileged_width_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="privileged must have shape"):
        Episode(
            obs=rng.integers(0, 256, (6, *OBS_SHAPE), dtype=np.uint8),
            actions=rng.integers(0, 4, 5).astype(np.int32),
            rewards=rng.standard_normal(5).astype(np.float32),
            terminated=np.zeros(5, dtype=bool),
            truncated=np.zeros(5, dtype=bool),
            privileged=rng.standard_normal((6, len(KEYS) - 2)).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )


def test_wrong_obs_dtype_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="obs must have dtype uint8"):
        Episode(
            obs=rng.standard_normal((6, *OBS_SHAPE)).astype(np.float32),
            actions=rng.integers(0, 4, 5).astype(np.int32),
            rewards=rng.standard_normal(5).astype(np.float32),
            terminated=np.zeros(5, dtype=bool),
            truncated=np.zeros(5, dtype=bool),
            privileged=rng.standard_normal((6, len(KEYS))).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_episode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data.episode'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/data/episode.py
"""Episode container and on-disk format.

One compressed `.npz` per episode. `obs` holds T+1 frames for T transitions, so
a window of length T yields T aligned (s_t, a_t, r_t, s_{t+1}) tuples.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mbfps.envs.protocol import OBS_SHAPE


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
        if self.obs.dtype != np.uint8:
            raise ValueError(f"obs must have dtype uint8, got {self.obs.dtype}")
        if self.obs.shape[1:] != OBS_SHAPE:
            raise ValueError(
                f"obs must have per-frame shape {OBS_SHAPE}, got {self.obs.shape[1:]}"
            )
        if self.obs.shape[0] != t + 1:
            raise ValueError(
                f"obs must have one more entry than actions; "
                f"got {self.obs.shape[0]} obs and {t} actions"
            )
        for name, arr, expected in (
            ("rewards", self.rewards, t),
            ("terminated", self.terminated, t),
            ("truncated", self.truncated, t),
        ):
            if arr.shape[0] != expected:
                raise ValueError(
                    f"{name} must have length {expected}, got {arr.shape[0]}"
                )
        expected_privileged_shape = (t + 1, len(self.privileged_keys))
        if self.privileged.shape != expected_privileged_shape:
            raise ValueError(
                f"privileged must have shape {expected_privileged_shape}, "
                f"got {self.privileged.shape}"
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
        privileged_keys=np.asarray(ep.privileged_keys, dtype=np.str_),
        policy_name=ep.policy_name,
        seed=ep.seed,
        scenario=ep.scenario,
    )


def load_episode(path: Path) -> Episode:
    """Read an episode written by `save_episode`."""
    # Pickle is not used: privileged_keys is stored as a fixed-width unicode
    # array (numpy infers the dtype), which round-trips exactly through npz
    # without allow_pickle, including the empty-tuple case. Leaving
    # allow_pickle at its default of False means loading an episode can never
    # execute arbitrary code embedded in the file.
    with np.load(path) as data:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_episode.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data/episode.py tests/data/test_episode.py
git commit -m "feat: episode container with verified npz round-trip"
```

---

## Task 9: Collector with crash recovery

**Files:**
- Create: `src/mbfps/data/collector.py`
- Create: `tests/data/test_collector.py`

**Interfaces:**
- Consumes: `EnvProtocol`, `Policy`, `Episode`.
- Produces: `Collector(env_factory: Callable[[], EnvProtocol], policy: Policy, max_steps: int = 1000)` with `collect_episode(seed: int) -> Episode | None` (None on engine crash, after restarting the env), `close() -> None`, and attribute `crash_count: int`.

**The failure this task must avoid.** `ViZDoomEnv.privileged_state` returns `None` once the episode is finished (verified against ViZDoom 1.3.0 — `get_state()` is None at that point). A collector that queries privileged state *after* the terminal step, or that derives the key set *after* the loop, produces a ragged array; `np.stack` then raises, the blanket `except Exception` reclassifies it as an engine crash, and **every episode is discarded**. So:

- Capture the key set **once, at reset**, before any stepping.
- On the terminal frame, carry the last valid row forward rather than emitting a zero-width row.
- Raise a **distinct `DataIntegrityError`** for malformed data and let it escape the crash handler. An `AssertionError` would be caught by `except Exception` and counted as an engine fault — landing in the very failure mode the guard exists to prevent.
- Do **not** reshape the stacked array. `np.stack` of zero-length rows already gives `(T+1, 0)`; `reshape(-1, 0)` raises `cannot reshape array of size 0` (measured on numpy 2.5.2), which the crash handler would swallow.
- `Episode.__post_init__` raises **`ValueError`** for obs/actions/privileged shape and dtype violations. That is our own bug reached via a different axis than the ragged-privileged-array case above, and the blanket `except Exception` catches it identically — a good episode gets discarded and miscounted as an engine fault. Wrap the `Episode(...)` construction in its own `try/except ValueError` and re-raise as `DataIntegrityError`. Do **not** force-cast `obs` to `uint8` before that call: a real engine already emits `uint8` frames, so the cast is a no-op for good data and would only mask a malformed one, defeating the validation it's supposed to trigger.
- Let **`MemoryError`** escape the crash handler too — `np.stack` on a long episode can raise it, and it is exactly as much our own resource problem as a `DataIntegrityError` is our own data problem, not an engine fault.
- `terminated` and `truncated` are a load-bearing distinction, not an interchangeable pair: a time-limit cutoff is not a true terminal state, and swapping the two arguments in the `Episode(...)` call, or narrowing `if terminated or truncated: break` to `if terminated:`, must be caught by a stub environment capable of actually setting `truncated=True` — a stub that always returns `truncated=False` cannot detect either bug.
- The episode seed passed to `collect_episode` must reach `self._env.reset(seed=seed)`. A stub that ignores its `seed` argument cannot detect a dropped env seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_collector.py
import numpy as np
import pytest
from gymnasium import spaces

from mbfps.data.collector import Collector, DataIntegrityError
from mbfps.data.policies import RandomPolicy
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


class _StubEnv:
    """Stand-in that mirrors ViZDoom's real terminal behaviour.

    Critically, `privileged_state` returns None once the episode is finished,
    exactly as ViZDoom's get_state() does. A more forgiving stub would let the
    ragged-array bug through.
    """

    instances = 0

    def __init__(self, episode_len=6, crash_at=None, truncate: bool = False):
        type(self).instances += 1
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.button_names = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
        self.scenario = "stub"
        self._episode_len = episode_len
        self._crash_at = crash_at
        self._truncate = truncate
        self._t = 0
        self._done = False
        self.seeds_seen: list = []

    def reset(self, *, seed=None):
        self._t = 0
        self._done = False
        self.seeds_seen.append(seed)
        return np.full(OBS_SHAPE, 1, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        if self._crash_at is not None and self._t == self._crash_at:
            raise RuntimeError("simulated engine crash")
        self._done = self._t >= self._episode_len
        obs = np.full(OBS_SHAPE, self._t % 256, dtype=np.uint8)
        if self._done and self._truncate:
            return obs, 1.0, False, True, {}
        return obs, 1.0, self._done, False, {}

    def close(self):
        pass

    @property
    def privileged_state(self):
        if self._done:
            return None
        return {k: float(self._t) for k in KEYS}


def _collector(episode_len=6, crash_at=None, max_steps=1000):
    return Collector(
        lambda: _StubEnv(episode_len, crash_at), RandomPolicy(4, seed=0), max_steps
    )


def test_collect_episode_returns_episode():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep is not None and ep.length == 6
    c.close()


def test_terminal_episode_is_not_treated_as_a_crash():
    """The bug this guards: a None privileged_state on the terminal frame
    produced a ragged array, so every real episode was discarded."""
    c = _collector(episode_len=6)
    assert c.collect_episode(seed=0) is not None
    assert c.crash_count == 0
    c.close()


def test_privileged_keys_survive_termination():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep.privileged_keys == KEYS, "keys must be captured at reset, not after"
    c.close()


def test_privileged_has_one_row_per_frame():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep.privileged.shape == (ep.length + 1, len(KEYS))
    c.close()


def test_terminal_privileged_row_carries_the_last_value_forward():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert np.array_equal(ep.privileged[-1], ep.privileged[-2])
    c.close()


def test_obs_has_one_more_frame_than_actions():
    c = _collector(episode_len=6)
    ep = c.collect_episode(seed=0)
    assert ep.obs.shape[0] == ep.actions.shape[0] + 1
    c.close()


def test_episode_records_policy_name_and_seed():
    c = _collector(episode_len=4)
    ep = c.collect_episode(seed=77)
    assert ep.policy_name == "random" and ep.seed == 77
    c.close()


def test_each_episode_gets_a_different_action_sequence():
    """The bug this guards: resetting the policy to a fixed seed made every
    collected episode replay one identical action sequence."""
    c = _collector(episode_len=40, max_steps=40)
    a = c.collect_episode(seed=1)
    b = c.collect_episode(seed=2)
    assert not np.array_equal(a.actions, b.actions)
    c.close()


def test_same_seed_reproduces_the_action_sequence():
    a = _collector(episode_len=40, max_steps=40).collect_episode(seed=5)
    b = _collector(episode_len=40, max_steps=40).collect_episode(seed=5)
    assert np.array_equal(a.actions, b.actions)


def test_same_seed_reproduces_actions_on_a_reused_collector():
    """Guards the per-episode policy reseed.

    A fresh collector per episode cannot detect a missing reset(): two new
    RandomPolicy(seed=0) objects start from the same RNG state regardless.
    Reusing ONE collector is what exposes it -- without the reseed the second
    episode simply continues the first episode's RNG stream.

    This is the failure a pre-flight audit rated critical: every collected
    episode replaying one identical action sequence.
    """
    c = _collector(episode_len=40, max_steps=40)
    try:
        first = c.collect_episode(seed=5)
        second = c.collect_episode(seed=5)
        assert first is not None and second is not None
        assert np.array_equal(first.actions, second.actions), (
            "same seed on a reused collector produced different actions -- "
            "the per-episode policy reseed is missing"
        )
    finally:
        c.close()


def test_different_seeds_differ_on_a_reused_collector():
    """The complement: reseeding must actually track the seed, not ignore it."""
    c = _collector(episode_len=40, max_steps=40)
    try:
        a = c.collect_episode(seed=5)
        b = c.collect_episode(seed=6)
        assert not np.array_equal(a.actions, b.actions), (
            "different seeds on a reused collector produced identical actions"
        )
    finally:
        c.close()


def test_max_steps_truncates():
    c = _collector(episode_len=999, max_steps=5)
    assert c.collect_episode(seed=0).length == 5
    c.close()


def test_crash_returns_none_and_is_counted():
    c = _collector(episode_len=20, crash_at=3)
    assert c.collect_episode(seed=0) is None
    assert c.crash_count == 1
    c.close()


def test_crash_rebuilds_the_environment():
    _StubEnv.instances = 0
    c = _collector(episode_len=20, crash_at=3)
    c.collect_episode(seed=0)
    assert _StubEnv.instances >= 2, "collector must rebuild the env after a crash"
    c.close()


def test_collector_recovers_and_keeps_collecting():
    envs = iter([_StubEnv(20, crash_at=3), _StubEnv(6), _StubEnv(6)])
    c = Collector(lambda: next(envs), RandomPolicy(4, seed=0))
    assert c.collect_episode(seed=0) is None
    assert c.collect_episode(seed=1) is not None
    c.close()


def test_env_with_no_privileged_state_still_collects():
    """A zero-width privileged array must not be mistaken for a crash.

    `np.stack` of zero-length rows gives shape (T+1, 0), which is correct.
    Reshaping it would raise `cannot reshape array of size 0` (measured on
    numpy 2.5.2), and the crash handler would swallow that as a phantom
    engine fault -- discarding every episode from such an env.
    """

    class _Bare(_StubEnv):
        @property
        def privileged_state(self):
            return None

    c = Collector(lambda: _Bare(6), RandomPolicy(4, seed=0))
    ep = c.collect_episode(seed=0)
    assert ep is not None
    assert c.crash_count == 0
    assert ep.privileged_keys == ()
    assert ep.privileged.shape == (ep.length + 1, 0)
    c.close()


def test_malformed_data_is_not_reported_as_an_engine_crash():
    """A key vanishing mid-episode is our bug, not the engine's.

    Without a distinct exception type this raises inside `_collect`, is caught
    by the blanket crash handler, and is silently counted as an engine fault --
    the exact failure mode this collector was rewritten to prevent.
    """

    class _Shifting(_StubEnv):
        @property
        def privileged_state(self):
            if self._done:
                return None
            if self._t > 2:
                return {k: 1.0 for k in KEYS[:-1]}  # drops "angle"
            return {k: float(self._t) for k in KEYS}

    c = Collector(lambda: _Shifting(6), RandomPolicy(4, seed=0))
    with pytest.raises(DataIntegrityError, match="lost keys mid-episode"):
        c.collect_episode(seed=0)
    assert c.crash_count == 0
    c.close()


def test_truncated_episode_ends_the_loop():
    """A time-limit cutoff must end collection just like a terminal state."""
    c = Collector(lambda: _StubEnv(6, truncate=True), RandomPolicy(4, seed=0), 1000)
    ep = c.collect_episode(seed=0)
    assert ep is not None and ep.length == 6
    c.close()


def test_truncation_is_recorded_separately_from_termination():
    """Collapsing the two is the time-limit bootstrapping bug."""
    c = Collector(lambda: _StubEnv(6, truncate=True), RandomPolicy(4, seed=0), 1000)
    ep = c.collect_episode(seed=0)
    assert ep.truncated[-1], "final step should be flagged truncated"
    assert not ep.terminated[-1], "a time limit is not a true terminal state"
    assert not ep.terminated.any()
    c.close()


def test_collector_passes_the_episode_seed_to_the_env():
    """Without this, a dropped env seed is only caught at the ViZDoomEnv level."""
    env = _StubEnv(6)
    c = Collector(lambda: env, RandomPolicy(4, seed=0), 1000)
    c.collect_episode(seed=11)
    c.collect_episode(seed=12)
    assert env.seeds_seen == [11, 12]
    c.close()


def test_malformed_obs_is_not_reported_as_an_engine_crash():
    """Episode validation failures are our bug, not the engine's."""

    class _BadObs(_StubEnv):
        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            return obs.astype(np.float32), reward, term, trunc, info

    c = Collector(lambda: _BadObs(6), RandomPolicy(4, seed=0), 1000)
    with pytest.raises(DataIntegrityError, match="failed validation"):
        c.collect_episode(seed=0)
    assert c.crash_count == 0
    c.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_collector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data.collector'`

- [ ] **Step 3: Write the implementation**

```python
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
    This also covers episodes that fail `Episode`'s own validation --
    `__post_init__` raises `ValueError` for obs/actions/privileged shape and
    dtype violations, and that failure is our bug too, not the engine's.
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
        except (DataIntegrityError, MemoryError):
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

        # obs is intentionally NOT force-cast to uint8 here: a real engine
        # already emits uint8 frames, so casting is a no-op for good data and
        # would only paper over a malformed one -- defeating Episode's own
        # dtype check below and letting the exact bug this wrapper exists to
        # catch slip back out as a phantom engine crash instead.
        try:
            return Episode(
                obs=np.stack(frames),
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
        except ValueError as exc:
            raise DataIntegrityError(
                f"collected episode failed validation: {exc}"
            ) from exc

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_collector.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data/collector.py tests/data/test_collector.py
git commit -m "feat: episode collector with crash recovery and terminal-frame handling"
```

## Task 10: Replay buffer with eviction

**Files:**
- Create: `src/mbfps/data/buffer.py`
- Create: `tests/data/test_buffer.py`

**Interfaces:**
- Consumes: `Episode`, `save_episode`, `load_episode`.
- Produces: `ReplayBuffer(root: Path, capacity_transitions: int)` with `add(ep: Episode) -> Path`, `episode_paths() -> list[Path]`, `stray_paths() -> list[Path]`, `load_all() -> list[Episode]`, `n_episodes -> int`, `n_transitions -> int`.

**Episode length is encoded in the filename** (`ep_000042_len00318.npz`). Reading `.length` by decompressing the file would make `n_transitions` — called on every `add()` and every progress print — decompress every stored episode's full frame stack, giving O(n²) collection and putting M1's bounded-wall-clock criterion out of reach. The filename is metadata that costs nothing to read.

Three failure modes fall out of trusting the filename this much, and each needs its own guard rather than a shared one:

- A file matching `ep_*.npz` but not the strict name pattern (`ep_.npz`, `ep_1_len.npz`) is invisible to `episode_paths()`, so it is never counted and never evicted — a silent, permanent disk leak. Auto-deleting it would risk destroying something a user placed there, so `stray_paths()` surfaces it instead, and the constructor logs a warning naming the count and up to three examples.
- The encoded length is trusted without verification everywhere except `load_all()`, which already decompresses every episode to build `Episode` objects — the one place checking the actual `.length` against the filename costs nothing extra. `load_all()` raises `ValueError` naming both numbers on a mismatch.
- The counter that picks the next index is seeded once at construction and never resynced, so two `ReplayBuffer` instances over the same directory can compute the same next index; since length is part of the filename, a bare `exists()` check would miss the collision anyway. `add()` reserves its index by checking the directory itself (`_next_path`), advancing past any index another writer already used.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_buffer.py
import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, policy: str = "random", seed: int = 0) -> Episode:
    return Episode(
        obs=np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name=policy,
        seed=seed,
        scenario="my_way_home",
    )


def test_add_writes_a_file(tmp_path):
    assert ReplayBuffer(tmp_path, capacity_transitions=100).add(make_episode(10)).is_file()


def test_counts_track_contents(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    buf.add(make_episode(10))
    buf.add(make_episode(15))
    assert buf.n_episodes == 2
    assert buf.n_transitions == 25


def _poison_load_episode(monkeypatch):
    """Make any load_episode call fail, so decompression becomes detectable."""
    import mbfps.data.buffer as buffer_module

    def _boom(*args, **kwargs):
        raise AssertionError("capacity accounting must not load episode files")

    monkeypatch.setattr(buffer_module, "load_episode", _boom)


def test_n_transitions_does_not_decompress_episodes(tmp_path, monkeypatch):
    buf = ReplayBuffer(tmp_path, capacity_transitions=1000)
    for _ in range(3):
        buf.add(make_episode(10))
    _poison_load_episode(monkeypatch)
    assert buf.n_transitions == 30


def test_eviction_does_not_decompress_episodes(tmp_path, monkeypatch):
    """`_evict` is the hotter path -- `add()` calls it on every episode.

    Poison BEFORE the adds and use a capacity that forces eviction, so this
    actually covers `_evict`. Poisoning afterwards leaves the O(n^2)
    regression restorable with the whole suite still green.
    """
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    _poison_load_episode(monkeypatch)
    for _ in range(4):
        buf.add(make_episode(10))
    assert buf.n_episodes == 2, "eviction must have run without decompressing"
    assert buf.n_transitions <= 25


def test_eviction_respects_capacity(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    for _ in range(5):
        buf.add(make_episode(10))
    assert buf.n_transitions <= 25


def test_eviction_removes_oldest_first(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    for seed in (1, 2, 3):
        buf.add(make_episode(10, seed=seed))
    seeds = {ep.seed for ep in buf.load_all()}
    assert 1 not in seeds, "oldest episode should have been evicted"
    assert {2, 3} <= seeds


def test_eviction_also_removes_cached_features(tmp_path):
    """Orphaned .features.npy files would defeat the disk-growth bound."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    first = buf.add(make_episode(10, seed=1))
    features = first.with_suffix(".features.npy")
    np.save(features, np.zeros((11, 4, 8), dtype=np.float16))
    assert features.is_file()
    buf.add(make_episode(10, seed=2))
    buf.add(make_episode(10, seed=3))
    assert not first.is_file()
    assert not features.is_file(), "feature cache outlived its episode"


def test_buffer_reopens_existing_directory(tmp_path):
    ReplayBuffer(tmp_path, capacity_transitions=100).add(make_episode(10))
    assert ReplayBuffer(tmp_path, capacity_transitions=100).n_episodes == 1


def test_indices_continue_after_reopen(tmp_path):
    ReplayBuffer(tmp_path, capacity_transitions=1000).add(make_episode(4, seed=1))
    reopened = ReplayBuffer(tmp_path, capacity_transitions=1000)
    reopened.add(make_episode(4, seed=2))
    assert reopened.n_episodes == 2, "second buffer must not overwrite the first file"


def test_load_all_returns_episodes(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    buf.add(make_episode(4, policy="scripted"))
    episodes = buf.load_all()
    assert len(episodes) == 1
    assert episodes[0].policy_name == "scripted"


def test_single_oversized_episode_is_kept(tmp_path):
    """Evicting to empty would make the buffer useless; keep the newest."""
    buf = ReplayBuffer(tmp_path, capacity_transitions=5)
    buf.add(make_episode(50))
    assert buf.n_episodes == 1


def test_stray_files_are_reported(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    buf.add(make_episode(10))
    stray_a = tmp_path / "ep_.npz"
    stray_b = tmp_path / "ep_1_len.npz"
    stray_a.touch()
    stray_b.touch()
    assert set(buf.stray_paths()) == {stray_a, stray_b}
    assert buf.n_episodes == 1, "strays must not be counted as episodes"


def test_stray_files_are_not_deleted(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=5)
    stray_a = tmp_path / "ep_.npz"
    stray_b = tmp_path / "ep_1_len.npz"
    stray_a.touch()
    stray_b.touch()
    for seed in (1, 2, 3):
        buf.add(make_episode(10, seed=seed))  # forces eviction, capacity=5
    assert stray_a.is_file(), "eviction must never delete unrecognised files"
    assert stray_b.is_file(), "eviction must never delete unrecognised files"


def test_load_all_detects_filename_length_divergence(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    path = buf.add(make_episode(10))
    bad_path = path.with_name(path.name.replace("_len00010.npz", "_len00099.npz"))
    path.rename(bad_path)
    with pytest.raises(ValueError, match=r"99.*10"):
        buf.load_all()


def test_two_buffers_over_one_directory_do_not_overwrite(tmp_path):
    buf_a = ReplayBuffer(tmp_path, capacity_transitions=1000)
    buf_b = ReplayBuffer(tmp_path, capacity_transitions=1000)
    buf_a.add(make_episode(10, seed=101))
    buf_b.add(make_episode(10, seed=202))
    assert buf_a.n_episodes == 2
    seeds = {ep.seed for ep in buf_a.load_all()}
    assert seeds == {101, 202}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data.buffer'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/data/buffer.py
"""Fixed-capacity, disk-backed episode store.

Capacity is measured in transitions rather than episodes because episode length
varies with policy and scenario; transitions are what actually bound disk use
and training-set size.

Filenames carry both the ordering index and the episode length
(`ep_000042_len00318.npz`), so capacity accounting never has to open a file.
"""

import itertools
import logging
import re
from pathlib import Path

from mbfps.data.episode import Episode, load_episode, save_episode

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^ep_(\d+)_len(\d+)\.npz$")


class ReplayBuffer:
    """Episodes on disk, oldest evicted once capacity is exceeded."""

    def __init__(self, root: Path, capacity_transitions: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_transitions = capacity_transitions
        indices = [self._parse(p)[0] for p in self.episode_paths()]
        self._counter = itertools.count(max(indices) + 1 if indices else 0)
        strays = self.stray_paths()
        if strays:
            examples = ", ".join(p.name for p in strays[:3])
            logger.warning(
                "%d file(s) in %s match the episode glob but not the episode "
                "filename pattern; they are excluded from all counts and from "
                "eviction and will accumulate forever (e.g. %s)",
                len(strays),
                self.root,
                examples,
            )

    @staticmethod
    def _parse(path: Path) -> tuple[int, int]:
        """Return (index, length) parsed from an episode filename."""
        match = _NAME_RE.match(path.name)
        if match is None:
            raise ValueError(f"malformed episode filename: {path.name}")
        return int(match.group(1)), int(match.group(2))

    def episode_paths(self) -> list[Path]:
        """Episode files, oldest first."""
        return sorted(
            (p for p in self.root.glob("ep_*.npz") if _NAME_RE.match(p.name)),
            key=lambda p: self._parse(p)[0],
        )

    def stray_paths(self) -> list[Path]:
        """Files matching the episode glob whose names this class cannot parse.

        They are excluded from every count AND from eviction, so they accumulate
        forever. Deleting them automatically would risk destroying something a
        user placed here, so they are surfaced instead.
        """
        return sorted(
            p for p in self.root.glob("ep_*.npz") if not _NAME_RE.match(p.name)
        )

    @property
    def n_episodes(self) -> int:
        return len(self.episode_paths())

    @property
    def n_transitions(self) -> int:
        """Total transitions stored, read from filenames without decompressing."""
        return sum(self._parse(p)[1] for p in self.episode_paths())

    def add(self, ep: Episode) -> Path:
        """Write `ep` and evict oldest episodes until within capacity."""
        path = self._next_path(ep.length)
        save_episode(ep, path)
        self._evict()
        return path

    def _next_path(self, length: int) -> Path:
        """Reserve the next unused index, resyncing if another writer advanced.

        The counter alone is not enough: a second ReplayBuffer over the same
        directory seeds its own counter at construction and can hand out an
        index this one has already used. Length is part of the filename, so a
        bare exists() check would miss the collision.
        """
        while True:
            index = next(self._counter)
            if not any(self.root.glob(f"ep_{index:06d}_len*.npz")):
                return self.root / f"ep_{index:06d}_len{length:05d}.npz"

    def load_all(self) -> list[Episode]:
        """Load every stored episode, oldest first.

        This is the one place a filename/content divergence is detectable: the
        encoded length is otherwise trusted as-is (by `n_transitions` and
        `_evict`) to keep capacity accounting O(1), but `load_all` already pays
        the decompression cost for every episode, so verifying here is free.
        """
        episodes = []
        for path in self.episode_paths():
            ep = load_episode(path)
            _, encoded_length = self._parse(path)
            if ep.length != encoded_length:
                raise ValueError(
                    f"{path.name}: filename encodes length {encoded_length}, "
                    f"but the episode actually has {ep.length} transitions"
                )
            episodes.append(ep)
        return episodes

    def _evict(self) -> None:
        paths = self.episode_paths()
        lengths = [self._parse(p)[1] for p in paths]
        total = sum(lengths)
        # Never evict the newest episode: an empty buffer is worse than an
        # oversized one.
        for path, length in zip(paths[:-1], lengths[:-1]):
            if total <= self.capacity_transitions:
                break
            path.unlink()
            path.with_suffix(".features.npy").unlink(missing_ok=True)
            total -= length
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_buffer.py -v`
Expected: 15 passed.

Sanity-check all four guards by mutation, restoring between each:

1. Make `stray_paths()` return `[]` -- confirm `test_stray_files_are_reported` fails.
2. Remove the divergence check from `load_all()` -- confirm `test_load_all_detects_filename_length_divergence` fails.
3. Revert `add()` to build the path directly from `next(self._counter)` without the directory check -- confirm `test_two_buffers_over_one_directory_do_not_overwrite` fails.
4. Revert `n_transitions` to `sum(load_episode(p).length for p in self.episode_paths())` -- confirm the existing O(1) guard tests (`test_n_transitions_does_not_decompress_episodes`, `test_eviction_does_not_decompress_episodes`) fail.

A mutation that leaves the suite green means that guard is not guarding.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data/buffer.py tests/data/test_buffer.py
git commit -m "feat: replay buffer with O(1) capacity accounting and feature-cache eviction"
```

## Task 11: Sequence loader

**Files:**
- Create: `src/mbfps/data/loader.py`
- Create: `tests/data/test_loader.py`

**Interfaces:**
- Consumes: `ReplayBuffer`, `Episode`.
- Produces: `SequenceLoader(buffer: ReplayBuffer, batch_size: int = 16, seq_len: int = 64, seed: int = 0)` with `sample(include_privileged: bool = False) -> dict[str, np.ndarray]`. Keys: `obs (B, T+1, 112, 112, 3) uint8`, `actions (B, T) int32`, `rewards (B, T) float32`, `terminated (B, T) bool`, `truncated (B, T) bool`, `episode_index (B,) int32`. Adds `privileged (B, T+1, K) float32` **only** when `include_privileged=True`.

`include_privileged` defaults to False so the safe path is the default one: a training loop that never passes the flag can never receive privileged state.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_loader.py
import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    """Every frame is filled with `fill`, so windows carry episode identity."""
    return Episode(
        obs=np.full((t + 1, *OBS_SHAPE), fill, dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.full((t + 1, len(KEYS)), float(fill), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3, 4):
        buf.add(make_episode(t=80, fill=fill))
    return buf


def test_batch_shapes(buffer):
    batch = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0).sample()
    assert batch["obs"].shape == (4, 17, *OBS_SHAPE)
    assert batch["actions"].shape == (4, 16)
    assert batch["rewards"].shape == (4, 16)
    assert batch["terminated"].shape == (4, 16)
    assert batch["truncated"].shape == (4, 16)
    assert batch["episode_index"].shape == (4,)


def test_batch_dtypes(buffer):
    batch = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0).sample()
    assert batch["obs"].dtype == np.uint8
    assert batch["actions"].dtype == np.int32
    assert batch["rewards"].dtype == np.float32
    assert batch["terminated"].dtype == bool
    assert batch["truncated"].dtype == bool


def test_windows_never_cross_episode_boundaries(buffer):
    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    for _ in range(20):
        for window in loader.sample()["obs"]:
            assert len(np.unique(window)) == 1, "window spans two episodes"


def test_privileged_absent_by_default(buffer):
    """The safe path is the default path."""
    assert "privileged" not in SequenceLoader(buffer, 4, 16, seed=0).sample()


def test_privileged_present_only_when_requested(buffer):
    batch = SequenceLoader(buffer, 4, 16, seed=0).sample(include_privileged=True)
    assert batch["privileged"].shape == (4, 17, len(KEYS))


def test_sampling_is_reproducible(buffer):
    a = SequenceLoader(buffer, 4, 16, seed=99).sample()
    b = SequenceLoader(buffer, 4, 16, seed=99).sample()
    assert np.array_equal(a["obs"], b["obs"])
    assert np.array_equal(a["episode_index"], b["episode_index"])


def test_batch_draws_from_multiple_episodes(buffer):
    """A loader stuck on one episode would train on a fraction of the data.

    The buffer fixture holds four usable episodes, so eight draws landing on a
    single one has probability ~6e-5 per batch; over five batches it is
    negligible, and the loader is seeded, so this is deterministic.
    """
    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    seen = set()
    for _ in range(5):
        seen.update(loader.sample()["episode_index"].tolist())
    assert len(seen) > 1, f"every sample came from episode(s) {seen}"


def test_window_offsets_vary(buffer):
    """A loader fixed at offset 0 would only ever show each episode's opening.

    Each episode here is filled with a constant, so the offset is invisible in
    the pixels. Detect it through the frames themselves: write a per-timestep
    marker into one episode and check that sampled windows start at different
    points within it.
    """
    from mbfps.data.episode import load_episode, save_episode

    path = buffer.episode_paths()[0]
    episode = load_episode(path)
    for t in range(episode.obs.shape[0]):
        episode.obs[t, 0, 0, 0] = t % 251  # a per-timestep marker
    save_episode(episode, path)

    loader = SequenceLoader(buffer, batch_size=8, seq_len=16, seed=0)
    starts = set()
    for _ in range(10):
        batch = loader.sample()
        for row, idx in zip(batch["obs"], batch["episode_index"]):
            if idx == 0:
                starts.add(int(row[0, 0, 0, 0]))
    assert len(starts) > 1, f"all sampled windows began at the same offset: {starts}"


def test_episodes_shorter_than_seq_len_are_skipped(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=4, fill=9))
    buf.add(make_episode(t=80, fill=7))
    batch = SequenceLoader(buf, batch_size=8, seq_len=16, seed=0).sample()
    assert np.all(batch["obs"] == 7)


def test_no_usable_episodes_raises(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=4, fill=9))
    with pytest.raises(ValueError, match="no episodes long enough"):
        SequenceLoader(buf, batch_size=4, seq_len=64, seed=0).sample()


def test_large_buffer_logs_a_memory_warning(tmp_path, caplog, monkeypatch):
    """The eager load is a real constraint on a 16 GB machine; make it visible."""
    import mbfps.data.loader as loader_module

    monkeypatch.setattr(loader_module, "_WARN_BYTES", 1)
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(make_episode(t=80, fill=1))
    with caplog.at_level("WARNING"):
        SequenceLoader(buf, batch_size=2, seq_len=16, seed=0)
    assert "resident in RAM" in caplog.text


def test_small_buffer_logs_no_warning(buffer, caplog):
    with caplog.at_level("WARNING"):
        SequenceLoader(buffer, batch_size=2, seq_len=16, seed=0)
    assert "resident in RAM" not in caplog.text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data.loader'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/data/loader.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_loader.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data/loader.py tests/data/test_loader.py
git commit -m "feat: sequence loader with episode-boundary safety"
```

---

## Task 12: Frozen DINOv2 feature cache

**Files:**
- Create: `src/mbfps/data/features.py`
- Create: `tests/data/test_features.py`
- Modify: `pyproject.toml` (no change needed — `transformers` is already a dependency)

**Interfaces:**
- Consumes: `get_device` from `mbfps.utils.device`; `OBS_SHAPE`.
- Produces: `FeatureExtractor(model_name: str = "facebook/dinov2-small", device: str = "mps")` with `encode(frames: np.ndarray) -> np.ndarray` mapping `(N, 112, 112, 3) uint8` to `(N, 64, 384) float16`; constants `N_PATCHES = 64`, `FEATURE_DIM = 384`. Also `cache_episode_features(ep_path: Path, extractor: FeatureExtractor) -> Path` writing a sibling `.features.npy`.

Per spec §3.3: the learned bottleneck sits **downstream** of this cache, so the cache stores the frozen backbone's raw 64x384 patch features. Stored as float16 (~49KB/frame, about 1.3x a raw uint8 frame).

**Determinism is per-device, not cross-device.** Repeat-encoding the same frame on the same device is byte-identical, but CPU and MPS outputs for the same frame differ in roughly 3% of float16 elements (measured on `facebook/dinov2-small`). Collection defaults to MPS. Constraint for later plans: **the feature cache must be generated once on a single device and reused for every experimental arm** — regenerating it per arm, or mixing devices within one dataset, silently introduces a confound indistinguishable from a real signal.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_features.py
import numpy as np
import pytest
import torch

from mbfps.data.features import (
    FEATURE_DIM,
    N_PATCHES,
    FeatureExtractor,
    cache_episode_features,
)
from mbfps.envs.protocol import OBS_SHAPE

pytestmark = pytest.mark.slow  # downloads ~88MB on first run


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(device="cpu")


def test_patch_grid_is_8x8():
    assert N_PATCHES == 64
    assert FEATURE_DIM == 384


def test_encode_output_shape(extractor):
    frames = np.random.randint(0, 256, (3, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).shape == (3, N_PATCHES, FEATURE_DIM)


def test_encode_output_dtype_is_float16(extractor):
    frames = np.random.randint(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert extractor.encode(frames).dtype == np.float16


def test_encode_is_byte_identical_on_repeat(extractor):
    """The cache is only sound if the frozen backbone is deterministic."""
    frames = np.random.randint(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    a = extractor.encode(frames)
    b = extractor.encode(frames)
    assert np.array_equal(a, b)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_encode_is_byte_identical_on_repeat_on_mps():
    """Collection runs on MPS by default, so CPU repeatability is not enough."""
    mps_extractor = FeatureExtractor(device="mps")
    frames = np.random.default_rng(1).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    assert np.array_equal(mps_extractor.encode(frames), mps_extractor.encode(frames))


def test_encode_distinguishes_different_frames(extractor):
    frames = np.stack(
        [
            np.zeros(OBS_SHAPE, dtype=np.uint8),
            np.full(OBS_SHAPE, 255, dtype=np.uint8),
        ]
    )
    out = extractor.encode(frames)
    assert not np.allclose(out[0].astype(np.float32), out[1].astype(np.float32))


def test_encode_rejects_wrong_shape(extractor):
    with pytest.raises(ValueError, match="expected frames of shape"):
        extractor.encode(np.zeros((2, 64, 64, 3), dtype=np.uint8))


def _encode_preprocessed(extractor, x):
    """Run the frozen backbone on already-preprocessed NHWC float32 input."""
    import torch

    tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(extractor.device)
    with torch.no_grad():
        out = extractor.model(pixel_values=tensor).last_hidden_state
    return out[:, 1:, :].cpu().numpy().astype(np.float32)


def test_encode_applies_imagenet_normalization(extractor):
    """Normalisation must be applied -- but any equivalent formulation is fine.

    An exact-equality check against the implementation's own expression is a
    change-detector: a mathematically-equivalent refactor shifts results by
    ~1e-6, which survives the float16 cast and fails the comparison. So this
    asserts closeness to the reference AND clear separation from the
    un-normalised alternative.
    """
    frames = np.random.default_rng(0).integers(0, 256, (2, *OBS_SHAPE), dtype=np.uint8)
    actual = extractor.encode(frames).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    scaled = frames.astype(np.float32) / 255.0

    normalised = _encode_preprocessed(extractor, (scaled - mean) / std)
    unnormalised = _encode_preprocessed(extractor, scaled)

    assert np.allclose(actual, normalised, atol=2e-2), (
        "encode() does not match ImageNet-normalised input"
    )
    separation = np.abs(normalised - unnormalised).mean()
    assert separation > 1e-2, (
        f"normalised and un-normalised encodings differ by only {separation:.2e}; "
        "this test cannot detect whether normalisation is applied"
    )


def test_cache_episode_features_writes_sibling_file(tmp_path, extractor):
    from mbfps.data.episode import Episode, save_episode

    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    ep = Episode(
        obs=np.zeros((4, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(3, dtype=np.int32),
        rewards=np.zeros(3, dtype=np.float32),
        terminated=np.zeros(3, dtype=bool),
        truncated=np.zeros(3, dtype=bool),
        privileged=np.zeros((4, len(keys)), dtype=np.float32),
        privileged_keys=keys,
        policy_name="random",
        seed=0,
        scenario="my_way_home",
    )
    path = tmp_path / "ep_000000.npz"
    save_episode(ep, path)
    out = cache_episode_features(path, extractor)
    assert out.is_file()
    saved = np.load(out)
    assert saved.shape == (4, N_PATCHES, FEATURE_DIM)
    # Mutation-testing gap-fill: shape alone doesn't catch writing float32
    # instead of float16, which would blow the ~49KB/frame storage budget
    # per spec §3.3.
    assert saved.dtype == np.float16
```

- [ ] **Step 2: Register the `slow` marker**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: downloads model weights or runs a long benchmark"]
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data.features'`

- [ ] **Step 4: Write the implementation**

```python
# src/mbfps/data/features.py
"""Frozen DINOv2 patch-feature cache.

The backbone never updates, so each frame is encoded once at collection time and
the features are reused for every training step. The learned bottleneck lives
downstream, so this cache holds the raw 64x384 patch grid rather than a
bottlenecked vector.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from mbfps.envs.protocol import OBS_SHAPE
from mbfps.utils.device import get_device

N_PATCHES = 64
"""112 / 14 = 8, so DINOv2's patch grid is 8x8."""

FEATURE_DIM = 384
"""Hidden size of DINOv2-small."""

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FeatureExtractor:
    """Encodes frames with a frozen self-supervised backbone."""

    def __init__(
        self, model_name: str = "facebook/dinov2-small", device: str = "mps"
    ) -> None:
        self.device = get_device(prefer=device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode `(N, 112, 112, 3)` uint8 frames to `(N, 64, 384)` float16.

        Raises:
            ValueError: if `frames` does not match the expected shape.
        """
        if frames.ndim != 4 or frames.shape[1:] != OBS_SHAPE:
            raise ValueError(
                f"expected frames of shape (N, {OBS_SHAPE}), got {frames.shape}"
            )
        x = frames.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        out = self.model(pixel_values=tensor).last_hidden_state
        patches = out[:, 1:, :]  # drop the CLS token
        if patches.shape[1] != N_PATCHES:
            raise ValueError(
                f"expected {N_PATCHES} patch tokens, got {patches.shape[1]}; "
                f"check that the input is 112x112 and the patch size is 14"
            )
        return patches.to(torch.float16).cpu().numpy()


def cache_episode_features(
    ep_path: Path, extractor: FeatureExtractor, batch_size: int = 32
) -> Path:
    """Encode an episode's frames and write a sibling `.features.npy`.

    Called by `scripts/collect.py --cache-features` as each episode is written.
    Frames are encoded in batches: an episode can run to a thousand frames, and
    a single forward pass over all of them would exhaust unified memory.

    Returns:
        Path to the written feature file.
    """
    ep_path = Path(ep_path)
    # allow_pickle is left at its default of False, matching load_episode's
    # rationale in episode.py: obs is a plain uint8 array and round-trips
    # through npz without pickle, so there is no need to enable arbitrary
    # code execution on load.
    with np.load(ep_path) as data:
        obs = data["obs"]
    chunks = [
        extractor.encode(obs[i : i + batch_size]) for i in range(0, len(obs), batch_size)
    ]
    features = np.concatenate(chunks, axis=0)
    out_path = ep_path.with_suffix(".features.npy")
    np.save(out_path, features)
    return out_path
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_features.py -v`
Expected: 9 passed on a machine with MPS available (8 passed, 1 skipped otherwise — `test_encode_is_byte_identical_on_repeat_on_mps` requires an MPS device). Weights are cached locally after the first run.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/data/features.py tests/data/test_features.py pyproject.toml
git commit -m "feat: frozen DINOv2 patch-feature cache"
```

---

## Task 13: M1 exit gate — collection CLI, coverage report, privileged isolation

**Files:**
- Create: `scripts/collect.py`
- Create: `scripts/coverage_report.py`
- Create: `tests/test_privileged_isolation.py`
- Modify: `pyproject.toml` (add `matplotlib`)

**Interfaces:**
- Consumes: everything from Tasks 1–12.
- Produces: `scripts/collect.py` writing episodes under `data/<scenario>/`; `scripts/coverage_report.py` writing `runs/coverage_<scenario>.png` and printing per-policy visitation statistics. No new library API.

This is the M1 exit gate: **a dataset exists, the loader is benchmarked, and the scripted policy demonstrably reaches states random play does not.**

- [ ] **Step 1: Add matplotlib**

```bash
.venv/bin/python -m pip install "matplotlib>=3.8"
```

In `pyproject.toml`, add `"matplotlib>=3.8",` to the `dependencies` list.

- [ ] **Step 2: Write the failing isolation test**

```python
# tests/test_privileged_isolation.py
"""Privileged state must never reach a training tensor.

The spec makes `privileged_state` evaluation-only. A leak would let the world
model or policy read ground truth the agent could not observe, silently
invalidating every result. This is asserted rather than documented.
"""

import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")
SENTINEL = 12345.0


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    buf.add(
        Episode(
            obs=np.zeros((81, *OBS_SHAPE), dtype=np.uint8),
            actions=np.zeros(80, dtype=np.int32),
            rewards=np.zeros(80, dtype=np.float32),
            terminated=np.zeros(80, dtype=bool),
            truncated=np.zeros(80, dtype=bool),
            privileged=np.full((81, len(KEYS)), SENTINEL, dtype=np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="my_way_home",
        )
    )
    return buf


def test_default_batch_has_no_privileged_key(buffer):
    assert "privileged" not in SequenceLoader(buffer, 4, 16, seed=0).sample()


def test_sentinel_absent_from_every_default_batch_array(buffer):
    """No array in a training batch may contain the privileged sentinel."""
    batch = SequenceLoader(buffer, 4, 16, seed=0).sample()
    for name, array in batch.items():
        assert not np.any(
            np.asarray(array, dtype=np.float64) == SENTINEL
        ), f"privileged sentinel leaked into training array {name!r}"


def test_privileged_reachable_only_via_explicit_flag(buffer):
    batch = SequenceLoader(buffer, 4, 16, seed=0).sample(include_privileged=True)
    assert np.all(batch["privileged"] == SENTINEL)


def test_env_privileged_values_are_not_in_the_observation():
    """The engine's own observation must not encode privileged values."""
    from mbfps.envs.vizdoom_env import ViZDoomEnv

    env = ViZDoomEnv(scenario="my_way_home", frame_skip=4, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        state = env.privileged_state
        assert obs.dtype == np.uint8, "observations are pixels, not state vectors"
        assert obs.shape == OBS_SHAPE
        assert state is not None and set(state) == set(KEYS)
    finally:
        env.close()
```

- [ ] **Step 3: Run the isolation test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_privileged_isolation.py -v`
Expected: 4 passed. These pass by construction given Task 11's design — they exist to fail loudly if a future change adds privileged data to the default path.

- [ ] **Step 4: Write the collection script**

```python
# scripts/collect.py
"""Collect a mixed-policy ViZDoom dataset.

Per spec §5.2, a single random policy has poor state coverage. This alternates
`random` and `scripted` collection so the world model sees geometry random play
never reaches.
"""

import argparse
import time
from pathlib import Path

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.collector import Collector
from mbfps.data.features import cache_episode_features
from mbfps.data.policies import RandomPolicy, ScriptedPolicy
from mbfps.envs.registry import make_env
from mbfps.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="my_way_home")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="encode each episode with the frozen DINOv2 backbone as it is written",
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    # A real episode is ~19.8 MB resident once loaded by SequenceLoader (526
    # frames of 112x112x3 uint8). At the default below (60,000 transitions,
    # ~114 episodes) that is ~2.3 GB resident -- comfortably inside a 16 GB
    # unified-memory machine shared with the model. Raise this only knowing
    # what you are buying: 200,000 transitions (~380 episodes) costs ~7.5 GB.
    parser.add_argument("--capacity", type=int, default=60_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    out = args.out or Path("data") / args.scenario
    buffer = ReplayBuffer(out, capacity_transitions=args.capacity)

    def env_factory():
        return make_env(
            "vizdoom", scenario=args.scenario, frame_skip=args.frame_skip
        )

    probe = env_factory()
    n_actions, button_names = probe.action_space.n, probe.button_names
    probe.close()

    collectors = {
        "random": Collector(
            env_factory, RandomPolicy(n_actions, seed=args.seed), args.max_steps
        ),
        "scripted": Collector(
            env_factory, ScriptedPolicy(button_names, seed=args.seed), args.max_steps
        ),
    }

    extractor = None
    if args.cache_features:
        from mbfps.data.features import FeatureExtractor

        extractor = FeatureExtractor()

    start, kept = time.perf_counter(), 0
    for i in range(args.episodes):
        name = "random" if i % 2 == 0 else "scripted"
        episode = collectors[name].collect_episode(seed=args.seed + i)
        if episode is not None:
            path = buffer.add(episode)
            if extractor is not None and path.is_file():
                cache_episode_features(path, extractor)
            kept += 1
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{args.episodes}] kept={kept} transitions={buffer.n_transitions}")

    elapsed = time.perf_counter() - start
    for collector in collectors.values():
        collector.close()

    print(f"episodes_kept={kept} crashes={sum(c.crash_count for c in collectors.values())}")
    print(f"transitions={buffer.n_transitions} elapsed_s={elapsed:.1f}")
    print(f"transitions_per_second={buffer.n_transitions / elapsed:.1f}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run a small collection to verify it works end to end**

Run: `.venv/bin/python scripts/collect.py --episodes 20`
Expected: prints progress, then `episodes_kept=20`, a positive `transitions_per_second`, and `output=data/my_way_home`.

**`episodes_kept` must equal the episode count.** A run reporting `episodes_kept=0 crashes=20` means the collector is discarding every terminating episode — re-read Task 9's terminal-frame handling before continuing rather than lowering the expectation.

- [ ] **Step 6: Write the coverage report**

The gate is **per-episode reach**, not aggregate reach pooled across episodes. See
`scripts/coverage_report.py`:

```python
# scripts/coverage_report.py
"""Per-policy state-visitation histogram and per-episode coverage gate.

Coverage gaps are invisible at the milestone that creates them and only become
symptomatic three milestones later, as an M4 agent with high imagined return and
near-zero real return. This makes them visible now.

The gate compares *per-episode* reach (occupied cells within a single episode),
not aggregate reach pooled across many episodes. Aggregate cell counts scale
with maze size and episode count: once enough episodes have been collected on
a small, fixed maze like `my_way_home`, both policies fill nearly all of the
reachable grid and the aggregate count saturates -- at that point it measures
how big the maze is and how much data was collected, not how good the policy
is. Per-episode reach does not have this problem: it asks how much ground one
episode of a given policy covers, which is exactly what the scripted policy
exists to improve over random.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbfps.data.buffer import ReplayBuffer  # noqa: E402


def _occupied_mask(xy: np.ndarray, bins: int, x_range, y_range) -> np.ndarray:
    hist = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[x_range, y_range])[0]
    return hist > 0


def _occupied_count(xy: np.ndarray, bins: int, x_range, y_range) -> int:
    return int(_occupied_mask(xy, bins, x_range, y_range).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/my_way_home"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--gate-ratio",
        type=float,
        default=1.2,
        help=(
            "Pass/fail threshold for the per-episode-reach gate: the mean "
            "occupied-cell count reached per scripted episode must exceed "
            "the mean per random episode by at least this multiplicative "
            "margin, i.e. mean(scripted) > mean(random) * ratio. The gate "
            "does not depend on a frame budget -- episode length is itself "
            "part of what a policy controls, so per-episode frame counts "
            "are not equalised."
        ),
    )
    args = parser.parse_args()

    episodes = ReplayBuffer(args.data, capacity_transitions=10**9).load_all()
    if not episodes:
        raise SystemExit(f"no episodes found in {args.data}")

    keys = episodes[0].privileged_keys
    x_i, y_i = keys.index("pos_x"), keys.index("pos_y")

    by_policy: dict[str, list[np.ndarray]] = defaultdict(list)
    for ep in episodes:
        by_policy[ep.policy_name].append(ep.privileged[:, [x_i, y_i]])

    names = sorted(by_policy)
    stacked = {n: np.concatenate(by_policy[n]) for n in names}

    # Shared bin range across all policies and episodes so that every
    # occupied-cell count below -- per-episode, aggregate, or union -- refers
    # to the same grid and is directly comparable.
    all_xy = np.concatenate(list(stacked.values()))
    x_range = (all_xy[:, 0].min(), all_xy[:, 0].max())
    y_range = (all_xy[:, 1].min(), all_xy[:, 1].max())

    # --- primary gate: per-episode reach -------------------------------
    # Frame counts differ slightly per episode; they are intentionally NOT
    # equalised here -- episode length is itself part of what the policy
    # controls, and all episodes in this dataset run to the same timeout,
    # so a policy that reaches more of the maze before timing out is doing
    # exactly what it is supposed to do.
    per_episode_counts: dict[str, np.ndarray] = {}
    for name in names:
        per_episode_counts[name] = np.array(
            [_occupied_count(xy, args.bins, x_range, y_range) for xy in by_policy[name]],
            dtype=float,
        )

    print("--- per-episode reach (the gate) ---")
    means: dict[str, float] = {}
    for name in names:
        counts = per_episode_counts[name]
        means[name] = float(counts.mean())
        sd = float(counts.std(ddof=0))
        print(
            f"{name}: n_episodes={len(counts)} "
            f"mean_cells_per_episode={counts.mean():.1f} sd={sd:.1f}"
        )

    have_both = "random" in means and "scripted" in means
    ratio = means["scripted"] / means["random"] if have_both else float("nan")
    if have_both:
        print(f"ratio (mean scripted / mean random) = {ratio:.3f}")

    # --- union analysis: does more of the same data buy coverage? ------
    print()
    print("--- union analysis (does more of the same data buy coverage?) ---")
    masks = {n: _occupied_mask(stacked[n], args.bins, x_range, y_range) for n in names}
    if have_both:
        random_mask, scripted_mask = masks["random"], masks["scripted"]
        random_alone = int(random_mask.sum())
        scripted_alone = int(scripted_mask.sum())
        scripted_only = int((scripted_mask & ~random_mask).sum())
        union = int((random_mask | scripted_mask).sum())
        pct_over_random = 100.0 * (union - random_alone) / random_alone
        print(f"random alone            : {random_alone} cells")
        print(f"scripted alone          : {scripted_alone} cells")
        print(f"scripted-only cells     : {scripted_only}   (states random never reaches)")
        print(
            f"union                   : {union}   "
            f"(+{pct_over_random:.1f}% over random alone)"
        )
    else:
        print("union analysis requires both 'random' and 'scripted' policies; skipping")

    # --- budget ladder: DIAGNOSTIC ONLY, not the gate ------------------
    # Occupied-cell counts pooled across many episodes scale with sample
    # size and, on a small fixed maze, saturate once the reachable area is
    # filled -- both policies eventually cover most of the grid and the
    # ladder flattens out regardless of policy quality. Kept here only as
    # a diagnostic; see the per-episode reach above for the gate.
    max_budget = min(len(v) for v in stacked.values())
    ladder = [b for b in (2_000, 5_000, 10_000, 25_000) if b <= max_budget]
    ladder.append(max_budget)
    ladder = sorted(set(ladder))

    rng = np.random.default_rng(0)
    ladder_counts: dict[int, dict[str, int]] = {}
    max_budget_sampled: dict[str, np.ndarray] = {}
    for budget in ladder:
        counts = {}
        for name in names:
            xy = stacked[name]
            sub = xy[rng.choice(len(xy), size=budget, replace=False)]
            counts[name] = _occupied_count(sub, args.bins, x_range, y_range)
            if budget == max_budget:
                max_budget_sampled[name] = sub
        ladder_counts[budget] = counts

    # Figure is drawn at the max equalised budget, as before.
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        xy = max_budget_sampled[name]
        ax.hist2d(xy[:, 0], xy[:, 1], bins=args.bins, range=[x_range, y_range])
        ax.set_title(f"{name}  (n={len(xy)}, budget={max_budget})")
        ax.set_xlabel("pos_x")
        ax.set_ylabel("pos_y")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"coverage_{args.data.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)

    print()
    print(f"figure={out_path}")
    print(f"max_equalised_budget={max_budget}")
    for name in names:
        xy = max_budget_sampled[name]
        print(
            f"{name}: total_frames={len(stacked[name])} "
            f"occupied_cells={ladder_counts[max_budget][name]}/{args.bins ** 2} "
            f"(at {max_budget} frames) "
            f"x_span={np.ptp(xy[:, 0]):.1f} y_span={np.ptp(xy[:, 1]):.1f}"
        )

    print()
    print("--- aggregate coverage budget ladder (DIAGNOSTIC ONLY -- not the gate) ---")
    print(
        "Aggregate coverage saturates once episodes have filled the reachable "
        "area, so it tracks maze size and episode count rather than policy "
        "quality; it is not used for pass/fail."
    )
    print(f"{'budget':>9}  {'random':>8}  {'scripted':>8}  {'ratio':>6}")
    for budget in ladder:
        counts = ladder_counts[budget]
        agg_ratio = counts.get("scripted", 0) / max(counts.get("random", 1), 1)
        print(
            f"{budget:>9}  {counts.get('random', 0):>8}  "
            f"{counts.get('scripted', 0):>8}  {agg_ratio:>6.3f}"
        )

    # --- gate: mean per-episode reach -----------------------------------
    print()
    if not have_both:
        raise SystemExit(
            "coverage gate requires both 'random' and 'scripted' policies in the dataset"
        )
    passed = means["scripted"] > means["random"] * args.gate_ratio
    print(
        f"GATE: {'PASS' if passed else 'FAIL'} "
        f"(mean_scripted={means['scripted']:.1f}, mean_random={means['random']:.1f}, "
        f"ratio={ratio:.3f}, required_ratio={args.gate_ratio:.2f})"
    )
    if not passed:
        raise SystemExit(
            f"coverage gate failed: mean(scripted)={means['scripted']:.1f} does not "
            f"exceed mean(random)={means['random']:.1f} * {args.gate_ratio:.2f} "
            f"(ratio={ratio:.3f} < required {args.gate_ratio:.2f})"
        )


if __name__ == "__main__":
    main()
```

**Why the gate is per-episode reach, not aggregate reach:** an earlier version of this
script gated on occupied-cell counts pooled across all episodes at a mid-range frame
budget. Measured on the existing 200-episode `my_way_home` dataset (61 episodes per
policy): aggregate occupied cells were 999 (random) vs 1018 (scripted) -- a ratio of only
~1.02, and independent ~5,800-frame trials gave ratios of 1.199, 1.260, 1.169, 1.481
(mean 1.28) with no stable value as the budget ladder grew. That instability is not noise
in the policy; it is the metric going blind. `my_way_home` is a small, fixed maze, and by
61 episodes per policy the *union* of cells random alone reaches (999) already covers all
but 26 of the 1,025 cells either policy ever reaches (+2.6%). Once episodes have nearly
saturated the reachable area, the aggregate count is dominated by maze size and episode
count, not by policy quality -- exactly backwards for a gate meant to catch a policy that
stopped exploring.

**Per-episode reach does not saturate** and directly measures what the scripted policy
exists to do: on the same dataset, scripted episodes reach a mean 125.0 cells (sd 40.1)
against random's mean 86.7 cells (sd 33.5) -- a decisive, stable +44% per episode (ratio
1.441, n=61 each), regardless of how many episodes have been collected. The gate is
`mean(scripted) > mean(random) * --gate-ratio` (default `1.2`). The aggregate budget
ladder is kept as a labelled diagnostic only -- useful for eyeballing where coverage
saturates -- and a union analysis is printed alongside it so the next plan can tell
whether collecting more episodes of these two policies would buy additional state
coverage (see the constraint recorded after the exit criteria below: on `my_way_home`,
at this episode count, it would not).

- [ ] **Step 6b: Gate on episode length against the training window**

`SequenceLoader` silently discards any episode shorter than `seq_len + 1`. Without a gate that shows up later as a mysteriously small training set, not as an error. Measure it now:

```bash
.venv/bin/python -c "
from mbfps.data.buffer import ReplayBuffer
import numpy as np
SEQ_LEN = 64
buf = ReplayBuffer('data/my_way_home', capacity_transitions=10**9)
lengths = np.array([ep.length for ep in buf.load_all()])
usable = int((lengths >= SEQ_LEN).sum())
frac = usable / len(lengths)
print(f'episodes={len(lengths)} mean_len={lengths.mean():.1f} min={lengths.min()} max={lengths.max()}')
print(f'usable_at_seq_len_{SEQ_LEN}={usable}/{len(lengths)} ({frac:.0%})')
print('GATE:', 'PASS' if frac >= 0.8 else 'FAIL')
"
```

**The gate: at least 80% of episodes must be at least `seq_len` (64) transitions long.**

If it fails, the dataset cannot supply the training window and Plan 2 will not be able to train. Do not proceed. Diagnose in this order:
1. Confirm the scenario is `my_way_home` — measured mean episode length 489 (random) / 525 (scripted), 12/12 above threshold.
2. Check whether `doom_skill` is raising difficulty and shortening episodes; pass a lower `--doom-skill` if so.
3. Only then consider whether `frame_skip` is too large (it divides episode length directly).

Record the measured `mean_len` in the commit message — Plan 2 needs it to size training.

- [ ] **Step 7: Run the coverage report and confirm the scripted policy reaches further**

Run: `.venv/bin/python scripts/coverage_report.py`

Expected: prints a `--- per-episode reach (the gate) ---` block with `n_episodes`,
`mean_cells_per_episode`, and `sd` per policy plus the ratio; a
`--- union analysis (does more of the same data buy coverage?) ---` block reporting cells
reached by random alone, by scripted alone, scripted-only cells (states random never
reaches), and the union with its percentage gain over random alone; the figure path and
per-policy `total_frames`/`occupied_cells`/`x_span`/`y_span` diagnostic line at the max
equalised budget; a `--- aggregate coverage budget ladder (DIAGNOSTIC ONLY -- not the
gate) ---` table --

```
   budget    random  scripted   ratio
     2000       ...       ...    ...
     5000       ...       ...    ...
    10000       ...       ...    ...
    25000       ...       ...    ...
    <max>       ...       ...    ...
```

-- followed by a `GATE: PASS` or `GATE: FAIL` line naming the per-episode means, the ratio,
and the required ratio. **The gate: `mean(scripted occupied cells per episode) >
mean(random occupied cells per episode) * --gate-ratio` (default `1.2`).** The script exits
non-zero on `GATE: FAIL` so the gate cannot be passed over silently.

Measured on the existing 200-episode dataset (61 episodes per policy): random reaches a
mean 86.7 cells/episode (sd 33.5), scripted reaches a mean 125.0 cells/episode (sd 40.1) --
ratio 1.441, comfortably above the default `1.2` threshold and stable regardless of how
many episodes are collected, unlike the old aggregate-budget gate (see Step 6's rationale
above and the constraint recorded after the exit criteria below).

If `GATE: FAIL`, first confirm the policy is not degenerate — run `.venv/bin/python -c "from mbfps.envs.registry import make_env; e=make_env('vizdoom'); print(e.button_names); e.close()"` and check that `MOVE_FORWARD`, `TURN_LEFT` and `TURN_RIGHT` are present. If they are, tune `_ADVANCE_PROB` or `_SWEEP_LEN` in `src/mbfps/data/policies.py`, re-collect, and re-run. Do not proceed to Plan 2 with a scripted policy that adds no coverage — it is dead weight in the dataset and the M1 rationale no longer holds. **Do not, however, re-tune merely to chase a passing number on a dataset that already exists** — the policy constants are tuned against the environment's exploration dynamics, not against one dataset's gate result; if a gate failure on an existing collection does not resolve after confirming the policy is non-degenerate, treat it as a real finding to report rather than a bug to tune away.

- [ ] **Step 8: Benchmark the loader**

```bash
.venv/bin/python -c "
import time
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
buf = ReplayBuffer('data/my_way_home', capacity_transitions=10**9)
loader = SequenceLoader(buf, batch_size=16, seq_len=64, seed=0)
loader.sample()  # warm up
start = time.perf_counter()
for _ in range(20):
    loader.sample()
print(f'batches_per_second={20 / (time.perf_counter() - start):.2f}')
"
```

Expected: prints a positive `batches_per_second`. Record it in the commit message — Plan 2 needs it to size world-model training steps.

- [ ] **Step 9: Run the full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass (excluding none; `slow` tests run too).

- [ ] **Step 10: Commit**

```bash
git add scripts/collect.py scripts/coverage_report.py tests/test_privileged_isolation.py pyproject.toml
git commit -m "feat: mixed-policy collection CLI, coverage report, privileged isolation gate"
```

---

## Exit criteria for Plan 1

Both milestones' spec criteria, restated as things you can run:

**M0:**
- [ ] `pytest tests/envs/test_determinism.py` passes — same seed gives bit-identical frames and rewards, different seeds diverge, and a recorded episode replays from its saved action sequence in a fresh engine.
- [ ] `pytest tests/envs/test_vizdoom_env.py` passes — including that a timeout sets `truncated` (not `terminated`), a goal sets `terminated` (not `truncated`), and `privileged_state` is `None` after the episode ends.
- [ ] `scripts/benchmark_env.py` reports a recorded `steps_per_second`.

**M1:**
- [ ] `scripts/collect.py` produces a dataset with a reported `transitions_per_second` **and `episodes_kept` equal to the requested episode count** (zero kept means the collector is discarding terminating episodes).
- [ ] The loader benchmark reports `batches_per_second`.
- [ ] `pytest tests/data/test_episode.py` passes — stored data equals collected data.
- [ ] `pytest tests/data/test_features.py` passes — the feature cache is byte-identical on repeat **within a process on a given device** (CPU and MPS outputs for the same frame differ in roughly 3% of float16 elements, so the cache must be generated once on one device and reused across every experimental arm, never regenerated per arm).
- [ ] The episode-length gate passes: at least 80% of episodes are >= 64 transitions, with `mean_len` recorded.
- [ ] `scripts/coverage_report.py` prints `GATE: PASS` — the mean occupied-cell count reached **per scripted episode** exceeds the mean **per random episode** by at least `--gate-ratio` (default `1.2`), i.e. `mean(scripted) > mean(random) * 1.2` (measured: 125.0 vs 86.7 cells/episode, ratio 1.441). This is a **per-episode**, not aggregate, gate: aggregate cell counts pooled across many episodes saturate once a fixed maze's reachable area is filled and then track maze size and episode count rather than policy quality (measured on this dataset: aggregate cells 999 (random) vs 1018 (scripted), ratio ~1.02, union only +2.6% over random alone) — see the constraint below.
- [ ] `pytest tests/test_privileged_isolation.py` passes.

## Constraint carried into Plan 2: `my_way_home` coverage is near-saturated at 61 episodes/policy

Measured while building the M1 coverage gate (see Task 13, Step 6): on the existing
200-episode `my_way_home` dataset (61 episodes per policy), random alone already reaches
999 of the 1,025 cells either policy ever reaches, and scripted adds only 26 cells (+2.6%)
that random never visits. The two policies' *per-episode* reach differs sharply and
reliably (scripted 125.0 cells/episode vs random's 86.7, +44%, the M1 gate) but their
*aggregate* footprint over many episodes has nearly filled the small, fixed maze.

**Implication for Plan 2 and beyond: collecting more episodes of `RandomPolicy` and
`ScriptedPolicy` on `my_way_home` will not meaningfully increase state coverage.** The
ceiling here is the maze's reachable area, not the sample size. If a later milestone needs
broader state coverage than this dataset provides (e.g. the world model hallucinating in
regions neither policy visits), the fix is additional scenarios or map variety — not more
episodes of these two policies on this one map. Re-run `scripts/coverage_report.py --data
data/<scenario>` (union analysis) before collecting a larger dataset on the same scenario,
to check whether that scenario has already saturated the same way.

## Deferred to Plan 2 (M2–M3)

Encoders (all three arms), the RSSM, decoders, reward/continue heads, the world-model trainer, open-loop rollout evaluation, and the latent-to-privileged linear probe.

## Deferred to Plan 3 (M4–M5)

Actor, critic, imagination rollouts, the imagination-isolation test, the online Dreamer loop, and the model-free (PPO/DQN) sample-efficiency baseline.
