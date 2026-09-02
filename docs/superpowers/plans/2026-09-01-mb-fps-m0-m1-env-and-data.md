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
- **No network access at training time.** The only download is the DINOv2 checkpoint, fetched once in Task 14 and cached locally.
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
- Create: `src/mbfps/__init__.py`
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


def test_get_device_sets_mps_fallback_env_var():
    get_device()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


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
"""Model-Based First-Person Shooter Agent."""
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
"""

import os

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(prefer: str = "mps") -> torch.device:
    """Return the best available device, honouring `prefer`.

    Args:
        prefer: "mps", "cuda", or "cpu". Falls back to CPU when unavailable.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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

Expected: 6 passed.

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/test_protocol.py -v`
Expected: 5 passed.

Note: `test_make_env_lists_available_names_in_error` passes only once `vizdoom_env.py` exists (Task 3). Until then `_register_builtins` swallows the `ImportError` and the registry is empty. Mark that one test `@pytest.mark.xfail(reason="ViZDoomEnv lands in Task 3")` now and remove the marker in Task 3.

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


def test_output_values_stay_in_uint8_range():
    frame = np.full((120, 160, 3), 255, dtype=np.uint8)
    out = preprocess_frame(frame)
    assert out.min() >= 0 and out.max() <= 255


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

_TARGET_HW = (OBS_SHAPE[1], OBS_SHAPE[0])  # cv2.resize takes (width, height)


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
    resized = cv2.resize(frame, _TARGET_HW, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/test_wrappers.py -v`
Expected: 7 passed.

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
- Produces: `ViZDoomEnv(scenario: str = "basic", frame_skip: int = 4, seed: int = 0)` implementing `EnvProtocol`. Exposes `PRIVILEGED_KEYS: tuple[str, ...] = ("health", "pos_x", "pos_y", "pos_z", "angle")`.

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
    e = ViZDoomEnv(scenario="basic", frame_skip=4, seed=0)
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


def test_episode_eventually_terminates(env):
    env.reset(seed=0)
    for _ in range(2000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            return
    pytest.fail("episode did not terminate within 2000 steps")


def test_observation_after_termination_is_still_valid(env):
    env.reset(seed=0)
    for _ in range(2000):
        obs, _, terminated, truncated, _ = env.step(0)
        assert obs.shape == OBS_SHAPE, "terminal observation must stay well-formed"
        if terminated or truncated:
            break


def test_privileged_state_has_expected_keys(env):
    env.reset(seed=0)
    state = env.privileged_state
    assert set(state) == set(PRIVILEGED_KEYS)
    assert all(isinstance(v, float) for v in state.values())


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
from mbfps.envs.registry import register
from mbfps.envs.wrappers import preprocess_frame

PRIVILEGED_KEYS: tuple[str, ...] = ("health", "pos_x", "pos_y", "pos_z", "angle")
"""Keys of `privileged_state`. EVALUATION ONLY -- never a training input."""

_PRIVILEGED_VARS = (
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.POSITION_Z,
    vzd.GameVariable.ANGLE,
)


class ViZDoomEnv:
    """A headless, seedable ViZDoom environment."""

    def __init__(
        self, scenario: str = "basic", frame_skip: int = 4, seed: int = 0
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
        for var in _PRIVILEGED_VARS:
            self._game.add_available_game_variable(var)
        self._game.set_seed(seed)
        self._game.init()

        self._actions = build_action_set(len(self._game.get_available_buttons()))
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._actions))
        self._last_obs = np.zeros(OBS_SHAPE, dtype=np.uint8)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode, optionally reseeding first."""
        if seed is not None:
            self._seed = seed
            self._game.set_seed(seed)
        self._game.new_episode()
        self._last_obs = self._observe()
        return self._last_obs, {"scenario": self.scenario}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance `frame_skip` tics with the button vector for `action`."""
        reward = float(self._game.make_action(self._actions[action], self.frame_skip))
        terminated = bool(self._game.is_episode_finished())
        if not terminated:
            self._last_obs = self._observe()
        return self._last_obs, reward, terminated, False, {}

    def close(self) -> None:
        """Release the ViZDoom instance. Safe to call more than once."""
        if not self._closed:
            self._game.close()
            self._closed = True

    @property
    def privileged_state(self) -> dict[str, float] | None:
        """Ground-truth engine state. EVALUATION PROBES ONLY."""
        state = self._game.get_state()
        if state is None:
            return None
        values = state.game_variables[-len(_PRIVILEGED_VARS) :]
        return {k: float(v) for k, v in zip(PRIVILEGED_KEYS, values)}

    def _observe(self) -> np.ndarray:
        state = self._game.get_state()
        if state is None:
            return self._last_obs
        return preprocess_frame(state.screen_buffer)


register("vizdoom", ViZDoomEnv)
```

- [ ] **Step 4: Remove the xfail marker from Task 2's test**

In `tests/envs/test_protocol.py`, delete the `@pytest.mark.xfail(...)` decorator above `test_make_env_lists_available_names_in_error`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/envs/ -v`
Expected: all passed, including the previously-xfailed registry test.

If `test_privileged_state_has_expected_keys` fails because `basic.cfg` already declares game variables, the slice `game_variables[-5:]` still selects ours because `add_available_game_variable` appends. If it fails for another reason, print `self._game.get_available_game_variables()` and align the slice with the actual ordering before proceeding.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/envs/vizdoom_env.py tests/envs/
git commit -m "feat: headless seedable ViZDoom environment"
```

---

## Task 6: Determinism, action replay, and throughput

**Files:**
- Create: `tests/envs/test_determinism.py`
- Create: `scripts/benchmark_env.py`

**Interfaces:**
- Consumes: `ViZDoomEnv`, `make_env`.
- Produces: `scripts/benchmark_env.py` printing `steps_per_second=<float>`. No new library API.

This task is the M0 exit gate: **same seed produces a bit-identical episode, and a recorded episode replays from its action sequence alone.**

- [ ] **Step 1: Write the failing test**

```python
# tests/envs/test_determinism.py
import numpy as np
import pytest

from mbfps.envs.vizdoom_env import ViZDoomEnv


def _rollout(seed: int, actions: list[int]) -> tuple[list[np.ndarray], list[float]]:
    env = ViZDoomEnv(scenario="basic", frame_skip=4, seed=seed)
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
    rng = np.random.default_rng(0)
    return [int(a) for a in rng.integers(0, 4, size=40)]


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


def test_episode_replays_from_action_sequence_alone(actions):
    """The M0 exit criterion: actions + seed fully determine the episode."""
    original_frames, original_rewards = _rollout(seed=11, actions=actions)
    replay_frames, replay_rewards = _rollout(seed=11, actions=actions)
    assert all(
        np.array_equal(a, b) for a, b in zip(original_frames, replay_frames)
    )
    assert original_rewards == replay_rewards


def test_reset_reseeds_within_one_instance(actions):
    env = ViZDoomEnv(scenario="basic", frame_skip=4, seed=0)
    try:
        first, _ = env.reset(seed=21)
        env.step(1)
        second, _ = env.reset(seed=21)
        assert np.array_equal(first, second)
    finally:
        env.close()
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/envs/test_determinism.py -v`

Expected: PASS if `set_seed` before `new_episode` reseeds correctly.

**If `test_reset_reseeds_within_one_instance` fails**, ViZDoom is not honouring a post-`init` reseed. Apply this fallback in `src/mbfps/envs/vizdoom_env.py` — replace the seeding block in `reset` with a full engine restart:

```python
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode, optionally reseeding first.

        ViZDoom only honours `set_seed` before `init`, so a reseed requires
        closing and reinitialising the engine. This costs ~100ms and happens
        once per reseed, not once per episode.
        """
        if seed is not None and seed != self._seed:
            self._seed = seed
            self._game.close()
            self._game.set_seed(seed)
            self._game.init()
        self._game.new_episode()
        self._last_obs = self._observe()
        return self._last_obs, {"scenario": self.scenario}
```

Re-run until all four tests pass. Do not proceed to Task 7 with a failing determinism test — every downstream arm comparison depends on it.

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
    parser.add_argument("--scenario", default="basic")
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
Expected: prints `steps_per_second=<N>`. Record N in the commit message — it is the M0 baseline and sizes every collection run in M1.

- [ ] **Step 5: Commit**

```bash
git add tests/envs/test_determinism.py scripts/benchmark_env.py
git commit -m "test: determinism and action-replay gate; add env throughput benchmark"
```

---

## Task 7: Collection policies

**Files:**
- Create: `src/mbfps/data/__init__.py`
- Create: `src/mbfps/data/policies.py`
- Create: `tests/data/test_policies.py`
- Modify: `src/mbfps/envs/vizdoom_env.py` (add a `button_names` property)

**Interfaces:**
- Consumes: `ViZDoomEnv`.
- Produces: `Policy` Protocol with `name: str` and `act(obs: np.ndarray) -> int` and `reset() -> None`; `RandomPolicy(n_actions: int, seed: int)`; `ScriptedPolicy(button_names: Sequence[str], seed: int)`. `ViZDoomEnv.button_names -> tuple[str, ...]`.

Rationale (spec §5.2): a single random policy has poor state coverage, and the world model hallucinates wherever the collector never went. The M4 actor-critic then optimises straight into those hallucinations, producing high imagined return and near-zero real return. `ScriptedPolicy` exists to reach states `RandomPolicy` does not.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_policies.py
import numpy as np
import pytest

from mbfps.data.policies import Policy, RandomPolicy, ScriptedPolicy
from mbfps.envs.protocol import OBS_SHAPE

BUTTONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK")


@pytest.fixture
def obs():
    return np.zeros(OBS_SHAPE, dtype=np.uint8)


def test_random_policy_satisfies_protocol():
    assert isinstance(RandomPolicy(n_actions=5, seed=0), Policy)


def test_scripted_policy_satisfies_protocol():
    assert isinstance(ScriptedPolicy(BUTTONS, seed=0), Policy)


def test_policies_expose_distinct_names():
    assert RandomPolicy(5, seed=0).name == "random"
    assert ScriptedPolicy(BUTTONS, seed=0).name == "scripted"


def test_random_policy_stays_in_range(obs):
    policy = RandomPolicy(n_actions=5, seed=0)
    assert all(0 <= policy.act(obs) < 5 for _ in range(200))


def test_random_policy_covers_every_action(obs):
    policy = RandomPolicy(n_actions=5, seed=0)
    assert {policy.act(obs) for _ in range(500)} == {0, 1, 2, 3, 4}


def test_random_policy_is_reproducible(obs):
    p, q = RandomPolicy(5, seed=3), RandomPolicy(5, seed=3)
    assert [p.act(obs) for _ in range(50)] == [q.act(obs) for _ in range(50)]


def test_random_policy_reset_replays_the_same_sequence(obs):
    policy = RandomPolicy(5, seed=3)
    first = [policy.act(obs) for _ in range(50)]
    policy.reset()
    assert [policy.act(obs) for _ in range(50)] == first


def test_scripted_policy_returns_valid_indices(obs):
    policy = ScriptedPolicy(BUTTONS, seed=0)
    assert all(0 <= policy.act(obs) <= len(BUTTONS) for _ in range(300))


def test_scripted_policy_is_forward_biased(obs):
    """Forward must dominate, or coverage is no better than random."""
    policy = ScriptedPolicy(BUTTONS, seed=0)
    actions = [policy.act(obs) for _ in range(1000)]
    forward_index = BUTTONS.index("MOVE_FORWARD") + 1  # +1 skips the no-op
    assert actions.count(forward_index) / len(actions) > 0.4


def test_scripted_policy_sweeps_both_directions(obs):
    policy = ScriptedPolicy(BUTTONS, seed=0)
    actions = {policy.act(obs) for _ in range(1000)}
    assert BUTTONS.index("TURN_LEFT") + 1 in actions
    assert BUTTONS.index("TURN_RIGHT") + 1 in actions


def test_scripted_policy_is_reproducible(obs):
    p, q = ScriptedPolicy(BUTTONS, seed=5), ScriptedPolicy(BUTTONS, seed=5)
    assert [p.act(obs) for _ in range(100)] == [q.act(obs) for _ in range(100)]


def test_scripted_policy_reset_restarts_the_sweep(obs):
    policy = ScriptedPolicy(BUTTONS, seed=5)
    first = [policy.act(obs) for _ in range(30)]
    policy.reset()
    assert [policy.act(obs) for _ in range(30)] == first


def test_scripted_policy_without_movement_buttons_falls_back(obs):
    policy = ScriptedPolicy(("ATTACK",), seed=0)
    assert all(0 <= policy.act(obs) <= 1 for _ in range(50))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_policies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbfps.data'`

- [ ] **Step 3: Write the implementation**

```python
# src/mbfps/data/__init__.py
```

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

    def reset(self) -> None:
        """Reset any per-episode internal state."""
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

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)


class ScriptedPolicy:
    """Forward-biased movement with a periodic left/right aim sweep.

    Random play in a corridor scenario tends to oscillate near the spawn point.
    This policy commits to forward motion, sweeping the view so that the world
    model sees walls, corners, and distant geometry that random play rarely
    reaches.
    """

    _FORWARD_PROB = 0.6
    _SWEEP_LEN = 8

    def __init__(self, button_names: Sequence[str], seed: int = 0) -> None:
        self.name = "scripted"
        self._seed = seed
        self._n = len(button_names) + 1  # +1 for the no-op
        self._forward = self._index_of(button_names, "MOVE_FORWARD")
        self._left = self._index_of(button_names, "TURN_LEFT")
        self._right = self._index_of(button_names, "TURN_RIGHT")
        self._attack = self._index_of(button_names, "ATTACK")
        self.reset()

    @staticmethod
    def _index_of(names: Sequence[str], target: str) -> int | None:
        """Return the action index for `target`, or None if absent."""
        for i, name in enumerate(names):
            if name == target:
                return i + 1
        return None

    def act(self, obs: np.ndarray) -> int:
        self._t += 1
        sweeping_left = (self._t // self._SWEEP_LEN) % 2 == 0
        turn = self._left if sweeping_left else self._right

        if self._forward is not None and self._rng.random() < self._FORWARD_PROB:
            return self._forward
        if turn is not None and self._rng.random() < 0.75:
            return turn
        if self._attack is not None:
            return self._attack
        return int(self._rng.integers(0, self._n))

    def reset(self) -> None:
        self._t = 0
        self._rng = np.random.default_rng(self._seed)
```

- [ ] **Step 4: Add `button_names` to the environment**

In `src/mbfps/envs/vizdoom_env.py`, add this property after `privileged_state`:

```python
    @property
    def button_names(self) -> tuple[str, ...]:
        """Names of the scenario's available buttons, in button-vector order."""
        return tuple(b.name for b in self._game.get_available_buttons())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_policies.py -v`
Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add src/mbfps/data tests/data src/mbfps/envs/vizdoom_env.py
git commit -m "feat: random and scripted collection policies"
```

---

## Task 8: Episode storage

**Files:**
- Create: `src/mbfps/data/episode.py`
- Create: `tests/data/test_episode.py`

**Interfaces:**
- Consumes: nothing. (`privileged_keys` travels with each episode as data, so this
  module has no dependency on any particular engine's key set.)
- Produces: `Episode` dataclass with fields `obs (T+1, 112, 112, 3) uint8`, `actions (T,) int32`, `rewards (T,) float32`, `terminated (T,) bool`, `privileged (T+1, K) float32`, `privileged_keys tuple[str, ...]`, `policy_name str`, `seed int`, `scenario str`; property `length -> int` (= T). Functions `save_episode(ep: Episode, path: Path) -> None` and `load_episode(path: Path) -> Episode`.

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
        privileged=rng.standard_normal((t + 1, len(KEYS))).astype(np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=42,
        scenario="basic",
    )


def test_length_is_number_of_transitions():
    assert make_episode(t=5).length == 5


def test_obs_has_one_more_entry_than_actions():
    ep = make_episode(t=5)
    assert ep.obs.shape[0] == ep.actions.shape[0] + 1


def test_round_trip_preserves_arrays(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert np.array_equal(loaded.obs, ep.obs)
    assert np.array_equal(loaded.actions, ep.actions)
    assert np.array_equal(loaded.rewards, ep.rewards)
    assert np.array_equal(loaded.terminated, ep.terminated)
    assert np.array_equal(loaded.privileged, ep.privileged)


def test_round_trip_preserves_metadata(tmp_path):
    ep = make_episode()
    path = tmp_path / "ep.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    assert loaded.policy_name == "random"
    assert loaded.seed == 42
    assert loaded.scenario == "basic"
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


def test_mismatched_lengths_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="obs must have one more"):
        Episode(
            obs=rng.integers(0, 256, (5, *OBS_SHAPE), dtype=np.uint8),
            actions=rng.integers(0, 4, 5).astype(np.int32),
            rewards=rng.standard_normal(5).astype(np.float32),
            terminated=np.zeros(5, dtype=bool),
            privileged=rng.standard_normal((5, len(KEYS))).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="basic",
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


@dataclass
class Episode:
    """A single recorded episode."""

    obs: np.ndarray  # (T+1, 112, 112, 3) uint8
    actions: np.ndarray  # (T,) int32
    rewards: np.ndarray  # (T,) float32
    terminated: np.ndarray  # (T,) bool
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
        privileged=ep.privileged,
        privileged_keys=np.array(ep.privileged_keys, dtype=object),
        policy_name=ep.policy_name,
        seed=ep.seed,
        scenario=ep.scenario,
    )


def load_episode(path: Path) -> Episode:
    """Read an episode written by `save_episode`."""
    with np.load(path, allow_pickle=True) as data:
        return Episode(
            obs=data["obs"],
            actions=data["actions"],
            rewards=data["rewards"],
            terminated=data["terminated"],
            privileged=data["privileged"],
            privileged_keys=tuple(data["privileged_keys"].tolist()),
            policy_name=str(data["policy_name"]),
            seed=int(data["seed"]),
            scenario=str(data["scenario"]),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_episode.py -v`
Expected: 6 passed.

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
- Consumes: `EnvProtocol`, `Policy`, `Episode`, `PRIVILEGED_KEYS`.
- Produces: `Collector(env_factory: Callable[[], EnvProtocol], policy: Policy, max_steps: int = 1000)` with `collect_episode(seed: int) -> Episode | None` (None on engine crash, after restarting the env), `close() -> None`, and attribute `crash_count: int`.

ViZDoom can segfault or raise mid-episode. A crash must discard the partial episode and restart the engine rather than poisoning the dataset with a truncated trajectory that looks terminal but is not.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_collector.py
import numpy as np
import pytest
from gymnasium import spaces

from mbfps.data.collector import Collector
from mbfps.data.policies import RandomPolicy
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


class _StubEnv:
    """Deterministic stand-in that can be told to crash at a given step."""

    instances = 0

    def __init__(self, episode_len=6, crash_at=None):
        type(self).instances += 1
        self.observation_space = spaces.Box(0, 255, OBS_SHAPE, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.button_names = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
        self._episode_len = episode_len
        self._crash_at = crash_at
        self._t = 0

    def reset(self, *, seed=None):
        self._t = 0
        return np.full(OBS_SHAPE, 1, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        if self._crash_at is not None and self._t == self._crash_at:
            raise RuntimeError("simulated engine crash")
        obs = np.full(OBS_SHAPE, self._t % 256, dtype=np.uint8)
        return obs, 1.0, self._t >= self._episode_len, False, {}

    def close(self):
        pass

    @property
    def privileged_state(self):
        return {k: float(self._t) for k in KEYS}


def test_collect_episode_returns_episode():
    c = Collector(lambda: _StubEnv(episode_len=6), RandomPolicy(4, seed=0))
    ep = c.collect_episode(seed=0)
    assert ep is not None and ep.length == 6
    c.close()


def test_obs_has_one_more_frame_than_actions():
    c = Collector(lambda: _StubEnv(episode_len=6), RandomPolicy(4, seed=0))
    ep = c.collect_episode(seed=0)
    assert ep.obs.shape[0] == ep.actions.shape[0] + 1
    c.close()


def test_episode_records_policy_name_and_seed():
    c = Collector(lambda: _StubEnv(episode_len=4), RandomPolicy(4, seed=0))
    ep = c.collect_episode(seed=77)
    assert ep.policy_name == "random" and ep.seed == 77
    c.close()


def test_privileged_has_one_row_per_frame():
    c = Collector(lambda: _StubEnv(episode_len=6), RandomPolicy(4, seed=0))
    ep = c.collect_episode(seed=0)
    assert ep.privileged.shape == (ep.length + 1, len(KEYS))
    assert ep.privileged_keys == KEYS
    c.close()


def test_max_steps_truncates():
    c = Collector(lambda: _StubEnv(episode_len=999), RandomPolicy(4, seed=0), max_steps=5)
    ep = c.collect_episode(seed=0)
    assert ep.length == 5
    c.close()


def test_crash_returns_none_and_is_counted():
    c = Collector(lambda: _StubEnv(episode_len=20, crash_at=3), RandomPolicy(4, seed=0))
    assert c.collect_episode(seed=0) is None
    assert c.crash_count == 1
    c.close()


def test_crash_rebuilds_the_environment():
    _StubEnv.instances = 0
    c = Collector(lambda: _StubEnv(episode_len=20, crash_at=3), RandomPolicy(4, seed=0))
    c.collect_episode(seed=0)
    assert _StubEnv.instances >= 2, "collector must rebuild the env after a crash"
    c.close()


def test_collector_recovers_and_keeps_collecting():
    envs = iter([_StubEnv(20, crash_at=3), _StubEnv(6), _StubEnv(6)])
    c = Collector(lambda: next(envs), RandomPolicy(4, seed=0))
    assert c.collect_episode(seed=0) is None
    assert c.collect_episode(seed=1) is not None
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
        except Exception:
            self.crash_count += 1
            logger.warning("engine crashed during collection; restarting", exc_info=True)
            self._restart()
            return None

    def _collect(self, seed: int) -> Episode:
        self._policy.reset()
        obs, _ = self._env.reset(seed=seed)

        frames = [obs.copy()]
        privileged = [self._privileged_row()]
        actions: list[int] = []
        rewards: list[float] = []
        terminated_flags: list[bool] = []

        for _ in range(self._max_steps):
            action = self._policy.act(obs)
            obs, reward, terminated, truncated, _ = self._env.step(action)
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            frames.append(obs.copy())
            privileged.append(self._privileged_row())
            if terminated or truncated:
                break

        return Episode(
            obs=np.stack(frames).astype(np.uint8),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated_flags, dtype=bool),
            privileged=np.stack(privileged).astype(np.float32),
            privileged_keys=self._privileged_keys(),
            policy_name=self._policy.name,
            seed=seed,
            scenario=getattr(self._env, "scenario", "unknown"),
        )

    def _privileged_keys(self) -> tuple[str, ...]:
        state = self._env.privileged_state
        return tuple(state) if state else ()

    def _privileged_row(self) -> np.ndarray:
        state = self._env.privileged_state
        if not state:
            return np.zeros(len(self._privileged_keys()), dtype=np.float32)
        return np.asarray(list(state.values()), dtype=np.float32)

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
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data/collector.py tests/data/test_collector.py
git commit -m "feat: episode collector with engine crash recovery"
```

---

## Task 10: Replay buffer with eviction

**Files:**
- Create: `src/mbfps/data/buffer.py`
- Create: `tests/data/test_buffer.py`

**Interfaces:**
- Consumes: `Episode`, `save_episode`, `load_episode`.
- Produces: `ReplayBuffer(root: Path, capacity_transitions: int)` with `add(ep: Episode) -> Path`, `episode_paths() -> list[Path]`, `load_all() -> list[Episode]`, `n_episodes -> int`, `n_transitions -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_buffer.py
import numpy as np

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
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name=policy,
        seed=seed,
        scenario="basic",
    )


def test_add_writes_a_file(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    path = buf.add(make_episode(10))
    assert path.is_file()


def test_counts_track_contents(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=100)
    buf.add(make_episode(10))
    buf.add(make_episode(15))
    assert buf.n_episodes == 2
    assert buf.n_transitions == 25


def test_eviction_respects_capacity(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    for _ in range(5):
        buf.add(make_episode(10))
    assert buf.n_transitions <= 25


def test_eviction_removes_oldest_first(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=25)
    buf.add(make_episode(10, seed=1))
    buf.add(make_episode(10, seed=2))
    buf.add(make_episode(10, seed=3))
    seeds = {ep.seed for ep in buf.load_all()}
    assert 1 not in seeds, "oldest episode should have been evicted"
    assert {2, 3} <= seeds


def test_buffer_reopens_existing_directory(tmp_path):
    ReplayBuffer(tmp_path, capacity_transitions=100).add(make_episode(10))
    assert ReplayBuffer(tmp_path, capacity_transitions=100).n_episodes == 1


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
"""

import itertools
from pathlib import Path

from mbfps.data.episode import Episode, load_episode, save_episode


class ReplayBuffer:
    """Episodes on disk, oldest evicted once capacity is exceeded."""

    def __init__(self, root: Path, capacity_transitions: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_transitions = capacity_transitions
        self._counter = itertools.count(self._next_index())

    def _next_index(self) -> int:
        indices = [int(p.stem.split("_")[1]) for p in self.episode_paths()]
        return max(indices) + 1 if indices else 0

    def episode_paths(self) -> list[Path]:
        """Episode files, oldest first."""
        return sorted(self.root.glob("ep_*.npz"))

    @property
    def n_episodes(self) -> int:
        return len(self.episode_paths())

    @property
    def n_transitions(self) -> int:
        return sum(ep.length for ep in self.load_all())

    def add(self, ep: Episode) -> Path:
        """Write `ep` and evict oldest episodes until within capacity."""
        path = self.root / f"ep_{next(self._counter):06d}.npz"
        save_episode(ep, path)
        self._evict()
        return path

    def load_all(self) -> list[Episode]:
        """Load every stored episode, oldest first."""
        return [load_episode(p) for p in self.episode_paths()]

    def _evict(self) -> None:
        paths = self.episode_paths()
        lengths = [load_episode(p).length for p in paths]
        total = sum(lengths)
        # Never evict the newest episode: an empty buffer is worse than an
        # oversized one.
        for path, length in zip(paths[:-1], lengths[:-1]):
            if total <= self.capacity_transitions:
                break
            path.unlink()
            total -= length
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_buffer.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mbfps/data/buffer.py tests/data/test_buffer.py
git commit -m "feat: fixed-capacity replay buffer with oldest-first eviction"
```

---

## Task 11: Sequence loader

**Files:**
- Create: `src/mbfps/data/loader.py`
- Create: `tests/data/test_loader.py`

**Interfaces:**
- Consumes: `ReplayBuffer`, `Episode`.
- Produces: `SequenceLoader(buffer: ReplayBuffer, batch_size: int = 16, seq_len: int = 64, seed: int = 0)` with `sample(include_privileged: bool = False) -> dict[str, np.ndarray]`. Keys: `obs (B, T+1, 112, 112, 3) uint8`, `actions (B, T) int32`, `rewards (B, T) float32`, `terminated (B, T) bool`, `episode_index (B,) int32`. Adds `privileged (B, T+1, K) float32` **only** when `include_privileged=True`.

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
        privileged=np.full((t + 1, len(KEYS)), float(fill), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="basic",
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
    assert batch["episode_index"].shape == (4,)


def test_batch_dtypes(buffer):
    batch = SequenceLoader(buffer, batch_size=4, seq_len=16, seed=0).sample()
    assert batch["obs"].dtype == np.uint8
    assert batch["actions"].dtype == np.int32
    assert batch["rewards"].dtype == np.float32
    assert batch["terminated"].dtype == bool


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

        obs, actions, rewards, terminated, privileged, indices = [], [], [], [], [], []
        for _ in range(self.batch_size):
            idx = int(self._rng.choice(usable))
            ep = self._episodes[idx]
            start = int(self._rng.integers(0, ep.length - self.seq_len + 1))
            end = start + self.seq_len
            obs.append(ep.obs[start : end + 1])
            actions.append(ep.actions[start:end])
            rewards.append(ep.rewards[start:end])
            terminated.append(ep.terminated[start:end])
            indices.append(idx)
            if include_privileged:
                privileged.append(ep.privileged[start : end + 1])

        batch = {
            "obs": np.stack(obs).astype(np.uint8),
            "actions": np.stack(actions).astype(np.int32),
            "rewards": np.stack(rewards).astype(np.float32),
            "terminated": np.stack(terminated).astype(bool),
            "episode_index": np.asarray(indices, dtype=np.int32),
        }
        if include_privileged:
            batch["privileged"] = np.stack(privileged).astype(np.float32)
        return batch
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_loader.py -v`
Expected: 8 passed.

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

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_features.py
import numpy as np
import pytest

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


def test_cache_episode_features_writes_sibling_file(tmp_path, extractor):
    from mbfps.data.episode import Episode, save_episode

    keys = ("health", "pos_x", "pos_y", "pos_z", "angle")
    ep = Episode(
        obs=np.zeros((4, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(3, dtype=np.int32),
        rewards=np.zeros(3, dtype=np.float32),
        terminated=np.zeros(3, dtype=bool),
        privileged=np.zeros((4, len(keys)), dtype=np.float32),
        privileged_keys=keys,
        policy_name="random",
        seed=0,
        scenario="basic",
    )
    path = tmp_path / "ep_000000.npz"
    save_episode(ep, path)
    out = cache_episode_features(path, extractor)
    assert out.is_file()
    assert np.load(out).shape == (4, N_PATCHES, FEATURE_DIM)
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


def cache_episode_features(ep_path: Path, extractor: FeatureExtractor) -> Path:
    """Encode an episode's frames and write a sibling `.features.npy`.

    Returns:
        Path to the written feature file.
    """
    ep_path = Path(ep_path)
    with np.load(ep_path, allow_pickle=True) as data:
        obs = data["obs"]
    features = extractor.encode(obs)
    out_path = ep_path.with_suffix(".features.npy")
    np.save(out_path, features)
    return out_path
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_features.py -v`
Expected: 7 passed. First run downloads ~88MB.

If `test_encode_output_shape` fails with a patch-count mismatch, the installed `transformers` is not interpolating position embeddings for a 112x112 input. Pass `interpolate_pos_encoding=True` to the model call:

```python
        out = self.model(
            pixel_values=tensor, interpolate_pos_encoding=True
        ).last_hidden_state
```

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
            privileged=np.full((81, len(KEYS)), SENTINEL, dtype=np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=0,
            scenario="basic",
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

    env = ViZDoomEnv(scenario="basic", frame_skip=4, seed=0)
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
from mbfps.data.policies import RandomPolicy, ScriptedPolicy
from mbfps.envs.registry import make_env
from mbfps.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="basic")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=200_000)
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

    start, kept = time.perf_counter(), 0
    for i in range(args.episodes):
        name = "random" if i % 2 == 0 else "scripted"
        episode = collectors[name].collect_episode(seed=args.seed + i)
        if episode is not None:
            buffer.add(episode)
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

Run: `.venv/bin/python scripts/collect.py --scenario basic --episodes 20`
Expected: prints progress, then `episodes_kept=20` (or fewer with crashes reported), a positive `transitions_per_second`, and `output=data/basic`.

- [ ] **Step 6: Write the coverage report**

```python
# scripts/coverage_report.py
"""Per-policy state-visitation histogram.

Coverage gaps are invisible at the milestone that creates them and only become
symptomatic three milestones later, as an M4 agent with high imagined return and
near-zero real return. This makes them visible now.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbfps.data.buffer import ReplayBuffer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/basic"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--bins", type=int, default=60)
    args = parser.parse_args()

    episodes = ReplayBuffer(args.data, capacity_transitions=10**9).load_all()
    if not episodes:
        raise SystemExit(f"no episodes found in {args.data}")

    keys = episodes[0].privileged_keys
    x_i, y_i = keys.index("pos_x"), keys.index("pos_y")

    positions: dict[str, list[np.ndarray]] = defaultdict(list)
    for ep in episodes:
        positions[ep.policy_name].append(ep.privileged[:, [x_i, y_i]])

    names = sorted(positions)
    stacked = {n: np.concatenate(positions[n]) for n in names}
    all_xy = np.concatenate(list(stacked.values()))
    x_range = (all_xy[:, 0].min(), all_xy[:, 0].max())
    y_range = (all_xy[:, 1].min(), all_xy[:, 1].max())

    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        xy = stacked[name]
        ax.hist2d(xy[:, 0], xy[:, 1], bins=args.bins, range=[x_range, y_range])
        ax.set_title(f"{name}  (n={len(xy)})")
        ax.set_xlabel("pos_x")
        ax.set_ylabel("pos_y")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"coverage_{args.data.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)

    print(f"figure={out_path}")
    for name in names:
        xy = stacked[name]
        occupied = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=args.bins, range=[x_range, y_range]
        )[0]
        print(
            f"{name}: frames={len(xy)} "
            f"occupied_cells={int((occupied > 0).sum())}/{args.bins ** 2} "
            f"x_span={np.ptp(xy[:, 0]):.1f} y_span={np.ptp(xy[:, 1]):.1f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the coverage report and confirm the scripted policy reaches further**

Run: `.venv/bin/python scripts/coverage_report.py --data data/basic`

Expected: prints `figure=runs/coverage_basic.png` and one line per policy. **The gate: `scripted` must show a larger `occupied_cells` count than `random`.** If it does not, raise `_FORWARD_PROB` in `src/mbfps/data/policies.py` or lengthen `_SWEEP_LEN`, re-collect, and re-run. Do not proceed to Plan 2 with a scripted policy that adds no coverage — it is dead weight in the dataset and the M1 rationale no longer holds.

- [ ] **Step 8: Benchmark the loader**

```bash
.venv/bin/python -c "
import time
from mbfps.data.buffer import ReplayBuffer
from mbfps.data.loader import SequenceLoader
buf = ReplayBuffer('data/basic', capacity_transitions=10**9)
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
- [ ] `pytest tests/envs/test_determinism.py` passes — same seed gives bit-identical frames and rewards, and an episode replays from its action sequence alone.
- [ ] `scripts/benchmark_env.py` reports a recorded `steps_per_second`.

**M1:**
- [ ] `scripts/collect.py` produces a dataset with a reported `transitions_per_second`.
- [ ] The loader benchmark reports `batches_per_second`.
- [ ] `pytest tests/data/test_episode.py` passes — stored data equals collected data.
- [ ] `pytest tests/data/test_features.py` passes — the feature cache is byte-identical on repeat.
- [ ] `scripts/coverage_report.py` shows `scripted` occupying more cells than `random`.
- [ ] `pytest tests/test_privileged_isolation.py` passes.

## Deferred to Plan 2 (M2–M3)

Encoders (all three arms), the RSSM, decoders, reward/continue heads, the world-model trainer, open-loop rollout evaluation, and the latent-to-privileged linear probe.

## Deferred to Plan 3 (M4–M5)

Actor, critic, imagination rollouts, the imagination-isolation test, the online Dreamer loop, and the model-free (PPO/DQN) sample-efficiency baseline.
