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
