"""The cache-size estimate reads the backbone's geometry, not a constant.

`scripts/cache_features.py` refuses to start when the disk cannot hold the
cache it is about to write. That estimate used to multiply two module
constants, `N_PATCHES * FEATURE_DIM`, that described the ViT backbones only.
A backbone with narrower rows would have been sized as if it were 384 wide
and refused on a disk with room for it. The arithmetic is now a function that
reads the registry, so it can be tested without a buffer or a backbone.
"""

import importlib.util
from pathlib import Path

import pytest

from mbfps.data.features import BACKBONE_GEOMETRY

_SPEC = importlib.util.spec_from_file_location(
    "cache_features_script",
    Path(__file__).resolve().parents[2] / "scripts" / "cache_features.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


def test_cache_bytes_is_frames_times_geometry_times_float16():
    assert script.cache_bytes(10, "dinov2") == 10 * 64 * 384 * 2
    assert script.cache_bytes(10, "random_vit") == 10 * 64 * 384 * 2


def test_cache_bytes_reads_the_registry_not_a_constant(monkeypatch):
    """Both real backbones are (64, 384), so the test above cannot tell a
    registry read from the old constants. A geometry that matches neither
    constant can."""
    monkeypatch.setitem(BACKBONE_GEOMETRY, "random_vit", (64, 32))
    assert script.cache_bytes(10, "random_vit") == 10 * 64 * 32 * 2


def test_cache_bytes_rejects_an_unknown_backbone():
    with pytest.raises(KeyError):
        script.cache_bytes(10, "nope")
