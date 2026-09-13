"""The backbone registry's home is a leaf module, and `features` re-exports it.

`mbfps.data.features` imports `transformers` at module top (1.5-2.2 s), and
until the registry moved, `mbfps.models.encoders` -- and through it every
`WorldModel` build -- paid that import to read one dict. The two things this
file pins are the two that make the move worth anything: that the cheap
readers really are cheap (no `transformers`, no `features`, in the process
after importing them), and that the re-export is the SAME object rather than
a copy, so a test that rebinds an entry through either spelling rebinds the
dict the loader and the bottleneck read.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import mbfps.data.features as features
import mbfps.data.geometry as geometry
from mbfps.data.geometry import (
    BACKBONE_GEOMETRY,
    BACKBONES,
    PIXEL_AE_CHECKPOINT,
    geometry_mismatch,
)


def test_the_registry_literals_live_in_the_leaf_module():
    """Literal, not derived, like `tests/data/test_features.py`'s own copy:
    a typo here is a matmul error at step 0."""
    assert BACKBONES == ("dinov2", "random_vit", "pixel_ae")
    assert BACKBONE_GEOMETRY == {
        "dinov2": (64, 384),
        "random_vit": (64, 384),
        "pixel_ae": (64, 32),
    }
    assert PIXEL_AE_CHECKPOINT == Path("runs/m2_fixed/autoencoder_cnn.pt")


def test_features_re_exports_the_same_objects_not_copies():
    """IDENTITY, not equality. `tests/models/test_encoders.py` and
    `tests/data/test_cache_features_script.py` do `monkeypatch.setitem` on
    `features.BACKBONE_GEOMETRY` and expect the encoder and the cache-size
    estimate to see the rebound entry; an equal copy would leave those tests
    green while patching a dict nobody reads."""
    assert features.BACKBONE_GEOMETRY is geometry.BACKBONE_GEOMETRY
    assert features.BACKBONES is geometry.BACKBONES
    assert features.PIXEL_AE_CHECKPOINT is geometry.PIXEL_AE_CHECKPOINT
    assert features.geometry_mismatch is geometry.geometry_mismatch


def test_the_registry_readers_do_not_import_transformers():
    """In a FRESH process: importing the encoder and the loader must leave
    `transformers` and `mbfps.data.features` unimported. In-process the
    suite has long since imported both, so `sys.modules` here says nothing;
    a subprocess is the only honest witness."""
    probe = (
        "import sys\n"
        "import mbfps.models.encoders, mbfps.data.loader\n"
        "leaked = sorted(m for m in sys.modules\n"
        "                if m == 'transformers' or m.startswith('transformers.')\n"
        "                or m == 'mbfps.data.features')\n"
        "print(leaked)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        timeout=120, check=True,
    )
    assert completed.stdout.strip() == "[]", (
        "a model build or a loader construction is paying for the "
        f"transformers import again: {completed.stdout.strip()}")


def test_the_leaf_module_imports_nothing_heavy():
    """The other half of the same guard: the leaf must stay a leaf. `torch`
    would be the obvious way to break it (a `torch.device` default, say)."""
    probe = (
        "import sys\n"
        "import mbfps.data.geometry\n"
        "print(sorted(m for m in ('torch', 'numpy', 'transformers')"
        " if m in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        timeout=120, check=True,
    )
    assert completed.stdout.strip() == "[]", completed.stdout


# --- geometry_mismatch: one message for four guards ---------------------------

def test_geometry_mismatch_names_backbone_expected_then_got():
    """Expected BEFORE got, each labelled: the loader's test asserts the two
    labelled substrings separately so a swap is visible, and the encoder's
    and extractor's regexes read `backbone.*expected.*got` in that order."""
    text = geometry_mismatch("pixel_ae", (64, 32), (64, 384))
    assert "'pixel_ae'" in text
    assert "does not match backbone 'pixel_ae'" in text
    assert "expected rows of shape (64, 32)" in text
    assert "got (64, 384)" in text
    assert text.index("(64, 32)") < text.index("(64, 384)")


def test_geometry_mismatch_offers_the_recache_hint_only_for_a_file():
    """A cache on disk is fixed by re-caching that backbone; rows handed to a
    forward pass are not, so the hint would send the user to the wrong tool."""
    path = Path("data/x/ep_000007.features_pixel_ae.npy")
    with_file = geometry_mismatch("pixel_ae", (64, 32), (2048,), path=path)
    assert str(path) in with_file
    assert "Re-run scripts/cache_features.py --backbone pixel_ae" in with_file
    assert "got (2048,)" in with_file

    without = geometry_mismatch("pixel_ae", (64, 32), (2048,))
    assert "cache_features" not in without
    assert str(path) not in without
    assert "does not match backbone 'pixel_ae'" in without


@pytest.mark.parametrize("expected,got", [
    ([64, 32], (64, 384)),          # a list where a tuple belongs
    ((64, 32), [64, 384]),
])
def test_geometry_mismatch_renders_shapes_as_tuples_whatever_it_is_handed(
    expected, got
):
    """`torch.Size` and lists both arrive here; the message must read the same."""
    text = geometry_mismatch("dinov2", expected, got)
    assert "(64, 32)" in text and "(64, 384)" in text
    assert "[64" not in text
