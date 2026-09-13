"""M2's three autoencoder scripts take `choices=KINDS`, not `choices=ARMS`.

M3c retired `cnn` from `ARMS`. These scripts are how M2 trained, rendered and
scored the end-to-end `CNNEncoder`, and `runs/m2_fixed/autoencoder_cnn.pt` --
which is now the `pixel_ae` backbone -- was produced by the first of them. If
they took `ARMS` they would refuse the one kind they exist for.

The parser is captured rather than run: each script's `main` calls
`parse_args()` inline and then trains, so `parse_args` is replaced with a
function that hands the parser back and stops. What is asserted is the
parser's real configuration, not the source text.
"""

import argparse
import importlib.util
from pathlib import Path

import pytest

from mbfps.utils.config import ARMS, KINDS

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_for_kinds_test", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Captured(Exception):
    """Raised from the stand-in `parse_args` so `main` never reaches training."""


def _capture_parser(monkeypatch, script) -> argparse.ArgumentParser:
    holder = {}

    def parse_args(self, args=None, namespace=None):
        holder["parser"] = self
        raise _Captured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    with pytest.raises(_Captured):
        script.main()
    return holder["parser"]


def _choices(parser: argparse.ArgumentParser, dest: str):
    action = next(a for a in parser._actions if a.dest == dest)
    return tuple(action.choices)


def test_kinds_is_arms_plus_cnn():
    """L7 guard for the parametrisations below."""
    assert KINDS == ("cnn", "pixel_ae", "frozen_ssl", "random_vit")
    assert "cnn" in KINDS and "cnn" not in ARMS


@pytest.mark.parametrize("name", ["train_autoencoder", "reconstruction_grid"])
def test_single_arm_m2_scripts_offer_every_kind(monkeypatch, name):
    parser = _capture_parser(monkeypatch, _load(name))
    assert _choices(parser, "arm") == KINDS, (
        f"scripts/{name}.py must take choices=KINDS: `cnn` is the encoder it "
        "exists to build, and `ARMS` no longer contains it")


def test_eval_reconstruction_offers_every_kind_and_defaults_to_the_m2_arms(
    monkeypatch,
):
    """`--arms` is new: the script used to loop over `ARMS`, which after M3c
    would silently drop `cnn` from M2's paired comparison. The default is the
    three kinds M2 actually trained -- `pixel_ae` is constructible but has no
    `autoencoder_pixel_ae.pt`, so it is opt-in rather than a guaranteed
    FileNotFoundError on the default command line."""
    parser = _capture_parser(monkeypatch, _load("eval_reconstruction"))
    assert _choices(parser, "arms") == KINDS
    action = next(a for a in parser._actions if a.dest == "arms")
    assert list(action.default) == ["cnn", "frozen_ssl", "random_vit"]
    assert action.nargs == "+"
