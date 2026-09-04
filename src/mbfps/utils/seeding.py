"""Seed every RNG this project touches from a single integer."""

import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed `random`, NumPy, and PyTorch (CPU + all accelerators)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_SEED_SPACE = 2**31


def fork_seed(base_seed: int, name: str) -> int:
    """A stable seed for `name`, derived from `base_seed`.

    Uses blake2b rather than `hash()`, whose string hashing is randomised per
    process by PYTHONHASHSEED -- that would make module init non-reproducible
    across runs, which is the opposite of what this is for.
    """
    digest = hashlib.blake2b(
        name.encode("utf-8"), digest_size=8, key=str(base_seed).encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % _SEED_SPACE


@contextmanager
def seeded_init(base_seed: int, name: str):
    """Build a module under an RNG seeded only by `base_seed` and `name`.

    Module construction otherwise inherits whatever the global RNG has already
    produced, so a module's weights depend on how many draws every module built
    before it consumed. That made M2's shared decoder initialise differently for
    the pixel arm than for the feature arms -- an arm-parity violation in a
    module that was supposed to be identical across all three.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fork_seed(base_seed, name))
        yield
