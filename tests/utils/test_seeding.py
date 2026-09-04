import torch
import torch.nn as nn

from mbfps.utils.seeding import fork_seed, seed_everything, seeded_init


def test_fork_seed_is_deterministic():
    assert fork_seed(0, "decoder") == fork_seed(0, "decoder")


def test_fork_seed_differs_by_name():
    assert fork_seed(0, "decoder") != fork_seed(0, "rssm")


def test_fork_seed_differs_by_base_seed():
    assert fork_seed(0, "decoder") != fork_seed(1, "decoder")


def test_seeded_init_is_independent_of_prior_rng_consumption():
    """The whole point: a module's weights must not depend on what came before.

    This is the M2 decoder-parity bug in miniature -- the CNN arm drew 26,382,304
    values before the decoder was built and the bottleneck arms drew 12,320, so
    the "shared" decoder started from different weights per arm.
    """
    def build(waste: int) -> torch.Tensor:
        seed_everything(0)
        torch.randn(waste)  # simulate a differently-sized encoder
        with seeded_init(0, "decoder"):
            layer = nn.Linear(8, 8)
        return layer.weight.detach().clone()

    assert torch.equal(build(0), build(1_000_000))


def test_seeded_init_restores_the_outer_rng():
    """It must not disturb the stream its caller is using."""
    seed_everything(0)
    before = torch.randn(4)
    seed_everything(0)
    with seeded_init(0, "whatever"):
        nn.Linear(64, 64)
    after = torch.randn(4)
    assert torch.equal(before, after)
