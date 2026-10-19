"""Reward transforms (milestone M4).

`basic`'s per-step rewards span -9 to +99: a squared error on the raw scale is
dominated by the kill by three orders of magnitude. DreamerV3 trains the reward
head on `symlog` and reads it through `symexp`. A checkpoint records which
transform its head was trained with (`reward_transform`), and every consumer
reads the head through `read_reward` -- never by assumption, because the M3
checkpoints are `identity`.
"""

import torch

REWARD_TRANSFORMS: tuple[str, ...] = ("identity", "symlog")


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * ln(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(y: torch.Tensor) -> torch.Tensor:
    """sign(y) * (exp(|y|) - 1), the inverse of `symlog`."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def _require_known(transform: str) -> None:
    if transform not in REWARD_TRANSFORMS:
        raise ValueError(
            f"unknown reward_transform {transform!r}; available: {list(REWARD_TRANSFORMS)}"
        )


def reward_target(rewards: torch.Tensor, transform: str) -> torch.Tensor:
    """What the reward head is trained to output for `rewards`."""
    _require_known(transform)
    return symlog(rewards) if transform == "symlog" else rewards


def read_reward(
    head_output: torch.Tensor, transform: str, bounds: tuple[float, float] | None = None
) -> torch.Tensor:
    """The reward head's output in reward units.

    `bounds` (M5a §4.3): `(r_min, r_max)` in reward units. The head's output is
    clipped to `[reward_target(r_min), reward_target(r_max)]` BEFORE the inverse
    transform, so an off-distribution head output cannot extrapolate through
    `symexp`. None leaves the readout exactly as M4 had it."""
    _require_known(transform)
    if bounds is not None:
        lo, hi = (reward_target(torch.tensor(float(b)), transform).item() for b in bounds)
        head_output = head_output.clamp(min=lo, max=hi)
    return symexp(head_output) if transform == "symlog" else head_output
