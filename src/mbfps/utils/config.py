"""Experiment configuration.

The study compares three arms that must differ in exactly one respect: the
space the model predicts in. Everything else -- batch size, sequence length,
learning rate, seed, embedding width -- is held identical.

That invariant is enforced structurally rather than by discipline: every arm is
built from the same `TrainConfig` instance and the same `EncoderConfig`
defaults, overriding only `kind`. A configuration file format would let one
arm's settings drift silently; here a stray field is a TypeError at call time.
"""

import dataclasses
from dataclasses import dataclass, replace

ARMS: tuple[str, ...] = ("cnn", "frozen_ssl", "random_vit")
"""The three arms. `cnn` is the baseline, `frozen_ssl` the treatment,
`random_vit` the control that separates pretraining from target stability."""


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder settings. `kind` is the ONLY field that may differ across arms.

    There is deliberately no `patch_dim` here. The width of a cached feature
    row is the frozen BACKBONE's property, not an arm-level setting, and lives
    in `mbfps.data.features.BACKBONE_GEOMETRY` beside its patch count. A
    shared 384 used to sit here and was read by exactly two lines of
    `BottleneckEncoder`; a backbone with narrower rows would have been built
    against 384 with no error until the first matmul. `bottleneck_dim` and
    `embed_dim` stay: they are the study's choices, held identical across
    arms, and the encoder checks `n_patches * bottleneck_dim == embed_dim`
    against each backbone's own patch count.
    """

    kind: str
    embed_dim: int = 2048
    cnn_depth: int = 32
    bottleneck_dim: int = 32
    standardise_features: bool = True
    """Normalise each cached patch vector before the bottleneck.

    Held identical across arms, like every other field here. It exists because
    the two feature caches arrive at very different scales -- DINOv2 std 2.3559
    against random_vit's 1.0000 -- and feeding both into a bare `nn.Linear` at
    one learning rate confounds the treatment/control contrast with optimisation
    conditioning. `False` reproduces the pre-2026-09-03 behaviour.
    """


@dataclass(frozen=True)
class TrainConfig:
    """Training settings, shared byte-for-byte across all arms."""

    batch_size: int = 16
    seq_len: int = 64
    lr: float = 1e-4
    steps: int = 20_000
    seed: int = 0
    device: str = "mps"


@dataclass(frozen=True)
class Config:
    """A complete experiment configuration."""

    arm: str
    encoder: EncoderConfig
    train: TrainConfig
    data_root: str = "data/my_way_home"


def get_config(arm: str, **overrides) -> Config:
    """Build the configuration for `arm`, applying `overrides` to TrainConfig.

    Args:
        arm: one of `ARMS`.
        **overrides: field names of `TrainConfig`.

    Raises:
        KeyError: if `arm` is not registered.
        TypeError: if an override names a field `TrainConfig` does not have.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; available: {list(ARMS)}")
    train = replace(TrainConfig(), **overrides) if overrides else TrainConfig()
    return Config(arm=arm, encoder=EncoderConfig(kind=arm), train=train)
