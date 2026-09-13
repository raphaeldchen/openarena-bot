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

ARMS: tuple[str, ...] = ("pixel_ae", "frozen_ssl", "random_vit")
"""The study's arms. Every M3 tool -- the driver, report, aggregation, rollout
evaluation, diagnostics, pooling -- takes `choices=ARMS`.

`pixel_ae` is the frozen M2 pixel autoencoder's encoder, `frozen_ssl` the
DINOv2 treatment, `random_vit` the control that separates pretraining from
target stability. M3b's end-to-end `cnn` arm is not here: measured, its
embedding target was born collapsed (`embedding_loss` 0.0008 at step 0 against
0.32-0.41 for the feature arms), its dynamics KL never cleared the free-bits
floor, and its prior never received a gradient. The order here is the row
order of every report table."""

KINDS: tuple[str, ...] = ("cnn",) + ARMS
"""Every encoder `build_encoder` can construct. `get_config` validates against
this, not `ARMS`, because M2's autoencoder scripts and tests still build the
end-to-end `CNNEncoder` under the name `cnn` and must keep doing so -- and the
shipped M3b checkpoints (`runs/m3_study/world_model_cnn_seed*.pt`) load through
`get_config("cnn")`. Buildable is not selectable: no M3 tool offers `cnn`, so
no M3c artefact can be written under the old name."""


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
        arm: one of `KINDS` -- a study arm, or `cnn` for M2's autoencoder and
            the shipped M3b artefacts.
        **overrides: field names of `TrainConfig`.

    Raises:
        KeyError: if `arm` is not a registered kind. The message lists `KINDS`,
            so a reader sees both that `cnn` is still a kind and which three
            the study runs.
        TypeError: if an override names a field `TrainConfig` does not have.
    """
    if arm not in KINDS:
        raise KeyError(f"unknown arm {arm!r}; available: {list(KINDS)}")
    train = replace(TrainConfig(), **overrides) if overrides else TrainConfig()
    return Config(arm=arm, encoder=EncoderConfig(kind=arm), train=train)
