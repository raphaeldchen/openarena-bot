"""`scripts/cache_features.py` with a third backbone whose rows are not (64, 384).

The script is the only thing that writes a feature cache, and a cache is read
by every training step of two-thirds of the study without anything checking
how it was made. So the tests are arranged around the ways a `pixel_ae` cache
can be wrong while looking right: a file written under the ViT suffix (so the
`frozen_ssl` arm silently trains on autoencoder features), a file whose rows
are (64, 384) or a flat 2048 (refused at load, but only after five minutes of
encoding), a disk guard that demands the ViT cache's 2.93 GB for a 0.24 GB
write, and -- the defect this project keeps meeting -- an encoder that is not
the one the checkpoint holds. Every fixture builds its own M2-shaped
checkpoint from a real `CNNEncoder`, so nothing here depends on
`runs/m2_fixed/autoencoder_cnn.pt` existing.

The script is loaded by path, as `tests/eval/test_eval_rollout_script.py`
does, because `scripts/` is not a package.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.features import BACKBONE_GEOMETRY, BACKBONES, PIXEL_AE_CHECKPOINT
from mbfps.envs.protocol import OBS_SHAPE
from mbfps.models.encoders import CNNEncoder
from mbfps.utils.config import EncoderConfig

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


KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")
T = 5
"""Transitions per fixture episode: T + 1 = 6 frames, one batch of 32."""
N_FRAMES = T + 1

# Hand-written, NOT derived from BACKBONE_GEOMETRY (L7: parametrising the
# expectation over the registry under test would let a wrong registry entry
# agree with itself). The literal rows are what the disk guard must be told
# for one six-frame episode: 64 x 384 float16 rows for the ViTs, 64 x 32 for
# the autoencoder. `test_expectations_cover_every_backbone` forces a new row
# the day a fourth backbone is registered.
EXPECTED_BYTES = {
    "dinov2": N_FRAMES * 64 * 384 * 2,
    "random_vit": N_FRAMES * 64 * 384 * 2,
    "pixel_ae": N_FRAMES * 64 * 32 * 2,
}
EXPECTED_SUFFIX = {
    "dinov2": ".features.npy",
    "random_vit": ".features_random_vit.npy",
    "pixel_ae": ".features_pixel_ae.npy",
}


def _episode(seed: int) -> Episode:
    """Six random frames: random so that no two encode to the same row."""
    rng = np.random.default_rng(seed)
    return Episode(
        obs=rng.integers(0, 256, (N_FRAMES, *OBS_SHAPE), dtype=np.uint8),
        actions=np.zeros(T, dtype=np.int32),
        rewards=np.zeros(T, dtype=np.float32),
        terminated=np.zeros(T, dtype=bool),
        truncated=np.zeros(T, dtype=bool),
        privileged=np.zeros((N_FRAMES, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=seed,
        scenario="my_way_home",
    )


@pytest.fixture
def data_dir(tmp_path) -> Path:
    root = tmp_path / "data"
    ReplayBuffer(root, capacity_transitions=10_000).add(_episode(seed=0))
    return root


def _fake_m2_checkpoint(path: Path, seed: int) -> Path:
    """An M2-shaped checkpoint from a real CNNEncoder, without runs/m2_fixed.

    `mbfps.training.autoencoder.train_autoencoder` saves
    `{"arm": cfg.arm, "state_dict": model.state_dict()}` where the model holds
    its encoder under `encoder.` and its decoder under `decoder.`. Only the
    encoder subset matters to the backbone loader, and a random-init encoder is
    enough to prove that THIS file's weights are the ones that reached the
    cache -- two seeds give two different encoders.
    """
    torch.manual_seed(seed)
    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    torch.save(
        {
            "arm": "cnn",
            "state_dict": {f"encoder.{k}": v for k, v in encoder.state_dict().items()},
        },
        path,
    )
    return path


@pytest.fixture
def spy_extractor(monkeypatch) -> list[dict]:
    """Replace the script's FeatureExtractor with one that records its kwargs.

    It still encodes -- zeros of the backbone's own geometry -- so `main` runs
    to completion and the recorded call can be asserted (L8: a spy whose calls
    are never asserted is not a test).
    """
    calls: list[dict] = []

    class _Spy:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.backbone = kwargs["backbone"]

        def encode(self, frames):
            n_patches, patch_dim = BACKBONE_GEOMETRY[self.backbone]
            return np.zeros((len(frames), n_patches, patch_dim), dtype=np.float16)

    monkeypatch.setattr(script, "FeatureExtractor", _Spy)
    return calls


# --- the command line -------------------------------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_registered_backbone_is_accepted(backbone):
    assert script._parser().parse_args(["--backbone", backbone]).backbone == backbone


def test_pixel_ae_is_accepted_and_the_arm_name_cnn_is_not():
    """The backbone is `pixel_ae`; `cnn` is an encoder kind, not a cache.

    Stated with literals rather than over BACKBONES: if the registry lost
    `pixel_ae`, the parametrised test above would shrink and stay green.
    """
    assert script._parser().parse_args(["--backbone", "pixel_ae"]).backbone == "pixel_ae"
    with pytest.raises(SystemExit):
        script._parser().parse_args(["--backbone", "cnn"])


def test_checkpoint_defaults_to_the_m2_encoder_and_parses_as_a_path():
    parser = script._parser()
    default = parser.parse_args([]).checkpoint
    assert default == PIXEL_AE_CHECKPOINT
    assert isinstance(default, Path)
    explicit = parser.parse_args(["--checkpoint", "runs/other/autoencoder_cnn.pt"]).checkpoint
    assert explicit == Path("runs/other/autoencoder_cnn.pt")
    assert isinstance(explicit, Path), "a str here reaches FeatureExtractor as a str"


# --- the disk guard ---------------------------------------------------------


def test_expectations_cover_every_backbone():
    assert set(EXPECTED_BYTES) == set(BACKBONES), "add a hand-written row"
    assert set(EXPECTED_SUFFIX) == set(BACKBONES), "add a hand-written row"


def test_cache_bytes_matches_the_m1_dataset_numbers():
    """The M1 buffer is 122 episodes, 59,633 frames (summed from the filenames).

    Its ViT caches are 2.93 GB on disk, so `dinov2` must reproduce that; the
    `pixel_ae` cache is 12x smaller. Stated in bytes so a 4-byte (float32)
    slip or a ViT-width slip for `pixel_ae` is a different number, not a
    rounding.
    """
    assert script.cache_bytes(59_633, "dinov2") == 2_931_081_216
    assert script.cache_bytes(59_633, "random_vit") == 2_931_081_216
    assert script.cache_bytes(59_633, "pixel_ae") == 244_256_768


@pytest.mark.parametrize("backbone", BACKBONES)
def test_main_asks_the_disk_for_the_backbones_own_byte_count(
    backbone, data_dir, spy_extractor, monkeypatch
):
    """The old estimate was `frames * 64 * 384 * 2` for every backbone.

    For `pixel_ae` that demands 2.93 GB free to write 0.24 GB -- on the
    2.3-GB-free machine this script was written for, the one cache that fits
    would be the one refused.
    """
    asked: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        script, "require_free_bytes", lambda path, needed: asked.append((path, needed))
    )
    script.main(
        ["--data", str(data_dir), "--backbone", backbone, "--device", "cpu",
         "--checkpoint", str(data_dir / "unused.pt")]
    )
    assert asked == [(data_dir, EXPECTED_BYTES[backbone])]


# --- the checkpoint reaches the extractor -----------------------------------


@pytest.mark.parametrize("backbone", BACKBONES)
def test_main_forwards_checkpoint_to_the_extractor(backbone, data_dir, spy_extractor):
    """Every backbone, not only `pixel_ae`: the flag is forwarded unconditionally
    and `build_backbone` decides what to do with it. A conditional here would
    be a second place that knows which backbone reads a checkpoint."""
    checkpoint = data_dir / "some_autoencoder.pt"
    script.main(
        ["--data", str(data_dir), "--backbone", backbone, "--device", "cpu",
         "--seed", "3", "--checkpoint", str(checkpoint)]
    )
    assert spy_extractor == [
        {"backbone": backbone, "device": "cpu", "seed": 3, "checkpoint": checkpoint}
    ]


def test_main_forwards_the_default_checkpoint_when_none_is_given(data_dir, spy_extractor):
    script.main(["--data", str(data_dir), "--backbone", "pixel_ae", "--device", "cpu"])
    [call] = spy_extractor
    assert call["checkpoint"] == PIXEL_AE_CHECKPOINT


# --- the cache itself -------------------------------------------------------


def test_pixel_ae_cache_is_the_loaded_encoders_output(data_dir, tmp_path):
    """End to end through the real FeatureExtractor on CPU.

    The file must sit under the `pixel_ae` suffix, hold (T+1, 64, 32) float16,
    and be bit-for-bit the forward pass of the encoder in `--checkpoint` (M8:
    shape and dtype alone are satisfied by a random-init encoder, or by
    whatever `PIXEL_AE_CHECKPOINT` happens to hold on this machine). Both
    sides run one CPU batch of the same six frames, so bit-equality is the
    right bar -- `test_features.py` holds the DINOv2 path to the same one.
    """
    checkpoint = _fake_m2_checkpoint(tmp_path / "autoencoder_cnn.pt", seed=0)
    script.main(
        ["--data", str(data_dir), "--backbone", "pixel_ae", "--device", "cpu",
         "--checkpoint", str(checkpoint)]
    )

    [ep_path] = ReplayBuffer(data_dir, capacity_transitions=10_000).episode_paths()
    out = ep_path.with_suffix(".features_pixel_ae.npy")  # literal: the name IS the contract
    assert out.is_file(), sorted(p.name for p in data_dir.iterdir())
    assert not ep_path.with_suffix(".features.npy").exists(), (
        "written under the dinov2 suffix -- frozen_ssl would train on these"
    )
    saved = np.load(out)
    with np.load(ep_path) as data:
        obs = data["obs"]
    assert saved.shape == (obs.shape[0], 64, 32)
    assert saved.dtype == np.float16

    encoder = CNNEncoder(EncoderConfig(kind="cnn"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items()})
    encoder.eval()
    with torch.no_grad():
        reference = encoder(torch.from_numpy(obs)).reshape(N_FRAMES, 64, 32)
    np.testing.assert_array_equal(saved, reference.to(torch.float16).numpy())
    # L1: if every frame encoded to one row, the equality above could not
    # distinguish a cache in frame order from one that is not.
    assert not np.array_equal(saved[0], saved[1])

    # The contents follow --checkpoint: a second encoder, a second cache.
    other = _fake_m2_checkpoint(tmp_path / "autoencoder_cnn_other.pt", seed=1)
    script.main(
        ["--data", str(data_dir), "--backbone", "pixel_ae", "--device", "cpu",
         "--checkpoint", str(other)]
    )
    assert not np.array_equal(np.load(out), saved)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_clear_removes_only_the_named_backbones_cache(backbone, data_dir, spy_extractor):
    """Three caches coexist now; `--clear` must take exactly the one named.

    It must also build no extractor: clearing the `pixel_ae` cache cannot be
    made to depend on the checkpoint that produced it still existing.
    """
    [ep_path] = ReplayBuffer(data_dir, capacity_transitions=10_000).episode_paths()
    for suffix in EXPECTED_SUFFIX.values():
        np.save(ep_path.with_suffix(suffix), np.zeros((N_FRAMES, 1, 1), dtype=np.float16))

    script.main(["--data", str(data_dir), "--backbone", backbone, "--clear"])

    survivors = {
        name for name, suffix in EXPECTED_SUFFIX.items()
        if ep_path.with_suffix(suffix).is_file()
    }
    assert survivors == set(BACKBONES) - {backbone}
    assert spy_extractor == [], "--clear constructed a backbone"
