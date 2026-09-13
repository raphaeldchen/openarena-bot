import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


@pytest.fixture
def small_buffer(tmp_path):
    """Six episodes long enough for context+horizon, with cached features.

    Six, not four: `episode_split(val_fraction=0.2)` floors to zero validation
    episodes below five and raises.
    """
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, 7):
        t = 40
        steps = np.arange(t)
        obs = np.zeros((t + 1, *OBS_SHAPE), dtype=np.uint8)
        obs[:, 0, 0, 0] = np.arange(t + 1, dtype=np.uint8)
        obs[:, 0, 0, 1] = fill
        buf.add(Episode(
            obs=obs,
            actions=(steps % 6).astype(np.int32),
            rewards=(steps % 3).astype(np.float32),
            terminated=(steps == t - 1),
            truncated=np.zeros(t, dtype=bool),
            privileged=np.stack([
                np.full(t + 1, 100.0),
                np.arange(t + 1, dtype=np.float32) * 3.0,
                np.arange(t + 1, dtype=np.float32) * -2.0,
                np.zeros(t + 1),
                (np.arange(t + 1) * 7.0) % 360.0,
            ], axis=1).astype(np.float32),
            privileged_keys=KEYS,
            policy_name="random",
            seed=fill,
            scenario="my_way_home",
        ))
    rng = np.random.default_rng(0)
    for path in buf.episode_paths():
        # One cache per study backbone, each at ITS OWN row width: the ViT
        # arms read (64, 384), pixel_ae reads (64, 32). A pixel_ae cache
        # written 384 wide would be refused by the loader's geometry check.
        for suffix, width in ((".features.npy", 384),
                              (".features_random_vit.npy", 384),
                              (".features_pixel_ae.npy", 32)):
            np.save(path.with_suffix(suffix),
                    rng.random((41, 64, width)).astype(np.float16))
    return buf
