import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def _positions(t: int, fill: int, curved: bool) -> tuple[np.ndarray, np.ndarray]:
    """`(pos_x, pos_y)` over `t + 1` frames.

    STRAIGHT is the original and the default: `3 * i` and `-2 * i`, linear in
    the frame index. Every existing fixture keeps it, and every number pinned
    against those fixtures is unchanged.

    CURVED exists because a straight line makes `p(t + k) - p(t)` the SAME
    VECTOR in every window, at every k, in every episode -- so a control that
    permutes which window's displacement a prediction is read against is
    bitwise the treatment, and a test of that control cannot fail. A curve
    that is not a polynomial in `i` gives each window its own displacement;
    `fill` shifts its phase so two episodes are not the same curve either.
    """
    i = np.arange(t + 1, dtype=np.float64)
    if not curved:
        return (i * 3.0).astype(np.float32), (i * -2.0).astype(np.float32)
    phase = 0.37 * float(fill)
    return (
        (120.0 * np.sin(0.21 * i + phase) + 2.0 * i).astype(np.float32),
        (90.0 * np.cos(0.13 * i - phase) - 1.5 * i).astype(np.float32),
    )


def _build_buffer(tmp_path, n_episodes: int, curved: bool = False) -> ReplayBuffer:
    """`n_episodes` synthetic 40-step episodes with cached features, one
    per study backbone at its own row width. `fill` runs 1..n so the
    six-episode fixture is exactly what it always was."""
    buf = ReplayBuffer(tmp_path / "data", capacity_transitions=100_000)
    for fill in range(1, n_episodes + 1):
        t = 40
        steps = np.arange(t)
        pos_x, pos_y = _positions(t, fill, curved)
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
                pos_x,
                pos_y,
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


@pytest.fixture
def small_buffer(tmp_path):
    """Six episodes long enough for context+horizon, with cached features.

    Six, not four: `episode_split(val_fraction=0.2)` floors to zero validation
    episodes below five and raises.
    """
    return _build_buffer(tmp_path, 6)


@pytest.fixture
def wide_buffer(tmp_path):
    """Thirty episodes: `episode_split(0.2)` holds out six, leaving 24 to
    train on -- more than `PROBE_EPISODE_LIMIT`, so the split-gap strata are
    non-empty: train_probe 20, train_held 4, val 6."""
    return _build_buffer(tmp_path, 30)


@pytest.fixture
def curved_buffer(tmp_path):
    """`wide_buffer`'s thirty episodes on a CURVED path.

    The straight path gives every window the same `p(t + k) - p(t)` at every
    k, in every episode -- which makes a permuted-pairing control bitwise
    identical to the treatment it exists to differ from, and a test of that
    control one that cannot fail. M3i's measure and read phases are read
    against this fixture for that reason.
    """
    return _build_buffer(tmp_path, 30, curved=True)
