import numpy as np
import pytest

from mbfps.data.buffer import ReplayBuffer
from mbfps.data.episode import Episode
from mbfps.data.loader import SequenceLoader
from mbfps.data.prefetch import Prefetcher
from mbfps.envs.protocol import OBS_SHAPE

KEYS = ("health", "pos_x", "pos_y", "pos_z", "angle")


def make_episode(t: int, fill: int) -> Episode:
    return Episode(
        obs=np.full((t + 1, *OBS_SHAPE), fill, dtype=np.uint8),
        actions=np.zeros(t, dtype=np.int32),
        rewards=np.zeros(t, dtype=np.float32),
        terminated=np.zeros(t, dtype=bool),
        truncated=np.zeros(t, dtype=bool),
        privileged=np.zeros((t + 1, len(KEYS)), dtype=np.float32),
        privileged_keys=KEYS,
        policy_name="random",
        seed=fill,
        scenario="my_way_home",
    )


@pytest.fixture
def buffer(tmp_path):
    buf = ReplayBuffer(tmp_path, capacity_transitions=10_000)
    for fill in (1, 2, 3):
        buf.add(make_episode(t=80, fill=fill))
    return buf


def test_yields_the_requested_number_of_batches(buffer):
    loader = SequenceLoader(buffer, batch_size=2, seq_len=16, seed=0)
    with Prefetcher(loader, depth=2) as pf:
        batches = [next(iter(pf)) for _ in range(3)]
    assert len(batches) == 3
    assert all(b["actions"].shape == (2, 16) for b in batches)


def test_batches_match_direct_sampling_in_order(buffer):
    """Prefetching must not change what the loader would have produced."""
    direct_loader = SequenceLoader(buffer, batch_size=2, seq_len=16, seed=7)
    expected = [direct_loader.sample() for _ in range(4)]

    pf_loader = SequenceLoader(buffer, batch_size=2, seq_len=16, seed=7)
    with Prefetcher(pf_loader, depth=2) as pf:
        it = iter(pf)
        got = [next(it) for _ in range(4)]

    for i, (a, b) in enumerate(zip(expected, got)):
        assert np.array_equal(a["obs"], b["obs"]), f"batch {i} differs"
        assert np.array_equal(a["episode_index"], b["episode_index"])
        assert np.array_equal(a["window_start"], b["window_start"])


def test_close_is_idempotent(buffer):
    pf = Prefetcher(SequenceLoader(buffer, 2, 16, seed=0), depth=1)
    next(iter(pf))
    pf.close()
    pf.close()


def test_context_manager_closes_the_thread(buffer):
    import threading

    before = threading.active_count()
    with Prefetcher(SequenceLoader(buffer, 2, 16, seed=0), depth=1) as pf:
        next(iter(pf))
    assert threading.active_count() == before, "worker thread outlived the context"


def test_worker_exception_propagates_to_the_consumer(buffer):
    """A failing loader must raise here, not hang the consumer forever."""

    class _Broken(SequenceLoader):
        def sample(self, include_privileged: bool = False):
            raise RuntimeError("loader exploded")

    with Prefetcher(_Broken(buffer, 2, 16, seed=0), depth=1) as pf:
        with pytest.raises(RuntimeError, match="loader exploded"):
            next(iter(pf))


def test_depth_must_be_positive(buffer):
    with pytest.raises(ValueError, match="depth must be at least 1"):
        Prefetcher(SequenceLoader(buffer, 2, 16, seed=0), depth=0)
