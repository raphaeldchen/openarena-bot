"""Overlap batch loading with computation.

Reading a feature batch costs ~101 ms (51 MB from memmap) while an SSL-arm
training step costs ~161 ms, so serial loading adds ~63% to the arms that are
otherwise fast. A single background thread is enough: NumPy releases the GIL
while copying, so the read genuinely overlaps the forward/backward pass.

The pixel arm sees ~6% overhead from the same read and does not need this, but
using one path for every arm keeps the arms comparable.
"""

import queue
import threading
from typing import Any, Iterator

from mbfps.data.loader import SequenceLoader

_SENTINEL = object()


class Prefetcher:
    """Yields batches from `loader`, produced on a background thread."""

    def __init__(self, loader: SequenceLoader, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self._loader = loader
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        try:
            while not self._stop.is_set():
                batch = self._loader.sample()
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            # Deliver the failure rather than dying silently and leaving the
            # consumer blocked on an empty queue forever.
            self._queue.put(exc)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        """Stop the worker thread. Safe to call more than once."""
        if self._stop.is_set():
            return
        self._stop.set()
        # Drain so a worker blocked on put() can observe the stop flag.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "Prefetcher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
