from __future__ import annotations

import queue
import threading

import numpy as np


class AudioQueue:
    def __init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._closed = threading.Event()

    def put(self, block: np.ndarray) -> None:
        if not self._closed.is_set():
            self._queue.put(block)

    def get(self, timeout: float | None = None) -> np.ndarray | None:
        if self._closed.is_set() and self._queue.empty():
            return None
        try:
            if timeout is None:
                return self._queue.get()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed.set()
