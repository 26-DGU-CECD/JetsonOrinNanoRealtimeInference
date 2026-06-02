from __future__ import annotations

import numpy as np

from config import CHUNK_SAMPLES


class AudioBuffer:
    def __init__(self, chunk_samples: int = CHUNK_SAMPLES) -> None:
        self.chunk_samples = int(chunk_samples)
        self._pending_blocks: list[np.ndarray] = []
        self._pending_samples = 0

    def add_block(self, block: np.ndarray) -> list[np.ndarray]:
        block = np.asarray(block, dtype=np.float32)
        if block.size == 0:
            return []

        self._pending_blocks.append(block.copy())
        self._pending_samples += block.shape[0]
        if self._pending_samples < self.chunk_samples:
            return []

        joined = np.concatenate(self._pending_blocks, axis=0)
        chunks: list[np.ndarray] = []
        offset = 0
        while joined.shape[0] - offset >= self.chunk_samples:
            chunks.append(joined[offset : offset + self.chunk_samples].copy())
            offset += self.chunk_samples

        remainder = joined[offset:]
        self._pending_blocks = [remainder.copy()] if remainder.size else []
        self._pending_samples = remainder.shape[0]
        return chunks

    def reset(self) -> None:
        self._pending_blocks.clear()
        self._pending_samples = 0
