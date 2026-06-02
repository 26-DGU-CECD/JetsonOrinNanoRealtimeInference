from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import CHUNK_SAMPLES, DB_EPSILON


@dataclass(frozen=True)
class EnhancementResult:
    waveform: np.ndarray
    clipped: bool
    quiet_gain: float
    loud_gain: float


class AudioProcessor:
    def __init__(
        self,
        *,
        min_db: float,
        enhance_threshold_db: float,
        noise_reduction_db: float,
        main_gain_db: float,
        enhance_sharpness: float,
        chunk_samples: int = CHUNK_SAMPLES,
    ) -> None:
        self.chunk_samples = int(chunk_samples)
        self.threshold_dbfs = self.dbfs_threshold(min_db)
        self.enhance_threshold_db = float(enhance_threshold_db)
        self.noise_reduction_db = float(noise_reduction_db)
        self.main_gain_db = float(main_gain_db)
        self.enhance_sharpness = float(enhance_sharpness)
        self._pending_blocks: list[np.ndarray] = []
        self._pending_samples = 0

    def chunks_from_block(self, block: np.ndarray) -> list[np.ndarray]:
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

    def is_low_signal(self, dbfs: float) -> bool:
        return float(dbfs) < self.threshold_dbfs

    def enhance(self, waveform: np.ndarray) -> EnhancementResult:
        waveform = np.asarray(waveform, dtype=np.float32)
        threshold = self._enhance_threshold_amplitude(self.enhance_threshold_db)
        quiet_gain = float(10.0 ** (-abs(self.noise_reduction_db) / 20.0))
        loud_gain = float(10.0 ** (self.main_gain_db / 20.0))
        sharpness = max(self.enhance_sharpness, 0.1)

        relative_level = np.abs(waveform) / max(threshold, DB_EPSILON)
        loud_weight = np.power(relative_level, sharpness)
        loud_weight = loud_weight / (1.0 + loud_weight)
        gain = quiet_gain + (loud_gain - quiet_gain) * loud_weight

        enhanced = waveform * gain.astype(np.float32, copy=False)
        clipped = bool(np.any(np.abs(enhanced) > 1.0))
        enhanced = np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)
        return EnhancementResult(enhanced, clipped, quiet_gain, loud_gain)

    @classmethod
    def _enhance_threshold_amplitude(cls, enhance_threshold_db: float) -> float:
        dbfs = cls.dbfs_threshold(float(enhance_threshold_db))
        return float(10.0 ** (dbfs / 20.0))

    @staticmethod
    def dbfs_threshold(value: float) -> float:
        if value > 0:
            return -float(value)
        return float(value)

    @staticmethod
    def format_optional_dbfs_threshold(value: float | None) -> str:
        if value is None:
            return "off"
        return f"{float(value):+.1f}"

    @staticmethod
    def rms_dbfs(waveform: np.ndarray) -> float:
        waveform = np.asarray(waveform, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        return 20.0 * np.log10(max(rms, DB_EPSILON))


class MicrophoneInput:
    def __init__(self, *, device_index: int, sample_rate: int, channels: int) -> None:
        self.device_index = int(device_index)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._closed = threading.Event()
        self._stream: Any | None = None

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"Audio input status: {status}", file=sys.stderr, flush=True)
        if not self._closed.is_set():
            self._queue.put(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._closed.clear()
        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

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
        self.stop()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def __enter__(self) -> "MicrophoneInput":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
