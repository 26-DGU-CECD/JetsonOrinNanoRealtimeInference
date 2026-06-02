from __future__ import annotations

import sys
from typing import Any

import sounddevice as sd

from audio_queue import AudioQueue


class MicrophoneModule:
    def __init__(
        self,
        *,
        device_index: int,
        sample_rate: int,
        channels: int,
        audio_queue: AudioQueue,
    ) -> None:
        self.device_index = int(device_index)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.audio_queue = audio_queue
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"Audio input status: {status}", file=sys.stderr, flush=True)
        self.audio_queue.put(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def __enter__(self) -> "MicrophoneModule":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
