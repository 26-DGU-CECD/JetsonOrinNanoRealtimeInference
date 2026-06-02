from __future__ import annotations

import numpy as np

from db_threshold_gate import DbThresholdGate
from doa_audio_estimator import DOAAudioEstimator
from doa_reading import DOAReading
from doa_usb_reader import DOAUsbReader


class DOASelector:
    def __init__(
        self,
        *,
        source: str,
        usb_reader: DOAUsbReader,
        audio_estimator: DOAAudioEstimator,
    ) -> None:
        self.source = source
        self.usb_reader = usb_reader
        self.audio_estimator = audio_estimator

    def select(self, chunk: np.ndarray) -> DOAReading:
        if self.source == "audio":
            return self.audio_estimator.estimate(chunk)
        if self.source == "usb":
            return self.usb_reader.snapshot()

        audio_reading = self.audio_estimator.estimate(chunk)
        if audio_reading.raw_angle is not None:
            return audio_reading

        usb_reading = self.usb_reader.snapshot()
        if usb_reading.raw_angle is not None:
            return usb_reading

        return DOAReading(
            None,
            "none",
            f"{audio_reading.status};{usb_reading.status}",
        )

    def status_summary(self) -> str:
        audio_min = DbThresholdGate.format_optional_dbfs_threshold(self.audio_estimator.min_dbfs)
        return (
            f"doa_source={self.source}, audio_doa_min_dbfs={audio_min}, "
            f"usb_doa={self.usb_reader.status}, audio_doa={self.audio_estimator.status}"
        )
