from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DB_EPSILON
from db_threshold_gate import DbThresholdGate


@dataclass(frozen=True)
class EnhancementResult:
    waveform: np.ndarray
    clipped: bool
    quiet_gain: float
    loud_gain: float


class AudioPreprocessor:
    def __init__(
        self,
        enhance_threshold_db: float,
        noise_reduction_db: float,
        main_gain_db: float,
        enhance_sharpness: float,
    ) -> None:
        self.enhance_threshold_db = float(enhance_threshold_db)
        self.noise_reduction_db = float(noise_reduction_db)
        self.main_gain_db = float(main_gain_db)
        self.enhance_sharpness = float(enhance_sharpness)

    def process(self, waveform: np.ndarray) -> EnhancementResult:
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

    @staticmethod
    def _enhance_threshold_amplitude(enhance_threshold_db: float) -> float:
        dbfs = DbThresholdGate.dbfs_threshold(float(enhance_threshold_db))
        return float(10.0 ** (dbfs / 20.0))
