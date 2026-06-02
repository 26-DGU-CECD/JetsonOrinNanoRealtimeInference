from __future__ import annotations

import numpy as np

from config import DB_EPSILON


class AudioLevelMeter:
    @staticmethod
    def rms_dbfs(waveform: np.ndarray) -> float:
        waveform = np.asarray(waveform, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        return 20.0 * np.log10(max(rms, DB_EPSILON))
