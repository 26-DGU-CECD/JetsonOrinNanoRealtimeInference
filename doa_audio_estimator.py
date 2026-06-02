from __future__ import annotations

import numpy as np

from audio_level_meter import AudioLevelMeter
from config import RAW_DOA_CHANNELS, RAW_DOA_MIC_POSITIONS_M, SPEED_OF_SOUND_M_S
from db_threshold_gate import DbThresholdGate
from doa_reading import DOAReading


class DOAAudioEstimator:
    def __init__(
        self,
        *,
        enabled: bool,
        stream_channels: int,
        sample_rate: int,
        min_db: float | None,
        window_ms: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.stream_channels = int(stream_channels)
        self.sample_rate = int(sample_rate)
        self.min_dbfs = None if min_db is None else DbThresholdGate.dbfs_threshold(min_db)
        self.window_samples = max(
            256,
            int(round(self.sample_rate * max(20.0, float(window_ms)) / 1000.0)),
        )
        self.status = "disabled"
        self.channel_indices = tuple(RAW_DOA_CHANNELS)
        self.mic_positions = np.asarray(RAW_DOA_MIC_POSITIONS_M, dtype=np.float32)
        self.pairs = [
            (left, right)
            for left in range(len(self.channel_indices))
            for right in range(left + 1, len(self.channel_indices))
        ]
        self.expected_taus = self._build_expected_taus()

        if not self.enabled:
            return
        if self.stream_channels <= max(self.channel_indices):
            self.status = "audio_channels_unavailable"
            return
        self.status = "audio_enabled"

    def _build_expected_taus(self) -> np.ndarray:
        compass_degrees = np.arange(360, dtype=np.float32)
        radians = np.deg2rad(compass_degrees)
        directions = np.stack((np.sin(radians), np.cos(radians)), axis=1)
        expected = []
        for left, right in self.pairs:
            delta = self.mic_positions[left] - self.mic_positions[right]
            expected.append(-(directions @ delta) / SPEED_OF_SOUND_M_S)
        return np.stack(expected, axis=1).astype(np.float32)

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        result = 1
        while result < value:
            result <<= 1
        return result

    def _gcc_phat(
        self,
        sig: np.ndarray,
        refsig: np.ndarray,
        *,
        max_tau: float,
        interp: int = 16,
    ) -> float | None:
        n = self._next_power_of_two(sig.size + refsig.size)
        sig_fft = np.fft.rfft(sig, n=n)
        ref_fft = np.fft.rfft(refsig, n=n)
        cross_power = sig_fft * np.conj(ref_fft)
        cross_power /= np.abs(cross_power) + 1e-12
        cc = np.fft.irfft(cross_power, n=interp * n)

        max_shift = min(int(round(interp * self.sample_rate * max_tau)), (interp * n) // 2)
        if max_shift < 1:
            return None

        cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
        shift = int(np.argmax(cc)) - max_shift
        return float(shift) / float(interp * self.sample_rate)

    def _select_loudest_window(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] <= self.window_samples:
            return audio

        step = max(128, self.window_samples // 4)
        best_start = 0
        best_power = -1.0
        last_start = audio.shape[0] - self.window_samples
        for start in range(0, last_start + 1, step):
            window = audio[start : start + self.window_samples]
            power = float(np.mean(np.square(window)))
            if power > best_power:
                best_power = power
                best_start = start
        return audio[best_start : best_start + self.window_samples]

    def estimate(self, chunk: np.ndarray) -> DOAReading:
        if not self.enabled:
            return DOAReading(None, "audio", "audio_disabled")
        if self.status == "audio_channels_unavailable":
            return DOAReading(None, "audio", self.status)
        if chunk.ndim != 2 or chunk.shape[1] <= max(self.channel_indices):
            self.status = "audio_channels_unavailable"
            return DOAReading(None, "audio", self.status)

        raw = chunk[:, self.channel_indices].astype(np.float32, copy=True)
        raw = self._select_loudest_window(raw)
        raw_dbfs = AudioLevelMeter.rms_dbfs(raw.reshape(-1))
        if self.min_dbfs is not None and raw_dbfs < self.min_dbfs:
            self.status = f"audio_low_signal {raw_dbfs:+.1f}dBFS"
            return DOAReading(None, "audio", self.status)

        raw -= np.mean(raw, axis=0, keepdims=True)
        channel_rms = np.sqrt(np.mean(np.square(raw), axis=0) + 1e-12)
        usable_channels = channel_rms > 1e-5
        if int(np.count_nonzero(usable_channels)) < 2:
            self.status = "audio_not_enough_active_channels"
            return DOAReading(None, "audio", self.status)

        raw /= channel_rms.reshape(1, -1)
        raw *= np.hanning(raw.shape[0]).astype(np.float32).reshape(-1, 1)

        measured_taus = []
        pair_indices = []
        for pair_index, (left, right) in enumerate(self.pairs):
            if not (usable_channels[left] and usable_channels[right]):
                continue
            max_tau = (
                float(np.linalg.norm(self.mic_positions[left] - self.mic_positions[right]))
                / SPEED_OF_SOUND_M_S
            )
            tau = self._gcc_phat(raw[:, left], raw[:, right], max_tau=max_tau)
            if tau is None:
                continue
            measured_taus.append(tau)
            pair_indices.append(pair_index)

        if len(measured_taus) < 2:
            self.status = "audio_not_enough_tdoa_pairs"
            return DOAReading(None, "audio", self.status)

        measured = np.asarray(measured_taus, dtype=np.float32)
        expected = self.expected_taus[:, pair_indices]
        errors = np.mean(np.square(expected - measured.reshape(1, -1)), axis=1)
        angle = int(np.argmin(errors)) % 360
        error_us = float(np.sqrt(np.min(errors)) * 1_000_000.0)
        self.status = f"audio_active {raw_dbfs:+.1f}dBFS err={error_us:.1f}us"
        return DOAReading(angle, "audio", self.status)
