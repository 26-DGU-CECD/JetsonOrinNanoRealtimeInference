from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from audio_processing import AudioProcessor
from config import (
    CARDINAL_SUFFIX,
    RAW_DOA_CHANNELS,
    RAW_DOA_MIC_POSITIONS_M,
    RESPEAKER_USB_PRODUCT_ID,
    RESPEAKER_USB_VENDOR_ID,
    SPEED_OF_SOUND_M_S,
)


@dataclass(frozen=True)
class DOAReading:
    raw_angle: int | None
    source: str
    status: str


def angle_to_cardinal(angle: float) -> str:
    corrected = float(angle) % 360.0
    if corrected < 45.0 or corrected >= 315.0:
        return "북"
    if corrected < 135.0:
        return "동"
    if corrected < 225.0:
        return "남"
    return "서"


def corrected_angle(raw_angle: float, north_offset: float) -> int:
    return int(round((float(raw_angle) - float(north_offset)) % 360.0)) % 360


def direction_text(angle: int | None) -> tuple[str, str]:
    if angle is None:
        return "", ""
    direction = angle_to_cardinal(angle)
    return direction, f"{CARDINAL_SUFFIX[direction]} {angle}도"


class DOAManager:
    def __init__(
        self,
        *,
        source: str,
        disabled: bool,
        stream_channels: int,
        sample_rate: int,
        usb_poll_interval: float,
        audio_min_db: float | None,
        audio_window_ms: float,
    ) -> None:
        self.source = source
        self.stream_channels = int(stream_channels)
        self.sample_rate = int(sample_rate)
        self.usb_poll_interval = max(0.02, float(usb_poll_interval))

        self.usb_ok = False
        self.usb_tuning = None
        self.usb_status = "disabled"
        self._usb_lock = threading.Lock()
        self._usb_stop_event = threading.Event()
        self._usb_thread: threading.Thread | None = None
        self._usb_last_angle: int | None = None
        self._usb_last_voice: bool | None = None
        self._usb_last_read_at: float | None = None
        self._usb_last_error: str | None = None

        self.audio_enabled = not disabled and source in ("auto", "audio")
        self.audio_min_dbfs = None if audio_min_db is None else AudioProcessor.dbfs_threshold(audio_min_db)
        self.audio_window_samples = max(
            256,
            int(round(self.sample_rate * max(20.0, float(audio_window_ms)) / 1000.0)),
        )
        self.audio_status = "disabled"
        self.audio_channel_indices = tuple(RAW_DOA_CHANNELS)
        self.audio_mic_positions = np.asarray(RAW_DOA_MIC_POSITIONS_M, dtype=np.float32)
        self.audio_pairs = [
            (left, right)
            for left in range(len(self.audio_channel_indices))
            for right in range(left + 1, len(self.audio_channel_indices))
        ]
        self.audio_expected_taus = self._build_expected_taus()
        self._configure_audio()

        usb_enabled = not disabled and source in ("auto", "usb")
        if usb_enabled:
            self._start_usb()
        else:
            reason = (
                "disabled by --disable-doa"
                if disabled
                else "disabled because --doa-source=audio"
            )
            print(f"[DOA] USB reader {reason}", file=sys.stderr, flush=True)

    def _configure_audio(self) -> None:
        if not self.audio_enabled:
            return
        if self.stream_channels <= max(self.audio_channel_indices):
            self.audio_status = "audio_channels_unavailable"
            return
        self.audio_status = "audio_enabled"

    def _start_usb(self) -> None:
        try:
            import usb.core  # type: ignore
            from tuning import Tuning  # type: ignore

            dev = usb.core.find(
                idVendor=RESPEAKER_USB_VENDOR_ID,
                idProduct=RESPEAKER_USB_PRODUCT_ID,
            )
            if dev is None:
                self.usb_status = "device_not_found"
                print("[DOA] ReSpeaker USB control device not found.", file=sys.stderr, flush=True)
                return

            self.usb_tuning = Tuning(dev)
            self.usb_ok = True
            self.usb_status = "enabled"
            self._poll_usb_once()
            self._usb_thread = threading.Thread(target=self._poll_usb_loop, daemon=True)
            self._usb_thread.start()
            if self.usb_status == "enabled":
                print("[DOA] ReSpeaker USB DOA reader enabled.", file=sys.stderr, flush=True)
            else:
                usb_status = self.usb_status
                if self._usb_last_error:
                    usb_status = f"{usb_status}:{self._usb_last_error}"
                print(
                    f"[DOA] ReSpeaker USB control found, but reads failed: {usb_status}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            self.usb_status = "unavailable"
            print(f"[DOA] USB reader disabled: {exc!r}", file=sys.stderr, flush=True)

    def select(self, chunk: np.ndarray) -> DOAReading:
        if self.source == "audio":
            return self._estimate_audio(chunk)
        if self.source == "usb":
            return self._snapshot_usb()

        audio_reading = self._estimate_audio(chunk)
        if audio_reading.raw_angle is not None:
            return audio_reading

        usb_reading = self._snapshot_usb()
        if usb_reading.raw_angle is not None:
            return usb_reading

        return DOAReading(
            None,
            "none",
            f"{audio_reading.status};{usb_reading.status}",
        )

    def status_summary(self) -> str:
        audio_min = AudioProcessor.format_optional_dbfs_threshold(self.audio_min_dbfs)
        return (
            f"doa_source={self.source}, audio_doa_min_dbfs={audio_min}, "
            f"usb_doa={self.usb_status}, audio_doa={self.audio_status}"
        )

    def read_angle(self) -> int | None:
        return self._snapshot_usb().raw_angle

    def stop(self) -> None:
        self._usb_stop_event.set()
        if self._usb_thread is not None:
            self._usb_thread.join(timeout=1.0)
            self._usb_thread = None
        if self.usb_tuning is not None:
            try:
                self.usb_tuning.close()
            except Exception:
                pass
            self.usb_tuning = None

    def _read_usb_device_locked(self) -> tuple[int | None, bool | None]:
        if self.usb_tuning is None:
            return None, None

        voice = None
        try:
            voice_value = self.usb_tuning.is_voice()
            if voice_value is not None:
                voice = bool(int(voice_value))
        except Exception:
            voice = None

        angle = self.usb_tuning.direction
        if angle is None:
            return None, voice
        return int(float(angle)) % 360, voice

    def _poll_usb_once(self) -> None:
        if not self.usb_ok or self.usb_tuning is None:
            return

        try:
            with self._usb_lock:
                angle, voice = self._read_usb_device_locked()
            self._usb_last_read_at = time.monotonic()
            self._usb_last_voice = voice
            if angle is not None:
                self._usb_last_angle = angle
            self.usb_status = "enabled"
            self._usb_last_error = None
        except Exception as exc:
            self.usb_status = "read_error"
            self._usb_last_error = type(exc).__name__

    def _poll_usb_loop(self) -> None:
        while not self._usb_stop_event.wait(self.usb_poll_interval):
            self._poll_usb_once()

    def _snapshot_usb(self) -> DOAReading:
        if not self.usb_ok:
            return DOAReading(None, "usb", self.usb_status)

        self._poll_usb_once()
        angle = self._usb_last_angle
        if angle is None:
            status = self.usb_status
            if self._usb_last_error:
                status = f"{status}:{self._usb_last_error}"
            if status == "enabled":
                status = "usb_no_angle"
            return DOAReading(None, "usb", status)

        now = time.monotonic()
        age = None if self._usb_last_read_at is None else now - self._usb_last_read_at
        if age is not None and age > max(1.0, self.usb_poll_interval * 5.0):
            return DOAReading(None, "usb", "usb_stale")

        status = "usb_no_voice" if self._usb_last_voice is False else "usb_active"
        return DOAReading(angle, "usb", status)

    def _build_expected_taus(self) -> np.ndarray:
        compass_degrees = np.arange(360, dtype=np.float32)
        radians = np.deg2rad(compass_degrees)
        directions = np.stack((np.sin(radians), np.cos(radians)), axis=1)
        expected = []
        for left, right in self.audio_pairs:
            delta = self.audio_mic_positions[left] - self.audio_mic_positions[right]
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
        if audio.shape[0] <= self.audio_window_samples:
            return audio

        step = max(128, self.audio_window_samples // 4)
        best_start = 0
        best_power = -1.0
        last_start = audio.shape[0] - self.audio_window_samples
        for start in range(0, last_start + 1, step):
            window = audio[start : start + self.audio_window_samples]
            power = float(np.mean(np.square(window)))
            if power > best_power:
                best_power = power
                best_start = start
        return audio[best_start : best_start + self.audio_window_samples]

    def _estimate_audio(self, chunk: np.ndarray) -> DOAReading:
        if not self.audio_enabled:
            return DOAReading(None, "audio", "audio_disabled")
        if self.audio_status == "audio_channels_unavailable":
            return DOAReading(None, "audio", self.audio_status)
        if chunk.ndim != 2 or chunk.shape[1] <= max(self.audio_channel_indices):
            self.audio_status = "audio_channels_unavailable"
            return DOAReading(None, "audio", self.audio_status)

        raw = chunk[:, self.audio_channel_indices].astype(np.float32, copy=True)
        raw = self._select_loudest_window(raw)
        raw_dbfs = AudioProcessor.rms_dbfs(raw.reshape(-1))
        if self.audio_min_dbfs is not None and raw_dbfs < self.audio_min_dbfs:
            self.audio_status = f"audio_low_signal {raw_dbfs:+.1f}dBFS"
            return DOAReading(None, "audio", self.audio_status)

        raw -= np.mean(raw, axis=0, keepdims=True)
        channel_rms = np.sqrt(np.mean(np.square(raw), axis=0) + 1e-12)
        usable_channels = channel_rms > 1e-5
        if int(np.count_nonzero(usable_channels)) < 2:
            self.audio_status = "audio_not_enough_active_channels"
            return DOAReading(None, "audio", self.audio_status)

        raw /= channel_rms.reshape(1, -1)
        raw *= np.hanning(raw.shape[0]).astype(np.float32).reshape(-1, 1)

        measured_taus = []
        pair_indices = []
        for pair_index, (left, right) in enumerate(self.audio_pairs):
            if not (usable_channels[left] and usable_channels[right]):
                continue
            max_tau = (
                float(np.linalg.norm(self.audio_mic_positions[left] - self.audio_mic_positions[right]))
                / SPEED_OF_SOUND_M_S
            )
            tau = self._gcc_phat(raw[:, left], raw[:, right], max_tau=max_tau)
            if tau is None:
                continue
            measured_taus.append(tau)
            pair_indices.append(pair_index)

        if len(measured_taus) < 2:
            self.audio_status = "audio_not_enough_tdoa_pairs"
            return DOAReading(None, "audio", self.audio_status)

        measured = np.asarray(measured_taus, dtype=np.float32)
        expected = self.audio_expected_taus[:, pair_indices]
        errors = np.mean(np.square(expected - measured.reshape(1, -1)), axis=1)
        angle = int(np.argmin(errors)) % 360
        error_us = float(np.sqrt(np.min(errors)) * 1_000_000.0)
        self.audio_status = f"audio_active {raw_dbfs:+.1f}dBFS err={error_us:.1f}us"
        return DOAReading(angle, "audio", self.audio_status)
