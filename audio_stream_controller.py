from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from app_packet_builder import AppPacketBuilder
from audio_buffer import AudioBuffer
from audio_level_meter import AudioLevelMeter
from audio_preprocessor import AudioPreprocessor
from audio_queue import AudioQueue
from config import (
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    CHUNK_SECONDS,
    REQUIRED_INPUT_CHANNELS,
    SAMPLE_RATE,
)
from db_threshold_gate import DbThresholdGate
from direction_utils import DirectionUtils
from doa_selector import DOASelector
from microphone_module import MicrophoneModule
from sound_classifier import SoundClassifier


@dataclass(frozen=True)
class AudioStreamSettings:
    device_index: int
    device_info: dict[str, Any]
    stream_channels: int
    channel_index: int
    model_sample_rate: int
    model_input_seconds: int
    debug: bool
    min_db: float
    enhance_threshold_db: float
    noise_reduction_db: float
    main_gain_db: float
    enhance_sharpness: float
    min_score: float


class AudioStreamController:
    def __init__(
        self,
        *,
        settings: AudioStreamSettings,
        classifier: SoundClassifier,
        doa_selector: DOASelector,
        packet_builder: AppPacketBuilder,
        ble_server: Any,
    ) -> None:
        self.settings = settings
        self.classifier = classifier
        self.doa_selector = doa_selector
        self.packet_builder = packet_builder
        self.ble_server = ble_server
        self.audio_queue = AudioQueue()
        self.audio_buffer = AudioBuffer()
        self.level_meter = AudioLevelMeter()
        self.threshold_gate = DbThresholdGate(settings.min_db)
        self.preprocessor = AudioPreprocessor(
            settings.enhance_threshold_db,
            settings.noise_reduction_db,
            settings.main_gain_db,
            settings.enhance_sharpness,
        )
        self.microphone = MicrophoneModule(
            device_index=settings.device_index,
            sample_rate=SAMPLE_RATE,
            channels=settings.stream_channels,
            audio_queue=self.audio_queue,
        )
        self._stop_event = threading.Event()

    def run_forever(self) -> None:
        self._validate_settings()
        self._print_startup_summary()
        print("Ctrl+C to stop.")

        with self.microphone:
            while not self._stop_event.is_set():
                block = self.audio_queue.get(timeout=0.2)
                if block is None:
                    continue
                for chunk_multi in self.audio_buffer.add_block(block):
                    self._process_chunk(chunk_multi)

    def stop(self) -> None:
        self._stop_event.set()
        self.audio_queue.close()
        self.microphone.stop()

    def _validate_settings(self) -> None:
        if self.settings.channel_index < 0 or self.settings.channel_index >= self.settings.stream_channels:
            raise RuntimeError(
                "Selected channel index is out of range: "
                f"channel={self.settings.channel_index}, "
                f"available=0..{self.settings.stream_channels - 1}"
            )

        if self.settings.stream_channels < REQUIRED_INPUT_CHANNELS:
            print(
                f"Warning: selected input device reports only {self.settings.stream_channels} "
                "input channels. Continuing with the available channel.",
                file=sys.stderr,
                flush=True,
            )

    def _print_startup_summary(self) -> None:
        print(
            f"Input device: [{self.settings.device_index}] {self.settings.device_info.get('name')} | "
            f"channels={self.settings.stream_channels}, mic_sr={SAMPLE_RATE}, "
            f"model_sr={self.settings.model_sample_rate}, chunk={CHUNK_SECONDS}s, "
            f"model_input={self.settings.model_input_seconds}s, channel={self.settings.channel_index}, "
            f"min_dbfs={self.threshold_gate.threshold_dbfs:+.1f}, "
            f"enhance_threshold_dbfs={DbThresholdGate.dbfs_threshold(self.settings.enhance_threshold_db):+.1f}, "
            f"noise_reduction_db={self.settings.noise_reduction_db:.1f}, "
            f"main_gain_db={self.settings.main_gain_db:+.1f}, "
            f"min_score={self.settings.min_score:.1%}, "
            f"{self.doa_selector.status_summary()}"
        )

    def _process_chunk(self, chunk_multi: np.ndarray) -> None:
        chunk_started = time.perf_counter()
        chunk = chunk_multi[:, self.settings.channel_index].astype(np.float32, copy=True)

        timestamp = datetime.now().strftime("%H:%M:%S")
        chunk_dbfs = self.level_meter.rms_dbfs(chunk)
        enhancement = self.preprocessor.process(chunk)
        enhanced_dbfs = self.level_meter.rms_dbfs(enhancement.waveform)

        try:
            infer_started = time.perf_counter()
            classification = self.classifier.predict(enhancement.waveform)
            infer_sec = time.perf_counter() - infer_started
        except Exception as exc:
            print(
                f"[{timestamp}] inference error: {exc} | skipping chunk",
                file=sys.stderr,
                flush=True,
            )
            return

        status_reasons = []
        if self.threshold_gate.is_low(chunk_dbfs):
            status_reasons.append(
                f"low_signal {chunk_dbfs:+.1f}<{self.threshold_gate.threshold_dbfs:+.1f}dBFS"
            )
        if classification.best_score < self.settings.min_score:
            status_reasons.append(
                f"low_score {classification.best_score:.1%}<{self.settings.min_score:.1%}"
            )

        if status_reasons:
            status_text = "low(" + ", ".join(status_reasons) + ")"
            line_color = ANSI_RED
        else:
            status_text = "detected"
            line_color = ANSI_GREEN

        doa_reading = self.doa_selector.select(chunk_multi)
        raw_angle = doa_reading.raw_angle
        angle = (
            DirectionUtils.corrected_angle(raw_angle, self.packet_builder.north_offset)
            if raw_angle is not None
            else None
        )
        direction = DirectionUtils.angle_to_cardinal(angle) if angle is not None else ""
        if angle is None:
            doa_text = (
                f" | DOA=unavailable source={doa_reading.source} "
                f"status={doa_reading.status}"
            )
        else:
            doa_text = (
                f" | DOA={direction} {angle}deg raw={raw_angle} "
                f"source={doa_reading.source} status={doa_reading.status}"
            )

        app_db = self.packet_builder.app_db_from_dbfs(chunk_dbfs)
        line = (
            f"[{timestamp}] predict: {classification.best_label} ({classification.best_score:.1%}) | "
            f"status={status_text} | "
            f"level={chunk_dbfs:+.1f} dBFS | "
            f"app_db={app_db:.1f} dB | "
            f"enhanced={enhanced_dbfs:+.1f} dBFS | "
            f"infer={infer_sec:.3f}s | "
            f"quiet_gain={enhancement.quiet_gain:.2f}x loud_gain={enhancement.loud_gain:.2f}x"
            f"{' clipped' if enhancement.clipped else ''}{doa_text} | "
            f"all: {self._format_scores(classification.scores)}"
        )
        print(self._colorize(line, line_color), flush=True)

        total_sec = CHUNK_SECONDS + (time.perf_counter() - chunk_started)
        self.ble_server.publish(
            self.packet_builder.build(
                timestamp=timestamp,
                label=classification.best_label,
                score=classification.best_score,
                scores=classification.scores,
                infer_sec=infer_sec,
                total_sec=total_sec,
                chunk_dbfs=chunk_dbfs,
                status_text=status_text,
                raw_line=line,
                raw_angle=raw_angle,
                doa_status=doa_reading.status,
                doa_source=doa_reading.source,
            )
        )

    @staticmethod
    def _format_scores(scores: dict[str, float]) -> str:
        return ", ".join(f"{label}={probability:.1%}" for label, probability in scores.items())

    @staticmethod
    def _colorize(text: str, color_code: str) -> str:
        if not sys.stdout.isatty():
            return text
        return f"{color_code}{text}{ANSI_RESET}"
