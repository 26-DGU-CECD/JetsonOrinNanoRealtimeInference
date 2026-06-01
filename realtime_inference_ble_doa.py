#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torchaudio

import realtime_inference_ble as ble
from realtime_inference import (
    ANSI_GREEN,
    ANSI_RED,
    CHUNK_SAMPLES,
    CHUNK_SECONDS,
    DEFAULT_ENHANCE_SHARPNESS,
    DEFAULT_ENHANCE_THRESHOLD_DB,
    DEFAULT_MAIN_GAIN_DB,
    DEFAULT_MIN_DB,
    DEFAULT_MIN_SCORE,
    DEFAULT_NOISE_REDUCTION_DB,
    MIC_CHANNEL_INDEX,
    MODEL_INPUT_SECONDS,
    MODEL_SAMPLE_RATE,
    REQUIRED_INPUT_CHANNELS,
    SAMPLE_RATE,
    build_custom_label_indices,
    colorize,
    db_gate_threshold,
    enhance_chunk,
    find_respeaker_device,
    format_scores,
    load_efficientat,
    predict_chunk,
    print_input_devices,
    rms_dbfs,
)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )

if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


CARDINAL_SUFFIX = {
    "북": "북쪽",
    "동": "동쪽",
    "남": "남쪽",
    "서": "서쪽",
}

DANGER_LABELS = {
    "gunshot",
    "alarm_siren",
    "horn",
    "glass_shatter",
}

CAUTION_LABELS = {
    "construction",
    "water",
    "knock",
    "appliances",
    "baby_cry",
    "animal_cry",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime EfficientAT inference from ReSpeaker Array V3 over BLE, "
            "with ReSpeaker DOA fields for EdgeAudioRecognition."
        )
    )
    parser.add_argument(
        "--efficientat-dir",
        default=str(Path(__file__).resolve().parent / "EfficientAT"),
        help="Path to cloned fschmid56/EfficientAT repository.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional sounddevice input device index. Defaults to automatic ReSpeaker search.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=MIC_CHANNEL_INDEX,
        help=(
            "Input channel to use. ReSpeaker 4 Mic Array 6-channel firmware is usually "
            "ch0=processed audio, ch1-4=raw microphones, ch5=playback."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input devices and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print waveform, mel, logits, and top AudioSet sigmoid scores for each chunk.",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=DEFAULT_MIN_DB,
        help=(
            "Mark chunks quieter than this level as low signal. Positive values are treated as "
            "dB below full scale, so 30 means -30 dBFS. Use 0 or a negative value "
            "to pass an explicit dBFS threshold."
        ),
    )
    parser.add_argument(
        "--enhance-threshold-db",
        type=float,
        default=DEFAULT_ENHANCE_THRESHOLD_DB,
        help=(
            "Sample-level enhancement threshold. Positive values are treated as "
            "dB below full scale, so 35 means -35 dBFS."
        ),
    )
    parser.add_argument(
        "--noise-reduction-db",
        type=float,
        default=DEFAULT_NOISE_REDUCTION_DB,
        help="Reduce quieter waveform parts by this many dB before inference.",
    )
    parser.add_argument(
        "--main-gain-db",
        type=float,
        default=DEFAULT_MAIN_GAIN_DB,
        help="Boost louder waveform parts by this many dB before inference.",
    )
    parser.add_argument("--gain-db", type=float, dest="main_gain_db", help=argparse.SUPPRESS)
    parser.add_argument(
        "--enhance-sharpness",
        type=float,
        default=DEFAULT_ENHANCE_SHARPNESS,
        help="Higher values separate quiet noise and loud events more aggressively.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Mark predictions whose best custom sigmoid score is below this value as low confidence.",
    )
    parser.add_argument(
        "--ble-name",
        default="JHello",
        help="BLE advertising name. Keep JHello to match the Flutter scanner.",
    )
    parser.add_argument(
        "--ble-chunk-bytes",
        type=int,
        default=244,
        help=(
            "Expected maximum BLE notify bytes. This file sends one App-compatible JSON "
            "notification instead of the framed protocol used by realtime_inference_ble.py."
        ),
    )
    parser.add_argument(
        "--north-offset",
        type=float,
        default=0.0,
        help="Raw DOA angle that should be treated as North. Default: 0.",
    )
    parser.add_argument(
        "--disable-doa",
        action="store_true",
        help="Run BLE inference without trying to read ReSpeaker DOA.",
    )
    parser.add_argument(
        "--db-offset",
        type=float,
        default=80.0,
        help=(
            "Convert internal dBFS to the positive dB value expected by the app: "
            "app_db=max(0, dBFS + offset). Default: 80."
        ),
    )
    parser.add_argument(
        "--full-packet",
        action="store_true",
        help="Include raw/items fields. Requires a BLE MTU large enough for the bigger JSON.",
    )
    return parser.parse_args()


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


class DOAReader:
    def __init__(self, enabled: bool = True):
        self.ok = False
        self.tuning = None
        self.status = "disabled"
        self._lock = threading.Lock()

        if not enabled:
            print("[DOA] disabled by --disable-doa", file=sys.stderr, flush=True)
            return

        try:
            import usb.core  # type: ignore
            from tuning import Tuning  # type: ignore

            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev is None:
                self.status = "device_not_found"
                print("[DOA] ReSpeaker USB control device not found.", file=sys.stderr, flush=True)
                return

            self.tuning = Tuning(dev)
            self.ok = True
            self.status = "enabled"
            print("[DOA] ReSpeaker DOA reader enabled.", file=sys.stderr, flush=True)
        except Exception as exc:
            self.status = "unavailable"
            print(f"[DOA] disabled: {exc!r}", file=sys.stderr, flush=True)

    def read_angle(self) -> int | None:
        if not self.ok or self.tuning is None:
            return None

        try:
            with self._lock:
                angle = self.tuning.direction
            if angle is None:
                return None
            return int(float(angle)) % 360
        except Exception:
            self.status = "read_error"
            return None


class AppInferenceCharacteristic(ble.InferenceCharacteristic):
    def _notify_latest(self) -> bool:
        if not self.notifying:
            return False

        self.sequence += 1
        payload_bytes = self.latest_payload.encode("utf-8")
        if len(payload_bytes) > self.chunk_bytes:
            print(
                f"warning: BLE JSON is {len(payload_bytes)} bytes; "
                f"larger than --ble-chunk-bytes={self.chunk_bytes}. "
                "If the app does not receive packets, increase Android MTU or omit --full-packet.",
                file=sys.stderr,
                flush=True,
            )

        self.PropertiesChanged(
            ble.GATT_CHRC_IFACE,
            ble.dbus.Dictionary({"Value": ble.byte_array(payload_bytes)}, signature="sv"),
            ble.dbus.Array([], signature="s"),
        )
        print(
            f"sent EdgeAudioRecognition notification seq={self.sequence} "
            f"bytes={len(payload_bytes)}",
            flush=True,
        )
        return False


class AppBleInferenceServer(ble.BleInferenceServer):
    def publish(self, data: dict) -> None:
        if self.characteristic is None:
            return
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        self.characteristic.notify_text(payload)


def install_app_compatible_ble_characteristic() -> None:
    ble.InferenceCharacteristic = AppInferenceCharacteristic


def risk_level(label: str) -> str:
    key = str(label).strip().lower()
    if key in DANGER_LABELS:
        return "danger"
    if key in CAUTION_LABELS:
        return "caution"
    return "info"


def app_db_from_dbfs(dbfs: float, offset: float) -> float:
    return round(max(0.0, float(dbfs) + float(offset)), 1)


def build_app_sound_packet(
    *,
    timestamp: str,
    label: str,
    score: float,
    scores: Dict[str, float],
    infer_sec: float,
    total_sec: float,
    chunk_dbfs: float,
    app_db: float,
    status_text: str,
    raw_line: str,
    raw_angle: int | None,
    north_offset: float,
    doa_status: str,
    full_packet: bool,
) -> dict:
    if raw_angle is None:
        angle = 0
        direction = ""
        direction_text = ""
    else:
        angle = corrected_angle(raw_angle, north_offset)
        direction = angle_to_cardinal(angle)
        direction_text = f"{CARDINAL_SUFFIX[direction]} {angle}도"

    packet = {
        "status": "ok",
        "time": timestamp,
        "label": label,
        "score": round(float(score), 6),
        "infer_sec": round(float(infer_sec), 3),
        "total_sec": round(float(total_sec), 3),
        "db": app_db,
        "level": risk_level(label),
        "direction": direction,
        "angle": float(angle),
        "angle_raw": float(raw_angle if raw_angle is not None else 0),
        "direction_text": direction_text,
        "doa_status": doa_status,
    }

    if full_packet:
        packet["display_label"] = label
        packet["dbfs"] = round(float(chunk_dbfs), 2)
        packet["status_text"] = status_text
        packet["raw"] = raw_line
        packet["items"] = [
            {
                "label": item_label,
                "display_label": item_label,
                "score": round(float(item_score), 6),
                "direction": direction,
            }
            for item_label, item_score in scores.items()
        ]

    return packet


def run_stream_ble_doa(
    *,
    device_index: int,
    device_info: dict,
    stream_channels: int,
    channel_index: int,
    model: torch.nn.Module,
    mel: torch.nn.Module,
    resampler: torch.nn.Module | None,
    custom_indices: Dict[str, List[int]],
    audioset_labels: Sequence[str],
    device: torch.device,
    debug: bool,
    min_db: float,
    enhance_threshold_db: float,
    noise_reduction_db: float,
    main_gain_db: float,
    enhance_sharpness: float,
    min_score: float,
    ble_server: AppBleInferenceServer,
    doa_reader: DOAReader,
    north_offset: float,
    db_offset: float,
    full_packet: bool,
) -> None:
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    if channel_index < 0 or channel_index >= stream_channels:
        raise RuntimeError(
            f"Selected channel index is out of range: "
            f"channel={channel_index}, available=0..{stream_channels - 1}"
        )

    if stream_channels < REQUIRED_INPUT_CHANNELS:
        print(
            f"Warning: selected input device reports only {stream_channels} input channels. "
            "Continuing with the available channel.",
            file=sys.stderr,
            flush=True,
        )

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"Audio input status: {status}", file=sys.stderr, flush=True)
        audio_queue.put(indata.copy())

    print(
        f"Input device: [{device_index}] {device_info.get('name')} | "
        f"channels={stream_channels}, mic_sr={SAMPLE_RATE}, "
        f"model_sr={MODEL_SAMPLE_RATE}, chunk={CHUNK_SECONDS}s, "
        f"model_input={MODEL_INPUT_SECONDS}s, channel={channel_index}, "
        f"min_dbfs={db_gate_threshold(min_db):+.1f}, "
        f"enhance_threshold_dbfs={db_gate_threshold(enhance_threshold_db):+.1f}, "
        f"noise_reduction_db={noise_reduction_db:.1f}, main_gain_db={main_gain_db:+.1f}, "
        f"min_score={min_score:.1%}, doa={doa_reader.status}"
    )
    print("Ctrl+C to stop.")

    pending_blocks: List[np.ndarray] = []
    pending_samples = 0

    with ble.sd.InputStream(
        device=device_index,
        samplerate=SAMPLE_RATE,
        channels=stream_channels,
        dtype="float32",
        callback=callback,
    ):
        while True:
            block = audio_queue.get()
            mono = block[:, channel_index].astype(np.float32, copy=True)
            pending_blocks.append(mono)
            pending_samples += mono.shape[0]

            if pending_samples < CHUNK_SAMPLES:
                continue

            joined = np.concatenate(pending_blocks)
            offset = 0
            while joined.shape[0] - offset >= CHUNK_SAMPLES:
                chunk_started = time.perf_counter()
                chunk = joined[offset: offset + CHUNK_SAMPLES]
                offset += CHUNK_SAMPLES

                timestamp = datetime.now().strftime("%H:%M:%S")
                chunk_dbfs = rms_dbfs(chunk)
                min_dbfs = db_gate_threshold(min_db)

                inference_chunk, clipped, quiet_gain, loud_gain = enhance_chunk(
                    chunk,
                    enhance_threshold_db,
                    noise_reduction_db,
                    main_gain_db,
                    enhance_sharpness,
                )
                enhanced_dbfs = rms_dbfs(inference_chunk)

                try:
                    infer_started = time.perf_counter()
                    best_label, best_probability, scores = predict_chunk(
                        inference_chunk,
                        model,
                        mel,
                        resampler,
                        custom_indices,
                        audioset_labels,
                        device,
                        debug=debug,
                    )
                    infer_sec = time.perf_counter() - infer_started
                except Exception as exc:
                    print(
                        f"[{timestamp}] inference error: {exc} | skipping chunk",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                status_reasons = []
                if chunk_dbfs < min_dbfs:
                    status_reasons.append(f"low_signal {chunk_dbfs:+.1f}<{min_dbfs:+.1f}dBFS")
                if best_probability < min_score:
                    status_reasons.append(f"low_score {best_probability:.1%}<{min_score:.1%}")

                if status_reasons:
                    status_text = "low(" + ", ".join(status_reasons) + ")"
                    line_color = ANSI_RED
                else:
                    status_text = "detected"
                    line_color = ANSI_GREEN

                raw_angle = doa_reader.read_angle()
                angle = corrected_angle(raw_angle, north_offset) if raw_angle is not None else None
                direction = angle_to_cardinal(angle) if angle is not None else ""
                doa_text = f" | DOA={direction} {angle}deg raw={raw_angle}" if angle is not None else ""

                line = (
                    f"[{timestamp}] predict: {best_label} ({best_probability:.1%}) | "
                    f"status={status_text} | "
                    f"level={chunk_dbfs:+.1f} dBFS | "
                    f"app_db={app_db_from_dbfs(chunk_dbfs, db_offset):.1f} dB | "
                    f"enhanced={enhanced_dbfs:+.1f} dBFS | "
                    f"infer={infer_sec:.3f}s | "
                    f"quiet_gain={quiet_gain:.2f}x loud_gain={loud_gain:.2f}x"
                    f"{' clipped' if clipped else ''}{doa_text} | all: {format_scores(scores)}"
                )
                print(colorize(line, line_color), flush=True)

                total_sec = CHUNK_SECONDS + (time.perf_counter() - chunk_started)
                ble_server.publish(
                    build_app_sound_packet(
                        timestamp=timestamp,
                        label=best_label,
                        score=best_probability,
                        scores=scores,
                        infer_sec=infer_sec,
                        total_sec=total_sec,
                        chunk_dbfs=chunk_dbfs,
                        app_db=app_db_from_dbfs(chunk_dbfs, db_offset),
                        status_text=status_text,
                        raw_line=line,
                        raw_angle=raw_angle,
                        north_offset=north_offset,
                        doa_status=doa_reader.status,
                        full_packet=full_packet,
                    )
                )

            remainder = joined[offset:]
            pending_blocks = [remainder] if remainder.size else []
            pending_samples = remainder.shape[0]


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    try:
        device_index, device_info, stream_channels = find_respeaker_device(args.device_index)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        print_input_devices()
        return 1

    install_app_compatible_ble_characteristic()
    ble_server = AppBleInferenceServer(args.ble_name, args.ble_chunk_bytes)
    try:
        ble_server.start()
    except Exception as exc:
        print(f"BLE startup error: {exc}", file=sys.stderr, flush=True)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    resampler = None
    if SAMPLE_RATE != MODEL_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(device).eval()

    try:
        model, mel, audioset_labels = load_efficientat(Path(args.efficientat_dir), device)
        custom_indices = build_custom_label_indices(audioset_labels)
    except Exception as exc:
        print(f"Model initialization error: {exc}", file=sys.stderr, flush=True)
        ble_server.stop()
        return 1

    doa_reader = DOAReader(enabled=not args.disable_doa)
    stop_requested = False

    def stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        run_stream_ble_doa(
            device_index=device_index,
            device_info=device_info,
            stream_channels=stream_channels,
            channel_index=args.channel_index,
            model=model,
            mel=mel,
            resampler=resampler,
            custom_indices=custom_indices,
            audioset_labels=audioset_labels,
            device=device,
            debug=args.debug,
            min_db=args.min_db,
            enhance_threshold_db=args.enhance_threshold_db,
            noise_reduction_db=args.noise_reduction_db,
            main_gain_db=args.main_gain_db,
            enhance_sharpness=args.enhance_sharpness,
            min_score=args.min_score,
            ble_server=ble_server,
            doa_reader=doa_reader,
            north_offset=args.north_offset,
            db_offset=args.db_offset,
            full_packet=args.full_packet,
        )
    except KeyboardInterrupt:
        print("\nStopping.")
        return 0
    except Exception as exc:
        print(f"Audio stream error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if stop_requested:
            print("Stopping BLE server...", flush=True)
        ble_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
