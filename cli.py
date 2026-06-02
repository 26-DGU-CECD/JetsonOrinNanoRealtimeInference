from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from config import (
    DEFAULT_EFFICIENTAT_DIR,
    DEFAULT_ENHANCE_SHARPNESS,
    DEFAULT_ENHANCE_THRESHOLD_DB,
    DEFAULT_MAIN_GAIN_DB,
    DEFAULT_MIN_DB,
    DEFAULT_MIN_SCORE,
    DEFAULT_NOISE_REDUCTION_DB,
    MIC_CHANNEL_INDEX,
)


def parse_optional_db_gate(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"off", "none", "disable", "disabled", "all"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a number, or one of: off, none, disabled, all"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime EfficientAT inference from ReSpeaker Array V3 over BLE, "
            "with audio/USB DOA fields for EdgeAudioRecognition."
        )
    )
    parser.add_argument(
        "--efficientat-dir",
        default=str(Path(DEFAULT_EFFICIENTAT_DIR)),
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
            "Expected maximum BLE notify bytes. App-compatible JSON is sent as one "
            "notification; increase Android MTU or omit --full-packet if packets are too large."
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
        "--doa-source",
        choices=("auto", "audio", "usb"),
        default="auto",
        help=(
            "DOA source. auto uses raw mic audio when available and falls back to "
            "the ReSpeaker USB DSP angle. Default: auto."
        ),
    )
    parser.add_argument(
        "--doa-poll-interval",
        type=float,
        default=0.1,
        help="Seconds between background USB DSP DOA polls. Default: 0.1.",
    )
    parser.add_argument(
        "--audio-doa-min-db",
        type=parse_optional_db_gate,
        default=None,
        help=(
            "Minimum raw mic level for audio-based DOA. Positive values are treated "
            "as dB below full scale, so 45 means -45 dBFS. Use 'off' to calculate "
            "DOA even for quiet raw mic windows. Default: off."
        ),
    )
    parser.add_argument(
        "--audio-doa-window-ms",
        type=float,
        default=250.0,
        help="Loudest raw mic window length used for audio-based DOA. Default: 250.",
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
