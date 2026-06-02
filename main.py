#!/usr/bin/env python3
from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Sequence

from cli import parse_args
from io_setup import configure_stdio


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)

    from audio_device_finder import AudioDeviceFinder

    device_finder = AudioDeviceFinder()
    if args.list_devices:
        device_finder.print_input_devices()
        return 0

    try:
        audio_device = device_finder.find(args.device_index)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        device_finder.print_input_devices()
        return 1

    try:
        from ble_inference_server import BleInferenceServer

        ble_server = BleInferenceServer(
            args.ble_name,
            args.ble_chunk_bytes,
            app_compatible=True,
        )
        ble_server.start()
    except Exception as exc:
        print(f"BLE startup error: {exc}", file=sys.stderr, flush=True)
        return 1

    controller = None
    doa_manager = None
    stop_requested = False

    try:
        from app_packet_builder import AppPacketBuilder
        from audio_stream_controller import AudioStreamController, AudioStreamSettings
        from config import MODEL_INPUT_SECONDS, MODEL_SAMPLE_RATE, SAMPLE_RATE
        from doa import DOAManager
        from sound_classifier import SoundClassifier

        device = SoundClassifier.select_device()
        print(f"Inference device: {device}")
        classifier = SoundClassifier.from_efficientat(
            Path(args.efficientat_dir),
            device=device,
            debug=args.debug,
        )
        doa_manager = DOAManager(
            source=args.doa_source,
            disabled=args.disable_doa,
            stream_channels=audio_device.stream_channels,
            sample_rate=SAMPLE_RATE,
            usb_poll_interval=args.doa_poll_interval,
            audio_min_db=args.audio_doa_min_db,
            audio_window_ms=args.audio_doa_window_ms,
        )
        packet_builder = AppPacketBuilder(
            north_offset=args.north_offset,
            db_offset=args.db_offset,
            full_packet=args.full_packet,
        )
        settings = AudioStreamSettings(
            device_index=audio_device.index,
            device_info=audio_device.info,
            stream_channels=audio_device.stream_channels,
            channel_index=args.channel_index,
            model_sample_rate=MODEL_SAMPLE_RATE,
            model_input_seconds=MODEL_INPUT_SECONDS,
            debug=args.debug,
            min_db=args.min_db,
            enhance_threshold_db=args.enhance_threshold_db,
            noise_reduction_db=args.noise_reduction_db,
            main_gain_db=args.main_gain_db,
            enhance_sharpness=args.enhance_sharpness,
            min_score=args.min_score,
        )
        controller = AudioStreamController(
            settings=settings,
            classifier=classifier,
            doa_manager=doa_manager,
            packet_builder=packet_builder,
            ble_server=ble_server,
        )
    except Exception as exc:
        print(f"Initialization error: {exc}", file=sys.stderr, flush=True)
        if doa_manager is not None:
            doa_manager.stop()
        ble_server.stop()
        return 1

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        if controller is not None:
            controller.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        controller.run_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        return 0
    except Exception as exc:
        print(f"Audio stream error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if stop_requested:
            print("Stopping services...", flush=True)
        controller.stop()
        if doa_manager is not None:
            doa_manager.stop()
        ble_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
