from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sounddevice as sd

from config import MIC_CHANNEL_INDEX, MIC_NAME_KEYWORDS, REQUIRED_INPUT_CHANNELS, SAMPLE_RATE


@dataclass(frozen=True)
class AudioDeviceSelection:
    index: int
    info: dict[str, Any]
    stream_channels: int


class AudioDeviceFinder:
    def input_devices(self) -> list[tuple[int, dict[str, Any]]]:
        return [
            (index, dict(device))
            for index, device in enumerate(sd.query_devices())
            if int(device.get("max_input_channels", 0)) > 0
        ]

    def print_input_devices(self) -> None:
        devices = self.input_devices()
        if not devices:
            print("No available input devices.")
            return

        print("Available input devices:")
        for index, device in devices:
            print(
                f"  [{index}] {device.get('name')} | "
                f"inputs={device.get('max_input_channels')} | "
                f"default_sr={device.get('default_samplerate')}"
            )

    def find(self, device_index: int | None = None) -> AudioDeviceSelection:
        devices = self.input_devices()

        if device_index is not None:
            for index, device in devices:
                if index == device_index:
                    channels = min(REQUIRED_INPUT_CHANNELS, int(device["max_input_channels"]))
                    return AudioDeviceSelection(index, device, channels)
            raise RuntimeError(f"Input device index {device_index} was not found.")

        candidates: list[tuple[int, dict[str, Any]]] = []
        for index, device in devices:
            name = str(device.get("name", "")).lower()
            max_channels = int(device.get("max_input_channels", 0))
            if max_channels <= MIC_CHANNEL_INDEX:
                continue
            if any(keyword in name for keyword in MIC_NAME_KEYWORDS):
                candidates.append((index, device))

        if candidates:
            candidates.sort(
                key=lambda item: (
                    int(item[1].get("default_samplerate", 0)) != SAMPLE_RATE,
                    int(item[1].get("max_input_channels", 0)) < REQUIRED_INPUT_CHANNELS,
                    item[0],
                )
            )
            index, device = candidates[0]
            channels = min(REQUIRED_INPUT_CHANNELS, int(device["max_input_channels"]))
            return AudioDeviceSelection(index, device, channels)

        raise RuntimeError("Could not automatically find a ReSpeaker Array V3 input device.")
