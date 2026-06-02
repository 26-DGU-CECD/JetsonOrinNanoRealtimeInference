from __future__ import annotations

import sys
import threading
import time

from config import RESPEAKER_USB_PRODUCT_ID, RESPEAKER_USB_VENDOR_ID
from doa_reading import DOAReading


class DOAUsbReader:
    def __init__(
        self,
        *,
        enabled: bool = True,
        poll_interval: float = 0.1,
        disabled_reason: str = "disabled",
    ) -> None:
        self.ok = False
        self.tuning = None
        self.status = "disabled"
        self.poll_interval = max(0.02, float(poll_interval))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_angle: int | None = None
        self._last_voice: bool | None = None
        self._last_read_at: float | None = None
        self._last_error: str | None = None

        if not enabled:
            print(f"[DOA] USB reader {disabled_reason}", file=sys.stderr, flush=True)
            return

        try:
            import usb.core  # type: ignore
            from tuning import Tuning  # type: ignore

            dev = usb.core.find(
                idVendor=RESPEAKER_USB_VENDOR_ID,
                idProduct=RESPEAKER_USB_PRODUCT_ID,
            )
            if dev is None:
                self.status = "device_not_found"
                print("[DOA] ReSpeaker USB control device not found.", file=sys.stderr, flush=True)
                return

            self.tuning = Tuning(dev)
            self.ok = True
            self.status = "enabled"
            self._poll_once()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            if self.status == "enabled":
                print("[DOA] ReSpeaker USB DOA reader enabled.", file=sys.stderr, flush=True)
            else:
                usb_status = self.status
                if self._last_error:
                    usb_status = f"{usb_status}:{self._last_error}"
                print(
                    f"[DOA] ReSpeaker USB control found, but reads failed: {usb_status}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            self.status = "unavailable"
            print(f"[DOA] USB reader disabled: {exc!r}", file=sys.stderr, flush=True)

    def _read_device_locked(self) -> tuple[int | None, bool | None]:
        if self.tuning is None:
            return None, None

        voice = None
        try:
            voice_value = self.tuning.is_voice()
            if voice_value is not None:
                voice = bool(int(voice_value))
        except Exception:
            voice = None

        angle = self.tuning.direction
        if angle is None:
            return None, voice
        return int(float(angle)) % 360, voice

    def _poll_once(self) -> None:
        if not self.ok or self.tuning is None:
            return

        try:
            with self._lock:
                angle, voice = self._read_device_locked()
            self._last_read_at = time.monotonic()
            self._last_voice = voice
            if angle is not None:
                self._last_angle = angle
            self.status = "enabled"
            self._last_error = None
        except Exception as exc:
            self.status = "read_error"
            self._last_error = type(exc).__name__

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            self._poll_once()

    def read_angle(self) -> int | None:
        return self.snapshot().raw_angle

    def snapshot(self) -> DOAReading:
        if not self.ok:
            return DOAReading(None, "usb", self.status)

        self._poll_once()
        angle = self._last_angle
        if angle is None:
            status = self.status
            if self._last_error:
                status = f"{status}:{self._last_error}"
            if status == "enabled":
                status = "usb_no_angle"
            return DOAReading(None, "usb", status)

        now = time.monotonic()
        age = None if self._last_read_at is None else now - self._last_read_at
        if age is not None and age > max(1.0, self.poll_interval * 5.0):
            return DOAReading(None, "usb", "usb_stale")

        status = "usb_no_voice" if self._last_voice is False else "usb_active"
        return DOAReading(angle, "usb", status)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.tuning is not None:
            try:
                self.tuning.close()
            except Exception:
                pass
            self.tuning = None
