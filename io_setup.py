from __future__ import annotations

import io
import sys


class IOSetup:
    @staticmethod
    def configure_stdio() -> None:
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            if hasattr(stream, "buffer"):
                setattr(
                    sys,
                    name,
                    io.TextIOWrapper(
                        stream.buffer,
                        encoding="utf-8",
                        errors="replace",
                        line_buffering=True,
                    ),
                )
