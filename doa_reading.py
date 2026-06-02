from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DOAReading:
    raw_angle: int | None
    source: str
    status: str
