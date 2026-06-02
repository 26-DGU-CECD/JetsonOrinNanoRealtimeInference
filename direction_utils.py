from __future__ import annotations

from config import CARDINAL_SUFFIX


class DirectionUtils:
    @staticmethod
    def angle_to_cardinal(angle: float) -> str:
        corrected = float(angle) % 360.0
        if corrected < 45.0 or corrected >= 315.0:
            return "북"
        if corrected < 135.0:
            return "동"
        if corrected < 225.0:
            return "남"
        return "서"

    @staticmethod
    def corrected_angle(raw_angle: float, north_offset: float) -> int:
        return int(round((float(raw_angle) - float(north_offset)) % 360.0)) % 360

    @classmethod
    def direction_text(cls, corrected_angle: int | None) -> tuple[str, str]:
        if corrected_angle is None:
            return "", ""
        direction = cls.angle_to_cardinal(corrected_angle)
        return direction, f"{CARDINAL_SUFFIX[direction]} {corrected_angle}도"
