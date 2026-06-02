from __future__ import annotations

from config import CAUTION_LABELS, DANGER_LABELS
from direction_utils import DirectionUtils


class AppPacketBuilder:
    def __init__(
        self,
        *,
        north_offset: float,
        db_offset: float,
        full_packet: bool,
    ) -> None:
        self.north_offset = float(north_offset)
        self.db_offset = float(db_offset)
        self.full_packet = bool(full_packet)

    def risk_level(self, label: str) -> str:
        key = str(label).strip().lower()
        if key in DANGER_LABELS:
            return "danger"
        if key in CAUTION_LABELS:
            return "caution"
        return "info"

    def app_db_from_dbfs(self, dbfs: float) -> float:
        return round(max(0.0, float(dbfs) + self.db_offset), 1)

    def build(
        self,
        *,
        timestamp: str,
        label: str,
        score: float,
        scores: dict[str, float],
        infer_sec: float,
        total_sec: float,
        chunk_dbfs: float,
        status_text: str,
        raw_line: str,
        raw_angle: int | None,
        doa_status: str,
        doa_source: str,
    ) -> dict:
        if raw_angle is None:
            angle = None
            direction = ""
            direction_text = ""
        else:
            angle = DirectionUtils.corrected_angle(raw_angle, self.north_offset)
            direction, direction_text = DirectionUtils.direction_text(angle)

        app_db = self.app_db_from_dbfs(chunk_dbfs)
        packet = {
            "status": "ok",
            "time": timestamp,
            "label": label,
            "score": round(float(score), 6),
            "infer_sec": round(float(infer_sec), 3),
            "total_sec": round(float(total_sec), 3),
            "db": app_db,
            "level": self.risk_level(label),
            "direction": direction,
            "angle": float(angle) if angle is not None else None,
            "angle_raw": float(raw_angle) if raw_angle is not None else None,
            "direction_text": direction_text,
            "doa_status": doa_status,
            "doa_source": doa_source,
            "has_doa": raw_angle is not None,
        }

        if self.full_packet:
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
