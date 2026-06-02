from __future__ import annotations


class DbThresholdGate:
    def __init__(self, min_db: float) -> None:
        self.min_db = float(min_db)
        self.threshold_dbfs = self.dbfs_threshold(self.min_db)

    @staticmethod
    def dbfs_threshold(value: float) -> float:
        if value > 0:
            return -float(value)
        return float(value)

    @staticmethod
    def format_optional_dbfs_threshold(value: float | None) -> str:
        if value is None:
            return "off"
        return f"{float(value):+.1f}"

    def is_low(self, dbfs: float) -> bool:
        return float(dbfs) < self.threshold_dbfs
