from __future__ import annotations

from collections.abc import Sequence

from config import AUDIOSET_CLASS_COUNT, LABEL_MAPPING


class LabelMapper:
    def __init__(self, audioset_labels: Sequence[str]) -> None:
        self.audioset_labels = list(audioset_labels)

    def build_custom_label_indices(self) -> dict[str, list[int]]:
        if len(self.audioset_labels) != AUDIOSET_CLASS_COUNT:
            raise RuntimeError(
                f"AudioSet label count is not {AUDIOSET_CLASS_COUNT}: {len(self.audioset_labels)}"
            )

        label_to_index = {label: index for index, label in enumerate(self.audioset_labels)}
        indices: dict[str, list[int]] = {}
        missing: dict[str, list[str]] = {}

        for custom_label, audioset_labels in LABEL_MAPPING.items():
            matched = [label_to_index[label] for label in audioset_labels if label in label_to_index]
            not_found = [label for label in audioset_labels if label not in label_to_index]
            indices[custom_label] = matched
            if not_found:
                missing[custom_label] = not_found

        if missing:
            details = "; ".join(
                f"{custom_label}: {', '.join(labels)}"
                for custom_label, labels in missing.items()
            )
            raise RuntimeError(f"AudioSet labels were not found: {details}")

        return indices
