from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from config import AUDIOSET_CLASS_COUNT, MODEL_INPUT_SAMPLES


@dataclass(frozen=True)
class ClassificationResult:
    best_label: str
    best_score: float
    scores: dict[str, float]
    infer_sec: float = 0.0


class SoundClassifier:
    def __init__(
        self,
        *,
        model: Any,
        mel: Any,
        resampler: Any | None,
        custom_indices: dict[str, list[int]],
        audioset_labels: Sequence[str],
        device: Any,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.mel = mel
        self.resampler = resampler
        self.custom_indices = custom_indices
        self.audioset_labels = list(audioset_labels)
        self.device = device
        self.debug = bool(debug)

    def predict(self, waveform: np.ndarray) -> ClassificationResult:
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = np.clip(waveform, -1.0, 1.0)
        input_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        if self.resampler is not None:
            input_tensor = self.resampler(input_tensor)
        if input_tensor.shape[1] < MODEL_INPUT_SAMPLES:
            input_tensor = torch.nn.functional.pad(
                input_tensor,
                (0, MODEL_INPUT_SAMPLES - input_tensor.shape[1]),
            )
        elif input_tensor.shape[1] > MODEL_INPUT_SAMPLES:
            input_tensor = input_tensor[:, :MODEL_INPUT_SAMPLES]

        amp_context = (
            torch.amp.autocast("cuda", enabled=True)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), amp_context:
            spec = self.mel(input_tensor)
            logits, _ = self.model(spec.unsqueeze(0))
            probabilities = torch.sigmoid(logits.float()).squeeze(0).detach().cpu().numpy()

        if probabilities.shape[0] != AUDIOSET_CLASS_COUNT:
            raise RuntimeError(
                f"Model output class count is not {AUDIOSET_CLASS_COUNT}: {probabilities.shape[0]}"
            )

        if self.debug:
            self._print_debug(input_tensor, spec, logits, probabilities)

        scores = {
            custom_label: float(np.max(probabilities[label_indices]))
            for custom_label, label_indices in self.custom_indices.items()
        }
        best_label = max(scores, key=scores.get)
        return ClassificationResult(best_label, scores[best_label], scores)

    def _print_debug(
        self,
        input_tensor: Any,
        spec: Any,
        logits: Any,
        probabilities: np.ndarray,
    ) -> None:
        logits_cpu = logits.float().squeeze(0).detach().cpu().numpy()
        spec_cpu = spec.detach().float().cpu().numpy()
        input_cpu = input_tensor.detach().float().cpu().numpy().squeeze(0)
        top_indices = np.argsort(probabilities)[::-1][:10]
        top_text = ", ".join(
            f"{self.audioset_labels[index]}={probabilities[index]:.4f}/logit={logits_cpu[index]:+.2f}"
            for index in top_indices
        )
        print(
            "DEBUG "
            f"wav[min={input_cpu.min():+.4f}, max={input_cpu.max():+.4f}, "
            f"rms={np.sqrt(np.mean(np.square(input_cpu))):.6f}] | "
            f"mel[min={spec_cpu.min():+.3f}, max={spec_cpu.max():+.3f}, mean={spec_cpu.mean():+.3f}] | "
            f"logits[min={logits_cpu.min():+.2f}, max={logits_cpu.max():+.2f}, "
            f"mean={logits_cpu.mean():+.2f}] | "
            f"top={top_text}",
            file=sys.stderr,
            flush=True,
        )
