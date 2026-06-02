from __future__ import annotations

import io
import os
import sys
import warnings
from collections.abc import Sequence
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import (
    AUDIOSET_CLASS_COUNT,
    FMAX,
    HOP_SIZE,
    LABEL_MAPPING,
    MODEL_INPUT_SAMPLES,
    MODEL_NAME,
    MODEL_SAMPLE_RATE,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SIZE,
)


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

    @classmethod
    def from_efficientat(
        cls,
        efficientat_dir: Path | str,
        *,
        device: Any | None = None,
        debug: bool = False,
    ) -> "SoundClassifier":
        if device is None:
            device = cls.select_device()

        model, mel, audioset_labels, resampler = cls._load_efficientat(
            Path(efficientat_dir),
            device,
        )
        return cls(
            model=model,
            mel=mel,
            resampler=resampler,
            custom_indices=cls._build_custom_label_indices(audioset_labels),
            audioset_labels=audioset_labels,
            device=device,
            debug=debug,
        )

    @staticmethod
    def select_device() -> Any:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def _load_efficientat(cls, efficientat_dir: Path, device: Any) -> tuple[Any, Any, list[str], Any | None]:
        if not efficientat_dir.exists():
            raise RuntimeError(
                f"EfficientAT repository was not found: {efficientat_dir}\n"
                "Run `bash install.sh` first or pass --efficientat-dir."
            )

        repo_dir = efficientat_dir.resolve()
        repo_text = str(repo_dir)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)

        old_cwd = os.getcwd()
        try:
            # EfficientAT helper modules load metadata/resources by relative path.
            os.chdir(repo_text)
            from helpers.utils import NAME_TO_WIDTH, labels  # type: ignore
            from models.mn.model import get_model as get_mn  # type: ignore
            from models.preprocess import AugmentMelSTFT  # type: ignore

            with redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Don't use ConvNormActivation directly.*",
                    category=UserWarning,
                    module="torchvision\\.ops\\.misc",
                )
                model = get_mn(
                    width_mult=NAME_TO_WIDTH(MODEL_NAME),
                    pretrained_name=MODEL_NAME,
                    strides=(2, 2, 2, 2),
                    head_type="mlp",
                )

            mel = AugmentMelSTFT(
                n_mels=N_MELS,
                sr=MODEL_SAMPLE_RATE,
                win_length=WINDOW_SIZE,
                hopsize=HOP_SIZE,
                n_fft=N_FFT,
                fmax=FMAX,
                freqm=0,
                timem=0,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torchvision":
                raise RuntimeError(
                    "EfficientAT model loading requires torchvision. "
                    "Install it with `pip install torchvision` and retry."
                ) from exc
            raise
        finally:
            os.chdir(old_cwd)

        model.to(device).eval()
        mel.to(device).eval()
        return model, mel, list(labels), cls._create_resampler(device)

    @staticmethod
    def _create_resampler(device: Any) -> Any | None:
        if SAMPLE_RATE == MODEL_SAMPLE_RATE:
            return None

        import torchaudio

        return torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(device).eval()

    @staticmethod
    def _build_custom_label_indices(audioset_labels: Sequence[str]) -> dict[str, list[int]]:
        if len(audioset_labels) != AUDIOSET_CLASS_COUNT:
            raise RuntimeError(
                f"AudioSet label count is not {AUDIOSET_CLASS_COUNT}: {len(audioset_labels)}"
            )

        label_to_index = {label: index for index, label in enumerate(audioset_labels)}
        indices: dict[str, list[int]] = {}
        missing: dict[str, list[str]] = {}

        for custom_label, labels in LABEL_MAPPING.items():
            matched = [label_to_index[label] for label in labels if label in label_to_index]
            not_found = [label for label in labels if label not in label_to_index]
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
