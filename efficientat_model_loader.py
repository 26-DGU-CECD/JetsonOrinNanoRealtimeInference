from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    FMAX,
    HOP_SIZE,
    MODEL_NAME,
    MODEL_SAMPLE_RATE,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SIZE,
)


@dataclass(frozen=True)
class EfficientATArtifacts:
    model: Any
    mel: Any
    audioset_labels: list[str]
    device: Any
    resampler: Any | None


class EfficientATModelLoader:
    def __init__(self, efficientat_dir: Path | str) -> None:
        self.efficientat_dir = Path(efficientat_dir)

    @staticmethod
    def select_device() -> Any:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def create_resampler(device: Any) -> Any | None:
        if SAMPLE_RATE == MODEL_SAMPLE_RATE:
            return None

        import torchaudio

        return torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(device).eval()

    def load(self, device: Any | None = None) -> EfficientATArtifacts:
        if device is None:
            device = self.select_device()

        if not self.efficientat_dir.exists():
            raise RuntimeError(
                f"EfficientAT repository was not found: {self.efficientat_dir}\n"
                "Run `bash install.sh` first or pass --efficientat-dir."
            )

        repo_dir = self.efficientat_dir.resolve()
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
        resampler = self.create_resampler(device)
        return EfficientATArtifacts(
            model=model,
            mel=mel,
            audioset_labels=list(labels),
            device=device,
            resampler=resampler,
        )
