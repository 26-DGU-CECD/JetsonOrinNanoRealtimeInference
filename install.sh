#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "EfficientAT" ]; then
  git clone https://github.com/fschmid56/EfficientAT.git
else
  echo "EfficientAT already exists; skipping clone."
fi

pip install torch torchaudio sounddevice "numpy<2"

# EfficientAT's MobileNet model imports torchvision.ops.misc.
pip install torchvision
