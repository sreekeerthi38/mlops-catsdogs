"""Central configuration: filesystem paths, class labels, and params.yaml loader.

Kept dependency-light on purpose (only PyYAML) so it can be imported by every
part of the pipeline (training, serving, tests) without pulling in torch.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# --- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"      # processed/{train,val,test}/{cats,dogs}
MODELS_DIR = ROOT_DIR / "models"
PARAMS_PATH = ROOT_DIR / "params.yaml"

# Model path used by the inference service. Overridable via env for containers.
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "model.pt")))

# --- Labels ------------------------------------------------------------------
# Index order MUST match training. torchvision.ImageFolder sorts folders
# alphabetically -> "cats"=0, "dogs"=1. We keep short labels for the API.
CLASSES = ["cat", "dog"]

# --- Image / normalization constants ----------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_params(path: Path | str = PARAMS_PATH) -> dict:
    """Load params.yaml. Returns {} if the file is missing so tests never crash."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
