"""Inference utilities used by the API and by unit tests.

`predict` is the "model utility/inference function" that M3 asks us to unit-test.
It is pure: given a model and a preprocessed array it returns a plain dict, so it
can be tested with a randomly-initialized model (no trained weights required).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .config import CLASSES, MODEL_PATH
from .model import SimpleCNN


def load_model(model_path: Path | str = MODEL_PATH, num_classes: int = 2):
    """Instantiate SimpleCNN and load a saved state_dict. Returns an eval-mode model."""
    import torch

    model_path = Path(model_path)
    model = SimpleCNN(num_classes=num_classes)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, arr: np.ndarray, classes=CLASSES) -> Dict:
    """Run a forward pass on a single preprocessed array (shape (3, H, W)).

    Returns
    -------
    dict with keys: label, confidence, probabilities.
    """
    import torch

    if arr.ndim == 3:
        arr = arr[None, ...]          # add batch dim -> (1, 3, H, W)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "label": classes[idx],
        "confidence": round(float(probs[idx]), 6),
        "probabilities": {c: round(float(p), 6) for c, p in zip(classes, probs)},
    }
