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
from .model import build_model


def load_model(model_path: Path | str = MODEL_PATH):
    """Rebuild the saved architecture and load its state_dict.

    The checkpoint carries ``arch`` and ``classes`` alongside the weights, so
    the serving container reconstructs the *same* network that was trained
    without being told which one it was. ``weights_only=True`` restricts
    unpickling to tensors and primitive containers (torch >= 2.4), which
    removes the arbitrary-code-execution surface of a plain ``torch.load``.

    The rebuilt backbone uses ``pretrained=False`` so the container never
    reaches out to download ImageNet weights at startup.
    """
    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        arch = checkpoint.get("arch", "simple_cnn")
        classes = list(checkpoint.get("classes", CLASSES))
    else:                                   # bare state_dict (legacy)
        state_dict, arch, classes = checkpoint, "simple_cnn", list(CLASSES)

    model = build_model(arch, num_classes=len(classes), pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()
    model.classes = classes                 # consumed by the API layer
    model.arch = arch
    return model


def predict(model, arr: np.ndarray, classes=None) -> Dict:
    """Run a forward pass on a single preprocessed array (shape (3, H, W)).

    Returns
    -------
    dict with keys: label, confidence, probabilities.
    """
    import torch

    if classes is None:
        classes = getattr(model, "classes", CLASSES)

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
