"""M3: unit tests for the model utility / inference function.

Uses a randomly-initialized model, so no trained weights are needed. Guarded with
importorskip so the suite still passes in a torch-free environment (the CI job
installs the full requirements and runs these for real).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model import SimpleCNN
from src.inference import predict
from src.config import CLASSES, IMG_SIZE


def _dummy_array():
    return np.random.randn(3, IMG_SIZE, IMG_SIZE).astype(np.float32)


def test_forward_pass_output_shape():
    model = SimpleCNN(num_classes=len(CLASSES))
    out = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    assert out.shape == (2, len(CLASSES))


def test_predict_returns_valid_label():
    model = SimpleCNN(num_classes=len(CLASSES))
    result = predict(model, _dummy_array())
    assert result["label"] in CLASSES
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_probabilities_sum_to_one():
    model = SimpleCNN(num_classes=len(CLASSES))
    result = predict(model, _dummy_array())
    total = sum(result["probabilities"].values())
    assert result["probabilities"].keys() == set(CLASSES) or set(result["probabilities"]) == set(CLASSES)
    assert abs(total - 1.0) < 1e-4
