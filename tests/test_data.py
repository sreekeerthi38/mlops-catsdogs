"""M3: unit tests for a data pre-processing function (torch-free)."""
import numpy as np
import pytest
from PIL import Image

from src.data import preprocess_image
from src.config import IMG_SIZE


def _random_image(w=300, h=200, mode="RGB") -> Image.Image:
    arr = np.random.randint(0, 256, (h, w, 3 if mode == "RGB" else 1), dtype=np.uint8)
    return Image.fromarray(arr.squeeze(), mode=mode)


def test_output_shape_and_dtype():
    out = preprocess_image(_random_image())
    assert out.shape == (3, IMG_SIZE, IMG_SIZE)
    assert out.dtype == np.float32


def test_converts_grayscale_to_three_channels():
    gray = _random_image(mode="L")
    out = preprocess_image(gray)
    assert out.shape[0] == 3  # forced to RGB


def test_values_are_finite_and_normalized():
    out = preprocess_image(_random_image())
    assert np.isfinite(out).all()
    # after ImageNet normalization values fall roughly in [-2.7, 2.7]
    assert out.min() > -3.0 and out.max() < 3.0


def test_rejects_non_pil_input():
    with pytest.raises(TypeError):
        preprocess_image(np.zeros((10, 10, 3)))
