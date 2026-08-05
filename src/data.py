"""Data layer: preprocessing, augmentation transforms, dataloaders, dataset split.

`preprocess_image` is deliberately implemented with only Pillow + numpy so it is
(a) identical between training and serving and (b) unit-testable without torch.
The torchvision transforms used for *training* apply the same resize + ImageNet
normalization, so the model sees consistent inputs at train and inference time.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing (unit-tested; no torch dependency)
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Convert a PIL image into a normalized, channels-first float32 array.

    Steps: force RGB -> resize to (size, size) -> scale to [0, 1] ->
    ImageNet-normalize -> transpose to (C, H, W).

    Returns
    -------
    np.ndarray of shape (3, size, size), dtype float32.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected PIL.Image, got {type(image)!r}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = (arr - _MEAN) / _STD                                 # normalize
    arr = np.transpose(arr, (2, 0, 1))                         # (3, H, W)
    return np.ascontiguousarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# torchvision transforms + dataloaders (import torch lazily so this module
# stays importable in a torch-free test environment)
# ---------------------------------------------------------------------------
def build_transforms(train: bool, size: int = IMG_SIZE):
    from torchvision import transforms

    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])


def make_dataloaders(processed_dir: Path, batch_size: int = 32,
                     num_workers: int = 2, size: int = IMG_SIZE):
    """Build train/val/test dataloaders from an ImageFolder-style directory."""
    import torch
    from torchvision import datasets

    processed_dir = Path(processed_dir)
    loaders = {}
    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_dir} not found. Run scripts/prepare_data.py first."
            )
        ds = datasets.ImageFolder(split_dir, transform=build_transforms(split == "train", size))
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=False,
        )
    return loaders["train"], loaders["val"], loaders["test"]


# ---------------------------------------------------------------------------
# Dataset splitting (used by scripts/prepare_data.py)
# ---------------------------------------------------------------------------
def split_dataset(raw_pairs, out_dir: Path,
                  ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                  seed: int = 42) -> dict:
    """Copy (src_path, class_name) pairs into out_dir/{train,val,test}/{class}.

    Returns a dict of counts per split.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    out_dir = Path(out_dir)
    rng = random.Random(seed)

    # group by class then shuffle within class for a stratified split
    by_class: dict[str, list[Path]] = {}
    for src, cls in raw_pairs:
        by_class.setdefault(cls, []).append(Path(src))

    counts = {"train": 0, "val": 0, "test": 0}
    for cls, paths in by_class.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        buckets = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        for split, items in buckets.items():
            dst_dir = out_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                shutil.copy(src, dst_dir / f"{cls}_{i:05d}{src.suffix.lower()}")
            counts[split] += len(items)
    return counts
