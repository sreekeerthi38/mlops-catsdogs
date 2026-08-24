"""Models for 224x224 RGB binary classification.

Two architectures are available and both are logged to MLflow, so the
experiment tracker holds a real comparison rather than a single run:

* ``simple_cnn``   -- the from-scratch baseline (4 conv blocks + GAP).
                      Fast, CPU-friendly, weak (~63% on a 2k subset).
* ``mobilenet_v2`` -- ImageNet backbone frozen, classifier head retrained.
                      Same 3 CPU epochs, far better separation.

``pretrained`` MUST be False when rebuilding a model for inference: the
serving container has no internet and does not need ImageNet weights, since
the trained ``state_dict`` is loaded straight over the top.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Canonical architecture keys. Older checkpoints wrote the class name, so the
# alias map keeps `load_model` backward compatible with model.pt files that
# were saved before this refactor.
ARCH_ALIASES = {
    "SimpleCNN": "simple_cnn",
    "simple_cnn": "simple_cnn",
    "MobileNetV2": "mobilenet_v2",
    "mobilenet_v2": "mobilenet_v2",
}


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """From-scratch baseline CNN."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 16),      # 224 -> 112
            _block(16, 32),     # 112 -> 56
            _block(32, 64),     # 56  -> 28
            _block(64, 128),    # 28  -> 14
            nn.AdaptiveAvgPool2d(1),   # -> (128, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_mobilenet(num_classes: int = 2, pretrained: bool = True,
                    freeze_backbone: bool = True) -> nn.Module:
    """MobileNetV2 with a fresh classifier head.

    Freezing ``features`` means only ~2.5k parameters train, so a CPU epoch
    over 2000 images stays in the same time budget as the baseline CNN.
    """
    from torchvision import models

    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    if freeze_backbone:
        for p in m.features.parameters():
            p.requires_grad = False
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


def build_model(arch: str = "mobilenet_v2", num_classes: int = 2,
                pretrained: bool = True) -> nn.Module:
    """Factory. ``arch`` accepts canonical keys or legacy class names."""
    key = ARCH_ALIASES.get(arch)
    if key is None:
        raise ValueError(
            f"unknown arch {arch!r}; expected one of {sorted(set(ARCH_ALIASES.values()))}"
        )
    if key == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    return build_mobilenet(num_classes=num_classes, pretrained=pretrained)
