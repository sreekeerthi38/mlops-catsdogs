"""Baseline model: a small, CPU-friendly CNN for 224x224 RGB binary classification.

Kept intentionally lightweight (4 conv blocks + global average pool) so it trains
in a few minutes on CPU over a small subset. Swap in transfer learning
(torchvision.models.mobilenet_v2/resnet18) if you have a GPU and want accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
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
