"""1D CNN baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


CNN_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "channels": (32, 64, 128, 256),
    "kernel_sizes": (7, 5, 5, 3),
    "pool_sizes": (2, 2, 2, None),
    "dropout": 0.3,
}


class ConvBlock(nn.Module):
    """Conv-BN-ReLU block with optional max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        pool_size: int | None = None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool_size is not None:
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN(nn.Module):
    """Lightweight 1D CNN baseline for local spectral peak patterns."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, ...] = (7, 5, 5, 3),
        pool_sizes: tuple[int | None, ...] = (2, 2, 2, None),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(kernel_sizes) == len(pool_sizes)):
            raise ValueError("channels, kernel_sizes, and pool_sizes must have the same length")

        blocks: list[nn.Module] = []
        in_channels = input_channels
        for out_channels, kernel_size, pool_size in zip(channels, kernel_sizes, pool_sizes):
            blocks.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, pool_size=pool_size))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels[-1], num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def default_cnn_params() -> dict:
    """Return tunable default parameters for the 1D CNN baseline."""
    return copy_default_params(CNN_DEFAULT_PARAMS)
