"""1D VGG-style baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


VGG1D_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "channels_per_block": ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512)),
    "kernel_size": 3,
    "pool_size": 2,
    "classifier_hidden_dim": 256,
    "dropout": 0.4,
}


class VGGBlock1D(nn.Module):
    """Stacked Conv-BN-ReLU layers followed by optional max pooling."""

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        *,
        kernel_size: int = 3,
        pool_size: int | None = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = out_channels
        if pool_size is not None:
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
        self.block = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG1D(nn.Module):
    """VGG-style 1D CNN for hierarchical local spectral feature extraction."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        channels_per_block: tuple[tuple[int, ...], ...] = ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512)),
        kernel_size: int = 3,
        pool_size: int | None = 2,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if not channels_per_block:
            raise ValueError("channels_per_block must contain at least one block")

        blocks: list[nn.Module] = []
        current_channels = input_channels
        for block_channels in channels_per_block:
            if not block_channels:
                raise ValueError("each VGG block must contain at least one channel")
            block = VGGBlock1D(current_channels, tuple(block_channels), kernel_size=kernel_size, pool_size=pool_size)
            blocks.append(block)
            current_channels = block.out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def default_vgg1d_params() -> dict:
    """Return tunable default parameters for the 1D VGG baseline."""
    return copy_default_params(VGG1D_DEFAULT_PARAMS)
