"""1D Inception-style baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


INCEPTION1D_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "stem_channels": 32,
    "branch_channels": ((32, 32, 32, 32), (64, 64, 64, 64), (96, 96, 96, 96)),
    "kernel_sizes": (5, 9),
    "pool_sizes": (2, 2, None),
    "dropout": 0.35,
}


class InceptionBlock1D(nn.Module):
    """Parallel 1D convolution branches for multi-scale spectral peaks."""

    def __init__(
        self,
        in_channels: int,
        branch_channels: tuple[int, int, int, int],
        *,
        kernel_sizes: tuple[int, int] = (5, 9),
    ) -> None:
        super().__init__()
        if len(branch_channels) != 4:
            raise ValueError("branch_channels must contain four branch widths")
        if len(kernel_sizes) != 2:
            raise ValueError("kernel_sizes must contain two convolution kernels")

        branch_1x1, branch_small, branch_large, branch_pool = branch_channels
        small_kernel, large_kernel = kernel_sizes
        self.branch_1x1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_1x1, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_1x1),
            nn.ReLU(inplace=True),
        )
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, branch_small, kernel_size=small_kernel, padding=small_kernel // 2, bias=False),
            nn.BatchNorm1d(branch_small),
            nn.ReLU(inplace=True),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, branch_large, kernel_size=large_kernel, padding=large_kernel // 2, bias=False),
            nn.BatchNorm1d(branch_large),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_pool, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_pool),
            nn.ReLU(inplace=True),
        )
        self.out_channels = sum(branch_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.branch_1x1(x), self.branch_small(x), self.branch_large(x), self.branch_pool(x))
        return torch.cat(branches, dim=1)


class Inception1D(nn.Module):
    """Inception-style 1D CNN for multi-scale LIBS spectral feature extraction."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        stem_channels: int = 32,
        branch_channels: tuple[tuple[int, int, int, int], ...] = ((32, 32, 32, 32), (64, 64, 64, 64), (96, 96, 96, 96)),
        kernel_sizes: tuple[int, int] = (5, 9),
        pool_sizes: tuple[int | None, ...] = (2, 2, None),
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if len(branch_channels) != len(pool_sizes):
            raise ValueError("branch_channels and pool_sizes must have the same length")

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        current_channels = stem_channels
        for block_channels, pool_size in zip(branch_channels, pool_sizes):
            block = InceptionBlock1D(current_channels, tuple(block_channels), kernel_sizes=tuple(kernel_sizes))
            blocks.append(block)
            if pool_size is not None:
                blocks.append(nn.MaxPool1d(kernel_size=pool_size))
            current_channels = block.out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(current_channels, num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def default_inception1d_params() -> dict:
    """Return tunable default parameters for the 1D Inception baseline."""
    return copy_default_params(INCEPTION1D_DEFAULT_PARAMS)
