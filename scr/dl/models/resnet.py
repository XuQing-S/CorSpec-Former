"""1D ResNet baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


RESNET_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "base_channels": 32,
    "blocks_per_stage": (2, 2, 2, 2),
    "stem_kernel_size": 7,
    "stem_stride": 2,
    "dropout": 0.3,
}


class BasicBlock1D(nn.Module):
    """Residual block for 1D spectra."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """1D ResNet baseline for deeper LIBS spectral feature extraction."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        base_channels: int = 32,
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != 4:
            raise ValueError("ResNet1DMetalClassifier expects four stages")

        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                base_channels,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=stem_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(base_channels, blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(base_channels * 2, blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(base_channels * 4, blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(base_channels * 8, blocks_per_stage[3], stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(base_channels * 8, num_classes),
        )
        self.apply(initialize_weights)

    def _make_stage(self, out_channels: int, block_count: int, stride: int) -> nn.Sequential:
        blocks: list[nn.Module] = [BasicBlock1D(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, block_count):
            blocks.append(BasicBlock1D(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def default_resnet_params() -> dict:
    """Return tunable default parameters for the 1D ResNet baseline."""
    return copy_default_params(RESNET_DEFAULT_PARAMS)
