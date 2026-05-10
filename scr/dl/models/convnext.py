"""1D ConvNeXt-style baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


CONVNEXT1D_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "depths": (2, 2, 4, 2),
    "dims": (64, 128, 256, 512),
    "kernel_size": 7,
    "stem_kernel_size": 4,
    "stem_stride": 4,
    "mlp_ratio": 4,
    "layer_scale_init": 1e-6,
    "dropout": 0.35,
}


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt block adapted to channel-first 1D spectra."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        hidden_dim = channels * mlp_ratio
        self.pointwise = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        if layer_scale_init > 0:
            self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))
        else:
            self.layer_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pointwise(x)
        if self.layer_scale is not None:
            x = x * self.layer_scale
        x = x.transpose(1, 2)
        return residual + x


class ConvNeXt1D(nn.Module):
    """ConvNeXt-style 1D CNN for long LIBS spectra."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        depths: tuple[int, ...] = (2, 2, 4, 2),
        dims: tuple[int, ...] = (64, 128, 256, 512),
        kernel_size: int = 7,
        stem_kernel_size: int = 4,
        stem_stride: int = 4,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-6,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if len(depths) != len(dims):
            raise ValueError("depths and dims must have the same length")
        if not depths:
            raise ValueError("ConvNeXt1D expects at least one stage")

        layers: list[nn.Module] = [
            nn.Conv1d(
                input_channels,
                dims[0],
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=stem_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(dims[0]),
        ]
        for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
            if stage_idx > 0:
                layers.extend(
                    [
                        nn.BatchNorm1d(dims[stage_idx - 1]),
                        nn.Conv1d(dims[stage_idx - 1], dim, kernel_size=2, stride=2, bias=False),
                    ]
                )
            for _ in range(depth):
                layers.append(
                    ConvNeXtBlock1D(
                        dim,
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        layer_scale_init=layer_scale_init,
                    )
                )

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(dims[-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dims[-1], num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.norm(x)
        return self.classifier(x)


def default_convnext1d_params() -> dict:
    """Return tunable default parameters for the 1D ConvNeXt baseline."""
    return copy_default_params(CONVNEXT1D_DEFAULT_PARAMS)
