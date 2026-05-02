"""Conv-LSTM baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


LSTM_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "conv_channels": 64,
    "conv_kernel_size": 7,
    "conv_stride": 4,
    "hidden_size": 128,
    "num_layers": 1,
    "bidirectional": True,
    "pooling": "mean",
    "dropout": 0.3,
}


class LSTM(nn.Module):
    """Conv-LSTM baseline for long-range spectral dependencies."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        conv_channels: int = 64,
        conv_kernel_size: int = 7,
        conv_stride: int = 4,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        pooling: str = "mean",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")

        self.pooling = pooling
        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_channels,
                conv_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        feature_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.transpose(1, 2)
        sequence_output, _ = self.lstm(x)
        if self.pooling == "last":
            x = sequence_output[:, -1, :]
        else:
            x = sequence_output.mean(dim=1)
        return self.classifier(x)


def default_lstm_params() -> dict:
    """Return tunable default parameters for the Conv-LSTM baseline."""
    return copy_default_params(LSTM_DEFAULT_PARAMS)
