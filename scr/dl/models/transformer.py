"""1D Transformer baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


TRANSFORMER1D_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.2,
    "patch_size": 16,
    "conv_kernel_size": 16,
    "pooling": "mean",
    "max_sequence_length": 520,
}


class Transformer1D(nn.Module):
    """Patch-based Transformer encoder for global LIBS spectral dependencies."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        patch_size: int = 16,
        conv_kernel_size: int = 16,
        pooling: str = "cls",
        max_sequence_length: int = 520,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.pooling = pooling
        self.max_sequence_length = max_sequence_length
        self.patch_embed = nn.Conv1d(
            input_channels,
            d_model,
            kernel_size=conv_kernel_size,
            stride=patch_size,
            padding=conv_kernel_size // 2,
            bias=False,
        )
        positional_length = max_sequence_length + (1 if pooling == "cls" else 0)
        self.position_embedding = nn.Parameter(torch.zeros(1, positional_length, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model, num_classes),
        )
        self.apply(initialize_weights)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).transpose(1, 2)
        if x.size(1) > self.max_sequence_length:
            raise ValueError("embedded sequence length exceeds max_sequence_length")
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        if self.pooling == "cls":
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


def default_transformer1d_params() -> dict:
    """Return tunable default parameters for the 1D Transformer baseline."""
    return copy_default_params(TRANSFORMER1D_DEFAULT_PARAMS)
