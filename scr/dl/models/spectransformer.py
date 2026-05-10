"""1D spectral Transformer with replaceable attention for LIBS classification."""

from __future__ import annotations

import torch
from torch import nn

from .attention import SpectralAttention, make_self_attention
from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights


SPECTRANSFORMER1D_DEFAULT_PARAMS = {
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
    "attention_type": "spectral",  # spectral, mhsa, scsa, fdsa, hfsa
    "frequency_bands": 64,
    "local_kernel_size": 5,
}


class SpecTransformerEncoderLayer1D(nn.Module):
    """Transformer encoder layer whose attention branch can be replaced."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        attention_type: str = "spectral",
        frequency_bands: int = 64,
        local_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = self._make_attention(
            attention_type=attention_type,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            frequency_bands=frequency_bands,
            local_kernel_size=local_kernel_size,
        )
        self.drop_path = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    @staticmethod
    def _make_attention(
        attention_type: str,
        d_model: int,
        nhead: int,
        dropout: float,
        frequency_bands: int,
        local_kernel_size: int,
    ) -> nn.Module:
        if attention_type == "spectral":
            return SpectralAttention(hidden_size=d_model, num_blocks=nhead, input_format="BLC")
        return make_self_attention(attention_type, d_model, nhead, dropout, frequency_bands, local_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SpecTransformer1D(nn.Module):
    """Patch-based Transformer with configurable spectral attention."""

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
        pooling: str = "mean",
        max_sequence_length: int = 520,
        attention_type: str = "spectral",
        frequency_bands: int = 64,
        local_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if attention_type not in {"mhsa", "scsa", "fdsa", "hfsa", "spectral"}:
            raise ValueError("attention_type must be 'mhsa', 'scsa', 'fdsa', 'hfsa', or 'spectral'")

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
        self.encoder = nn.Sequential(
            *[
                SpecTransformerEncoderLayer1D(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_type=attention_type,
                    frequency_bands=frequency_bands,
                    local_kernel_size=local_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )
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


def default_spectransformer1d_params() -> dict:
    """Return tunable default parameters for the spectral Transformer."""
    return copy_default_params(SPECTRANSFORMER1D_DEFAULT_PARAMS)
