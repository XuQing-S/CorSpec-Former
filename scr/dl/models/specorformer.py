"""Multi-scale 1D SpeCorformer for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .attention import (
    CrossAttention1D,
    FrequencyDomainSelfAttention1D,
    HybridFrequencySelfAttention1D,
    MultiHeadSelfAttention1D,
    SpectralCorrelationSelfAttention1D,
    make_self_attention,
)
from .common import INPUT_CHANNELS, NUM_METAL_CLASSES, copy_default_params, initialize_weights

__all__ = [
    "CrossAttention1D",
    "FrequencyDomainSelfAttention1D",
    "HybridFrequencySelfAttention1D",
    "MultiHeadSelfAttention1D",
    "SpeCorformer1D",
    "SpeCorformerDecoderBlock1D",
    "SpeCorformerEncoderBlock1D",
    "SpeCorformerDecoderStage1D",
    "SpeCorformerEncoderStage1D",
    "SpectralCorrelationSelfAttention1D",
    "SpectrumPatchEmbedding1D",
    "default_specorformer1d_params",
]


SPECORFORMER1D_DEFAULT_PARAMS = {
    "input_channels": INPUT_CHANNELS,
    "num_classes": NUM_METAL_CLASSES,
    "embed_dims": (64, 128, 256),
    "patch_sizes": (4, 16, 64),
    "patch_kernel_sizes": (9, 33, 129),
    "depths": (2, 2, 4),
    "self_nheads": (2, 4, 8),
    "cross_nheads": (4, 8),
    "mlp_ratios": (4.0, 4.0, 4.0),
    "self_attention_type": "hfsa",
    "frequency_bands": (64, 64, 64),
    "local_kernel_sizes": (5, 5, 5),
    "cross_attention_type": "mhca",
    "dropout": 0.2,
    "pooling": "mean",
}


class SpectrumPatchEmbedding1D(nn.Module):
    """Patch embedding that samples one scale directly from the raw spectrum."""

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_size: int,
        kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be greater than or equal to 1")
        if kernel_size < patch_size:
            raise ValueError("kernel_size should be greater than or equal to patch_size")

        self.proj = nn.Sequential(
            nn.Conv1d(
                input_channels,
                embed_dim,
                kernel_size=kernel_size,
                stride=patch_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2)


class SpeCorformerEncoderBlock1D(nn.Module):
    """First-scale block: self attention followed by an MLP."""

    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        attention_type: str = "scsa",
        frequency_bands: int = 64,
        local_kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = make_self_attention(attention_type, dim, nhead, dropout, frequency_bands, local_kernel_size)
        self.drop_path = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SpeCorformerDecoderBlock1D(nn.Module):
    """Later-scale block: self attention, cross attention, then an MLP."""

    def __init__(
        self,
        dim: int,
        memory_dim: int,
        self_nhead: int,
        cross_nhead: int,
        mlp_ratio: float = 4.0,
        self_attention_type: str = "fdsa",
        cross_attention_type: str = "mhca",
        frequency_bands: int = 64,
        local_kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if cross_attention_type != "mhca":
            raise ValueError("cross_attention_type must be 'mhca'")

        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = make_self_attention(
            self_attention_type,
            dim,
            self_nhead,
            dropout,
            frequency_bands,
            local_kernel_size,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention1D(dim, memory_dim, cross_nhead, dropout)
        self.drop_path = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        x = x + self.drop_path(self.cross_attn(self.norm2(x), memory))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class SpeCorformerEncoderStage1D(nn.Module):
    """First scale that only performs self-attention over its own tokens."""

    def __init__(
        self,
        dim: int,
        depth: int,
        nhead: int,
        mlp_ratio: float,
        self_attention_type: str,
        frequency_bands: int,
        local_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                SpeCorformerEncoderBlock1D(
                    dim,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    attention_type=self_attention_type,
                    frequency_bands=frequency_bands,
                    local_kernel_size=local_kernel_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SpeCorformerDecoderStage1D(nn.Module):
    """Later scale that fuses its self-attended tokens with previous-scale memory."""

    def __init__(
        self,
        dim: int,
        memory_dim: int,
        depth: int,
        self_nhead: int,
        cross_nhead: int,
        mlp_ratio: float,
        self_attention_type: str,
        cross_attention_type: str,
        frequency_bands: int,
        local_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SpeCorformerDecoderBlock1D(
                    dim,
                    memory_dim=memory_dim,
                    self_nhead=self_nhead,
                    cross_nhead=cross_nhead,
                    mlp_ratio=mlp_ratio,
                    self_attention_type=self_attention_type,
                    cross_attention_type=cross_attention_type,
                    frequency_bands=frequency_bands,
                    local_kernel_size=local_kernel_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, memory)
        return x


class SpeCorformer1D(nn.Module):
    """Multi-scale SpeCorformer with decoder-like cross-scale token fusion."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_METAL_CLASSES,
        embed_dims: tuple[int, ...] = (64, 128, 256),
        patch_sizes: tuple[int, ...] = (4, 16, 64),
        patch_kernel_sizes: tuple[int, ...] = (9, 33, 129),
        depths: tuple[int, ...] = (2, 2, 4),
        self_nheads: tuple[int, ...] = (2, 4, 8),
        cross_nheads: tuple[int, ...] = (4, 8),
        mlp_ratios: tuple[float, ...] = (4.0, 4.0, 4.0),
        self_attention_type: str = "hfsa",
        frequency_bands: tuple[int, ...] = (64, 64, 64),
        local_kernel_sizes: tuple[int, ...] = (5, 5, 5),
        cross_attention_type: str = "mhca",
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max", "concat_mean"}:
            raise ValueError("pooling must be 'mean', 'max', or 'concat_mean'")
        if len(embed_dims) < 2:
            raise ValueError("SpeCorformer1D expects at least two scales")
        if not (
            len(embed_dims)
            == len(patch_sizes)
            == len(patch_kernel_sizes)
            == len(depths)
            == len(self_nheads)
            == len(mlp_ratios)
            == len(frequency_bands)
            == len(local_kernel_sizes)
        ):
            raise ValueError(
                "embed_dims, patch_sizes, patch_kernel_sizes, depths, self_nheads, mlp_ratios, "
                "frequency_bands, and local_kernel_sizes must have the same length"
            )
        if len(cross_nheads) != len(embed_dims) - 1:
            raise ValueError("cross_nheads must contain one value for each cross-scale fusion stage")

        self.pooling = pooling
        self.patch_embeddings = nn.ModuleList(
            [
                SpectrumPatchEmbedding1D(
                    input_channels=input_channels,
                    embed_dim=embed_dim,
                    patch_size=patch_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for embed_dim, patch_size, kernel_size in zip(embed_dims, patch_sizes, patch_kernel_sizes)
            ]
        )

        self.encoder_stage = SpeCorformerEncoderStage1D(
            embed_dims[0],
            depth=depths[0],
            nhead=self_nheads[0],
            mlp_ratio=mlp_ratios[0],
            self_attention_type=self_attention_type,
            frequency_bands=frequency_bands[0],
            local_kernel_size=local_kernel_sizes[0],
            dropout=dropout,
        )
        self.decoder_stages = nn.ModuleList(
            SpeCorformerDecoderStage1D(
                embed_dims[index],
                memory_dim=embed_dims[index - 1],
                depth=depths[index],
                self_nhead=self_nheads[index],
                cross_nhead=cross_nheads[index - 1],
                mlp_ratio=mlp_ratios[index],
                self_attention_type=self_attention_type,
                cross_attention_type=cross_attention_type,
                frequency_bands=frequency_bands[index],
                local_kernel_size=local_kernel_sizes[index],
                dropout=dropout,
            )
            for index in range(1, len(embed_dims))
        )

        classifier_dim = sum(embed_dims) if pooling == "concat_mean" else embed_dims[-1]
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for embed_dim in embed_dims)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_dim, num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_tokens = [patch_embed(x) for patch_embed in self.patch_embeddings]

        features: list[torch.Tensor] = [self.encoder_stage(scale_tokens[0])]
        for tokens, stage in zip(scale_tokens[1:], self.decoder_stages):
            features.append(stage(tokens, features[-1]))

        if self.pooling == "concat_mean":
            pooled = [norm(feature).mean(dim=1) for norm, feature in zip(self.norms, features)]
            x = torch.cat(pooled, dim=-1)
        else:
            x = self.norms[-1](features[-1])
            if self.pooling == "mean":
                x = x.mean(dim=1)
            else:
                x = x.max(dim=1).values
        return self.classifier(x)


def default_specorformer1d_params() -> dict:
    """Return tunable default parameters for the 1D SpeCorformer."""
    return copy_default_params(SPECORFORMER1D_DEFAULT_PARAMS)
