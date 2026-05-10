"""Attention modules for 1D LIBS spectral models."""

# Pylint in the current environment treats torch.fft functions as non-callable.
# pylint: disable=not-callable

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpectralCorrelationSelfAttention1D(nn.Module):
    """Token-wise 1D spectral correlation self-attention."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q - q.mean(dim=-1, keepdim=True)
        k = k - k.mean(dim=-1, keepdim=True)

        q2 = q.pow(2)
        k2 = k.pow(2)
        q2 = q2 / (q2.sum(dim=-1, keepdim=True) + 1e-7)
        k2 = k2 / (k2.sum(dim=-1, keepdim=True) + 1e-7)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-2)
        correlation_context = k2.transpose(-2, -1) @ v
        correlation_tokens = q2 @ correlation_context / math.sqrt(num_tokens)

        return self.proj(v + correlation_tokens).reshape(batch_size, num_tokens, channels)


class MultiHeadSelfAttention1D(nn.Module):
    """Standard token-wise multi-head self-attention for 1D spectra."""

    def __init__(self, dim: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.attn(x, x, x, need_weights=False)
        return x


class GatedSelfAttention1D(nn.Module):
    """Multi-head self-attention with sigmoid gating after SDPA outputs.

    This follows the strongest variant reported in "Gated Attention for Large
    Language Models": head-specific multiplicative sigmoid gates are computed
    from the input hidden states and applied to each attention head output.
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        dropout: float = 0.0,
        gate_granularity: str = "elementwise",
    ) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError("dim must be divisible by nhead for gated self-attention")
        if gate_granularity not in {"elementwise", "headwise"}:
            raise ValueError("gate_granularity must be 'elementwise' or 'headwise'")

        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.gate_granularity = gate_granularity

        self.qkv = nn.Linear(dim, dim * 3)
        gate_dim = dim if gate_granularity == "elementwise" else nhead
        self.gate_proj = nn.Linear(dim, gate_dim)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous()

        gate = torch.sigmoid(self.gate_proj(x))
        if self.gate_granularity == "elementwise":
            gate = gate.reshape(batch_size, num_tokens, self.nhead, self.head_dim)
        else:
            gate = gate[:, :, :, None]

        gated_output = attn_output * gate
        gated_output = gated_output.reshape(batch_size, num_tokens, self.dim)
        return self.proj(gated_output)


class FrequencyDomainSelfAttention1D(nn.Module):
    """Frequency-domain self-attention for 1D spectral token sequences."""

    def __init__(self, dim: int, num_frequency_bands: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if num_frequency_bands < 1:
            raise ValueError("num_frequency_bands must be greater than or equal to 1")

        self.num_frequency_bands = num_frequency_bands
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.frequency_mixer = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * 2, dim),
        )
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.size(1)
        spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
        amplitude = torch.abs(spectrum)

        if amplitude.size(1) > self.num_frequency_bands:
            pooled_amplitude = F.adaptive_avg_pool1d(
                amplitude.transpose(1, 2),
                self.num_frequency_bands,
            ).transpose(1, 2)
        else:
            pooled_amplitude = amplitude

        channel_gate = self.channel_gate(pooled_amplitude.mean(dim=1)).unsqueeze(1)
        frequency_delta = torch.tanh(self.frequency_mixer(amplitude))
        enhanced_spectrum = spectrum * (1.0 + self.frequency_scale * frequency_delta * channel_gate)

        x = torch.fft.irfft(enhanced_spectrum, n=num_tokens, dim=1, norm="ortho")
        return self.proj(x)


class HybridFrequencySelfAttention1D(nn.Module):
    """Frequency-domain attention with a local convolution residual branch."""

    def __init__(
        self,
        dim: int,
        num_frequency_bands: int = 64,
        local_kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if local_kernel_size < 1:
            raise ValueError("local_kernel_size must be greater than or equal to 1")

        self.frequency_attn = FrequencyDomainSelfAttention1D(dim, num_frequency_bands, dropout)
        self.local_branch = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=local_kernel_size,
                padding=local_kernel_size // 2,
                groups=dim,
                bias=False,
            ),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        self.local_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency_tokens = self.frequency_attn(x)
        local_tokens = self.local_branch(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(frequency_tokens + self.local_scale * local_tokens)


class CrossAttention1D(nn.Module):
    """Cross attention from current-scale queries to previous-scale memory tokens."""

    def __init__(self, query_dim: int, memory_dim: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if query_dim % nhead != 0:
            raise ValueError("query_dim must be divisible by nhead for cross attention")

        self.memory_proj = nn.Linear(memory_dim, query_dim) if memory_dim != query_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(query_dim, nhead, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        memory = self.memory_proj(memory)
        x, _ = self.attn(query=x, key=memory, value=memory, need_weights=False)
        return x


class HaarDWT1D(nn.Module):
    """One-level Haar discrete wavelet transform for 1D signals."""

    def __init__(self) -> None:
        super().__init__()
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, L] to [B, 2C, ceil(L / 2)]."""
        if x.dim() != 3:
            raise ValueError(f"HaarDWT1D expects [B, C, L], got shape {tuple(x.shape)}")

        if x.size(-1) % 2 == 1:
            x = F.pad(x, (0, 1), mode="replicate")

        even = x[..., 0::2]
        odd = x[..., 1::2]
        low = (even + odd) * self.inv_sqrt2
        high = (even - odd) * self.inv_sqrt2
        return torch.cat([low, high], dim=1)


class HaarIDWT1D(nn.Module):
    """Inverse one-level Haar wavelet transform for 1D signals."""

    def __init__(self) -> None:
        super().__init__()
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        """Convert [B, 2C, L] to [B, C, 2L], optionally cropped to output_size."""
        if x.dim() != 3:
            raise ValueError(f"HaarIDWT1D expects [B, 2C, L], got shape {tuple(x.shape)}")
        if x.size(1) % 2 != 0:
            raise ValueError("The channel dimension must be even for HaarIDWT1D.")

        low, high = torch.chunk(x, 2, dim=1)
        even = (low + high) * self.inv_sqrt2
        odd = (low - high) * self.inv_sqrt2

        out = torch.empty(
            x.size(0),
            low.size(1),
            low.size(2) * 2,
            device=x.device,
            dtype=x.dtype,
        )
        out[..., 0::2] = even
        out[..., 1::2] = odd

        if output_size is not None:
            out = out[..., :output_size]
        return out


class ELULinearAttention(nn.Module):
    """Linear attention with phi(x) = elu(x) + 1."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = F.elu(queries) + 1.0
        k = F.elu(keys) + 1.0

        if key_padding_mask is not None:
            mask = key_padding_mask[:, :, None, None].to(dtype=k.dtype, device=k.device)
            k = k * mask
            values = values * mask

        kv = torch.einsum("blhd,blhm->bhmd", k, values)
        normalizer = 1.0 / (torch.einsum("blhd,bhd->blh", q, k.sum(dim=1)) + self.eps)
        out = torch.einsum("blhd,bhmd,blh->blhm", q, kv, normalizer)
        return out.contiguous()


class SpectralAttention(nn.Module):
    """Fourier-wavelet spectral attention for 1D spectral token sequences."""

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
        is_filter: bool = True,
        input_format: str = "BLC",
    ) -> None:
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError(f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}")
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size should be even because the wavelet branch halves channels first.")
        if input_format not in {"BLC", "BCL"}:
            raise ValueError('input_format must be either "BLC" or "BCL".')

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.input_format = input_format

        self.w1 = nn.Parameter(
            0.02
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            0.02 * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            0.02
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(0.02 * torch.randn(2, self.num_blocks, self.block_size))

        self.num_heads = num_blocks
        self.reduced_channels = hidden_size // 2

        self.dwt = HaarDWT1D()
        self.idwt = HaarIDWT1D()

        self.reduce = nn.Sequential(
            nn.Conv1d(hidden_size, self.reduced_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.reduced_channels),
            nn.GELU(),
        )

        self.filter = (
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
            )
            if is_filter
            else nn.Identity()
        )

        self.qkv = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 3),
        )
        self.inner_attention = ELULinearAttention()

        self.proj = nn.Linear(hidden_size + self.reduced_channels, hidden_size)
        self.merge_linear = nn.Linear(hidden_size * 2, hidden_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv1d):
            fan_out = module.kernel_size[0] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _to_blc(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SpectralAttention expects a 3D tensor, got shape {tuple(x.shape)}")
        if self.input_format == "BCL":
            return x.transpose(1, 2).contiguous()
        return x

    def _from_blc(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_format == "BCL":
            return x.transpose(1, 2).contiguous()
        return x

    def _fourier_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, channels = x.shape
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho")
        x_freq = x_freq.reshape(batch_size, x_freq.shape[1], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            batch_size,
            x_freq.shape[1],
            self.num_blocks,
            self.block_size * self.hidden_size_factor,
            device=x.device,
            dtype=x.dtype,
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros_like(x_freq.real)
        o2_imag = torch.zeros_like(x_freq.imag)

        total_modes = length // 2 + 1
        kept_modes = max(1, int(total_modes * self.hard_thresholding_fraction))
        kept_modes = min(kept_modes, total_modes)

        o1_real[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x_freq[:, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x_freq[:, :kept_modes].imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x_freq[:, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x_freq[:, :kept_modes].real, self.w1[1])
            + self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[1])
            + self.b2[0]
        )
        o2_imag[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x_freq = torch.stack([o2_real, o2_imag], dim=-1)
        x_freq = torch.view_as_complex(x_freq).reshape(batch_size, x_freq.shape[1], channels)
        x_out = torch.fft.irfft(x_freq, n=length, dim=1, norm="ortho")
        return x_out + x

    def _wavelet_attention(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, length, channels = x.shape

        x_channels_first = x.transpose(1, 2).contiguous()
        x_dwt = self.dwt(self.reduce(x_channels_first))
        x_dwt = self.filter(x_dwt)

        dwt_length = x_dwt.size(-1)
        kv = x_dwt.transpose(1, 2).contiguous()
        qkv = self.qkv(kv)
        qkv = qkv.reshape(batch_size, dwt_length, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        dwt_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, length):
                raise ValueError("key_padding_mask should have shape [B, L] before wavelet downsampling.")
            if length % 2 == 1:
                key_padding_mask = F.pad(key_padding_mask, (0, 1), value=False)
            dwt_mask = key_padding_mask.reshape(batch_size, -1, 2).any(dim=-1)

        x_attn = self.inner_attention(q, k, v, dwt_mask)
        x_attn = x_attn.reshape(batch_size, dwt_length, channels).transpose(1, 2).contiguous()

        x_idwt = self.idwt(x_attn, output_size=length)
        x_idwt = x_idwt.transpose(1, 2).contiguous()
        return self.proj(torch.cat([x, x_idwt], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_blc = self._to_blc(x)
        if x_blc.size(-1) != self.hidden_size:
            raise ValueError(f"Expected feature dimension {self.hidden_size}, got {x_blc.size(-1)}.")

        x_fft = self._fourier_attention(x_blc)
        x_wave = self._wavelet_attention(x_blc, key_padding_mask)

        gate = torch.sigmoid(self.merge_linear(torch.cat([x_fft, x_wave], dim=-1)))
        out = gate * x_fft + (1.0 - gate) * x_wave
        return self._from_blc(out)


def make_self_attention(
    attention_type: str,
    dim: int,
    nhead: int,
    dropout: float,
    frequency_bands: int,
    local_kernel_size: int,
) -> nn.Module:
    """Build a self-attention module used by SpeCorformer blocks."""
    if attention_type == "mhsa":
        if dim % nhead != 0:
            raise ValueError("dim must be divisible by nhead when using mhsa")
        return MultiHeadSelfAttention1D(dim, nhead, dropout)
    if attention_type == "gated":
        return GatedSelfAttention1D(dim, nhead, dropout)
    if attention_type == "scsa":
        return SpectralCorrelationSelfAttention1D(dim, dropout)
    if attention_type == "fdsa":
        return FrequencyDomainSelfAttention1D(dim, frequency_bands, dropout)
    if attention_type == "hfsa":
        return HybridFrequencySelfAttention1D(dim, frequency_bands, local_kernel_size, dropout)
    raise ValueError("self_attention_type must be 'mhsa', 'gated', 'scsa', 'fdsa', or 'hfsa'")
