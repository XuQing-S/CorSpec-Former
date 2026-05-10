"""Model package for LIBS metal classification."""

from .cnn import CNN, ConvBlock, default_cnn_params
from .common import (
    INPUT_CHANNELS,
    MODEL_NAMES,
    NUM_METAL_CLASSES,
    SIGNAL_LENGTH,
    count_trainable_parameters,
    initialize_weights,
)
from .convnext import ConvNeXt1D, ConvNeXtBlock1D, default_convnext1d_params
from .factory import create_model, get_default_model_params, normalize_model_name
from .fnn import FNN, default_fnn_params
from .inception import Inception1D, InceptionBlock1D, default_inception1d_params
from .lstm import LSTM, default_lstm_params
from .resnet import BasicBlock1D, ResNet1D, default_resnet_params
from .specorformer import (
    CrossAttention1D,
    FrequencyDomainSelfAttention1D,
    HybridFrequencySelfAttention1D,
    SpeCorformer1D,
    SpeCorformerDecoderBlock1D,
    SpeCorformerEncoderBlock1D,
    SpectralCorrelationSelfAttention1D,
    SpectrumPatchEmbedding1D,
    default_specorformer1d_params,
)
from .transformer import Transformer1D, default_transformer1d_params
from .vgg import VGG1D, VGGBlock1D, default_vgg1d_params


__all__ = [
    "BasicBlock1D",
    "CNN",
    "ConvNeXt1D",
    "ConvNeXtBlock1D",
    "ConvBlock",
    "FNN",
    "INPUT_CHANNELS",
    "Inception1D",
    "InceptionBlock1D",
    "LSTM",
    "MODEL_NAMES",
    "NUM_METAL_CLASSES",
    "ResNet1D",
    "SIGNAL_LENGTH",
    "CrossAttention1D",
    "FrequencyDomainSelfAttention1D",
    "HybridFrequencySelfAttention1D",
    "SpeCorformer1D",
    "SpeCorformerDecoderBlock1D",
    "SpeCorformerEncoderBlock1D",
    "SpectralCorrelationSelfAttention1D",
    "SpectrumPatchEmbedding1D",
    "Transformer1D",
    "VGG1D",
    "VGGBlock1D",
    "count_trainable_parameters",
    "create_model",
    "default_cnn_params",
    "default_convnext1d_params",
    "default_fnn_params",
    "default_inception1d_params",
    "default_lstm_params",
    "default_resnet_params",
    "default_specorformer1d_params",
    "default_transformer1d_params",
    "default_vgg1d_params",
    "get_default_model_params",
    "initialize_weights",
    "normalize_model_name",
]
