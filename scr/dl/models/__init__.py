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
from .factory import create_model, get_default_model_params, normalize_model_name
from .fnn import FNN, default_fnn_params
from .lstm import LSTM, default_lstm_params
from .resnet import BasicBlock1D, ResNet1D, default_resnet_params


__all__ = [
    "BasicBlock1D",
    "CNN",
    "ConvBlock",
    "FNN",
    "INPUT_CHANNELS",
    "LSTM",
    "MODEL_NAMES",
    "NUM_METAL_CLASSES",
    "ResNet1D",
    "SIGNAL_LENGTH",
    "count_trainable_parameters",
    "create_model",
    "default_cnn_params",
    "default_fnn_params",
    "default_lstm_params",
    "default_resnet_params",
    "get_default_model_params",
    "initialize_weights",
    "normalize_model_name",
]
