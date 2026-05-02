"""Model factory and tunable defaults for LIBS metal classification."""

from __future__ import annotations

from torch import nn

from .cnn import CNN, default_cnn_params
from .common import MODEL_NAMES, NUM_METAL_CLASSES
from .fnn import FNN, default_fnn_params
from .lstm import LSTM, default_lstm_params
from .resnet import ResNet1D, default_resnet_params


MODEL_ALIASES = {
    "fnn": "fnn",
    "mlp": "fnn",
    "cnn": "cnn",
    "1d_cnn": "cnn",
    "cnn1d": "cnn",
    "lstm": "lstm",
    "cnn_lstm": "lstm",
    "conv_lstm": "lstm",
    "resnet": "resnet",
    "1d_resnet": "resnet",
    "resnet1d": "resnet",
}

MODEL_CLASSES = {
    "fnn": FNN,
    "cnn": CNN,
    "lstm": LSTM,
    "resnet": ResNet1D,
}

MODEL_DEFAULTS = {
    "fnn": default_fnn_params,
    "cnn": default_cnn_params,
    "lstm": default_lstm_params,
    "resnet": default_resnet_params,
}


def normalize_model_name(model_name: str) -> str:
    normalized_name = model_name.strip().lower().replace("-", "_")
    if normalized_name not in MODEL_ALIASES:
        raise ValueError(f"unsupported model: {model_name!r}; expected one of {MODEL_NAMES}")
    return MODEL_ALIASES[normalized_name]


def get_default_model_params(model_name: str) -> dict:
    """Return a mutable copy of tunable default parameters for a model."""
    normalized_name = normalize_model_name(model_name)
    return MODEL_DEFAULTS[normalized_name]()


def create_model(model_name: str, num_classes: int = NUM_METAL_CLASSES, **overrides) -> nn.Module:
    """Create a metal classification model by name.

    Hyperparameters can be tuned by passing keyword overrides, for example:
    ``create_model("cnn", channels=(16, 32, 64, 128), dropout=0.2)``.
    """
    normalized_name = normalize_model_name(model_name)
    params = get_default_model_params(normalized_name)
    params.update(overrides)
    params["num_classes"] = num_classes
    return MODEL_CLASSES[normalized_name](**params)
