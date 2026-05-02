"""Shared utilities for LIBS metal classification models."""

from __future__ import annotations

from copy import deepcopy

from torch import nn


INPUT_CHANNELS = 1
SIGNAL_LENGTH = 8192
NUM_METAL_CLASSES = 6
MODEL_NAMES = ("fnn", "cnn", "lstm", "resnet")


def initialize_weights(module: nn.Module) -> None:
    """Initialize common trainable layers with stable defaults."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def copy_default_params(params: dict) -> dict:
    """Return a defensive copy of a model default-parameter dictionary."""
    return deepcopy(params)
