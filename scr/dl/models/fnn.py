"""FNN baseline for LIBS metal classification."""

from __future__ import annotations

import torch
from torch import nn

from .common import NUM_METAL_CLASSES, SIGNAL_LENGTH, copy_default_params, initialize_weights


FNN_DEFAULT_PARAMS = {
    "input_length": SIGNAL_LENGTH,
    "num_classes": NUM_METAL_CLASSES,
    "hidden_dims": (1024, 256, 64),
    "dropout": 0.3,
}


class FNN(nn.Module):
    """Fully connected baseline for flattened LIBS spectra."""

    def __init__(
        self,
        input_length: int = SIGNAL_LENGTH,
        num_classes: int = NUM_METAL_CLASSES,
        hidden_dims: tuple[int, ...] = (1024, 256, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_length
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def default_fnn_params() -> dict:
    """Return tunable default parameters for the FNN baseline."""
    return copy_default_params(FNN_DEFAULT_PARAMS)
