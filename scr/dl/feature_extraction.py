"""Feature extraction helpers for trained LIBS classifiers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _find_final_linear(model: nn.Module) -> nn.Linear:
    """Return the final linear layer used as the classification projection."""
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and len(classifier) > 0 and isinstance(classifier[-1], nn.Linear):
        return classifier[-1]
    if isinstance(classifier, nn.Linear):
        return classifier

    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("model does not contain a Linear classification layer")
    return linear_layers[-1]


@torch.no_grad()
def extract_pre_classifier_features(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the feature vector immediately before the final classifier."""
    final_linear = _find_final_linear(model)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    captured_inputs: list[torch.Tensor] = []
    was_training = model.training

    def capture_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        captured_inputs.append(inputs[0].detach().cpu())

    hook = final_linear.register_forward_hook(capture_input)
    model.eval()
    try:
        for spectra, batch_labels in loader:
            captured_inputs.clear()
            spectra = spectra.to(device, non_blocking=True)
            _ = model(spectra)
            if not captured_inputs:
                raise RuntimeError("failed to capture pre-classifier features")
            batch_features = captured_inputs[0]
            features.append(batch_features.reshape(batch_features.shape[0], -1).numpy())
            labels.append(batch_labels.detach().cpu().numpy())
    finally:
        hook.remove()
        if was_training:
            model.train()

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def extract_input_saliency_by_class(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    num_classes: int,
    max_samples_per_class: int = 80,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Extract absolute input-gradient saliency maps and inputs grouped by true class."""
    if max_samples_per_class <= 0:
        raise ValueError("max_samples_per_class must be positive")

    saliency_by_class: dict[int, list[np.ndarray]] = {label: [] for label in range(num_classes)}
    inputs_by_class: dict[int, list[np.ndarray]] = {label: [] for label in range(num_classes)}
    was_training = model.training
    model.eval()
    try:
        for inputs, labels in loader:
            if all(len(items) >= max_samples_per_class for items in saliency_by_class.values()):
                break

            inputs = inputs.to(device, non_blocking=True).detach().requires_grad_(True)
            labels = labels.to(device, non_blocking=True)
            model.zero_grad()
            logits = model(inputs)
            target_scores = logits.gather(1, labels.view(-1, 1)).sum()
            target_scores.backward()

            batch_saliency = inputs.grad.detach().abs().cpu().numpy()
            batch_inputs = inputs.detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy().astype(int)
            for sample_saliency, sample_input, label in zip(batch_saliency, batch_inputs, batch_labels):
                class_items = saliency_by_class.get(int(label))
                if class_items is not None and len(class_items) < max_samples_per_class:
                    class_items.append(sample_saliency)
                    inputs_by_class[int(label)].append(sample_input)
    finally:
        model.zero_grad()
        if was_training:
            model.train()

    return {
        label: (np.asarray(items), np.asarray(inputs_by_class[label]))
        for label, items in saliency_by_class.items() if items
    }
