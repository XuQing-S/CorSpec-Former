"""Interpretability output helpers for LIBS classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.visualize import plot_saliency_heatmap, plot_spectrum_heatmap
from scr.dl.feature_extraction import extract_input_saliency_by_class


def safe_label_name(label_name: str) -> str:
    """Return a filesystem-friendly label name."""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label_name)


def save_saliency_heatmaps_by_class(
    figures_dir: Path,
    model: nn.Module,
    loader,
    label_names: list[str],
    device: torch.device,
    *,
    split_name: str,
    max_samples_per_class: int = 80,
) -> None:
    """Save one input-gradient saliency heatmap for each metal class."""
    num_classes = len(label_names) if label_names else 6
    saliency_by_class = extract_input_saliency_by_class(
        model,
        loader,
        device,
        num_classes=num_classes,
        max_samples_per_class=max_samples_per_class,
    )
    saliency_dir = figures_dir / "saliency_by_class"
    for label, (saliency, spectra) in saliency_by_class.items():
        display_name = label_names[label] if label < len(label_names) else str(label)
        plot_saliency_heatmap(
            saliency_dir / f"{split_name}_{label:02d}_{safe_label_name(display_name)}_saliency_heatmap.png",
            saliency,
            spectra=spectra,
            max_samples=max_samples_per_class,
            title=f"{split_name.capitalize()} {display_name} Input Saliency Heatmap",
        )


def save_spectrum_heatmaps_by_class(
    figures_dir: Path,
    spectra: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    *,
    split_name: str,
    max_samples_per_class: int = 80,
) -> None:
    """Save one row-normalized spectrum heatmap for each metal class."""
    matrix = np.asarray(spectra)
    labels = np.asarray(labels).reshape(-1).astype(int)
    spectrum_dir = figures_dir / "spectrum_heatmaps_by_class"
    for label in sorted({int(value) for value in labels}):
        class_spectra = matrix[labels == label]
        if class_spectra.size == 0:
            continue
        display_name = label_names[label] if label < len(label_names) else str(label)
        plot_spectrum_heatmap(
            spectrum_dir / f"{split_name}_{label:02d}_{safe_label_name(display_name)}_spectrum_heatmap.png",
            class_spectra,
            max_samples=max_samples_per_class,
            title=f"{split_name.capitalize()} {display_name} Spectrum Heatmap",
        )
