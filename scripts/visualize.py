"""Visualization helpers for LIBS deep-learning experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_figure(fig, path: str | Path, *, dpi: int = 180) -> None:
    """Save a figure with explicit image format and verify it can be decoded."""
    output_path = _ensure_parent(path)
    suffix = output_path.suffix.lower().lstrip(".") or "png"
    image_format = "png" if suffix == "png" else suffix
    fig.savefig(
        output_path,
        format=image_format,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    try:
        plt.imread(str(output_path))
    except Exception as error:
        raise RuntimeError(f"saved figure cannot be decoded: {output_path}") from error


def _extract_series(history: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in history:
        value = row.get(key)
        if value in (None, ""):
            values.append(np.nan)
        else:
            values.append(float(value))
    return values


def _as_2d_features(values: np.ndarray) -> np.ndarray:
    """Convert spectra or feature tensors to ``[N, D]`` matrices."""
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        return matrix.reshape(1, -1)
    if matrix.ndim == 2:
        return matrix
    return matrix.reshape(matrix.shape[0], -1)


def _as_1d_spectrum(values: np.ndarray) -> np.ndarray:
    """Convert one spectrum-like array to a 1D vector."""
    spectrum = np.asarray(values)
    return spectrum.reshape(-1)


def _label_names(labels: np.ndarray, label_names: list[str] | None = None) -> list[str]:
    if label_names:
        return label_names
    unique_labels = sorted({int(label) for label in labels})
    return [str(label) for label in unique_labels]


def _reduce_to_2d(
    features: np.ndarray,
    *,
    method: str = "pca",
    standardize: bool = True,
    random_seed: int = 42,
    max_samples: int | None = None,
    perplexity: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D embedding and selected row indices."""
    matrix = _as_2d_features(features).astype(np.float64)
    indices = np.arange(matrix.shape[0])
    if max_samples is not None and matrix.shape[0] > max_samples:
        rng = np.random.RandomState(random_seed)  # pylint: disable=no-member
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
        matrix = matrix[indices]

    if standardize:
        matrix = StandardScaler().fit_transform(matrix)

    normalized_method = method.lower()
    if normalized_method == "pca":
        embedding = PCA(n_components=2, random_state=random_seed).fit_transform(matrix)
    elif normalized_method in {"tsne", "t-sne"}:
        effective_perplexity = min(perplexity, max(1.0, (matrix.shape[0] - 1) / 3.0))
        embedding = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            random_state=random_seed,
        ).fit_transform(matrix)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    return embedding, indices


def plot_loss_curve(path: str | Path, history: list[dict]) -> None:
    """Plot train and validation loss curves."""
    output_path = _ensure_parent(path)
    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    train_loss = _extract_series(history, "train_loss")
    val_loss = _extract_series(history, "val_loss")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss, label="Validation Loss", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_macro_f1_curve(path: str | Path, history: list[dict]) -> None:
    """Plot validation Macro-F1 over epochs."""
    output_path = _ensure_parent(path)
    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    macro_f1 = _extract_series(history, "val_macro_f1")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, macro_f1, label="Validation Macro-F1", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Validation Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_confusion_matrix(
    path: str | Path,
    matrix: np.ndarray,
    label_names: list[str] | None = None,
    *,
    normalize: bool = False,
    title: str = "Metal Classification Confusion Matrix",
) -> None:
    """Plot a confusion matrix for metal classification."""
    output_path = _ensure_parent(path)
    labels = label_names if label_names else [str(index) for index in range(matrix.shape[0])]
    display_matrix = matrix.astype(np.float64)
    if normalize:
        row_sum = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(display_matrix, row_sum, out=np.zeros_like(display_matrix), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(display_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = display_matrix.max() / 2.0 if display_matrix.size else 0.0
    for row_index in range(display_matrix.shape[0]):
        for col_index in range(display_matrix.shape[1]):
            value = display_matrix[row_index, col_index]
            text = f"{value:.2f}" if normalize else str(int(matrix[row_index, col_index]))
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_metrics_bar(
    path: str | Path,
    metrics: dict[str, float],
    *,
    title: str = "Classification Metrics",
) -> None:
    """Plot one experiment's classification metrics as a bar chart."""
    output_path = _ensure_parent(path)
    names = list(metrics.keys())
    values = [float(metrics[name]) for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_embedding_2d(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | None = None,
    *,
    method: str = "pca",
    title: str | None = None,
    standardize: bool = True,
    random_seed: int = 42,
    max_samples: int | None = 3000,
    perplexity: float = 30.0,
) -> None:
    """Plot a 2D PCA/t-SNE embedding colored by metal class."""
    output_path = _ensure_parent(path)
    labels = np.asarray(labels).reshape(-1)
    embedding, indices = _reduce_to_2d(
        features,
        method=method,
        standardize=standardize,
        random_seed=random_seed,
        max_samples=max_samples,
        perplexity=perplexity,
    )
    selected_labels = labels[indices]
    names = _label_names(selected_labels, label_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = sorted({int(label) for label in selected_labels})
    cmap = plt.get_cmap("tab10")
    for color_index, label in enumerate(unique_labels):
        mask = selected_labels == label
        display_name = names[label] if label < len(names) else str(label)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.75,
            color=cmap(color_index % 10),
            label=display_name,
        )
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title or f"{method.upper()} Embedding")
    ax.grid(True, alpha=0.25)
    ax.legend(markerscale=1.5, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_split_embedding_2d(
    path: str | Path,
    features_by_split: dict[str, np.ndarray],
    *,
    method: str = "pca",
    title: str | None = None,
    standardize: bool = True,
    random_seed: int = 42,
    max_samples: int | None = 3000,
    perplexity: float = 30.0,
) -> None:
    """Plot train/val/test distribution in one 2D embedding."""
    split_names = list(features_by_split)
    matrix = np.concatenate([_as_2d_features(features_by_split[name]) for name in split_names], axis=0)
    split_labels = np.concatenate(
        [np.full(_as_2d_features(features_by_split[name]).shape[0], index) for index, name in enumerate(split_names)]
    )
    plot_embedding_2d(
        path,
        matrix,
        split_labels,
        split_names,
        method=method,
        title=title or f"{method.upper()} Split Distribution",
        standardize=standardize,
        random_seed=random_seed,
        max_samples=max_samples,
        perplexity=perplexity,
    )


def plot_class_mean_spectra(
    path: str | Path,
    spectra: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | None = None,
    *,
    wavelength: np.ndarray | None = None,
    title: str = "Class Mean Spectra",
) -> None:
    """Plot one mean spectrum per metal class."""
    output_path = _ensure_parent(path)
    matrix = _as_2d_features(spectra)
    labels = np.asarray(labels).reshape(-1)
    x_axis = np.asarray(wavelength).reshape(-1) if wavelength is not None else np.arange(matrix.shape[1])
    names = _label_names(labels, label_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label in sorted({int(label) for label in labels}):
        class_spectra = matrix[labels == label]
        mean_spectrum = class_spectra.mean(axis=0)
        display_name = names[label] if label < len(names) else str(label)
        ax.plot(x_axis, mean_spectrum, linewidth=1.2, label=display_name)
    ax.set_xlabel("Wavelength Index" if wavelength is None else "Wavelength")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_class_spectrum_band(
    path: str | Path,
    spectra: np.ndarray,
    labels: np.ndarray,
    label: int,
    label_name: str | None = None,
    *,
    wavelength: np.ndarray | None = None,
    percentile_band: tuple[float, float] = (25.0, 75.0),
    title: str | None = None,
) -> None:
    """Plot mean spectrum and percentile band for one metal class."""
    output_path = _ensure_parent(path)
    matrix = _as_2d_features(spectra)
    labels = np.asarray(labels).reshape(-1)
    class_spectra = matrix[labels == label]
    if class_spectra.size == 0:
        raise ValueError(f"no spectra found for label {label}")

    x_axis = np.asarray(wavelength).reshape(-1) if wavelength is not None else np.arange(matrix.shape[1])
    mean_spectrum = class_spectra.mean(axis=0)
    lower = np.percentile(class_spectra, percentile_band[0], axis=0)
    upper = np.percentile(class_spectra, percentile_band[1], axis=0)
    display_name = label_name or str(label)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, mean_spectrum, linewidth=1.2, label=f"{display_name} mean")
    ax.fill_between(x_axis, lower, upper, alpha=0.25, label=f"{percentile_band[0]:.0f}-{percentile_band[1]:.0f} percentile")
    ax.set_xlabel("Wavelength Index" if wavelength is None else "Wavelength")
    ax.set_ylabel("Intensity")
    ax.set_title(title or f"{display_name} Spectrum Band")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_spectrum_heatmap(
    path: str | Path,
    spectra: np.ndarray,
    *,
    max_samples: int = 80,
    title: str = "Spectrum Heatmap",
) -> None:
    """Plot multiple spectra as a row-normalized heatmap, with mean spectrum overlay."""
    output_path = _ensure_parent(path)
    matrix = _as_2d_features(spectra).astype(np.float64)
    if matrix.shape[0] == 0:
        raise ValueError("spectra must contain at least one sample")
    
    spectra_matrix = matrix
    if matrix.shape[0] > max_samples:
        matrix = matrix[:max_samples]
        spectra_matrix = spectra_matrix[:max_samples]

    row_mean = matrix.mean(axis=1, keepdims=True)
    row_std = matrix.std(axis=1, keepdims=True) + 1e-8
    display_matrix = (matrix - row_mean) / row_std
    lower, upper = np.percentile(display_matrix, [1.0, 99.0])
    display_matrix = np.clip(display_matrix, lower, upper)

    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(display_matrix, aspect="auto", interpolation="nearest", cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="Row-normalized Intensity", fraction=0.035, pad=0.02)
    ax.set_xlabel("Wavelength Index")
    ax.set_ylabel("Sample")
    ax.set_title(title)

    mean_spectrum = spectra_matrix.mean(axis=0)
    mean_spectrum_norm = (mean_spectrum - mean_spectrum.min()) / (mean_spectrum.max() - mean_spectrum.min() + 1e-8)
    overlay_y = (1.0 - mean_spectrum_norm) * max(matrix.shape[0] - 1, 1)
    ax.plot(overlay_y, linewidth=2.5, color="white", alpha=0.7)
    ax.plot(overlay_y, linewidth=1.5, color="cyan", alpha=0.9, label="Mean Spectrum (Scaled)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), borderaxespad=0.0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_saliency_curve(
    path: str | Path,
    spectrum: np.ndarray,
    saliency: np.ndarray,
    *,
    wavelength: np.ndarray | None = None,
    title: str = "Spectrum Saliency",
) -> None:
    """Plot one spectrum together with a saliency curve."""
    output_path = _ensure_parent(path)
    spectrum_1d = _as_1d_spectrum(spectrum)
    saliency_1d = np.abs(_as_1d_spectrum(saliency))
    if spectrum_1d.shape[0] != saliency_1d.shape[0]:
        raise ValueError("spectrum and saliency must have the same length")

    x_axis = np.asarray(wavelength).reshape(-1) if wavelength is not None else np.arange(spectrum_1d.shape[0])
    normalized_saliency = saliency_1d / (saliency_1d.max() + 1e-8)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_axis, spectrum_1d, linewidth=1.0, color="tab:blue", label="Spectrum")
    ax1.set_xlabel("Wavelength Index" if wavelength is None else "Wavelength")
    ax1.set_ylabel("Intensity", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x_axis, normalized_saliency, linewidth=1.0, color="tab:red", alpha=0.8, label="Saliency")
    ax2.set_ylabel("Normalized Saliency", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0.0, 1.05)
    ax1.set_title(title)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_saliency_heatmap(
    path: str | Path,
    saliency: np.ndarray,
    labels: np.ndarray | None = None,
    label_names: list[str] | None = None,
    *,
    spectra: np.ndarray | None = None,
    max_samples: int = 80,
    title: str = "Saliency Heatmap",
) -> None:
    """Plot saliency values for multiple spectra as a heatmap, optionally with mean spectrum overlay."""
    output_path = _ensure_parent(path)
    matrix = np.abs(_as_2d_features(saliency))
    if labels is not None:
        order = np.argsort(np.asarray(labels).reshape(-1))
        matrix = matrix[order]
        labels = np.asarray(labels).reshape(-1)[order]
        if spectra is not None:
            spectra = _as_2d_features(spectra)[order]
    
    if spectra is not None:
        spectra_matrix = _as_2d_features(spectra)

    if matrix.shape[0] > max_samples:
        matrix = matrix[:max_samples]
        if labels is not None:
            labels = labels[:max_samples]
        if spectra is not None:
            spectra_matrix = spectra_matrix[:max_samples]

    row_max = matrix.max(axis=1, keepdims=True) + 1e-8
    display_matrix = matrix / row_max

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(display_matrix, aspect="auto", interpolation="nearest", cmap="magma")
    fig.colorbar(image, ax=ax, label="Normalized Saliency", fraction=0.035, pad=0.02)
    ax.set_xlabel("Wavelength Index")
    ax.set_ylabel("Sample")
    ax.set_title(title)

    if spectra is not None:
        mean_spectrum = spectra_matrix.mean(axis=0)
        mean_spectrum_norm = (mean_spectrum - mean_spectrum.min()) / (mean_spectrum.max() - mean_spectrum.min() + 1e-8)
        overlay_y = (1.0 - mean_spectrum_norm) * max(matrix.shape[0] - 1, 1)
        ax.plot(overlay_y, linewidth=2.5, color="white", alpha=0.7)
        ax.plot(overlay_y, linewidth=1.5, color="cyan", alpha=0.9, label="Mean Spectrum (Scaled)")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), borderaxespad=0.0)

    if labels is not None and label_names:
        tick_positions = []
        tick_labels = []
        for label in sorted({int(value) for value in labels}):
            positions = np.where(labels == label)[0]
            if positions.size:
                tick_positions.append(int(positions.mean()))
                tick_labels.append(label_names[label] if label < len(label_names) else str(label))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_top_saliency_regions(
    path: str | Path,
    saliency: np.ndarray,
    *,
    top_k: int = 20,
    window_size: int = 16,
    title: str = "Top Saliency Regions",
) -> None:
    """Plot the strongest saliency windows along the spectrum."""
    output_path = _ensure_parent(path)
    saliency_1d = np.abs(_as_1d_spectrum(saliency))
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    usable_length = (saliency_1d.shape[0] // window_size) * window_size
    if usable_length == 0:
        raise ValueError("saliency length is shorter than window_size")
    window_scores = saliency_1d[:usable_length].reshape(-1, window_size).mean(axis=1)
    top_indices = np.argsort(window_scores)[-top_k:][::-1]
    labels = [f"{index * window_size}-{(index + 1) * window_size - 1}" for index in top_indices]
    scores = window_scores[top_indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(scores)), scores)
    ax.set_xticks(np.arange(len(scores)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Wavelength Index Window")
    ax.set_ylabel("Mean Saliency")
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_activation_map(
    path: str | Path,
    activation: np.ndarray,
    *,
    title: str = "Activation Map",
) -> None:
    """Plot a CNN activation tensor shaped ``[channels, length]`` as a heatmap."""
    output_path = _ensure_parent(path)
    matrix = np.asarray(activation)
    if matrix.ndim == 3:
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise ValueError("activation must have shape [channels, length] or [1, channels, length]")

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    fig.colorbar(image, ax=ax, label="Activation")
    ax.set_xlabel("Reduced Spectrum Position")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def save_training_figures(figures_dir: str | Path, history: list[dict]) -> None:
    """Save standard training curves into a figures directory."""
    directory = Path(figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(directory / "loss_curve.png", history)
    plot_macro_f1_curve(directory / "macro_f1_curve.png", history)
