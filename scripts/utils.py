"""Utility helpers for LIBS deep-learning experiments.

The functions here are intentionally lightweight and reusable by future
``scr/dl/train.py`` code. PyTorch is imported lazily so this module can still be
checked in environments where torch is not installed.
"""

from __future__ import annotations

import csv
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


DEFAULT_RANDOM_SEED = 42
DEFAULT_DL_OUTPUT_ROOT = Path("experiments/dl/metal_classification")


def set_random_seed(seed: int = DEFAULT_RANDOM_SEED, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch if available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> str:
    """Return ``cuda`` when available, otherwise ``cpu``."""
    if preferred != "auto":
        return preferred
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_run_dir(output_root: str | Path = DEFAULT_DL_OUTPUT_ROOT, run_name: str = "") -> Path:
    """Create a deep-learning experiment directory with standard subfolders."""
    root = Path(output_root)
    run_dir = root / run_name if run_name else root / datetime.now().strftime("%Y%m%d-%H%M%S")
    for subdir in ("checkpoints", "predictions", "figures"):
        ensure_dir(run_dir / subdir)
    return run_dir


def to_jsonable(value: Any) -> Any:
    """Convert common NumPy values to JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: str | Path, payload: dict) -> None:
    """Save a dictionary as UTF-8 JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    """Load a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_rows_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Save a list of dictionaries to CSV."""
    if not rows and fieldnames is None:
        return

    output_path = Path(path)
    ensure_dir(output_path.parent)
    columns = fieldnames or list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(path: str | Path, history: list[dict]) -> None:
    """Save per-epoch training history."""
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "val_weighted_f1",
        "lr",
        "epoch_seconds",
    ]
    rows = []
    for row in history:
        rows.append({name: row.get(name, "") for name in fieldnames})
    save_rows_csv(path, rows, fieldnames)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute metal-classification metrics used by ML and DL baselines."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(_score_with_zero_division(precision_score, y_true, y_pred, average="macro")),
        "macro_recall": float(_score_with_zero_division(recall_score, y_true, y_pred, average="macro")),
        "macro_f1": float(_score_with_zero_division(f1_score, y_true, y_pred, average="macro")),
        "weighted_f1": float(_score_with_zero_division(f1_score, y_true, y_pred, average="weighted")),
    }


def _score_with_zero_division(metric_func, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Run a sklearn classification metric with quiet zero-division handling."""
    try:
        return metric_func(y_true, y_pred, zero_division=0, **kwargs)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            return metric_func(y_true, y_pred, **kwargs)


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> str:
    """Build a sklearn classification report without undefined-metric warnings."""
    target_names = label_names if label_names else None
    try:
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            return classification_report(y_true, y_pred, target_names=target_names)


def save_predictions_csv(
    path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
    probabilities: np.ndarray | None = None,
) -> None:
    """Save test-set predictions for metal classification."""
    rows: list[dict[str, Any]] = []
    labels = label_names or []
    for sample_id, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        true_index = int(true_label)
        pred_index = int(pred_label)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "y_true": true_index,
            "y_pred": pred_index,
            "true_name": labels[true_index] if labels else true_index,
            "pred_name": labels[pred_index] if labels else pred_index,
        }
        if probabilities is not None:
            for class_index, probability in enumerate(probabilities[sample_id]):
                class_name = labels[class_index] if labels else str(class_index)
                row[f"prob_{class_name}"] = float(probability)
        rows.append(row)
    save_rows_csv(path, rows)


def save_confusion_matrix_csv(path: str | Path, matrix: np.ndarray, label_names: list[str] | None = None) -> None:
    """Save a confusion matrix to CSV."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    labels = label_names if label_names else [str(index) for index in range(matrix.shape[0])]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def write_experiment_report(
    path: str | Path,
    model_name: str,
    dataset_root: str | Path,
    best_epoch: int,
    best_val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    params: dict,
) -> None:
    """Write a compact Markdown report for one DL experiment."""
    lines = [
        "# Deep Learning Metal Classification Report",
        "",
        f"- model: `{model_name}`",
        f"- dataset_root: `{dataset_root}`",
        f"- best_epoch: `{best_epoch}`",
        f"- params: `{json.dumps(to_jsonable(params), ensure_ascii=False)}`",
        "",
        "## Best Validation Metrics",
        "",
    ]
    lines.extend(f"- {name}: {value:.6f}" for name, value in best_val_metrics.items())
    lines.extend(["", "## Test Metrics", ""])
    lines.extend(f"- {name}: {value:.6f}" for name, value in test_metrics.items())

    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines), encoding="utf-8")
