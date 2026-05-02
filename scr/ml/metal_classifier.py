"""Train common machine-learning models for metal classification.

The script uses feature matrices generated under ``data/features`` and metal
labels from ``data/processed``. It trains one selected classifier, evaluates on
validation and test splits, and saves metrics, predictions, model, and a
confusion-matrix figure under ``experiments/ml/metal_classification``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

try:
    import joblib
except ModuleNotFoundError:
    joblib = None


FEATURE_NAME_MAP = {                    # 特征类型映射; 可选 feature,pca,stat,pls_metal
    "feature": "x_feature",
    "pca": "x_pca",
    "stat": "x_stat",
    "pls_metal": "x_pls_metal",
}
MODEL_NAMES = ("logistic_regression", "svm", "random_forest", "extra_trees")  # 机器学习模型类型
DEFAULT_RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a metal classification model.")
    parser.add_argument("--feature", choices=sorted(FEATURE_NAME_MAP), default="pls_metal", help="输入特征类型")
    parser.add_argument("--model", choices=MODEL_NAMES, default="logistic_regression", help="机器学习模型类型")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/noise_shift/3"), help="特征目录")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/noise_shift/3"), help="标签目录")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/ml/metal_classification"), help="实验输出根目录")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="随机种子")
    parser.add_argument("--grid-search", action="store_true", help="启用小范围网格搜索")
    parser.add_argument("--run-name", type=str, default="", help="自定义运行目录名")
    return parser.parse_args()


def load_features(features_root: Path, feature: str) -> dict[str, np.ndarray]:
    feature_suffix = FEATURE_NAME_MAP[feature]
    return {
        split: np.load(features_root / f"{split}_{feature_suffix}.npy").astype(np.float32)
        for split in ("train", "val", "test")
    }


def load_labels(processed_root: Path) -> dict[str, np.ndarray]:
    return {
        split: np.load(processed_root / f"{split}_y_metal.npy").astype(np.int64)
        for split in ("train", "val", "test")
    }


def load_label_names(processed_root: Path) -> list[str]:
    mapping_path = processed_root / "label_mapping.json"
    if not mapping_path.exists():
        return []

    with mapping_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    metal_mapping = payload.get("metal", {})
    index_to_name = {int(index): name for name, index in metal_mapping.items()}
    return [index_to_name[index] for index in sorted(index_to_name)]


def create_model(model_name: str, random_seed: int, params: dict | None = None):
    params = params or {}
    if model_name == "logistic_regression":
        defaults = {
            "max_iter": 5000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": random_seed,
        }
        defaults.update(params)
        return LogisticRegression(**defaults)
    if model_name == "svm":
        defaults = {
            "C": 10.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": False,
            "random_state": random_seed,
        }
        defaults.update(params)
        return SVC(**defaults)
    if model_name == "random_forest":
        defaults = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": random_seed,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if model_name == "extra_trees":
        defaults = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": random_seed,
        }
        defaults.update(params)
        return ExtraTreesClassifier(**defaults)
    raise ValueError(f"Unsupported model: {model_name}")


def get_param_grid(model_name: str) -> list[dict]:
    if model_name == "logistic_regression":
        return list(ParameterGrid({"C": [0.1, 1.0, 10.0]}))
    if model_name == "svm":
        return list(ParameterGrid({"C": [0.1, 1.0, 10.0, 100.0], "kernel": ["linear", "rbf"], "gamma": ["scale"]}))
    if model_name == "random_forest":
        return list(ParameterGrid({"n_estimators": [200, 500], "max_depth": [None, 20, 40]}))
    if model_name == "extra_trees":
        return list(ParameterGrid({"n_estimators": [200, 500], "max_depth": [None, 20, 40]}))
    return [{}]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def train_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
    grid_search: bool,
) -> tuple[object, dict, list[dict]]:
    search_params = get_param_grid(model_name) if grid_search else [{}]
    best_model = None
    best_params: dict = {}
    best_score = -1.0
    search_rows: list[dict] = []

    for params in search_params:
        model = create_model(model_name, random_seed, params)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_metrics = compute_metrics(y_val, val_pred)
        row = {"params": json.dumps(params, ensure_ascii=False), **val_metrics}
        search_rows.append(row)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_params = params
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained.")
    return best_model, best_params, search_rows


def make_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        run_dir = output_root / run_name
    else:
        run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_object(path: Path, payload: object) -> None:
    if joblib is not None:
        joblib.dump(payload, path)
        return
    with path.open("wb") as file:
        pickle.dump(payload, file)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_search_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "y_true", "y_pred", "true_name", "pred_name"])
        writer.writeheader()
        for sample_id, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            true_index = int(true_label)
            pred_index = int(pred_label)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "y_true": true_index,
                    "y_pred": pred_index,
                    "true_name": label_names[true_index] if label_names else true_index,
                    "pred_name": label_names[pred_index] if label_names else pred_index,
                }
            )


def save_confusion_matrix_csv(path: Path, matrix: np.ndarray, label_names: list[str]) -> None:
    labels = label_names if label_names else [str(index) for index in range(matrix.shape[0])]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def plot_confusion_matrix(path: Path, matrix: np.ndarray, label_names: list[str]) -> None:
    labels = label_names if label_names else [str(index) for index in range(matrix.shape[0])]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Metal Classification Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    feature: str,
    model_name: str,
    best_params: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    lines = [
        "# Metal Classification Report",
        "",
        f"- feature: `{feature}`",
        f"- model: `{model_name}`",
        f"- best_params: `{json.dumps(best_params, ensure_ascii=False)}`",
        "",
        "## Validation Metrics",
        "",
    ]
    lines.extend(f"- {name}: {value:.6f}" for name, value in val_metrics.items())
    lines.extend(["", "## Test Metrics", ""])
    lines.extend(f"- {name}: {value:.6f}" for name, value in test_metrics.items())
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    features = load_features(args.features_root, args.feature)
    labels = load_labels(args.processed_root)
    label_names = load_label_names(args.processed_root)
    run_dir = make_run_dir(args.output_root, args.run_name)

    model, best_params, search_rows = train_model(
        args.model,
        features["train"],
        labels["train"],
        features["val"],
        labels["val"],
        args.random_seed,
        args.grid_search,
    )

    val_pred = model.predict(features["val"])
    test_pred = model.predict(features["test"])
    val_metrics = compute_metrics(labels["val"], val_pred)
    test_metrics = compute_metrics(labels["test"], test_pred)
    test_matrix = confusion_matrix(labels["test"], test_pred)

    config = {
        "feature": args.feature,
        "model": args.model,
        "best_params": best_params,
        "random_seed": args.random_seed,
        "grid_search": args.grid_search,
        "features_root": str(args.features_root),
        "processed_root": str(args.processed_root),
    }
    metrics = {"val": val_metrics, "test": test_metrics}
    report_text = classification_report(
        labels["test"],
        test_pred,
        target_names=label_names if label_names else None,
    )

    model_path = run_dir / "models" / f"metal_{args.feature}_{args.model}.joblib"
    save_object(model_path, {"model": model, "config": config, "label_names": label_names})
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "classification_report.json", {"text": report_text})
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_search_rows(run_dir / "validation_search.csv", search_rows)
    save_predictions(run_dir / "predictions" / f"metal_{args.feature}_{args.model}_test.csv", labels["test"], test_pred, label_names)
    save_confusion_matrix_csv(run_dir / "confusion_matrix.csv", test_matrix, label_names)
    plot_confusion_matrix(run_dir / "figures" / "confusion_matrix.png", test_matrix, label_names)
    write_report(run_dir / "report.md", args.feature, args.model, best_params, val_metrics, test_metrics)

    print(
        "metal classification finished: "
        f"feature={args.feature}, model={args.model}, "
        f"val_macro_f1={val_metrics['macro_f1']:.4f}, test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
