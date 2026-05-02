"""Evaluate a trained LIBS deep-learning metal classifier."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        _ = kwargs
        return iterable

from scripts.utils import (
    classification_report_text,
    compute_classification_metrics,
    ensure_dir,
    get_device,
    load_json,
    save_confusion_matrix_csv,
    save_json,
    save_predictions_csv,
)
from scripts.visualize import (
    plot_class_mean_spectra,
    plot_confusion_matrix,
    plot_embedding_2d,
    plot_metrics_bar,
)
from scr.dl.dataset import LIBSDataset, LIBSFeatureDataset, create_dataloader, load_metal_label_names
from scr.dl.feature_extraction import extract_pre_classifier_features
from scr.dl.interpretability import save_saliency_heatmaps_by_class, save_spectrum_heatmaps_by_class
from scr.dl.models import create_model


DEFAULT_PROCESSED_ROOT = Path("data/processed/balanced")
DEFAULT_FEATURES_ROOT = Path("data/features/balanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LIBS metal classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint 路径，通常为 best_model.pt")
    parser.add_argument("--config", type=Path, default=None, help="训练生成的 config.json；默认自动查找 checkpoint 上两级目录")
    parser.add_argument("--processed-root", type=Path, default=None, help="测试数据目录；默认使用 config 中的 processed_root")
    parser.add_argument("--features-root", type=Path, default=None, help="特征目录；默认使用 config 中的 features_root")
    parser.add_argument("--input-kind", choices=("spectrum", "feature"), default=None, help="输入类型；默认使用 config 中的 input_kind")
    parser.add_argument("--feature-name", choices=("feature", "pca", "stat", "pls_metal"), default=None, help="特征类型；默认使用 config 中的 feature_name")
    parser.add_argument("--output-dir", type=Path, default=None, help="测试输出目录；默认写入 run_dir/test_eval")
    parser.add_argument("--batch-size", type=int, default=None, help="测试 batch size；默认使用 config 中 batch_size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数量")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda 或具体设备名")
    parser.add_argument("--disable-progress", action="store_true", help="关闭 tqdm 测试进度条")
    parser.add_argument("--no-figures", action="store_true", help="不保存测试可视化图")
    parser.add_argument("--save-embedding", action="store_true", help="额外保存测试集 PCA embedding")
    parser.add_argument("--visualize-max-samples", type=int, default=3000, help="降维可视化最大抽样样本数")
    parser.add_argument("--saliency-max-samples", type=int, default=80, help="每类 saliency 热图最多抽样样本数")
    return parser.parse_args()


def infer_config_path(checkpoint_path: Path) -> Path:
    run_dir = checkpoint_path.parent.parent
    return run_dir / "config.json"


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    show_progress: bool,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    progress = tqdm(loader, desc="test", leave=False, disable=not show_progress)
    for spectra, labels in progress:
        spectra = spectra.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(spectra)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=total_loss / max(total_samples, 1))

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    metrics = compute_classification_metrics(y_true, y_pred)
    return total_loss / max(total_samples, 1), metrics, y_true, y_pred, probabilities


def write_test_report(
    path: Path,
    checkpoint_path: Path,
    processed_root: Path,
    input_kind: str,
    metrics: dict[str, float],
    classification_text: str,
) -> None:
    lines = [
        "# Deep Learning Test Report",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- processed_root: `{processed_root}`",
        f"- input_kind: `{input_kind}`",
        "",
        "## Test Metrics",
        "",
    ]
    lines.extend(f"- {name}: {value:.6f}" for name, value in metrics.items())
    lines.extend(["", "## Classification Report", "", "```text", classification_text, "```"])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_test_figures(
    output_dir: Path,
    model: nn.Module,
    loader,
    dataset: LIBSDataset,
    matrix: np.ndarray,
    metrics: dict[str, float],
    y_true: np.ndarray,
    label_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if args.no_figures:
        return

    figures_dir = ensure_dir(output_dir / "figures")
    plot_confusion_matrix(figures_dir / "confusion_matrix.png", matrix, label_names)
    plot_confusion_matrix(
        figures_dir / "confusion_matrix_normalized.png",
        matrix,
        label_names,
        normalize=True,
        title="Normalized Test Confusion Matrix",
    )
    plot_metrics_bar(
        figures_dir / "test_metrics.png",
        {key: value for key, value in metrics.items() if key != "loss"},
        title="Test Metrics",
    )
    if isinstance(dataset, LIBSDataset):
        plot_class_mean_spectra(
            figures_dir / "test_class_mean_spectra.png",
            np.asarray(dataset.x),
            y_true,
            label_names,
            title="Test Class Mean Spectra",
        )
        save_spectrum_heatmaps_by_class(
            figures_dir,
            np.asarray(dataset.x),
            y_true,
            label_names,
            split_name="test",
            max_samples_per_class=args.saliency_max_samples,
        )
        save_saliency_heatmaps_by_class(
            figures_dir,
            model,
            loader,
            label_names,
            device,
            split_name="test",
            max_samples_per_class=args.saliency_max_samples,
        )
    if args.save_embedding:
        plot_embedding_2d(
            figures_dir / "test_pca_embedding.png",
            np.asarray(dataset.x),
            y_true,
            label_names,
            method="pca",
            title="Test PCA Embedding",
            max_samples=args.visualize_max_samples,
        )
        deep_features, feature_labels = extract_pre_classifier_features(model, loader, device)
        plot_embedding_2d(
            figures_dir / "test_pre_classifier_feature_pca_embedding.png",
            deep_features,
            feature_labels,
            label_names,
            method="pca",
            title="Test Pre-Classifier Feature PCA Embedding",
            max_samples=args.visualize_max_samples,
        )


def main() -> None:
    args = parse_args()
    device = torch.device(get_device(args.device))
    config_path = args.config or infer_config_path(args.checkpoint)
    config = load_json(config_path)
    checkpoint = load_checkpoint(args.checkpoint, device)

    processed_root = Path(args.processed_root or config.get("processed_root", DEFAULT_PROCESSED_ROOT))
    features_root = Path(args.features_root or config.get("features_root", DEFAULT_FEATURES_ROOT))
    input_kind = args.input_kind or str(config.get("input_kind", "spectrum"))
    feature_name = args.feature_name or str(config.get("feature_name", "feature"))
    output_dir = ensure_dir(args.output_dir or (args.checkpoint.parent.parent / "test_eval"))
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    label_names = load_metal_label_names(processed_root)
    num_classes = int(config.get("num_classes", len(label_names) if label_names else 6))
    model_name = str(config["model"])
    model_params = dict(config.get("model_params", {}))
    model_params.pop("num_classes", None)

    if input_kind == "feature":
        if model_name != "fnn":
            raise ValueError("input-kind=feature is only supported by the fnn model")
        dataset = LIBSFeatureDataset(
            features_root,
            processed_root,
            "test",
            feature_name=feature_name,
            load_sample_index=True,
        )
        model_params["input_length"] = int(dataset.x.shape[1])
    else:
        dataset = LIBSDataset(processed_root, "test", load_sample_index=True)
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = create_model(model_name, num_classes=num_classes, **model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    test_loss, metrics, y_true, y_pred, probabilities = evaluate(
        model,
        loader,
        criterion,
        device,
        show_progress=not args.disable_progress,
    )
    metrics = {"loss": float(test_loss), **metrics}
    elapsed_seconds = time.time() - start_time
    matrix = confusion_matrix(y_true, y_pred)
    report_text = classification_report_text(y_true, y_pred, label_names)

    save_json(
        output_dir / "test_metrics.json",
        {
            "checkpoint": str(args.checkpoint),
            "config": str(config_path),
            "processed_root": str(processed_root),
            "features_root": str(features_root),
            "input_kind": input_kind,
            "feature_name": feature_name,
            "elapsed_seconds": elapsed_seconds,
            "metrics": metrics,
        },
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_predictions_csv(output_dir / "predictions" / f"metal_{model_name}_test.csv", y_true, y_pred, label_names, probabilities)
    save_confusion_matrix_csv(output_dir / "confusion_matrix.csv", matrix, label_names)
    save_test_figures(output_dir, model, loader, dataset, matrix, metrics, y_true, label_names, args, device)
    write_test_report(output_dir / "test_report.md", args.checkpoint, processed_root, input_kind, metrics, report_text)

    print(
        "test finished: "
        f"model={model_name}, test_macro_f1={metrics['macro_f1']:.6f}, "
        f"elapsed={elapsed_seconds:.1f}s, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
