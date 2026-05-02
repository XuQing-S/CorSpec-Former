"""Train deep-learning baselines for LIBS metal classification."""

from __future__ import annotations

import argparse
import json
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
    get_device,
    make_run_dir,
    save_confusion_matrix_csv,
    save_history_csv,
    save_json,
    save_predictions_csv,
    set_random_seed,
    write_experiment_report,
)
from scripts.visualize import (
    plot_class_mean_spectra,
    plot_confusion_matrix,
    plot_embedding_2d,
    plot_metrics_bar,
    plot_split_embedding_2d,
    save_training_figures,
)
from scr.dl.dataset import LIBSDataset, create_feature_metal_dataloaders, create_metal_dataloaders, load_metal_label_names
from scr.dl.feature_extraction import extract_pre_classifier_features
from scr.dl.interpretability import save_saliency_heatmaps_by_class, save_spectrum_heatmaps_by_class
from scr.dl.models import count_trainable_parameters, create_model, get_default_model_params


DEFAULT_PROCESSED_ROOT = Path("data/processed/balanced")
DEFAULT_FEATURES_ROOT = Path("data/features/balanced")
DEFAULT_OUTPUT_ROOT = Path("experiments/dl")


def load_training_config(path: str | Path | None) -> dict:
    if path is None or str(path) == "":
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_config_value(config: dict, key: str, default):
    if key in config:
        return config[key]
    for section in ("data", "experiment", "training", "runtime", "visualization", "optimization"):
        section_payload = config.get(section, {})
        if isinstance(section_payload, dict) and key in section_payload:
            return section_payload[key]
    return default


def flatten_training_config(config: dict) -> dict:
    """Flatten only explicitly configured values into argparse defaults."""
    defaults: dict = {}
    for key, value in config.items():
        if key == "model_params":
            defaults[key] = value
            continue
        if key == "description" or isinstance(value, dict):
            continue
        defaults[key] = value

    section_to_keys = {
        "data": ("processed_root", "features_root", "input_kind", "feature_name"),
        "experiment": ("output_root", "run_name"),
        "training": ("epochs", "batch_size", "patience", "class_weight"),
        "optimization": ("learning_rate", "weight_decay", "lr_scheduler", "lr_factor", "lr_patience", "min_lr"),
        "runtime": ("num_workers", "device", "random_seed", "deterministic", "disable_progress"),
        "visualization": ("no_data_figures", "visualize_max_samples", "saliency_max_samples", "save_tsne"),
    }
    for section, keys in section_to_keys.items():
        payload = config.get(section, {})
        if not isinstance(payload, dict):
            continue
        for key in keys:
            if key in payload:
                defaults[key] = payload[key]
    return defaults


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None, help="训练配置 JSON 文件")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_training_config(pre_args.config)
    configured_defaults = flatten_training_config(config)

    parser = argparse.ArgumentParser(description="Train a deep-learning metal classifier for LIBS spectra.")
    parser.add_argument("--config", type=Path, default=pre_args.config, help="训练配置 JSON 文件")

    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT, help="预处理张量目录")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT, help="特征工程输出目录")
    parser.add_argument("--input-kind", choices=("spectrum", "feature"), default="spectrum", help="输入类型：原始光谱或融合/降维特征")
    parser.add_argument("--feature-name", choices=("feature", "pca", "stat", "pls_metal"), default="feature", help="input-kind=feature 时使用的特征类型")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="实验输出根目录")
    parser.add_argument("--run-name", type=str, default="", help="自定义运行目录名")

    parser.add_argument("--model", choices=("fnn", "cnn", "lstm", "resnet"), default="cnn", help="模型类型")
    parser.add_argument("--model-params", default={}, help="模型参数 JSON 字符串或 JSON 文件路径，用于覆盖默认参数")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")

    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none", help="类别权重策略")
    parser.add_argument("--lr-scheduler", choices=("none", "plateau", "cosine"), default="none", help="学习率调度策略")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau 衰减因子")
    parser.add_argument("--lr-patience", type=int, default=5, help="ReduceLROnPlateau 耐心轮数")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="CosineAnnealingLR 最小学习率")

    parser.add_argument("--epochs", type=int, default=200, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--patience", type=int, default=20, help="验证集 Macro-F1 早停耐心轮数")

    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数量")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda 或具体设备名")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子")
    parser.add_argument("--deterministic", action="store_true", default=False, help="启用确定性 CUDA 设置")

    parser.add_argument("--disable-progress", action="store_true", default=False, help="关闭 tqdm 训练进度条")
    parser.add_argument("--no-data-figures", action="store_true", default=False, help="不保存光谱分布类可视化图")
    parser.add_argument("--visualize-max-samples", type=int, default=3000, help="降维可视化最大抽样样本数")
    parser.add_argument("--saliency-max-samples", type=int, default=80, help="每类 saliency 热图最多抽样样本数")
    parser.add_argument("--save-tsne", action="store_true", default=False, help="额外保存 t-SNE 降维图，耗时较长")
    parser.set_defaults(**configured_defaults)
    args = parser.parse_args()
    args.config_payload = config
    return args


def load_model_param_overrides(value) -> dict:
    """Load model parameter overrides from JSON text or a JSON file."""
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    candidate_path = Path(value)
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return json.loads(value)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        loader,
        desc=f"train {epoch}/{total_epochs}",
        leave=False,
        disable=not show_progress,
    )
    for spectra, labels in progress:
        spectra = spectra.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(spectra)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=total_loss / max(total_samples, 1), lr=get_current_lr(optimizer))
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    stage: str,
    show_progress: bool,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    progress = tqdm(loader, desc=stage, leave=False, disable=not show_progress)
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_val_macro_f1: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def dataset_arrays(loader) -> tuple[np.ndarray, np.ndarray]:
    dataset = loader.dataset
    return np.asarray(dataset.x), np.asarray(dataset.y)


def save_visualization_outputs(
    run_dir: Path,
    model: nn.Module,
    dataloaders: dict,
    history: list[dict],
    matrix: np.ndarray,
    test_metrics: dict[str, float],
    y_true: np.ndarray,
    label_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Save standard and data-distribution figures for one training run."""
    figures_dir = run_dir / "figures"
    save_training_figures(figures_dir, history)
    plot_confusion_matrix(figures_dir / "confusion_matrix.png", matrix, label_names)
    plot_confusion_matrix(
        figures_dir / "confusion_matrix_normalized.png",
        matrix,
        label_names,
        normalize=True,
        title="Normalized Metal Classification Confusion Matrix",
    )
    classification_metrics = {key: value for key, value in test_metrics.items() if key != "loss"}
    plot_metrics_bar(figures_dir / "test_metrics.png", classification_metrics, title=f"{args.model} Test Metrics")

    if args.no_data_figures:
        return

    try:
        train_x, train_y = dataset_arrays(dataloaders["train"])
        val_x, _ = dataset_arrays(dataloaders["val"])
        test_x, _ = dataset_arrays(dataloaders["test"])

        if args.input_kind == "spectrum":
            plot_class_mean_spectra(
                figures_dir / "train_class_mean_spectra.png",
                train_x,
                train_y,
                label_names,
                title="Train Class Mean Spectra",
            )
            plot_class_mean_spectra(
                figures_dir / "test_class_mean_spectra.png",
                test_x,
                y_true,
                label_names,
                title="Test Class Mean Spectra",
            )
            save_spectrum_heatmaps_by_class(
                figures_dir,
                train_x,
                train_y,
                label_names,
                split_name="train",
                max_samples_per_class=args.saliency_max_samples,
            )
            save_spectrum_heatmaps_by_class(
                figures_dir,
                test_x,
                y_true,
                label_names,
                split_name="test",
                max_samples_per_class=args.saliency_max_samples,
            )
            save_saliency_heatmaps_by_class(
                figures_dir,
                model,
                dataloaders["train"],
                label_names,
                device,
                split_name="train",
                max_samples_per_class=args.saliency_max_samples,
            )
            save_saliency_heatmaps_by_class(
                figures_dir,
                model,
                dataloaders["test"],
                label_names,
                device,
                split_name="test",
                max_samples_per_class=args.saliency_max_samples,
            )
        plot_embedding_2d(
            figures_dir / "test_pca_embedding.png",
            test_x,
            y_true,
            label_names,
            method="pca",
            title="Test PCA Embedding",
            random_seed=args.random_seed,
            max_samples=args.visualize_max_samples,
        )
        plot_split_embedding_2d(
            figures_dir / "split_pca_distribution.png",
            {"train": train_x, "val": val_x, "test": test_x},
            method="pca",
            title="Train/Val/Test PCA Distribution",
            random_seed=args.random_seed,
            max_samples=args.visualize_max_samples,
        )

        train_features, _ = extract_pre_classifier_features(model, dataloaders["train"], device)
        val_features, _ = extract_pre_classifier_features(model, dataloaders["val"], device)
        test_features, test_feature_y = extract_pre_classifier_features(model, dataloaders["test"], device)
        plot_embedding_2d(
            figures_dir / "test_pre_classifier_feature_pca_embedding.png",
            test_features,
            test_feature_y,
            label_names,
            method="pca",
            title="Test Pre-Classifier Feature PCA Embedding",
            random_seed=args.random_seed,
            max_samples=args.visualize_max_samples,
        )
        plot_split_embedding_2d(
            figures_dir / "split_pre_classifier_feature_pca_distribution.png",
            {"train": train_features, "val": val_features, "test": test_features},
            method="pca",
            title="Train/Val/Test Pre-Classifier Feature PCA Distribution",
            random_seed=args.random_seed,
            max_samples=args.visualize_max_samples,
        )
        if args.save_tsne:
            plot_embedding_2d(
                figures_dir / "test_tsne_embedding.png",
                test_x,
                y_true,
                label_names,
                method="tsne",
                title="Test t-SNE Embedding",
                random_seed=args.random_seed,
                max_samples=args.visualize_max_samples,
            )
            plot_embedding_2d(
                figures_dir / "test_pre_classifier_feature_tsne_embedding.png",
                test_features,
                test_feature_y,
                label_names,
                method="tsne",
                title="Test Pre-Classifier Feature t-SNE Embedding",
                random_seed=args.random_seed,
                max_samples=args.visualize_max_samples,
            )
    except (ValueError, RuntimeError, OSError) as error:
        print(f"warning: failed to save data visualizations: {error}")


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_seed, deterministic=args.deterministic)
    device = torch.device(get_device(args.device))

    model_params = get_default_model_params(args.model)
    model_params.update(load_model_param_overrides(args.model_params))
    label_names = load_metal_label_names(args.processed_root)
    num_classes = len(label_names) if label_names else 6
    model_params.pop("num_classes", None)

    if args.input_kind == "feature" and args.model != "fnn":
        raise ValueError("input-kind=feature is only supported by the fnn model")

    if args.input_kind == "feature":
        dataloaders = create_feature_metal_dataloaders(
            args.features_root,
            args.processed_root,
            feature_name=args.feature_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        dataloaders = create_metal_dataloaders(
            args.processed_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    train_dataset = dataloaders["train"].dataset
    if args.input_kind == "feature":
        model_params["input_length"] = int(train_dataset.x.shape[1])
    model = create_model(args.model, num_classes=num_classes, **model_params).to(device)

    class_weights = None
    if args.class_weight == "balanced":
        class_weights = compute_class_weights(np.asarray(train_dataset.y), num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )

    run_dir = make_run_dir(args.output_root, args.run_name)
    config = {
        "model": args.model,
        "model_params": model_params,
        "config_path": str(args.config) if args.config else "",
        "config_payload": args.config_payload,
        "input_kind": args.input_kind,
        "feature_name": args.feature_name,
        "processed_root": str(args.processed_root),
        "features_root": str(args.features_root),
        "output_root": str(args.output_root),
        "run_dir": str(run_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "device": str(device),
        "random_seed": args.random_seed,
        "class_weight": args.class_weight,
        "lr_scheduler": args.lr_scheduler,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
        "disable_progress": args.disable_progress,
        "no_data_figures": args.no_data_figures,
        "visualize_max_samples": args.visualize_max_samples,
        "saliency_max_samples": args.saliency_max_samples,
        "save_tsne": args.save_tsne,
        "num_classes": num_classes,
        "label_names": label_names,
        "trainable_parameters": count_trainable_parameters(model),
    }
    save_json(run_dir / "config.json", config)

    history: list[dict] = []
    best_epoch = 0
    best_val_macro_f1 = -1.0
    best_val_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    show_progress = not args.disable_progress

    print(
        "training started: "
        f"model={args.model}, device={device}, epochs={args.epochs}, "
        f"trainable_parameters={config['trainable_parameters']}"
    )
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
        )
        val_loss, val_metrics, _, _, _ = evaluate(
            model,
            dataloaders["val"],
            criterion,
            device,
            stage=f"val {epoch}/{args.epochs}",
            show_progress=show_progress,
        )
        if scheduler is not None and args.lr_scheduler == "plateau":
            scheduler.step(val_metrics["macro_f1"])
        elif scheduler is not None:
            scheduler.step()

        epoch_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": get_current_lr(optimizer),
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} lr={row['lr']:.6g} "
            f"time={epoch_seconds:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "checkpoints" / "best_model.pt",
                model,
                optimizer,
                epoch,
                best_epoch,
                best_val_macro_f1,
                config,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            run_dir / "checkpoints" / "last_model.pt",
            model,
            optimizer,
            epoch,
            best_epoch,
            best_val_macro_f1,
            config,
        )
        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}, best_epoch={best_epoch}")
            break

    best_checkpoint = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_metrics, y_true, y_pred, probabilities = evaluate(
        model,
        dataloaders["test"],
        criterion,
        device,
        stage="test",
        show_progress=show_progress,
    )
    test_metrics = {"loss": float(test_loss), **test_metrics}
    metrics = {
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "test": test_metrics,
        "total_seconds": time.time() - start_time,
    }

    matrix = confusion_matrix(y_true, y_pred)
    report_text = classification_report_text(y_true, y_pred, label_names)

    save_json(run_dir / "metrics.json", metrics)
    save_history_csv(run_dir / "history.csv", history)
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_predictions_csv(run_dir / "predictions" / f"metal_{args.model}_test.csv", y_true, y_pred, label_names, probabilities)
    save_confusion_matrix_csv(run_dir / "confusion_matrix.csv", matrix, label_names)
    save_visualization_outputs(run_dir, model, dataloaders, history, matrix, test_metrics, y_true, label_names, args, device)
    write_experiment_report(
        run_dir / "report.md",
        model_name=args.model,
        dataset_root=args.processed_root,
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        params=model_params,
    )

    print(
        "training finished: "
        f"model={args.model}, best_epoch={best_epoch}, "
        f"test_macro_f1={test_metrics['macro_f1']:.6f}, run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()
