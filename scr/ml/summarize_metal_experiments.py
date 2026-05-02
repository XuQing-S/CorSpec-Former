"""Summarize metal-classification experiment runs.

The script scans experiment directories produced by ``metal_classifier.py`` and
creates a consolidated report, CSV summaries, and comparison figures.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize metal classification experiments.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments/ml/metal_classification"),
        help="金属分类实验根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="汇总结果输出目录；默认 experiments-root/summary",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_runs(experiments_root: Path) -> list[Path]:
    runs: list[Path] = []
    for path in sorted(experiments_root.iterdir()):
        if not path.is_dir() or path.name == "summary":
            continue
        if (path / "config.json").exists() and (path / "metrics.json").exists():
            runs.append(path)
    return runs


def read_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        pred_labels = header[1:]
        for row in reader:
            labels.append(row[0])
            rows.append([int(value) for value in row[1:]])
    return pred_labels if pred_labels else labels, np.asarray(rows, dtype=np.int64)


def collect_experiments(runs: list[Path]) -> tuple[list[dict], dict[str, tuple[list[str], np.ndarray]]]:
    rows: list[dict] = []
    confusion_matrices: dict[str, tuple[list[str], np.ndarray]] = {}
    for run_dir in runs:
        config = read_json(run_dir / "config.json")
        metrics = read_json(run_dir / "metrics.json")
        row = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "feature": config.get("feature", ""),
            "model": config.get("model", ""),
            "grid_search": str(config.get("grid_search", False)),
            "best_params": json.dumps(config.get("best_params", {}), ensure_ascii=False),
        }
        for split in ("val", "test"):
            for metric in METRIC_NAMES:
                row[f"{split}_{metric}"] = float(metrics.get(split, {}).get(metric, np.nan))
        row["macro_f1_gap"] = row["val_macro_f1"] - row["test_macro_f1"]
        rows.append(row)

        matrix_path = run_dir / "confusion_matrix.csv"
        if matrix_path.exists():
            confusion_matrices[run_dir.name] = read_confusion_matrix(matrix_path)
    rows.sort(key=lambda item: item["test_macro_f1"], reverse=True)
    return rows, confusion_matrices


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_feature_model_matrix(rows: list[dict], metric_name: str) -> tuple[list[str], list[str], np.ndarray]:
    features = sorted({row["feature"] for row in rows})
    models = sorted({row["model"] for row in rows})
    matrix = np.full((len(features), len(models)), np.nan, dtype=np.float64)
    for row in rows:
        feature_index = features.index(row["feature"])
        model_index = models.index(row["model"])
        matrix[feature_index, model_index] = row[metric_name]
    return features, models, matrix


def write_matrix_csv(path: Path, features: list[str], models: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", *models])
        for feature, values in zip(features, matrix):
            writer.writerow([feature, *[f"{value:.6f}" if np.isfinite(value) else "" for value in values]])


def plot_metric_by_experiment(rows: list[dict], output_path: Path) -> None:
    sorted_rows = sorted(rows, key=lambda item: item["test_macro_f1"])
    labels = [f"{row['feature']} / {row['model']}" for row in sorted_rows]
    values = [row["test_macro_f1"] for row in sorted_rows]

    fig_height = max(6, 0.35 * len(sorted_rows))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(np.arange(len(labels)), values, color="#4C72B0")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("Metal Classification Experiment Ranking")
    ax.set_xlim(0.0, 1.0)
    for index, value in enumerate(values):
        ax.text(min(value + 0.01, 0.98), index, f"{value:.4f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_model_heatmap(features: list[str], models: list[str], matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Test Macro-F1")
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticklabels(features)
    ax.set_title("Feature x Model Test Macro-F1")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_val_test_scatter(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    features = sorted({row["feature"] for row in rows})
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(features), 1)))
    color_map = {feature: colors[index] for index, feature in enumerate(features)}
    for row in rows:
        ax.scatter(row["val_macro_f1"], row["test_macro_f1"], color=color_map[row["feature"]], s=60)
        ax.text(row["val_macro_f1"] + 0.003, row["test_macro_f1"], row["model"], fontsize=8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Validation Macro-F1")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Validation vs Test Macro-F1")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=feature, markerfacecolor=color_map[feature], markersize=8)
        for feature in features
    ]
    ax.legend(handles=handles, title="feature", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_confusion_matrix(
    best_run_id: str,
    confusion_matrices: dict[str, tuple[list[str], np.ndarray]],
    output_path: Path,
) -> None:
    if best_run_id not in confusion_matrices:
        return
    labels, matrix = confusion_matrices[best_run_id]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Best Run Confusion Matrix ({best_run_id})")
    threshold = matrix.max() / 2.0
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
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict], columns: list[str], limit: int | None = None) -> str:
    selected_rows = rows if limit is None else rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in selected_rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_best_by_key(rows: list[dict], key: str) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        group = row[key]
        if group not in best or row["test_macro_f1"] > best[group]["test_macro_f1"]:
            best[group] = row
    return sorted(best.values(), key=lambda item: item[key])


def write_report(output_path: Path, rows: list[dict], figure_dir_name: str) -> None:
    best = rows[0]
    columns = ["run_id", "feature", "model", "val_macro_f1", "test_macro_f1", "test_accuracy", "macro_f1_gap"]
    lines = [
        "# 金属分类 16 组实验汇总报告",
        "",
        "## 总览",
        "",
        f"- 实验数量：{len(rows)}",
        f"- 最佳实验：`{best['run_id']}`",
        f"- 最佳特征/模型：`{best['feature']}` + `{best['model']}`",
        f"- 最佳测试集 Macro-F1：{best['test_macro_f1']:.6f}",
        f"- 最佳测试集 Accuracy：{best['test_accuracy']:.6f}",
        "",
        "## 实验排名",
        "",
        markdown_table(rows, columns),
        "",
        "## 各特征最佳实验",
        "",
        markdown_table(summarize_best_by_key(rows, "feature"), columns),
        "",
        "## 各模型最佳实验",
        "",
        markdown_table(summarize_best_by_key(rows, "model"), columns),
        "",
        "## 生成文件",
        "",
        "- `experiment_summary.csv`：16 组实验完整指标表",
        "- `feature_model_test_macro_f1.csv`：特征与模型二维对比表",
        f"- `{figure_dir_name}/test_macro_f1_ranking.png`：测试集 Macro-F1 排名图",
        f"- `{figure_dir_name}/feature_model_heatmap.png`：特征-模型测试 Macro-F1 热力图",
        f"- `{figure_dir_name}/val_vs_test_macro_f1.png`：验证集与测试集 Macro-F1 对比散点图",
        f"- `{figure_dir_name}/best_confusion_matrix.png`：最佳实验混淆矩阵",
        "",
        "## 初步结论",
        "",
        "- 以测试集 Macro-F1 作为主排序指标，优先选择排名靠前且验证/测试差距较小的组合。",
        "- 若某个组合验证集明显高于测试集，说明可能存在过拟合或对划分敏感，需要谨慎使用。",
        "- 后续可围绕最佳特征和最佳模型开启更细粒度的网格搜索。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.experiments_root / "summary")
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.experiments_root)
    if not runs:
        raise FileNotFoundError(f"No experiment runs found in {args.experiments_root}")

    rows, confusion_matrices = collect_experiments(runs)
    write_csv(output_dir / "experiment_summary.csv", rows)
    features, models, matrix = build_feature_model_matrix(rows, "test_macro_f1")
    write_matrix_csv(output_dir / "feature_model_test_macro_f1.csv", features, models, matrix)

    plot_metric_by_experiment(rows, figure_dir / "test_macro_f1_ranking.png")
    plot_feature_model_heatmap(features, models, matrix, figure_dir / "feature_model_heatmap.png")
    plot_val_test_scatter(rows, figure_dir / "val_vs_test_macro_f1.png")
    plot_best_confusion_matrix(rows[0]["run_id"], confusion_matrices, figure_dir / "best_confusion_matrix.png")
    write_report(output_dir / "summary_report.md", rows, "figures")

    print(f"summarized experiments={len(rows)}")
    print(f"summary directory: {output_dir}")


if __name__ == "__main__":
    main()
