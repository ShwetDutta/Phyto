import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from phyto.config import Config

# Published Benchmark Literature Data (IEEE ICCSCE 2025 Paper Reference)
LITERATURE_BENCHMARKS = [
    {"model_name": "MobileNetV2", "paper_ref": "ICCSCE 2025", "accuracy": 91.02, "latency_ms": 14.50, "fps": 68.98},
    {"model_name": "MobileNetV3-Small", "paper_ref": "ICCSCE 2025", "accuracy": 87.40, "latency_ms": 10.63, "fps": 94.08},
    {"model_name": "MobileNetV3-Large", "paper_ref": "ICCSCE 2025", "accuracy": 82.88, "latency_ms": 14.15, "fps": 70.70},
    {"model_name": "ShuffleNetV2 x0.5", "paper_ref": "ICCSCE 2025", "accuracy": 98.06, "latency_ms": 9.41, "fps": 106.76},
    {"model_name": "ShuffleNetV2 x1.0", "paper_ref": "ICCSCE 2025", "accuracy": 97.66, "latency_ms": 12.30, "fps": 81.33},
    {"model_name": "ShuffleNetV2 x1.5", "paper_ref": "ICCSCE 2025", "accuracy": 97.21, "latency_ms": 14.80, "fps": 67.84},
    {"model_name": "ShuffleNetV2 x2.0", "paper_ref": "ICCSCE 2025", "accuracy": 95.62, "latency_ms": 17.12, "fps": 58.44},
    {"model_name": "VGG-16", "paper_ref": "ICCSCE 2025", "accuracy": 95.84, "latency_ms": 82.40, "fps": 12.22},
]


def plot_training_history(history_dict: Dict[str, Dict[str, List[float]]], save_path: str = "results/training_history.png"):
    """
    Plots training and validation loss and accuracy curves for multiple models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    for model_name, hist in history_dict.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["val_loss"], label=f"{model_name} Val Loss", linewidth=2)
        axes[1].plot(epochs, hist["val_acc"], label=f"{model_name} Val Acc", linewidth=2)

    axes[0].set_title("Validation Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation Accuracy Curve (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Training history curves saved to: {save_path}")


def plot_confusion_matrices(
    results_list: List[Dict[str, Any]],
    class_names: List[str] = Config.CLASS_NAMES,
    save_path: str = "results/confusion_matrices.png"
):
    """
    Plots confusion matrix heatmaps side by side for all evaluated models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(results_list)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))

    if num_models == 1:
        axes = [axes]

    for i, res in enumerate(results_list):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=axes[i]
        )
        axes[i].set_title(f"{res['model_name']}\nAccuracy: {res['accuracy']:.2f}%", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Confusion matrices saved to: {save_path}")


def plot_comparison_bar_charts(
    results_list: List[Dict[str, Any]],
    save_path: str = "results/model_comparison.png"
):
    """
    Generates comparative bar charts for Accuracy, F1-Score, Inference Latency, and Model Size.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(results_list)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    # 1. Accuracy
    sns.barplot(data=df, x="model_name", y="accuracy", ax=axes[0, 0], palette=colors[:len(df)])
    axes[0, 0].set_title("Test Accuracy (%) [Higher is Better]", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy (%)")
    for p in axes[0, 0].patches:
        axes[0, 0].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    # 2. F1-Score
    sns.barplot(data=df, x="model_name", y="f1_macro", ax=axes[0, 1], palette=colors[:len(df)])
    axes[0, 1].set_title("Macro F1-Score (%) [Higher is Better]", fontweight="bold")
    axes[0, 1].set_ylabel("F1 Score (%)")
    for p in axes[0, 1].patches:
        axes[0, 1].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    # 3. Latency
    sns.barplot(data=df, x="model_name", y="latency_ms", ax=axes[1, 0], palette=colors[:len(df)])
    axes[1, 0].set_title("Inference Latency (ms/sample) [Lower is Better]", fontweight="bold")
    axes[1, 0].set_ylabel("Latency (ms)")
    for p in axes[1, 0].patches:
        axes[1, 0].annotate(f"{p.get_height():.2f} ms", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    # 4. Model Size
    sns.barplot(data=df, x="model_name", y="model_size_mb", ax=axes[1, 1], palette=colors[:len(df)])
    axes[1, 1].set_title("Model File Size (MB) [Lower is Better]", fontweight="bold")
    axes[1, 1].set_ylabel("Size (MB)")
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f"{p.get_height():.2f} MB", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    for ax in axes.flat:
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Comparative bar charts saved to: {save_path}")


def generate_summary_markdown_table(results_list: List[Dict[str, Any]]) -> str:
    """
    Produces clean Markdown comparative table for baseline vs proposed model evaluation,
    including paper baseline literature comparison.
    """
    table_md = "### Experimental Model Benchmarks (Phyto Project)\n\n"
    table_md += "| Model Name | Attention | Distillation | Accuracy (%) | Macro F1 (%) | Latency (ms) | Throughput (FPS) | Size (MB) | Params (M) |\n"
    table_md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for r in results_list:
        has_cbam = "CBAM" if "CBAM" in r["model_name"] or "Proposed" in r["model_name"] else "None"
        has_kd = "Yes" if "KD" in r["model_name"] or "Proposed" in r["model_name"] else "No"

        table_md += f"| **{r['model_name']}** | {has_cbam} | {has_kd} | {r['accuracy']:.2f}% | {r['f1_macro']:.2f}% | {r['latency_ms']:.2f} ms | {r['fps']:.1f} FPS | {r['model_size_mb']:.2f} MB | {r['total_params_m']:.2f}M |\n"

    table_md += "\n### Reference Paper Benchmarks (IEEE ICCSCE 2025)\n\n"
    table_md += "| Model Name | Reference Source | Classification Accuracy (%) | Inference Time (ms) | Throughput (FPS) |\n"
    table_md += "| :--- | :---: | :---: | :---: | :---: |\n"

    for lit in LITERATURE_BENCHMARKS:
        table_md += f"| **{lit['model_name']}** | {lit['paper_ref']} | {lit['accuracy']:.2f}% | {lit['latency_ms']:.2f} ms | {lit['fps']:.2f} FPS |\n"

    return table_md
