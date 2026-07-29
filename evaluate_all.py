"""
Project Phyto: Comprehensive Evaluation & Benchmarking CLI Script
Evaluates Teacher, Baseline, Proposed, and Quantized Models on Test Set
Generates metrics tables and comparison plots.
"""

import os
import argparse
import torch
from phyto.config import Config
from phyto.dataset import build_dataloaders
from phyto.models import TeacherResNet50, BaselineShuffleNetV2, CBAMShuffleNetV2
from phyto.evaluator import evaluate_comprehensive
from phyto.quantization import quantize_model_int8
from phyto.utils import (
    plot_confusion_matrices, plot_comparison_bar_charts, generate_summary_markdown_table
)

def load_checkpoint_or_init(model: torch.nn.Module, checkpoint_name: str, device: str = Config.DEVICE) -> str:
    path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_name)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[Checkpoint] Loaded state dict from: {path}")
        return path
    else:
        print(f"[Warning] Checkpoint not found at {path}. Evaluating model with base weights.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Phyto Evaluation & Benchmarking Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Raw_Data folder (Optional: auto-resolves via kagglehub)")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    Config.setup_directories()

    _, _, test_loader, _, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    results = []

    # 1. Evaluate Teacher Model
    print("\n[Evaluation] Benchmarking Teacher Model (ResNet50)...")
    teacher = TeacherResNet50(num_classes=Config.NUM_CLASSES)
    teacher_ckpt = load_checkpoint_or_init(teacher, "teacher_resnet50.pth")
    res_teacher = evaluate_comprehensive(teacher, test_loader, "Teacher (ResNet50)", teacher_ckpt)
    results.append(res_teacher)

    # 2. Evaluate Baseline Model
    print("\n[Evaluation] Benchmarking Baseline Model (ShuffleNetV2 1.0x)...")
    baseline = BaselineShuffleNetV2(num_classes=Config.NUM_CLASSES)
    baseline_ckpt = load_checkpoint_or_init(baseline, "baseline_shufflenetv2.pth")
    res_baseline = evaluate_comprehensive(baseline, test_loader, "Baseline (ShuffleNetV2)", baseline_ckpt)
    results.append(res_baseline)

    # 3. Evaluate Proposed Model
    print("\n[Evaluation] Benchmarking Proposed Model (CBAM-ShuffleNetV2 + KD)...")
    proposed = CBAMShuffleNetV2(num_classes=Config.NUM_CLASSES)
    proposed_ckpt = load_checkpoint_or_init(proposed, "proposed_cbam_shufflenetv2_kd.pth")
    res_proposed = evaluate_comprehensive(proposed, test_loader, "Proposed (CBAM-ShuffleNet+KD)", proposed_ckpt)
    results.append(res_proposed)

    # 4. Evaluate Quantized Model
    print("\n[Evaluation] Benchmarking Quantized Proposed Model (INT8 CBAM-ShuffleNetV2)...")
    quantized = quantize_model_int8(proposed)
    quant_ckpt = os.path.join(Config.CHECKPOINT_DIR, "proposed_cbam_shufflenetv2_int8.pth")
    res_quant = evaluate_comprehensive(quantized, test_loader, "Proposed INT8 (Quantized)", quant_ckpt, device="cpu")
    results.append(res_quant)

    # Generate Comparative Artifacts
    print("\n========================================================")
    print("PHYTO PERFORMANCE & EDGE BENCHMARKING SUMMARY")
    print("========================================================")
    table_md = generate_summary_markdown_table(results)
    print(table_md)

    with open("results/metrics_comparison.md", "w") as f:
        f.write("# Phyto Benchmarking Summary\n\n" + table_md)

    plot_confusion_matrices(results, save_path="results/confusion_matrices.png")
    plot_comparison_bar_charts(results, save_path="results/model_comparison.png")

    print("\n[SUCCESS] All evaluation plots and summary table written to 'results/' directory.")

if __name__ == "__main__":
    main()
