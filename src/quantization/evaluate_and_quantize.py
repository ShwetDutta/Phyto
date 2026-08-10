"""
INT8 Quantization and Benchmarking Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Measures FP32 vs Dynamic INT8 Quantized model size, accuracy, latency, and FPS.
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.phyto_dataset import PhytoDataset, load_split_manifest
from src.models import create_shufflenet_v2_x0_5
from src.quantization.quantize import quantize_model_dynamic, get_model_size_mb
from src.evaluation.metrics import evaluate_model, benchmark_inference


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate & Dynamic Quantize KD Student Model")
    parser.add_argument("--manifest-path", type=str, default="results/new_dataset_manifest/groundnut_dataset_split_manifest.csv")
    parser.add_argument("--raw-data-root", type=str, default=r"c:\Users\Shwet\Desktop\Groundnut_Leaf_dataset")
    parser.add_argument("--student-checkpoint", type=str, default="results/checkpoints/kd_shufflenet_v05_from_resnet50.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=str, default="results/checkpoints/quantization_benchmark_results.json")
    args = parser.parse_args()

    test_df = load_split_manifest(args.manifest_path, split="test")
    test_transform = get_test_transform()
    test_dataset = PhytoDataset(test_df, raw_data_root=args.raw_data_root, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FP32 model
    student_model = create_shufflenet_v2_x0_5(num_classes=6, pretrained=False)
    s_ckpt_p = Path(args.student_checkpoint)
    if not s_ckpt_p.exists():
        raise FileNotFoundError(f"Student checkpoint not found at: {s_ckpt_p.resolve()}")

    s_ckpt = torch.load(s_ckpt_p, map_location=device, weights_only=False)
    student_model.load_state_dict(s_ckpt["model_state_dict"])
    student_model.eval()

    fp32_size_mb = get_model_size_mb(student_model)
    fp32_test_metrics = evaluate_model(student_model, test_loader, device=device)
    fp32_bench = benchmark_inference(student_model, test_loader, device=device)

    print("\n=== FP32 Student Benchmarks ===")
    print(f"Model Size:        {fp32_size_mb:.2f} MB")
    print(f"Test Accuracy:     {fp32_test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1:          {fp32_test_metrics['macro_f1_score'] * 100:.2f}%")
    print(f"Latency ({device}): {fp32_bench['average_latency_ms']:.2f} ms/img ({fp32_bench['fps']:.1f} FPS)")

    # Dynamic INT8 Quantization (runs on CPU)
    int8_model = quantize_model_dynamic(student_model)
    int8_size_mb = get_model_size_mb(int8_model)

    cpu_device = torch.device("cpu")
    test_loader_cpu = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    int8_test_metrics = evaluate_model(int8_model, test_loader_cpu, device=cpu_device)
    int8_bench = benchmark_inference(int8_model, test_loader_cpu, device=cpu_device)

    print("\n=== INT8 Quantized Student Benchmarks (CPU) ===")
    print(f"Model Size:        {int8_size_mb:.2f} MB")
    print(f"Test Accuracy:     {int8_test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1:          {int8_test_metrics['macro_f1_score'] * 100:.2f}%")
    print(f"Latency (CPU):     {int8_bench['average_latency_ms']:.2f} ms/img ({int8_bench['fps']:.1f} FPS)")

    results = {
        "fp32": {
            "size_mb": fp32_size_mb,
            "metrics": fp32_test_metrics,
            "benchmark": fp32_bench,
            "device": str(device),
        },
        "int8_dynamic": {
            "size_mb": int8_size_mb,
            "metrics": int8_test_metrics,
            "benchmark": int8_bench,
            "device": "cpu",
        },
    }

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved quantization benchmarking results to {out_p}")


if __name__ == "__main__":
    main()
