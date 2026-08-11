"""
Master Google Colab End-to-End Pipeline Execution Runner for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Executes all 5 experimental stages sequentially on Tesla T4 GPU:
1. ShuffleNetV2 x0.5 Baseline Training & Test Evaluation
2. ResNet50 + CBAM Teacher Training & Test Evaluation
3. Knowledge Distillation into ShuffleNetV2 x0.5 Student
4. Dynamic INT8 Quantization & CPU Benchmarking
5. ONNX Export & Edge Engine Benchmarking
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Auto-resolve repository root into sys.path for Google Colab
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def run_cmd(cmd: list[str]) -> None:
    cmd_str = " ".join(cmd)
    print(f"\n==========================================")
    print(f"Executing: {cmd_str}")
    print(f"==========================================\n", flush=True)
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    res = subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd_str}")



def main():
    parser = argparse.ArgumentParser(description="Master End-to-End Phyto Pipeline Execution Runner")
    parser.add_argument("--manifest-path", type=str, default="results/new_dataset_manifest/groundnut_dataset_split_manifest.csv")
    parser.add_argument("--raw-data-root", type=str, default="/content/drive/MyDrive/Datasets for phyto/raw_data_new/Groundnut_Leaf_dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="/content/drive/MyDrive/Datasets for phyto/models")
    parser.add_argument("--image-size", type=int, default=256, help="Input resolution (default: 256 for base paper match)")
    parser.add_argument("--baseline-epochs", type=int, default=35, help="Baseline training epochs (default: 35)")
    parser.add_argument("--teacher-epochs", type=int, default=25, help="Teacher training epochs (default: 25)")
    parser.add_argument("--kd-epochs", type=int, default=25, help="KD student training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Baseline optimizer (default: sgd)")
    parser.add_argument("--lr", type=float, default=0.01, help="Baseline learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Force re-training even if checkpoints exist")
    args = parser.parse_args()

    import torch
    print("=======================================================")
    print("PHYTO MASTER PIPELINE INITIALIZATION")
    if torch.cuda.is_available():
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("WARNING: CUDA GPU NOT DETECTED! Running on CPU will be extremely slow.")
        print("Switch runtime in Colab: Runtime -> Change runtime type -> T4 GPU")
    print("=======================================================\n", flush=True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = ckpt_dir / "baseline_shufflenet_v05.pth"
    base_json = ckpt_dir / "baseline_shufflenet_v05_metrics.json"

    teacher_resnet_ckpt = ckpt_dir / "cbam_teacher_resnet50.pth"
    teacher_resnet_json = ckpt_dir / "cbam_teacher_resnet50_metrics.json"

    teacher_eff_ckpt = ckpt_dir / "teacher_efficientnet_b0.pth"
    teacher_eff_json = ckpt_dir / "teacher_efficientnet_b0_metrics.json"

    kd_ckpt = ckpt_dir / "kd_shufflenet_v05_from_resnet50.pth"
    kd_json = ckpt_dir / "kd_shufflenet_v05_metrics.json"

    quant_json = ckpt_dir / "quantization_benchmark_results.json"
    onnx_file = ckpt_dir / "kd_shufflenet_v05.onnx"

    python_exe = sys.executable

    # Stage 1: Manifest check
    manifest_p = Path(args.manifest_path)
    if not manifest_p.exists():
        print("Split manifest missing. Generating manifest and split...")
        run_cmd([python_exe, "-m", "src.data.build_new_dataset_manifest", "--dataset-root", args.raw_data_root])
        run_cmd([python_exe, "-m", "src.data.create_new_dataset_split", "--seed", str(args.seed)])

    # Stage 2: Train Baseline ShuffleNetV2 x0.5 (Paper-Matching Config: 256x256, SGD, 35 epochs)
    print("\n>>> STAGE 1/5: Baseline ShuffleNetV2 x0.5 (Paper Reproducibility Audit Config)...")
    if base_ckpt.exists() and base_json.exists() and not args.force:
        print(f"[Notice] Baseline checkpoint already exists at {base_ckpt}. Skipping training!")
    else:
        run_cmd([
            python_exe, "-m", "src.training.train_baseline",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--image-size", str(args.image_size),
            "--epochs", str(args.baseline_epochs),
            "--batch-size", str(args.batch_size),
            "--optimizer", args.optimizer,
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--checkpoint-path", str(base_ckpt),
            "--output-metrics-json", str(base_json),
        ])

    # Stage 3: Train Candidate Teachers (ResNet50 + CBAM & EfficientNet-B0)
    print("\n>>> STAGE 2/5: Training Candidate Teachers for Validation Selection...")
    if teacher_resnet_ckpt.exists() and teacher_resnet_json.exists() and not args.force:
        print(f"[Notice] Teacher (ResNet50 + CBAM) checkpoint already exists at {teacher_resnet_ckpt}.")
    else:
        run_cmd([
            python_exe, "-m", "src.training.train_teacher",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--teacher-type", "resnet50_cbam",
            "--image-size", str(args.image_size),
            "--epochs", str(args.teacher_epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(teacher_resnet_ckpt),
            "--output-metrics-json", str(teacher_resnet_json),
        ])

    if teacher_eff_ckpt.exists() and teacher_eff_json.exists() and not args.force:
        print(f"[Notice] Teacher (EfficientNet-B0) checkpoint already exists at {teacher_eff_ckpt}.")
    else:
        run_cmd([
            python_exe, "-m", "src.training.train_teacher",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--teacher-type", "efficientnet_b0",
            "--image-size", str(args.image_size),
            "--epochs", str(args.teacher_epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(teacher_eff_ckpt),
            "--output-metrics-json", str(teacher_eff_json),
        ])

    # Select best teacher based on validation metrics
    selected_teacher_ckpt = teacher_resnet_ckpt
    selected_teacher_type = "resnet50_cbam"

    try:
        r_val = torch.load(teacher_resnet_ckpt, map_location="cpu", weights_only=False).get("best_validation_accuracy", 0.0)
        e_val = torch.load(teacher_eff_ckpt, map_location="cpu", weights_only=False).get("best_validation_accuracy", 0.0)
        print(f"\n[Teacher Selection] ResNet50+CBAM Val Acc: {r_val:.2f}% | EfficientNet-B0 Val Acc: {e_val:.2f}%")
        if e_val > r_val:
            selected_teacher_ckpt = teacher_eff_ckpt
            selected_teacher_type = "efficientnet_b0"
            print("Selected EfficientNet-B0 as superior teacher model!")
        else:
            print("Selected ResNet50 + CBAM as superior teacher model!")
    except Exception as e:
        print(f"[Notice] Could not compare teacher validation checkpoints ({e}), defaulting to ResNet50 + CBAM.")

    # Stage 4: Distill Selected Teacher into ShuffleNetV2 x0.5 Student
    print(f"\n>>> STAGE 3/5: Executing Knowledge Distillation ({selected_teacher_type} -> Student)...")
    if kd_ckpt.exists() and kd_json.exists() and not args.force:
        print(f"[Notice] KD Student checkpoint already exists at {kd_ckpt}. Skipping training!")
    else:
        run_cmd([
            python_exe, "-m", "src.training.train_kd_student",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--teacher-type", selected_teacher_type,
            "--teacher-checkpoint", str(selected_teacher_ckpt),
            "--image-size", str(args.image_size),
            "--epochs", str(args.kd_epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(kd_ckpt),
            "--output-metrics-json", str(kd_json),
        ])

    # Stage 5: INT8 Quantization & Benchmarking
    print("\n>>> STAGE 4/5: Performing INT8 Dynamic Quantization & Benchmarking...")
    run_cmd([
        python_exe, "-m", "src.quantization.evaluate_and_quantize",
        "--manifest-path", str(manifest_p),
        "--raw-data-root", args.raw_data_root,
        "--student-checkpoint", str(kd_ckpt),
        "--output-json", str(quant_json),
    ])

    # Stage 6: ONNX Export & Edge Engine Benchmarking
    print("\n>>> STAGE 5/5: Exporting ONNX Engine & Benchmarking...")
    run_cmd([
        python_exe, "-m", "src.quantization.export_onnx_tensorrt",
        "--student-checkpoint", str(kd_ckpt),
        "--onnx-path", str(onnx_file),
    ])

    print("\n=======================================================")
    print("ALL PHYTO EXPERIMENT STAGES COMPLETED SUCCESSFULLY!")
    print("Checkpoints and JSON metrics saved to:", ckpt_dir.resolve())
    print("=======================================================\n")


if __name__ == "__main__":
    main()
