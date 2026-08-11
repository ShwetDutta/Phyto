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
    parser.add_argument("--baseline-epochs", type=int, default=15)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--kd-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
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

    teacher_ckpt = ckpt_dir / "cbam_teacher_resnet50.pth"
    teacher_json = ckpt_dir / "cbam_teacher_resnet50_metrics.json"

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

    # Stage 2: Train Baseline ShuffleNetV2 x0.5
    print("\n>>> STAGE 1/5: Baseline ShuffleNetV2 x0.5...")
    if base_ckpt.exists() and base_json.exists() and not args.force:
        print(f"[Notice] Baseline checkpoint already exists at {base_ckpt}. Skipping training!")
    else:
        run_cmd([
            python_exe, "-m", "src.training.train_baseline",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--epochs", str(args.baseline_epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(base_ckpt),
            "--output-metrics-json", str(base_json),
        ])

    # Stage 3: Train ResNet50 + CBAM Teacher
    print("\n>>> STAGE 2/5: High-Capacity ResNet50 + CBAM Teacher...")
    if teacher_ckpt.exists() and teacher_json.exists() and not args.force:
        print(f"[Notice] Teacher checkpoint already exists at {teacher_ckpt}. Skipping training!")
    else:
        run_cmd([
            python_exe, "-m", "src.training.train_teacher",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--epochs", str(args.teacher_epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(teacher_ckpt),
            "--output-metrics-json", str(teacher_json),
        ])

    # Stage 4: Distill ResNet50 + CBAM Teacher into ShuffleNetV2 x0.5 Student
    print("\n>>> STAGE 3/5: Knowledge Distillation into Student...")
    if kd_ckpt.exists() and kd_json.exists() and not args.force:
        print(f"[Notice] KD Student checkpoint already exists at {kd_ckpt}. Skipping training!")
    else:
        run_cmd([
            python_exe, "-m", "src.training.train_kd_student",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--teacher-checkpoint", str(teacher_ckpt),
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
