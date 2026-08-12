"""
Master Controlled Knowledge Distillation & Model Ablation Execution Runner.
Phyto Project - Groundnut Plant Disease Classification (Edge-AI Framework).

Executes controlled experiments:
1. ShuffleNetV2 x0.5 Baseline (No CBAM, No KD)
2. ShuffleNetV2 x0.5 + CBAM (No KD)
3. EfficientNet-B0 Teacher -> ShuffleNetV2 x0.5 Logit KD (Hyperparameter sweep T & Alpha on Val Set)
4. EfficientNet-B0 Teacher -> ShuffleNetV2 x0.5 Feature KD (Hyperparameter sweep Beta on Val Set)
5. EfficientNet-B0 Teacher -> ShuffleNetV2 x0.5 / x0.5+CBAM Combined Logit + Feature KD

Generates full Markdown ablation table and JSON summary report.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import (
    create_shufflenet_v2_x0_5,
    create_shufflenet_v2_x0_5_cbam,
    create_efficientnet_b0,
)
from src.data.phyto_dataset import PhytoDataset, load_split_manifest
from src.evaluation.metrics import evaluate_model
from torch.utils.data import DataLoader
from torchvision import transforms


def count_parameters(model: nn.Module) -> float:
    """Returns total parameter count in Millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_model_size_mb(model_path: Path) -> float:
    """Returns model checkpoint size on disk in MB."""
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


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
    parser = argparse.ArgumentParser(description="Master Controlled KD & Ablation Runner")
    parser.add_argument("--manifest-path", type=str, default="results/new_dataset_manifest/groundnut_dataset_split_manifest.csv")
    parser.add_argument("--raw-data-root", type=str, default=r"c:\Users\Shwet\Desktop\Groundnut_Leaf_dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="results/ablation_models")
    parser.add_argument("--teacher-checkpoint", type=str, default="results/checkpoints/teacher_efficientnet_b0.pth")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Force retraining even if checkpoints exist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=======================================================")
    print("PHYTO CONTROLLED KNOWLEDGE DISTILLATION ABLATION RUNNER")
    print(f"Device: {device}")
    print("=======================================================\n", flush=True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_p = Path(args.manifest_path)
    python_exe = sys.executable

    # Setup test dataloader for evaluation
    test_df = load_split_manifest(args.manifest_path, split="test")
    test_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = PhytoDataset(test_df, raw_data_root=args.raw_data_root, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Dictionary to hold final ablation results
    ablation_results = []

    # -------------------------------------------------------------
    # Exp 1: Baseline ShuffleNetV2 x0.5 (No CBAM, No KD)
    # -------------------------------------------------------------
    print("\n>>> ABLATION 1/5: ShuffleNetV2 x0.5 Baseline...")
    exp1_ckpt = ckpt_dir / "exp1_shufflenet_v05_baseline.pth"
    exp1_json = ckpt_dir / "exp1_shufflenet_v05_baseline_metrics.json"

    if not exp1_ckpt.exists() or args.force:
        run_cmd([
            python_exe, "-m", "src.training.train_baseline",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--image-size", str(args.image_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(exp1_ckpt),
            "--output-metrics-json", str(exp1_json),
        ])

    m1 = create_shufflenet_v2_x0_5(num_classes=6, pretrained=False)
    m1.load_state_dict(torch.load(exp1_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    e1 = evaluate_model(m1, test_loader, device=device)
    ablation_results.append({
        "Model": "ShuffleNetV2 x0.5 Baseline (No KD)",
        "Accuracy (%)": round(e1["accuracy"] * 100, 2),
        "Weighted F1 (%)": round(e1["f1_score"] * 100, 2),
        "Macro F1 (%)": round(e1["macro_f1_score"] * 100, 2),
        "Parameter Count (M)": round(count_parameters(m1), 2),
        "Model Size (MB)": round(get_model_size_mb(exp1_ckpt), 2),
    })

    # -------------------------------------------------------------
    # Exp 2: ShuffleNetV2 x0.5 + CBAM (No KD)
    # -------------------------------------------------------------
    print("\n>>> ABLATION 2/5: ShuffleNetV2 x0.5 + CBAM...")
    exp2_ckpt = ckpt_dir / "exp2_shufflenet_v05_cbam.pth"
    exp2_json = ckpt_dir / "exp2_shufflenet_v05_cbam_metrics.json"

    if not exp2_ckpt.exists() or args.force:
        run_cmd([
            python_exe, "-m", "src.training.train_baseline",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--image-size", str(args.image_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(exp2_ckpt),
            "--output-metrics-json", str(exp2_json),
        ])

    m2 = create_shufflenet_v2_x0_5_cbam(num_classes=6, pretrained=False)
    m2.load_state_dict(torch.load(exp2_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    e2 = evaluate_model(m2, test_loader, device=device)
    ablation_results.append({
        "Model": "ShuffleNetV2 x0.5 + CBAM (No KD)",
        "Accuracy (%)": round(e2["accuracy"] * 100, 2),
        "Weighted F1 (%)": round(e2["f1_score"] * 100, 2),
        "Macro F1 (%)": round(e2["macro_f1_score"] * 100, 2),
        "Parameter Count (M)": round(count_parameters(m2), 2),
        "Model Size (MB)": round(get_model_size_mb(exp2_ckpt), 2),
    })

    # -------------------------------------------------------------
    # Exp 3: EfficientNet-B0 Teacher -> ShuffleNetV2 x0.5 Logit KD
    # Controlled Hyperparameter Sweep (Temperature & Alpha) on Validation set
    # -------------------------------------------------------------
    print("\n>>> ABLATION 3/5: Logit Knowledge Distillation Sweep...")
    logit_configs = [
        {"T": 2.0, "alpha": 0.5},
        {"T": 4.0, "alpha": 0.7},
        {"T": 6.0, "alpha": 0.9},
    ]

    best_logit_val_acc = -1.0
    best_logit_cfg = logit_configs[1]

    for cfg in logit_configs:
        t_val, a_val = cfg["T"], cfg["alpha"]
        ckpt_name = f"exp3_logit_kd_T{int(t_val)}_a{int(a_val*10)}.pth"
        ckpt_p = ckpt_dir / ckpt_name
        json_p = ckpt_dir / f"exp3_logit_kd_T{int(t_val)}_a{int(a_val*10)}_metrics.json"

        if not ckpt_p.exists() or args.force:
            run_cmd([
                python_exe, "-m", "src.training.train_kd_student",
                "--manifest-path", str(manifest_p),
                "--raw-data-root", args.raw_data_root,
                "--teacher-type", "efficientnet_b0",
                "--teacher-checkpoint", args.teacher_checkpoint,
                "--student-type", "shufflenet_v05",
                "--temperature", str(t_val),
                "--alpha", str(a_val),
                "--feature-weight", "0.0",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(args.seed),
                "--checkpoint-path", str(ckpt_p),
                "--output-metrics-json", str(json_p),
            ])

        val_acc = torch.load(ckpt_p, map_location=device, weights_only=False).get("best_validation_accuracy", 0.0)
        print(f"[Logit KD Sweep] T={t_val}, Alpha={a_val} -> Val Acc: {val_acc:.2f}%")
        if val_acc > best_logit_val_acc:
            best_logit_val_acc = val_acc
            best_logit_cfg = cfg

    best_exp3_ckpt = ckpt_dir / f"exp3_logit_kd_T{int(best_logit_cfg['T'])}_a{int(best_logit_cfg['alpha']*10)}.pth"
    m3 = create_shufflenet_v2_x0_5(num_classes=6, pretrained=False)
    m3.load_state_dict(torch.load(best_exp3_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    e3 = evaluate_model(m3, test_loader, device=device)
    ablation_results.append({
        "Model": f"ShuffleNetV2 x0.5 Logit KD (T={best_logit_cfg['T']}, α={best_logit_cfg['alpha']})",
        "Accuracy (%)": round(e3["accuracy"] * 100, 2),
        "Weighted F1 (%)": round(e3["f1_score"] * 100, 2),
        "Macro F1 (%)": round(e3["macro_f1_score"] * 100, 2),
        "Parameter Count (M)": round(count_parameters(m3), 2),
        "Model Size (MB)": round(get_model_size_mb(best_exp3_ckpt), 2),
    })

    # -------------------------------------------------------------
    # Exp 4: EfficientNet-B0 Teacher -> ShuffleNetV2 x0.5 Feature KD
    # Controlled Hyperparameter Sweep (Beta) on Validation set
    # -------------------------------------------------------------
    print("\n>>> ABLATION 4/5: Feature Knowledge Distillation Sweep...")
    feature_weights = [0.1, 0.5, 1.0]
    best_feat_val_acc = -1.0
    best_feat_weight = feature_weights[1]

    for beta in feature_weights:
        ckpt_name = f"exp4_feature_kd_b{int(beta*10)}.pth"
        ckpt_p = ckpt_dir / ckpt_name
        json_p = ckpt_dir / f"exp4_feature_kd_b{int(beta*10)}_metrics.json"

        if not ckpt_p.exists() or args.force:
            run_cmd([
                python_exe, "-m", "src.training.train_kd_student",
                "--manifest-path", str(manifest_p),
                "--raw-data-root", args.raw_data_root,
                "--teacher-type", "efficientnet_b0",
                "--teacher-checkpoint", args.teacher_checkpoint,
                "--student-type", "shufflenet_v05",
                "--alpha", "0.0",
                "--feature-weight", str(beta),
                "--feature-loss-type", "mse",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(args.seed),
                "--checkpoint-path", str(ckpt_p),
                "--output-metrics-json", str(json_p),
            ])

        val_acc = torch.load(ckpt_p, map_location=device, weights_only=False).get("best_validation_accuracy", 0.0)
        print(f"[Feature KD Sweep] Beta={beta} -> Val Acc: {val_acc:.2f}%")
        if val_acc > best_feat_val_acc:
            best_feat_val_acc = val_acc
            best_feat_weight = beta

    best_exp4_ckpt = ckpt_dir / f"exp4_feature_kd_b{int(best_feat_weight*10)}.pth"
    m4 = create_shufflenet_v2_x0_5(num_classes=6, pretrained=False)
    m4.load_state_dict(torch.load(best_exp4_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    e4 = evaluate_model(m4, test_loader, device=device)
    ablation_results.append({
        "Model": f"ShuffleNetV2 x0.5 Feature KD (β={best_feat_weight})",
        "Accuracy (%)": round(e4["accuracy"] * 100, 2),
        "Weighted F1 (%)": round(e4["f1_score"] * 100, 2),
        "Macro F1 (%)": round(e4["macro_f1_score"] * 100, 2),
        "Parameter Count (M)": round(count_parameters(m4), 2),
        "Model Size (MB)": round(get_model_size_mb(best_exp4_ckpt), 2),
    })

    # -------------------------------------------------------------
    # Exp 5: Combined Logit + Feature KD (ShuffleNetV2 x0.5 & x0.5+CBAM)
    # -------------------------------------------------------------
    print("\n>>> ABLATION 5/5: Combined Logit + Feature Knowledge Distillation...")
    opt_T = best_logit_cfg["T"]
    opt_alpha = best_logit_cfg["alpha"]
    opt_beta = best_feat_weight

    # Exp 5a: Standard Student
    exp5a_ckpt = ckpt_dir / "exp5a_combined_kd_shufflenet_v05.pth"
    exp5a_json = ckpt_dir / "exp5a_combined_kd_shufflenet_v05_metrics.json"

    if not exp5a_ckpt.exists() or args.force:
        run_cmd([
            python_exe, "-m", "src.training.train_kd_student",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--teacher-type", "efficientnet_b0",
            "--teacher-checkpoint", args.teacher_checkpoint,
            "--student-type", "shufflenet_v05",
            "--temperature", str(opt_T),
            "--alpha", str(opt_alpha),
            "--feature-weight", str(opt_beta),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(exp5a_ckpt),
            "--output-metrics-json", str(exp5a_json),
        ])

    m5a = create_shufflenet_v2_x0_5(num_classes=6, pretrained=False)
    m5a.load_state_dict(torch.load(exp5a_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    e5a = evaluate_model(m5a, test_loader, device=device)
    ablation_results.append({
        "Model": f"ShuffleNetV2 x0.5 Combined KD (T={opt_T}, α={opt_alpha}, β={opt_beta})",
        "Accuracy (%)": round(e5a["accuracy"] * 100, 2),
        "Weighted F1 (%)": round(e5a["f1_score"] * 100, 2),
        "Macro F1 (%)": round(e5a["macro_f1_score"] * 100, 2),
        "Parameter Count (M)": round(count_parameters(m5a), 2),
        "Model Size (MB)": round(get_model_size_mb(exp5a_ckpt), 2),
    })

    # Exp 5b: CBAM Student
    exp5b_ckpt = ckpt_dir / "exp5b_combined_kd_shufflenet_v05_cbam.pth"
    exp5b_json = ckpt_dir / "exp5b_combined_kd_shufflenet_v05_cbam_metrics.json"

    if not exp5b_ckpt.exists() or args.force:
        run_cmd([
            python_exe, "-m", "src.training.train_kd_student",
            "--manifest-path", str(manifest_p),
            "--raw-data-root", args.raw_data_root,
            "--teacher-type", "efficientnet_b0",
            "--teacher-checkpoint", args.teacher_checkpoint,
            "--student-type", "shufflenet_v05_cbam",
            "--temperature", str(opt_T),
            "--alpha", str(opt_alpha),
            "--feature-weight", str(opt_beta),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoint-path", str(exp5b_ckpt),
            "--output-metrics-json", str(exp5b_json),
        ])

    m5b = create_shufflenet_v2_x0_5_cbam(num_classes=6, pretrained=False)
    m5b.load_state_dict(torch.load(exp5b_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    e5b = evaluate_model(m5b, test_loader, device=device)
    ablation_results.append({
        "Model": f"ShuffleNetV2 x0.5 + CBAM Combined KD (T={opt_T}, α={opt_alpha}, β={opt_beta})",
        "Accuracy (%)": round(e5b["accuracy"] * 100, 2),
        "Weighted F1 (%)": round(e5b["f1_score"] * 100, 2),
        "Macro F1 (%)": round(e5b["macro_f1_score"] * 100, 2),
        "Parameter Count (M)": round(count_parameters(m5b), 2),
        "Model Size (MB)": round(get_model_size_mb(exp5b_ckpt), 2),
    })

    # Save full Ablation Table JSON
    out_table_p = ckpt_dir / "ablation_summary_table.json"
    with open(out_table_p, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Print Markdown Ablation Table
    print("\n\n=======================================================")
    print("PHYTO CONTROLLED KNOWLEDGE DISTILLATION ABLATION TABLE")
    print("=======================================================")
    headers = ["Model", "Accuracy (%)", "Weighted F1 (%)", "Macro F1 (%)", "Parameter Count (M)", "Model Size (MB)"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---'] * len(headers))} |")
    for r in ablation_results:
        row_str = " | ".join(str(r[h]) for h in headers)
        print(f"| {row_str} |")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
