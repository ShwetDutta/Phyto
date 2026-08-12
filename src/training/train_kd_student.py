"""
Knowledge Distillation Training Script for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Distills feature representation from trained ResNet50 + CBAM Teacher to ShuffleNetV2 x0.5 Student.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.phyto_dataset import PhytoDataset, load_split_manifest
from src.models import (
    create_resnet50_cbam,
    create_efficientnet_b0,
    create_shufflenet_v2_x0_5,
    create_shufflenet_v2_x0_5_cbam,
    FeatureKDAdapter,
)
from src.training.distillation import train_distillation_model
from src.training.train import set_seed
from src.evaluation.metrics import evaluate_model



def get_default_transforms(image_size: int = 256):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_test_transform


def main():
    parser = argparse.ArgumentParser(description="Train ShuffleNetV2 x0.5 Student via Knowledge Distillation")
    parser.add_argument("--manifest-path", type=str, default="results/new_dataset_manifest/groundnut_dataset_split_manifest.csv")
    parser.add_argument("--raw-data-root", type=str, default=r"c:\Users\Shwet\Desktop\Groundnut_Leaf_dataset")
    parser.add_argument("--teacher-type", type=str, choices=["resnet50_cbam", "efficientnet_b0"], default="efficientnet_b0")
    parser.add_argument("--teacher-checkpoint", type=str, default="results/checkpoints/teacher_efficientnet_b0.pth")
    parser.add_argument("--student-type", type=str, choices=["shufflenet_v05", "shufflenet_v05_cbam"], default="shufflenet_v05")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    parser.add_argument("--feature-loss-type", type=str, choices=["mse", "cosine"], default="mse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="results/checkpoints/kd_shufflenet_v05_from_efficientnet.pth")
    parser.add_argument("--output-metrics-json", type=str, default="results/checkpoints/kd_shufflenet_v05_metrics.json")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint file to resume training from")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data splits
    train_df = load_split_manifest(args.manifest_path, split="train")
    val_df = load_split_manifest(args.manifest_path, split="validation")
    test_df = load_split_manifest(args.manifest_path, split="test")

    train_tf, val_test_tf = get_default_transforms(image_size=args.image_size)

    train_dataset = PhytoDataset(train_df, raw_data_root=args.raw_data_root, transform=train_tf)
    val_dataset = PhytoDataset(val_df, raw_data_root=args.raw_data_root, transform=val_test_tf)
    test_dataset = PhytoDataset(test_df, raw_data_root=args.raw_data_root, transform=val_test_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Knowledge Distillation ({args.teacher_type} -> {args.student_type}) on device: {device}")
    print(f"Distill Config: Alpha={args.alpha}, Temperature={args.temperature}, FeatureWeight={args.feature_weight}")

    # Load teacher model
    if args.teacher_type == "efficientnet_b0":
        teacher_model = create_efficientnet_b0(num_classes=6, pretrained=False)
        t_channels = 1280
    else:
        teacher_model = create_resnet50_cbam(num_classes=6, pretrained=False)
        t_channels = 2048

    t_ckpt_p = Path(args.teacher_checkpoint)
    if not t_ckpt_p.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found at: {t_ckpt_p.resolve()}")

    t_ckpt = torch.load(t_ckpt_p, map_location=device, weights_only=False)
    teacher_model.load_state_dict(t_ckpt["model_state_dict"])
    teacher_model.to(device)
    teacher_model.eval()
    print(f"Successfully loaded {args.teacher_type} Teacher weights from {t_ckpt_p}")

    # Instantiate student model
    if args.student_type == "shufflenet_v05_cbam":
        student_model = create_shufflenet_v2_x0_5_cbam(num_classes=6, pretrained=True)
    else:
        student_model = create_shufflenet_v2_x0_5(num_classes=6, pretrained=True)

    # Feature Adapter
    feature_adapter = None
    if args.feature_weight > 0.0:
        s_channels = 1024
        feature_adapter = FeatureKDAdapter(teacher_channels=t_channels, student_channels=s_channels).to(device)
        trainable_params = list(student_model.parameters()) + list(feature_adapter.parameters())
    else:
        trainable_params = list(student_model.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_last{ckpt_path.suffix}")

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is None and last_ckpt_path.exists():
        resume_ckpt = str(last_ckpt_path)
        print(f"[Auto-Resume] Found existing epoch checkpoint at {last_ckpt_path}. Resuming training...")

    history = train_distillation_model(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        alpha=args.alpha,
        temperature=args.temperature,
        feature_weight=args.feature_weight,
        feature_loss_type=args.feature_loss_type,
        feature_adapter=feature_adapter,
        checkpoint_path=ckpt_path,
        resume_from_checkpoint=resume_ckpt,
    )

    # Load best student checkpoint for test evaluation
    if ckpt_path.exists():
        s_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student_model.load_state_dict(s_ckpt["model_state_dict"])
        print(f"Loaded best student checkpoint from {ckpt_path} (Val Acc: {s_ckpt.get('best_validation_accuracy', 0.0):.2f}%)")

    test_metrics = evaluate_model(student_model, test_loader, device=device)
    print("\n=== ShuffleNetV2 x0.5 KD Student Test Results ===")
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted F1:   {test_metrics['f1_score'] * 100:.2f}%")
    print(f"Macro F1:      {test_metrics['macro_f1_score'] * 100:.2f}%")
    print(f"Rust Recall:   {test_metrics['per_class_recall'].get('rust', 0.0) * 100:.2f}%")

    out_metrics_p = Path(args.output_metrics_json)
    out_metrics_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_p, "w") as f:
        json.dump({"test_metrics": test_metrics, "history": history, "kd_config": {"temperature": args.temperature, "alpha": args.alpha}}, f, indent=2)

    print(f"Saved student metrics to {out_metrics_p}")


if __name__ == "__main__":
    main()
