"""
Training script for ShuffleNetV2 x0.5 Baseline Model for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Trains ShuffleNetV2 x0.5 baseline (without CBAM or Distillation) on the 6-class Phyto dataset.
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
from src.models import create_shufflenet_v2_x0_5
from src.training.train import set_seed, train_model
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
    parser = argparse.ArgumentParser(description="Train ShuffleNetV2 x0.5 Baseline Model")
    parser.add_argument("--manifest-path", type=str, default="results/new_dataset_manifest/groundnut_dataset_split_manifest.csv")
    parser.add_argument("--raw-data-root", type=str, default=r"c:\Users\Shwet\Desktop\Groundnut_Leaf_dataset")
    parser.add_argument("--image-size", type=int, default=256, help="Input resolution (default: 256 for paper match)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs (default: 35 for paper match)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="sgd", help="Optimizer type")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01 for SGD)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="results/checkpoints/baseline_shufflenet_v05.pth")
    parser.add_argument("--output-metrics-json", type=str, default="results/checkpoints/baseline_shufflenet_v05_metrics.json")
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
    print(f"Training ShuffleNetV2 x0.5 Baseline on device: {device} ({args.image_size}x{args.image_size}, {args.optimizer.upper()}, LR={args.lr}, Epochs={args.epochs})")

    model = create_shufflenet_v2_x0_5(num_classes=6, pretrained=True)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_path=ckpt_path,
    )

    # Load best model for evaluation on test set
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best baseline checkpoint from {ckpt_path} (Val Acc: {checkpoint.get('best_validation_accuracy', 0.0):.2f}%)")

    test_metrics = evaluate_model(model, test_loader, device=device)
    print("\n=== ShuffleNetV2 x0.5 Baseline Test Results ===")
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted F1:   {test_metrics['f1_score'] * 100:.2f}%")
    print(f"Macro F1:      {test_metrics['macro_f1_score'] * 100:.2f}%")

    out_metrics_p = Path(args.output_metrics_json)
    out_metrics_p.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "model_name": "ShuffleNetV2 x0.5 Baseline",
        "config": {
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
        },
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(out_metrics_p, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved baseline metrics to {out_metrics_p}")



if __name__ == "__main__":
    main()
