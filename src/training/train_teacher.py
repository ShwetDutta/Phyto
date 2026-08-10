"""
Training script for CBAM-enhanced ResNet50 Teacher Model for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.phyto_dataset import PhytoDataset, load_split_manifest
from src.models import create_resnet50_cbam
from src.training.train import set_seed, train_model
from src.evaluation.metrics import evaluate_model


def get_default_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_test_transform


def main():
    parser = argparse.ArgumentParser(description="Train CBAM ResNet50 Teacher Model")
    parser.add_argument("--manifest-path", type=str, default="results/new_dataset_manifest/groundnut_dataset_split_manifest.csv")
    parser.add_argument("--raw-data-root", type=str, default=r"c:\Users\Shwet\Desktop\Groundnut_Leaf_dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="results/checkpoints/cbam_teacher_resnet50.pth")
    parser.add_argument("--output-metrics-json", type=str, default="results/checkpoints/cbam_teacher_resnet50_metrics.json")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data splits
    train_df = load_split_manifest(args.manifest_path, split="train")
    val_df = load_split_manifest(args.manifest_path, split="validation")
    test_df = load_split_manifest(args.manifest_path, split="test")

    train_tf, val_test_tf = get_default_transforms()

    train_dataset = PhytoDataset(train_df, raw_data_root=args.raw_data_root, transform=train_tf)
    val_dataset = PhytoDataset(val_df, raw_data_root=args.raw_data_root, transform=val_test_tf)
    test_dataset = PhytoDataset(test_df, raw_data_root=args.raw_data_root, transform=val_test_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ResNet50 + CBAM Teacher on device: {device}")

    model = create_resnet50_cbam(num_classes=6, pretrained=True)
    criterion = nn.CrossEntropyLoss()
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
        print(f"Loaded best checkpoint from {ckpt_path} (Val Acc: {checkpoint.get('best_validation_accuracy', 0.0):.2f}%)")

    test_metrics = evaluate_model(model, test_loader, device=device)
    print("\n=== ResNet50 + CBAM Teacher Test Results ===")
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted F1:   {test_metrics['f1_score'] * 100:.2f}%")
    print(f"Macro F1:      {test_metrics['macro_f1_score'] * 100:.2f}%")
    print(f"Rust Recall:   {test_metrics['per_class_recall'].get('rust', 0.0) * 100:.2f}%")

    out_metrics_p = Path(args.output_metrics_json)
    out_metrics_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_p, "w") as f:
        json.dump({"test_metrics": test_metrics, "history": history}, f, indent=2)

    print(f"Saved teacher metrics to {out_metrics_p}")


if __name__ == "__main__":
    main()
