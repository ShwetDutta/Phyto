"""
Project Phyto: Training CLI Script
Trains Teacher Model, Baseline ShuffleNetV2, and Proposed Knowledge-Distilled CBAM-ShuffleNetV2.
"""

import os
import argparse
from phyto.config import Config
from phyto.dataset import build_dataloaders
from phyto.models import TeacherResNet50, BaselineShuffleNetV2, CBAMShuffleNetV2
from phyto.trainer import train_model
from phyto.quantization import quantize_model_int8, export_to_onnx

def main():
    parser = argparse.ArgumentParser(description="Phyto Local Model Training & Optimization Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Raw_Data folder (Optional: auto-resolves via kagglehub)")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs-teacher", type=int, default=Config.NUM_EPOCHS_TEACHER, help="Teacher epochs")
    parser.add_argument("--epochs-student", type=int, default=Config.NUM_EPOCHS_STUDENT, help="Student epochs")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    Config.setup_directories()

    # 1. Load Data
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # 2. Train Teacher Model (ResNet50)
    print("\n>>> Phase 1: Training Teacher Model (ResNet50)...")
    teacher = TeacherResNet50(num_classes=Config.NUM_CLASSES)
    train_model(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs_teacher,
        lr=args.lr,
        save_filename="teacher_resnet50.pth"
    )

    # 3. Train Baseline Model (ShuffleNetV2 1.0x without Attention or KD)
    print("\n>>> Phase 2: Training Baseline Model (ShuffleNetV2 1.0x)...")
    baseline = BaselineShuffleNetV2(num_classes=Config.NUM_CLASSES)
    train_model(
        model=baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs_student,
        lr=args.lr,
        save_filename="baseline_shufflenetv2.pth"
    )

    # 4. Train Proposed Model (CBAM-ShuffleNetV2 + Knowledge Distillation)
    print("\n>>> Phase 3: Training Proposed Model (CBAM-ShuffleNetV2 + KD)...")
    proposed = CBAMShuffleNetV2(num_classes=Config.NUM_CLASSES)
    train_model(
        model=proposed,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs_student,
        lr=args.lr,
        teacher_model=teacher,
        save_filename="proposed_cbam_shufflenetv2_kd.pth"
    )

    # 5. Quantize & Export Edge Model
    print("\n>>> Phase 4: Quantization & Edge Export...")
    quantized_model = quantize_model_int8(
        proposed,
        save_path="checkpoints/proposed_cbam_shufflenetv2_int8.pth"
    )
    export_to_onnx(
        proposed,
        save_path="checkpoints/proposed_cbam_shufflenetv2.onnx"
    )

    print("\n[SUCCESS] Phyto Training & Optimization Pipeline Executed Successfully!")

if __name__ == "__main__":
    main()
