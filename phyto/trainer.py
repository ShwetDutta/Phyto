import os
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from phyto.config import Config
from phyto.loss import KnowledgeDistillationLoss

def evaluate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """
    Evaluates model performance on validation/test dataloader.
    Returns (avg_loss, accuracy_percentage).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_standard_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Executes 1 training epoch using standard supervised learning.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_distillation_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kd_criterion: KnowledgeDistillationLoss,
    device: str
) -> Tuple[float, float]:
    """
    Executes 1 training epoch for Student model guided by Teacher model via Knowledge Distillation.
    """
    student_model.train()
    teacher_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(inputs)

        optimizer.zero_grad()
        student_logits = student_model(inputs)
        loss = kd_criterion(student_logits, teacher_logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = Config.LEARNING_RATE,
    teacher_model: nn.Module = None,
    save_filename: str = "best_model.pth",
    device: str = Config.DEVICE
) -> Dict[str, List[float]]:
    """
    Orchestrates full training loop with validation checks, learning rate decay, and best checkpoint saving.
    Supports both standard training and Knowledge Distillation training.
    """
    Config.setup_directories()
    save_path = os.path.join(Config.CHECKPOINT_DIR, save_filename)

    model = model.to(device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_criterion = nn.CrossEntropyLoss()
    kd_criterion = KnowledgeDistillationLoss(temperature=Config.KD_TEMPERATURE, alpha=Config.KD_ALPHA)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = -1.0
    start_time = time.time()

    mode_str = "Knowledge Distillation" if teacher_model is not None else "Standard Supervised"
    print(f"\n========================================================")
    print(f"Starting Training: {save_filename.replace('.pth','')} [{mode_str}]")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"========================================================")

    for epoch in range(1, epochs + 1):
        if teacher_model is not None:
            train_loss, train_acc = train_distillation_epoch(
                model, teacher_model, train_loader, optimizer, kd_criterion, device
            )
        else:
            train_loss, train_acc = train_standard_epoch(
                model, train_loader, optimizer, ce_criterion, device
            )

        val_loss, val_acc = evaluate_epoch(model, val_loader, ce_criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "optimizer_state_dict": optimizer.state_dict()
            }, save_path)
            print(f" ---> Best Checkpoint Saved! (Val Acc: {best_val_acc:.2f}%)")

    # Final guarantee save if checkpoint doesn't exist
    if not os.path.exists(save_path):
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_acc": history["val_acc"][-1],
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path)

    elapsed = time.time() - start_time
    print(f"Training Complete in {elapsed/60:.2f} mins. Best Val Acc: {best_val_acc:.2f}%\n")
    return history
