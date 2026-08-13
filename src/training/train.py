"""
Training and Inference Benchmarking Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Provides model training with Colab-resumable checkpointing, seed initialization,
checkpoint loading, and inference latency benchmarking.
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import benchmark_inference


def set_seed(seed: int = 42) -> None:
    """
    Sets random seed for Python, NumPy, and PyTorch across CPU and CUDA for reproducibility.

    Args:
        seed: Fixed integer seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[
    nn.Module,
    Optional[torch.optim.Optimizer],
    Optional[Any],
    int,
    float,
    Dict[str, List[float]],
]:
    """
    Loads a model checkpoint and restores model, optimizer, and scheduler states.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model instance to load weights into
        optimizer: Optional PyTorch optimizer to restore state
        scheduler: Optional learning rate scheduler to restore state
        device: Target execution device ('cuda' or 'cpu'); auto-detects if None

    Returns:
        Tuple containing:
            - model: Restored PyTorch model
            - optimizer: Restored optimizer (or None if not provided)
            - scheduler: Restored scheduler (or None if not provided)
            - start_epoch: Int representing the completed epoch count from checkpoint
            - best_validation_accuracy: Best validation accuracy recorded
            - history: Dictionary of training history metrics
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_p.resolve()}")

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    checkpoint = torch.load(ckpt_p, map_location=target_device, weights_only=False)

    model = model.to(target_device)
    if "model_state_dict" in checkpoint:
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            if (
                optimizer is not None
                and "optimizer_state_dict" in checkpoint
                and checkpoint["optimizer_state_dict"] is not None
            ):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if (
                scheduler is not None
                and "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"] is not None
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            print(f"\n[Warning] Auto-resume checkpoint state_dict mismatch for {ckpt_p.name}: {e}")
            print("[Notice] Overriding auto-resume checkpoint due to architecture mismatch. Starting training from scratch.\n")
            return model, optimizer, scheduler, 0, 0.0, {
                "train_loss": [],
                "train_accuracy": [],
                "validation_loss": [],
                "validation_accuracy": [],
                "learning_rate": [],
                "epoch_time": [],
            }
    else:
        raise KeyError("Checkpoint missing required key 'model_state_dict'")

    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_acc = float(checkpoint.get("best_validation_accuracy", 0.0))
    history = checkpoint.get(
        "training_history",
        {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "learning_rate": [],
            "epoch_time": [],
        },
    )

    return model, optimizer, scheduler, start_epoch, best_val_acc, history


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: Optional[Union[torch.device, str]] = None,
    epochs: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
    last_checkpoint_path: Optional[Union[str, Path]] = None,
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
) -> Dict[str, List[float]]:
    """
    Executes model training loop over specified epochs, tracks performance history,
    saves the best model checkpoint on validation improvement, and saves a last checkpoint
    after every completed epoch for Colab resilience.

    Args:
        model: PyTorch classification model instance
        train_loader: DataLoader for training set
        validation_loader: DataLoader for validation set
        criterion: Loss function module
        optimizer: PyTorch optimizer instance
        scheduler: Optional learning rate scheduler
        device: Execution device ('cuda' or 'cpu'); auto-detects if None
        epochs: Total target training epochs
        checkpoint_path: Path to save the best model checkpoint
        last_checkpoint_path: Optional explicit path to save the last epoch checkpoint
        resume_from_checkpoint: Optional checkpoint path to resume training from

    Returns:
        Dict[str, List[float]]: Training history dictionary containing epoch metrics.
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    model = model.to(target_device)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
        "epoch_time": [],
    }

    best_val_acc: float = -1.0
    start_epoch: int = 1

    # Resume from checkpoint if requested
    if resume_from_checkpoint is not None:
        (
            model,
            optimizer_loaded,
            scheduler_loaded,
            completed_epoch,
            loaded_best_val_acc,
            loaded_history,
        ) = load_checkpoint(
            checkpoint_path=resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=target_device,
        )
        if optimizer_loaded is not None:
            optimizer = optimizer_loaded
        if scheduler_loaded is not None:
            scheduler = scheduler_loaded

        start_epoch = completed_epoch + 1
        best_val_acc = loaded_best_val_acc
        history = loaded_history
        print(f"Resuming training from epoch {start_epoch} (completed {completed_epoch}/{epochs}).")

    if start_epoch > epochs:
        print(f"Training already completed ({start_epoch - 1}/{epochs} epochs). Returning history.")
        return history

    # Determine last checkpoint path for epoch-level saving
    target_last_path: Optional[Path] = None
    if last_checkpoint_path is not None:
        target_last_path = Path(last_checkpoint_path)
    elif checkpoint_path is not None:
        cp = Path(checkpoint_path)
        target_last_path = cp.with_name(f"{cp.stem}_last{cp.suffix}")

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.perf_counter()

        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        total_batches = len(train_loader)

        print(f"Epoch {epoch}/{epochs} Training Progress:")
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_train_loss += float(loss.item()) * batch_size
            _, preds = torch.max(outputs, 1)
            correct_train += int((preds == labels).sum().item())
            total_train += batch_size

            if batch_idx % 50 == 0 or batch_idx == total_batches:
                curr_acc = (correct_train / total_train) * 100.0 if total_train > 0 else 0.0
                curr_loss = running_train_loss / total_train if total_train > 0 else 0.0
                print(
                    f"  Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{total_batches}] "
                    f"- Loss: {curr_loss:.4f} | Acc: {curr_acc:.2f}%",
                    flush=True
                )

        epoch_train_loss = running_train_loss / total_train if total_train > 0 else 0.0
        epoch_train_acc = (correct_train / total_train) * 100.0 if total_train > 0 else 0.0

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(target_device)
                labels = labels.to(target_device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                running_val_loss += float(loss.item()) * batch_size
                _, preds = torch.max(outputs, 1)
                correct_val += int((preds == labels).sum().item())
                total_val += batch_size

        epoch_val_loss = running_val_loss / total_val if total_val > 0 else 0.0
        epoch_val_acc = (correct_val / total_val) * 100.0 if total_val > 0 else 0.0

        current_lr = float(optimizer.param_groups[0]["lr"])

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        elapsed_time = time.perf_counter() - start_time

        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["validation_loss"].append(epoch_val_loss)
        history["validation_accuracy"].append(epoch_val_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(elapsed_time)

        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc

        checkpoint_data: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_validation_accuracy": best_val_acc,
            "training_history": history,
        }

        # Save BEST model whenever validation accuracy improves
        if is_best and checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            ckpt_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, ckpt_p)

        # Save LAST checkpoint after every completed epoch
        if target_last_path is not None:
            target_last_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, target_last_path)

        best_flag = " [BEST]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - "
            f"Time: {elapsed_time:.2f}s - LR: {current_lr:.6f} - "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%{best_flag}"
        )

    return history


__all__ = ["set_seed", "load_checkpoint", "train_model", "benchmark_inference"]
