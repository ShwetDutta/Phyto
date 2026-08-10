"""
Training and Inference Benchmarking Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Provides model training, seed initialization, and inference latency benchmarking.
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
) -> Dict[str, List[float]]:
    """
    Executes model training loop over specified epochs, tracks performance history,
    and saves the best model checkpoint based on validation accuracy.

    Args:
        model: PyTorch classification model instance
        train_loader: DataLoader for training set
        validation_loader: DataLoader for validation set
        criterion: Loss function module
        optimizer: PyTorch optimizer instance
        scheduler: Optional learning rate scheduler
        device: Execution device ('cuda' or 'cpu'); auto-detects if None
        epochs: Number of training epochs
        checkpoint_path: Optional file path to save best model checkpoint

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

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()

        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
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
            if checkpoint_path is not None:
                ckpt_p = Path(checkpoint_path)
                ckpt_p.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_data: Dict[str, Any] = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_validation_accuracy": best_val_acc,
                    "training_history": history,
                }
                torch.save(checkpoint_data, ckpt_p)

        best_flag = " [BEST]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - "
            f"Time: {elapsed_time:.2f}s - LR: {current_lr:.6f} - "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%{best_flag}"
        )

    return history


def benchmark_inference(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: Optional[Union[torch.device, str]] = None,
    num_warmup_batches: int = 5,
) -> Dict[str, float]:
    """
    Measures model inference latency and frames-per-second (FPS) on a given DataLoader.
    Excludes image loading and I/O time from inference latency calculation.

    Args:
        model: PyTorch model instance
        dataloader: DataLoader providing input batches
        device: Device to run inference on ('cuda' or 'cpu')
        num_warmup_batches: Number of initial batches for GPU/CPU warmup

    Returns:
        Dict[str, float]: Dictionary containing average_latency_ms, median_latency_ms, and fps.
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    model = model.to(target_device)
    model.eval()

    is_cuda = target_device.type == "cuda"
    per_image_latencies_ms: List[float] = []

    with torch.no_grad():
        batch_idx = 0
        for images, _ in dataloader:
            images = images.to(target_device)
            batch_size = images.size(0)

            if batch_idx < num_warmup_batches:
                _ = model(images)
                if is_cuda:
                    torch.cuda.synchronize()
                batch_idx += 1
                continue

            if is_cuda:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(images)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latency_ms = (t1 - t0) * 1000.0
            per_image_latency_ms = batch_latency_ms / batch_size
            per_image_latencies_ms.extend([per_image_latency_ms] * batch_size)
            batch_idx += 1

    if not per_image_latencies_ms:
        raise ValueError("No inference samples available for benchmarking.")

    avg_latency_ms = float(np.mean(per_image_latencies_ms))
    median_latency_ms = float(np.median(per_image_latencies_ms))
    fps = (1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    return {
        "average_latency_ms": round(avg_latency_ms, 4),
        "median_latency_ms": round(median_latency_ms, 4),
        "fps": round(fps, 2),
    }
