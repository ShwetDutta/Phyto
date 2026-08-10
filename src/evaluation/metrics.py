"""
Evaluation Metrics Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Calculates accuracy, precision, recall, F1-score, confusion matrix,
per-class breakdown, and inference latency benchmarks.
"""

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

DEFAULT_CLASS_NAMES: List[str] = [
    "early_leaf_spot",
    "healthy_leaf",
    "late_leaf_spot",
    "nutrition_deficiency",
    "rust",
]


def calculate_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes global and per-class evaluation metrics given ground truth and predicted labels.

    Args:
        y_true: Ground truth integer class labels
        y_pred: Predicted integer class labels
        class_names: Optional list of class label strings in index order (0..4)

    Returns:
        Dict[str, Any]: Structured dictionary containing:
            - accuracy
            - precision (weighted)
            - recall (weighted)
            - f1_score (weighted)
            - macro_precision
            - macro_recall
            - macro_f1_score
            - confusion_matrix (as 2D list of ints)
            - per_class_precision (dict)
            - per_class_recall (dict)
            - per_class_f1 (dict)
    """
    labels = class_names if class_names is not None else DEFAULT_CLASS_NAMES
    num_classes = len(labels)
    class_indices = list(range(num_classes))

    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))

    # Overall weighted metrics
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="weighted", zero_division=0  # pyright: ignore [reportArgumentType]
    )

    # Macro averaged metrics
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0  # pyright: ignore [reportArgumentType]
    )

    # Per-class metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        y_t, y_p, labels=class_indices, average=None, zero_division=0  # pyright: ignore [reportArgumentType]
    )

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=class_indices)
    cm_list: List[List[int]] = cm.tolist()

    p_arr: np.ndarray = np.asarray(p_class)
    r_arr: np.ndarray = np.asarray(r_class)
    f1_arr: np.ndarray = np.asarray(f1_class)

    per_class_precision: Dict[str, float] = {
        cls_name: round(float(prec), 4) for cls_name, prec in zip(labels, p_arr.tolist())
    }
    per_class_recall: Dict[str, float] = {
        cls_name: round(float(rec), 4) for cls_name, rec in zip(labels, r_arr.tolist())
    }
    per_class_f1: Dict[str, float] = {
        cls_name: round(float(f1), 4) for cls_name, f1 in zip(labels, f1_arr.tolist())
    }

    return {
        "accuracy": round(acc, 4),
        "precision": round(float(w_prec), 4),
        "recall": round(float(w_rec), 4),
        "f1_score": round(float(w_f1), 4),
        "macro_precision": round(float(m_prec), 4),
        "macro_recall": round(float(m_rec), 4),
        "macro_f1_score": round(float(m_f1), 4),
        "confusion_matrix": cm_list,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader[Any],
    device: Optional[Union[torch.device, str]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates PyTorch classification model on the test dataset loader.

    Args:
        model: Trained PyTorch model instance
        test_loader: DataLoader for test set
        device: Device to run evaluation on ('cuda' or 'cpu')
        class_names: Optional list of class names

    Returns:
        Dict[str, Any]: Evaluation metrics dictionary.
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    model = model.to(target_device)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(target_device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            if isinstance(targets, torch.Tensor):
                all_targets.extend(targets.cpu().numpy().tolist())
            else:
                all_targets.extend(list(targets))

    return calculate_metrics(all_targets, all_preds, class_names=class_names)


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
