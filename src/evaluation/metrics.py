"""
Evaluation Metrics Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Calculates accuracy, precision, recall, F1-score, confusion matrix,
and per-class breakdown for model evaluation on test data using scikit-learn.
"""

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
