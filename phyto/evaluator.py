import os
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from phyto.config import Config
from phyto.quantization import benchmark_inference

def get_model_size_mb(model: nn.Module, filepath: str = None) -> float:
    """
    Calculates model disk file size in Megabytes (MB).
    """
    if filepath and os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    
    # Estimate from state dict parameters
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def get_parameter_count(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters) in model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def evaluate_comprehensive(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str = "Model",
    checkpoint_path: str = None,
    device: str = Config.DEVICE
) -> Dict[str, Any]:
    """
    Performs rigorous test set evaluation computing accuracy, F1, confusion matrix,
    model size, parameters, and inference latency/FPS.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Classification Metrics
    accuracy = accuracy_score(all_targets, all_preds) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    conf_mat = confusion_matrix(all_targets, all_preds)

    # Physical Model Metrics
    total_params, trainable_params = get_parameter_count(model)
    model_size_mb = get_model_size_mb(model, checkpoint_path)

    # Latency Benchmarking (1 sample batch)
    dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
    timing_res = benchmark_inference(model, dummy_input, num_runs=50, device=device)

    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision * 100.0,
        "recall_macro": recall * 100.0,
        "f1_macro": f1 * 100.0,
        "confusion_matrix": conf_mat,
        "total_params_m": total_params / 1e6,
        "model_size_mb": model_size_mb,
        "latency_ms": timing_res["latency_ms"],
        "fps": timing_res["fps"]
    }

    return results
