"""
PyTorch INT8 Dynamic & Static Quantization Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Provides dynamic post-training INT8 quantization and model file size utilities.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def quantize_model_dynamic(
    model: nn.Module,
    qconfig_spec: Optional[Any] = None,
) -> nn.Module:
    """
    Applies dynamic INT8 post-training quantization to model Linear layers.

    Args:
        model: Floating-point PyTorch model
        qconfig_spec: Optional layer specification for dynamic quantization (defaults to nn.Linear)

    Returns:
        nn.Module: Dynamically quantized PyTorch model.
    """
    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()

    if qconfig_spec is None:
        qconfig_spec = {nn.Linear}

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8,
    )
    return quantized_model


def get_model_size_mb(model: nn.Module, temp_filepath: Union[str, Path] = "temp_model.pt") -> float:
    """
    Calculates physical file size of saved model weights in Megabytes (MB).

    Args:
        model: PyTorch model instance
        temp_filepath: Temporary path to write state dict for size measurement

    Returns:
        float: File size in MB rounded to 4 decimal places.
    """
    p = Path(temp_filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    size_bytes = p.stat().st_size
    if p.exists():
        p.unlink()
    return round(size_bytes / (1024 * 1024), 4)
