import os
import time
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from phyto.config import Config

def export_to_onnx(
    model: nn.Module,
    save_path: str = "checkpoints/phyto_model.onnx",
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu"
) -> str:
    """
    Exports PyTorch model to ONNX format for TensorRT / ONNX Runtime edge execution.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"[ONNX Export] Model successfully exported to: {save_path}")
    return save_path


def quantize_model_int8(
    model: nn.Module,
    calibration_loader: DataLoader = None,
    save_path: str = "checkpoints/phyto_model_int8.pth",
    device: str = "cpu"
) -> nn.Module:
    """
    Applies Post-Training INT8 Dynamic/Static Quantization to reduce model size and accelerate CPU/Edge inference.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    model.to("cpu")

    # PyTorch Dynamic Quantization for Linear / Conv layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    torch.save(quantized_model.state_dict(), save_path)
    print(f"[Quantization] INT8 Quantized model saved to: {save_path}")
    return quantized_model


def benchmark_inference(
    model: nn.Module,
    dummy_input: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
    device: str = Config.DEVICE
) -> Dict[str, float]:
    """
    Measures average inference latency (ms/image) and throughput (FPS).
    """
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Latency timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time

    return {
        "latency_ms": avg_latency_ms,
        "fps": fps
    }
