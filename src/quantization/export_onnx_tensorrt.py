"""
ONNX Export and TensorRT / ONNX Runtime Edge Benchmarking Pipeline for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.models import create_shufflenet_v2_x0_5



def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str = "results/checkpoints/kd_shufflenet_v05.onnx",
    input_shape: tuple = (1, 3, 224, 224),
    device: str = "cpu",
) -> Path:
    target_p = Path(onnx_path)
    target_p.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape, device=device)
    model.to(device)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        str(target_p),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Successfully exported ONNX model to: {target_p.resolve()} ({target_p.stat().st_size / (1024*1024):.2f} MB)")
    return target_p


def benchmark_onnxruntime(onnx_path: str, num_runs: int = 100, warmup: int = 10):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Notice] `onnxruntime` is not installed. Skipping ONNX Runtime benchmark.")
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_lat = float(np.mean(latencies))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    active_provider = session.get_providers()[0]

    print(f"ONNX Runtime ({active_provider}) Latency: {avg_lat:.2f} ms/img ({fps:.1f} FPS)")
    return {"latency_ms": round(avg_lat, 4), "fps": round(fps, 2), "provider": active_provider}


def benchmark_tensorrt(onnx_path: str):
    try:
        import tensorrt as trt  # pyright: ignore [reportMissingImports]
        print("[Info] TensorRT library detected.")
    except ImportError:
        print("[Notice] `tensorrt` is not installed in the current environment.")
        print("         To build TensorRT engines, run on an NVIDIA Jetson or CUDA environment with TensorRT installed.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export KD Student Model to ONNX & Benchmark TensorRT/ONNX Runtime")
    parser.add_argument("--student-checkpoint", type=str, default="results/checkpoints/kd_shufflenet_v05_from_resnet50.pth")
    parser.add_argument("--onnx-path", type=str, default="results/checkpoints/kd_shufflenet_v05.onnx")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model = create_shufflenet_v2_x0_5(num_classes=6, pretrained=False)

    ckpt_p = Path(args.student_checkpoint)
    if ckpt_p.exists():
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        student_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded student weights from {ckpt_p}")

    onnx_file = export_to_onnx(student_model, onnx_path=args.onnx_path, device=device)

    print("\n--- ONNX Runtime Benchmark ---")
    benchmark_onnxruntime(str(onnx_file))

    print("\n--- TensorRT Benchmark ---")
    benchmark_tensorrt(str(onnx_file))


if __name__ == "__main__":
    main()
