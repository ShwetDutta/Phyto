# Phyto: Edge-AI Groundnut Leaf Disease Classification Benchmark Report

## 1. Project Overview & Objective

The **Phyto** project presents a lightweight, Edge-AI oriented deep learning framework for classifying groundnut (*Arachis hypogaea*) leaf diseases. Implemented using **PyTorch**, the pipeline addresses critical agricultural constraints: deployment on resource-constrained embedded/mobile edge devices (e.g., Raspberry Pi, Android handheld scanners, drones) requiring minimal model storage, low latency, high throughput, and robust detection of minority disease classes.

---

## 2. Dataset & Authoritative Ground Truth

- **Authoritative Data Source**: Physical ground truth files stored in `Raw_Data/` (3,058 RGB images, $1200 \times 800$ resolution). `Metadata.xlsx` was audited but excluded due to missing crop references and label contradictions.
- **Disease Classes (5)**:
  1. `early_leaf_spot` ($N=885$)
  2. `healthy_leaf` ($N=929$)
  3. `late_leaf_spot` ($N=689$)
  4. `nutrition_deficiency` ($N=329$)
  5. `rust` ($N=226$, minority class)
- **Data Split Standard**: Deterministic, stratified **70/15/15** ratio (Seed: 42).
  - **Train**: 2,140 images ($69.98\%$)
  - **Validation**: 458 images ($14.98\%$)
  - **Test**: 460 images ($15.04\%$)
  - **Data Isolation**: 100% SHA256 hash isolation across splits (zero leakage).

---

## 3. Master Experimental Results Comparison

All models were evaluated on the untouched 460-image test set. GPU latency & throughput were measured on an NVIDIA Tesla T4; CPU metrics were measured via PyTorch dynamic INT8 quantization.

| Model Architecture / Variant | Quantization | Size (MB) | Size Reduction | Test Acc (%) | Weighted F1 (%) | Macro F1 (%) | Rust Recall (%) | GPU FPS | CPU FPS | CPU Latency (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ShuffleNetV2 x1.0 (Baseline)** | FP32 | 4.96 MB | 0.0% | 82.39% | 81.84% | 79.52% | 44.12% | 1,414.11 | 18.25 | 54.79 ms |
| **ShuffleNetV2 x1.0 (Baseline)** | INT8 (CPU) | 4.95 MB | -0.2% | 82.39% | 81.84% | 79.52% | 44.12% | — | 37.02 | 27.01 ms |
| **ShuffleNetV2 + CBAM (Exp 2)** | FP32 | 5.60 MB | +12.9% | **89.57%** | **89.50%** | **91.67%** | **97.06%** | 1,204.49 | 16.92 | 59.10 ms |
| **ShuffleNetV2 + CBAM (Exp 2)** | INT8 (CPU) | 5.59 MB | +12.7% | **89.57%** | **89.50%** | **91.67%** | **97.06%** | — | 36.79 | 27.18 ms |
| **KD Student x0.5 (Exp 3)** | FP32 | 1.46 MB | **-70.6%** | 87.83% | 87.73% | 88.51% | 82.35% | **1,642.90** | 32.40 | 30.86 ms |
| **KD Student x0.5 (Exp 3)** | INT8 (CPU) | **1.44 MB** | **-71.0%** | 87.83% | 87.73% | 88.51% | 82.35% | — | **61.30** | **16.31 ms** |

---

## 4. Key Scientific Findings & Analysis

### A. Impact of Attention Gating (CBAM)
- **Rust Recall Breakthrough**: The FP32 baseline ShuffleNetV2 x1.0 suffered from severe minority class failure, achieving only **44.12% Rust Recall** (missing $19/34$ infected leaves). Integrating CBAM (Channel & Spatial Attention) raised Rust Recall to **97.06%** ($33/34$ detected) while boosting overall Test Accuracy from **82.39%** to **89.57%** (+7.18%).
- **Macro F1 Improvement**: Macro F1 jumped by **+12.15%** (to $91.67\%$), proving attention mechanisms eliminate biased feature representation on imbalanced plant disease datasets.

### B. Efficacy of Knowledge Distillation (KD)
- **Superiority over Baseline**: Distilling the high-capacity CBAM Teacher into an ultra-lightweight **ShuffleNetV2 x0.5 Student** yielded **87.83% Test Accuracy** — outperforming the $4.96\text{ MB}$ FP32 Baseline by **+5.44%** while reducing model memory footprint by **70.6%** ($1.46\text{ MB}$).
- **Preserved Minority Protection**: The KD Student preserved an **82.35% Rust Recall** (nearly double the baseline rate).

### C. INT8 Post-Training Quantization on Edge CPUs
- **Zero Loss in Accuracy**: Dynamic INT8 quantization preserved 100% of floating-point test metrics across all variants.
- **CPU Speedup**: Quantizing linear layers doubled CPU throughput on the KD Student model from **32.40 FPS** ($30.86\text{ ms}$) to **61.30 FPS** ($16.31\text{ ms}$ per image), enabling real-time $60\text{ FPS}$ mobile inference.

---

## 5. Deployment Recommendation Matrix

1. **Maximum Precision Deployment (Cloud / Drone Edge Gateways with GPU)**:
   - **Model**: `ShuffleNetV2 + CBAM (FP32)`
   - **Metrics**: $89.57\%$ Test Accuracy | $97.06\%$ Rust Recall | $1,204.49\text{ GPU FPS}$ ($0.83\text{ ms/img}$)
2. **Ultra-Lightweight Real-Time Edge Deployment (Mobile / Microcontrollers / Raspberry Pi CPU)**:
   - **Model**: `KD Student x0.5 (INT8 Dynamic Quantized)`
   - **Metrics**: $87.83\%$ Test Accuracy | $1.44\text{ MB}$ Storage | $61.30\text{ CPU FPS}$ ($16.31\text{ ms/img}$)
