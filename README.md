# Phyto: Edge-AI Groundnut Leaf Disease Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Phyto** is an Edge-AI deep learning framework engineered for lightweight, real-time classification of groundnut (*Arachis hypogaea*) leaf diseases. Designed specifically for deployment on resource-constrained embedded systems, drones, and mobile devices, the framework integrates **Convolutional Block Attention Modules (CBAM)**, **Knowledge Distillation (KD)**, and **INT8 Dynamic Quantization** to achieve high accuracy and throughput with minimal memory footprint.

---

## Key Experimental Results

Evaluated on an untouched 460-image stratified test set ($70/15/15$ split, seed 42) across 5 physical classes (`early_leaf_spot`, `healthy_leaf`, `late_leaf_spot`, `nutrition_deficiency`, `rust`):

| Model Architecture | Variant / Mode | Model Size | Test Acc | Weighted F1 | Macro F1 | Rust Recall | GPU FPS (Tesla T4) | CPU FPS | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ShuffleNetV2 x1.0** | FP32 Baseline | 4.96 MB | 82.39% | 81.84% | 79.52% | 44.12% | 1,414 FPS | 18.3 FPS | 54.79 ms |
| **ShuffleNetV2 + CBAM** | FP32 Teacher | 5.60 MB | **89.57%** | **89.50%** | **91.67%** | **97.06%** | 1,204 FPS | 16.9 FPS | 59.10 ms |
| **ShuffleNetV2 x0.5** | KD Student (FP32) | 1.46 MB | 87.83% | 87.73% | 88.51% | 82.35% | **1,643 FPS** | 32.4 FPS | 30.86 ms |
| **ShuffleNetV2 x0.5** | **KD Student (INT8)** | **1.44 MB** | **87.83%** | **87.73%** | **88.51%** | **82.35%** | — | **61.3 FPS** | **16.31 ms** |

### Research Breakthrough Highlights

- **Overcoming Minority Class Failure**: The baseline model missed **55.88% of rust disease infections** ($44.12\%$ recall). Integrating **CBAM Attention** increased Rust Recall to **97.06%** ($+52.94\%$ gain) while raising overall test accuracy to **89.57%**.
- **Distillation Outperforms Un-distilled Baseline**: Knowledge Distillation into a **ShuffleNetV2 x0.5 Student** achieved **87.83% test accuracy** ($+5.44\%$ higher than the 1.0x baseline) while reducing parameter size by **70.6%** ($1.44\text{ MB}$).
- **Real-Time Edge CPU Execution**: INT8 dynamic quantization enables **61.3 FPS** ($16.31\text{ ms/image}$) processing on CPU.

---

## Repository Structure

```
Phyto/
├── src/
│   ├── data/
│   │   ├── dataset_audit.py            # Physical image audit & integrity check
│   │   ├── build_physical_manifest.py  # Image manifest builder
│   │   ├── create_dataset_split.py     # Deterministic 70/15/15 stratified split generator
│   │   └── phyto_dataset.py            # PyTorch Dataset loader & path resolver
│   ├── models/
│   │   ├── cbam.py                     # Channel & Spatial Attention Module (CBAM)
│   │   └── shufflenet_v2.py            # Baseline, CBAM, and x0.5 ShuffleNetV2 factories
│   ├── training/
│   │   ├── train.py                    # Training loop with Colab-resumable checkpointing
│   │   └── distillation.py             # Knowledge Distillation loss & training pipeline
│   ├── evaluation/
│   │   └── metrics.py                  # Evaluation metrics & latency benchmark engine
│   └── quantization/
│       └── quantize.py                 # INT8 dynamic quantization & size benchmarking
├── results/
│   ├── dataset_audit/                  # Audit outputs & physical image manifest
│   ├── dataset_manifest/               # Stratified dataset split manifest & summary
│   └── final_experiment_report.md      # Comprehensive experimental benchmark report
└── requirements.txt                    # Dependencies
```

---

## Quick Start & Usage

### 1. Installation
```bash
git clone https://github.com/ShwetDutta/Phyto.git
cd Phyto
pip install -r requirements.txt
```

### 2. Loading Dataset
```python
import pandas as pd
from src.data.phyto_dataset import PhytoDataset, load_split_manifest

manifest_df = load_split_manifest("results/dataset_manifest/dataset_split_manifest.csv")
train_df = manifest_df[manifest_df["split"] == "train"]

dataset = PhytoDataset(
    manifest_df=train_df,
    raw_data_root="path/to/Raw_Data",
    transform=None
)
```

### 3. Model Training & Knowledge Distillation
```python
from src.models import create_shufflenet_v2_cbam, create_shufflenet_v2_x0_5
from src.training import train_model, train_distillation_model

# Train CBAM Teacher Model
teacher_model = create_shufflenet_v2_cbam(num_classes=5, pretrained=True)
train_model(teacher_model, train_loader, val_loader, criterion, optimizer, epochs=15)

# Distill into x0.5 Student Model
student_model = create_shufflenet_v2_x0_5(num_classes=5, pretrained=True)
train_distillation_model(student_model, teacher_model, train_loader, val_loader, optimizer, epochs=15)
```

### 4. INT8 Dynamic Quantization
```python
from src.quantization import quantize_model_dynamic, get_model_size_mb

quantized_student = quantize_model_dynamic(student_model)
size_mb = get_model_size_mb(quantized_student)
print(f"Quantized Model Size: {size_mb} MB")
```

---

## Citation & License
This project is licensed under the [MIT License](LICENSE).
