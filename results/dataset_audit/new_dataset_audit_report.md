# Phyto New Dataset Read-Only Audit Report

## 1. Summary Overview

- **Dataset Path**: `c:/Users/Shwet/Desktop/Groundnut_Leaf_dataset`
- **Total Files in Train**: **7,910**
- **Total Files in Test**: **2,451**
- **Total Files Overall**: **10,361**
- **Train + Test equals Total**: **True**
- **Corrupt / Unreadable Images**: **0**
- **Unique Duplicate Filenames**: **147**
- **Unique Duplicate SHA256 Hashes**: **10**

## 2. Per-Class Image Breakdown

| Class Folder Name | Train Count | Test Count | Total Count | Train % | Test % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `early_leaf_spot_1` | 1,322 | 409 | **1,731** | 16.71% | 16.69% |
| `early_rust_1` | 1,065 | 409 | **1,474** | 13.46% | 16.69% |
| `healthy_leaf_1` | 1,462 | 409 | **1,871** | 18.48% | 16.69% |
| `late_leaf_spot_1` | 1,491 | 405 | **1,896** | 18.85% | 16.52% |
| `nutrition_deficiency_1` | 1,255 | 410 | **1,665** | 15.87% | 16.73% |
| `rust_1` | 1,315 | 409 | **1,724** | 16.62% | 16.69% |
| **Total** | **7,910** | **2,451** | **10,361** | 100.00% | 100.00% |

## 3. File Properties & Integrity

### File Extensions
- `.jpg`: 10,360 files
- ``: 1 files

### Image Color Modes
- `RGB`: 10,361 images

### Image Dimensions Distribution
- `256x256`: 10,361 images

## 4. Class Isolation & Completeness

- **Missing Classes in Train**: `[]`
- **Missing Classes in Test**: `[]`

## 5. Duplicate Analysis

- **Duplicate Filenames Count**: 147
- **Duplicate Image Hashes Count**: 10

## 6. Base Paper Comparison Analysis

Comparing audited counts with statistics reported in the base research paper (Abu Talib et al., ICCSCE 2025 / Mendeley Data V3):

- **Base Paper Reported Training Samples**: 7,910 images
- **Audited New Dataset Train Count**: 7,910 images
- **Audited New Dataset Test Count**: 2,451 images
- **Audited New Dataset Total Count**: 10,361 images

> **Comparison Finding**: The new dataset on disk has **10,361 total images** divided into `train` (7,910) and `test` (2,451) splits with 6 class directories.