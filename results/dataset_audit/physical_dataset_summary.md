# Phyto Physical Dataset Summary & Image Manifest Audit

**Date of Manifest Generation**: 2026-08-10  
**Audit Purpose**: Ground-truth physical dataset verification (READ-ONLY)  
**Manifest CSV**: [`physical_image_manifest.csv`](file:///C:/Users/Shwet/Desktop/Phyto_1/results/dataset_audit/physical_image_manifest.csv)  

---

## 1. Physical Dataset Overview

- **Total Physical Images**: **3058**
- **Manifest Row Count**: **3058** (Exactly 1 row per physical image)
- **Unique SHA256 Hashes**: **3058**
- **Duplicate Content Images (SHA256 collisions)**: **0**
- **Corrupt Image Files**: **0**
- **Single Parent Folder Compliance**: **100%** (Every image has exactly 1 parent folder)

---

## 2. Image Distribution by Physical Class

| Physical Folder Name | Standard Class Label | Image Count | Percentage (%) |
| :--- | :--- | :---: | :---: |
| `early_leaf_spot` | `early_leaf_spot` | 885 | 28.94% |
| `healthy leaf` | `healthy_leaf` | 929 | 30.38% |
| `late leaf spot` | `late_leaf_spot` | 689 | 22.53% |
| `nutrition deficiency` | `nutrition_deficiency` | 329 | 10.76% |
| `rust` | `rust` | 226 | 7.39% |
| **Total** | **5 Physical Classes** | **3058** | **100.00%** |

---

## 3. Physical Image Technical Properties

- **Image Formats / Extensions**: {'.JPG': 3058}
- **Image Modes**: {'RGB': 3058}
- **Image Dimensions (Width x Height)**:
  - `1200 x 800`: **3058 images** (100.00%)
- **File Size Range**: Min = 135,440 bytes, Max = 806,018 bytes, Mean = 264,790 bytes

---

## 4. Cross-Folder Filename Collision Analysis

- **Duplicate Filenames Across Folders**: **47 filenames** (occurring across 94 physical images total).
- **SHA256 Hash Verification**: **CONFIRMED** - All 47 duplicate filename pairs possess **distinct SHA256 hashes**. They are different physical images sharing identical generic camera filenames across class folders.

### Sample Duplicate Filename Collisions:

| Filename | Occurrences | Folders | SHA256 Hash Status |
| :--- | :---: | :--- | :--- |
| `img_3483.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3484.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3485.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3486.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3487.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3488.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3489.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3490.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3491.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |
| `img_3492.jpg` | 2 | `early_leaf_spot, nutrition deficiency` | Distinct SHA256 (2/2) |

---

## 5. Verification Checklist & Compliance

1. **Manifest Coverage**: Exactly 3,058 rows corresponding to 3,058 physical files.
2. **Independent Image ID**: Each row assigned a unique ID (`PHYTO_RAW_0001` through `PHYTO_RAW_3058`) independent of filename.
3. **No Metadata Interference**: Classes assigned strictly from `Raw_Data/` parent directory names.
4. **Zero Modifying Operations**: No files renamed, moved, deleted, preprocessed, or split.
5. **No Early Rust Alterations**: No `early_rust` class created; no sub-categorization applied.

---
