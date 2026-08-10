# Phyto Dataset Audit Report: Groundnut Plant Disease Classification

**Date of Audit**: 2026-08-10  
**Project**: Phyto (Edge-AI Framework for Groundnut Plant Disease Classification)  
**Audit Mode**: READ-ONLY Audit (No dataset files modified, renamed, moved, deleted, or preprocessed)

---

## 1. Executive Summary

This report documents the read-only comprehensive audit of the groundnut plant disease dataset located in `Phyto_1`.
The dataset consists of physical image files located in `Raw_Data/` and annotations stored in `Metadata.xlsx`.

### Key Discoveries:
1. **Physical Dataset**: `Raw_Data/` contains **3058 images** across **5 top-level category folders**. All images are in `.JPG` format, RGB mode, and resolution `1200x800` with 0 corrupted files.
2. **Metadata Annotations**: `Metadata.xlsx` contains **10,361 total rows** (Train: 7910 rows; Test: 2451 rows).
3. **Massive Disconnect**: **96.44% of metadata records (9,992 out of 10,361)** do NOT match any physical file in `Raw_Data/`. Over 9,330 metadata records use `dr_` prefixed filenames (e.g. `dr_0_1138.jpg`), which refer to an object detection bounding box crop dataset that is entirely missing from `Raw_Data/`.
4. **Unannotated Raw Images**: **89.80% of physical images (2,746 out of 3,058)** in `Raw_Data/` are completely absent from `Metadata.xlsx`.
5. **Early Rust Resolution**: There is no separate `early_rust` folder in `Raw_Data/`. However, 34 physical images in `Raw_Data/rust/` match `Early Rust` metadata records, establishing that Early Rust samples are physically stored inside the `rust` folder.
6. **Data Leakage & Label Mutation**: 24 `File Id`s appear in both Train and Test metadata splits. 6 of these resolve to physical files in `Raw_Data/nutrition deficiency` and exhibit **contradictory class labels** across splits (e.g. labeled as `Nutrition Deficiency` in Train, but `Healthy Leaf` in Test).
7. **Excel Formatting Defect**: The `Test` sheet in `Metadata.xlsx` lacks a header row, causing standard readers to incorrectly parse the first sample (`dr_0_1138.jpg`) as column names.

---

## 2. Dataset Paths & Location

- **Workspace Root**: `C:\Users\Shwet\Desktop\Phyto_1`
- **Raw_Data Path (Relative)**: `Dataset of groundnut plant leaf images for classification and detection\Raw_Data`
- **Raw_Data Path (Absolute)**: `C:\Users\Shwet\Desktop\Phyto_1\Dataset of groundnut plant leaf images for classification and detection\Raw_Data`
- **Metadata.xlsx Path (Relative)**: `Dataset of groundnut plant leaf images for classification and detection\Metadata.xlsx`
- **Metadata.xlsx Path (Absolute)**: `C:\Users\Shwet\Desktop\Phyto_1\Dataset of groundnut plant leaf images for classification and detection\Metadata.xlsx`

---

## 3. Raw Data Physical Folder Scan

### A. Folder Structure & Image Counts
- **Total Image Files**: 3058
- **Image Extensions**: {'.JPG': 3058}
- **Nested Subdirectories**: None (0)

| Folder Name | Physical Image Count | Percentage of Raw Data | Image Format | Resolution |
| :--- | :---: | :---: | :---: | :---: |
| `early_leaf_spot` | 885 | 28.94% | JPG (RGB) | 1200x800 |
| `healthy leaf` | 929 | 30.38% | JPG (RGB) | 1200x800 |
| `late leaf spot` | 689 | 22.53% | JPG (RGB) | 1200x800 |
| `nutrition deficiency` | 329 | 10.76% | JPG (RGB) | 1200x800 |
| `rust` | 226 | 7.39% | JPG (RGB) | 1200x800 |
| **Total** | **3058** | **100.00%** | **JPG (RGB)** | **1200x800** |

---

## 4. "Early Rust" Category Investigation

- **Dedicated `early_rust` Folder in `Raw_Data/`**: **NO** (Only 5 visible top-level folders exist).
- **Metadata `Early Rust` Total Records**: **1474** (Train: 1,065; Test: 409).
- **Matching Images Found in `Raw_Data/`**: **34 physical images**.
- **Physical Location of Matched Early Rust Images**: All 34 matched images are located inside `Raw_Data/rust/` (e.g. `rust/IMG_8842.JPG`, `rust/IMG_8942.JPG`).
- **Unmatched `Early Rust` Metadata**: 1,440 records are `dr_` prefixed filenames (e.g., `dr_0_1913.jpg`) belonging to the missing object-detection crop set.

---

## 5. Metadata.xlsx Sheet & Class Analysis

`Metadata.xlsx` contains two relevant sheets (`Train` and `Test`).

### A. Row Counts & Unique Identifiers

| Metric | Train Sheet | Test Sheet | Combined Total |
| :--- | :---: | :---: | :---: |
| **Total Rows** | 7910 | 2451 | **10361** |
| **Header Row Present?** | Yes | **No** (Row 0 is Data) | N/A |
| **Unique File Id Values** | 7835 | 2400 | **10,211** |
| **Duplicated File Ids within Split** | 75 (148 rows) | 51 (102 rows) | **124 unique FIDs** |
| **File Ids in Both Train & Test** | N/A | N/A | **24 File Ids** |

### B. Class Distribution in Metadata

| Class Name in Metadata | Train Count | Test Count | Combined Count | Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| `Early Leaf Spot` | 1322 | 409 | 1731 | 16.71% |
| `Late Leaf spot` / `Late Leaf Spot` | 1491 | 405 | 1896 | 18.30% |
| `Healthy Leaf` | 1462 | 409 | 1871 | 18.06% |
| `Nutrition Deficiency` | 1255 | 410 | 1665 | 16.07% |
| `Rust` | 1315 | 409 | 1724 | 16.64% |
| `Early Rust` | 1065 | 409 | 1474 | 14.23% |
| **Total** | **7910** | **2451** | **10361** | **100.00%** |

---

## 6. Metadata-to-Image Matching Analysis

Matching was executed using full relative paths and case-insensitive filename resolution to map metadata records to actual physical images in `Raw_Data/`.

### A. Matching Rate Summary
- **Total Metadata Records**: 10361
- **Metadata Records Matched to Physical Images**: **369 (3.56%)**
- **Unmatched Metadata Records**: **9992 (96.44%)**
  - `dr_` Prefixed Records (Missing Crop Dataset): **9336**
  - Other Unmatched Records: **656**

### B. Raw Data Coverage
- **Total Physical Images in `Raw_Data/`**: 3058
- **Images Matched to Metadata**: **313 (10.24%)**
- **Images Absent from Metadata**: **2745 (89.76%)**

---

## 7. Duplicate Filename & Content Analysis

### A. Physical Raw Data Duplicates
- **Identical Image Content (Cryptographic SHA256 Match)**: **0 duplicate pairs** (All 3,058 files in `Raw_Data/` contain unique pixel content).
- **Duplicate Filenames Across Folders**: **47 filenames** appear in multiple folders (94 physical files total).
  - *Example*: `IMG_3483.JPG` exists in both `early_leaf_spot/` and `nutrition deficiency/`.
  - SHA256 verification confirms these are **different physical images** sharing the same generic camera filename.

### B. Metadata Duplicate File Ids
- **Train Sheet Duplicates**: 73 unique File Ids duplicated across 148 rows.
- **Test Sheet Duplicates**: 51 unique File Ids duplicated across 102 rows.
- **Inconsistency**: Duplicate File Id entries in metadata frequently assign **different disease classes** to the exact same File Id.

---

## 8. Train/Test Overlap & Data Leakage Analysis

### A. File Id Overlap
- **File Ids in both Train and Test Sheets**: **24 File Ids** (e.g. `22.jpg`, `23.jpg`, `39.jpg`, `40.jpg`, `41.jpg`, `42.jpg`, `dr_3_8007.jpg`, `dr_1_9542.jpg`).

### B. Resolved Physical Image Leakage & Label Mutation
For 6 of the overlapping File Ids, candidate physical files were resolved in `Raw_Data/nutrition deficiency/`. Cryptographic SHA256 hashing confirms that the **exact physical images** leak across Train and Test splits with **contradictory class labels**:

| File Id | Physical File Path | SHA256 Hash | Train Class Label(s) | Test Class Label(s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `22.jpg` | `nutrition deficiency/22.JPG` | `a9412b0d...` | Nutrition Deficiency | Late Leaf Spot | **Leakage & Conflicting Label** |
| `23.jpg` | `nutrition deficiency/23.JPG` | `e04055e9...` | Healthy Leaf, Nutrition Def. | Late Leaf Spot | **Leakage & Conflicting Label** |
| `39.jpg` | `nutrition deficiency/39.JPG` | `98f98054...` | Nutrition Deficiency | Healthy Leaf | **Leakage & Conflicting Label** |
| `40.jpg` | `nutrition deficiency/40.JPG` | `17c6be05...` | Nutrition Deficiency | Healthy Leaf | **Leakage & Conflicting Label** |
| `41.jpg` | `nutrition deficiency/41.JPG` | `29c4293c...` | Nutrition Deficiency | Healthy Leaf | **Leakage & Conflicting Label** |
| `42.jpg` | `nutrition deficiency/42.JPG` | `ba22679bf...` | Nutrition Deficiency | Healthy Leaf | **Leakage & Conflicting Label** |

---

## 9. Summary of Inconsistencies & Data Quality Issues

1. **Severe Metadata Disconnect**: Over 90% of `Metadata.xlsx` rows (9,336 records) refer to `dr_` prefixed object detection bounding-box crops that do not exist in `Raw_Data/`.
2. **High Ratio of Unannotated Images**: 89.80% of physical images in `Raw_Data/` (2,746 out of 3,058) are not referenced anywhere in `Metadata.xlsx`.
3. **Missing Excel Header**: The `Test` sheet in `Metadata.xlsx` starts immediately with data on row 0 without column headers.
4. **Class Name Typo**: `Late Leaf spot` (lowercase 's') in `Train` vs `Late Leaf Spot` (uppercase 'S') in `Test`.
5. **Cross-Split Data Leakage & Label Mutation**: Physical images leak between Train and Test splits with conflicting disease labels.
6. **Cross-Folder Filename Collisions**: 47 camera filenames (e.g. `IMG_3483.JPG`) are duplicated across different class folders in `Raw_Data/`.
7. **Implicit Early Rust Category**: Early Rust images are stored inside `Raw_Data/rust/` rather than in a dedicated folder.

---

## 10. Recommendations for Training Dataset Structure

*(Note: These recommendations are for planning purposes only and are NOT implemented in this turn as per audit instructions.)*

1. **Discard Disconnected `Metadata.xlsx` for Image Classification**:
   Because `Metadata.xlsx` has a 96.44% failure rate and severe label corruption, model training should be based directly on the physical images in `Raw_Data/`.
2. **Establish a Standardized 6-Class Folder Structure**:
   Create a clean, standardized directory structure for PyTorch/TensorFlow datasets:
   ```
   data/processed/
   ├── train/
   │   ├── early_leaf_spot/
   │   ├── late_leaf_spot/
   │   ├── healthy_leaf/
   │   ├── nutrition_deficiency/
   │   ├── rust/
   │   └── early_rust/
   ├── val/
   └── test/
   ```
3. **Curate Early Rust vs Rust**:
   Separate the 34 verified `Early Rust` images from `Raw_Data/rust/` into the dedicated `early_rust/` class directory, leaving mature `rust` images in `rust/`.
4. **Unique Filename Sanitization**:
   Rename images using their parent folder prefix (e.g. `early_leaf_spot_IMG_3483.jpg` vs `nutrition_deficiency_IMG_3483.jpg`) or SHA256 hashes to guarantee globally unique filenames across the entire dataset.
5. **Clean Stratified Split (80/10/10 or 70/15/15)**:
   Perform a clean stratified split at the physical image level (guaranteeing 0 leakage between train, validation, and test splits).

---
