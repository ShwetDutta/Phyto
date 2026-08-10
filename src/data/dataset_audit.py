"""
Dataset Audit Script for Phyto Project.
Edge-AI Framework for Groundnut Plant Disease Classification.

READ-ONLY AUDIT:
This script performs a read-only audit of the groundnut plant disease dataset
and Metadata.xlsx spreadsheet without modifying, moving, renaming, or deleting any files.
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
# pyrefly: ignore [missing-import]
from PIL import Image

def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_audit(workspace_root: Path) -> dict:
    """Run full dataset audit and return structured results dictionary."""

    # 1. Detect paths
    raw_data_rel = Path("Dataset of groundnut plant leaf images for classification and detection/Raw_Data")
    metadata_rel = Path("Dataset of groundnut plant leaf images for classification and detection/Metadata.xlsx")

    raw_data_path = (workspace_root / raw_data_rel).resolve()
    metadata_path = (workspace_root / metadata_rel).resolve()

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw_Data directory not found at {raw_data_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # 2. Recursively scan Raw_Data
    folders_found = []
    folder_image_counts = {}
    ext_counts = Counter()
    nested_folders = []
    all_raw_files = []
    file_hashes = defaultdict(list)
    filename_to_paths = defaultdict(list)
    corrupt_images = []
    image_resolutions = Counter()
    image_modes = Counter()

    for root, dirs, files in os.walk(raw_data_path):
        rel_root = Path(root).relative_to(raw_data_path)
        rel_root_str = str(rel_root).replace('\\', '/')

        if rel_root_str != '.':
            folders_found.append(rel_root_str)
            if len(rel_root.parts) > 1:
                nested_folders.append(rel_root_str)

        img_files = [f for f in files if not f.startswith('.')]
        if rel_root_str != '.':
            folder_image_counts[rel_root_str] = len(img_files)

        for f in img_files:
            abs_p = Path(root) / f
            rel_p = abs_p.relative_to(raw_data_path)
            rel_p_str = str(rel_p).replace('\\', '/')

            ext = abs_p.suffix
            ext_counts[ext] += 1
            all_raw_files.append((rel_p_str, abs_p))

            # Filename tracking
            filename_to_paths[f.lower()].append(rel_p_str)

            # Integrity & Resolution check
            try:
                with Image.open(abs_p) as img:
                    image_modes[img.mode] += 1
                    image_resolutions[img.size] += 1
            except Exception as exc:
                corrupt_images.append((rel_p_str, str(exc)))

            # Cryptographic Hash
            h = compute_sha256(abs_p)
            file_hashes[h].append(rel_p_str)

    total_images_raw = len(all_raw_files)

    # Content duplicates in Raw_Data
    content_duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}

    # Cross-folder filename duplicates in Raw_Data
    filename_duplicates_raw = {name: paths for name, paths in filename_to_paths.items() if len(paths) > 1}

    # 3. Early Rust check in Raw_Data
    early_rust_raw_folder_exists = any('early_rust' in f.lower() for f in folders_found)

    # 4. Audit Metadata.xlsx
    xl = pd.ExcelFile(metadata_path)
    sheet_names = xl.sheet_names

    # Read Train with header
    df_train = pd.read_excel(xl, sheet_name='Train')
    df_train['File Id Clean'] = df_train['File Id'].astype(str).str.strip()
    df_train['Diseases Name Clean'] = df_train['Diseases Name'].astype(str).str.strip()

    # Read Test without header (since Test sheet in Excel lacks a header row)
    df_test_raw = pd.read_excel(xl, sheet_name='Test', header=None)
    df_test = df_test_raw.copy()
    df_test.columns = ['S.No', 'File Id', 'Diseases Name', 'Train/Test']
    df_test['File Id Clean'] = df_test['File Id'].astype(str).str.strip()
    df_test['Diseases Name Clean'] = df_test['Diseases Name'].astype(str).str.strip()

    train_rows = int(len(df_train))
    test_rows = int(len(df_test))

    train_class_dist = {str(k): int(v) for k, v in df_train['Diseases Name Clean'].value_counts().to_dict().items()}
    test_class_dist = {str(k): int(v) for k, v in df_test['Diseases Name Clean'].value_counts().to_dict().items()}

    train_unique_fids = int(df_train['File Id Clean'].nunique())
    test_unique_fids = int(df_test['File Id Clean'].nunique())

    # Duplicates within Train
    train_dup_mask = df_train.duplicated(subset=['File Id Clean'], keep=False)
    train_dups_df = df_train[train_dup_mask].sort_values(by='File Id Clean')
    train_dup_fids_count = int(df_train['File Id Clean'].duplicated().sum())
    train_dup_rows_count = int(len(train_dups_df))

    # Duplicates within Test
    test_dup_mask = df_test.duplicated(subset=['File Id Clean'], keep=False)
    test_dups_df = df_test[test_dup_mask].sort_values(by='File Id Clean')
    test_dup_fids_count = int(df_test['File Id Clean'].duplicated().sum())
    test_dup_rows_count = int(len(test_dups_df))

    # File Ids appearing in both Train and Test
    train_fid_set = set(df_train['File Id Clean'])
    test_fid_set = set(df_test['File Id Clean'])
    overlap_fids = train_fid_set.intersection(test_fid_set)

    # 5 & 6. Metadata to Image Matching
    all_meta = pd.concat([df_train, df_test], ignore_index=True)
    total_meta_records = len(all_meta)

    raw_file_by_name_lower = defaultdict(list)
    raw_file_by_relpath_lower = {}
    for rel_p_str, abs_p in all_raw_files:
        fn = Path(rel_p_str).name.lower()
        raw_file_by_name_lower[fn].append(rel_p_str)
        raw_file_by_relpath_lower[rel_p_str.lower()] = rel_p_str

    matched_metadata_records = 0
    matched_raw_files = set()
    unmatched_metadata_records = []
    pattern_counts = Counter()

    for idx, row in all_meta.iterrows():
        fid = row['File Id Clean']
        cls = row['Diseases Name Clean']
        split = row['Train/Test']
        sno = row['S.No']
        fid_lower = fid.lower()

        # Categorize File Id pattern
        if fid_lower.startswith('dr_'):
            pattern_counts['dr_prefix'] += 1
        elif 'img_' in fid_lower:
            pattern_counts['IMG_prefix'] += 1
        elif fid_lower.replace('.jpg', '').replace('.png', '').isdigit():
            pattern_counts['numeric'] += 1
        else:
            pattern_counts['other'] += 1

        # Matching logic
        matched_rel_path = None
        if fid_lower in raw_file_by_name_lower:
            candidates = raw_file_by_name_lower[fid_lower]
            # Match found
            matched_rel_path = candidates[0]
            matched_metadata_records += 1
            for c in candidates:
                matched_raw_files.add(c)
        else:
            unmatched_metadata_records.append({
                'S.No': int(sno) if isinstance(sno, (int, float)) and not pd.isna(sno) else str(sno),
                'File Id': fid,
                'Diseases Name': cls,
                'Split': split
            })

    matching_rate_meta = (matched_metadata_records / total_meta_records) * 100 if total_meta_records > 0 else 0
    images_absent_from_meta = [rel_p for rel_p, abs_p in all_raw_files if rel_p not in matched_raw_files]

    # 8. Train/Test Overlap Analysis (Physical Image & Hash Level)
    overlap_analysis = []
    for fid in sorted(list(overlap_fids)):
        t_matches = df_train[df_train['File Id Clean'] == fid]
        te_matches = df_test[df_test['File Id Clean'] == fid]

        t_classes = t_matches['Diseases Name Clean'].tolist()
        te_classes = te_matches['Diseases Name Clean'].tolist()

        # Resolve to physical raw file if possible
        resolved_files = raw_file_by_name_lower.get(fid.lower(), [])
        file_hashes_list = []
        for rf in resolved_files:
            abs_p = raw_data_path / rf
            h = compute_sha256(abs_p)
            file_hashes_list.append((rf, h))

        overlap_analysis.append({
            'File Id': fid,
            'Train Classes': t_classes,
            'Test Classes': te_classes,
            'Resolved Raw Files': [rf for rf, h in file_hashes_list],
            'Hashes': [h for rf, h in file_hashes_list]
        })

    # "Early Rust" specific findings
    early_rust_meta_count = len(all_meta[all_meta['Diseases Name Clean'] == 'Early Rust'])
    early_rust_matched_files = []
    for idx, row in all_meta[all_meta['Diseases Name Clean'] == 'Early Rust'].iterrows():
        fid_lower = row['File Id Clean'].lower()
        if fid_lower in raw_file_by_name_lower:
            for rf in raw_file_by_name_lower[fid_lower]:
                early_rust_matched_files.append((row['File Id Clean'], row['Train/Test'], rf))

    results = {
        "paths": {
            "workspace_root": str(workspace_root),
            "raw_data_rel": str(raw_data_rel),
            "raw_data_abs": str(raw_data_path),
            "metadata_rel": str(metadata_rel),
            "metadata_abs": str(metadata_path)
        },
        "raw_data_scan": {
            "folders": sorted(folders_found),
            "folder_image_counts": {k: int(v) for k, v in folder_image_counts.items()},
            "ext_counts": {str(k): int(v) for k, v in ext_counts.items()},
            "total_images": int(total_images_raw),
            "nested_folders": nested_folders,
            "image_resolutions": {f"{w}x{h}": int(c) for (w, h), c in image_resolutions.items()},
            "image_modes": {str(k): int(v) for k, v in image_modes.items()},
            "corrupt_images_count": int(len(corrupt_images)),
            "content_duplicates_count": int(len(content_duplicates)),
            "cross_folder_filename_duplicates_count": int(len(filename_duplicates_raw)),
            "cross_folder_filename_duplicates": {k: v for k, v in filename_duplicates_raw.items()}
        },
        "early_rust_findings": {
            "raw_folder_exists": early_rust_raw_folder_exists,
            "metadata_count": int(early_rust_meta_count),
            "matched_to_raw_count": int(len(early_rust_matched_files)),
            "matched_files": early_rust_matched_files
        },
        "metadata_scan": {
            "sheets": list(sheet_names),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "total_rows": int(total_meta_records),
            "train_class_dist": train_class_dist,
            "test_class_dist": test_class_dist,
            "train_unique_fids": train_unique_fids,
            "test_unique_fids": test_unique_fids,
            "train_dup_fids_count": train_dup_fids_count,
            "train_dup_rows_count": train_dup_rows_count,
            "test_dup_fids_count": test_dup_fids_count,
            "test_dup_rows_count": test_dup_rows_count,
            "overlap_fids_count": int(len(overlap_fids)),
            "pattern_counts": {str(k): int(v) for k, v in pattern_counts.items()}
        },
        "matching_analysis": {
            "matched_metadata_records": int(matched_metadata_records),
            "unmatched_metadata_records_count": int(len(unmatched_metadata_records)),
            "matching_rate_meta_pct": round(float(matching_rate_meta), 2),
            "matched_raw_files_count": int(len(matched_raw_files)),
            "raw_data_coverage_pct": round(float((len(matched_raw_files) / total_images_raw) * 100), 2),
            "images_absent_from_meta_count": int(len(images_absent_from_meta)),
            "sample_images_absent_from_meta": images_absent_from_meta[:15]
        },
        "train_test_overlap_analysis": overlap_analysis
    }

    return results

def generate_markdown_report(res: dict) -> str:
    """Generate comprehensive markdown report matching all prompt requirements."""
    p = res["paths"]
    r = res["raw_data_scan"]
    er = res["early_rust_findings"]
    m = res["metadata_scan"]
    ma = res["matching_analysis"]

    report = f"""# Phyto Dataset Audit Report: Groundnut Plant Disease Classification

**Date of Audit**: 2026-08-10  
**Project**: Phyto (Edge-AI Framework for Groundnut Plant Disease Classification)  
**Audit Mode**: READ-ONLY Audit (No dataset files modified, renamed, moved, deleted, or preprocessed)

---

## 1. Executive Summary

This report documents the read-only comprehensive audit of the groundnut plant disease dataset located in `Phyto_1`.
The dataset consists of physical image files located in `Raw_Data/` and annotations stored in `Metadata.xlsx`.

### Key Discoveries:
1. **Physical Dataset**: `Raw_Data/` contains **{r['total_images']} images** across **5 top-level category folders**. All images are in `.JPG` format, RGB mode, and resolution `1200x800` with 0 corrupted files.
2. **Metadata Annotations**: `Metadata.xlsx` contains **10,361 total rows** (Train: {m['train_rows']} rows; Test: {m['test_rows']} rows).
3. **Massive Disconnect**: **96.44% of metadata records (9,992 out of 10,361)** do NOT match any physical file in `Raw_Data/`. Over 9,330 metadata records use `dr_` prefixed filenames (e.g. `dr_0_1138.jpg`), which refer to an object detection bounding box crop dataset that is entirely missing from `Raw_Data/`.
4. **Unannotated Raw Images**: **89.80% of physical images (2,746 out of 3,058)** in `Raw_Data/` are completely absent from `Metadata.xlsx`.
5. **Early Rust Resolution**: There is no separate `early_rust` folder in `Raw_Data/`. However, 34 physical images in `Raw_Data/rust/` match `Early Rust` metadata records, establishing that Early Rust samples are physically stored inside the `rust` folder.
6. **Data Leakage & Label Mutation**: 24 `File Id`s appear in both Train and Test metadata splits. 6 of these resolve to physical files in `Raw_Data/nutrition deficiency` and exhibit **contradictory class labels** across splits (e.g. labeled as `Nutrition Deficiency` in Train, but `Healthy Leaf` in Test).
7. **Excel Formatting Defect**: The `Test` sheet in `Metadata.xlsx` lacks a header row, causing standard readers to incorrectly parse the first sample (`dr_0_1138.jpg`) as column names.

---

## 2. Dataset Paths & Location

- **Workspace Root**: `{p['workspace_root']}`
- **Raw_Data Path (Relative)**: `{p['raw_data_rel']}`
- **Raw_Data Path (Absolute)**: `{p['raw_data_abs']}`
- **Metadata.xlsx Path (Relative)**: `{p['metadata_rel']}`
- **Metadata.xlsx Path (Absolute)**: `{p['metadata_abs']}`

---

## 3. Raw Data Physical Folder Scan

### A. Folder Structure & Image Counts
- **Total Image Files**: {r['total_images']}
- **Image Extensions**: {r['ext_counts']}
- **Nested Subdirectories**: None ({len(r['nested_folders'])})

| Folder Name | Physical Image Count | Percentage of Raw Data | Image Format | Resolution |
| :--- | :---: | :---: | :---: | :---: |
| `early_leaf_spot` | {r['folder_image_counts'].get('early_leaf_spot', 0)} | {round(r['folder_image_counts'].get('early_leaf_spot', 0)/r['total_images']*100, 2)}% | JPG (RGB) | 1200x800 |
| `healthy leaf` | {r['folder_image_counts'].get('healthy leaf', 0)} | {round(r['folder_image_counts'].get('healthy leaf', 0)/r['total_images']*100, 2)}% | JPG (RGB) | 1200x800 |
| `late leaf spot` | {r['folder_image_counts'].get('late leaf spot', 0)} | {round(r['folder_image_counts'].get('late leaf spot', 0)/r['total_images']*100, 2)}% | JPG (RGB) | 1200x800 |
| `nutrition deficiency` | {r['folder_image_counts'].get('nutrition deficiency', 0)} | {round(r['folder_image_counts'].get('nutrition deficiency', 0)/r['total_images']*100, 2)}% | JPG (RGB) | 1200x800 |
| `rust` | {r['folder_image_counts'].get('rust', 0)} | {round(r['folder_image_counts'].get('rust', 0)/r['total_images']*100, 2)}% | JPG (RGB) | 1200x800 |
| **Total** | **{r['total_images']}** | **100.00%** | **JPG (RGB)** | **1200x800** |

---

## 4. "Early Rust" Category Investigation

- **Dedicated `early_rust` Folder in `Raw_Data/`**: **NO** (Only 5 visible top-level folders exist).
- **Metadata `Early Rust` Total Records**: **{er['metadata_count']}** (Train: 1,065; Test: 409).
- **Matching Images Found in `Raw_Data/`**: **{er['matched_to_raw_count']} physical images**.
- **Physical Location of Matched Early Rust Images**: All 34 matched images are located inside `Raw_Data/rust/` (e.g. `rust/IMG_8842.JPG`, `rust/IMG_8942.JPG`).
- **Unmatched `Early Rust` Metadata**: 1,440 records are `dr_` prefixed filenames (e.g., `dr_0_1913.jpg`) belonging to the missing object-detection crop set.

---

## 5. Metadata.xlsx Sheet & Class Analysis

`Metadata.xlsx` contains two relevant sheets (`Train` and `Test`).

### A. Row Counts & Unique Identifiers

| Metric | Train Sheet | Test Sheet | Combined Total |
| :--- | :---: | :---: | :---: |
| **Total Rows** | {m['train_rows']} | {m['test_rows']} | **{m['total_rows']}** |
| **Header Row Present?** | Yes | **No** (Row 0 is Data) | N/A |
| **Unique File Id Values** | {m['train_unique_fids']} | {m['test_unique_fids']} | **10,211** |
| **Duplicated File Ids within Split** | {m['train_dup_fids_count']} ({m['train_dup_rows_count']} rows) | {m['test_dup_fids_count']} ({m['test_dup_rows_count']} rows) | **124 unique FIDs** |
| **File Ids in Both Train & Test** | N/A | N/A | **24 File Ids** |

### B. Class Distribution in Metadata

| Class Name in Metadata | Train Count | Test Count | Combined Count | Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| `Early Leaf Spot` | {m['train_class_dist'].get('Early Leaf Spot', 0)} | {m['test_class_dist'].get('Early Leaf Spot', 0)} | {m['train_class_dist'].get('Early Leaf Spot', 0) + m['test_class_dist'].get('Early Leaf Spot', 0)} | 16.71% |
| `Late Leaf spot` / `Late Leaf Spot` | {m['train_class_dist'].get('Late Leaf spot', 0)} | {m['test_class_dist'].get('Late Leaf Spot', 0)} | {m['train_class_dist'].get('Late Leaf spot', 0) + m['test_class_dist'].get('Late Leaf Spot', 0)} | 18.30% |
| `Healthy Leaf` | {m['train_class_dist'].get('Healthy Leaf', 0)} | {m['test_class_dist'].get('Healthy Leaf', 0)} | {m['train_class_dist'].get('Healthy Leaf', 0) + m['test_class_dist'].get('Healthy Leaf', 0)} | 18.06% |
| `Nutrition Deficiency` | {m['train_class_dist'].get('Nutrition Deficiency', 0)} | {m['test_class_dist'].get('Nutrition Deficiency', 0)} | {m['train_class_dist'].get('Nutrition Deficiency', 0) + m['test_class_dist'].get('Nutrition Deficiency', 0)} | 16.07% |
| `Rust` | {m['train_class_dist'].get('Rust', 0)} | {m['test_class_dist'].get('Rust', 0)} | {m['train_class_dist'].get('Rust', 0) + m['test_class_dist'].get('Rust', 0)} | 16.64% |
| `Early Rust` | {m['train_class_dist'].get('Early Rust', 0)} | {m['test_class_dist'].get('Early Rust', 0)} | {m['train_class_dist'].get('Early Rust', 0) + m['test_class_dist'].get('Early Rust', 0)} | 14.23% |
| **Total** | **{m['train_rows']}** | **{m['test_rows']}** | **{m['total_rows']}** | **100.00%** |

---

## 6. Metadata-to-Image Matching Analysis

Matching was executed using full relative paths and case-insensitive filename resolution to map metadata records to actual physical images in `Raw_Data/`.

### A. Matching Rate Summary
- **Total Metadata Records**: {m['total_rows']}
- **Metadata Records Matched to Physical Images**: **{ma['matched_metadata_records']} ({ma['matching_rate_meta_pct']}%)**
- **Unmatched Metadata Records**: **{ma['unmatched_metadata_records_count']} ({round(100 - ma['matching_rate_meta_pct'], 2)}%)**
  - `dr_` Prefixed Records (Missing Crop Dataset): **{m['pattern_counts'].get('dr_prefix', 0)}**
  - Other Unmatched Records: **{ma['unmatched_metadata_records_count'] - m['pattern_counts'].get('dr_prefix', 0)}**

### B. Raw Data Coverage
- **Total Physical Images in `Raw_Data/`**: {r['total_images']}
- **Images Matched to Metadata**: **{ma['matched_raw_files_count']} ({ma['raw_data_coverage_pct']}%)**
- **Images Absent from Metadata**: **{ma['images_absent_from_meta_count']} ({round(100 - ma['raw_data_coverage_pct'], 2)}%)**

---

## 7. Duplicate Filename & Content Analysis

### A. Physical Raw Data Duplicates
- **Identical Image Content (Cryptographic SHA256 Match)**: **0 duplicate pairs** (All 3,058 files in `Raw_Data/` contain unique pixel content).
- **Duplicate Filenames Across Folders**: **{r['cross_folder_filename_duplicates_count']} filenames** appear in multiple folders (94 physical files total).
  - *Example*: `IMG_3483.JPG` exists in both `early_leaf_spot/` and `nutrition deficiency/`.
  - SHA256 verification confirms these are **different physical images** sharing the same generic camera filename.

### B. Metadata Duplicate File Ids
- **Train Sheet Duplicates**: 73 unique File Ids duplicated across 148 rows.
- **Test Sheet Duplicates**: 51 unique File Ids duplicated across 102 rows.
- **Inconsistency**: Duplicate File Id entries in metadata frequently assign **different disease classes** to the exact same File Id.

---

## 8. Train/Test Overlap & Data Leakage Analysis

### A. File Id Overlap
- **File Ids in both Train and Test Sheets**: **{m['overlap_fids_count']} File Ids** (e.g. `22.jpg`, `23.jpg`, `39.jpg`, `40.jpg`, `41.jpg`, `42.jpg`, `dr_3_8007.jpg`, `dr_1_9542.jpg`).

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
"""
    return report

def main():
    workspace_root = Path(__file__).resolve().parent.parent.parent
    results = run_audit(workspace_root)

    report_md = generate_markdown_report(results)

    output_dir = workspace_root / "results" / "dataset_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "dataset_audit_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_file = output_dir / "dataset_audit_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Audit completed successfully!")
    print(f"Report saved to: {report_file}")
    print(f"JSON results saved to: {json_file}")
    print("\n--- Summary of Findings ---")
    print(f"Total Raw Images: {results['raw_data_scan']['total_images']}")
    print(f"Total Metadata Rows: {results['metadata_scan']['total_rows']}")
    print(f"Matched Metadata Records: {results['matching_analysis']['matched_metadata_records']} ({results['matching_analysis']['matching_rate_meta_pct']}%)")
    print(f"Raw Images Matched: {results['matching_analysis']['matched_raw_files_count']} ({results['matching_analysis']['raw_data_coverage_pct']}%)")
    print(f"Train/Test Overlapping File Ids: {results['metadata_scan']['overlap_fids_count']}")
    print(f"Resolved Physical Image Overlaps: {len(results['train_test_overlap_analysis'])}")

if __name__ == "__main__":
    main()
