"""
Dataset Split Generator for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Generates a reproducible 70/15/15 stratified train/validation/test split
from results/dataset_audit/physical_image_manifest.csv without copying,
moving, or modifying original image files.
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


def create_stratified_split(
    input_csv_path: Path,
    seed: int = 42
) -> pd.DataFrame:
    """
    Reads physical_image_manifest.csv and performs a 70/15/15 class-stratified split.
    
    Args:
        input_csv_path: Path to physical_image_manifest.csv
        seed: Fixed random seed for reproducibility
        
    Returns:
        pd.DataFrame containing split manifest with required columns.
    """
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    # Required ground truth column check
    if 'class_label' not in df.columns:
        raise KeyError("Column 'class_label' not found in input manifest.")
    if 'sha256' not in df.columns:
        raise KeyError("Column 'sha256' not found in input manifest.")

    # Sort deterministically by image_id prior to splitting
    df = df.sort_values(by='image_id').reset_index(drop=True)

    # Initialize split column
    df['split'] = ''

    # Perform stratified split per class
    rng = np.random.RandomState(seed)

    for class_name, group in df.groupby('class_label'):
        indices = np.array(group.index.values, copy=True)
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(np.round(0.70 * n_total))
        n_val = int(np.round(0.15 * n_total))
        # Remaining assigned to test to ensure exact total match
        n_test = n_total - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        df.loc[train_idx, 'split'] = 'train'
        df.loc[val_idx, 'split'] = 'validation'
        df.loc[test_idx, 'split'] = 'test'

    # Map column names to required output schema
    column_mapping = {
        'image_mode': 'mode',
        'file_size_bytes': 'file_size'
    }
    df = df.rename(columns=column_mapping)

    required_columns = [
        'image_id',
        'relative_path',
        'filename',
        'class_label',
        'sha256',
        'width',
        'height',
        'mode',
        'file_size',
        'split'
    ]

    # Filter and reorder to exact required columns
    out_df = pd.DataFrame(df[required_columns].copy())
    return out_df


def validate_split_manifest(df: pd.DataFrame) -> bool:
    """
    Independently validates the generated split manifest against all integrity constraints.
    
    Checks:
    1. Total row count is exactly 3,058.
    2. train_count + validation_count + test_count == 3,058.
    3. All required columns exist.
    4. Every image_id occurs exactly once (no nulls, 3,058 unique).
    5. Every relative_path occurs exactly once (no nulls, 3,058 unique).
    6. Every row has exactly one split ('train', 'validation', or 'test').
    7. All 5 classes exist in all 3 splits.
    8. Every class's train + validation + test counts equals its total.
    9. Zero SHA256 overlap across different splits.
    
    Returns:
        bool: True if all validation checks pass. Raises ValueError/KeyError otherwise.
    """
    # 1. Total row count check
    if len(df) != 3058:
        raise ValueError(f"Expected 3058 total rows, got {len(df)}")

    # 2. Required columns check
    expected_cols = [
        'image_id', 'relative_path', 'filename', 'class_label',
        'sha256', 'width', 'height', 'mode', 'file_size', 'split'
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")

    # 3. Split count sum check
    train_count = len(df[df['split'] == 'train'])
    val_count = len(df[df['split'] == 'validation'])
    test_count = len(df[df['split'] == 'test'])
    total_split_sum = train_count + val_count + test_count
    if total_split_sum != 3058:
        raise ValueError(
            f"Sum of train ({train_count}) + validation ({val_count}) + "
            f"test ({test_count}) is {total_split_sum}, expected 3058"
        )

    # 4. Unique and non-null image_id check
    if len(df['image_id']) != 3058 or df['image_id'].isnull().any():
        raise ValueError("image_id column contains nulls or missing entries.")
    if df['image_id'].nunique() != 3058:
        raise ValueError(f"Duplicate image_id detected! Unique count: {df['image_id'].nunique()}")

    # 5. Unique and non-null relative_path check
    if len(df['relative_path']) != 3058 or df['relative_path'].isnull().any():
        raise ValueError("relative_path column contains nulls or missing entries.")
    if df['relative_path'].nunique() != 3058:
        raise ValueError(f"Duplicate relative_path detected! Unique count: {df['relative_path'].nunique()}")

    # 6. Exactly one split per row check
    valid_splits = {'train', 'validation', 'test'}
    if df['split'].isnull().any() or (df['split'].astype(str).str.strip() == '').any():
        raise ValueError("Empty or null split labels detected!")
    actual_splits = set(df['split'].unique())
    invalid_splits = actual_splits - valid_splits
    if invalid_splits:
        raise ValueError(f"Unexpected split labels found: {invalid_splits}")

    # 7. Class coverage and per-class total split sum check
    expected_classes = {
        'early_leaf_spot', 'healthy_leaf', 'late_leaf_spot',
        'nutrition_deficiency', 'rust'
    }

    for s in ['train', 'validation', 'test']:
        split_classes = set(np.unique(df.loc[df['split'] == s, 'class_label']))
        missing_classes = expected_classes - split_classes
        if missing_classes:
            raise ValueError(f"Split '{s}' is missing classes: {missing_classes}")

    for c in expected_classes:
        c_df = df[df['class_label'] == c]
        c_total = len(c_df)
        c_train = len(c_df[c_df['split'] == 'train'])
        c_val = len(c_df[c_df['split'] == 'validation'])
        c_test = len(c_df[c_df['split'] == 'test'])
        c_sum = c_train + c_val + c_test
        if c_sum != c_total:
            raise ValueError(
                f"Class '{c}' split sum mismatch: train ({c_train}) + val ({c_val}) + "
                f"test ({c_test}) = {c_sum} != class total ({c_total})"
            )

    # 8. SHA256 integrity: No hash in more than one split
    train_hashes = set(df[df['split'] == 'train']['sha256'])
    val_hashes = set(df[df['split'] == 'validation']['sha256'])
    test_hashes = set(df[df['split'] == 'test']['sha256'])

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            f"Duplicate SHA256 hashes detected across splits! "
            f"Train/Val overlap: {len(train_val_overlap)}, "
            f"Train/Test overlap: {len(train_test_overlap)}, "
            f"Val/Test overlap: {len(val_test_overlap)}"
        )

    return True


def generate_markdown_summary(df: pd.DataFrame, seed: int = 42) -> str:
    """
    Generates dataset split summary report in Markdown format.
    """
    total_count = len(df)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']

    train_cnt = len(train_df)
    val_cnt = len(val_df)
    test_cnt = len(test_df)

    train_pct = (train_cnt / total_count) * 100
    val_pct = (val_cnt / total_count) * 100
    test_pct = (test_cnt / total_count) * 100

    classes = sorted(df['class_label'].unique())

    summary_lines = [
        "# Phyto Dataset Split Summary Report",
        "",
        "## Overview",
        f"- **Total Physical Images**: {total_count:,}",
        f"- **Random Seed**: {seed}",
        f"- **Stratification Standard**: 70% Train / 15% Validation / 15% Test",
        "",
        "## Overall Split Distribution",
        "",
        "| Split | Image Count | Percentage |",
        "|---|---:|---:|",
        f"| Train | {train_cnt:,} | {train_pct:.2f}% |",
        f"| Validation | {val_cnt:,} | {val_pct:.2f}% |",
        f"| Test | {test_cnt:,} | {test_pct:.2f}% |",
        f"| **Total** | **{total_count:,}** | **100.00%** |",
        "",
        "## Class-Wise Split Distribution",
        "",
        "| Class Label | Total Images | Train Count ( % ) | Validation Count ( % ) | Test Count ( % ) |",
        "|---|---:|---:|---:|---:|"
    ]

    for c in classes:
        c_total = len(df[df['class_label'] == c])
        c_train = len(df[(df['class_label'] == c) & (df['split'] == 'train')])
        c_val = len(df[(df['class_label'] == c) & (df['split'] == 'validation')])
        c_test = len(df[(df['class_label'] == c) & (df['split'] == 'test')])

        c_tr_pct = (c_train / c_total) * 100
        c_va_pct = (c_val / c_total) * 100
        c_te_pct = (c_test / c_total) * 100

        summary_lines.append(
            f"| `{c}` | {c_total:,} | {c_train:,} ({c_tr_pct:.2f}%) | {c_val:,} ({c_va_pct:.2f}%) | {c_test:,} ({c_te_pct:.2f}%) |"
        )

    summary_lines.extend([
        "",
        "## Integrity & Validation Status",
        "- [x] All 5 physical groundnut disease classes exist across all 3 splits.",
        "- [x] Exactly 3,058 physical image records preserved (zero missing, zero duplicated).",
        "- [x] Zero SHA256 hash collision/overlap across splits (100% hash isolation).",
        "- [x] No physical image files were modified, moved, copied, or renamed.",
        ""
    ])

    return "\n".join(summary_lines)


def main():
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent

    input_csv = workspace_root / "results" / "dataset_audit" / "physical_image_manifest.csv"
    output_dir = workspace_root / "results" / "dataset_manifest"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "dataset_split_manifest.csv"
    output_summary_md = output_dir / "dataset_split_summary.md"

    seed = 42
    print(f"Creating stratified dataset split (seed={seed})...")
    split_df = create_stratified_split(input_csv, seed=seed)

    print("Validating dataset split manifest...")
    validate_split_manifest(split_df)
    print("Validation PASSED successfully.")

    print(f"Writing split manifest CSV to: {output_csv}")
    split_df.to_csv(output_csv, index=False)

    print("Generating Markdown summary report...")
    summary_md = generate_markdown_summary(split_df, seed=seed)

    print(f"Writing summary report to: {output_summary_md}")
    with open(output_summary_md, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("\nDataset split completed successfully!")


if __name__ == "__main__":
    main()
