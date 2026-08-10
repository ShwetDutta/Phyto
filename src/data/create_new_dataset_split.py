"""
Stratified Validation Split Generator for New 10,361-Image Groundnut Dataset.
Phyto Project - Groundnut Plant Disease Classification (Edge-AI Framework).

Creates a 90/10 train/validation split from the 7,910 original training images,
leaving the 2,451 original test images completely untouched (Seed: 42).
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def create_split(manifest_csv: Path, output_csv: Path, seed: int = 42, val_ratio: float = 0.10) -> pd.DataFrame:
    print(f"Loading manifest CSV from: {manifest_csv.resolve()}")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    assert len(df) == 10361, f"Expected 10,361 total rows, found {len(df)}"

    train_original = df[df["original_split"] == "train"].copy()
    test_original = df[df["original_split"] == "test"].copy()

    assert len(train_original) == 7910, f"Expected 7,910 train rows, found {len(train_original)}"
    assert len(test_original) == 2451, f"Expected 2,451 test rows, found {len(test_original)}"

    # Stratified split of original 7,910 train set into actual train (90%) and validation (10%)
    train_subset, val_subset = train_test_split(
        train_original,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_original["class_label"],
    )

    train_subset["split"] = "train"
    val_subset["split"] = "validation"
    test_original["split"] = "test"

    split_df = pd.concat([train_subset, val_subset, test_original], ignore_index=True)

    # Verification checks
    assert len(split_df) == 10361, f"Total rows changed to {len(split_df)}"
    assert len(split_df[split_df["split"] == "test"]) == 2451, "Test count changed!"
    assert len(split_df[split_df["split"] == "train"]) + len(split_df[split_df["split"] == "validation"]) == 7910, "Train + Val count != 7,910!"

    test_hashes = set(test_original["sha256"])
    train_val_hashes = set(split_df[split_df["split"].isin(["train", "validation"])]["sha256"])
    sha256_overlap = test_hashes.intersection(train_val_hashes)
    assert len(sha256_overlap) == 0, f"Found {len(sha256_overlap)} overlapping hashes between test and train/val!"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)
    print(f"Saved stratified split manifest CSV to: {output_csv.resolve()}")

    print("\n=== Split Summary (Seed: {}) ===".format(seed))
    print(f"Train:      {len(train_subset):,} images ({len(train_subset)/7910*100:.2f}% of orig train)")
    print(f"Validation: {len(val_subset):,} images ({len(val_subset)/7910*100:.2f}% of orig train)")
    print(f"Test:       {len(test_original):,} images (100% untouched original test)")
    print(f"Total:      {len(split_df):,} images")

    print("\n=== Per-Class Breakdown Across Splits ===")
    split_table = pd.crosstab(split_df["class_label"], split_df["split"])[["train", "validation", "test"]]
    split_table["total"] = split_table.sum(axis=1)
    print(split_table.to_string())

    return split_df


def main():
    parser = argparse.ArgumentParser(description="Create Validation Split for New Groundnut Dataset")
    parser.add_argument("--manifest-csv", type=str, default="results/new_dataset_manifest/groundnut_dataset_manifest.csv")
    parser.add_argument("--output-csv", type=str, default="results/new_dataset_manifest/groundnut_dataset_split_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    create_split(Path(args.manifest_csv), Path(args.output_csv), seed=args.seed, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
