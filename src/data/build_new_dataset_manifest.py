"""
Manifest Builder for New 10,361-Image 6-Class Groundnut Leaf Dataset.
Phyto Project - Groundnut Plant Disease Classification (Edge-AI Framework).

Builds results/new_dataset_manifest/groundnut_dataset_manifest.csv and summary documentation.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from PIL import Image

CLASS_TO_IDX: Dict[str, int] = {
    "early_leaf_spot": 0,
    "early_rust": 1,
    "healthy_leaf": 2,
    "late_leaf_spot": 3,
    "nutrition_deficiency": 4,
    "rust": 5,
}

FOLDER_TO_CLASS: Dict[str, str] = {
    "early_leaf_spot_1": "early_leaf_spot",
    "early_rust_1": "early_rust",
    "healthy_leaf_1": "healthy_leaf",
    "late_leaf_spot_1": "late_leaf_spot",
    "nutrition_deficiency_1": "nutrition_deficiency",
    "rust_1": "rust",
    "early_leaf_spot": "early_leaf_spot",
    "early_rust": "early_rust",
    "healthy_leaf": "healthy_leaf",
    "late_leaf_spot": "late_leaf_spot",
    "nutrition_deficiency": "nutrition_deficiency",
    "rust": "rust",
}


def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_manifest(dataset_root: Path, output_dir: Path) -> pd.DataFrame:
    print(f"Building manifest for dataset at: {dataset_root.resolve()}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    splits = {"train": train_dir, "test": test_dir}
    records: List[Dict[str, Any]] = []
    
    hash_tracker: Dict[str, List[str]] = defaultdict(list)

    for split_name, split_path in splits.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            folder_name = class_dir.name
            if folder_name not in FOLDER_TO_CLASS:
                raise ValueError(f"Unknown class folder '{folder_name}' in {split_path}")

            class_label = FOLDER_TO_CLASS[folder_name]
            class_idx = CLASS_TO_IDX[class_label]

            for file_path in sorted(class_dir.iterdir()):
                if not file_path.is_file():
                    continue

                rel_path = str(file_path.relative_to(dataset_root)).replace("\\", "/")
                filename = file_path.name
                file_size = file_path.stat().st_size
                sha256_hash = compute_sha256(file_path)

                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode

                records.append({
                    "image_id": f"{split_name}_{class_label}_{filename}",
                    "relative_path": rel_path,
                    "filename": filename,
                    "folder_name": folder_name,
                    "class_label": class_label,
                    "class_idx": class_idx,
                    "original_split": split_name,
                    "sha256": sha256_hash,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_size_bytes": file_size,
                })
                
                hash_tracker[sha256_hash].append(rel_path)

    df = pd.DataFrame(records)

    # Flag duplicate hashes
    duplicate_hashes = {h for h, paths in hash_tracker.items() if len(paths) > 1}
    df["is_duplicate_hash"] = df["sha256"].isin(duplicate_hashes)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "groundnut_dataset_manifest.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved manifest CSV ({len(df)} rows) to: {csv_path.resolve()}")

    # Summary report
    summary_path = output_dir / "groundnut_dataset_manifest_summary.md"
    summary_md = []
    summary_md.append("# Phyto New Groundnut Dataset Manifest Summary\n")
    summary_md.append(f"- **Total Images**: **{len(df):,}**")
    summary_md.append(f"- **Train Images**: **{len(df[df['original_split'] == 'train']):,}**")
    summary_md.append(f"- **Test Images**: **{len(df[df['original_split'] == 'test']):,}**")
    summary_md.append(f"- **Total Classes**: **{len(df['class_label'].unique())}**")
    summary_md.append(f"- **Flagged Duplicate Hashes**: **{df['is_duplicate_hash'].sum()} images** ({len(duplicate_hashes)} unique hashes)")

    summary_md.append("\n## Class Breakdown Across Original Splits\n")
    summary_md.append("| Class Label | Class Index | Train Count | Test Count | Total Count |")
    summary_md.append("| :--- | :---: | :---: | :---: | :---: |")

    for cls, idx in sorted(CLASS_TO_IDX.items(), key=lambda x: x[1]):
        tr_cnt = len(df[(df["original_split"] == "train") & (df["class_label"] == cls)])
        te_cnt = len(df[(df["original_split"] == "test") & (df["class_label"] == cls)])
        tot_cnt = len(df[df["class_label"] == cls])
        summary_md.append(f"| `{cls}` | {idx} | {tr_cnt:,} | {te_cnt:,} | **{tot_cnt:,}** |")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))
    print(f"Saved manifest summary to: {summary_path.resolve()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build Manifest for New 10,361-Image Groundnut Dataset")
    parser.add_argument("--dataset-root", type=str, default=r"c:\Users\Shwet\Desktop\Groundnut_Leaf_dataset")
    parser.add_argument("--output-dir", type=str, default="results/new_dataset_manifest")
    args = parser.parse_args()

    build_manifest(Path(args.dataset_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
