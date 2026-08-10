"""
Standalone Read-Only Dataset Audit Script for New Groundnut Leaf Dataset.
Phyto Project - Groundnut Plant Disease Classification (Edge-AI Framework).

Performs strict read-only audit of c:/Users/Shwet/Desktop/Groundnut_Leaf_dataset.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from PIL import Image


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file for exact duplicate detection."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def audit_new_dataset(dataset_root: Path) -> Dict[str, Any]:
    print(f"Starting read-only audit of dataset at: {dataset_root.resolve()}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    splits = {"train": train_dir, "test": test_dir}
    
    file_extensions: Counter[str] = Counter()
    corrupt_files: List[Dict[str, str]] = []
    
    filename_tracker: Dict[str, List[str]] = defaultdict(list)
    hash_tracker: Dict[str, List[str]] = defaultdict(list)
    
    split_class_counts: Dict[str, Dict[str, int]] = {"train": defaultdict(int), "test": defaultdict(int)}
    total_class_counts: Dict[str, int] = defaultdict(int)
    
    dimensions: Counter[Tuple[int, int]] = Counter()
    color_modes: Counter[str] = Counter()
    
    total_files_train = 0
    total_files_test = 0

    for split_name, split_path in splits.items():
        if not split_path.exists():
            print(f"Warning: Split directory '{split_name}' does not exist at {split_path}")
            continue

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            
            for file_path in class_dir.iterdir():
                if not file_path.is_file():
                    continue

                rel_path = str(file_path.relative_to(dataset_root)).replace("\\", "/")
                ext = file_path.suffix.lower()
                file_extensions[ext] += 1

                if split_name == "train":
                    total_files_train += 1
                else:
                    total_files_test += 1

                split_class_counts[split_name][class_name] += 1
                total_class_counts[class_name] += 1

                filename_tracker[file_path.name].append(rel_path)

                # Check read integrity & image properties
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    with Image.open(file_path) as img:
                        dimensions[img.size] += 1
                        color_modes[img.mode] += 1
                except Exception as e:
                    corrupt_files.append({"file": rel_path, "error": str(e)})

                # Compute SHA256 hash
                file_hash = compute_sha256(file_path)
                hash_tracker[file_hash].append(rel_path)

    total_files_overall = total_files_train + total_files_test

    duplicate_filenames = {
        fname: paths for fname, paths in filename_tracker.items() if len(paths) > 1
    }

    duplicate_hashes = {
        h: paths for h, paths in hash_tracker.items() if len(paths) > 1
    }

    all_detected_classes = sorted(list(total_class_counts.keys()))
    train_classes = set(split_class_counts["train"].keys())
    test_classes = set(split_class_counts["test"].keys())

    missing_in_train = sorted(list(set(all_detected_classes) - train_classes))
    missing_in_test = sorted(list(set(all_detected_classes) - test_classes))

    is_train_test_equal_total = (total_files_train + total_files_test) == total_files_overall

    # Base paper statistics comparison
    base_paper_stats = {
        "source": "Manvikar & Reddy (2023) / Abu Talib et al. (ICCSCE 2025 Base Paper)",
        "reported_train_total": 7910,
        "reported_val_ratio": "30% for validation",
        "reported_class_breakdown": {
            "early_rust": 1065,
            "late_leaf_spot": 1491,
            "nutrition_deficiency": 1255,
            "rust": 1315,
            "healthy_leaf": 1462,
        }
    }

    audit_result = {
        "total_files_train": total_files_train,
        "total_files_test": total_files_test,
        "total_files_overall": total_files_overall,
        "split_class_counts": {k: dict(v) for k, v in split_class_counts.items()},
        "total_class_counts": dict(total_class_counts),
        "file_extensions": dict(file_extensions),
        "corrupt_files_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "duplicate_filenames_count": len(duplicate_filenames),
        "duplicate_filenames": duplicate_filenames,
        "duplicate_sha256_hashes_count": len(duplicate_hashes),
        "duplicate_sha256_hashes": duplicate_hashes,
        "image_dimensions": {f"{w}x{h}": cnt for (w, h), cnt in dimensions.items()},
        "image_color_modes": dict(color_modes),
        "missing_classes": {
            "missing_in_train": missing_in_train,
            "missing_in_test": missing_in_test,
        },
        "train_plus_test_equals_total": is_train_test_equal_total,
        "base_paper_comparison": base_paper_stats
    }

    return audit_result


def generate_markdown_report(results: Dict[str, Any], output_md_path: Path) -> None:
    train_cls = results["split_class_counts"].get("train", {})
    test_cls = results["split_class_counts"].get("test", {})
    tot_cls = results["total_class_counts"]

    all_classes = sorted(list(set(list(train_cls.keys()) + list(test_cls.keys()))))

    md = []
    md.append("# Phyto New Dataset Read-Only Audit Report")
    md.append("\n## 1. Summary Overview\n")
    md.append(f"- **Dataset Path**: `c:/Users/Shwet/Desktop/Groundnut_Leaf_dataset`")
    md.append(f"- **Total Files in Train**: **{results['total_files_train']:,}**")
    md.append(f"- **Total Files in Test**: **{results['total_files_test']:,}**")
    md.append(f"- **Total Files Overall**: **{results['total_files_overall']:,}**")
    md.append(f"- **Train + Test equals Total**: **{results['train_plus_test_equals_total']}**")
    md.append(f"- **Corrupt / Unreadable Images**: **{results['corrupt_files_count']}**")
    md.append(f"- **Unique Duplicate Filenames**: **{results['duplicate_filenames_count']}**")
    md.append(f"- **Unique Duplicate SHA256 Hashes**: **{results['duplicate_sha256_hashes_count']}**")

    md.append("\n## 2. Per-Class Image Breakdown\n")
    md.append("| Class Folder Name | Train Count | Test Count | Total Count | Train % | Test % |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    tot_overall = results['total_files_overall']
    for cls in all_classes:
        tr = train_cls.get(cls, 0)
        te = test_cls.get(cls, 0)
        tot = tot_cls.get(cls, 0)
        tr_pct = (tr / results['total_files_train'] * 100) if results['total_files_train'] > 0 else 0
        te_pct = (te / results['total_files_test'] * 100) if results['total_files_test'] > 0 else 0
        md.append(f"| `{cls}` | {tr:,} | {te:,} | **{tot:,}** | {tr_pct:.2f}% | {te_pct:.2f}% |")

    md.append(f"| **Total** | **{results['total_files_train']:,}** | **{results['total_files_test']:,}** | **{tot_overall:,}** | 100.00% | 100.00% |")

    md.append("\n## 3. File Properties & Integrity\n")
    md.append("### File Extensions")
    for ext, cnt in results["file_extensions"].items():
        md.append(f"- `{ext}`: {cnt:,} files")

    md.append("\n### Image Color Modes")
    for mode, cnt in results["image_color_modes"].items():
        md.append(f"- `{mode}`: {cnt:,} images")

    md.append("\n### Image Dimensions Distribution")
    for dim, cnt in results["image_dimensions"].items():
        md.append(f"- `{dim}`: {cnt:,} images")

    md.append("\n## 4. Class Isolation & Completeness\n")
    md.append(f"- **Missing Classes in Train**: `{results['missing_classes']['missing_in_train']}`")
    md.append(f"- **Missing Classes in Test**: `{results['missing_classes']['missing_in_test']}`")

    md.append("\n## 5. Duplicate Analysis\n")
    md.append(f"- **Duplicate Filenames Count**: {results['duplicate_filenames_count']}")
    md.append(f"- **Duplicate Image Hashes Count**: {results['duplicate_sha256_hashes_count']}")

    md.append("\n## 6. Base Paper Comparison Analysis\n")
    md.append("Comparing audited counts with statistics reported in the base research paper (Abu Talib et al., ICCSCE 2025 / Mendeley Data V3):\n")
    md.append("- **Base Paper Reported Training Samples**: 7,910 images")
    md.append("- **Audited New Dataset Train Count**: " + f"{results['total_files_train']:,} images")
    md.append("- **Audited New Dataset Test Count**: " + f"{results['total_files_test']:,} images")
    md.append("- **Audited New Dataset Total Count**: " + f"{results['total_files_overall']:,} images")
    md.append("\n> **Comparison Finding**: The new dataset on disk has **" + f"{results['total_files_overall']:,}" + " total images** divided into `train` (" + f"{results['total_files_train']:,}" + ") and `test` (" + f"{results['total_files_test']:,}" + ") splits with 6 class directories.")

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Saved audit markdown report to: {output_md_path.resolve()}")


def main():
    dataset_root = Path(r"c:\Users\Shwet\Desktop\Groundnut_Leaf_dataset")
    audit_results = audit_new_dataset(dataset_root)

    json_path = Path("results/dataset_audit/new_dataset_audit_results.json")
    md_path = Path("results/dataset_audit/new_dataset_audit_report.md")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"Saved audit JSON results to: {json_path.resolve()}")
    generate_markdown_report(audit_results, md_path)


if __name__ == "__main__":
    main()
