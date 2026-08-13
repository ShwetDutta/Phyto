"""
PyTorch Dataset Loader for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Provides the PhytoDataset class and helper functions to load images
from dataset split manifests using PyTorch.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_TO_IDX: Dict[str, int] = {
    'early_leaf_spot': 0,
    'early_rust': 1,
    'healthy_leaf': 2,
    'late_leaf_spot': 3,
    'nutrition_deficiency': 4,
    'rust': 5,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

CLASS_TO_FOLDER: Dict[str, str] = {
    'early_leaf_spot': 'early_leaf_spot_1',
    'early_rust': 'early_rust_1',
    'healthy_leaf': 'healthy_leaf_1',
    'late_leaf_spot': 'late_leaf_spot_1',
    'nutrition_deficiency': 'nutrition_deficiency_1',
    'rust': 'rust_1',
    # Old dataset folder fallbacks
    'healthy_leaf_old': 'healthy leaf',
    'late_leaf_spot_old': 'late leaf spot',
    'nutrition_deficiency_old': 'nutrition deficiency',
}


def get_class_names() -> List[str]:
    """Returns list of class labels ordered by integer index (0..5)."""
    return [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]


def get_class_to_idx() -> Dict[str, int]:
    """Returns mapping from class label string to integer class index."""
    return CLASS_TO_IDX.copy()


def load_split_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads dataset split manifest CSV and optionally filters by split.

    Args:
        manifest_path: Path to dataset split manifest CSV
        split: Optional split name to filter ('train', 'validation', 'test')

    Returns:
        pd.DataFrame containing split manifest entries.
    """
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_p.resolve()}")

    df = pd.read_csv(manifest_p)

    required_cols = {'filename', 'class_label', 'split'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing required columns: {missing}")

    if split is not None:
        valid_splits = {'train', 'validation', 'test'}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
        filtered_df = df[df['split'] == split]
        assert isinstance(filtered_df, pd.DataFrame)
        df = filtered_df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No records found for split '{split}' in manifest {manifest_p}")

    return df


def resolve_image_path(
    raw_data_root: Path,
    row: pd.Series
) -> Path:
    """
    Resolves physical image file path with multi-level fallbacks for diverse folder structures.
    """
    filename = str(row['filename'])
    class_label = str(row['class_label'])

    # 1. Primary path check: relative_path as-is
    if 'relative_path' in row and pd.notna(row['relative_path']):
        rel_p = str(row['relative_path'])
        candidate = raw_data_root / rel_p
        if candidate.exists():
            return candidate

        # 1b. Try stripping leading split folder (e.g. 'train/' or 'test/')
        rel_parts = Path(rel_p).parts
        if len(rel_parts) > 1:
            candidate = raw_data_root / Path(*rel_parts[1:])
            if candidate.exists():
                return candidate

        # 1c. Try nested directory (e.g. raw_data_root / raw_data_root.name / relative_path)
        candidate = raw_data_root / raw_data_root.name / rel_p
        if candidate.exists():
            return candidate

    # 2. Try folder_name column + filename
    if 'folder_name' in row and pd.notna(row['folder_name']):
        candidate = raw_data_root / str(row['folder_name']) / filename
        if candidate.exists():
            return candidate

    # 3. Try CLASS_TO_FOLDER mapping + filename
    folder_name = CLASS_TO_FOLDER.get(class_label, class_label)
    candidate = raw_data_root / folder_name / filename
    if candidate.exists():
        return candidate

    # 4. Try legacy folder names (with spaces) + filename
    old_folder = CLASS_TO_FOLDER.get(f"{class_label}_old", folder_name)
    candidate = raw_data_root / old_folder / filename
    if candidate.exists():
        return candidate

    # 5. Try searching immediate subdirectories in raw_data_root
    if raw_data_root.exists():
        try:
            for sub in raw_data_root.iterdir():
                if sub.is_dir():
                    if 'relative_path' in row and pd.notna(row['relative_path']):
                        cand = sub / str(row['relative_path'])
                        if cand.exists():
                            return cand
                    if 'folder_name' in row and pd.notna(row['folder_name']):
                        cand = sub / str(row['folder_name']) / filename
                        if cand.exists():
                            return cand
                    cand = sub / folder_name / filename
                    if cand.exists():
                        return cand
        except Exception:
            pass

    # 6. Default return path for error reporting
    if 'relative_path' in row and pd.notna(row['relative_path']):
        return raw_data_root / str(row['relative_path'])
    return raw_data_root / folder_name / filename


def verify_manifest_paths(
    manifest_df: pd.DataFrame,
    raw_data_root: Union[str, Path]
) -> bool:
    """
    Verifies that every row in the manifest resolves to an existing physical image file.
    """
    root_p = Path(raw_data_root)

    for idx, row in manifest_df.iterrows():
        img_path = resolve_image_path(root_p, row)
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image file not found for row {idx} (`{row.get('class_label')}` / `{row.get('filename')}`): {img_path.resolve()}"
            )

    return True


class PhytoDataset(Dataset[Tuple[Any, int]]):
    """
    PyTorch Dataset for Phyto groundnut leaf image classification.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        raw_data_root: Union[str, Path],
        transform: Optional[Callable[[Image.Image], Any]] = None
    ) -> None:
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.raw_data_root = Path(raw_data_root)
        self.transform = transform

        # Determine class_to_idx mapping dynamically or use standard 6-class mapping
        manifest_classes = self.manifest_df['class_label'].unique()
        self.class_to_idx = CLASS_TO_IDX.copy()
        
        for cls_label in manifest_classes:
            if str(cls_label) not in self.class_to_idx:
                raise ValueError(
                    f"Unknown class label '{cls_label}' in manifest. "
                    f"Expected one of: {list(self.class_to_idx.keys())}"
                )

    def __len__(self) -> int:
        return len(self.manifest_df)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        row = self.manifest_df.iloc[index]
        img_path = resolve_image_path(self.raw_data_root, row)

        if not img_path.exists():
            root_exists = self.raw_data_root.exists()
            subdirs = [p.name for p in self.raw_data_root.iterdir() if p.is_dir()] if root_exists else []
            raise FileNotFoundError(
                f"\n=======================================================\n"
                f"PHYTO DATASET PATH RESOLUTION ERROR\n"
                f"=======================================================\n"
                f"Image file not found: {img_path.resolve()}\n"
                f"Target Filename: {row.get('filename')}\n"
                f"Raw Data Root Passed: {self.raw_data_root.resolve()} (Exists: {root_exists})\n"
                f"Directories found in Raw Data Root: {subdirs}\n\n"
                f"DIAGNOSIS & SOLUTIONS:\n"
                f"1. NEW 10,361-IMAGE DATASET:\n"
                f"   If you intend to run the NEW 10,361 dataset manifest:\n"
                f"   Ensure --raw-data-root points to the directory containing 'train' and 'test' folders.\n"
                f"   (Example: --raw-data-root '/content/drive/MyDrive/Datasets for phyto/raw_data_new/Groundnut_Leaf_dataset')\n\n"
                f"2. ORIGINAL 3,060-IMAGE DATASET:\n"
                f"   If your local/Colab folder at '/content/Groundnut_Leaf_dataset' is the original 3,060 dataset:\n"
                f"   Update --manifest-path to: 'results/dataset_manifest/dataset_split_manifest.csv'\n"
                f"=======================================================\n"
            )

        image = Image.open(img_path).convert('RGB')
        
        if 'class_idx' in row and pd.notna(row['class_idx']):
            label_idx = int(row['class_idx'])
        else:
            label_idx = self.class_to_idx[str(row['class_label'])]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx
