"""
PyTorch Dataset Loader for Phyto Project.
Groundnut Plant Disease Classification (Edge-AI Framework).

Provides the PhytoDataset class and helper functions to load images
from dataset_split_manifest.csv using PyTorch.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_TO_IDX: Dict[str, int] = {
    'early_leaf_spot': 0,
    'healthy_leaf': 1,
    'late_leaf_spot': 2,
    'nutrition_deficiency': 3,
    'rust': 4,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

CLASS_TO_FOLDER: Dict[str, str] = {
    'early_leaf_spot': 'early_leaf_spot',
    'healthy_leaf': 'healthy leaf',
    'late_leaf_spot': 'late leaf spot',
    'nutrition_deficiency': 'nutrition deficiency',
    'rust': 'rust',
}


def get_class_names() -> List[str]:
    """Returns list of class labels ordered by integer index (0..4)."""
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
        manifest_path: Path to dataset_split_manifest.csv
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


def verify_manifest_paths(
    manifest_df: pd.DataFrame,
    raw_data_root: Union[str, Path]
) -> bool:
    """
    Verifies that every row in the manifest resolves to an existing physical image file.

    Args:
        manifest_df: DataFrame loaded from dataset_split_manifest.csv
        raw_data_root: Directory containing raw data folders

    Returns:
        bool: True if all images exist. Raises FileNotFoundError or ValueError otherwise.
    """
    root_p = Path(raw_data_root)

    for idx, row in manifest_df.iterrows():
        class_label = str(row['class_label'])
        filename = str(row['filename'])

        if class_label not in CLASS_TO_FOLDER:
            raise ValueError(f"Unknown class label '{class_label}' at row {idx}")

        actual_folder = CLASS_TO_FOLDER[class_label]
        img_path = root_p / actual_folder / filename

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image file not found for row {idx} (`{class_label}` / `{filename}`): {img_path.resolve()}"
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
        """
        Args:
            manifest_df: DataFrame containing image metadata ('filename', 'class_label', etc.)
            raw_data_root: Base directory containing raw dataset class folders
            transform: Optional image transformation/augmentation pipeline
        """
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.raw_data_root = Path(raw_data_root)
        self.transform = transform

        # Validate class labels in manifest
        for cls_label in self.manifest_df['class_label'].unique():
            if str(cls_label) not in CLASS_TO_IDX:
                raise ValueError(
                    f"Unknown class label '{cls_label}' in manifest. "
                    f"Expected one of: {list(CLASS_TO_IDX.keys())}"
                )

    def __len__(self) -> int:
        return len(self.manifest_df)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Loads image at index, applies transformations, and returns (image, label_idx).
        """
        row = self.manifest_df.iloc[index]
        class_label = str(row['class_label'])
        filename = str(row['filename'])

        if class_label not in CLASS_TO_FOLDER:
            raise ValueError(f"Unknown class label '{class_label}' at index {index}")

        folder_name = CLASS_TO_FOLDER[class_label]
        img_path = self.raw_data_root / folder_name / filename

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at path: {img_path.resolve()}")

        image = Image.open(img_path).convert('RGB')
        label_idx = CLASS_TO_IDX[class_label]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx
