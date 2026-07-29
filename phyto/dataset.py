import os
import glob
from typing import Tuple, Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from phyto.config import Config

class GroundnutDataset(Dataset):
    """
    Custom PyTorch Dataset for Groundnut Plant Leaf Images.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error reading image file {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def get_transforms(image_size: Tuple[int, int] = Config.IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Generates training and evaluation transforms with standardization & augmentation.
    """
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD)
    ])

    return train_transform, eval_transform


def resolve_dataset_directory(data_dir: str) -> str:
    """
    Resolves data directory path. If target folder does not exist locally,
    automatically downloads dataset from Kaggle via kagglehub ('warcoder/groundnut-plant-leaf-data').
    """
    if os.path.exists(data_dir):
        return data_dir

    # Search alternative relative locations
    alternative_local = os.path.join(
        "Dataset of groundnut plant leaf images for classification and detection",
        "Raw_Data"
    )
    if os.path.exists(alternative_local):
        return alternative_local

    # Fallback to automatic kagglehub download
    print(f"[Dataset Engine] Local path '{data_dir}' not found.")
    print("[Dataset Engine] Initiating automatic download via kagglehub ('warcoder/groundnut-plant-leaf-data')...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download("warcoder/groundnut-plant-leaf-data")
        raw_data_path = os.path.join(downloaded_path, "Raw_Data")
        if os.path.exists(raw_data_path):
            return raw_data_path
        return downloaded_path
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset directory not found at '{data_dir}' and kagglehub download failed: {e}"
        )


def load_dataset_filepaths(data_dir: str) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """
    Scans data_dir for class subdirectories and collects all valid image file paths and target labels.
    """
    actual_dir = resolve_dataset_directory(data_dir)

    subdirs = [d for d in os.listdir(actual_dir) if os.path.isdir(os.path.join(actual_dir, d))]
    subdirs.sort()

    class_to_idx = {cls_name: i for i, cls_name in enumerate(subdirs)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(subdirs)}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    labels = []

    for cls_name in subdirs:
        cls_dir = os.path.join(actual_dir, cls_name)
        cls_idx = class_to_idx[cls_name]
        
        for root, _, files in os.walk(cls_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
                    labels.append(cls_idx)

    return image_paths, labels, class_to_idx, idx_to_class


def build_dataloaders(
    data_dir: str = Config.DEFAULT_DATA_DIR,
    batch_size: int = Config.BATCH_SIZE,
    seed: int = Config.SEED,
    num_workers: int = Config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Splits groundnut leaf dataset into 70% Train, 15% Validation, and 15% Test using Stratified Split.
    Returns PyTorch DataLoaders for each split along with class mappings.
    """
    image_paths, labels, class_to_idx, idx_to_class = load_dataset_filepaths(data_dir)

    # 1. Stratified split into Train (70%) and Temp (30%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        stratify=labels,
        random_state=seed
    )

    # 2. Stratified split of Temp into Validation (50% of 30% = 15%) and Test (50% of 30% = 15%)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed
    )

    train_tf, eval_tf = get_transforms()

    train_ds = GroundnutDataset(train_paths, train_labels, transform=train_tf)
    val_ds = GroundnutDataset(val_paths, val_labels, transform=eval_tf)
    test_ds = GroundnutDataset(test_paths, test_labels, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"[Dataset Engine] Loaded {len(image_paths)} images across {len(class_to_idx)} classes.")
    print(f"[Dataset Engine] Stratified Split -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class
