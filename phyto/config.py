import os
import torch
import shutil

class Config:
    """
    Central Configuration Repository for Project Phyto
    """
    # Seeds & Reproducibility
    SEED = 42

    # Dataset Paths
    DEFAULT_DATA_DIR = os.path.join(
        "Dataset of groundnut plant leaf images for classification and detection",
        "Raw_Data"
    )
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # Class Definitions for Groundnut Leaf Dataset
    CLASS_NAMES = [
        "early_leaf_spot",
        "healthy leaf",
        "late leaf spot",
        "nutrition deficiency",
        "rust"
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    # Image Preprocessing & Augmentation
    IMAGE_SIZE = (224, 224)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # Data Split Ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_EPOCHS_TEACHER = 15
    NUM_EPOCHS_STUDENT = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Knowledge Distillation Parameters
    KD_TEMPERATURE = 4.0
    KD_ALPHA = 0.7  # Weight for Soft Loss vs Hard Loss: alpha*soft + (1-alpha)*hard

    # Hardware Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """Creates output directories for checkpoints and benchmarking results."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

    @classmethod
    def sync_to_gdrive(cls, gdrive_folder="Phyto_Checkpoints"):
        """
        Automatically backs up checkpoints and results to Google Drive if mounted.
        """
        gdrive_base = "/content/drive/MyDrive"
        if os.path.exists(gdrive_base):
            target_ckpt = os.path.join(gdrive_base, gdrive_folder, "checkpoints")
            target_res = os.path.join(gdrive_base, gdrive_folder, "results")
            os.makedirs(target_ckpt, exist_ok=True)
            os.makedirs(target_res, exist_ok=True)

            if os.path.exists(cls.CHECKPOINT_DIR):
                for f in os.listdir(cls.CHECKPOINT_DIR):
                    src = os.path.join(cls.CHECKPOINT_DIR, f)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(target_ckpt, f))

            if os.path.exists(cls.RESULTS_DIR):
                for f in os.listdir(cls.RESULTS_DIR):
                    src = os.path.join(cls.RESULTS_DIR, f)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(target_res, f))

            print(f"[Google Drive Backup] Successfully saved checkpoints & results to: '{gdrive_base}/{gdrive_folder}'")

    @classmethod
    def restore_from_gdrive(cls, gdrive_folder="Phyto_Checkpoints"):
        """
        Restores checkpoints and results from Google Drive to local session workspace.
        """
        gdrive_base = "/content/drive/MyDrive"
        source_ckpt = os.path.join(gdrive_base, gdrive_folder, "checkpoints")
        source_res = os.path.join(gdrive_base, gdrive_folder, "results")

        cls.setup_directories()
        restored_count = 0

        if os.path.exists(source_ckpt):
            for f in os.listdir(source_ckpt):
                src = os.path.join(source_ckpt, f)
                dst = os.path.join(cls.CHECKPOINT_DIR, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    restored_count += 1

        if os.path.exists(source_res):
            for f in os.listdir(source_res):
                src = os.path.join(source_res, f)
                dst = os.path.join(cls.RESULTS_DIR, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        if restored_count > 0:
            print(f"[Google Drive Restore] Restored {restored_count} trained checkpoint(s) from '{source_ckpt}'")
            return True
        return False
