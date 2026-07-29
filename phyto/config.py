import os
import torch

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
