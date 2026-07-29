import unittest
import os
import torch
from phyto.config import Config
from phyto.dataset import build_dataloaders, load_dataset_filepaths

class TestPhytoDataset(unittest.TestCase):

    def test_dataset_filepaths(self):
        data_dir = Config.DEFAULT_DATA_DIR
        self.assertTrue(os.path.exists(data_dir), f"Dataset directory {data_dir} does not exist")

        paths, labels, class_to_idx, idx_to_class = load_dataset_filepaths(data_dir)
        self.assertGreater(len(paths), 0, "No image paths found in dataset")
        self.assertEqual(len(class_to_idx), 5, f"Expected 5 classes, found {len(class_to_idx)}")
        self.assertEqual(len(paths), len(labels), "Length mismatch between image paths and labels")

    def test_dataloaders_split(self):
        data_dir = Config.DEFAULT_DATA_DIR
        train_loader, val_loader, test_loader, class_to_idx, _ = build_dataloaders(
            data_dir=data_dir, batch_size=4, num_workers=0
        )
        
        train_len = len(train_loader.dataset)
        val_len = len(val_loader.dataset)
        test_len = len(test_loader.dataset)
        total = train_len + val_len + test_len

        self.assertGreater(train_len, 0)
        self.assertGreater(val_len, 0)
        self.assertGreater(test_len, 0)

        # Check sample batch shape
        inputs, targets = next(iter(train_loader))
        self.assertEqual(inputs.shape, (4, 3, 224, 224), "Train sample batch image shape mismatch")
        self.assertEqual(targets.shape, (4,), "Train sample batch target shape mismatch")

if __name__ == "__main__":
    unittest.main()
