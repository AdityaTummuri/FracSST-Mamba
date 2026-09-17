"""Unit tests for UoGRadarAdapter, data schema compliance, and split integrity.

Ensures zero ML/CUDA dependencies (pure NumPy & Python standard library).
"""

import json
import unittest
from pathlib import Path
import numpy as np

from data.schema import RadarSample
from data.adapters.uog_adapter import UoGRadarAdapter


class TestUoGRadarAdapter(unittest.TestCase):
    """Test suite for the UoG dataset adapter and schema contracts."""

    @classmethod
    def setUpClass(cls):
        cls.raw_dir = Path("data/raw/Dataset_848")
        cls.adapter = UoGRadarAdapter(root_dir=cls.raw_dir)
        cls.discovered_files = cls.adapter.discover_samples()

    def test_discover_samples(self):
        """Verify discovery finds the expected 583 target files."""
        self.assertGreater(len(self.discovered_files), 0, "No radar files discovered.")
        self.assertEqual(
            len(self.discovered_files),
            583,
            f"Expected 583 files, found {len(self.discovered_files)}",
        )

    def test_filename_metadata_parsing(self):
        """Verify regex extraction of activity, subject ID, and repetitions."""
        meta_walk = self.adapter.parse_filename_metadata("1P36A01R01.dat")
        self.assertEqual(meta_walk["activity_label"], "Walk")
        self.assertEqual(meta_walk["subject_id"], "P36")
        self.assertEqual(meta_walk["repetition"], "R01")

        meta_sit = self.adapter.parse_filename_metadata("2P36A02R02.dat")
        self.assertEqual(meta_sit["activity_label"], "Sit")
        self.assertEqual(meta_sit["subject_id"], "P36")

        meta_fall = self.adapter.parse_filename_metadata("6P36A06R03.dat")
        self.assertEqual(meta_fall["activity_label"], "Fall")
        self.assertEqual(meta_fall["subject_id"], "P36")

    def test_load_sample_contract(self):
        """Verify load_sample returns a valid RadarSample instance."""
        if not self.discovered_files:
            self.skipTest("No files available for testing.")

        sample = self.adapter.load_sample(self.discovered_files[0])
        self.assertIsInstance(sample, RadarSample)

        # Shape and type checks
        self.assertTrue(sample.is_complex, "Radar signal must be complex (I + jQ).")
        self.assertEqual(sample.signal.ndim, 1, "Signal array must be 1-dimensional.")
        self.assertEqual(sample.sampling_rate, 1000.0, "Sampling rate must be 1000.0 Hz.")
        self.assertIn(sample.activity_label, {"Walk", "Sit", "Fall"})
        self.assertIsNotNone(sample.subject_id)
        self.assertGreater(sample.num_samples, 0)
        self.assertGreater(sample.duration_seconds, 0.0)

    def test_split_manifest_leakage_safety(self):
        """Verify that train, validation, and test splits have ZERO subject overlap."""
        split_path = Path("data/metadata/train_val_test_split.json")
        self.assertTrue(split_path.exists(), "Split manifest does not exist.")

        with open(split_path, "r") as f:
            manifest = json.load(f)

        summary = manifest["summary"]
        train_subs = set(summary["train"]["subjects"])
        val_subs = set(summary["validation"]["subjects"])
        test_subs = set(summary["test"]["subjects"])

        # Disjoint checks
        self.assertTrue(
            train_subs.isdisjoint(val_subs),
            "Leakage detected: Train and Val share subjects!",
        )
        self.assertTrue(
            train_subs.isdisjoint(test_subs),
            "Leakage detected: Train and Test share subjects!",
        )
        self.assertTrue(
            val_subs.isdisjoint(test_subs),
            "Leakage detected: Val and Test share subjects!",
        )

        # File count validation
        total_split_files = (
            len(manifest["train_files"])
            + len(manifest["val_files"])
            + len(manifest["test_files"])
        )
        self.assertEqual(
            total_split_files,
            583,
            f"Split manifest file count mismatch: {total_split_files} != 583",
        )

    def test_real_data_mamba_pipeline(self):
        """Verify real-data path: RadarMambaDataset (T, F) -> DataLoader (B, T, F) -> MambaClassifier (B, 3)."""
        import torch
        from data.dataset import RadarMambaDataset, create_dataloaders
        from models.mamba_block import MambaClassifier

        processed_dir = Path("data/processed")
        if not processed_dir.exists() or len(list(processed_dir.glob("*.pt"))) == 0:
            self.skipTest("No preprocessed .pt files found in data/processed")

        # 1. Single sample contract: (T, F), label
        dataset = RadarMambaDataset(data_dir=processed_dir)
        spec, label = dataset[0]
        self.assertEqual(spec.ndim, 2, f"Expected (T, F), got {spec.shape}")
        self.assertEqual(spec.shape[1], 256, f"Expected F=256, got {spec.shape[1]}")
        self.assertIn(spec.shape[0], (79, 157), f"Expected T in (79, 157), got {spec.shape[0]}")
        self.assertIsInstance(label, torch.Tensor)

        # 2. DataLoader batch contract: (B, T, F), (B,)
        train_loader, _ = create_dataloaders(data_dir=processed_dir, batch_size=4, seed=42)
        batch_specs, batch_labels = next(iter(train_loader))
        self.assertEqual(batch_specs.ndim, 3)
        self.assertEqual(batch_specs.shape[0], 4)
        self.assertEqual(batch_specs.shape[2], 256)
        self.assertEqual(batch_labels.shape[0], 4)

        # 3. MambaClassifier contract: (B, T, F) -> (B, 3)
        classifier = MambaClassifier(in_features=256, d_model=64, n_layers=2, num_classes=3)
        logits = classifier(batch_specs)
        self.assertEqual(logits.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()

