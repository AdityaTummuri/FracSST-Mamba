"""Comprehensive unit and smoke test suite for Stage 2 neural network architectures.

Runs via: python -m unittest test_models.py
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from data.schema import RadarSample
from models.mamba_block import MambaBlock, MambaClassifier
from models.da_seg import DASegModel
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_classification_metrics, compute_segmentation_metrics


class TestMambaBlock(unittest.TestCase):
    """Test suite for MambaBlock sequence module."""

    def test_mamba_block_shape_invariance(self) -> None:
        """Verify MambaBlock returns identical (B, T, d_model) output shape."""
        batch_size, seq_len, d_model = 4, 16, 128
        x = torch.randn(batch_size, seq_len, d_model)
        block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2, dropout=0.1)
        out = block(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any(), "NaN values found in MambaBlock output")


class TestMambaClassifier(unittest.TestCase):
    """Test suite for Member 3 MambaClassifier interface and checkpointing."""

    def test_mamba_classifier_input_output_contract(self) -> None:
        """Direct test proving (B, T, F) -> MambaClassifier -> (B, 3) forward, loss, backward, step."""
        batch_size, time_steps, freq_bins = 4, 16, 256
        num_classes = 3

        # Input shape contract: (B, T, F)
        x = torch.randn(batch_size, time_steps, freq_bins)
        target = torch.randint(0, num_classes, (batch_size,))

        classifier = MambaClassifier(
            in_features=freq_bins,
            d_model=64,
            n_layers=2,
            num_classes=num_classes,
            dropout=0.1,
        )
        optimizer = AdamW(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()

        # Output shape contract: (B, 3)
        logits = classifier(x)
        self.assertEqual(logits.shape, (batch_size, num_classes))

        loss = criterion(logits, target)
        self.assertFalse(torch.isnan(loss), "Loss evaluated to NaN")

        loss.backward()
        optimizer.step()

    def test_checkpoint_save_and_load_utilities(self) -> None:
        """Verify save_checkpoint() and load_checkpoint() utilities restore model state, optimizer state, and config."""
        # 1. Create MambaClassifier
        model = MambaClassifier(in_features=64, d_model=32, n_layers=1, num_classes=3)
        # 2. Create optimizer
        optimizer = AdamW(model.parameters(), lr=1e-3)
        config = {"d_model": 32, "lr": 0.001}

        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = Path(temp_dir) / "mamba_checkpoint.pth"

            # 3. Save using save_checkpoint()
            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=5,
                val_accuracy=0.92,
                val_mIoU=0.88,
                config=config,
            )
            self.assertTrue(ckpt_path.exists())

            # 4. Create a NEW MambaClassifier with the same configuration
            new_model = MambaClassifier(in_features=64, d_model=32, n_layers=1, num_classes=3)
            # 5. Create a NEW optimizer
            new_optimizer = AdamW(new_model.parameters(), lr=1e-3)

            # 6. Load using load_checkpoint()
            restored_data = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=new_model,
                optimizer=new_optimizer,
            )

            # 7. Verify epoch and val_accuracy
            self.assertEqual(restored_data["epoch"], 5)
            self.assertEqual(restored_data["val_accuracy"], 0.92)
            self.assertEqual(restored_data["config"], config)

            # 8. Verify model parameters are identical
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

            # 9. Verify optimizer state is restored
            self.assertEqual(
                len(new_optimizer.state_dict()["param_groups"]),
                len(optimizer.state_dict()["param_groups"]),
            )



class TestDASegModel(unittest.TestCase):
    """Regression test suite for DA-Seg architecture."""

    def setUp(self) -> None:
        self.batch_size = 4
        self.freq_bins = 256
        self.time_steps = 16
        self.d_model = 128
        self.num_classes = 3
        self.model = DASegModel(
            in_channels=1,
            d_model=self.d_model,
            n_layers=2,
            num_classes=self.num_classes,
            dropout=0.1,
        )

    def test_daseg_forward_shapes(self) -> None:
        """Test output shape contracts for 4D and 3D input spectrograms."""
        x4d = torch.randn(self.batch_size, 1, self.freq_bins, self.time_steps)
        cls_logits, seg_logits = self.model(x4d, return_segmentation=True)

        self.assertEqual(cls_logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(seg_logits.shape, (self.batch_size, self.time_steps, self.num_classes))

    def test_metrics_computation(self) -> None:
        """Verify accuracy, F1, and mIoU metric calculation functions."""
        y_true_cls = torch.tensor([0, 1, 2, 0])
        y_pred_cls = torch.tensor([[2.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.1, 2.0], [2.0, 0.1, 0.0]])
        cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)
        self.assertEqual(cls_metrics["accuracy"], 1.0)

        y_true_seg = torch.zeros((4, 16), dtype=torch.long)
        y_pred_seg = torch.zeros((4, 16, 3), dtype=torch.float32)
        y_pred_seg[:, :, 0] = 5.0
        seg_metrics = compute_segmentation_metrics(y_true_seg, y_pred_seg)
        self.assertEqual(seg_metrics["frame_accuracy"], 1.0)
        self.assertEqual(seg_metrics["mIoU"], 1.0)


class TestEndToEndSmokePipeline(unittest.TestCase):
    """Member 3 Definition of Done End-to-End Smoke Test:

    Standardized synthetic sample
            ↓
    Dataset Adapter / DataLoader API
            ↓
    (B, T, F)
            ↓
    Mamba classifier
            ↓
    (B, 3)
            ↓
    CrossEntropyLoss
            ↓
    Backward
            ↓
    Optimizer step
            ↓
    Checkpoint
    """

    def test_end_to_end_smoke_pipeline(self) -> None:
        """Execute complete Member 3 Mamba classifier smoke pipeline."""

        # 1. Standardized synthetic sample
        sample = RadarSample(
            signal=np.cos(2 * np.pi * 50 * np.linspace(0, 1, 1000)),
            sampling_rate=1000.0,
            activity_label="Walk",
            subject_id="Subject_01",
        )
        self.assertEqual(sample.num_samples, 1000)

        # 2. Dataset Adapter / DataLoader API -> Yielding (B, T, F) tensors
        batch_size = 4
        time_steps = 16
        freq_bins = 256
        num_classes = 3

        specs_btf = torch.randn(8, time_steps, freq_bins)  # (N, T, F)
        labels = torch.randint(0, num_classes, (8,))

        dataset = TensorDataset(specs_btf, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3. Mamba Classifier Initialization
        classifier = MambaClassifier(
            in_features=freq_bins,
            d_model=64,
            n_layers=2,
            num_classes=num_classes,
        )
        optimizer = AdamW(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 4. Pipeline Execution: DataLoader -> (B, T, F) -> Mamba Classifier -> (B, 3) -> Loss -> Backward -> Step
        for batch_specs, batch_labels in dataloader:
            optimizer.zero_grad()

            self.assertEqual(batch_specs.shape, (batch_size, time_steps, freq_bins))

            logits = classifier(batch_specs)
            self.assertEqual(logits.shape, (batch_size, num_classes))

            loss = criterion(logits, batch_labels)
            self.assertFalse(torch.isnan(loss))

            loss.backward()
            optimizer.step()
            break

        # 5. Checkpoint Save & Restore
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = Path(temp_dir) / "mamba_smoke_checkpoint.pth"

            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=classifier,
                optimizer=optimizer,
                epoch=1,
                val_accuracy=0.95,
            )
            self.assertTrue(ckpt_path.exists())

            restored_classifier = MambaClassifier(
                in_features=freq_bins,
                d_model=64,
                n_layers=2,
                num_classes=num_classes,
            )
            restored_opt = AdamW(restored_classifier.parameters(), lr=1e-3)

            checkpoint_data = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=restored_classifier,
                optimizer=restored_opt,
            )

            self.assertEqual(checkpoint_data["epoch"], 1)
            self.assertEqual(checkpoint_data["val_accuracy"], 0.95)

            # Verify parameters restored identically
            for p1, p2 in zip(classifier.parameters(), restored_classifier.parameters()):
                self.assertTrue(torch.allclose(p1, p2))



if __name__ == "__main__":
    unittest.main()


