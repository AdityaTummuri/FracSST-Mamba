"""PyTorch Dataset and DataLoader module for FracSST-Mamba radar data.

Loads preprocessed FracSST spectrogram tensors and corresponding sequence class labels.
Provides uniform (B, T, F) batch tensors for MambaClassifier sequence modeling.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class RadarMambaDataset(Dataset):
    """PyTorch Dataset for FracSST-Mamba preprocessed radar spectrograms.

    Produces:
        spectrogram_sequence: Tensor of shape (n_frames, n_freq_bins) i.e. (T, F).
        sequence_class_label: Scalar Tensor with sequence-level class index (0: Walk, 1: Sit, 2: Fall).

    Args:
        data_dir: Path to directory containing .pt preprocessed files.
        file_paths: Optional list of explicit file paths to load.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/processed",
        file_paths: Optional[List[Union[str, Path]]] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)

        if file_paths is not None:
            self.files = [Path(p) for p in file_paths]
        else:
            if not self.data_dir.exists():
                raise FileNotFoundError(
                    f"Data directory {self.data_dir} does not exist. "
                    "Please run preprocess.py first."
                )
            self.files = sorted(list(self.data_dir.glob("*.pt")))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .pt files found in {self.data_dir}. "
                "Please run preprocess.py to generate preprocessed radar tensors."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch preprocessed radar sample formatted for Mamba sequence models.

        Args:
            idx: Sample index.

        Returns:
            Tuple containing:
                - spectrogram_sequence: Tensor of shape (n_frames, n_freq_bins) i.e. (T, F).
                - sequence_class_label: Scalar Tensor with sequence-level class index.
        """
        file_path = self.files[idx]
        data = torch.load(file_path, weights_only=True)

        spectrogram = data["spectrogram"]
        label = data["label"]

        if not isinstance(spectrogram, torch.Tensor):
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        else:
            spectrogram = spectrogram.to(dtype=torch.float32)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.to(dtype=torch.long)

        # Standardize to (T, F) temporal sequence format
        if spectrogram.ndim == 3 and spectrogram.shape[0] == 1:
            spectrogram = spectrogram.squeeze(0)

        # If shape is (F, T) where frequency bins > time frames, transpose to (T, F)
        if spectrogram.ndim == 2 and spectrogram.shape[0] > spectrogram.shape[1]:
            spectrogram = spectrogram.transpose(0, 1)

        return spectrogram, label


def pad_collate_fn(batch):
    """Collate function that pads variable-length time sequences (T, F) to max_T in the batch.

    Ensures the DataLoader produces a uniform (B, max_T, F) tensor regardless of whether
    the batch contains 5s recordings (T=79), 10s recordings (T=157), or a mixture.
    """
    max_t = max(item[0].shape[0] for item in batch)
    f_dim = batch[0][0].shape[1]
    batch_size = len(batch)

    padded_specs = torch.zeros((batch_size, max_t, f_dim), dtype=torch.float32)
    labels = torch.empty((batch_size,), dtype=torch.long)

    for i, (spec, label) in enumerate(batch):
        t_len = spec.shape[0]
        padded_specs[i, :t_len, :] = spec
        labels[i] = label

    return padded_specs, labels



def create_dataloaders(
    data_dir: Union[str, Path] = "data/processed",
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    file_paths: Optional[List[Union[str, Path]]] = None,
    split_manifest: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Construct Train and Validation PyTorch DataLoaders yielding (B, T, F) tensors.

    If split_manifest is provided and exists, partitions are strictly loaded from the
    subject-independent split manifest (e.g. data/metadata/train_val_test_split.json)
    guaranteeing zero identity leakage. Otherwise, falls back to random_split.

    Args:
        data_dir: Path to directory containing preprocessed .pt files.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of dataset reserved for validation (used if no manifest).
        num_workers: DataLoader worker subprocesses count.
        seed: Random seed for dataset split (used if no manifest).
        file_paths: Optional explicit list of .pt files.
        split_manifest: Optional path to train_val_test_split.json.

    Returns:
        Tuple containing (train_loader, val_loader).
    """
    data_dir = Path(data_dir)

    # 1. Subject-independent manifest path resolution
    if split_manifest is not None and Path(split_manifest).exists():
        with open(split_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        train_entries = manifest.get("train_files", [])
        val_entries = manifest.get("val_files", [])

        train_files = [data_dir / f"{Path(item['filename']).stem}.pt" for item in train_entries]
        val_files = [data_dir / f"{Path(item['filename']).stem}.pt" for item in val_entries]

        train_existing = [p for p in train_files if p.exists()]
        val_existing = [p for p in val_files if p.exists()]

        if len(train_existing) > 0 and len(val_existing) > 0:
            train_set = RadarMambaDataset(file_paths=train_existing)
            val_set = RadarMambaDataset(file_paths=val_existing)

            train_loader = DataLoader(
                train_set,
                batch_size=min(batch_size, len(train_set)),
                shuffle=True,
                num_workers=num_workers,
                collate_fn=pad_collate_fn,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=min(batch_size, len(val_set)),
                shuffle=False,
                num_workers=num_workers,
                collate_fn=pad_collate_fn,
            )
            return train_loader, val_loader

    # 2. Fallback to random_split if no manifest or files not matched
    dataset = RadarMambaDataset(
        data_dir=data_dir, file_paths=file_paths
    )
    total_samples = len(dataset)

    val_size = max(1, int(total_samples * val_split))
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=min(batch_size, len(train_set)),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(batch_size, len(val_set)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
    )

    return train_loader, val_loader


def create_test_dataloader(
    data_dir: Union[str, Path] = "data/processed",
    split_manifest: Union[str, Path] = "data/metadata/train_val_test_split.json",
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    """Construct DataLoader for the untouched subject-independent test partition.

    Args:
        data_dir: Path to directory containing preprocessed .pt files.
        split_manifest: Path to train_val_test_split.json containing test_files manifest.
        batch_size: Batch size for DataLoader.
        num_workers: Subprocess workers count.

    Returns:
        PyTorch DataLoader yielding (B, max_T, F) test batches and (B,) labels.
    """
    data_dir = Path(data_dir)
    manifest_path = Path(split_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_entries = manifest.get("test_files", [])
    if not test_entries:
        raise ValueError(f"No test_files entries found in {manifest_path}")

    test_files = [data_dir / f"{Path(item['filename']).stem}.pt" for item in test_entries]
    test_existing = [p for p in test_files if p.exists()]

    if len(test_existing) == 0:
        raise FileNotFoundError(f"No preprocessed test files found in {data_dir} for manifest {manifest_path}")

    # pyrefly: ignore [bad-argument-type]
    test_set = RadarMambaDataset(file_paths=test_existing)
    return DataLoader(
        test_set,
        batch_size=min(batch_size, len(test_set)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
    )
