"""PyTorch Dataset and DataLoader module for FracSST-Mamba radar data.

Loads preprocessed FracSST spectrogram tensors and corresponding sequence class labels.
Provides uniform (B, T, F) batch tensors for MambaClassifier sequence modeling.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class RadarMambaDataset(Dataset):
    """PyTorch Dataset for FracSST-Mamba preprocessed radar spectrograms.

    Produces:
        spectrogram: Tensor of shape (T, F) representing time frames and frequency bins.
        label: Scalar Tensor with sequence-level class index (0: Walk, 1: Sit, 2: Fall).

    Args:
        data_dir: Path to directory containing .pt preprocessed files.
        file_paths: Optional list of explicit file paths to load.
        return_mask: If True, also returns temporal segmentation mask of shape (T,).
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/processed",
        file_paths: Optional[List[Union[str, Path]]] = None,
        return_mask: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.return_mask = return_mask

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
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Fetch preprocessed radar sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - spectrogram: Tensor of shape (T, F).
                - label: Scalar Tensor with class index.
                - (optional) segmentation_mask: Tensor of shape (T,) if return_mask=True.
        """
        file_path = self.files[idx]
        data = torch.load(file_path, weights_only=True)

        spectrogram = data["spectrogram"]  # (F, T) or (1, F, T) or (T, F)
        label = data["label"]  # scalar long

        # Squeeze channel dimension if present: (1, F, T) -> (F, T)
        if spectrogram.ndim == 3:
            spectrogram = spectrogram.squeeze(0)

        # Transpose to (T, F) if input is (F, T) where F=256 and T != 256
        if spectrogram.shape[0] == 256 and spectrogram.shape[1] != 256:
            spectrogram = spectrogram.transpose(0, 1)

        if self.return_mask:
            seg_mask = data.get("segmentation_mask")
            if seg_mask is None:
                seg_mask = torch.full((spectrogram.shape[0],), label.item(), dtype=torch.long)
            return spectrogram, label, seg_mask

        return spectrogram, label


def pad_collate_fn(batch):
    """Collate function that pads variable-length time sequences (T, F) to max_T in the batch.

    Ensures the DataLoader produces a uniform (B, max_T, F) tensor regardless of whether
    the batch contains 5s recordings (T=79), 10s recordings (T=157), or a mixture.
    """
    has_mask = len(batch[0]) == 3
    max_t = max(item[0].shape[0] for item in batch)
    f_dim = batch[0][0].shape[1]
    batch_size = len(batch)

    padded_specs = torch.zeros((batch_size, max_t, f_dim), dtype=torch.float32)
    labels = torch.empty((batch_size,), dtype=torch.long)

    if has_mask:
        padded_masks = torch.zeros((batch_size, max_t), dtype=torch.long)
        for i, (spec, label, mask) in enumerate(batch):
            t_len = spec.shape[0]
            padded_specs[i, :t_len, :] = spec
            labels[i] = label
            padded_masks[i, :t_len] = mask
        return padded_specs, labels, padded_masks

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
    return_mask: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Construct Train and Validation PyTorch DataLoaders yielding (B, T, F) tensors.

    Args:
        data_dir: Path to directory containing preprocessed .pt files.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of dataset reserved for validation.
        num_workers: DataLoader worker subprocesses count.
        seed: Random seed for dataset split.
        file_paths: Optional explicit list of .pt files.
        return_mask: If True, DataLoader batches include temporal segmentation masks.

    Returns:
        Tuple containing (train_loader, val_loader).
    """
    dataset = RadarMambaDataset(
        data_dir=data_dir, file_paths=file_paths, return_mask=return_mask
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
