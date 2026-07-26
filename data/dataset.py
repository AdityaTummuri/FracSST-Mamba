"""PyTorch Dataset and DataLoader module for FracSST-Mamba radar data.

Loads preprocessed FracSST spectrogram tensors and corresponding sequence class
and temporal segmentation targets.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class RadarMambaDataset(Dataset):
    """PyTorch Dataset for FracSST-Mamba preprocessed radar spectrograms.

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch preprocessed radar sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple containing:
                - spectrogram_tensor: Tensor of shape (1, n_freq_bins, n_frames).
                - sequence_class_label: Scalar Tensor with sequence-level class index.
                - frame_segmentation_labels: Tensor of shape (n_frames,) with frame-level class indices.
        """
        file_path = self.files[idx]
        data = torch.load(file_path, weights_only=True)

        spectrogram = data["spectrogram"]  # (n_freq_bins, n_frames)
        label = data["label"]  # scalar long
        segmentation_mask = data["segmentation_mask"]  # (n_frames,)

        # Ensure channel dimension is present: (1, F, T)
        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0)

        return spectrogram, label, segmentation_mask


def create_dataloaders(
    data_dir: Union[str, Path] = "data/processed",
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Construct Train and Validation PyTorch DataLoaders.

    Args:
        data_dir: Path to directory containing preprocessed .pt files.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of dataset reserved for validation.
        num_workers: DataLoader worker subprocesses count.
        seed: Random seed for dataset split.

    Returns:
        Tuple containing (train_loader, val_loader).
    """
    dataset = RadarMambaDataset(data_dir=data_dir)
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
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(batch_size, len(val_set)),
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
