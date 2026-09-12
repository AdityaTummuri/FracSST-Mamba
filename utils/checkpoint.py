"""Checkpoint utility functions for saving and loading PyTorch models.

Provides clean infrastructure for model checkpoint serialization, state restoration,
and training resumption.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    val_accuracy: Optional[float] = None,
    val_mIoU: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save PyTorch model weights, optimizer state, and training metadata to disk.

    Args:
        checkpoint_path: Destination path for .pth file.
        model: PyTorch model instance.
        optimizer: Optional PyTorch optimizer instance.
        epoch: Current training epoch index.
        val_accuracy: Validation classification accuracy metric.
        val_mIoU: Validation segmentation mIoU metric.
        config: Optional configuration dictionary.

    Returns:
        Path to saved checkpoint file.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if val_accuracy is not None:
        payload["val_accuracy"] = val_accuracy
    if val_mIoU is not None:
        payload["val_mIoU"] = val_mIoU
    if config is not None:
        payload["config"] = config

    torch.save(payload, path)
    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint weights and optional optimizer state.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        model: PyTorch model instance to populate.
        optimizer: Optional PyTorch optimizer instance to restore state.
        device: Target device for loading weights.

    Returns:
        Loaded checkpoint dictionary containing epoch, accuracy, mIoU, and config.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device or torch.device("cpu"), weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
