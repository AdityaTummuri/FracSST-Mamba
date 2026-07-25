"""Performance evaluation metrics for radar human activity recognition and segmentation.

Computes accuracy, F1 score, precision, recall, confusion matrix, and temporal frame-level
segmentation metrics.
"""

from typing import Dict, Any
import numpy as np
import torch


def compute_classification_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    """Compute overall classification metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class label indices.

    Returns:
        Dictionary containing accuracy, macro precision, recall, and F1-score.
    """
    raise NotImplementedError("compute_classification_metrics to be implemented.")


def compute_segmentation_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    """Compute frame-level temporal activity segmentation metrics.

    Args:
        y_true: Ground truth frame sequences.
        y_pred: Predicted frame sequence labels.

    Returns:
        Dictionary containing frame accuracy, mIoU, and boundary edit distance.
    """
    raise NotImplementedError("compute_segmentation_metrics to be implemented.")
