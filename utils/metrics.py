"""Performance evaluation metrics for radar human activity recognition and segmentation.

Computes accuracy, F1 score, precision, recall, confusion matrix, and temporal frame-level
segmentation metrics.
"""

from typing import Dict
import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_classification_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    """Compute overall classification metrics.

    Args:
        y_true: Ground truth target labels of shape (N,).
        y_pred: Predicted class label indices or logits of shape (N,) or (N, num_classes).

    Returns:
        Dictionary containing accuracy, macro precision, recall, and F1-score.
    """
    y_true = _to_numpy(y_true).ravel()
    y_pred = _to_numpy(y_pred)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_pred.ravel()

    accuracy = float(np.mean(y_true == y_pred))

    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s = [], [], []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
    }


def compute_segmentation_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    """Compute frame-level temporal activity segmentation metrics.

    Args:
        y_true: Ground truth frame sequences of shape (N, T).
        y_pred: Predicted frame sequence labels or logits of shape (N, T) or (N, T, num_classes).

    Returns:
        Dictionary containing frame accuracy and mean Intersection over Union (mIoU).
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_pred.ndim > y_true.ndim:
        y_pred = np.argmax(y_pred, axis=-1)

    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    frame_acc = float(np.mean(y_true_flat == y_pred_flat))

    classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    ious = []

    for c in classes:
        intersection = np.sum((y_true_flat == c) & (y_pred_flat == c))
        union = np.sum((y_true_flat == c) | (y_pred_flat == c))
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    miou = float(np.mean(ious)) if ious else 0.0

    return {
        "frame_accuracy": frame_acc,
        "mIoU": miou,
    }
