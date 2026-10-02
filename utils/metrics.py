"""Performance evaluation metrics for radar human activity recognition and segmentation.

Computes accuracy, F1 score, precision, recall, confusion matrix, and temporal frame-level
segmentation metrics.
"""

from typing import Dict, Optional, List, Any
import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_confusion_matrix(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute numerical classification confusion matrix.

    Args:
        y_true: Ground truth target labels of shape (N,).
        y_pred: Predicted class label indices or logits of shape (N,) or (N, num_classes).
        num_classes: Number of distinct classes. If None, inferred from labels/logits.

    Returns:
        Confusion matrix of shape (num_classes, num_classes) where rows are true
        labels and columns are predicted labels.
    """
    y_true_np = _to_numpy(y_true).ravel()
    y_pred_np = _to_numpy(y_pred)

    if y_pred_np.ndim > 1:
        if num_classes is None:
            num_classes = y_pred_np.shape[-1]
        y_pred_np = np.argmax(y_pred_np, axis=-1)
    y_pred_np = y_pred_np.ravel()

    if num_classes is None:
        if len(y_true_np) == 0 and len(y_pred_np) == 0:
            return np.zeros((0, 0), dtype=np.int64)
        all_labels = np.concatenate([y_true_np, y_pred_np])
        num_classes = int(np.max(all_labels)) + 1 if len(all_labels) > 0 else 0

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_np, y_pred_np):
        t_int, p_int = int(t), int(p)
        if 0 <= t_int < num_classes and 0 <= p_int < num_classes:
            cm[t_int, p_int] += 1

    return cm


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


def compute_detailed_classification_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute overall and per-class classification metrics.

    Args:
        y_true: Ground truth target labels of shape (N,).
        y_pred: Predicted class label indices or logits of shape (N,) or (N, num_classes).
        class_names: Optional list of class names (e.g. ['Walk', 'Sit', 'Fall']).

    Returns:
        Dictionary containing accuracy, macro metrics, per-class metrics, and confusion matrix.
    """
    y_true_np = _to_numpy(y_true).ravel()
    y_pred_np = _to_numpy(y_pred)

    if y_pred_np.ndim > 1:
        y_pred_np = np.argmax(y_pred_np, axis=-1)
    y_pred_np = y_pred_np.ravel()

    accuracy = float(np.mean(y_true_np == y_pred_np)) if len(y_true_np) > 0 else 0.0

    num_classes = len(class_names) if class_names else int(np.max(np.concatenate([y_true_np, y_pred_np]))) + 1
    classes = list(range(num_classes))

    precisions, recalls, f1s = [], [], []
    per_class = {}

    for c in classes:
        c_name = class_names[c] if (class_names and c < len(class_names)) else f"Class_{c}"
        tp = int(np.sum((y_true_np == c) & (y_pred_np == c)))
        fp = int(np.sum((y_true_np != c) & (y_pred_np == c)))
        fn = int(np.sum((y_true_np == c) & (y_pred_np != c)))
        support = int(np.sum(y_true_np == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        per_class[c_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": support,
        }

    cm = compute_confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
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
