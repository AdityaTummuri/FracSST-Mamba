"""Utility functions for FracSST-Mamba project."""

from .metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_segmentation_metrics,
)
from .checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_segmentation_metrics",
    "save_checkpoint",
    "load_checkpoint",
]

# Optional visualization imports (only if matplotlib and seaborn are installed)
try:
    from .visualization import plot_spectrogram, plot_confusion_matrix

    __all__ += ["plot_spectrogram", "plot_confusion_matrix"]
except ImportError:
    pass


