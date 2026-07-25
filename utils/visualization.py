"""Visualization utilities for micro-Doppler radar spectra and performance metrics.

Provides plotting helpers for STFT/FracSST spectrograms, confusion matrices,
and training history curves.
"""

from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_spectrogram(
    data: np.ndarray,
    title: str = "Micro-Doppler Spectrogram",
    xlabel: str = "Time (s)",
    ylabel: str = "Frequency (Hz)",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
) -> None:
    """Plot radar micro-Doppler time-frequency spectrogram.

    Args:
        data: 2D array of spectrogram magnitudes.
        title: Plot title text.
        xlabel: Label for horizontal time axis.
        ylabel: Label for vertical frequency axis.
        cmap: Matplotlib colormap string.
        save_path: Optional path to save image file.
    """
    raise NotImplementedError("plot_spectrogram visualization to be implemented.")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> None:
    """Plot classification confusion matrix heatmap.

    Args:
        cm: Confusion matrix 2D array.
        class_names: List of class category names.
        title: Title of heatmap plot.
        cmap: Seaborn colormap string.
        save_path: Optional output image file path.
    """
    raise NotImplementedError("plot_confusion_matrix visualization to be implemented.")
