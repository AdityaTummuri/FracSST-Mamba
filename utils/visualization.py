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
        data: 2D array of spectrogram magnitudes of shape (n_freq, n_time).
        title: Plot title text.
        xlabel: Label for horizontal time axis.
        ylabel: Label for vertical frequency axis.
        cmap: Matplotlib colormap string.
        save_path: Optional path to save image file.
    """
    plt.figure(figsize=(10, 5))
    # DB scale magnitude for better contrast if values are non-zero
    data_plot = np.abs(data)
    if np.max(data_plot) > 0:
        data_plot = 10 * np.log10(data_plot + 1e-12)

    plt.imshow(data_plot, aspect="auto", origin="lower", cmap=cmap)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.colorbar(label="Power (dB)" if np.max(data_plot) > 0 else "Magnitude")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> None:
    """Plot classification confusion matrix heatmap.

    Args:
        cm: Confusion matrix 2D array of shape (n_classes, n_classes).
        class_names: List of class category names.
        title: Title of heatmap plot.
        cmap: Seaborn colormap string.
        save_path: Optional output image file path.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d" if np.issubdtype(cm.dtype, np.integer) else ".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
