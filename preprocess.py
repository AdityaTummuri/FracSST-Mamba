"""Offline DSP preprocessing script for FracSST-Mamba.

Scans raw radar data directory for signals (or generates synthetic raw signals if empty),
applies 1D Fractional Synchrosqueezing Transform (FracSST) to eliminate micro-Doppler smearing,
and saves preprocessed spectrogram tensors with target class and segmentation masks into data/processed/.
"""

import os
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from dsp.synchrosqueezing import fracsst_1d


# Activity Classes: 0: Walk, 1: Sit, 2: Fall
CLASS_NAMES = ["Walk", "Sit", "Fall"]


def generate_synthetic_raw_data(
    raw_dir: Path, n_samples_per_class: int = 5, fs: float = 1000.0, duration: float = 1.0
) -> None:
    """Generate synthetic 1D radar signals for Walk, Sit, and Fall activities if raw data is missing.

    Args:
        raw_dir: Output directory path for raw .npy files.
        n_samples_per_class: Number of synthetic raw files per activity class.
        fs: Sampling rate in Hz.
        duration: Signal duration in seconds.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)

    sample_id = 0
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for s in range(n_samples_per_class):
            if class_idx == 0:
                # Walk: Sinusoidal Doppler chirp with base frequency & limb swing
                f0 = 30.0 + s * 2.0
                phase = 2.0 * np.pi * (f0 * t + 10.0 * (t**2)) + (15.0 / 3.0) * np.sin(2.0 * np.pi * 3.0 * t)
            elif class_idx == 1:
                # Sit: Low frequency micro-doppler baseline
                phase = 2.0 * np.pi * (8.0 * t + 0.5 * (t**2))
            else:
                # Fall: Sudden high acceleration quadratic frequency burst
                phase = 2.0 * np.pi * (20.0 * t + 120.0 * (t**2))

            signal = np.cos(phase) + 1j * np.sin(phase)

            file_path = raw_dir / f"sample_{sample_id:03d}_{class_name}_cls{class_idx}.npy"
            np.save(file_path, signal)
            sample_id += 1

    print(f"[Preprocess] Generated {sample_id} synthetic raw radar samples in {raw_dir}")


def preprocess_all(
    raw_dir: Path,
    processed_dir: Path,
    fs: float = 1000.0,
    n_fft: int = 256,
    hop_length: int = 64,
) -> int:
    """Run FracSST-1D DSP pipeline over raw radar files and save PyTorch tensors.

    Args:
        raw_dir: Source directory containing raw .npy radar signals.
        processed_dir: Destination directory for .pt preprocessed files.
        fs: Sampling rate in Hz.
        n_fft: FFT window length.
        hop_length: Window hop size.

    Returns:
        Number of preprocessed files created.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(list(raw_dir.glob("*.npy")))
    if not raw_files:
        print("[Preprocess] No raw radar files found. Generating synthetic dummy data...")
        generate_synthetic_raw_data(raw_dir)
        raw_files = sorted(list(raw_dir.glob("*.npy")))

    processed_count = 0
    print(f"[Preprocess] Processing {len(raw_files)} raw radar files using FracSST...")

    for file_path in raw_files:
        # Load raw signal
        signal = np.load(file_path)

        # Parse class label from filename (default to 0 if absent)
        class_label = 0
        filename_str = file_path.stem
        for idx, c_name in enumerate(CLASS_NAMES):
            if f"cls{idx}" in filename_str or c_name.lower() in filename_str.lower():
                class_label = idx
                break

        # Apply Stage 1 DSP: FracSST-1D
        spectrogram, freq_bins, optimal_alpha = fracsst_1d(
            signal_1d=signal,
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            alpha=None,  # Automatically search optimal alpha via Rényi entropy
        )

        n_freq_bins, n_frames = spectrogram.shape

        # Generate frame-level segmentation mask (homogeneous sequence class label for mock data)
        seg_mask = np.full((n_frames,), class_label, dtype=np.int64)

        # Save preprocessed sample dictionary into PyTorch tensor file
        output_data = {
            "spectrogram": torch.tensor(spectrogram, dtype=torch.float32),
            "label": torch.tensor(class_label, dtype=torch.long),
            "segmentation_mask": torch.tensor(seg_mask, dtype=torch.long),
            "optimal_alpha": float(optimal_alpha),
        }

        output_file = processed_dir / f"{file_path.stem}.pt"
        torch.save(output_data, output_file)
        processed_count += 1

    print(f"[Preprocess] Successfully preprocessed and saved {processed_count} files to {processed_dir}")
    return processed_count


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    preprocess_all(raw_dir, processed_dir)


if __name__ == "__main__":
    main()
