"""Offline DSP preprocessing script for FracSST-Mamba.

Connects UoGRadarAdapter to the existing 1D Fractional Synchrosqueezing Transform (fracsst_1d)
to convert real radar recordings into time-frequency spectrogram tensors saved into data/processed/.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch

from dsp.synchrosqueezing import fracsst_1d
from data.adapters.uog_adapter import UoGRadarAdapter
from data.synthetic_fixtures import create_synthetic_chirp_signal


# Canonical Activity Classes: 0: Walk, 1: Sit, 2: Fall
CLASS_NAMES = ["Walk", "Sit", "Fall"]
CLASS_MAP = {"Walk": 0, "Sit": 1, "Fall": 2}


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
                f0 = 30.0 + s * 2.0
                phase = 2.0 * np.pi * (f0 * t + 10.0 * (t**2)) + (15.0 / 3.0) * np.sin(2.0 * np.pi * 3.0 * t)
            elif class_idx == 1:
                phase = 2.0 * np.pi * (8.0 * t + 0.5 * (t**2))
            else:
                phase = 2.0 * np.pi * (20.0 * t + 120.0 * (t**2))

            signal = np.cos(phase) + 1j * np.sin(phase)

            file_path = raw_dir / f"sample_{sample_id:03d}_{class_name}_cls{class_idx}.npy"
            np.save(file_path, signal)
            sample_id += 1

    print(f"[Preprocess] Generated {sample_id} synthetic raw radar samples in {raw_dir}")


def _process_single_real_file(
    fpath_str: str,
    processed_dir_str: str,
    fs: float,
    n_fft: int,
    hop_length: int,
    alpha: Optional[float],
) -> str:
    """Worker function to process one real radar file via UoGRadarAdapter and FracSST."""
    fpath = Path(fpath_str)
    processed_dir = Path(processed_dir_str)
    adapter = UoGRadarAdapter()

    sample = adapter.load_sample(fpath)
    class_label = CLASS_MAP.get(sample.activity_label, 0)

    spectrogram, freq_bins, optimal_alpha = fracsst_1d(
        signal_1d=sample.signal,
        fs=sample.sampling_rate if sample.sampling_rate > 0 else fs,
        n_fft=n_fft,
        hop_length=hop_length,
        alpha=alpha,
    )

    n_freq_bins, n_frames = spectrogram.shape
    seg_mask = np.full((n_frames,), class_label, dtype=np.int64)

    output_data = {
        "spectrogram": torch.tensor(spectrogram, dtype=torch.float32),
        "label": torch.tensor(class_label, dtype=torch.long),
        "segmentation_mask": torch.tensor(seg_mask, dtype=torch.long),
        "optimal_alpha": float(optimal_alpha),
        "subject_id": sample.subject_id,
        "recording_id": sample.recording_id,
    }

    output_file = processed_dir / f"{sample.recording_id}.pt"
    torch.save(output_data, output_file)
    return output_file.name


def preprocess_all(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    fs: float = 1000.0,
    n_fft: int = 256,
    hop_length: int = 64,
    alpha: Optional[float] = 0.5,
    max_workers: Optional[int] = None,
) -> int:
    """Run FracSST-1D DSP pipeline over raw radar files and save PyTorch tensors.

    Discovers real .dat radar recordings using UoGRadarAdapter.
    If real radar data is missing, falls back to .npy or synthetic test fixtures.

    Args:
        raw_dir: Source directory containing raw radar files.
        processed_dir: Destination directory for .pt preprocessed files.
        fs: Sampling rate in Hz.
        n_fft: FFT window length.
        hop_length: Window hop size.
        alpha: Fractional order parameter (default 0.5, matching params.yaml).
        max_workers: Parallel worker count for multiprocessing.

    Returns:
        Number of preprocessed .pt files created.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check for real UoG radar .dat files
    adapter = UoGRadarAdapter(root_dir=raw_dir)
    real_files = adapter.discover_samples()

    if real_files:
        total_files = len(real_files)
        workers = max_workers or min(os.cpu_count() or 4, 12)
        print(f"[Preprocess] Found {total_files} real radar files via UoGRadarAdapter.")
        print(f"[Preprocess] Processing using existing FracSST (workers={workers}, alpha={alpha})...")

        processed_count = 0
        file_paths_str = [str(f) for f in real_files]
        processed_dir_str = str(processed_dir)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_single_real_file,
                    fp,
                    processed_dir_str,
                    fs,
                    n_fft,
                    hop_length,
                    alpha,
                ): fp
                for fp in file_paths_str
            }
            for future in as_completed(futures):
                future.result()
                processed_count += 1
                if processed_count % 50 == 0 or processed_count == total_files:
                    print(f"  [Progress] {processed_count}/{total_files} files preprocessed ({processed_count/total_files*100:.1f}%)")

        print(f"[Preprocess] Successfully preprocessed and saved {processed_count} files to {processed_dir}")
        return processed_count

    # 2. Fallback: check for .npy files
    raw_files = sorted(list(raw_dir.glob("*.npy")))
    if not raw_files:
        print("[Preprocess] No raw radar files found. Generating synthetic dummy data...")
        generate_synthetic_raw_data(raw_dir)
        raw_files = sorted(list(raw_dir.glob("*.npy")))

    processed_count = 0
    print(f"[Preprocess] Processing {len(raw_files)} raw .npy radar files using FracSST...")

    for file_path in raw_files:
        signal = np.load(file_path)
        class_label = 0
        filename_str = file_path.stem
        for idx, c_name in enumerate(CLASS_NAMES):
            if f"cls{idx}" in filename_str or c_name.lower() in filename_str.lower():
                class_label = idx
                break

        spectrogram, freq_bins, optimal_alpha = fracsst_1d(
            signal_1d=signal,
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            alpha=alpha,
        )

        n_freq_bins, n_frames = spectrogram.shape
        seg_mask = np.full((n_frames,), class_label, dtype=np.int64)

        output_data = {
            "spectrogram": torch.tensor(spectrogram, dtype=torch.float32),
            "label": torch.tensor(class_label, dtype=torch.long),
            "segmentation_mask": torch.tensor(seg_mask, dtype=torch.long),
            "optimal_alpha": float(optimal_alpha),
            "subject_id": "synthetic",
            "recording_id": file_path.stem,
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
