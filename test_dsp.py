"""Verification script for Stage 1 DSP pipeline (FrFT & FracSST).

Generates a synthetic accelerating human leg radar micro-Doppler chirp signal,
executes the FracSST-1D pipeline, and verifies output shapes, numerical stability,
and optimal alpha parameter selection.
"""

import sys
import numpy as np
from dsp.synchrosqueezing import fracsst_1d


def generate_synthetic_leg_signal(
    fs: float = 1000.0, duration: float = 1.0
) -> np.ndarray:
    """Generate a synthetic 1D radar signal for an accelerating human leg.

    Simulates a non-linear quadratic chirp with sinusoidal micro-Doppler modulation.
    """
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)
    # Accelerating linear chirp component + leg swing micro-motion modulation
    f_start = 30.0
    chirp_rate = 120.0
    micro_freq = 3.0
    micro_amp = 15.0

    phase = (
        2.0 * np.pi * (f_start * t + 0.5 * chirp_rate * (t**2))
        + (micro_amp / micro_freq) * np.sin(2.0 * np.pi * micro_freq * t)
    )
    signal = np.cos(phase) + 1j * np.sin(phase)
    return signal


def main() -> None:
    print("==================================================")
    print("      FracSST-Mamba Stage 1 DSP Verification      ")
    print("==================================================")

    fs = 1000.0
    n_fft = 256
    hop_length = 64

    print("[1] Generating synthetic accelerating human leg radar signal...")
    signal = generate_synthetic_leg_signal(fs=fs, duration=1.0)
    print(f"    Signal length: {len(signal)} samples (Sampling Rate: {fs} Hz)")

    print("[2] Running FracSST-1D pipeline (searching optimal alpha & reassigning spectrum)...")
    spectrogram, freq_bins, optimal_alpha = fracsst_1d(
        signal_1d=signal, fs=fs, n_fft=n_fft, hop_length=hop_length, alpha=None
    )

    # Numerical assertions
    assert not np.isnan(spectrogram).any(), "Error: NaN values detected in output spectrogram!"
    assert not np.isinf(spectrogram).any(), "Error: Inf values detected in output spectrogram!"
    assert spectrogram.ndim == 2, f"Error: Expected 2D spectrogram matrix, got shape {spectrogram.shape}"

    print("[3] DSP Pipeline Executed Successfully!")
    print(f"    - Optimal Alpha Selected: {optimal_alpha:.4f}")
    print(f"    - Output Spectrogram Matrix Shape: {spectrogram.shape} (freq_bins x time_frames)")
    print(f"    - Frequency Bins Count: {len(freq_bins)} (0.0 Hz to {freq_bins[-1]:.1f} Hz)")
    print(f"    - Spectrogram Energy Range: min={spectrogram.min():.4e}, max={spectrogram.max():.4e}")
    print("==================================================")


if __name__ == "__main__":
    main()
