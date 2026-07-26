"""Synchrosqueezing Transform (SST) and Fractional SST modules.

Implements energy reassignment over fractional time-frequency distributions
for concentrated micro-Doppler radar activity signatures.
"""

from typing import Tuple, Optional
import numpy as np
import scipy.signal
from .frft import frft, find_optimal_alpha


def fracsst_1d(
    signal_1d: np.ndarray,
    fs: float,
    n_fft: int,
    hop_length: int,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute 1D Fractional Synchrosqueezing Transform (FracSST).

    Rotates signal into the optimal fractional domain to align with acceleration chirps,
    computes STFT with standard and derivative windows, calculates instantaneous phase
    derivatives, and reassigns spectral energy into concentrated frequency ridges.

    Args:
        signal_1d: Input 1D time-series signal array.
        fs: Sampling frequency in Hz.
        n_fft: FFT window length.
        hop_length: Hop size between frame windows.
        alpha: Optional fractional rotation order. If None, optimal alpha is automatically search-selected.

    Returns:
        Tuple containing:
            - sharpened_spectrogram_matrix: 2D array of shape (n_freq_bins, n_frames).
            - frequency_bins: 1D array of frequency bin centers in Hz.
            - optimal_alpha: Selected fractional rotation order alpha.
    """
    signal_1d = np.asarray(signal_1d, dtype=complex)

    # 1. Search for optimal alpha if not provided
    if alpha is None:
        candidate_range = np.linspace(0.0, 1.5, 31)
        optimal_alpha = find_optimal_alpha(signal_1d, candidate_range)
    else:
        optimal_alpha = float(alpha)

    # 2. Rotate signal into fractional domain
    rotated_signal = frft(signal_1d, optimal_alpha)

    # 3. Prepare windows: standard Hamming window and its time derivative dw/dt
    N = len(rotated_signal)
    w = scipy.signal.windows.hamming(n_fft)
    dw = np.gradient(w, 1.0 / fs)

    pad_len = n_fft // 2
    f_pad = np.pad(rotated_signal, (pad_len, pad_len), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(f_pad, n_fft)[::hop_length]
    Zxx = np.fft.fftshift(np.fft.fft(frames * w, axis=1), axes=1).T


    Zxx_dw = np.fft.fftshift(np.fft.fft(frames * dw, axis=1), axes=1).T
    freq_bins = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))
    n_freq_bins, n_frames = Zxx.shape




    # 4. Calculate Instantaneous Frequencies
    # omega = Im(STFT_derivative / STFT_standard) / (2 * pi)
    abs_Zxx = np.abs(Zxx)
    thresh = 1e-6 * np.max(abs_Zxx) if abs_Zxx.size > 0 else 1e-12

    # Avoid division by zero
    ratio = np.where(abs_Zxx >= thresh, Zxx_dw / (Zxx + 1e-12), 0.0)
    freq_offset = np.imag(ratio) / (2.0 * np.pi)

    # Instantaneous frequency: f_inst = f_bin - freq_offset
    inst_freq = freq_bins[:, None] - freq_offset
    inst_freq = np.where(abs_Zxx < thresh, freq_bins[:, None], inst_freq)

    # 5. Energy Reassignment
    df = fs / n_fft
    sharpened_spectrogram = np.zeros((n_freq_bins, n_frames), dtype=float)

    for j in range(n_frames):
        target_bins = np.clip(
            np.round((inst_freq[:, j] - freq_bins[0]) / df).astype(int),
            0,
            n_freq_bins - 1,
        )
        np.add.at(sharpened_spectrogram[:, j], target_bins, abs_Zxx[:, j])

    return sharpened_spectrogram, freq_bins, optimal_alpha

