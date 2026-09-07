"""Synchrosqueezing Transform (SST) and Fractional SST modules.

Implements standard Synchrosqueezing (sst_1d), Fractional Synchrosqueezing (fracsst_1d),
and the unified high-level FrSSTResult data structure.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import scipy.signal
from .frft import frft, find_optimal_alpha


@dataclass
class FrSSTResult:
    """Structured container for FrSST representation and its coordinate axes.

    Attributes:
        representation: 2D float32 array of shape (n_freq_bins, n_frames).
        alpha: Fractional order parameter alpha used in the transform.
        frequency_axis: 1D array of frequency bin centers in Hz of shape (n_freq_bins,).
        time_axis: 1D array of time frame centers in seconds of shape (n_frames,).
        metadata: Dictionary containing operational parameters (fs, n_fft, hop_length, etc.).
    """
    representation: np.ndarray
    alpha: float
    frequency_axis: np.ndarray
    time_axis: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def sst_1d(
    signal_1d: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop_length: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute standard 1D Synchrosqueezing Transform (SST) without fractional rotation.

    Computes STFT with standard Hamming window and derivative window, estimates instantaneous
    phase derivatives, and reassigns energy into frequency ridges.

    Args:
        signal_1d: Input 1D time-series signal array of shape (N,).
        fs: Sampling frequency in Hz. Must be positive.
        n_fft: FFT window length. Must be positive and <= signal length.
        hop_length: Hop size between consecutive frames. Must be positive.

    Returns:
        Tuple of:
            - sharpened_spectrogram: 2D array of shape (n_freq_bins, n_frames).
            - frequency_bins: 1D array of frequency bin centers in Hz of shape (n_freq_bins,).
            - time_axis: 1D array of time frame positions in seconds of shape (n_frames,).

    Raises:
        ValueError: On invalid dimensions, empty signal, or non-positive parameters.
    """
    sig = np.asarray(signal_1d)
    if sig.ndim != 1:
        raise ValueError(f"sst_1d expects a 1D signal, got shape {sig.shape}")
    N = len(sig)
    if N == 0:
        raise ValueError("sst_1d received an empty signal.")
    if fs <= 0:
        raise ValueError(f"Sampling rate fs must be positive, got {fs}")
    if n_fft <= 0:
        raise ValueError(f"n_fft must be positive, got {n_fft}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")

    signal_complex = np.asarray(sig, dtype=complex)

    # Window preparation
    w = scipy.signal.windows.hamming(n_fft)
    dw = np.gradient(w, 1.0 / fs)

    # Symmetric STFT frame extraction with constant zero-padding
    pad_len = n_fft // 2
    f_pad = np.pad(signal_complex, (pad_len, pad_len), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(f_pad, n_fft)[::hop_length]
    n_frames = len(frames)

    # STFT with standard and derivative windows
    Zxx = np.fft.fftshift(np.fft.fft(frames * w, axis=1), axes=1).T
    Zxx_dw = np.fft.fftshift(np.fft.fft(frames * dw, axis=1), axes=1).T
    freq_bins = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))
    n_freq_bins = len(freq_bins)

    # Instantaneous frequency estimation:
    # omega_inst = Im(Zxx_dw / Zxx) / (2 * pi)
    abs_Zxx = np.abs(Zxx)
    thresh = 1e-6 * np.max(abs_Zxx) if abs_Zxx.size > 0 and np.max(abs_Zxx) > 0 else 1e-12

    ratio = np.where(abs_Zxx >= thresh, Zxx_dw / (Zxx + 1e-12), 0.0)
    freq_offset = np.imag(ratio) / (2.0 * np.pi)

    # Instantaneous frequency: f_inst = f_bin - freq_offset
    inst_freq = freq_bins[:, None] - freq_offset
    inst_freq = np.where(abs_Zxx < thresh, freq_bins[:, None], inst_freq)

    # Energy reassignment
    df = fs / n_fft
    sharpened_spectrogram = np.zeros((n_freq_bins, n_frames), dtype=float)

    for j in range(n_frames):
        target_bins = np.clip(
            np.round((inst_freq[:, j] - freq_bins[0]) / df).astype(int),
            0,
            n_freq_bins - 1,
        )
        np.add.at(sharpened_spectrogram[:, j], target_bins, abs_Zxx[:, j])

    # Time frame centers in seconds
    time_axis = (np.arange(n_frames) * hop_length) / fs

    return sharpened_spectrogram, freq_bins, time_axis


def fracsst_1d(
    signal_1d: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop_length: int = 64,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute 1D Fractional Synchrosqueezing Transform (FracSST).

    Preserves exact backward-compatible signature: returns (sharpened_spectrogram, freq_bins, optimal_alpha).

    Args:
        signal_1d: Input 1D time-series signal array.
        fs: Sampling frequency in Hz.
        n_fft: FFT window length.
        hop_length: Hop size between consecutive frame windows.
        alpha: Optional fractional rotation order. If None, optimal alpha is automatically search-selected.

    Returns:
        Tuple containing:
            - sharpened_spectrogram_matrix: 2D array of shape (n_freq_bins, n_frames).
            - frequency_bins: 1D array of frequency bin centers in Hz.
            - optimal_alpha: Selected fractional rotation order alpha.
    """
    sig = np.asarray(signal_1d)
    if sig.ndim != 1:
        raise ValueError(f"fracsst_1d expects a 1D signal, got shape {sig.shape}")

    # 1. Search for optimal alpha if not provided
    if alpha is None:
        candidate_range = np.linspace(0.0, 1.5, 31)
        optimal_alpha = find_optimal_alpha(sig, candidate_range, n_fft=n_fft, hop_length=hop_length)
    else:
        optimal_alpha = float(alpha)

    # 2. Rotate signal into fractional domain
    rotated_signal = frft(sig, optimal_alpha)

    # 3. Apply standard Synchrosqueezing to the rotated signal
    sharpened_spec, freq_bins, _ = sst_1d(
        rotated_signal,
        fs=fs,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    return sharpened_spec, freq_bins, optimal_alpha


def frsst(
    signal: np.ndarray,
    sampling_rate: float,
    n_fft: int = 256,
    hop_length: int = 64,
    alpha: Optional[float] = None,
    alpha_search_range: Optional[np.ndarray] = None,
) -> FrSSTResult:
    """High-level clean API for Fractional Synchrosqueezing Transform (FrSST).

    Rotates signal into the optimal fractional domain to concentrate chirp-like Doppler components,
    computes windowed STFT and phase derivatives, and reassigns energy into instantaneous frequency ridges.

    Args:
        signal: 1D complex or real radar signal array of shape (N,).
        sampling_rate: Sampling frequency in Hz.
        n_fft: FFT window size (frequency resolution). Defaults to 256.
        hop_length: Frame hop size (temporal resolution). Defaults to 64.
        alpha: Optional explicit fractional order. If None, selected via Rényi entropy minimization.
        alpha_search_range: Optional candidate alpha orders to evaluate if alpha is None.

    Returns:
        FrSSTResult containing:
            - representation: 2D ndarray of shape (n_freq_bins, n_frames) with reassigned energy.
            - alpha: Selected or supplied fractional order alpha.
            - frequency_axis: Frequency bins in Hz.
            - time_axis: Time positions in seconds.
            - metadata: Dictionary containing sampling_rate, n_fft, hop_length, input_length.
    """
    sig = np.asarray(signal)
    if sig.ndim != 1:
        raise ValueError(f"frsst expects a 1D signal, got shape {sig.shape}")
    N = len(sig)
    if N == 0:
        raise ValueError("frsst received an empty signal.")

    if alpha is None:
        if alpha_search_range is None:
            alpha_search_range = np.linspace(0.0, 1.5, 31)
        selected_alpha = find_optimal_alpha(
            sig, alpha_search_range, n_fft=n_fft, hop_length=hop_length
        )
    else:
        selected_alpha = float(alpha)

    # Rotate signal into optimal fractional domain
    rotated_signal = frft(sig, selected_alpha)

    # Perform SST reassignment on rotated signal
    sharpened_spec, freq_bins, time_axis = sst_1d(
        rotated_signal,
        fs=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    metadata = {
        "sampling_rate": float(sampling_rate),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "input_length": int(N),
        "optimal_alpha": float(selected_alpha),
    }

    return FrSSTResult(
        representation=sharpened_spec,
        alpha=selected_alpha,
        frequency_axis=freq_bins,
        time_axis=time_axis,
        metadata=metadata,
    )
