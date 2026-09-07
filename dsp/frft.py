"""Fractional Fourier Transform (FrFT) implementation and parameter search.

Provides functions for computing fast Fractional Fourier Transforms of 1D signals
specifically tailored for radar micro-Doppler spectral analysis and optimal alpha parameter selection.
"""

from typing import Union, Optional
import numpy as np
import scipy.signal


def frft(f: np.ndarray, a: float) -> np.ndarray:
    """Compute the 1D Fractional Fourier Transform (FrFT) of order a.

    Uses Ozaktas fast chirp-multiplication algorithm for arbitrary orders and handles
    modulo-4 boundary integer cases (a=0, 1, 2, 3).

    Args:
        f: Input 1D complex or real signal array of shape (N,).
        a: Fractional order parameter (any real number).

    Returns:
        Transformed 1D complex signal array of shape (N,) with preserved L2 energy norm.

    Raises:
        ValueError: If f is not a 1D array.
    """
    f_arr = np.asarray(f)
    if f_arr.ndim != 1:
        raise ValueError(f"frft expects a 1D array, but got shape {f_arr.shape}")

    f = np.asarray(f_arr, dtype=complex)
    N = len(f)
    if N == 0:
        return f.copy()

    # Reduce order to interval [-2.0, 2.0)
    a_mod = float(a) % 4.0
    if a_mod >= 2.0:
        a_mod -= 4.0

    # Boundary integer checks
    if np.isclose(a_mod, 0.0, atol=1e-10):
        return f.copy()
    if np.isclose(a_mod, 1.0, atol=1e-10):
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) / np.sqrt(N)
    if np.isclose(a_mod, -1.0, atol=1e-10):
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f))) * np.sqrt(N)
    if np.isclose(np.abs(a_mod), 2.0, atol=1e-10):
        return np.flip(f).copy()

    # Shift order into interval [-0.5, 0.5]
    if a_mod > 0.5:
        return frft(frft(f, 1.0), a_mod - 1.0)
    if a_mod < -0.5:
        return frft(frft(f, -1.0), a_mod + 1.0)

    # Now a_mod is strictly in [-0.5, 0.5]
    phi = a_mod * (np.pi / 2.0)
    if np.abs(phi) < 1e-10:
        return f.copy()

    tan_half = np.tan(phi / 2.0)
    sin_phi = np.sin(phi)
    csc_phi = 1.0 / sin_phi

    # Up-sample signal by factor of 2 in frequency domain (zero padding)
    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f)))
    F_padded = np.pad(F, (N // 2, (N + 1) // 2), mode="constant")
    f_2x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(F_padded))) * np.sqrt(2)

    # Grid for 2N points centered at 0
    m = np.arange(-N, N)
    u = m / (2.0 * np.sqrt(N))

    # First chirp multiplication
    g = f_2x * np.exp(-1j * np.pi * tan_half * (u**2))

    # Chirp convolution kernel
    m_h = np.arange(-2 * N, 2 * N + 1)
    u_h = m_h / (2.0 * np.sqrt(N))
    h = np.exp(1j * np.pi * csc_phi * (u_h**2))

    # FFT-based fast convolution
    conv = scipy.signal.fftconvolve(g, h, mode="same") / (2.0 * np.sqrt(N))

    # Second chirp multiplication
    g2 = conv * np.exp(-1j * np.pi * tan_half * (u**2))

    # Constant phase factor
    sgn = 1.0 if sin_phi >= 0 else -1.0
    c_phi = np.exp(-1j * (np.pi * sgn / 4.0 - phi / 2.0)) * np.sqrt(np.abs(csc_phi))

    g_out = c_phi * g2

    # Down-sample back to length N
    res = g_out[::2][:N]

    norm_f = np.linalg.norm(f)
    if norm_f > 0:
        norm_res = np.linalg.norm(res)
        if norm_res > 0:
            res = res * (norm_f / norm_res)

    return res


def find_optimal_alpha(
    f: np.ndarray,
    search_range: Optional[np.ndarray] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> float:
    """Find the optimal fractional order alpha minimizing Rényi Entropy (order=3).

    A lower Rényi entropy corresponds to higher time-frequency energy concentration
    (minimal spectral smearing).

    Args:
        f: Input 1D signal array of shape (N,).
        search_range: Array of candidate fractional orders alpha to evaluate.
                      Defaults to np.linspace(0.0, 1.5, 31).
        n_fft: Optional FFT window size for time-frequency entropy estimation.
               Defaults to min(256, len(f)).
        hop_length: Optional hop size between frames for entropy estimation.
                    Defaults to max(1, n_fft // 4).

    Returns:
        Optimal fractional order alpha that yields the minimal Rényi entropy.

    Raises:
        ValueError: If f is empty or not a 1D array.
    """
    f_arr = np.asarray(f)
    if f_arr.ndim != 1:
        raise ValueError(f"find_optimal_alpha expects a 1D array, got shape {f_arr.shape}")
    N = len(f_arr)
    if N == 0:
        raise ValueError("Cannot find optimal alpha for an empty signal.")

    if search_range is None:
        search_range = np.linspace(0.0, 1.5, 31)

    if n_fft is None:
        n_fft = min(256, N)
    else:
        n_fft = min(n_fft, N)

    if hop_length is None:
        hop_length = max(1, n_fft // 4)

    best_alpha = float(search_range[0])
    min_entropy = float("inf")

    w = scipy.signal.windows.hamming(n_fft)
    pad_len = n_fft // 2

    for alpha in search_range:
        f_rot = frft(f_arr, float(alpha))
        f_pad = np.pad(f_rot, (pad_len, pad_len), mode="constant")

        frames = np.lib.stride_tricks.sliding_window_view(f_pad, n_fft)[::hop_length]
        spec_mag = np.abs(np.fft.fft(frames * w, axis=1)) ** 2

        total_energy = np.sum(spec_mag)
        if total_energy < 1e-12:
            continue

        P = spec_mag / total_energy
        # Rényi Entropy of order 3: R3 = -0.5 * ln(sum(P^3))
        p3_sum = np.sum(P**3)
        entropy = -0.5 * np.log(p3_sum + 1e-12)

        if entropy < min_entropy:
            min_entropy = entropy
            best_alpha = float(alpha)

    return best_alpha
