"""Fractional Fourier Transform (FrFT) implementations.

Provides tools for computing fast Fractional Fourier Transforms of 1D/2D signals
specifically tailored for radar micro-Doppler spectral analysis.
"""

import numpy as np
import scipy.signal
import torch


def frft(signal: np.ndarray | torch.Tensor, alpha: float) -> np.ndarray | torch.Tensor:
    """Compute the Fractional Fourier Transform (FrFT) of order alpha.

    Args:
        signal: Input 1D or 2D signal array or tensor.
        alpha: Fractional order parameter (0 <= alpha <= 2).

    Returns:
        FrFT transformed complex signal array or tensor of same shape as input.
    """
    raise NotImplementedError("FrFT computation algorithm to be implemented.")
