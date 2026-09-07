"""Digital Signal Processing (DSP) module for FracSST-Mamba.

Provides genuine Fractional Fourier Transform (FrFT), Synchrosqueezing Transform (SST),
and the clean high-level FrSST representation pipeline.
"""

from .frft import frft, find_optimal_alpha
from .synchrosqueezing import sst_1d, fracsst_1d, frsst, FrSSTResult

synchrosqueezing = fracsst_1d

__all__ = [
    "frft",
    "find_optimal_alpha",
    "sst_1d",
    "fracsst_1d",
    "frsst",
    "FrSSTResult",
    "synchrosqueezing",
]
