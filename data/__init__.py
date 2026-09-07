"""Data package for FracSST-Mamba radar dataset, schema, and loaders."""

from .schema import RadarSample, BaseRadarDatasetAdapter

__all__ = [
    "RadarSample",
    "BaseRadarDatasetAdapter",
]

# Lazily export PyTorch-dependent dataset loaders so non-torch DSP environments don't break
try:
    from .dataset import RadarMambaDataset, create_dataloaders
    __all__ += ["RadarMambaDataset", "create_dataloaders"]
except ImportError:
    pass
