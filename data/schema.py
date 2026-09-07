"""Standard radar data structures and base dataset adapters.

Defines the internal RadarSample representation and abstract base adapter,
allowing external radar datasets (DIAT, CI4R, UoG, or proprietary hardware streams)
to be converted into a uniform format without modifying the DSP or model layers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np


@dataclass
class RadarSample:
    """Standard internal representation of a radar activity sample.

    Attributes:
        signal: 1D complex or real radar time series array (e.g. I + j*Q) of shape (N,).
        sampling_rate: Pulse repetition frequency / ADC sampling rate in Hz.
        activity_label: Activity class name or target index (e.g. 'Walk', 'Sit', 'Fall' or 0, 1, 2).
        num_samples: Total number of complex samples N. Automatically populated if omitted.
        subject_id: Optional identifier for the human subject (e.g. 'Subject_01') to allow
                    clean subject-independent (Leave-One-Subject-Out) train/val/test splits.
        recording_id: Optional session/trial/recording identifier.
        metadata: Optional dictionary storing radar hardware parameters, center frequency,
                  chirp duration, or environment notes.
    """
    signal: np.ndarray
    sampling_rate: float
    activity_label: Union[str, int]
    num_samples: int = 0
    subject_id: Optional[str] = None
    recording_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal)
        if self.signal.ndim != 1:
            raise ValueError(f"RadarSample signal must be 1D, got shape {self.signal.shape}")
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")
        if self.num_samples == 0:
            self.num_samples = len(self.signal)

    @property
    def duration_seconds(self) -> float:
        """Total duration of the radar sample in seconds."""
        return self.num_samples / self.sampling_rate

    @property
    def is_complex(self) -> bool:
        """Whether the signal contains complex I/Q data."""
        return np.iscomplexobj(self.signal)


class BaseRadarDatasetAdapter:
    """Abstract base adapter for transforming raw radar datasets into standard RadarSamples.

    Subclasses implement format-specific loading (e.g. HDF5, MATLAB .mat, binary ADC dumps,
    or CSV/NPY) for datasets like DIAT-μRadHAR, CI4R, or University of Glasgow micro-Doppler datasets.
    """

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root_dir = Path(root_dir)

    def discover_samples(self) -> List[Any]:
        """Scan directory and return a list of sample identifiers or file paths."""
        raise NotImplementedError("Subclasses must implement discover_samples()")

    def load_sample(self, identifier: Any) -> RadarSample:
        """Load and convert a single raw file/record into a standard RadarSample."""
        raise NotImplementedError("Subclasses must implement load_sample()")
