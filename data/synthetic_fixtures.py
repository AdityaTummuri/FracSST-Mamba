"""Synthetic radar signal generator fixtures for testing and verification.

NOTE: These synthetic signals are test fixtures only and must NOT be treated as
research data or substitutes for real radar datasets.
"""

from typing import Tuple
import numpy as np
from data.schema import RadarSample


def create_synthetic_chirp_signal(
    fs: float = 1000.0,
    duration: float = 1.0,
    f_start: float = 30.0,
    chirp_rate: float = 120.0,
    micro_freq: float = 3.0,
    micro_amp: float = 15.0,
) -> np.ndarray:
    """Generate a synthetic 1D non-linear quadratic radar micro-Doppler chirp signal.

    Args:
        fs: Sampling frequency in Hz.
        duration: Signal duration in seconds.
        f_start: Base carrier/Doppler frequency in Hz.
        chirp_rate: Quadratic chirp acceleration parameter.
        micro_freq: Micro-motion modulation frequency (e.g. limb swing rate).
        micro_amp: Micro-motion modulation amplitude.

    Returns:
        1D complex64/complex128 array of shape (N,) where N = int(fs * duration).
    """
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)
    phase = (
        2.0 * np.pi * (f_start * t + 0.5 * chirp_rate * (t**2))
        + (micro_amp / micro_freq) * np.sin(2.0 * np.pi * micro_freq * t)
    )
    return np.cos(phase) + 1j * np.sin(phase)


def create_synthetic_radar_sample(
    activity: str = "Walk",
    fs: float = 1000.0,
    duration: float = 1.0,
    subject_id: str = "TestSubject_01",
) -> RadarSample:
    """Generate a mock RadarSample instance for unit tests.

    Args:
        activity: One of 'Walk', 'Sit', 'Fall'.
        fs: Sampling frequency in Hz.
        duration: Duration in seconds.
        subject_id: Synthetic subject ID string.

    Returns:
        RadarSample instance wrapping the synthetic 1D complex radar signal.
    """
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)
    if activity == "Walk":
        phase = 2.0 * np.pi * (30.0 * t + 10.0 * (t**2)) + 5.0 * np.sin(2.0 * np.pi * 3.0 * t)
    elif activity == "Sit":
        phase = 2.0 * np.pi * (8.0 * t + 0.5 * (t**2))
    elif activity == "Fall":
        phase = 2.0 * np.pi * (20.0 * t + 120.0 * (t**2))
    else:
        phase = 2.0 * np.pi * 15.0 * t

    signal = np.cos(phase) + 1j * np.sin(phase)

    return RadarSample(
        signal=signal,
        sampling_rate=fs,
        activity_label=activity,
        subject_id=subject_id,
        recording_id=f"fixture_{activity.lower()}",
        metadata={"is_fixture": True, "duration": duration},
    )
