"""Dataset Adapter for the University of Glasgow (UoG) Micro-Doppler Radar Dataset.

Converts raw Ancortek FMCW .dat files into standard RadarSample instances.
Extracts slow-time Doppler time-series from 2D beat-note signals via fast-time Range-FFT
and MTI static clutter filtering.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from data.schema import BaseRadarDatasetAdapter, RadarSample


# Mapping raw activity prefixes to canonical activity labels
ACTIVITY_PREFIX_MAP = {
    "1": "Walk",
    "2": "Sit",
    "6": "Fall",
}


class UoGRadarAdapter(BaseRadarDatasetAdapter):
    """Adapter for University of Glasgow INSHEP FMCW radar dataset (.dat format).

    Hardware / Signal Structure:
        - Carrier frequency: 5.8 GHz (C-band)
        - Chirp duration: 1.0 ms (Slow-time PRF / fs = 1000 Hz)
        - ADC samples per chirp: 128
        - Bandwidth: 400 MHz
        - Antenna: Tx/Rx Yagi antennas (+17 dB gain)

    Args:
        root_dir: Directory containing raw .dat files (e.g. data/raw/Dataset_848).
        mode: Signal extraction mode: 'peak_bin' (default, selects range bin with
              maximum target Doppler variance) or 'integrated' (sums across range bins).
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/raw/Dataset_848",
        mode: str = "peak_bin",
    ) -> None:
        super().__init__(root_dir=root_dir)
        if mode not in {"peak_bin", "integrated"}:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'peak_bin' or 'integrated'.")
        self.mode = mode

    def discover_samples(self) -> List[Path]:
        """Discover all valid target activity .dat files (Walk, Sit, Fall).

        Returns:
            Sorted list of file Paths matching project target prefixes ('1', '2', '6').
        """
        if not self.root_dir.exists():
            return []

        all_dats = sorted(self.root_dir.glob("**/*.dat"))
        valid_files = [
            f for f in all_dats if f.name and f.name[0] in ACTIVITY_PREFIX_MAP
        ]
        return valid_files

    def parse_filename_metadata(self, filename: str) -> Dict[str, Any]:
        """Extract activity label, subject ID, and repetition from filename.

        Example filename: 1P36A01R01.dat -> prefix 1, Subject P36, Activity A01, Repetition R01.
        """
        match = re.match(r"^([126])([A-Za-z0-9]+?)A(\d+)R(\d+)", filename)
        if match:
            act_prefix, subject_id, act_code, repetition = match.groups()
            activity_label = ACTIVITY_PREFIX_MAP.get(act_prefix, "Unknown")
            return {
                "activity_label": activity_label,
                "raw_activity_code": f"A{act_code}",
                "subject_id": subject_id,
                "repetition": f"R{repetition}",
            }

        # Fallback if non-standard
        prefix = filename[0] if filename else ""
        return {
            "activity_label": ACTIVITY_PREFIX_MAP.get(prefix, "Unknown"),
            "raw_activity_code": "Unknown",
            "subject_id": "Unknown",
            "repetition": "Unknown",
        }

    def load_sample(self, file_path: Union[str, Path]) -> RadarSample:
        """Load and convert a single UoG .dat file into a standard RadarSample.

        Args:
            file_path: Path to the .dat file.

        Returns:
            Standard RadarSample containing 1D complex slow-time Doppler signal.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Radar file not found: {file_path}")

        # 1. Read header and data stream
        with open(file_path, "r", errors="ignore") as f:
            line0 = f.readline()
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()
            if not line3:
                raise ValueError(f"File {file_path} is too short or corrupted.")
            data_str = f.read().replace("i", "j")

        # 2. Parse 4-element header
        fc = float(line0.strip())               # Carrier frequency in Hz (e.g. 5.8e9)
        chirp_dur_ms = float(line1.strip())     # Chirp duration in ms (e.g. 1.0)
        nsamples_per_chirp = int(float(line2.strip())) # Fast-time samples (e.g. 128)
        bandwidth = float(line3.strip())        # Bandwidth in Hz (e.g. 400e6)

        sampling_rate = 1.0 / (chirp_dur_ms * 1e-3)  # PRF in Hz (e.g. 1000.0 Hz)

        # 3. Parse complex data string (real+imagj)
        complex_data = np.fromstring(data_str, sep="\n", dtype=np.complex128)

        # 4. Reshape into 2D Radar Matrix: (slow-time chirps, fast-time samples)
        total_samples = len(complex_data)
        num_chirps = total_samples // nsamples_per_chirp
        if num_chirps == 0:
            raise ValueError(f"Insufficient samples in {file_path}: {total_samples}")

        radar_matrix = complex_data[: num_chirps * nsamples_per_chirp].reshape(
            num_chirps, nsamples_per_chirp
        )

        # 5. Fast-time Range FFT
        range_profiles = np.fft.fft(radar_matrix, axis=1)

        # 6. Moving Target Indication (MTI) Filter: Subtract stationary clutter (mean along slow-time)
        mti_profiles = range_profiles - np.mean(range_profiles, axis=0, keepdims=True)

        # 7. Extract 1D slow-time Doppler signal
        if self.mode == "peak_bin":
            # Select range bin with maximum variance / power of moving targets
            bin_powers = np.mean(np.abs(mti_profiles) ** 2, axis=0)
            target_range_bin = int(np.argmax(bin_powers))
            slow_time_signal = mti_profiles[:, target_range_bin]
        else:
            # Range integration
            slow_time_signal = np.sum(mti_profiles, axis=1)
            target_range_bin = -1

        # 8. Parse metadata
        meta = self.parse_filename_metadata(file_path.name)
        hardware_meta = {
            "carrier_frequency_hz": fc,
            "chirp_duration_ms": chirp_dur_ms,
            "fast_time_samples_per_chirp": nsamples_per_chirp,
            "bandwidth_hz": bandwidth,
            "raw_chirp_count": num_chirps,
            "selected_range_bin": target_range_bin,
            "extraction_mode": self.mode,
            "file_source": str(file_path),
            "repetition": meta["repetition"],
        }

        return RadarSample(
            signal=slow_time_signal,
            sampling_rate=sampling_rate,
            activity_label=meta["activity_label"],
            num_samples=len(slow_time_signal),
            subject_id=meta["subject_id"],
            recording_id=file_path.stem,
            metadata=hardware_meta,
        )
