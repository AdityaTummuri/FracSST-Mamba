"""Comprehensive unit tests for Stage 1 DSP pipeline (FrFT, SST, FrSST, axes, invariance).

Runs without requiring PyTorch, Mamba, or any deep learning dependencies.
Can be run via: python -m unittest test_dsp.py
"""

import unittest
import numpy as np

from dsp.frft import frft, find_optimal_alpha
from dsp.synchrosqueezing import sst_1d, fracsst_1d, frsst, FrSSTResult
from data.synthetic_fixtures import create_synthetic_chirp_signal


class TestFrFT(unittest.TestCase):
    """Test suite for 1D Fractional Fourier Transform."""

    def setUp(self) -> None:
        self.fs = 1000.0
        self.signal = create_synthetic_chirp_signal(fs=self.fs, duration=0.5)

    def test_identity_order_zero(self) -> None:
        """FrFT with order a=0 must return the identical signal."""
        out = frft(self.signal, a=0.0)
        np.testing.assert_allclose(out, self.signal, atol=1e-7)

    def test_fourier_order_one(self) -> None:
        """FrFT with order a=1 must match scaled and shifted FFT."""
        out = frft(self.signal, a=1.0)
        N = len(self.signal)
        expected = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.signal))) / np.sqrt(N)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_inverse_fourier_order_minus_one(self) -> None:
        """FrFT with order a=-1 must match scaled and shifted IFFT."""
        out = frft(self.signal, a=-1.0)
        N = len(self.signal)
        expected = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(self.signal))) * np.sqrt(N)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_energy_preservation_parseval(self) -> None:
        """FrFT across arbitrary fractional orders must preserve L2 energy norm."""
        orig_energy = np.linalg.norm(self.signal)
        for order in [0.25, 0.5, 0.75, 1.2, -0.3, -1.5]:
            out = frft(self.signal, a=order)
            out_energy = np.linalg.norm(out)
            self.assertAlmostEqual(
                orig_energy,
                out_energy,
                places=5,
                msg=f"Parseval norm failed for order {order}: {orig_energy} vs {out_energy}",
            )

    def test_deterministic_behavior(self) -> None:
        """FrFT must produce identical outputs on repeated runs."""
        out1 = frft(self.signal, a=0.6)
        out2 = frft(self.signal, a=0.6)
        np.testing.assert_array_equal(out1, out2)

    def test_invalid_input_shape(self) -> None:
        """FrFT must reject non-1D arrays."""
        with self.assertRaises(ValueError):
            frft(np.ones((10, 10)), a=0.5)


class TestSST(unittest.TestCase):
    """Test suite for standard 1D Synchrosqueezing Transform (sst_1d)."""

    def setUp(self) -> None:
        self.fs = 1000.0
        self.signal = create_synthetic_chirp_signal(fs=self.fs, duration=0.5)

    def test_sst_shapes_and_types(self) -> None:
        """Check return shapes, axes, and float dtypes."""
        n_fft = 128
        hop_length = 32
        spec, freqs, times = sst_1d(self.signal, fs=self.fs, n_fft=n_fft, hop_length=hop_length)

        self.assertEqual(spec.ndim, 2)
        self.assertEqual(len(freqs), n_fft)
        self.assertEqual(spec.shape[0], n_fft)
        self.assertEqual(spec.shape[1], len(times))
        self.assertFalse(np.isnan(spec).any(), "NaN values found in SST spectrum")
        self.assertFalse(np.isinf(spec).any(), "Inf values found in SST spectrum")
        self.assertTrue((spec >= 0).all(), "Energy reassignment values must be non-negative")

    def test_invalid_parameters(self) -> None:
        """SST must raise ValueError for invalid inputs or negative parameters."""
        with self.assertRaises(ValueError):
            sst_1d(np.array([]), fs=1000.0)
        with self.assertRaises(ValueError):
            sst_1d(self.signal, fs=-100.0)
        with self.assertRaises(ValueError):
            sst_1d(self.signal, fs=1000.0, n_fft=0)


class TestFrSSTPipeline(unittest.TestCase):
    """Test suite for FrSST high-level interface and optimal alpha search."""

    def setUp(self) -> None:
        self.fs = 1000.0
        self.signal = create_synthetic_chirp_signal(fs=self.fs, duration=0.5)

    def test_optimal_alpha_search(self) -> None:
        """Check that optimal alpha search returns a valid float within search bounds."""
        search_range = np.array([0.0, 0.25, 0.5, 0.75])
        alpha = find_optimal_alpha(self.signal, search_range=search_range)
        self.assertIn(alpha, search_range)

    def test_frsst_high_level_api(self) -> None:
        """Test clean frsst API returns FrSSTResult with correct fields and metadata."""
        n_fft = 128
        hop_length = 32
        result = frsst(
            self.signal,
            sampling_rate=self.fs,
            n_fft=n_fft,
            hop_length=hop_length,
            alpha=0.5,
        )

        self.assertIsInstance(result, FrSSTResult)
        self.assertEqual(result.alpha, 0.5)
        self.assertEqual(result.representation.shape[0], n_fft)
        self.assertEqual(len(result.frequency_axis), n_fft)
        self.assertEqual(len(result.time_axis), result.representation.shape[1])
        self.assertEqual(result.metadata["sampling_rate"], self.fs)
        self.assertEqual(result.metadata["n_fft"], n_fft)

    def test_backward_compatibility_fracsst_1d(self) -> None:
        """Ensure existing fracsst_1d signature still works without breaking."""
        spec, freqs, alpha = fracsst_1d(
            self.signal,
            fs=self.fs,
            n_fft=128,
            hop_length=32,
            alpha=0.4,
        )
        self.assertEqual(alpha, 0.4)
        self.assertEqual(spec.shape[0], 128)
        self.assertEqual(len(freqs), 128)


if __name__ == "__main__":
    unittest.main()
