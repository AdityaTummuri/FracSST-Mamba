#!/usr/bin/env python3
"""Standalone DSP verification and demonstration script.

Runs completely independently of PyTorch, Mamba, CUDA, and deep learning dependencies.
Processes a radar test signal, computes FrFT, SST, and FrSST representations,
prints mathematical metadata, and saves a side-by-side comparison figure.

Usage:
    python scripts/run_dsp_demo.py
    python scripts/run_dsp_demo.py --no-plot
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dsp.frft import frft, find_optimal_alpha
from dsp.synchrosqueezing import sst_1d, frsst
from data.synthetic_fixtures import create_synthetic_chirp_signal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone DSP Demonstration for FracSST-Mamba (Zero PyTorch/Mamba dependency)."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Signal duration in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="Radar sampling frequency in Hz (default: 1000.0).",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=256,
        help="FFT window size (default: 256).",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=64,
        help="Hop size between frames (default: 64).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="assets/dsp_demo_comparison.png",
        help="Destination path for comparison plot.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("      FracSST-Mamba: Standalone Radar DSP Pipeline Demo      ")
    print("=" * 65)

    # 1. Generate or load test signal
    print(f"\n[1] Generating 1D radar micro-Doppler test signal...")
    signal = create_synthetic_chirp_signal(
        fs=args.fs,
        duration=args.duration,
        f_start=30.0,
        chirp_rate=120.0,
        micro_freq=3.0,
        micro_amp=15.0,
    )
    n_samples = len(signal)
    l2_norm = np.linalg.norm(signal)
    print(f"    - Duration:         {args.duration:.2f} s")
    print(f"    - Sampling Rate:    {args.fs:.1f} Hz")
    print(f"    - Number of samples: {n_samples}")
    print(f"    - Data Type:        {signal.dtype}")
    print(f"    - Initial L2 Norm:  {l2_norm:.4f}")

    # 2. Search for optimal fractional order alpha
    print(f"\n[2] Scanning fractional orders alpha in [0.0, 1.5] for minimum Rényi entropy...")
    search_grid = np.linspace(0.0, 1.5, 31)
    optimal_alpha = find_optimal_alpha(
        signal,
        search_range=search_grid,
        n_fft=args.n_fft,
        hop_length=args.hop,
    )
    print(f"    - Candidate orders: {len(search_grid)} evaluated")
    print(f"    - Optimal alpha*:   {optimal_alpha:.4f}")

    # 3. Fractional Fourier Transform (FrFT)
    print(f"\n[3] Computing Fractional Fourier Transform (FrFT)...")
    rotated_sig = frft(signal, a=optimal_alpha)
    rot_l2_norm = np.linalg.norm(rotated_sig)
    norm_delta = abs(l2_norm - rot_l2_norm)
    print(f"    - Rotated L2 Norm:  {rot_l2_norm:.4f} (Parseval delta: {norm_delta:.2e})")

    # 4. Standard SST (Baseline)
    print(f"\n[4] Computing Standard Synchrosqueezing Transform (SST baseline, alpha=0)...")
    sst_spec, freq_axis, time_axis = sst_1d(
        signal,
        fs=args.fs,
        n_fft=args.n_fft,
        hop_length=args.hop,
    )
    print(f"    - SST Matrix Shape: {sst_spec.shape} (Freq Bins x Time Frames)")
    print(f"    - Frequency Range:  [{freq_axis[0]:.1f} Hz, {freq_axis[-1]:.1f} Hz]")
    print(f"    - Time Span:        [{time_axis[0]:.3f} s, {time_axis[-1]:.3f} s]")

    # 5. Clean FrSST High-Level Call
    print(f"\n[5] Computing Fractional Synchrosqueezing Transform (FrSST)...")
    frsst_result = frsst(
        signal,
        sampling_rate=args.fs,
        n_fft=args.n_fft,
        hop_length=args.hop,
        alpha=optimal_alpha,
    )
    print(f"    - FrSST Matrix Shape: {frsst_result.representation.shape}")
    print(f"    - Alpha used:         {frsst_result.alpha:.4f}")
    print(f"    - Max Energy Peak:    {frsst_result.representation.max():.2e}")
    print(f"    - Metadata:           {frsst_result.metadata}")

    # 6. Optional visualization
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            print(f"\n[6] Rendering comparison figure...")
            save_dest = REPO_ROOT / args.save_path
            save_dest.parent.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#121212")

            # Standard SST
            sst_db = 10 * np.log10(sst_spec + 1e-12)
            im0 = axes[0].imshow(
                sst_db,
                aspect="auto",
                origin="lower",
                extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
                cmap="viridis",
            )
            axes[0].set_title("Standard SST (alpha = 0.0)", color="white", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("Time (s)", color="white")
            axes[0].set_ylabel("Doppler Frequency (Hz)", color="white")
            axes[0].tick_params(colors="white")
            axes[0].set_facecolor("#1e1e1e")
            fig.colorbar(im0, ax=axes[0], label="Power (dB)")

            # FrSST Sharpened
            frsst_db = 10 * np.log10(frsst_result.representation + 1e-12)
            im1 = axes[1].imshow(
                frsst_db,
                aspect="auto",
                origin="lower",
                extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
                cmap="magma",
            )
            axes[1].set_title(
                f"FrSST Sharpened (optimal alpha = {optimal_alpha:.3f})",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
            axes[1].set_xlabel("Time (s)", color="white")
            axes[1].set_ylabel("Doppler Frequency (Hz)", color="white")
            axes[1].tick_params(colors="white")
            axes[1].set_facecolor("#1e1e1e")
            fig.colorbar(im1, ax=axes[1], label="Power (dB)")

            fig.suptitle(
                "Radar Micro-Doppler Time-Frequency Reassignment Comparison",
                color="cyan",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(save_dest, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"    - Comparison plot saved to: {save_dest}")
        except ImportError as err:
            print(f"    - Matplotlib not installed; skipping plot generation ({err}).")

    print("\n" + "=" * 65)
    print("DSP standalone verification completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
