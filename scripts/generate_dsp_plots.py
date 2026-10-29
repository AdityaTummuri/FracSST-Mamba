"""Generate publication-quality DSP comparison graphs for README documentation.

Executes the exact mathematical DSP pipeline (FrFT + SST + FracSST) and produces:
  1. assets/dsp_comparison_dashboard.png (6-panel diagnostic dashboard)
  2. assets/stft_vs_fracsst_comparison.png (Direct high-resolution Before vs After)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dsp.frft import frft, find_optimal_alpha
from dsp.synchrosqueezing import sst_1d, fracsst_1d
from data.synthetic_fixtures import create_synthetic_chirp_signal, create_synthetic_radar_sample


def generate_dsp_comparison_plots():
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    fs = 1000.0
    duration = 1.5
    signal = create_synthetic_chirp_signal(fs=fs, duration=duration, f_start=20.0, chirp_rate=140.0, micro_freq=2.5, micro_amp=18.0)
    N = len(signal)
    t_sig = np.linspace(0.0, duration, N, endpoint=False)

    n_fft = 256
    hop_length = 64

    # 1. Classical STFT
    spec_sst_raw, freq_bins, time_axis = sst_1d(signal, fs=fs, n_fft=n_fft, hop_length=hop_length)

    # Compute raw STFT magnitude directly with same windowing for exact baseline comparison
    pad_len = n_fft // 2
    sig_pad = np.pad(signal, (pad_len, pad_len), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(sig_pad, n_fft)[::hop_length]
    w = np.hamming(n_fft)
    stft_mag = np.abs(np.fft.fftshift(np.fft.fft(frames * w, axis=1), axes=1)).T

    # 2. Optimal Alpha Search via Rényi Entropy
    alpha_grid = np.linspace(0.0, 1.5, 31)
    entropies = []
    min_entropy = float("inf")
    best_alpha = 0.0

    for a in alpha_grid:
        f_rot = frft(signal, a)
        f_p = np.pad(f_rot, (pad_len, pad_len), mode="constant")
        fr = np.lib.stride_tricks.sliding_window_view(f_p, n_fft)[::hop_length]
        smag = np.abs(np.fft.fft(fr * w, axis=1)) ** 2
        tot_e = np.sum(smag)
        if tot_e < 1e-12:
            entropies.append(float("inf"))
            continue
        P = smag / tot_e
        p3 = np.sum(P**3)
        ent = -0.5 * np.log(p3 + 1e-12)
        entropies.append(ent)
        if ent < min_entropy:
            min_entropy = ent
            best_alpha = float(a)

    entropies = np.array(entropies)

    # 3. Standard SST (alpha = 0)
    sst_spec, _, _ = sst_1d(signal, fs=fs, n_fft=n_fft, hop_length=hop_length)

    # 4. Fractional SST (FracSST with optimal alpha)
    fracsst_spec, _, _ = fracsst_1d(signal, fs=fs, n_fft=n_fft, hop_length=hop_length, alpha=best_alpha)

    # Decibel conversions with 45 dB dynamic range
    def to_db(M):
        norm_m = np.abs(M) / (np.max(np.abs(M)) + 1e-12)
        return 20.0 * np.log10(np.clip(norm_m, 1e-3, 1.0))

    stft_db = to_db(stft_mag)
    sst_db = to_db(sst_spec)
    fracsst_db = to_db(fracsst_spec)
    clim = (-40, 0)

    # Quantitative metrics
    def calc_renyi(S):
        P = (S**2) / (np.sum(S**2) + 1e-12)
        return -0.5 * np.log(np.sum(P**3) + 1e-12)

    def calc_shannon(S):
        P = (S**2) / (np.sum(S**2) + 1e-12)
        return -np.sum(P * np.log(P + 1e-12))

    def calc_ecm(S):
        return np.sum(S**4) / ((np.sum(S**2))**2 + 1e-12)

    r_stft = calc_renyi(stft_mag)
    r_sst = calc_renyi(sst_spec)
    r_fracsst = calc_renyi(fracsst_spec)

    sh_stft = calc_shannon(stft_mag)
    sh_sst = calc_shannon(sst_spec)
    sh_fracsst = calc_shannon(fracsst_spec)

    ecm_stft = calc_ecm(stft_mag)
    ecm_sst = calc_ecm(sst_spec)
    ecm_fracsst = calc_ecm(fracsst_spec)

    print("===============================================================")
    print(f"STFT:    R3 = {r_stft:.4f}, Shannon = {sh_stft:.4f}, ECM = {ecm_stft:.2e}")
    print(f"SST:     R3 = {r_sst:.4f}, Shannon = {sh_sst:.4f}, ECM = {ecm_sst:.2e}")
    print(f"FracSST: R3 = {r_fracsst:.4f}, Shannon = {sh_fracsst:.4f}, ECM = {ecm_fracsst:.2e}")
    print(f"Optimal Alpha: {best_alpha:.4f}")
    print("===============================================================")

    # -----------------------------------------------------------------
    # FIGURE 1: 6-Panel Diagnostic Dashboard
    # -----------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": "#445566",
        "axes.linewidth": 1.0,
    })

    fig1 = plt.figure(figsize=(15, 9), facecolor="#0e1117")
    gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.32, wspace=0.28)

    col_bg = "#151a24"
    col_text = "#e6edf3"
    col_accent = "#00d4ff"
    col_highlight = "#ff4757"

    # Panel 1: Raw Radar I/Q Signal
    ax1 = fig1.add_subplot(gs[0, 0], facecolor=col_bg)
    ax1.plot(t_sig, np.real(signal), color="#00d4ff", lw=1.2, label="In-Phase (I)")
    ax1.plot(t_sig, np.imag(signal), color="#ffa502", lw=1.0, alpha=0.85, label="Quadrature (Q)")
    ax1.set_title("(A) Raw Radar I/Q Signal", color=col_accent, fontweight="bold", fontsize=12)
    ax1.set_xlabel("Time (s)", color=col_text)
    ax1.set_ylabel("Amplitude", color=col_text)
    ax1.tick_params(colors=col_text)
    ax1.grid(True, color="#2c3e50", alpha=0.5, linestyle="--")
    ax1.legend(loc="upper right", facecolor="#1a2234", edgecolor="none", labelcolor=col_text, fontsize=8)
    ax1.set_xlim(0, duration)

    # Panel 2: Rényi Entropy vs Alpha Search
    ax2 = fig1.add_subplot(gs[0, 1], facecolor=col_bg)
    ax2.plot(alpha_grid, entropies, "-o", color="#a55eea", lw=1.8, markersize=4, label="Rényi R3 Profile")
    ax2.plot(best_alpha, min_entropy, "*", color=col_highlight, markersize=12, label=f"Optimal α* = {best_alpha:.2f}")
    ax2.set_title(f"(B) Optimal Order Search (α* = {best_alpha:.2f})", color=col_accent, fontweight="bold", fontsize=12)
    ax2.set_xlabel("Fractional Order α", color=col_text)
    ax2.set_ylabel("Order-3 Rényi Entropy R3", color=col_text)
    ax2.tick_params(colors=col_text)
    ax2.grid(True, color="#2c3e50", alpha=0.5, linestyle="--")
    ax2.legend(loc="upper right", facecolor="#1a2234", edgecolor="none", labelcolor=col_text, fontsize=8)

    # Panel 3: Spectral Slice Cross-Section (Peak Resolution Comparison)
    ax3 = fig1.add_subplot(gs[0, 2], facecolor=col_bg)
    peak_frame = len(time_axis) // 2
    slice_stft = stft_mag[:, peak_frame] / np.max(stft_mag[:, peak_frame])
    slice_sst = sst_spec[:, peak_frame] / np.max(sst_spec[:, peak_frame])
    slice_frac = fracsst_spec[:, peak_frame] / np.max(fracsst_spec[:, peak_frame])

    ax3.plot(freq_bins, slice_stft, "--", color="#747d8c", lw=1.2, label="STFT (Blurred)")
    ax3.plot(freq_bins, slice_sst, color="#2ed573", lw=1.5, label="Standard SST")
    ax3.plot(freq_bins, slice_frac, color=col_highlight, lw=2.0, label="FracSST (Sharp)")
    ax3.set_title(f"(C) Spectral Slice @ t = {time_axis[peak_frame]:.2f}s", color=col_accent, fontweight="bold", fontsize=12)
    ax3.set_xlabel("Doppler Frequency (Hz)", color=col_text)
    ax3.set_ylabel("Normalized Energy", color=col_text)
    ax3.tick_params(colors=col_text)
    ax3.grid(True, color="#2c3e50", alpha=0.5, linestyle="--")
    ax3.set_xlim(-200, 200)
    ax3.legend(loc="upper right", facecolor="#1a2234", edgecolor="none", labelcolor=col_text, fontsize=8)

    # Panel 4: STFT Spectrogram
    ax4 = fig1.add_subplot(gs[1, 0], facecolor=col_bg)
    im4 = ax4.imshow(stft_db, aspect="auto", origin="lower", cmap="inferno",
                     extent=[time_axis[0], time_axis[-1], freq_bins[0], freq_bins[-1]],
                     vmin=clim[0], vmax=clim[1])
    ax4.set_title(f"(D) Conventional STFT (R3 = {r_stft:.2f})", color=col_accent, fontweight="bold", fontsize=12)
    ax4.set_xlabel("Time (s)", color=col_text)
    ax4.set_ylabel("Frequency (Hz)", color=col_text)
    ax4.tick_params(colors=col_text)
    ax4.set_ylim(-250, 250)

    # Panel 5: Standard SST Spectrogram
    ax5 = fig1.add_subplot(gs[1, 1], facecolor=col_bg)
    im5 = ax5.imshow(sst_db, aspect="auto", origin="lower", cmap="inferno",
                     extent=[time_axis[0], time_axis[-1], freq_bins[0], freq_bins[-1]],
                     vmin=clim[0], vmax=clim[1])
    ax5.set_title(f"(E) Standard SST (α = 0, R3 = {r_sst:.2f})", color=col_accent, fontweight="bold", fontsize=12)
    ax5.set_xlabel("Time (s)", color=col_text)
    ax5.set_ylabel("Frequency (Hz)", color=col_text)
    ax5.tick_params(colors=col_text)
    ax5.set_ylim(-250, 250)

    # Panel 6: Proposed FracSST Spectrogram
    ax6 = fig1.add_subplot(gs[1, 2], facecolor=col_bg)
    im6 = ax6.imshow(fracsst_db, aspect="auto", origin="lower", cmap="inferno",
                     extent=[time_axis[0], time_axis[-1], freq_bins[0], freq_bins[-1]],
                     vmin=clim[0], vmax=clim[1])
    ax6.set_title(f"(F) Proposed FracSST (α* = {best_alpha:.2f}, R3 = {r_fracsst:.2f})",
                  color="#2ed573", fontweight="bold", fontsize=12)
    ax6.set_xlabel("Time (s)", color=col_text)
    ax6.set_ylabel("Frequency (Hz)", color=col_text)
    ax6.tick_params(colors=col_text)
    ax6.set_ylim(-250, 250)

    cb = fig1.colorbar(im6, ax=[ax4, ax5, ax6], location="bottom", pad=0.12, fraction=0.03, aspect=40)
    cb.set_label("Normalized Power Spectral Density (dB)", color=col_text, fontsize=10)
    cb.ax.tick_params(colors=col_text)

    fig1.suptitle("FracSST-Mamba Stage 1 DSP: Physics-Informed Micro-Doppler Feature Extraction",
                  color="white", fontsize=15, fontweight="bold", y=0.98)

    dashboard_path = assets_dir / "dsp_comparison_dashboard.png"
    fig1.savefig(dashboard_path, dpi=300, bbox_inches="tight", facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f"Saved: {dashboard_path}")

    # -----------------------------------------------------------------
    # FIGURE 2: Direct High-Resolution Before vs After Comparison
    # -----------------------------------------------------------------
    fig2, (b_ax1, b_ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0e1117")

    im_b1 = b_ax1.imshow(stft_db, aspect="auto", origin="lower", cmap="magma",
                         extent=[time_axis[0], time_axis[-1], freq_bins[0], freq_bins[-1]],
                         vmin=clim[0], vmax=clim[1])
    b_ax1.set_title("CONVENTIONAL STFT SPECTROGRAM\n(Severe Heisenberg-Gabor Smearing)",
                    color="#ff6b81", fontweight="bold", fontsize=13, pad=10)
    b_ax1.set_xlabel("Time (s)", color=col_text, fontweight="bold", fontsize=11)
    b_ax1.set_ylabel("Doppler Frequency (Hz)", color=col_text, fontweight="bold", fontsize=11)
    b_ax1.tick_params(colors=col_text)
    b_ax1.set_ylim(-250, 250)
    b_ax1.set_facecolor(col_bg)

    im_b2 = b_ax2.imshow(fracsst_db, aspect="auto", origin="lower", cmap="magma",
                         extent=[time_axis[0], time_axis[-1], freq_bins[0], freq_bins[-1]],
                         vmin=clim[0], vmax=clim[1])
    b_ax2.set_title(f"PROPOSED FracSST SPECTROGRAM (α* = {best_alpha:.2f})\n(Dechirped & Synchrosqueezed Energy Ridges)",
                    color="#2ed573", fontweight="bold", fontsize=13, pad=10)
    b_ax2.set_xlabel("Time (s)", color=col_text, fontweight="bold", fontsize=11)
    b_ax2.set_ylabel("Doppler Frequency (Hz)", color=col_text, fontweight="bold", fontsize=11)
    b_ax2.tick_params(colors=col_text)
    b_ax2.set_ylim(-250, 250)
    b_ax2.set_facecolor(col_bg)

    cb2 = fig2.colorbar(im_b2, ax=[b_ax1, b_ax2], location="bottom", pad=0.18, fraction=0.045, aspect=50)
    cb2.set_label("Power Spectral Density (dB)", color=col_text, fontsize=11)
    cb2.ax.tick_params(colors=col_text)

    fig2.suptitle("Direct Visual Resolution Comparison: Human Walking Micro-Doppler Signature",
                  color="white", fontsize=15, fontweight="bold", y=0.99)

    direct_path = assets_dir / "stft_vs_fracsst_comparison.png"
    fig2.savefig(direct_path, dpi=300, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"Saved: {direct_path}")


if __name__ == "__main__":
    generate_dsp_comparison_plots()
