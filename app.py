"""FracSST-Mamba Streamlit Dashboard — Simulated Live Radar HAR Feed.

Provides an interactive demonstration of the full FracSST-Mamba pipeline:
Stage 1 (DSP) spectrogram sharpening visualised side-by-side with
Stage 2 (Mamba AI) real-time activity classification and temporal segmentation.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from models.da_seg import DASegModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = ["Walk", "Sit", "Fall"]
CLASS_ICONS: Dict[int, str] = {0: "🚶", 1: "🪑", 2: "🚨"}
CLASS_COLORS: Dict[int, str] = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}
WEIGHTS_PATH = Path("weights/best_model.pth")
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Model Loading (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> Tuple[DASegModel, dict]:
    """Load the trained DASegModel from the best checkpoint.

    Returns:
        Tuple of (model, config_dict).
    """
    if not WEIGHTS_PATH.exists():
        st.error(
            f"Model checkpoint not found at `{WEIGHTS_PATH}`. "
            "Please run `python train.py` first."
        )
        st.stop()

    device = torch.device("cpu")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})

    model = DASegModel(
        in_channels=1,
        d_model=model_cfg.get("d_model", 128),
        n_layers=model_cfg.get("n_layers", 4),
        num_classes=model_cfg.get("num_classes", 3),
        dropout=0.0,  # No dropout at inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


@st.cache_data
def load_processed_samples() -> List[Path]:
    """Discover all preprocessed .pt files sorted by name.

    Returns:
        Sorted list of .pt file paths.
    """
    if not PROCESSED_DIR.exists() or len(list(PROCESSED_DIR.glob("*.pt"))) == 0:
        st.error(
            f"No preprocessed data in `{PROCESSED_DIR}`. "
            "Please run `python preprocess.py` first."
        )
        st.stop()
    return sorted(list(PROCESSED_DIR.glob("*.pt")))


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------
def create_spectrogram_figure(
    raw_signal: np.ndarray,
    sharpened_spec: np.ndarray,
    optimal_alpha: float,
    sample_name: str,
    fs: float = 1000.0,
    n_fft: int = 256,
) -> plt.Figure:
    """Create a side-by-side Raw STFT vs FracSST Sharpened spectrogram figure.

    Args:
        raw_signal: Original raw 1D radar signal loaded from data/raw/.
        sharpened_spec: 2D FracSST output spectrogram (n_freq, n_frames).
        optimal_alpha: The fractional order α selected by Rényi entropy.
        sample_name: Display name of the current sample.
        fs: Sampling frequency.
        n_fft: FFT window size.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=(14, 5), facecolor="#0e1117")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # --- Left Panel: Raw STFT ---
    ax1 = fig.add_subplot(gs[0])
    if raw_signal is not None and len(raw_signal) > 0:
        import scipy.signal as sig
        w = sig.windows.hamming(n_fft)
        f_stft, t_stft, Zxx_raw = sig.stft(
            raw_signal, fs=fs, window=w, nperseg=n_fft, noverlap=n_fft - n_fft // 4
        )
        raw_mag = np.abs(Zxx_raw)
        raw_db = 10 * np.log10(raw_mag + 1e-12)
        im1 = ax1.imshow(
            raw_db, aspect="auto", origin="lower", cmap="inferno",
            extent=[0, t_stft[-1], 0, f_stft[-1]],
        )
        fig.colorbar(im1, ax=ax1, label="Power (dB)", shrink=0.85)
    ax1.set_title("Raw STFT Spectrogram", fontsize=13, fontweight="bold", color="white")
    ax1.set_xlabel("Time (s)", fontsize=11, color="white")
    ax1.set_ylabel("Frequency (Hz)", fontsize=11, color="white")
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#1a1a2e")

    # --- Right Panel: FracSST Sharpened ---
    ax2 = fig.add_subplot(gs[1])
    sharp_db = 10 * np.log10(np.abs(sharpened_spec) + 1e-12)
    im2 = ax2.imshow(sharp_db, aspect="auto", origin="lower", cmap="magma")
    fig.colorbar(im2, ax=ax2, label="Power (dB)", shrink=0.85)
    ax2.set_title(
        f"FracSST Sharpened (α={optimal_alpha:.3f})",
        fontsize=13, fontweight="bold", color="white",
    )
    ax2.set_xlabel("Time Frame", fontsize=11, color="white")
    ax2.set_ylabel("Frequency Bin", fontsize=11, color="white")
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#1a1a2e")

    fig.suptitle(
        f"Stage 1 DSP — {sample_name}",
        fontsize=15, fontweight="bold", color="#00d4ff", y=1.02,
    )
    fig.patch.set_facecolor("#0e1117")

    return fig


def create_segmentation_figure(
    seg_probs: np.ndarray,
    pred_labels: np.ndarray,
    true_label: int,
    sample_name: str,
) -> plt.Figure:
    """Create a temporal segmentation probability bar chart.

    Args:
        seg_probs: Softmax probabilities (n_frames, num_classes).
        pred_labels: Predicted class per frame (n_frames,).
        true_label: Ground truth sequence label.
        sample_name: Display name.

    Returns:
        Matplotlib Figure.
    """
    n_frames = seg_probs.shape[0]
    fig, ax = plt.subplots(figsize=(14, 3), facecolor="#0e1117")

    frames = np.arange(n_frames)
    bottom = np.zeros(n_frames)
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for cls_idx in range(seg_probs.shape[1]):
        ax.bar(
            frames, seg_probs[:, cls_idx], bottom=bottom, width=0.8,
            color=colors[cls_idx % len(colors)], alpha=0.85,
            label=CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}",
        )
        bottom += seg_probs[:, cls_idx]

    ax.set_xlim(-0.5, n_frames - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time Frame", fontsize=11, color="white")
    ax.set_ylabel("Probability", fontsize=11, color="white")
    ax.set_title(
        f"Frame-Level Segmentation — Ground Truth: {CLASS_NAMES[true_label]}",
        fontsize=13, fontweight="bold", color="white",
    )
    ax.legend(loc="upper right", fontsize=9, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0e1117")
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the FracSST-Mamba Streamlit dashboard."""

    st.set_page_config(
        page_title="FracSST-Mamba Dashboard",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Custom CSS for dark-themed premium look ----
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 900;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6bcb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            font-size: 1.05rem;
            color: #8899aa;
            margin-bottom: 1.5rem;
        }
        .prediction-box {
            border-radius: 16px;
            padding: 2rem 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .prediction-box h1 {
            font-size: 3.5rem;
            margin: 0;
            font-weight: 900;
        }
        .prediction-box p {
            font-size: 1.1rem;
            margin-top: 0.5rem;
            opacity: 0.85;
        }
        .metric-row {
            display: flex;
            justify-content: space-around;
            gap: 1rem;
            margin-top: 1rem;
        }
        .metric-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            text-align: center;
            flex: 1;
        }
        .metric-card .label {
            font-size: 0.85rem;
            color: #8899aa;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e1117 0%, #16213e 100%);
        }
        .sidebar-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #00d4ff;
            margin-bottom: 0.5rem;
        }
        .status-idle {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 2px solid #3498db;
            color: #3498db;
        }
        .status-walk {
            background: linear-gradient(135deg, #0d3320, #145a32);
            border: 2px solid #2ecc71;
            color: #2ecc71;
        }
        .status-sit {
            background: linear-gradient(135deg, #0a2a4a, #154360);
            border: 2px solid #3498db;
            color: #3498db;
        }
        .status-fall {
            background: linear-gradient(135deg, #4a0a0a, #78281f);
            border: 2px solid #e74c3c;
            color: #e74c3c;
            animation: pulse 0.8s ease-in-out infinite alternate;
        }
        @keyframes pulse {
            from { box-shadow: 0 0 15px rgba(231,76,60,0.3); }
            to { box-shadow: 0 0 35px rgba(231,76,60,0.7); }
        }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    st.markdown('<div class="main-title">📡 FracSST-Mamba</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Physics-Aware Radar Human Activity Recognition — Live Demo Dashboard</div>',
        unsafe_allow_html=True,
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Control Panel</div>', unsafe_allow_html=True)
        st.divider()

        simulate_btn = st.button(
            "🛰️  Simulate Live Radar Feed",
            use_container_width=True,
            type="primary",
        )

        st.divider()
        feed_speed = st.slider("Feed Speed (seconds per frame)", 0.5, 3.0, 1.5, 0.5)

        st.divider()
        st.markdown("**Pipeline Architecture**")
        st.markdown(
            "1. **Stage 1 — DSP**: FracSST sharpens raw radar micro-Doppler spectrograms "
            "using Fractional Fourier Transform + Synchrosqueezing.\n"
            "2. **Stage 2 — AI**: Mamba SSM classifies the activity "
            "and segments temporal frames."
        )

        st.divider()
        st.markdown("**Model Info**")
        model, config = load_model()
        model_cfg = config.get("model", {})
        total_params = sum(p.numel() for p in model.parameters())
        st.markdown(
            f"- **Architecture**: DASegModel + MambaBlock\n"
            f"- **Parameters**: {total_params:,}\n"
            f"- **Classes**: {', '.join(CLASS_NAMES)}\n"
            f"- **d_model**: {model_cfg.get('d_model', 128)}\n"
            f"- **Mamba Layers**: {model_cfg.get('n_layers', 4)}"
        )

    # ---- Main Content Area ----
    col_dsp, col_ai = st.columns([3, 2], gap="large")

    with col_dsp:
        st.markdown("### 🔬 Stage 1 — DSP Front-End")
        spec_placeholder = st.empty()
        seg_placeholder = st.empty()

    with col_ai:
        st.markdown("### 🧠 Stage 2 — Mamba AI Prediction")
        prediction_placeholder = st.empty()
        metrics_placeholder = st.empty()
        progress_placeholder = st.empty()

    # ---- Default Idle State ----
    if not simulate_btn:
        with spec_placeholder.container():
            st.info("👈 Click **Simulate Live Radar Feed** in the sidebar to start the demo.")

        prediction_placeholder.markdown(
            '<div class="prediction-box status-idle">'
            '<h1>⏳ IDLE</h1>'
            '<p>Waiting for radar feed…</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ---- Simulation Loop ----
    model, config = load_model()
    sample_files = load_processed_samples()

    progress_bar = progress_placeholder.progress(0, text="Initialising radar feed…")

    for i, sample_path in enumerate(sample_files):
        progress_frac = (i + 1) / len(sample_files)
        progress_bar.progress(progress_frac, text=f"Processing sample {i + 1}/{len(sample_files)}")

        # Load preprocessed tensor
        data = torch.load(sample_path, weights_only=True)
        spectrogram = data["spectrogram"]          # (256, 16)
        true_label = int(data["label"].item())
        optimal_alpha = float(data["optimal_alpha"])
        sample_name = sample_path.stem.replace("_", " ").title()

        # Attempt to load matching raw signal for comparison plot
        raw_file = RAW_DIR / f"{sample_path.stem}.npy"
        raw_signal = np.load(raw_file) if raw_file.exists() else None

        # ---- Stage 1: DSP Visualization ----
        spec_fig = create_spectrogram_figure(
            raw_signal=raw_signal,
            sharpened_spec=spectrogram.numpy(),
            optimal_alpha=optimal_alpha,
            sample_name=sample_name,
        )
        spec_placeholder.pyplot(spec_fig, use_container_width=True)
        plt.close(spec_fig)

        # ---- Stage 2: Model Inference ----
        input_tensor = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 16)
        with torch.no_grad():
            cls_logits, seg_logits = model(input_tensor, return_segmentation=True)

        cls_probs = F.softmax(cls_logits, dim=-1).squeeze(0).numpy()
        seg_probs = F.softmax(seg_logits, dim=-1).squeeze(0).numpy()  # (T, C)
        pred_class = int(np.argmax(cls_probs))
        confidence = float(cls_probs[pred_class]) * 100.0
        pred_name = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else f"Class {pred_class}"
        pred_icon = CLASS_ICONS.get(pred_class, "❓")

        # Pick CSS class for visual styling
        status_css = ["status-walk", "status-sit", "status-fall"][pred_class] if pred_class < 3 else "status-idle"
        alert_label = f"{pred_icon} {pred_name.upper()}"
        if pred_class == 2:
            alert_label = "🚨 FALL DETECTED"

        # ---- Update Prediction Card ----
        prediction_placeholder.markdown(
            f'<div class="prediction-box {status_css}">'
            f'<h1>{alert_label}</h1>'
            f'<p>Confidence: {confidence:.1f}% &nbsp;|&nbsp; Ground Truth: {CLASS_NAMES[true_label]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ---- Update Metric Cards ----
        metrics_placeholder.markdown(
            f'<div class="metric-row">'
            f'  <div class="metric-card">'
            f'    <div class="label">Walk</div>'
            f'    <div class="value" style="color:#2ecc71">{cls_probs[0]*100:.1f}%</div>'
            f'  </div>'
            f'  <div class="metric-card">'
            f'    <div class="label">Sit</div>'
            f'    <div class="value" style="color:#3498db">{cls_probs[1]*100:.1f}%</div>'
            f'  </div>'
            f'  <div class="metric-card">'
            f'    <div class="label">Fall</div>'
            f'    <div class="value" style="color:#e74c3c">{cls_probs[2]*100:.1f}%</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ---- Update Segmentation Chart ----
        seg_fig = create_segmentation_figure(
            seg_probs=seg_probs,
            pred_labels=np.argmax(seg_probs, axis=-1),
            true_label=true_label,
            sample_name=sample_name,
        )
        seg_placeholder.pyplot(seg_fig, use_container_width=True)
        plt.close(seg_fig)

        time.sleep(feed_speed)

    progress_bar.progress(1.0, text="✅ Radar feed simulation complete.")
    st.balloons()


if __name__ == "__main__":
    main()
