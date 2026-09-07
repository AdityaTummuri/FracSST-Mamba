# FracSST-Mamba: Physics-Aware Radar Human Activity Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![DSP: Pure NumPy/SciPy](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy-013243.svg)](dsp/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b.svg)](https://streamlit.io/)

---

## Executive Summary

**FracSST-Mamba** is a physics-informed deep learning architecture designed for **Continuous-Wave (CW) and Micro-Doppler Radar Human Activity Recognition (HAR)** and temporal event segmentation.

Radar returns from dynamic human motions (walking, limb swinging, rapid falling) exhibit **non-linear, time-varying micro-Doppler frequency trajectories**. Conventional Short-Time Fourier Transforms (STFT) suffer severe spectral smearing due to Heisenberg-Gabor time-frequency uncertainty, while standard Transformers scale with quadratic complexity $\mathcal{O}(T^2)$.

FracSST-Mamba resolves these limitations through a clean two-stage decoupled paradigm:
1. **Stage 1 — Trainable-Free Physics DSP Front-End**: Rotates radar I/Q signals into the optimal fractional domain via Fractional Fourier Transform (FrFT)—selected by minimizing order-3 Rényi Entropy—and sharpens micro-Doppler ridges using Synchrosqueezing Transform (SST).
2. **Stage 2 — Dual-Attention Mamba Sequence Backbone**: Models temporal dynamics with linear complexity $\mathcal{O}(T)$ via Selective State-Space Models (SSM), jointly outputting sequence activity classifications and frame-level temporal segmentation masks.

```text
Raw Radar I/Q Signal
       ↓
[ Stage 1: DSP ]  Optimal α Search (Rényi Entropy) → FrFT Rotation → SST Reassignment
       ↓
Sharpened Micro-Doppler Representation (FrSSTResult)
       ↓
[ Stage 2: AI ]   2D Spectral Stem → Frequency Pooling → Dual Attention → Stacked Mamba SSM
       ↓
Activity Classification (Walk / Sit / Fall) & Temporal Frame Segmentation
```

---

## System Architecture

![FracSST-Mamba Architecture](assets/architecture.png)

### Stage 1: Deterministic DSP Front-End (`dsp/`)
The DSP front-end is **purely mathematical, deterministic, and free of PyTorch/CUDA dependencies**.

1. **Optimal Fractional Order Selection** (`dsp/frft.py` → `find_optimal_alpha`):
   - Evaluates candidate fractional rotation angles $\alpha \in [0, 1.5]$.
   - Quantifies time-frequency concentration using order-3 Rényi entropy:
     $$R_3(\alpha) = -\frac{1}{2} \ln \left( \sum_{i,j} P_\alpha(i, j)^3 \right), \quad P_\alpha = \frac{|Z_\alpha|^2}{\sum |Z_\alpha|^2}$$
   - Returns $\alpha^* = \arg\min_\alpha R_3(\alpha)$, aligning the transform coordinate axes with the principal acceleration chirps of human limb motions.

2. **Fractional Fourier Domain Rotation** (`dsp/frft.py` → `frft`):
   - Computes the continuous FrFT using the **Ozaktas fast chirp-multiplication algorithm** with 2× frequency oversampling and FFT-based fast convolution.
   - Enforces Parseval energy invariance ($\|F^a f\|_2 = \|f\|_2$) across all rotation orders.
   - Accurately handles modulo-4 boundary conditions ($a \in \{0, \pm 1, \pm 2\}$).

3. **Synchrosqueezing Energy Reassignment** (`dsp/synchrosqueezing.py` → `sst_1d`, `frsst`):
   - Computes STFT using both a standard window $w(t)$ and its analytical time derivative $w'(t) = \frac{dw}{dt}$.
   - Estimates instantaneous frequency offsets:
     $$\hat{\omega}(t, \omega) = \omega - \frac{1}{2\pi} \text{Im} \left( \frac{\text{STFT}_{w'}(t, \omega)}{\text{STFT}_w(t, \omega)} \right)$$
   - Reassigns spectral energy from smeared bins to instantaneous frequency ridges, producing concentrated $(N_{\text{freq}}, N_{\text{time}})$ representations.

---

### Stage 2: Dual-Attention Mamba Sequence Network (`models/`)
The neural network operates on the FrSST time-frequency representation:

```text
Input Spectrogram Tensor: [B, 1, F=256, T=16]
          │
          ▼
   2D Conv Stem (Conv2d → BN → SiLU ×2)  ──>  [B, 128, 256, 16]
          │
          ▼
Adaptive Frequency Pool (Collapse F to 1) ──> [B, T=16, d_model=128]
          │
          ▼
Dual Attention: Channel (Squeeze-Excitation) + Temporal Multi-Head Attention
          │
          ▼
Stacked Mamba SSM Blocks (×4 Layers, Selective State-Space Scan)
          │
    ┌─────┴──────────────────────────────┐
    ▼                                    ▼
Frame Segmentation Head          Global Average Pooling (over T)
[B, T=16, num_classes=3]                 │
                                         ▼
                             Classification Head [B, num_classes=3]
```

- **Mamba SSM Engine**: Features a dual-backend wrapper in `models/mamba_block.py`. Automatically utilizes native hardware-accelerated CUDA kernels via `mamba-ssm` if available, and gracefully falls back to a vectorized pure-PyTorch sequential scan (`PyTorchSelectiveSSM`) on CPU environments.

---

## Architectural Stratification

To ensure enterprise-grade modularity and future-proof dataset integration, the codebase is strictly decoupled across 6 clean layers:

| Layer | Directory | Responsibilities | Dependencies |
|---|---|---|---|
| **Data Layer** | `data/` | Standard `RadarSample` schema, `BaseRadarDatasetAdapter`, split manifests, test fixtures. | NumPy (Zero Torch in schema) |
| **Signal Processing** | `dsp/` | `frft()`, `find_optimal_alpha()`, `sst_1d()`, and `frsst()`. | Pure NumPy, SciPy |
| **Feature Layer** | `dsp/synchrosqueezing.py` | `FrSSTResult` container (reassigned matrix, time/frequency axes, metadata). | Pure NumPy |
| **Model Layer** | `models/` | `DASegModel`, `MambaBlock`, `ChannelAttention`, `TemporalAttention`. | PyTorch (optional `mamba_ssm`) |
| **Training & Evaluation** | `train.py`, `utils/` | Multi-task loss, validation loops, confusion matrices, frame mIoU. | PyTorch, PyYAML |
| **Demos & CLI** | `scripts/`, `app.py` | Standalone DSP demo script, Streamlit live radar simulation feed. | Matplotlib, Streamlit |

---

## Repository Structure

```text
FracSST-Mamba/
├── assets/
│   └── architecture.png       # End-to-end architecture schematic
├── config/
│   └── params.yaml            # Hyperparameters for DSP, network, and training
├── data/
│   ├── raw/                   # Raw radar signals (.npy, .mat, .h5, etc.)
│   ├── interim/               # Converted/filtered intermediate signals
│   ├── processed/             # FracSST preprocessed PyTorch tensors (.pt)
│   ├── metadata/              # Dataset manifests and subject-independent splits
│   ├── __init__.py
│   ├── schema.py              # Canonical RadarSample dataclass & dataset adapter ABC
│   ├── synthetic_fixtures.py  # Isolated test signals for unit & smoke testing
│   └── dataset.py             # PyTorch Dataset & DataLoader factory
├── dsp/
│   ├── __init__.py            # Clean exports of DSP routines and FrSSTResult
│   ├── frft.py                # 1D Ozaktas FrFT & Rényi Entropy grid search
│   └── synchrosqueezing.py    # sst_1d, fracsst_1d, and high-level frsst() API
├── models/
│   ├── __init__.py
│   ├── mamba_block.py         # Mamba SSM block (CUDA native + PyTorch CPU fallback)
│   └── da_seg.py              # Dual-Attention Convolutional-Projection Mamba network
├── scripts/
│   └── run_dsp_demo.py        # Standalone DSP verification (Zero PyTorch/Mamba dependency)
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # Classification metrics & frame-level segmentation mIoU
│   └── visualization.py       # Matplotlib & Seaborn spectrogram / CM plotters
├── weights/
│   └── best_model.pth         # Trained network checkpoint
├── app.py                     # Streamlit live radar feed demonstration dashboard
├── preprocess.py              # Offline DSP batch preprocessing utility
├── train.py                   # Model training and validation CLI
├── test_dsp.py                # Complete DSP unit test suite (unittests, zero ML dependency)
├── test_models.py             # Neural network shape & gradient verification script
└── requirements.txt           # Tiered dependency manifest
```

---

## Data Schema & Dataset Adapters

The data layer does **not** hardcode assumptions about any specific benchmark. Any external radar dataset (e.g. DIAT-μRadHAR, CI4R, University of Glasgow, or proprietary hardware streams) is normalized into a standard canonical `RadarSample` before passing to the DSP pipeline:

```python
from data.schema import RadarSample

sample = RadarSample(
    signal=complex_iq_array,        # 1D complex array (I + 1j*Q) of shape (N,)
    sampling_rate=1000.0,           # ADC sample rate / PRF in Hz
    activity_label="Walk",          # Activity target (Walk, Sit, Fall, etc.)
    subject_id="Subject_03",        # Human subject ID (prevents identity leakage)
    recording_id="Trial_001",       # Session identifier
    metadata={"sensor": "CW-24GHz"} # Radar center frequency, chirp slope, etc.
)
```

To ingest a new radar benchmark, simply subclass `BaseRadarDatasetAdapter` in `data/schema.py`:
```python
from data.schema import BaseRadarDatasetAdapter, RadarSample

class CustomRadarAdapter(BaseRadarDatasetAdapter):
    def discover_samples(self):
        return sorted(self.root_dir.glob("**/*.mat"))

    def load_sample(self, file_path) -> RadarSample:
        # Load raw file, parse I/Q channels and subject ID
        return RadarSample(signal=iq_signal, sampling_rate=fs, activity_label=label)
```

---

## Clean FrSST DSP Interface

The signal processing layer exposes a high-level API returning a structured `FrSSTResult`:

```python
import numpy as np
from dsp import frsst

# 1D complex radar signal (e.g., 1000 samples @ 1 kHz)
radar_iq = np.load("sample.npy")

# High-level clean call (automatically searches optimal alpha if None)
result = frsst(
    signal=radar_iq,
    sampling_rate=1000.0,
    n_fft=256,
    hop_length=64,
    alpha=None  # Triggers Rényi entropy minimization
)

print("Sharpened Spectrogram:", result.representation.shape) # (256, 16)
print("Optimal Alpha Order:  ", result.alpha)                 # e.g. 0.450
print("Frequency Range (Hz): ", result.frequency_axis[0], "to", result.frequency_axis[-1])
print("Time Duration (s):    ", result.time_axis[0], "to", result.time_axis[-1])
```

---

## Installation & Setup

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/AdityaTummuri/FracSST-Mamba.git
cd FracSST-Mamba

# Recommended: Python 3.10, 3.11, or 3.12
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
`requirements.txt` is tiered into Core DSP, Deep Learning, and Optional Acceleration:
```bash
# Full installation
pip install -r requirements.txt
```

> **Hardware Acceleration Note**: Native `mamba-ssm` and `causal-conv1d` kernels require an NVIDIA GPU with CUDA. When running on CPU or environments without CUDA toolchains, the codebase **automatically falls back** to the built-in pure-PyTorch selective scan (`PyTorchSelectiveSSM`) without error.

---

## Quickstart & Workflow

### 1. Run Standalone DSP Demo (Zero PyTorch/Mamba dependency)
Verify the signal processing mathematics on your local CPU without installing PyTorch:
```bash
python scripts/run_dsp_demo.py
```
*Generates a micro-Doppler chirp, executes FrFT, SST, and FrSST, prints mathematical metadata, and renders a side-by-side time-frequency comparison figure.*

### 2. Execute DSP Unit Test Suite
```bash
python -m unittest test_dsp.py
```
*Executes 11 unit tests covering boundary orders, Parseval energy conservation ($\Delta L_2 < 10^{-5}$), deterministic behavior, and input validation.*

### 3. Verify Neural Network Architecture
```bash
python test_models.py
```
*Verifies forward pass, backward gradient propagation, shape contracts, and parameter count (421,795 parameters).*

### 4. Run Offline DSP Preprocessor
```bash
python preprocess.py
```
*Processes raw radar signals from `data/raw/` through the FrSST pipeline and writes feature tensors to `data/processed/`.*

### 5. Train FracSST-Mamba Network
```bash
# Standard training run (parameters configured via config/params.yaml)
python train.py

# Inline overrides
python train.py --epochs 50 --batch-size 16

# Quick 1-epoch dry run
python train.py --dry-run
```

### 6. Launch Live Demonstration Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` and click **"🛰️ Simulate Live Radar Feed"** to observe real-time micro-Doppler sharpening and Mamba temporal segmentation.

---

## Activity Classes & Multi-Task Loss

The framework classifies target human activities and performs frame-level temporal event segmentation:

| Class ID | Activity | Micro-Doppler Signature |
|:---:|:---|:---|
| **0** | **Walk** | Periodic sinusoidal envelope with high-velocity limb-swing micro-Doppler modulations |
| **1** | **Sit** | Stationary low-frequency baseline chest/torso micro-motion |
| **2** | **Fall** | High-amplitude, rapid quadratic acceleration burst followed by cessation of motion |

### Loss Formulation
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}}$$

- $\mathcal{L}_{\text{cls}}$: Sequence-level `CrossEntropyLoss` over global activity predictions.
- $\mathcal{L}_{\text{seg}}$: Frame-level `CrossEntropyLoss` over temporal frame activity predictions.
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`.

---

## Citation

If you use FracSST-Mamba in your research or application, please cite:

```bibtex
@software{Aditya_FracSST-Mamba_2026,
  author    = {Aditya, T. and Krishna, Y. and Varshitha, G.},
  title     = {FracSST-Mamba: Physics-Aware Radar Human Activity Recognition},
  year      = {2026},
  version   = {1.0.0},
  url       = {https://github.com/AdityaTummuri/FracSST-Mamba}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
