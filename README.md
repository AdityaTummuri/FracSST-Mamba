# FracSST-Mamba: Physics-Informed Micro-Doppler Radar Activity Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![DSP: Pure NumPy/SciPy](https://img.shields.io/badge/DSP-NumPy%20%7C%20SciPy-013243.svg)](dsp/)

---

## 1. Project Overview

**FracSST-Mamba** is a physics-informed deep learning system for Continuous-Wave (CW) radar **Human Activity Recognition (HAR)**. It couples a parameter-free mathematical signal processing front-end with an efficient Selective State-Space sequence model (Mamba):

1. **Deterministic DSP Front-End (FracSST)**: Projects raw radar I/Q signals into the optimal fractional domain via the Fractional Fourier Transform (FrFT)—selected by minimizing order-3 Rényi entropy—and sharpens micro-Doppler ridges using the Synchrosqueezing Transform (SST).
2. **Selective State-Space Classifier (MambaClassifier)**: Processes time-frequency feature sequences with linear $\mathcal{O}(T)$ complexity, capturing long-range temporal motion dynamics without the quadratic computational bottleneck of standard Transformers.

---

## 2. Problem & Objectives

- **Micro-Doppler Smearing**: Human body motions (walking, sitting, falling) generate complex, non-stationary frequency trajectories. Conventional Short-Time Fourier Transforms (STFT) suffer from severe spectral smearing bounded by the Heisenberg-Gabor uncertainty principle.
- **Transformer Complexity**: Standard self-attention models scale quadratically ($\mathcal{O}(T^2)$) with spectrogram sequence length, requiring high compute budgets and memory footprint.
- **Subject Leakage in Radar Benchmarks**: Random sample splitting creates identity leakage because radar returns correlate heavily with individual gait, limb length, and body mass. Models evaluated without subject separation often report artificially inflated accuracy (>98%) that fails on new individuals.
- **Objective**: Develop an end-to-end, reproducible radar classification pipeline evaluated under **strict subject-independent conditions** (zero subject overlap between train, validation, and test splits).

---

## 3. End-to-End Pipeline

```text
Raw Radar I/Q Signal (.dat)
           │
           ▼
[ UoG Radar Dataset Adapter ]   ──> Canonical RadarSample (5.8 GHz CW, 4 kHz fs)
           │
           ▼
[ Fractional SST (FracSST) ]    ──> Rényi Entropy α* search → Fast FrFT (Ozaktas)
                                    → Instantaneous frequency SST reassignment
           │
           ▼
[ Sharpened Spectrograms ]      ──> Frequency representations (256 frequency bins)
           │
           ▼
[ MambaClassifier Network ]     ──> Linear projection stem → Stacked bidirectional
                                    Mamba SSM blocks (4 layers) → Global pooling
           │
           ▼
[ Activity Classification ]     ──> Walk | Sit | Fall
```

---

## 4. Supported Activity Classes

| Class ID | Activity | Micro-Doppler Motion Characteristics |
|:---:|:---|:---|
| **0** | **Walk** | Periodic sinusoidal envelopes with distinct high-velocity limb-swing modulations. |
| **1** | **Sit** | Controlled transitional deceleration ending in low-velocity torso micro-motion. |
| **2** | **Fall** | High-velocity, rapid non-linear acceleration burst followed by abrupt cessation of motion. |

---

## 5. Dataset Summary

The pipeline is integrated with the **University of Glasgow (UoG) 5.8 GHz Continuous-Wave Radar Activity Dataset**:

- **Radar Specifications**: 5.8 GHz carrier frequency, Continuous-Wave (CW), 4,000 Hz ADC sample rate, 16-bit complex (I/Q) channels.
- **Total Valid Recordings**: 583 radar recordings across 60 human subjects.
- **Activity Balance**:
  - **Walk**: 202 samples
  - **Sit**: 191 samples
  - **Fall**: 190 samples
- **Subject-Independent Partitioning** (`data/metadata/train_val_test_split.json`):
  - **Train Set**: 413 samples across 42 subjects (~70.8%)
  - **Validation Set**: 88 samples across 9 subjects (~15.1%)
  - **Test Set**: 82 samples across 9 unseen subjects (~14.1%)
  - **Subject Overlap**: Exactly **0** (strictly disjoint subject IDs across all sets).

---

## 6. Model Summary

- **Architecture**: `MambaClassifier` ([`models/da_seg.py`](models/da_seg.py))
- **Input Representation**: Temporal sequence of frequency vectors $(B, T, F)$ with $F=256$ frequency bins.
- **Backbone**:
  - Linear input projection (`d_freq=256` $\to$ `d_model=128`, LayerNorm, Dropout).
  - 4 stacked bidirectional Mamba Selective State-Space Model blocks (`d_state=16`, `d_conv=4`, `expand=2`).
  - Global temporal average pooling over sequence length $T$.
  - Multi-layer MLP classification head (`128` $\to$ `64` $\to$ `3`).
- **Total Trainable Parameters**: 421,795 (~1.6 MB memory footprint).
- **Execution Engine**: Features automatic dual-backend execution:
  - Hardware-accelerated CUDA kernels via `mamba-ssm` when GPU tools are available.
  - Native vectorized PyTorch sequential scan (`PyTorchSelectiveSSM`) on CPU environments, ensuring full functionality without C++ compilation requirements.

---

## 7. Current Verified Results

The model was trained on 413 samples, tuned on 88 validation samples, and evaluated on the **82 previously unseen test samples** from 9 held-out subjects:

### Summary Performance

| Metric | Validation Set (9 subjects) | Test Set (9 unseen subjects) |
|---|:---:|:---:|
| **Sample Count** | 88 samples | 82 samples |
| **Accuracy** | **93.18%** | **80.49%** (66 / 82) |
| **Macro F1-Score** | — | **0.8045** |
| **Macro Precision** | — | **0.8060** |
| **Macro Recall** | — | **0.8034** |

> **Note on Evaluation**: The reported **80.49%** is test accuracy on 82 samples from 9 completely unseen subjects (`P02`, `P08`, `P09`, `P15`, `P16`, `P28`, `P53`, `P60`, `P69`).

### Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Walk** | 96.30% | 92.86% | **0.9455** | 28 |
| **Sit** | 71.43% | 74.07% | **0.7273** | 27 |
| **Fall** | 74.07% | 74.07% | **0.7407** | 27 |

### Confusion Matrix (Test Set)

```text
                  Predicted
             Walk    Sit    Fall
True Walk  [  26      1       1  ]
     Sit   [   1     20       6  ]
     Fall  [   0      7      20  ]
```

- **Analysis**:
  - **Walk** is classified with high accuracy (F1: 0.9455) due to its distinctive periodic micro-Doppler rhythm.
  - The dominant source of classification error is mutual confusion between **Sit** and **Fall** (6 sitting trials predicted as fall; 7 falling trials predicted as sit). Both actions share downward torso acceleration vectors, differing primarily in velocity and terminal impact dynamics.

Artifacts:
- Checkpoint: `weights/best_model.pth`
- Test Metrics: `results/test_evaluation_results.json`
- Confusion Matrix Plot: `results/test_confusion_matrix.png`

---

## 8. Repository Structure

```text
FracSST-Mamba/
├── assets/
│   └── architecture.png              # Architecture schematic
├── config/
│   └── params.yaml                   # Model and training hyperparameters
├── data/
│   ├── adapters/
│   │   └── uog_adapter.py            # UoG radar binary parser
│   ├── metadata/
│   │   ├── train_val_test_split.json # Subject-independent partition manifest
│   │   └── label_mapping.json        # Activity class mappings
│   ├── schema.py                     # Canonical RadarSample dataclass
│   └── dataset.py                    # PyTorch Dataset and DataLoader loaders
├── dsp/
│   ├── frft.py                       # Fractional Fourier Transform & Rényi search
│   └── synchrosqueezing.py           # Synchrosqueezing transform routines
├── models/
│   ├── da_seg.py                     # MambaClassifier network architecture
│   └── mamba_block.py                # Mamba SSM block (CUDA + PyTorch CPU fallback)
├── results/
│   ├── test_evaluation_results.json  # Verified test set evaluation metrics
│   └── test_confusion_matrix.png     # Rendered test set confusion matrix
├── scripts/
│   ├── run_dsp_demo.py               # Standalone DSP demonstration script
│   ├── audit_data_quality.py         # Dataset validation utility
│   └── create_splits.py              # Subject-independent split generator
├── utils/
│   ├── checkpoint.py                 # Checkpoint save/load helpers
│   ├── metrics.py                    # Detailed classification & macro metrics
│   └── visualization.py              # Spectrogram and confusion matrix plotters
├── app.py                            # Streamlit interactive radar demo dashboard
├── preprocess.py                     # Batch offline DSP preprocessing CLI
├── train.py                          # Training, validation, and test evaluation CLI
├── test_data_adapter.py              # Unit tests for radar data adapter
├── test_dsp.py                       # Unit tests for FrFT and Synchrosqueezing
├── test_models.py                    # Unit tests for Mamba neural network
└── requirements.txt                  # Python dependency specifications
```

---

## 9. Installation & Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/AdityaTummuri/FracSST-Mamba.git
cd FracSST-Mamba

# Create and activate virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Hardware Acceleration**: The codebase includes an automatic CPU fallback (`PyTorchSelectiveSSM`). If an NVIDIA GPU with CUDA toolchains is available, native `mamba-ssm` can optionally be installed for accelerated scans.

---

## 10. Usage Instructions

### Run Unit Tests
Verify signal processing mathematics, model dimensions, and data adapters:
```bash
python -m unittest test_dsp.py test_models.py test_data_adapter.py
```
*(All 22 unit tests should pass).*

### Offline Preprocessing
Convert raw radar `.dat` recordings into preprocessed FracSST spectrogram tensors:
```bash
python preprocess.py
```
*Tensors are saved to `data/processed/` with shape `(F=256, T)`.*

### Training
Train the `MambaClassifier` using the subject-independent split:
```bash
python train.py --epochs 50 --batch-size 16 --lr 0.001
```
*The best model weights based on validation accuracy are saved to `weights/best_model.pth`.*

### Test Evaluation
Evaluate the trained checkpoint strictly on the untouched test partition:
```bash
python train.py --test
```
*Loads `weights/best_model.pth`, verifies zero subject leakage, evaluates the 82 unseen test samples, and saves updated metrics and confusion matrix plots to `results/`.*

### Interactive Demonstration Dashboard
Launch the Streamlit visualization interface:
```bash
streamlit run app.py
```

---

## 11. Subject-Independent Evaluation Note

In radar-based human sensing, evaluating models with random frame or sample splits is a common pitfall. Radar returns reflect individual body mass, limb length, and walking cadence. When samples from the same person appear in both training and test sets, models can memorize subject identities rather than learning motion dynamics, inflating test accuracy.

FracSST-Mamba enforces a **strict subject-independent protocol**:
- **42 subjects** are used exclusively for training.
- **9 subjects** are used for validation model selection.
- **9 unseen subjects** are held out exclusively for final testing.
- **Subject Leakage = 0**.

The 80.49% test accuracy reflects true out-of-subject generalization on previously unseen individuals.

---

## 12. Limitations & Future Work

- **Sit vs. Fall Ambiguity**: Both actions involve downward torso movement. Distinguishing rapid sitting from controlled falling requires incorporating higher Doppler resolution or multi-static radar geometries.
- **Controlled Dataset Scope**: Evaluated on 60 subjects recorded in controlled line-of-sight laboratory environments. Real-world continuous deployment will require addressing through-wall attenuation, multi-occupant interference, and arbitrary motion trajectories.
- **Future Directions**:
  - Incorporating multi-channel Doppler-range profiles.
  - Exploring contrastive self-supervised pretraining on unlabeled radar streams.
  - Edge optimization and INT8 quantization for real-time micro-controller execution.

---

## 13. Team & License

- **Authors**: T. Aditya, Y. Krishna, G. Varshitha
- **License**: This project is licensed under the [MIT License](LICENSE).
