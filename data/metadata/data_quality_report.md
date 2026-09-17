# Radar Dataset Quality & Integrity Report

- **Audit Date**: 2026-09-14
- **Dataset**: University of Glasgow Micro-Doppler Radar Dataset (`Dataset_848`)
- **Radar Hardware**: Ancortek C-Band FMCW (5.8 GHz, PRF 1000 Hz, Bandwidth 400 MHz)
- **Total Discovered Recordings**: 583

---

## 1. Executive Quality Summary

| Metric | Result | Status |
|:---|:---|:---:|
| **Total Discovered Recordings** | 583 | PASS |
| **Successfully Parsed & Validated** | 583 | PASS |
| **Corrupted / Unreadable Files** | 0 | PASS |
| **NaN / Inf Detections** | 0 | PASS |
| **Dead / Zero-Power Recordings** | 0 | PASS |
| **Signal Dynamic Range Integrity** | Mean Peak: 2.50e+04, Mean RMS: 8.64e+03 | PASS |

> [!NOTE]
> All 583 target radar recordings passed 100% of structural and mathematical sanity checks.
> There are zero corrupt files, zero NaNs, zero Infinities, and zero flatlined signals.

---

## 2. Activity Class Distribution

The pruned dataset displays balance across the three project classes:

| Class Label | FracSST-Mamba Index | File Count | Percentage | Balance Status |
|:---|:---:|:---:|:---:|:---:|
| **Walk** | `0` | 194 | 33.3% | Balanced |
| **Sit** | `1` | 194 | 33.3% | Balanced |
| **Fall** | `2` | 195 | 33.4% | Balanced |

---

## 3. Recording Duration Distribution

| Duration (seconds) | Slow-Time Chirps ($N$) | File Count | Percentage |
|:---:|:---:|:---:|:---:|
| **5.0s** | 5000 | 388 | 66.6% |
| **10.0s** | 10000 | 195 | 33.4% |

---

## 4. Subject Diversity & Integrity

- **Unique Human Subjects**: 60 distinct subjects
- **Average Recordings per Subject**: 9.7
- **Subject Identifiers**: Extracted directly from filenames (`P01` through `P72`).
- **Leakage Safeguard**: Subject IDs are fully preserved in `RadarSample.subject_id`, guaranteeing leak-free subject-independent evaluation.

---

## 5. Anomaly Details

- **Corrupted files**: None (0 files)
- **NaN/Inf files**: None (0 files)
- **Zero-power files**: None (0 files)

---

## 6. Certification for Pipeline Handoff

The dataset meets all criteria for FracSST-Mamba Stage 1 DSP and Stage 2 Neural Network processing:
1. Signal format is verified 1D complex `np.complex128`.
2. Sampling rate is strictly uniform across all samples ($f_s = 1000.0\text{ Hz}$).
3. Moving target Doppler frequency spectrum contains valid human micro-Doppler signatures.
4. Class balance is verified for multi-task classification and segmentation.
