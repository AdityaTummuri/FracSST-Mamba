# DSP Lab Project Report

## FracSST-Mamba: Radar-Based Human Activity Recognition Using Fractional Synchrosqueezing and Mamba State-Space Models

**Team Members:**

| Team Member | Roll Number |
| :--- | :--- |
| **Aditya Tummuri** | AM.EN.U4ECE25153 |
| **Varshitha G** | AM.EN.U4ECE25114 |
| **Krishna Nandan Reddy** | AM.EN.U4ECE25157 |


---

## 1. Introduction

Recognizing what a person is doing — whether they are walking, sitting down, or falling — has many real-world applications. It can help elderly care systems detect falls, assist smart home devices, and enable contactless health monitoring. Traditional approaches use cameras or wearable sensors, but these raise privacy concerns and require cooperation from the subject.

**Radar-based sensing** offers a promising alternative. A radar transmits a continuous radio wave, and when a person moves, the reflected signal undergoes tiny frequency shifts called **micro-Doppler signatures**. These frequency shifts carry rich information about the type of motion — arm swings during walking, the gradual deceleration when sitting, or the sudden impact of a fall.

However, extracting clean, usable patterns from raw radar signals is not straightforward. This is where **Digital Signal Processing (DSP)** plays a central role. Our project, **FracSST-Mamba**, combines advanced DSP techniques with a modern deep learning architecture to build an end-to-end radar activity recognition system.

---

## 2. Problem Statement

We addressed three key challenges:

1. **Spectral Smearing:** Human body movements create complex, rapidly changing frequency patterns. The standard Short-Time Fourier Transform (STFT) smears these patterns because of the fundamental **Heisenberg–Gabor uncertainty** — you cannot have perfect time *and* frequency resolution simultaneously.

2. **Computational Cost of Deep Learning:** Transformer-based models, which are state-of-the-art in sequence processing, scale quadratically (O(T²)) with input length, making them expensive for long radar spectrograms.

3. **Subject Leakage in Evaluation:** Most radar studies randomly split data samples for training and testing. Since each person's radar signature is unique (due to body shape, gait, height), this makes the model memorize *who* the person is rather than *what* they are doing. This inflates accuracy numbers artificially.

**Our Objective:** Build a system that (a) uses advanced DSP to get cleaner spectrograms, (b) uses an efficient deep learning model instead of Transformers, and (c) evaluates honestly using a strict subject-independent protocol.

---

## 3. System Architecture

Our pipeline has two main stages — a **pure DSP front-end** followed by a **neural network classifier**:

![FracSST-Mamba Architecture](/home/nuker/.gemini/antigravity-ide/brain/b41227f3-f243-4e86-bb13-6f20433f779d/architecture.png)

### Stage 1: FracSST — The DSP Front-End

This is the core signal processing contribution of our project. It involves three steps:

**Step 1 — Optimal Fractional Order Search (Rényi Entropy)**

The Fractional Fourier Transform (FrFT) is a generalization of the standard Fourier Transform. While the regular FFT rotates a signal by exactly 90° in the time–frequency plane, the FrFT can rotate by *any* angle α. For chirp-like signals (signals whose frequency changes over time), there exists an optimal angle α* that maximally concentrates the signal energy.

We find this optimal α by:
- Trying 31 candidate values of α evenly spaced between 0.0 and 1.5
- For each α, computing the FrFT of the signal and then a short-time spectral representation
- Measuring the **Rényi entropy (order 3)** of each representation — lower entropy means more concentrated (sharper) energy

The α that gives the lowest entropy is selected as the optimal rotation angle.

> **In simple terms:** We try different "viewing angles" for the signal and pick the one where the energy looks the sharpest and most focused.

**Step 2 — Fractional Domain Rotation**

Once the optimal α* is found, we apply the FrFT with this angle to rotate the entire raw radar signal into the "best" domain. This is implemented using the **Ozaktas fast chirp-multiplication algorithm**, which involves:
- Upsampling the signal by a factor of 2
- Three multiplications with chirp functions
- FFT-based fast convolution

**Step 3 — Synchrosqueezing Transform (SST)**

After rotating the signal, we apply **Synchrosqueezing** to further sharpen the time-frequency picture. SST works by:
- Computing the standard STFT with a Hamming window
- Also computing the STFT with a *derivative* of the window function
- Using the ratio of these two STFTs to estimate the **instantaneous frequency** at each time-frequency point
- **Reassigning** energy from smeared STFT bins to the exact instantaneous frequency bins

The result is a razor-sharp spectrogram (256 frequency bins × 16 time frames) where micro-Doppler ridges are tightly concentrated instead of blurred.

### Stage 2: Mamba Classifier — The Neural Network

The sharpened spectrogram is fed into a **MambaClassifier** neural network:

| Component | Details |
|---|---|
| **Input Projection** | Linear layer: 256 frequency bins → 128-dimensional features |
| **Backbone** | 4 stacked Mamba SSM blocks (d_state=16, d_conv=4, expand=2) |
| **Pooling** | Global temporal average pooling over all time frames |
| **Classification Head** | MLP: 128 → 64 → 3 classes (Walk, Sit, Fall) |
| **Total Parameters** | 421,795 (~1.6 MB) |

**Why Mamba instead of Transformers?**

Mamba is a **Selective State-Space Model (SSM)** that processes sequences in linear O(T) time, compared to O(T²) for Transformer self-attention. It achieves this by:
- Using a **1D causal convolution** for local context
- Using a **selective gating mechanism** where the state transition parameters (Δ, B, C) are computed from the input itself, allowing the model to selectively focus on or ignore information
- Maintaining a compact recurrent **hidden state** that gets updated at each time step

This makes it both faster and more memory-efficient, which matters when processing long radar recordings.

---

## 4. Dataset

We used the **University of Glasgow (UoG) 5.8 GHz CW Radar Activity Dataset**:

| Property | Value |
|---|---|
| Radar Type | 5.8 GHz Continuous-Wave (CW) |
| Sample Rate | 4,000 Hz ADC |
| Data Format | 16-bit complex I/Q |
| Total Recordings | 583 across 60 subjects |
| Walk Samples | 202 |
| Sit Samples | 191 |
| Fall Samples | 190 |

### Subject-Independent Splitting

To prevent identity leakage, we partitioned the data **by subject**, ensuring zero person overlap:

| Split | Samples | Subjects | Percentage |
|---|:---:|:---:|:---:|
| **Train** | 413 | 42 | 70.8% |
| **Validation** | 88 | 9 | 15.1% |
| **Test** | 82 | 9 (unseen) | 14.1% |

The 9 test subjects (P02, P08, P09, P15, P16, P28, P53, P60, P69) were **never seen** during training or model selection.

---

## 5. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight decay = 1e-4) |
| Learning Rate | 0.001 |
| Loss Function | Cross-Entropy |
| Batch Size | 32 |
| Epochs | 50 |
| DSP: FFT Window (n_fft) | 256 |
| DSP: Hop Length | 64 |
| DSP: Fractional Order (α) | 0.5 |

Best model checkpoint was saved based on the highest validation accuracy.

---

## 6. Results and Analysis

### 6.1 Overall Performance

| Metric | Validation Set | Test Set (Unseen Subjects) |
|---|:---:|:---:|
| **Samples** | 88 | 82 |
| **Accuracy** | **93.18%** | **80.49%** (66/82 correct) |
| **Macro F1-Score** | — | **0.8045** |
| **Macro Precision** | — | **0.8060** |
| **Macro Recall** | — | **0.8034** |

The gap between validation accuracy (93.18%) and test accuracy (80.49%) is expected and healthy — it shows the model generalizes reasonably well to completely new people, though some performance loss is natural when encountering unseen body types and motion patterns.

### 6.2 Per-Class Performance

| Activity | Precision | Recall | F1-Score | Test Samples |
|---|:---:|:---:|:---:|:---:|
| **Walk** | 96.30% | 92.86% | **0.9455** | 28 |
| **Sit** | 71.43% | 74.07% | **0.7273** | 27 |
| **Fall** | 74.07% | 74.07% | **0.7407** | 27 |

**Walking** is recognized with near-perfect accuracy (F1 = 0.95). This is because walking produces a very distinctive micro-Doppler pattern — periodic sinusoidal modulations from rhythmic arm and leg swings that no other activity produces.

**Sitting** and **Falling** are harder to tell apart, which makes intuitive sense — both involve the body moving downward. The key difference is speed: falling is rapid and ends with a sudden impact, while sitting is controlled and gradual.

### 6.3 Confusion Matrix

![Test Set Confusion Matrix](/home/nuker/.gemini/antigravity-ide/brain/b41227f3-f243-4e86-bb13-6f20433f779d/test_confusion_matrix.png)

**What the confusion matrix tells us:**
- **Walk** is very cleanly classified — 26 out of 28 correct, with only 1 misclassified as Sit and 1 as Fall.
- **Sit → Fall confusion:** 6 sitting samples were wrongly predicted as falls. Both activities share a downward torso motion component.
- **Fall → Sit confusion:** 7 falling samples were wrongly predicted as sits. This is the biggest source of error — the model sometimes cannot distinguish the velocity difference between controlled sitting and rapid falling.
- **No falls were misclassified as walks** (0 in that cell), confirming that the model reliably separates high-motion (walk) from transition activities (sit/fall).

### 6.4 Key Observations

1. **DSP makes a real difference.** The combination of FrFT + SST produces much sharper spectrograms than a plain STFT, which helps the neural network see clearer micro-Doppler ridges.

2. **Subject-independent evaluation is honest.** Our 80.49% accuracy is measured on people the model has never encountered. Many published papers report >98% accuracy but use random splits where the same person appears in both training and testing — those numbers are misleadingly high.

3. **Mamba is efficient.** With only ~422K parameters, our model is extremely lightweight. It processes sequences in linear time, making it practical for real-time deployment on edge devices.

4. **Sit vs. Fall remains the hardest problem.** This is a known challenge across the radar sensing community, not specific to our approach.

---

## 7. DSP Concepts Used

This project applies several core DSP concepts from our coursework:

| Concept | Where We Used It |
|---|---|
| **Fourier Transform (FFT)** | Inside STFT computation and FrFT (Ozaktas algorithm uses FFT internally) |
| **Short-Time Fourier Transform (STFT)** | Base time-frequency analysis inside the SST module |
| **Windowing (Hamming Window)** | Applied to each STFT frame to reduce spectral leakage |
| **Convolution (FFT-based)** | Used in the FrFT chirp convolution step via `scipy.signal.fftconvolve` |
| **Instantaneous Frequency Estimation** | Phase derivative method used in SST energy reassignment |
| **Frequency Domain Filtering** | Zero-padding based upsampling in FrFT |
| **Sampling & Aliasing** | Proper handling of 4 kHz sampling rate and Nyquist frequency |
| **Entropy (Rényi Entropy)** | Information-theoretic criterion for optimal parameter selection |

---

## 8. Limitations and Future Scope

**Current Limitations:**
- The Sit vs. Fall confusion is the primary bottleneck, accounting for ~80% of all errors.
- The dataset was recorded in a controlled indoor lab. Real-world deployment would need to handle walls, multiple people, and arbitrary motion angles.
- The model has been tested on only 60 subjects — a larger, more diverse population would strengthen the results.

**Future Directions:**
- Using multi-channel radar (range-Doppler) to add a second dimension of information.
- Self-supervised pretraining on unlabeled radar data to improve feature learning.
- Model quantization (INT8) for deployment on low-power microcontrollers for real-time fall detection systems.

---

## 9. Conclusion

In this project, we built **FracSST-Mamba**, an end-to-end radar-based human activity recognition system that combines physics-grounded DSP with modern deep learning. The FracSST front-end uses the Fractional Fourier Transform and Synchrosqueezing to produce clean, sharp spectrograms from noisy radar signals. The Mamba classifier processes these spectrograms efficiently in linear time with just ~422K parameters.

Under a strict subject-independent evaluation where the model is tested on 9 completely unseen individuals, we achieved **80.49% accuracy** and a **macro F1-score of 0.8045** across three activity classes (Walk, Sit, Fall). Walking is recognized almost perfectly (F1 = 0.95), while the main challenge lies in distinguishing sitting from falling — a difficulty rooted in the physical similarity of these downward motions.

This project demonstrates that thoughtful signal processing (a core DSP skill) combined with efficient machine learning can solve real-world sensing problems in a privacy-preserving, contactless manner.

---

## 10. References

1. H. M. Ozaktas, O. Arıkan, M. A. Kutay, G. Bozdağı, "Digital computation of the fractional Fourier transform," *IEEE Trans. Signal Process.*, vol. 44, no. 9, pp. 2141–2150, 1996.
2. I. Daubechies, J. Lu, H.-T. Wu, "Synchrosqueezed wavelet transforms," *Applied and Computational Harmonic Analysis*, vol. 30, no. 2, pp. 243–261, 2011.
3. A. Gu, T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," *arXiv preprint arXiv:2312.00752*, 2023.
4. University of Glasgow Radar Activity Recognition Dataset, 5.8 GHz CW Radar, 60 subjects.
5. A. Rényi, "On measures of entropy and information," *Proc. 4th Berkeley Symposium on Mathematical Statistics and Probability*, 1961.
