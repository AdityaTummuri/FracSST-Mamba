# FracSST-Mamba: MATLAB DSP Implementation & Demonstration Suite

This directory contains the standalone, pure MATLAB implementation of the **Digital Signal Processing (DSP) front-end** for the **FracSST-Mamba** project (*Physics-Informed Micro-Doppler Radar Human Activity Recognition*).

It allows researchers, evaluators, and students to run, verify, visualize, and benchmark the entire mathematical radar signal processing pipeline directly in MATLAB without requiring Python, PyTorch, CUDA, or deep learning libraries.

---

## 1. What is the DSP Layer and What Problem Does It Solve?

In Continuous-Wave (CW) and FMCW radar Human Activity Recognition (HAR), moving body parts (torso, arms, legs) generate frequency shifts called **micro-Doppler signatures**:
- **Walking**: Involves forward torso motion coupled with periodic sinusoidal arm and leg swings.
- **Sitting**: Involves controlled transitional deceleration ending in low-frequency torso motion.
- **Falling**: Involves high-velocity quadratic acceleration bursts ending abruptly on floor impact.

### The Fundamental Problem: Heisenberg-Gabor Spectral Smearing
Standard time-frequency analysis uses the Short-Time Fourier Transform (STFT):
$$\text{STFT}(t, \omega) = \int_{-\infty}^{\infty} s(\tau) w(\tau - t) e^{-j \omega \tau} d\tau$$
Due to the **Heisenberg-Gabor uncertainty principle** ($\Delta t \cdot \Delta \omega \ge \frac{1}{2}$), a window cannot achieve fine temporal and frequency resolution simultaneously. Accelerating limb swings and rapid fall bursts appear as wide, smeared, blurry energy clouds rather than distinct trajectory lines.

### The Solution: Fractional Synchrosqueezing Transform (FracSST)
The DSP layer solves this using a two-stage physics-informed mathematical pipeline:
1. **Fractional Fourier Transform (FrFT)**: Rotates the signal in the time-frequency Wigner-Ville distribution plane by an optimal fractional order $\alpha^*$ (found by minimizing order-3 Rényi entropy). This **dechirps** accelerating Doppler components, aligning slanted trajectories.
2. **Synchrosqueezing Transform (SST)**: Computes instantaneous phase derivatives to determine local instantaneous frequencies, and reassigns smeared spectral energy into razor-sharp frequency ridges while strictly preserving temporal causality.

```
Raw Radar I/Q Signal
       │
       ▼
[ Range-FFT + MTI Clutter Filter ]  ──> Extracts 1D slow-time complex Doppler signal
       │
       ▼
[ Rényi Entropy Optimization ]      ──> Evaluates α ∈ [0, 1.5] → discovers optimal α*
       │
       ▼
[ Fast Ozaktas FrFT Rotation ]      ──> Rotates signal into optimal fractional domain: f_rot = FrFT(f, α*)
       │
       ▼
[ Synchrosqueezing Reassignment ]   ──> Evaluates phase derivatives ω_inst → reallocates energy
       │
       ▼
[ Sharpened Spectrogram Matrix ]    ──> (256 frequency bins × T time frames) feeding Mamba AI
```

---

## 2. Directory Structure & File Manifest

| File | Purpose | Key Mathematical Operations |
|:---|:---|:---|
| [`frft.m`](file:///d:/FracSST-Mamba/matlab/frft.m) | Discrete Fractional Fourier Transform | Ozaktas $O(N \log N)$ chirp-multiplication algorithm, modulo-4 arithmetic, Parseval $L_2$ norm preservation. |
| [`find_optimal_alpha.m`](file:///d:/FracSST-Mamba/matlab/find_optimal_alpha.m) | Parameter Optimization | Grid search over $\alpha \in [0, 1.5]$ minimizing order-3 Rényi Entropy $R_3 = -\frac{1}{2} \ln \sum P^3$. |
| [`sst_1d.m`](file:///d:/FracSST-Mamba/matlab/sst_1d.m) | Standard 1D Synchrosqueezing | STFT with Hamming window $w(t)$ and derivative window $w'(t)$, phase derivative $\omega_{\text{inst}}$, energy reassignment. |
| [`fracsst_1d.m`](file:///d:/FracSST-Mamba/matlab/fracsst_1d.m) | Unified FracSST Pipeline | High-level API connecting `find_optimal_alpha` $\to$ `frft` $\to$ `sst_1d`. |
| [`uog_load_dat.m`](file:///d:/FracSST-Mamba/matlab/uog_load_dat.m) | Real Radar Data Loader | Parses 5.8 GHz FMCW UoG radar `.dat` files, computes fast-time Range-FFT, applies MTI clutter filter, extracts peak Doppler bin. |
| [`create_synthetic_radar_sample.m`](file:///d:/FracSST-Mamba/matlab/create_synthetic_radar_sample.m) | Benchmark Signal Generator | Generates synthetic complex I/Q radar returns for **Walk**, **Sit**, and **Fall** with physical gait and acceleration kinetics. |
| [`demo_dsp_visualizer.m`](file:///d:/FracSST-Mamba/matlab/demo_dsp_visualizer.m) | Interactive Visual Demonstration | Renders 6-panel comparative diagnostic dashboard and before/after high-resolution plots with quantitative metrics. |
| [`test_dsp.m`](file:///d:/FracSST-Mamba/matlab/test_dsp.m) | Automated Unit Test Suite | 10 mathematical tests verifying FrFT boundary conditions, Parseval energy conservation, SST dimensions, and non-negativity. |
| [`test_custom_data.m`](file:///d:/FracSST-Mamba/matlab/test_custom_data.m) | Quick-Start Custom Data Template | Plug-and-play script for evaluating custom signals from `.mat`, `.csv`, or arrays. |

---

## 3. How to Run and Test in MATLAB

### Option A: Run the Automated Unit Tests
Open MATLAB, navigate to `d:\FracSST-Mamba\matlab`, and execute:
```matlab
test_dsp
```
**Expected Output:**
```
====================================================================
        FracSST-Mamba: MATLAB DSP Automated Unit Test Suite         
====================================================================

>> [Group 1] Testing Fractional Fourier Transform (frft.m)...
   [PASS] Test 1: FrFT identity order (a = 0) returns identical signal (err = 0.00e+00)
   [PASS] Test 2: FrFT Fourier limit (a = 1) matches scaled FFT (err = 1.12e-16)
   [PASS] Test 3: FrFT inverse Fourier (a = -1) matches scaled IFFT (err = 1.12e-16)
   [PASS] Test 4: Parseval L2 energy preserved across all arbitrary orders (max err = 2.41e-15)
   [PASS] Test 5: Deterministic numerical reproducibility verified

>> [Group 2] Testing Synchrosqueezing Transform (sst_1d.m)...
   [PASS] Test 6: SST output matrix dimensions (128 x 16) match grid axes
   [PASS] Test 7: Reassigned spectral energy is strictly non-negative and finite (no NaN/Inf)

>> [Group 3] Testing Optimal Alpha & Full FracSST Pipeline...
   [PASS] Test 8: Optimal alpha search returns valid parameter alpha* = 0.500 (R3 = 4.1204)
   [PASS] Test 9: End-to-end FracSST pipeline completes successfully with verified integrity
   [PASS] Test 10: Multi-activity verification passed for Walk, Sit, and Fall signals

====================================================================
   TOTAL TESTS : 10 | PASSED : 10 | FAILED : 0
   >>> ALL TESTS PASSED! DSP LAYER IS FULLY OPERATIONAL IN MATLAB. <<<
====================================================================
```

---

### Option B: Run the Full Interactive Visualizer
Run the visualizer script to generate comparative figures:
```matlab
demo_dsp_visualizer
```
You can switch activities inside `demo_dsp_visualizer.m` on line 41:
```matlab
data_mode = 'synthetic_walk';  % Options: 'synthetic_walk', 'synthetic_fall', 'synthetic_sit', 'real_uog'
```

#### What Graphs Will You See?
1. **Figure 1: 6-Panel Diagnostic Dashboard**:
   - **(A) Raw Radar I/Q Signal**: Time-domain In-Phase ($I$) and Quadrature ($Q$) returns.
   - **(B) Rényi Entropy Search Curve**: Plot of $R_3(\alpha)$ vs $\alpha$ showing the exact convex minimum identifying $\alpha^*$.
   - **(C) Spectral Slice Cross-Section**: Overlaid frequency cross-sections comparing STFT vs SST vs FracSST at peak acceleration, illustrating significant Full-Width at Half-Maximum (FWHM) peak sharpening.
   - **(D) Conventional STFT Spectrogram**: Baseline blurred spectrogram.
   - **(E) Standard SST Spectrogram ($\alpha=0$)**: Partial reassignment without dechirping.
   - **(F) Proposed FracSST Spectrogram ($\alpha=\alpha^*$)**: Razor-thin micro-Doppler ridges.
2. **Figure 2: Before & After High-Resolution Comparison**:
   - Side-by-side view of Conventional STFT vs FracSST with synchronized frequency ranges ($-f_s/4$ to $+f_s/4$) and calibrated power colorbars in dB.

---

### Option C: Test Your Own Data
To evaluate your own custom radar recording or array:
1. Open [`test_custom_data.m`](file:///d:/FracSST-Mamba/matlab/test_custom_data.m).
2. Set your signal and sampling frequency:
   ```matlab
   % Example: loading your own .mat file
   load('my_radar_signal.mat', 'radar_array');
   my_signal = radar_array;
   fs = 1000; % your sampling rate in Hz
   ```
3. Run:
   ```matlab
   test_custom_data
   ```
4. The script will automatically search for $\alpha^*$, apply FracSST, and render the before/after comparison plot.

---

## 4. How to Explain the DSP Part During a Presentation or Viva

Use the following structured talking points when presenting this work:

### 1. Motivation (Why not just use STFT?)
> *"Human movement in Continuous-Wave radar produces non-stationary Doppler frequency shifts that accelerate over time. Under the classical Heisenberg-Gabor limit, standard STFT forces a trade-off: a short window blurs frequency, while a long window blurs fast-moving limb acceleration bursts. This creates blurry spectrograms that degrade deep learning classification accuracy."*

### 2. Fractional Fourier Transform Rotation (Panel B & F)
> *"The Fractional Fourier Transform generalizes the Fourier transform into fractional angular orders $\alpha$. When $\alpha=0$, we are in pure time; when $\alpha=1$, we are in pure frequency. At fractional orders $\alpha^*$, the Wigner-Ville distribution rotates by angle $\phi = \alpha^* \frac{\pi}{2}$. This rotation dechirps the quadratic phase acceleration of the human body, turning slanted, smeared chirps into stationary spectral components."*

### 3. Optimal Order Selection via Order-3 Rényi Entropy (Panel B)
> *"Rather than manually tuning $\alpha$, our DSP layer automatically searches for $\alpha^*$ by minimizing order-3 Rényi Entropy:
> $$R_3 = -\frac{1}{2} \ln \sum_{t, \omega} P(t, \omega)^3$$
> Because entropy measures disorder and spectral dispersion, the minimum of this curve corresponds mathematically to the highest energy concentration and minimal spectral smearing."*

### 4. Synchrosqueezing Reassignment (Panel C, E, & F)
> *"Standard Fourier analysis spreads energy across adjacent bins. Synchrosqueezing computes the instantaneous phase derivative using a window derivative $w'(t)$:
> $$\omega_{\text{inst}}(t, \omega) = \omega - \frac{1}{2\pi} \text{Im}\left( \frac{\text{STFT}_{w'}(t, \omega)}{\text{STFT}_w(t, \omega)} \right)$$
> It then reassigns spectral energy strictly along the frequency axis to the true instantaneous frequency ridge. This sharpens the spectral cross-section (as shown by the sharp needle in Panel C) without altering the temporal timeline."*

### 5. Quantitative Metric Verification (Command Window Table)
Show the evaluation table generated in MATLAB:
- **Order-3 Rényi Entropy ($R_3$)**: Decreases significantly from STFT $\to$ SST $\to$ FracSST (lower is better).
- **Shannon Entropy ($H$)**: Decreases, showing reduced informational entropy in the frequency domain.
- **Energy Concentration Measure (ECM)**: Increases by orders of magnitude, proving energy is concentrated into fewer, high-SNR trajectory ridges.

### 6. Hand-off to Stage 2 Mamba Sequence Model
> *"The resulting 2D matrix (256 frequency bins $\times$ $T$ time frames) represents clean, physics-sharpened micro-Doppler ridges. This serves as the input sequence to the 4-layer bidirectional Mamba Selective State-Space classifier, enabling linear $O(T)$ complexity Human Activity Recognition."*

---

## 5. Summary of Mathematical Formulas

1. **Fractional Order Reduction**:
   $$a_{\text{mod}} = a \pmod 4, \quad a_{\text{mod}} \in [-2, 2)$$
2. **Ozaktas Chirp Factor**:
   $$\phi = a_{\text{mod}} \cdot \frac{\pi}{2}, \quad g(u) = f_{2x}(u) \cdot \exp\left(-j \pi \tan\left(\frac{\phi}{2}\right) u^2\right)$$
3. **Parseval Energy Conservation**:
   $$\|F_\alpha\|_2 = \|f\|_2$$
4. **Instantaneous Frequency Estimator**:
   $$\hat{\omega}(t, \omega) = \omega - \frac{1}{2\pi} \text{Im}\left(\frac{Z_{xx}^{w'}(t, \omega)}{Z_{xx}^w(t, \omega) + \epsilon}\right)$$
5. **Reassignment Mapping**:
   $$S_{\text{FracSST}}(\omega_k, t_j) = \sum_{\omega \in \{ \omega : |\hat{\omega}(t_j, \omega) - \omega_k| \le \frac{\Delta \omega}{2} \}} |Z_{xx}(t_j, \omega)|$$
