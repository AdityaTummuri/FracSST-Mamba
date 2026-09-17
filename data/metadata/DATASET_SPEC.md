# Technical Dataset Specification — FracSST-Mamba Radar Data

## 1. Dataset Origin & Provenance
- **Dataset Name**: University of Glasgow INSHEP Micro-Doppler Radar Dataset (`Dataset_848`)
- **Project**: *Intelligent RF Sensing for Falls and Health Prediction (INSHEP)* (Grant EP/R041679/1)
- **Principal Investigators**: Dr. Syed Aziz Shah, Dr. Julien Le Kernec, Dr. Francesco Fioranelli (University of Glasgow)
- **Licensing / Citation**: Open academic research dataset. Cite Fioranelli et al. (University of Glasgow) and INSHEP project.

---

## 2. Radar Hardware & Physical Signal Parameters

| Parameter | Symbol | Value | Physical Significance |
|:---|:---:|:---:|:---|
| **Radar Architecture** | — | **FMCW (Frequency-Modulated Continuous-Wave)** | Ancortek Radar System |
| **Operating Carrier Frequency** | $f_c$ | **5.8 GHz** | C-band RF sensing wavelength $\lambda \approx 5.17\text{ cm}$ |
| **Chirp Repetition Duration** | $T_c$ | **1.0 ms** ($0.001\text{ s}$) | Determines slow-time Doppler sampling rate |
| **Slow-Time PRF / Sampling Rate** | $f_s$ | **1000.0 Hz** ($1 / T_c$) | Maximum unambiguous Doppler shift: $\pm 500\text{ Hz}$ |
| **RF Chirp Bandwidth** | $B$ | **400 MHz** ($400 \times 10^6\text{ Hz}$) | Range resolution $\Delta R = \frac{c}{2B} \approx 37.5\text{ cm}$ |
| **ADC Fast-Time Samples / Chirp** | $N_{\text{fast}}$ | **128 samples** | Fast-time range profile resolution |
| **Output Transmit Power** | $P_{\text{tx}}$ | **+18 dBm** | Contactless RF power level compliant with FCC/safety |
| **Antennas** | — | **Yagi Transmit & Receive** | Directive $+17\text{ dBi}$ gain |

---

## 3. Raw Data Encoding & File Structure

Original files reside in `data/raw/Dataset_848/`:
- **Format**: Text `.dat` files with a 4-line header:
  - Line 1: Carrier frequency ($5.8\text{ GHz} = 5800000000.0$)
  - Line 2: Chirp duration ($1.0\text{ ms} = 1.0$)
  - Line 3: Fast-time samples ($128$)
  - Line 4: Bandwidth ($400\text{ MHz} = 400000000.0$)
  - Line 5+: Sequence of complex beat-note samples in format `real+imagi` (e.g. `1812+1897i`).
- **File Lengths**:
  - $1,280,000$ samples: $10.0\text{ s}$ duration ($10,000\text{ chirps} \times 128\text{ samples/chirp}$)
  - $640,000$ samples: $5.0\text{ s}$ duration ($5,000\text{ chirps} \times 128\text{ samples/chirp}$)

---

## 4. Processing via `UoGRadarAdapter`

To feed the downstream DSP (`dsp.frsst`) and Mamba layers, `UoGRadarAdapter`:
1. Reshapes the beat-note samples into matrix $(M_{\text{chirps}}, 128)$.
2. Performs fast-time Range-FFT along axis 1 to yield range profiles.
3. Applies a Moving Target Indication (MTI) high-pass clutter filter to eliminate static background reflection ($0\text{ Hz}$ DC component).
4. Selects the peak moving target range bin to extract a clean 1D complex Doppler signal $x[n] \in \mathbb{C}^{M}$ at $f_s = 1000\text{ Hz}$.
5. Encapsulates into canonical `RadarSample` (defined in `data/schema.py`).
