# FracSST-Mamba: Physics-Aware Radar Human Activity Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
**FracSST-Mamba** is a physics-aware deep learning framework designed for Radar Human Activity Recognition (HAR). It combines Fractional Fourier Transform Synchrosqueezing (FracSST) for advanced Digital Signal Processing (DSP) with State Space Models (Mamba architecture) for efficient temporal sequence modeling.

## Project Structure
```
FracSST-Mamba/
├── LICENSE
├── CITATION.cff
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── params.yaml
├── data/
│   ├── raw/
│   └── processed/
├── dsp/
│   ├── __init__.py
│   ├── frft.py
│   └── synchrosqueezing.py
├── models/
│   ├── __init__.py
│   ├── mamba_block.py
│   └── da_seg.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   └── metrics.py
└── train.py
```

## Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/AdityaTummuri/FracSST-Mamba.git
cd FracSST-Mamba
pip install -r requirements.txt
```

## Usage
To start training the model with default configurations:
```bash
python train.py --config config/params.yaml
```

## Citation
If you use this repository in your research, please cite:
```bibtex
@software{Aditya_FracSST-Mamba_2026,
  author = {Aditya, T. and Krishna, Y. and Varshitha, G.},
  title = {FracSST-Mamba: Physics-Aware Radar Human Activity Recognition},
  version = {1.0.0},
  date = {2026-07-24}
}
```
