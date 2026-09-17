# Member 2 $\to$ Member 3 Dataset Handoff Dossier

**Author**: Member 2 (Dataset Owner)  
**Status**: Ready for DSP Preprocessing (Member 1) & Model Training (Member 3)  
**Date**: 2026-09-14  

---

## Definition of Done Verification Checklist

| Question | Verification & Answer | Code / File Reference |
|:---|:---|:---|
| **1. What is one radar sample?** | An instance of canonical `RadarSample` containing a 1D complex NumPy array (`np.complex128`) of slow-time micro-Doppler radar returns. | [`data/schema.py`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/schema.py) |
| **2. Where are I/Q values?** | Inside `sample.signal`. It is a 1D complex array ($I + jQ$), where real components represent In-phase ($I$) and imaginary components represent Quadrature ($Q$). | `sample.signal.real`, `sample.signal.imag` |
| **3. What is the sampling rate?** | Exactly **1000.0 Hz** ($T_c = 1.0\text{ ms}$, PRF $= 1000\text{ Hz}$). Verified uniform across all 583 files and matches `config/params.yaml`. | `sample.sampling_rate` |
| **4. What activity does it represent?** | One of the 3 canonical project classes: **`Walk`** (`0`), **`Sit`** (`1`), or **`Fall`** (`2`). | `sample.activity_label` & [`data/metadata/label_mapping.json`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/metadata/label_mapping.json) |
| **5. Which subject produced it?** | Fully preserved in `sample.subject_id` (e.g. `P01`, `P36` ... up to 60 distinct subjects). | `sample.subject_id` |
| **6. Which files belong to train/validation/test?** | Partitioned strictly subject-wise (zero identity leakage):<br>• **Train**: 42 subjects, 413 recordings (`Walk`: 137, `Sit`: 137, `Fall`: 139)<br>• **Val**: 9 subjects, 88 recordings (`Walk`: 29, `Sit`: 30, `Fall`: 29)<br>• **Test**: 9 subjects, 82 recordings (`Walk`: 28, `Sit`: 27, `Fall`: 27) | [`data/metadata/train_val_test_split.json`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/metadata/train_val_test_split.json) |
| **7. What preprocessing has already occurred?** | Fast-time Range-FFT and slow-time MTI stationary clutter removal performed by `UoGRadarAdapter` to isolate moving-target Doppler. Raw data files remain untouched in `data/raw/`. | [`data/adapters/uog_adapter.py`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/adapters/uog_adapter.py) |
| **8. Can it enter the DSP pipeline without guessing?** | **YES**. Calling `adapter.load_sample(file_path)` yields a validated `RadarSample` directly ingestible by `dsp.frsst()`. | See quickstart below |

---

## Quickstart for Member 3 (How to Load Data in 3 Lines)

```python
from data.adapters.uog_adapter import UoGRadarAdapter
import json

# 1. Initialize adapter
adapter = UoGRadarAdapter("data/raw/Dataset_848")

# 2. Load leakage-safe split manifest
with open("data/metadata/train_val_test_split.json") as f:
    splits = json.load(f)

# 3. Load any train/val/test sample without guessing
sample = adapter.load_sample(splits["train_files"][0]["file_path"])
print(sample.activity_label)  # 'Walk', 'Sit', or 'Fall'
print(sample.signal.shape)    # (10000,) or (5000,)
print(sample.sampling_rate)   # 1000.0 Hz
print(sample.subject_id)      # 'P36'
```

---

## Deliverables Summary

1. **Storage Optimization**: Pruned 1,171 non-required files, reducing raw data footprint by **9.33 GB**.
2. **Standard Dataset Adapter**: [`data/adapters/uog_adapter.py`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/adapters/uog_adapter.py)
3. **Data Quality Audit Report**: [`data/metadata/data_quality_report.md`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/metadata/data_quality_report.md) (0 corrupted, 0 NaNs, 100% pass)
4. **Label Mapping Specification**: [`data/metadata/label_mapping.json`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/metadata/label_mapping.json)
5. **Leak-Free Subject Splits**: [`data/metadata/train_val_test_split.json`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/metadata/train_val_test_split.json)
6. **Radar Specification**: [`data/metadata/DATASET_SPEC.md`](file:///home/nuker/Documents/projects/FracSST-Mamba_v01/data/metadata/DATASET_SPEC.md)
