# REPRODUCIBILITY PROTOCOL & EXECUTION SPECIFICATIONS

## 1. Deterministic Seeding Protocol

All stochastic processes—including pseudo-random weight initialization, data loader shuffling, stochastic augmentation parameters, and dropout masks—are governed by an explicit seeding utility in `src/utils/seed.py`:

```python
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Enforces full deterministic execution across all random number generators."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 2. Computational Environment & Dependency Specification

### 2.1 Hardware Requirements
- **Cloud Training Environment**: Kaggle Notebooks with GPU accelerator enabled:
  - GPU: NVIDIA Tesla T4 (16 GB GDDR6) or NVIDIA Tesla P100 (16 GB HBM2).
  - CPU: 4 vCPUs, 30 GB System RAM.
- **Local Development Environment**:
  - OS: Linux x86_64.
  - RAM: 12 GB.
  - Compute: CPU-only (used for repository scaffolding, unit tests on synthetic arrays, and documentation).

### 2.2 Software Stack & Versions
- **Python**: `3.10.x` or `3.11.x`
- **Core Frameworks**:
  - `torch==2.2.1`
  - `torchvision==0.17.1`
  - `segmentation-models-pytorch==0.3.3`
  - `albumentations==1.4.3`
  - `pydicom==2.4.4`
  - `h5py==3.10.0`
  - `scipy==1.12.0`
  - `scikit-learn==1.4.1.post1`
  - `pandas==2.2.1`
  - `numpy==1.26.4`
  - `pyyaml==6.0.1`
  - `pytest==8.0.2`

---

## 3. Configuration Management

All hyperparameters, file paths, and execution flags are strictly decoupled from code and defined in YAML files under `configs/`:
- `configs/baseline.yaml`: Deterministic baseline training parameters.
- `configs/mc_dropout.yaml`: Spatial dropout rate, MC inference sample count ($T=20$).
- `configs/ensemble.yaml`: Ensemble member count ($M=3$, seeds 42–44, as actually run), seed array, and aggregation rules.

---

## 4. End-to-End Execution Sequence

To reproduce all experimental results from scratch:

```bash
# Step 1: Install exact dependencies
pip install -r requirements.txt

# Step 2: Execute automated unit test suite
pytest tests/ -v

# Step 3: Run DICOM audit and metadata extraction
python -m src.data.dicom_audit --data_dir data/raw/dicom-images-train/ --output data/processed/metadata_master.csv

# Step 4: Generate patient-level stratified split manifest
python -m src.data.split --metadata data/processed/metadata_master.csv --output data/processed/patient_splits.csv

# Step 5: Convert and cache DICOMs to compressed 512x512 HDF5
python -m src.data.cache_hdf5 --split_csv data/processed/patient_splits.csv --output_dir data/processed/cache_512x512/

# Step 6: Train deterministic baseline (5 seeds)
python -m src.train --config configs/baseline.yaml

# Step 7: Train MC Dropout model
python -m src.train --config configs/mc_dropout.yaml

# Step 8: Train Deep Ensemble members (5 models)
python -m src.train --config configs/ensemble.yaml

# Step 9: Run evaluation and selective prediction benchmarks
python -m src.evaluate --test_manifest data/processed/patient_splits.csv --output_dir results/
```
