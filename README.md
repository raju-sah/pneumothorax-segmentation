# Uncertainty-Aware Pneumothorax Segmentation & Selective Prediction in Chest Radiographs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![License: Research Use](https://img.shields.io/badge/License-Research_Use-green.svg)](LICENSE)
[![Validation: Zero Patient Leakage](https://img.shields.io/badge/Validation-Zero_Patient_Leakage-purple.svg)](DATA_LEAKAGE.md)

An end-to-end medical AI research project investigating whether predictive uncertainty can identify spatial segmentation failures on 2D chest radiographs (CXRs) and support selective prediction (human-in-the-loop clinical triage) using the SIIM-ACR dataset.

Designed as a publication-grade research portfolio targeting venues such as **Medical Image Deep Learning (MIDL)** and **MICCAI**.

---

## 📖 Research Framework & Core Questions

Standard deep learning segmentation models frequently suffer from **silent overconfidence**—mistaking benign anatomical artifacts (e.g., skin folds, scapular borders, rib lines) for life-threatening pneumothoraces or missing subtle apical visceral pleural lines without signaling diagnostic hesitation.

This project addresses six central research questions:

1. **RQ1 (Baseline vs. Uncertainty)**: How does uncertainty-aware inference compare with a deterministic ResNet34 U-Net baseline?
2. **RQ2 (Error Correlation)**: Does predictive uncertainty correlate with actual pixel-level and lesion-level segmentation error?
3. **RQ3 (Failure Modes)**: Can epistemic uncertainty identify radiological confounders (pseudopneumothorax)?
4. **RQ4 (Lesion Difficulty)**: Does uncertainty elevate for subtle, small pneumothoraces ($<2\%$ hemithorax) and portable AP projections?
5. **RQ5 (Selective Prediction)**: Can uncertainty support automated clinical referral, improving the diagnostic accuracy of autonomously retained cases?
6. **RQ6 (MC Dropout vs. Ensembles)**: Does a 5-member Deep Ensemble justify its $5\times$ training cost over a 20-pass Monte Carlo Dropout model for clinical triage?

---

## 📂 Repository Documentation & Research Governance

All research decisions, dataset audits, mathematical formulations, and protocols are documented in depth:

| Document | Description |
| :--- | :--- |
| **[RESEARCH_PLAN.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/RESEARCH_PLAN.md)** | Executive research plan, central questions, hypotheses, and scope. |
| **[DATASET_PLAN.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/DATASET_PLAN.md)** | Data acquisition, directory layouts, preprocessing, and HDF5 caching. |
| **[DATASET_AUDIT.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/DATASET_AUDIT.md)** | Full audit of Stage 1 vs. Stage 2, DICOM metadata, and mask counts. |
| **[DATA_LEAKAGE.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/DATA_LEAKAGE.md)** | Patient leakage audit, `PatientID` grouping, and NIH ChestX-ray14 confounds. |
| **[LITERATURE_REVIEW.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/LITERATURE_REVIEW.md)** | Synthesis of CXR segmentation, UQ paradigms, and selective prediction. |
| **[RESEARCH_GAPS.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/RESEARCH_GAPS.md)** | Identification of three defensible scientific gaps (no trivial claims). |
| **[UNCERTAINTY_METHODS.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/UNCERTAINTY_METHODS.md)** | Mathematical derivations of MC Dropout, Ensembles, and Entropy. |
| **[METRICS.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/METRICS.md)** | Mathematical definitions of overlap, calibration, and referral metrics. |
| **[EXPERIMENT_PLAN.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/EXPERIMENT_PLAN.md)** | Protocols for EXP-01 through EXP-10 (hypotheses and failure criteria). |
| **[REPRODUCIBILITY.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/REPRODUCIBILITY.md)** | Deterministic seeding, hardware specifications, and reproduction commands. |
| **[LIMITATIONS.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/LIMITATIONS.md)** | Appraisal of 2D projection limits, single consensus, and clinical scope. |
| **[MODEL_CARD.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/MODEL_CARD.md)** | Standardized model card following Mitchell et al. (2019). |
| **[DATA_CARD.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/DATA_CARD.md)** | Standardized data card following Gebru et al. (2021). |
| **[RESEARCH_DECISIONS.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/RESEARCH_DECISIONS.md)** | Architecture Decision Records (ADRs) explaining all technical trade-offs. |
| **[PAPER_OUTLINE.md](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/PAPER_OUTLINE.md)** | Publication-ready manuscript outline formatted for MIDL/MICCAI. |

---

## 🏗️ Architecture & Methodological Highlights

```
                       Input Chest Radiograph (512x512)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    ResNet34 U-Net Core    │
                        └─────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    [Method A: Baseline]      [Method B: MC Dropout]   [Method C: Deep Ensemble]
      Single Pass T=1           Spatial Dropout p=0.2     5 Diverse Model Seeds
      Naive Shannon Entropy     T=20 Stochastic Passes    Mutual Information & Var
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                     ┌─────────────────────────────────┐
                     │ Dense-to-Case Uncertainty Gate  │
                     │ (Top-500 Pixel Variance Filter) │
                     └────────────────┬────────────────┘
                                      │
                       Uncertainty Score > Threshold?
                                ├── YES ──► Refer to Radiologist (Deferred Cohort)
                                └── NO  ──► Autonomous Diagnosis (Retained Cohort)
```

- **Backbone**: U-Net with ImageNet-pretrained ResNet34 encoder.
- **Resolution**: $512 \times 512$ with bicubic image and nearest-neighbor mask interpolation.
- **Loss**: $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{SoftDice}}$ ($\epsilon = 1.0$).
- **Partitioning**: Grouped Stratified 5-Fold Cross-Validation + 20% independent test holdout grouped by `PatientID`.

---

## 🚀 Quickstart & Verification

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/your-username/uncertainty-pneumothorax-segmentation.git
cd uncertainty-pneumothorax-segmentation
pip install -r requirements.txt

# 2. Run automated test suite to verify math and pipeline integrity
pytest tests/ -v
```
