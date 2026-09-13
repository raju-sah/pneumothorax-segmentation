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
6. **RQ6 (MC Dropout vs. Ensembles)**: Does a 3-member Deep Ensemble justify its $3\times$ training cost over a 20-pass Monte Carlo Dropout model for clinical triage?

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
      Single Pass T=1           Spatial Dropout p=0.2     3 Diverse Model Seeds
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

---

## 📊 Empirical Results (Test Holdout, $N=2,135$)

All experiments (EXP-01 through EXP-10) were executed using the zero-leakage test holdout ($2,135$ radiographs, $22.3\%$ positive prevalence):

### Table 1: Primary Experimental Benchmark (P10 verbatim run, $N=2,135$)
| Evaluation Metric | Deterministic Baseline (seed 42) | MC Dropout ($T=20$) (seed 42) | Deep Ensemble ($M=3$) |
| :--- | :---: | :---: | :---: |
| **$\text{DSC}_{\text{pos}}$** (Positive Cases) | 0.2803 [0.264, 0.297] | 0.0176* [0.010, 0.027] | **0.3158** [0.292, 0.339] |
| **$\text{DSC}_{\text{all}}$** (Overall Cohort) | 0.1112 [0.101, 0.122] | 0.7735* [0.756, 0.791] | **0.3275** [0.309, 0.346] |
| **AUROC-ED** (Error Detection, global pixels) $\uparrow$ | **0.9936** | 0.9884 | 0.9929 |
| **ESCE** (Calibration Error) $\downarrow$ | 0.00024 | 0.00015* | **0.00025** |
| **Brier Score** $\downarrow$ | 0.00009 | 0.00008* | 0.00010 |
| **AURC** (Risk-Coverage, standard) $\downarrow$ | 0.7687 | 0.1525* | **0.7100** |

\*MC degenerate (near-empty masks); not wins. ens−det DSC_all Δ=+0.2163 [0.1989, 0.2352], Wilcoxon p=2.3e-96. See `results/P10_FINDINGS.md` — including the disclosed checkpoint-selection variance (same-seed DSC_pos 0.61→0.28 across selection rules).

### Table 2: Clinical Selective Prediction & Referral Simulation (P10)
| Strategy | AURC $\downarrow$ | Retained Dice @ 100% | Retained Dice @ 90% | Retained Dice @ 80% | Retained Dice @ 70% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Referral** | 0.8888 | 0.1112 | 0.1112 | 0.1112 | 0.1112 |
| **Deterministic Entropy** | 0.7687 | 0.1112 | 0.1142 | 0.1176 | 0.1222 |
| **MC Dropout Variance*** | 0.1525 | 0.7735 | 0.7968 | 0.8122 | 0.8266 |
| **Deep Ensemble MutInfo** | 0.7100 | **0.3275** | **0.3228** | **0.3061** | **0.2914** |

### Generated Publication Visualizations
- **[Figure 1: Risk-Coverage Curves](docs/assets/fig1_risk_coverage_curves.png)**
- **[Figure 2: AUROC Error Detection ROC](docs/assets/fig3_auroc_error_detection.png)**
- **[Figure 3: AP vs. PA Subgroup Performance](docs/assets/fig4_ap_vs_pa_subgroups.png)**
- **[Figure 4: Qualitative Multi-Panel Uncertainty & Archetypes](docs/assets/fig5_qualitative_uncertainty_maps.png)** (panels illustrative, pre-P10 checkpoints)

---

## 🌐 Interactive Web Application (GitHub Pages)

A static, pure HTML5, CSS, and JavaScript interactive web application is hosted under `docs/`:
- **Live GitHub Pages URL**: [https://raju-sah.github.io/pneumothorax-segmentation/](https://raju-sah.github.io/pneumothorax-segmentation/)
- **Clinical Triage Studio**: Multi-layer canvas with interactive opacity, crosshairs, and pixel probe HUD.
- **Selective Prediction Simulator**: Interactive coverage slider $\Phi$ with real-time Pareto calculations and SVG risk curves.
- **Scientific Lightbox Gallery & Benchmark Explorer**: Interactive tabs for Tables 1-3 and high-resolution figures.

---

## 📄 Academic Paper Manuscript

The complete conference manuscript formatted for MIDL/MICCAI is available in the [`paper/`](file:///home/raju/AI-ML%20Projects/Pneumothorax%20segmentation/paper/) directory:
- **`paper/main.tex`**: Camera-ready LaTeX document with all empirical metrics, tables, and figures embedded.
- **`paper/references.bib`**: BibTeX bibliography with 15 relevant medical AI and uncertainty quantification citations.


