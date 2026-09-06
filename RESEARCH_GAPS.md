# RESEARCH GAPS & SCIENTIFIC NOVELTY

## 1. What Has Already Been Done on SIIM-ACR

A survey of published papers utilizing the SIIM-ACR Pneumothorax dataset reveals consistent patterns:
1. **Raw Leaderboard Maximization**: The overwhelming majority of papers focus exclusively on maximizing global Dice Similarity Coefficient (DSC) on test sets through increasingly complex vision backbones (e.g., Swin Transformer, TransUNet, SegFormer, EfficientNet-B7 ensembles).
2. **Qualitative Uncertainty Demonstrations**: A small number of studies apply Monte Carlo Dropout to medical segmentation models and display qualitative variance heatmaps for 2–3 cherry-picked examples without quantitative error correlation or calibration metrics.
3. **Absence of Clinical Referral Modeling**: Almost no studies investigate how uncertainty scores translate into operational referral decisions in clinical triage workflows.

---

## 2. What We Explicitly Do NOT Claim as Novelty

To adhere to rigorous scientific integrity, we explicitly state:
- **Combining U-Net with MC Dropout is NOT novel**: MC Dropout (Gal & Ghahramani, 2016) has been applied to medical imaging for nearly a decade.
- **Combining U-Net with Deep Ensembles is NOT novel**: Ensembling multiple networks (Lakshminarayanan et al., 2017) is a well-established baseline.
- **Reporting Dice and ECE is NOT novel**: Standard benchmark reporting does not constitute an original scientific contribution.

---

## 3. Defensible Research Gaps Addressed by This Project

This project addresses three specific, clinically actionable, and academically rigorous research gaps:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THREE CORE RESEARCH GAPS                                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ GAP 1: Dense-to-Case Selective Prediction for Pneumothorax Triage                     │
│ Gap: Existing selective prediction is formalized for classification; medical           │
│ segmentation models lack calibrated mechanisms to triage entire radiographs to human   │
│ radiologists based on spatially aggregated epistemic uncertainty.                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ GAP 2: The Disconnect Between Pixel Calibration and Case Referral Utility             │
│ Gap: Models with low Expected Segmentation Calibration Error (ESCE) may still fail to   │
│ rank clinically dangerous failure modes (e.g. false positives on skin folds) higher    │
│ than benign boundary noise. We quantify this divergence.                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ GAP 3: Epistemic Sensitivity to Lesion Scale and Anatomical Acquisition (AP vs. PA)   │
│ Gap: Prior studies treat CXRs homogeneously. We formally dissect whether uncertainty   │
│ increases for subtle hairline apical lesions (<2% hemithorax) and portable AP views.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Gap 1: Dense-to-Case Selective Prediction Formulation
While selective classification has been explored, segmenting life-threatening conditions requires translating dense $512 \times 512$ uncertainty distributions into actionable image-level referral decisions. We establish and benchmark top-$K$ pixel variance aggregation against mean entropy to build empirical **Risk-Coverage Pareto frontiers** for emergency chest radiography.

### Gap 2: Disconnect Between Calibration Metrics and Referral Utility
A standard assumption in the uncertainty literature is that lower calibration error (e.g., lower ESCE or Brier Score) automatically translates to superior clinical triage. We test whether this assumption holds, comparing whether post-hoc temperature scaling (which minimizes ESCE) actually improves the Area Under the Risk-Coverage Curve (AURC) in selective prediction.

### Gap 3: Anatomical Disambiguation of Radiographic Confounders
Pneumothorax false alarms frequently originate from benign anatomical artifacts (e.g., scapular margins, skin folds, rib intersections). We investigate whether epistemic uncertainty measures (mutual information in Deep Ensembles and predictive variance in MC Dropout) elevate specifically on these mimics, providing an interpretability safeguard against unnecessary chest tube interventions.
