# RESEARCH PLAN: Uncertainty-Aware Pneumothorax Segmentation with Selective Prediction

## 1. Executive Summary & Research Vision
This project investigates whether predictive uncertainty in deep neural networks can reliably detect clinical segmentation errors in chest radiographs (CXRs) and serve as an automated triage mechanism for selective prediction (human-in-the-loop clinical referral).

- **Target Clinical Problem**: Pneumothorax segmentation in 2D chest radiographs.
- **Dataset**: SIIM-ACR Pneumothorax Segmentation challenge dataset (Stage 2 annotated cohort).
- **Core Architecture**: ResNet34 U-Net ($512 \times 512$ resolution) with $0.5 \cdot \text{BCE} + 0.5 \cdot \text{SoftDice}$ loss.
- **Uncertainty Paradigms**: Deterministic Baseline vs. Monte Carlo (MC) Dropout ($T=20$) vs. Deep Ensembles ($M=5$).
- **Translational Deliverable**: Empirical Risk-Coverage Pareto frontiers demonstrating quantitative diagnostic gains when referring uncertain radiographs to thoracic radiologists.

---

## 2. Research Questions & Hypotheses

### Central Research Question
> *"Can predictive uncertainty identify where a deep learning model is likely to produce unreliable pneumothorax segmentation predictions, and can uncertainty support selective prediction for human review?"*

### Secondary Research Questions & Formal Hypotheses

- **RQ1: Deterministic vs. Uncertainty-Aware Inference**
  - *Question*: How does uncertainty-aware inference compare with a deterministic segmentation baseline in raw segmentation accuracy and boundary delineation?
  - *Hypothesis ($H_1$)*: Deep Ensembles will outperform the deterministic baseline in positive-case Dice ($\Delta \text{DSC} \ge +0.02$) and 95th-percentile Hausdorff Distance due to variance reduction across diverse non-convex loss minima, whereas MC Dropout will achieve comparable overlap to the baseline.

- **RQ2: Uncertainty / Error Correlation**
  - *Question*: Does predictive uncertainty correlate spatially and image-wise with actual segmentation errors?
  - *Hypothesis ($H_2$)*: Epistemic uncertainty (predictive variance in MC Dropout; mutual information in Deep Ensembles) will yield significantly higher Area Under the ROC Curve for Error Detection ($\text{AUROC-ED} \ge 0.82$) than naive deterministic entropy ($\text{AUROC-ED} \le 0.73$).

- **RQ3: Identification of Clinical Failure Modes**
  - *Question*: Can uncertainty differentiate between true pathology and common radiological confounders (skin folds, scapular borders, rib lines)?
  - *Hypothesis ($H_3$)*: Radiographic confounders that induce false positive predictions will exhibit elevated epistemic uncertainty, reflecting model ignorance of non-parenchymal artifacts.

- **RQ4: Lesion Scale & Anatomical Difficulty**
  - *Question*: Does predictive uncertainty increase for small or visually subtle pneumothoraces compared to gross collapsed lungs?
  - *Hypothesis ($H_4$)*: Aggregated uncertainty across the hemithorax will correlate negatively with lesion size (Spearman $\rho \le -0.40$), with subtle apical pneumothoraces ($< 2\%$ of hemithorax) showing peak boundary entropy.

- **RQ5: Selective Prediction & Human Referral**
  - *Question*: Can predictive uncertainty support selective prediction, where uncertain cases are referred for human review while confident predictions are autonomously retained?
  - *Hypothesis ($H_5$)*: Selective prediction based on top-$K$ pixel variance will achieve superior Area Under the Risk-Coverage curve ($\text{AURC}$) compared to random referral or deterministic confidence ranking, yielding a Dice gain $> +0.08$ at $80\%$ retained coverage ($c = 0.80$).

- **RQ6: MC Dropout vs. Deep Ensembles**
  - *Question*: Which uncertainty approach is more reliable in clinical triage: MC Dropout or Deep Ensembles?
  - *Hypothesis ($H_6$)*: Deep Ensembles ($M=5$) will achieve statistically superior calibration ($\text{ESCE}$) and lower $\text{AURC}$ than MC Dropout ($T=20$) ($p < 0.01$, Wilcoxon signed-rank test), justifying their $5\times$ training compute overhead for safety-critical deployment.

---

## 3. Scientific Scoping & Delimitations

1. **No External Segmentation Domain Shift**:
   - The NIH ChestX-ray14 dataset is **not** an independent segmentation dataset; SIIM-ACR is a subset of NIH ChestX-ray14 re-annotated by SIIM/ACR/STR radiologists.
   - NIH ChestX-ray14 lacks pixel-level pneumothorax masks.
   - Therefore, domain shift is evaluated internally via strict patient-isolated holdout sets, with out-of-distribution (OOD) tests restricted to negative-control rejection tasks (e.g., normal CXRs from CheXpert).

2. **Strict Patient-Level Partitioning**:
   - Random splitting is strictly prohibited to prevent data leakage from multiple radiographs of the same patient.
   - Partitioning is grouped by `PatientID` extracted from DICOM metadata.

3. **Separation of Positive and Negative Case Evaluation**:
   - Since ~78% of radiographs are negative (no pneumothorax), reporting a single aggregate Dice score (where empty ground truth + empty prediction = 1.0) artificially inflates performance to $>0.90$.
   - Metrics are strictly disaggregated into:
     - Positive-only Dice ($\text{DSC}_{\text{pos}}$).
     - Overall Dice ($\text{DSC}_{\text{all}}$).
     - Image-level Classification AUROC / Precision / Recall.
     - Pixel-level boundary Hausdorff Distance 95% ($\text{HD95}$).

4. **Realistic Compute Constraints**:
   - Designed for 8–10 hours/week part-time research effort.
   - GPU budget: 30 hours/week free Kaggle GPU allocation (NVIDIA T4/P100).
   - Local execution: Code verification, CPU-based unit tests, analysis, and paper drafting.
