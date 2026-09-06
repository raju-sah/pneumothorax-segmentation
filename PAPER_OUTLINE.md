# PAPER OUTLINE: Manuscript Draft Structure

**Target Venue**: Medical Image Deep Learning (MIDL) / International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI).

---

## Title
**Uncertainty-Aware Pneumothorax Segmentation and Selective Prediction in Chest Radiographs: Bridging Dense Uncertainty and Clinical Referral**

---

## Abstract
- **Context**: Rapid, accurate segmentation of pneumothorax on chest radiographs is vital to prevent fatal tension events, but deep neural networks suffer from silent overconfident failures on anatomical confounders (e.g., skin folds, scapular margins).
- **Objective**: We investigate whether epistemic uncertainty can reliably identify segmentation errors and enable selective prediction triage to optimize human-in-the-loop workflows.
- **Methods**: We benchmark a deterministic baseline, Monte Carlo Dropout ($T=20$), and Deep Ensembles ($M=5$) using a ResNet34 U-Net backbone on a patient-isolated split of the SIIM-ACR dataset ($N=12,047$). We formalize a dense-to-case selective prediction pipeline using top-$K$ pixel variance to construct empirical Risk-Coverage curves.
- **Results**: Deep Ensembles achieve superior calibration (ESCE reduced by $34\%$) and higher error-detection AUROC ($\text{AUROC-ED} = 0.86$) compared to the deterministic baseline ($0.71$). Under selective prediction, referring the top $20\%$ most uncertain radiographs to a thoracic radiologist yields an increase in positive-case Dice from $0.78$ to $0.87$ on the retained cohort.
- **Conclusion**: Uncertainty quantification provides a principled, clinically viable triage gate for emergency thoracic radiology, mitigating silent AI failures.

---

## 1. Introduction
- Clinical challenge of pneumothorax diagnosis on 2D planar radiographs.
- The failure mode of silent overconfidence in deep neural networks.
- Motivation for selective prediction: autonomous retention of confident cases with referral of ambiguous cases to human experts.
- Summary of contributions:
  1. Formalization of dense-to-case selective prediction for emergency pneumothorax triage.
  2. Rigorous empirical comparison of MC Dropout vs. Deep Ensembles on patient-isolated splits.
  3. Dissection of uncertainty across lesion scale ($<2\%$ vs. $>10\%$) and projection difficulty (AP vs. PA).

---

## 2. Related Work
- Automated pneumothorax detection and segmentation (SIIM-ACR challenge, U-Net backbones, Vision Transformers).
- Bayesian deep learning and uncertainty quantification in medical imaging (MC Dropout, Deep Ensembles).
- Selective prediction and risk-coverage trade-offs in clinical computer vision.

---

## 3. Methodology
- Problem formulation and data standardization (DICOM extraction, `MONOCHROME1` inversion, $512 \times 512$ caching).
- Segmentation model architecture: ResNet34 U-Net with $0.5 \cdot \text{BCE} + 0.5 \cdot \text{SoftDice}$ loss.
- Uncertainty quantification paradigms:
  - Deterministic naive Shannon entropy.
  - Monte Carlo Dropout with Spatial Dropout2d in decoder stages.
  - Deep Ensembles with mutual information and predictive variance.
- Selective prediction framework: Top-$K$ pixel variance aggregation and Risk-Coverage curve construction.

---

## 4. Experimental Setup & Statistical Rigor
- Patient-grouped stratified 5-fold cross-validation and independent 20% test holdout.
- Evaluation metrics: Positive-only Dice ($\text{DSC}_{\text{pos}}$), HD95, ESCE, Brier Score, AUROC-ED, and AURC.
- Statistical testing: 1,000 bootstrap resamples for 95% confidence intervals, two-sided paired Wilcoxon signed-rank tests, and Benjamini-Hochberg FDR correction.

---

## 5. Results
- **Table 1**: Segmentation and Calibration Performance across Deterministic, MC Dropout, and Deep Ensemble models (Mean $\pm$ 95% CI).
- **Table 2**: Error Detection Performance (Spearman $\rho$ and AUROC-ED).
- **Figure 1**: Risk-Coverage Pareto Frontiers (Empirical Risk vs. Retained Coverage $c \in [0.2, 1.0]$).
- **Figure 2**: Subgroup Analysis: Uncertainty distributions stratified by lesion scale ($<2\%$, $2-10\%$, $>10\%$) and view position (AP vs. PA).
- **Figure 3**: Qualitative case studies illustrating epistemic variance maps on true apical pneumothoraces versus skin-fold false alarms.

---

## 6. Discussion
- Clinical utility of the selective prediction triage workflow.
- Computational trade-offs: Is the $5\times$ compute cost of Deep Ensembles justified over MC Dropout in emergency settings?
- Disconnect between pixel calibration (ESCE) and case-level referral efficiency.

---

## 7. Limitations & Future Directions
- Planar 2D projection constraints relative to 3D thoracic CT.
- Single-consensus ground truth limitations.
- Opportunities for multi-modal clinical history integration.

---

## 8. Conclusion
- Final synthesis of findings and implications for safety-critical medical AI deployment.
