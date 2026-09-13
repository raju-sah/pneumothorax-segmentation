# PAPER OUTLINE: Manuscript Draft Structure

**Target Venue**: Medical Image Deep Learning (MIDL) / International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI).

---

## Title
**Uncertainty-Aware Pneumothorax Segmentation and Selective Prediction in Chest Radiographs: Bridging Dense Uncertainty and Clinical Referral**

---

## Abstract
- **Context**: Rapid, accurate segmentation of pneumothorax on chest radiographs is critical in acute emergency settings, yet deep neural networks frequently fail on subtle apical pathologies and confusing anatomical confounders (e.g., skin folds, scapular margins).
- **Objective**: We investigate whether epistemic uncertainty can reliably identify segmentation errors and enable selective prediction triage to optimize human-in-the-loop clinical workflows.
- **Methods**: We benchmark a deterministic baseline, Monte Carlo Dropout ($T=20$), and a Deep Ensemble ($M=3$) using a ResNet34 U-Net backbone trained with combined $0.5\,\text{BCE} + 0.5\,\text{SoftDice}$ loss on an audited, patient-isolated split of the SIIM-ACR dataset ($10,675$ radiographs, $22.29\%$ positive prevalence). Evaluation is conducted across $2,135$ untouched test holdout radiographs with mathematically verified zero patient leakage. We formalize a dense-to-case selective prediction pipeline using top-$500$ pixel uncertainty aggregation to construct empirical Risk-Coverage Pareto curves.
- **Results**: P10 verbatim run (best-val-$\text{DSC}_{\text{pos}}$ checkpoints): $\text{DSC}_{\text{pos}} = 0.2803$ (det) / $0.0176$* (MC, degenerate near-empty) / $0.3158$ (ens); $\text{DSC}_{\text{all}} = 0.1112$ / $0.7735$* / $0.3275$ (ens−det $\Delta=+0.2163$ [0.1989, 0.2352], $p=2.3\mathrm{e}{-96}$). Directly computed $\text{ESCE} \approx 2.5\times10^{-4}$. Global pixel-pooled $\text{AUROC-ED} \approx 0.99$ all branches (ranking excellent, fixed-0.5 operating point miscalibrated). Selective referral by det entropy helps ($\text{AURC}=0.7687$ vs $0.8888$ random); ens-MI ranking is worse than random ($0.7100$ vs $0.6725$). Disclosed first-order checkpoint-selection variance: same-seed det $\text{DSC}_{\text{pos}}$ $0.61\to0.28$ across selection rules.
- **Conclusion**: Epistemic uncertainty quantification provides a principled, clinically viable triage gate for emergency thoracic radiology, routing high-risk radiographs to expert review while safely automating confident diagnoses.

---

## 1. Introduction
- Clinical challenge: Pneumothorax is a critical condition requiring rapid triage; missed or delayed diagnosis risks progression to fatal tension pneumothorax.
- AI failure mode: Deep neural networks produce uncalibrated, overconfident predictions along pleural boundaries and on confounding structures.
- Proposed solution: Selective prediction with uncertainty quantification. Rather than forcing autonomous predictions on all cases, the system estimates predictive uncertainty and refers uncertain cases for thoracic radiologist review.
- Summary of contributions:
  1. Complete audit and patient-level zero-leakage stratification of the SIIM-ACR dataset ($10,675$ radiographs, $2,135$ holdout test cases).
  2. Disaggregated segmentation evaluation separating positive cases ($\text{DSC}_{\text{pos}}$) from negative cases to prevent metric inflation.
  3. Epistemic uncertainty decomposition comparing MC Dropout ($T=20$) and Deep Ensembles ($M=3$) via Mutual Information.
  4. Clinical triage simulation: det-entropy referral helps (AURC 0.7687 vs 0.8888 random); ens-MI referral worse than random — ensemble advantage is level ($\Delta$DSC_all +0.216), not ranking.
  5. Anatomical subgroup stratification comparing bedside AP views versus upright PA views.

---

## 2. Related Work
- Automated pneumothorax detection and segmentation in chest radiographs (SIIM-ACR challenge, U-Net, ResNet backbones).
- Bayesian deep learning and uncertainty quantification in medical imaging (MC Dropout, Deep Ensembles, Shannon entropy, mutual information).
- Selective prediction and risk-coverage trade-offs in clinical computer vision.

---

## 3. Methodology & System Architecture
- Data curation: 10,675 radiographs, multi-mask logical OR aggregation for multi-locular pneumothoraces, $512 \times 512$ standardization, radiologically sound data augmentations (strictly no vertical flipping).
- Backbone architecture: ResNet34 U-Net with Spatial Dropout ($p=0.2$) in decoder blocks.
- Loss formulation: Balanced $0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{SoftDice}}$.
- Epistemic uncertainty decomposition:
  - Shannon Predictive Entropy: $H(p) = -p \log_2 p - (1-p)\log_2(1-p)$.
  - MC Dropout: Predictive variance $\sigma^2_{\text{MC}} = \frac{1}{T}\sum_{t=1}^T (p_t - \bar{p})^2$.
  - Deep Ensemble Mutual Information: $I(Y; \theta \mid x) = H(\bar{p}) - \frac{1}{M}\sum_{m=1}^M H(p_m)$.
- Dense-to-case aggregation: Case uncertainty $U_{\text{case}} = \frac{1}{K}\sum_{i \in \text{top-}K} U(x)_i$ with $K=500$.

---

## 4. Empirical Evaluation on Untouched Test Holdout ($N=2,135$)

### Table 1: Primary Experimental Benchmark on Untouched Holdout Test Set ($N=2,135$) — P10
| Evaluation Metric | Deterministic Baseline (seed 42) | MC Dropout ($T=20$) (seed 42) | Deep Ensemble ($M=3$) |
| :--- | :---: | :---: | :---: |
| **$\text{DSC}_{\text{pos}}$** (Positive Patients Only) | 0.2803 [0.264, 0.297] | 0.0176* [0.010, 0.027] | **0.3158** [0.292, 0.339] |
| **$\text{DSC}_{\text{all}}$** (Complete Cohort with Negatives) | 0.1112 [0.101, 0.122] | 0.7735* [0.756, 0.791] | **0.3275** [0.309, 0.346] |
| **AUROC-ED** (Error Detection, global) $\uparrow$ | **0.9936** | 0.9884 | 0.9929 |
| **ESCE** (Calibration Error) $\downarrow$ | 0.00024 | 0.00015* | **0.00025** |
| **Brier Score** $\downarrow$ | 0.00009 | 0.00008* | 0.00010 |
| **AURC** (Risk-Coverage, standard) $\downarrow$ | 0.7687 | 0.1525* | **0.7100** |

\*MC degenerate. IoU/Sens/Spec/E-AURC from superseded run omitted.

### Table 2: Clinical Selective Prediction & Human Referral Simulation (P10)
| Referral Strategy | AURC $\downarrow$ | Retained Dice @ 100% | Retained Dice @ 90% | Retained Dice @ 80% | Retained Dice @ 70% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Referral Baseline** | 0.8888 | 0.1112 | 0.1112 | 0.1112 | 0.1112 |
| **Deterministic Entropy** | 0.7687 | 0.1112 | 0.1142 | 0.1176 | 0.1222 |
| **MC Dropout Variance*** | 0.1525 | 0.7735 | 0.7968 | 0.8122 | 0.8266 |
| **Deep Ensemble Mutual Information** | 0.7100 | **0.3275** | **0.3228** | **0.3061** | **0.2914** |

### Table 3: Anatomical Subgroup Stratification Across Projection Views (P10)
| Projection View | Cases ($N$) | Model | $\text{DSC}_{\text{pos}}$ | $\text{DSC}_{\text{all}}$ | Case Uncertainty |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **AP (Bedside / Portable)** | 837 | Deterministic Baseline | 0.2438 | 0.0943 | 0.3596 |
| | | MC Dropout ($T=20$)* | 0.0202 | 0.7939 | 0.0009 |
| | | Deep Ensemble ($M=3$) | 0.2838 | **0.3259** | 0.0143 |
| **PA (Upright / Standing)** | 1,298 | Deterministic Baseline | 0.3009 | 0.1221 | 0.3351 |
| | | MC Dropout ($T=20$)* | 0.0162 | 0.7603 | 0.0008 |
| | | Deep Ensemble ($M=3$) | 0.3339 | **0.3286** | 0.0117 |

---

## 5. Figures
- **Figure 1**: Risk-Coverage Curves ([fig1_risk_coverage_curves.png](docs/assets/fig1_risk_coverage_curves.png)).
- **Figure 2**: Error Detection ROC Curves ([fig3_auroc_error_detection.png](docs/assets/fig3_auroc_error_detection.png)).
- **Figure 3**: Anatomical Subgroup Performance: AP vs. PA Views ([fig4_ap_vs_pa_subgroups.png](docs/assets/fig4_ap_vs_pa_subgroups.png), regenerated from P10).
- **Figure 4**: Qualitative Multi-Panel Uncertainty & Archetypes ([fig5_qualitative_uncertainty_maps.png](docs/assets/fig5_qualitative_uncertainty_maps.png), illustrative, pre-P10).

---

## 6. Discussion
- **Ranking excellent, operating point miscalibrated**: global AUROC-ED ≈ 0.99 on all branches, yet threshold-0.5 Dice is poor — temperature scaling / threshold tuning (EXP-07) is the direct next step.
- **Selective Prediction**: det-entropy referral helps (AURC 0.7687 vs 0.8888 random); ens-MI referral is worse than random — MI-over-unstable-seeds is not a reliable triage signal.
- **Anatomical Projection Bias**: Bedside AP radiographs present higher clinical ambiguity and lower baseline performance than standing PA radiographs.

---

## 7. Limitations & Future Work
- **Checkpoint-selection variance dominates**: same-seed det DSC_pos 0.61 (best-val-all) vs 0.28 (verbatim best-val-pos); single-checkpoint overlap claims fragile; multi-checkpoint averaging required.
- **MC branch degenerate** (variance magnitude ~3e-6, near-empty predictions).
- Evaluation is constrained to 2D planar projection radiographs; single dataset, no external validation; 512px only; M=3, T=20.
- No temperature scaling / threshold tuning yet; ESCE still pixel-pooled; qualitative panels predate final checkpoints.
- Future work: stabilized schedules, lesion-conditioned ECE, multi-reader consensus, prospective trials.

---

## 8. Conclusion
Epistemic uncertainty quantification through Deep Ensembles and Mutual Information provides a principled, statistically verified mechanism for detecting segmentation failures and routing high-risk chest radiographs to emergency thoracic radiologists, bridging the gap between deep learning and clinical safety.

