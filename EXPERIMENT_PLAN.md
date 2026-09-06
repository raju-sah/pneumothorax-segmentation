# EXPERIMENT PLAN: Master Protocols (EXP-01 through EXP-10)

This document specifies the experimental execution protocol for the entire project. Each experiment defines its formal hypothesis, configuration, variables, evaluation metrics, expected interpretation, and explicit failure criteria.

---

## EXP-01: Rigorous Dataset Audit & DICOM Extraction

- **Hypothesis**: The SIIM-ACR dataset contains patient-level repeat scans, mixed photometric interpretations, and multi-lesion annotations that require structured extraction before modeling.
- **Setup**: Run `src/data/dicom_audit.py` over all 12,047 DICOM files in `data/raw/dicom-images-train/`. Extract headers, parse `train-rle.csv`, and generate `data/processed/metadata_master.csv`.
- **Variables**: DICOM attributes (`PatientID`, `ViewPosition`, `PhotometricInterpretation`, `PatientAge`, `PatientSex`, `PixelSpacing`).
- **Metrics**: Missing tag rate, unique patient count, positive/negative ratio, multi-mask instance distribution.
- **Expected Interpretation**: Confirms ~22.15% positive rate, identifies exact number of unique patients, and flags all `MONOCHROME1` instances for inversion.
- **Failure Criteria**: `PatientID` missing or identical across all files, preventing patient-level grouping.

---

## EXP-02: Preprocessing & Leakage-Free Stratified Partition

- **Hypothesis**: Stratified Group K-Fold splitting based on `PatientID` isolates patient scans while preserving the ~22% positive prevalence and AP/PA distribution across all folds.
- **Setup**: Execute `src/data/split.py` to create a 20% patient-isolated test holdout and 5 stratified folds on the remaining 80%. Generate $512 \times 512$ HDF5 cache (`images_512.h5`, `masks_512.h5`).
- **Variables**: Random splitting seed (42), fold count ($k=5$).
- **Metrics**: Patient intersection $|\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}}| = 0$, positive ratio variance across folds $< 1.0\%$.
- **Expected Interpretation**: Verified zero-leakage manifest with identical demographic and projection distributions across folds.
- **Failure Criteria**: Non-zero patient intersection between any train and test/validation sets.

---

## EXP-03: Deterministic Segmentation Baseline

- **Hypothesis**: A standard ResNet34 U-Net trained with $0.5 \cdot \text{BCE} + 0.5 \cdot \text{SoftDice}$ achieves competitive segmentation overlap ($\text{DSC}_{\text{pos}} \ge 0.75$), but exhibits high miscalibration (ESCE $> 0.12$) and silent false positives.
- **Setup**: Train ResNet34 U-Net for 35 epochs on Fold 0 using AdamW ($lr=3e-4$, cosine annealing) and mixed precision (`torch.cuda.amp`).
- **Variables**: Random initialization seeds (42, 43, 44, 45, 46).
- **Metrics**: $\text{DSC}_{\text{pos}}$, $\text{DSC}_{\text{all}}$, IoU, Sensitivity, Specificity, HD95, ESCE, Brier Score.
- **Expected Interpretation**: Strong segmentation overlap on clear pneumothoraces, but overconfident probability estimates on skin folds and rib borders.
- **Failure Criteria**: Validation $\text{DSC}_{\text{pos}} < 0.65$ after 20 epochs.

---

## EXP-04: Monte Carlo Dropout Implementation & Optimization

- **Hypothesis**: Test-time spatial dropout ($p=0.2$) generates spatial variance maps that correlate with ambiguous lesion boundaries without degrading raw segmentation accuracy.
- **Setup**: Train ResNet34 U-Net with `nn.Dropout2d(p=0.2)` in bottleneck and decoder blocks. Sample $T \in \{5, 10, 20, 30\}$ stochastic passes at inference time.
- **Variables**: Sample count $T$, dropout rate $p \in \{0.1, 0.2, 0.3\}$.
- **Metrics**: Predictive Variance, Predictive Entropy, $\text{DSC}_{\text{pos}}$, ESCE, inference latency per image.
- **Expected Interpretation**: Variance concentrates along pleural margins; $T=20$ yields optimal trade-off between variance stability and inference speed.
- **Failure Criteria**: MC Dropout reduces $\text{DSC}_{\text{pos}}$ by $> 0.05$ compared to deterministic baseline.

---

## EXP-05: Deep Ensemble Formulation ($M=5$)

- **Hypothesis**: Deep Ensembles capture epistemic uncertainty through diverse optimization trajectories, outperforming MC Dropout in calibration and boundary delineation.
- **Setup**: Train 5 identical ResNet34 U-Nets across independent seeds (42–46). Compute ensemble mean prediction, mutual information, and ensemble variance.
- **Variables**: Ensemble size $M \in \{2, 3, 4, 5\}$.
- **Metrics**: Ensemble $\text{DSC}_{\text{pos}}$, HD95, ESCE, Brier Score, Mutual Information.
- **Expected Interpretation**: Ensemble improves $\text{DSC}_{\text{pos}}$ by $+0.02 - 0.04$ over single baseline and reduces ESCE by $> 30\%$.
- **Failure Criteria**: Mutual information is uniformly zero across all images (indicating identical local minima).

---

## EXP-06: Uncertainty / Error Correlation Analysis

- **Hypothesis**: Epistemic uncertainty metrics (Mutual Information in Ensembles, Predictive Variance in MC Dropout) achieve higher error-detection AUROC ($\text{AUROC-ED} \ge 0.82$) than naive deterministic entropy ($\le 0.73$).
- **Setup**: Compute absolute error maps $E = |Y - \hat{Y}|$ across the test set. Evaluate pixel-level Spearman rank correlation $\rho$ and AUROC-ED for each uncertainty formulation.
- **Variables**: Uncertainty formulation (Deterministic Entropy, MC Variance, Ensemble Mutual Information).
- **Metrics**: Spearman $\rho(U, E)$, AUROC-ED.
- **Expected Interpretation**: Ensemble Mutual Information displays the strongest correlation with true segmentation errors.
- **Failure Criteria**: AUROC-ED $\le 0.50$ (uncertainty exhibits no discriminative power for errors).

---

## EXP-07: Calibration & Temperature Scaling

- **Hypothesis**: Deep Ensembles inherently calibrate predictions, and post-hoc temperature scaling further minimizes Expected Segmentation Calibration Error (ESCE).
- **Setup**: Construct reliability diagrams (10 bins). Fit scalar and vector temperature scaling on the validation split; evaluate on the test split.
- **Variables**: Uncalibrated vs. Temperature-scaled vs. Ensembled predictions.
- **Metrics**: ESCE, Brier Score, Reliability Diagram slope and intercept.
- **Expected Interpretation**: Ensembling achieves lower ESCE than temperature-scaled single models.
- **Failure Criteria**: Temperature scaling causes probability collapse to uniform distribution.

---

## EXP-08: Selective Prediction & Clinical Referral Simulation

- **Hypothesis**: Referring radiographs with top-$K$ pixel variance to human radiologists produces an empirical Risk-Coverage curve that improves retained cohort Dice significantly over random or deterministic referral.
- **Setup**: Aggregate image-level uncertainty via top-$K$ pixel variance ($K=500$). Sweep retained coverage $c \in [0.2, 1.0]$. Compute retained cohort Dice and AURC.
- **Variables**: Uncertainty aggregation metric, referral selection policy (Random, Deterministic Entropy, MC Dropout, Deep Ensemble).
- **Metrics**: Risk-Coverage Curve, Area Under Risk-Coverage (AURC), Retained Dice at $c \in \{0.7, 0.8, 0.9\}$.
- **Expected Interpretation**: Deep Ensemble achieves the lowest AURC; retaining the 80% most confident cases boosts $\text{DSC}_{\text{pos}}$ by $> +0.08$.
- **Failure Criteria**: Risk-Coverage curve is flat or increasing (referring uncertain cases does not reduce cohort error).

---

## EXP-09: Radiological Difficulty Analysis & Anatomical Stratification

- **Hypothesis**: Predictive uncertainty is significantly elevated on subtle small pneumothoraces ($<2\%$ hemithorax) and portable AP projections compared to large lesions and PA views.
- **Setup**: Stratify the test set by: (a) Lesion size ($<2\%$, $2-10\%$, $>10\%$), (b) Projection (`ViewPosition`: AP vs. PA), and (c) Rib/clavicle intersection zones.
- **Variables**: Clinical difficulty strata.
- **Metrics**: Subgroup $\text{DSC}_{\text{pos}}$, Subgroup Mean Uncertainty, Subgroup AUROC-ED, two-sided Wilcoxon test $p$-values.
- **Expected Interpretation**: AP views exhibit significantly higher background uncertainty ($p < 0.01$); small lesions exhibit peak boundary variance.
- **Failure Criteria**: Subtle missed pneumothoraces produce zero uncertainty (silent false negatives).

---

## EXP-10: Robustness, Ablations & Negative-Control OOD Evaluation

- **Hypothesis**: $512 \times 512$ resolution is necessary to resolve subtle visceral pleural lines, and the uncertainty framework safely flags out-of-distribution external radiographs with high epistemic uncertainty.
- **Setup**: (a) Train ResNet34 U-Net at $256 \times 256$ and compare metrics with $512 \times 512$; (b) Evaluate uncertainty models on external normal CXRs (from CheXpert).
- **Variables**: Input resolution ($256$ vs $512$), In-distribution test vs. Out-of-distribution external radiographs.
- **Metrics**: $\Delta \text{DSC}_{\text{pos}}$, OOD Rejection AUROC (separating in-distribution pneumothorax cases from external normal radiographs based on uncertainty).
- **Expected Interpretation**: $512 \times 512$ improves small lesion Dice by $> 10\%$; external normal CXRs trigger near-zero predicted masks and high epistemic flags when foreign artifacts are present.
- **Failure Criteria**: Model predicts high-confidence false-positive pneumothoraces across external normal radiographs.
