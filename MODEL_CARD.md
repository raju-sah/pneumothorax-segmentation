# MODEL CARD: Uncertainty-Aware ResNet34 U-Net for Pneumothorax Segmentation

Following the standardized model card framework proposed by Mitchell et al. (2019).

---

## 1. Model Details
- **Model Name**: Uncertainty-Aware ResNet34 U-Net (Deterministic, MC Dropout, and Deep Ensemble variants).
- **Architecture**: U-Net with ImageNet-pretrained ResNet34 encoder; Spatial Dropout (`p=0.2`) in decoder/bottleneck for MC Dropout; $M=5$ models for Deep Ensemble.
- **Input Resolution**: $512 \times 512 \times 1$ (Grayscale chest radiograph, normalized $[0, 1]$).
- **Output**: Pixel-wise probability map $\hat{p} \in [0, 1]^{512 \times 512}$ and spatial epistemic uncertainty map $U(x) \in \mathbb{R}^{512 \times 512}$.
- **Loss Function**: $0.5 \cdot \text{BCE} + 0.5 \cdot \text{SoftDice}$.
- **Primary Use**: Academic research portfolio project investigating uncertainty quantification, error detection, and selective prediction for thoracic radiography.

---

## 2. Intended Use & Clinical Scope
- **Intended Use Case**: Automated quality assurance and selective triage in emergency radiology. Confident predictions are displayed to attending clinicians; high-uncertainty cases are automatically flagged and prioritized for human expert re-reading.
- **Out-of-Scope / Prohibited Uses**:
  - Direct autonomous patient management or unilateral chest tube placement without radiologist verification.
  - Deployment on pediatric neonates (neonatal pneumothorax exhibits distinct radiographic anatomy).
  - Cross-sectional imaging (CT or MRI).

---

## 3. Factors & Subgroup Stratification
The model is explicitly benchmarked across:
- **Radiographic Projection**: AP (portable bedside) vs. PA (standard erect).
- **Lesion Scale**: Small ($<2\%$ hemithorax), Medium ($2-10\%$), Large ($>10\%$).
- **Demographic Cohorts**: Patient sex (Male vs. Female) and age brackets.

---

## 4. Performance Metrics
- **Segmentation**: Positive-only Dice ($\text{DSC}_{\text{pos}}$), Overall Dice ($\text{DSC}_{\text{all}}$), IoU, Sensitivity, Specificity, Hausdorff Distance 95% ($\text{HD95}$).
- **Uncertainty Calibration**: Expected Segmentation Calibration Error (ESCE), Brier Score.
- **Error Detection**: Spearman rank correlation $\rho$, AUROC for Error Detection (AUROC-ED).
- **Selective Prediction**: Risk-Coverage Curve ($\mathcal{R}(c)$), Area Under Risk-Coverage (AURC), Retained Dice at coverage $c \in \{0.7, 0.8, 0.9\}$.

---

## 5. Training & Evaluation Data
- **Training Cohort**: 80% patient-grouped development partition of the 12,047 Stage 2 SIIM-ACR dataset.
- **Evaluation Cohort**: 20% untouched patient-held-out test partition (~2,410 radiographs) with guaranteed zero patient overlap.

---

## 6. Ethical Considerations & Risk Mitigation
- **The Silent Failure Risk**: A standard deep learning model that fails silently on a subtle tension pneumothorax risks patient death. Incorporating epistemic uncertainty provides an intrinsic safety mechanism, explicitly conveying model hesitation.
- **Bias Mitigation**: Grouped stratification ensures portable AP radiographs (predominantly from sicker, immobilized patients) are equally represented during model training and evaluation.
