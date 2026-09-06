# EVALUATION METRICS: Mathematical Formulations & Clinical Interpretations

## 1. Segmentation Overlap & Boundary Metrics

### 1.1 Dice Similarity Coefficient (DSC)
Given binary ground truth $Y \in \{0, 1\}^{H \times W}$ and thresholded binary prediction $\hat{Y} = \mathbb{I}(\bar{p} > 0.5)$:
$$\text{DSC}(Y, \hat{Y}) = \frac{2 \sum_{i=1}^N Y_i \hat{Y}_i}{\sum_{i=1}^N Y_i + \sum_{i=1}^N \hat{Y}_i}$$

#### Edge Case Resolution (Negative Cases)
In the SIIM-ACR dataset, ~78% of radiographs are negative ($\sum Y_i = 0$):
- **Both $Y$ and $\hat{Y}$ empty**: $\sum Y_i = 0$ and $\sum \hat{Y}_i = 0 \implies \text{DSC} = 1.0$ (correct negative prediction).
- **$Y$ empty, $\hat{Y}$ non-empty**: $\sum Y_i = 0$ and $\sum \hat{Y}_i > 0 \implies \text{DSC} = 0.0$ (false positive alarm).
- **$Y$ non-empty, $\hat{Y}$ empty**: $\sum Y_i > 0$ and $\sum \hat{Y}_i = 0 \implies \text{DSC} = 0.0$ (missed pneumothorax).

#### Disaggregated Reporting
To prevent artificial inflation from negative cases, we report:
1. **$\text{DSC}_{\text{pos}}$ (Positive-Only Dice)**: Computed strictly over images with confirmed pneumothorax ($|Y| > 0$).
2. **$\text{DSC}_{\text{all}}$ (Overall Challenge Dice)**: Evaluated across all images including correct negative predictions.

---

### 1.2 Hausdorff Distance 95th Percentile (HD95)
Measures the maximum distance of a set to the nearest point in the other set, trimmed at the 95th percentile to eliminate outlier speckle noise:
$$d_H(A, B) = \max \left( \sup_{a \in A} \inf_{b \in B} d(a, b), \sup_{b \in B} \inf_{a \in A} d(a, b) \right)$$
- **Clinical Rationale**: Dice measures volumetric overlap, which can be high even if the model predicts distant disconnected false-positive lesions. HD95 penalizes anatomically nonsensical predictions (e.g., false-positive flags on the contralateral chest wall).
- **Empty Mask Handling**: If either $Y$ or $\hat{Y}$ is empty when the other is non-empty, HD95 is penalized with the maximum bounding diagonal ($512\sqrt{2} \approx 724$ pixels).

---

### 1.3 Sensitivity, Specificity, Precision
- **Sensitivity (Recall)**: $\frac{TP}{TP + FN}$ — Fraction of true pneumothorax pixels identified.
- **Specificity**: $\frac{TN}{TN + FP}$ — Fraction of normal parenchyma correctly rejected.
- **Precision (PPV)**: $\frac{TP}{TP + FP}$ — Reliability of positive pixel alarms.

---

## 2. Uncertainty Calibration Metrics

### 2.1 Expected Segmentation Calibration Error (ESCE)
Evaluates whether predicted probabilities match empirical observation frequencies across $B = 10$ bins:
$$\text{ESCE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
Where:
$$\text{conf}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} \hat{p}_i, \quad \text{acc}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} y_i$$

### 2.2 Brier Score
The mean squared difference between predicted probabilities and ground truth:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
A strictly proper scoring rule where lower values denote simultaneously better accuracy and sharper calibration.

---

## 3. Error Detection & Correlation Metrics

### 3.1 Spearman Rank Correlation ($\rho$)
Computes the rank-order correlation between the uncertainty map $U(x)$ and the absolute pixel error $E(x) = |Y(x) - \hat{Y}(x)|$:
$$\rho = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$
A positive $\rho$ indicates that as uncertainty rises, true segmentation errors increase monotonically.

### 3.2 AUROC for Error Detection (AUROC-ED)
Treats the binary error map $E(x) \in \{0, 1\}$ as ground truth and the uncertainty map $U(x)$ as the continuous prediction score:
- **Interpretation**: $\text{AUROC-ED} = 1.0$ means high uncertainty perfectly predicts every mistaken pixel; $\text{AUROC-ED} = 0.5$ means uncertainty is no better than random guessing at identifying errors.

---

## 4. Selective Prediction & Clinical Referral Metrics

### 4.1 Risk-Coverage (RC) Curve
Let $S(x)$ be the aggregated case-level uncertainty score. For a target coverage $c \in [0.2, 1.0]$, we determine the uncertainty threshold $\tau_c$ such that a fraction $c$ of the dataset satisfies $S(x) \le \tau_c$.
- **Retained Risk**:
  $$\mathcal{R}(c) = 1 - \frac{1}{| \mathcal{D}_{\text{retained}}(c) |} \sum_{x \in \mathcal{D}_{\text{retained}}(c)} \text{DSC}(Y(x), \hat{Y}(x))$$
- **Plot**: $\mathcal{R}(c)$ on the y-axis versus coverage $c$ on the x-axis.

### 4.2 Area Under Risk-Coverage (AURC) & Excess AURC
- **AURC**:
  $$\text{AURC} = \int_0^1 \mathcal{R}(c) \, dc$$
  Lower AURC indicates superior triage efficiency.
- **Excess AURC (E-AURC)**:
  $$\text{E-AURC} = \text{AURC} - \text{AURC}^*$$
  Where $\text{AURC}^*$ is the theoretical optimum achieved by an oracle that ranks cases by true error.
