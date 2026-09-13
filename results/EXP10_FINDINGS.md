# EXP-10 Findings: Robustness, Lesion Size Stratification & Negative-Control Audit

**Protocol:** Evaluates model robustness across:
1. **Lesion Size Stratification:** Distribution of pneumothorax sizes in the test holdout ($N=476$ positive scans), assessing overlap recovery across subtle ($<2\%$) and medium ($2-10\%$) pathologies.
2. **Negative-Control Specificity Audit:** False-positive activation behavior across $1,659$ verified negative control radiographs.
3. **Pathology vs. Negative-Control Discrimination:** AUROC of case-level uncertainty in separating active pneumothorax from healthy control radiographs.

---

## 1. Lesion Size Stratification (Positive Cohort, $N=476$)

| Strata (Pneumothorax Extent) | Cases ($N$) | % of Positives | Mean Lesion Area | Det Ref ($t=0.50$) | Det Tuned ($t^*=0.25$) | Det Gain | Ens Ref ($t=0.50$) | Ens Tuned ($t^*=0.25$) | Ens Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small Lesions ($<2\%$ hemithorax)** | 380 | **79.8%** | 0.54% of pixels | 0.3211 | **0.4513** | **+40.6%** | 0.3515 | **0.4297** | **+22.3%** |
| **Medium Lesions ($2-10\%$)** | 96 | **20.2%** | 4.18% of pixels | 0.1189 | **0.3030** | **+154.9%** | 0.1744 | **0.3387** | **+94.2%** |
| **Large Lesions ($>10\%$)** | 0 | 0.0% | — | — | — | — | — | — | — |

### Key Observations:
- **Severe Anatomical Subtlety:** Roughly 80% of pneumothoraces occupy $<2\%$ of the image area (mean $0.54\% \approx 5,600$ pixels out of $1,048,576$). Clinically, this reflects emergency screening where patients present with subtle apical caps or small pneumothoraces rather than total tension collapse.
- **Operating-Point Leverage Across Strata:** Tuning the threshold to $t^* = 0.25$ provides dramatic overlap improvements across both groups:
  - Small subtle lesions: $+40.6\%$ relative gain for deterministic ($0.321 \to 0.451$).
  - Medium lesions: $+154.9\%$ relative gain for deterministic ($0.119 \to 0.303$) and $+94.2\%$ for ensemble ($0.174 \to 0.339$), overcoming the severe under-segmentation of default $0.50$ cutoffs.

---

## 2. Negative-Control Specificity & False Alarm Audit ($N=1,659$)

| Metric on True Negative Radiographs | Deterministic Baseline | Deep Ensemble ($M=3$) | Clinical Implication |
| :--- | :---: | :---: | :--- |
| **Clean Negative Rate @ $t=0.50$** (Dice = 1.0, 0 FP pixels) | 6.3% | **33.1%** ($5.3\times$ higher) | Ensemble multi-seed probability averaging effectively quashes background artifacts |
| **Clean Negative Rate @ $t^*=0.25$** (Dice = 1.0, 0 FP pixels) | 6.0% | 5.5% | Lower threshold allows minor boundary activations to trigger false alarms |
| **Mean Case Uncertainty ($U_{\text{case}}$)** | 0.3419 | 0.0124 | Epistemic mutual information remains near zero on negative controls |

### Clinical Translation:
- Under conservative standard thresholding ($t=0.50$), the Deep Ensemble acts as an automated negative-filter, correctly predicting completely empty masks on **$33.1\%$** of negative radiographs without radiologist input.
- Lowering the operating point to $t^*=0.25$ is optimal for positive lesion detection, but increases false-positive pixel triggers on negative controls.
- **Recommendation:** Clinical systems should deploy a **two-stage dual-threshold cascade**:
  1. *Screening / Triage Stage ($t=0.50$):* Ensemble screens out confident true negatives ($DSC = 1.0$).
  2. *Lesion Delineation Stage ($t^*=0.25$):* High-sensitivity operating point delineates subtle pneumothorax boundaries on flagged positive cases.

---

## 3. Negative-Control Discrimination (Pathology vs. Controls)

Can case uncertainty distinguish active pneumothorax radiographs from negative controls?

- **MC Dropout Variance:** $\text{AUROC} = \mathbf{0.6533}$
- **Deterministic Entropy:** $\text{AUROC} = 0.5872$
- **Deep Ensemble Mutual Information:** $\text{AUROC} = 0.5563$

Case uncertainty is higher on positive scans (mean det unc $0.3544$ vs $0.3419$ negative; ens unc $0.0139$ vs $0.0124$). However, because top-$K$ pixel aggregation concentrates on the 500 most uncertain pixels, anatomical confounders on negative controls (scapular borders, ribs, skin folds) also produce elevated localized hesitation, capping image-level classification AUROC at $0.56 - 0.65$.

---

## 4. Artifacts Produced
- Analysis script: `scripts/run_exp10_analysis.py`
- Metrics data: `results/exp10_metrics.json`
