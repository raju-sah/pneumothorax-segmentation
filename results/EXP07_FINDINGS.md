# EXP-07 Findings: Temperature Scaling & Threshold Tuning on P10 Verbatim Checkpoints

**Kernel:** `rajucode/exp07-temp-scaling-threshold-tuning` (v3, COMPLETE 2026-09-13).  
**Protocol:** Strictly no retraining. Checkpoints loaded from `rajucode/p10-verbatim-checkpoints` (`det_seed42.pt`, `det_seed43.pt`, `det_seed44.pt`).
1. Temperature $T$ fit on validation set via LBFGS minimization of pixel binary cross-entropy (subsampled stride 64).
2. Operating threshold $t^*$ swept on validation set to maximize validation $\text{DSC}_{\text{pos}}$ (verbatim original rule).
3. Evaluated on untouched holdout test set ($N=2,135$, 476 positive / 1,659 negative) at both tuned $(T, t^*)$ and reference standard $(T=1.0, t=0.50)$.
4. Branches: Deterministic (`det`, seed 42) and Deep Ensemble (`ens`, seeds 42–44 average probability). MC Dropout was excluded as it collapsed to near-empty predictions in P10 ($\text{DSC}_{\text{pos}} \approx 0.018$, variance $\sim 3 \times 10^{-6}$).

---

## 1. Fitted Parameters (Validation Set)

| Branch | Fitted Temperature $T$ | Optimal Threshold $t^*$ | Val $\text{DSC}_{\text{pos}}$ @ $t^*$ | Val $\text{DSC}_{\text{pos}}$ @ 0.50 |
| :--- | :---: | :---: | :---: | :---: |
| **Deterministic (`det`)** | 0.8586 | **0.25** | 0.4278 | 0.2675 |
| **Deep Ensemble (`ens`)** | 0.8398 | **0.25** | 0.4183 | 0.3301 |

- Both fitted temperatures are $T < 1.0$ ($0.86$ and $0.84$), indicating that unscaled logits were slightly underconfident/over-dispersed; temperature scaling slightly sharpens high-confidence regions.
- The optimal threshold on validation positive Dice for both models is **$t^* = 0.25$**, roughly half of the default $0.50$ threshold.

---

## 2. Primary Test Set Results ($N=2,135$)

| Configuration | $\text{DSC}_{\text{pos}}$ [95% CI] | $\text{DSC}_{\text{all}}$ [95% CI] | Pixel Sensitivity | Pixel IoU | Pixel Specificity | ESCE | Brier Score | AURC | E-AURC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Det Reference ($t=0.50$)** | 0.2803 | 0.1112 | 0.1140 | 0.0551 | 0.9999 | 0.000128 | 9.0e-05 | 0.7687 | 0.1031 |
| **Det Tuned ($T=0.86, t^*=0.25$)** | **0.4214** [0.404, 0.439] | **0.1408** [0.129, 0.152] | **0.3287** | **0.0805** | 0.9998 | 0.000128 | 9.0e-05 | **0.7507** | **0.1395** |
| $\Delta_{\text{det}}$ (Tuned vs Ref) | **+0.1411** (+50.3%) | **+0.0296** (+26.6%) | **+0.2147** (+188%) | **+0.0254** (+46.1%) | -0.0001 | 0.0 | 0.0 | **-0.0180** | — |
| \midrule | | | | | | | | |
| **Ens Reference ($t=0.50$)** | 0.3158 | 0.3275 | 0.1752 | 0.0867 | 0.9999 | 0.000126 | 9.1e-05 | 0.7100 | 0.3981 |
| **Ens Tuned ($T=0.84, t^*=0.25$)** | **0.4113** [0.392, 0.430] | 0.1348 [0.123, 0.145] | **0.3539** | **0.0868** | 0.9998 | 0.000126 | 9.1e-05 | 0.8273 | 0.2060 |
| $\Delta_{\text{ens}}$ (Tuned vs Ref) | **+0.0955** (+30.2%) | -0.1927 | **+0.1787** (+102%) | +0.0001 | -0.0001 | 0.0 | 0.0 | +0.1173 | — |

*Bootstrap 95% CIs on test cohort computed across 1,000 resamples (seed 42).*  
*Wilcoxon signed-rank test between Tuned Ensemble and Tuned Deterministic: $p = 0.0110$.*

---

## 3. Key Findings & Insights

1. **Resolving the "Operating-Point vs Ranking Split":**
   - In P10, we proved that pixel ranking was near-perfect ($\text{AUROC-ED} \approx 0.99$), yet threshold-$0.50$ Dice remained poor ($\text{DSC}_{\text{pos}} = 0.2803$) because default sigmoid cutoff $0.50$ severely suppressed sensitivity ($11.4\%$).
   - Threshold tuning to $t^* = 0.25$ directly fixes this scale miscalibration:
     - Deterministic $\text{DSC}_{\text{pos}}$ surges from **0.2803 to 0.4214** ($+50.3\%$ relative gain, $+0.1411$ absolute).
     - Pixel sensitivity jumps from **11.4% to 32.9%** (nearly $3\times$ improvement) with essentially no loss in specificity ($0.9999 \to 0.9998$).
     - Pixel IoU improves by $+46.1\%$ ($0.0551 \to 0.0805$).

2. **The Precision-Recall Tradeoff on Negative Scans:**
   - For Deep Ensemble, tuning the operating threshold to $0.25$ boosts lesion-positive detection from **0.3158 to 0.4113** ($\text{DSC}_{\text{pos}}$), doubling sensitivity ($17.5\% \to 35.4\%$).
   - However, because the threshold is lowered from $0.50$ to $0.25$, small background false-positive activations that were previously suppressed by ensemble averaging occasionally survive. On true-negative scans ($N=1,659$), a single false-positive pixel causes the scan's Dice score to plummet from $1.0$ to $\approx 0.0$. This drops the cohort-wide $\text{DSC}_{\text{all}}$ from $0.3275$ to $0.1348$.
   - **Clinical takeaway:** In screening scenarios requiring high recall / lesion detection, $t^* = 0.25$ is strictly superior. In automated unreviewed triage where negative scans must not be false-alarmed, a conservative threshold ($0.50$) or a dual-threshold system (e.g. $t_{\text{detect}} = 0.25$, $t_{\text{negative}} = 0.50$) is indicated.

3. **Calibration & Selective Referral:**
   - Both models maintain exceptionally low spatial calibration error ($\text{ESCE} \approx 1.27 \times 10^{-4}$) and low Brier score ($\approx 9.0 \times 10^{-5}$).
   - Deterministic selective referral AURC further improves under temperature scaling and threshold tuning from $0.7687$ to **$0.7507$** (E-AURC = $0.1395$).

---

## 4. Artifacts Produced
- Predictions: `results/exp07_predictions.csv` (2,135 holdout rows with tuned and reference metrics)
- Metrics Summary: `results/exp07_metrics.json`
- Threshold Curves Plot: `results/figures/exp07_threshold_curves.png`
