### Table 1: Primary Experimental Benchmark on Untouched Holdout Test Set ($N=2,135$) — P10 verbatim run

| Evaluation Metric | Deterministic Baseline (seed 42) | MC Dropout ($T=20$) (seed 42) | Deep Ensemble ($M=3$) |
| :--- | :---: | :---: | :---: |
| **$DSC_{pos}$** (Positive Patients Only) | 0.2803 [0.264, 0.297] | 0.0176* [0.010, 0.027] | **0.3158** [0.292, 0.339] |
| **$DSC_{all}$** (Complete Cohort with Negatives) | 0.1112 [0.101, 0.122] | 0.7735* [0.756, 0.791] | **0.3275** [0.309, 0.346] |
| **AUROC-ED** (Error Detection, global pixels) $\uparrow$ | **0.9936** | 0.9884 | 0.9929 |
| **ESCE** (Calibration Error) $\downarrow$ | 0.00024 | 0.00015* | **0.00025** |
| **Brier Score** $\downarrow$ | 0.00009 | 0.00008* | 0.00010 |
| **AURC** (Risk-Coverage, standard) $\downarrow$ | 0.7687 [0.743, 0.797] | 0.1525* | **0.7100** [0.686, 0.736] |
| **E-AURC** (excess vs oracle) $\downarrow$ | **0.1031** | 0.1247* | 0.3981 |

\*MC degenerate (near-empty masks); do not read as wins. ens−det DSC_all Δ=+0.2163 [0.1989, 0.2352], Wilcoxon p=2.3e-96. IoU/Sens/Spec/E-AURC from superseded run omitted.

### Table 1b: Post-Hoc Calibrated Benchmark (EXP-07 Temperature Scaling + Threshold Tuning, $t^*=0.25$)

| Evaluation Metric | Deterministic Tuned ($T=0.86, t^*=0.25$) | Deep Ensemble Tuned ($T=0.84, t^*=0.25$) | $\Delta$ (Tuned vs Ref Det) |
| :--- | :---: | :---: | :---: |
| **$DSC_{pos}$** (Positive Patients Only) | **0.4214** [0.404, 0.439] | 0.4113 [0.392, 0.430] | **+0.1411** (+50.3%) |
| **$DSC_{all}$** (Complete Cohort with Negatives) | 0.1408 [0.129, 0.152] | 0.1348 [0.123, 0.145] | **+0.0296** (+26.6%) |
| **Pixel Sensitivity** | 0.3287 | **0.3539** | **+0.2147** (+188%) |
| **Pixel IoU** | 0.0805 | **0.0868** | **+0.0254** (+46.1%) |
| **ESCE** (Calibration Error) $\downarrow$ | 0.00013 | 0.00013 | -0.00011 |
| **Brier Score** $\downarrow$ | 0.00009 | 0.00009 | 0.00000 |
| **AURC** (Risk-Coverage, standard) $\downarrow$ | **0.7507** | 0.8273 | **-0.0180** |
| **E-AURC** (excess vs oracle) $\downarrow$ | **0.1395** | 0.2060 | +0.0364 |

*Tuned threshold $t^*=0.25$ selected via validation $DSC_{pos}$ grid sweep; temperature $T$ fit via validation pixel BCE with LBFGS. Wilcoxon signed-rank $p = 0.0110$.*
