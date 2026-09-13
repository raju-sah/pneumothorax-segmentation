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
