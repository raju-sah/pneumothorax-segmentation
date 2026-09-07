### Table 1: Primary Experimental Benchmark on Untouched Holdout Test Set ($N=2,135$)

| Evaluation Metric | Deterministic Baseline (EXP-03) | MC Dropout ($T=20$) (EXP-04) | Deep Ensemble ($M=3$) (EXP-05) |
| :--- | :---: | :---: | :---: |
| **$DSC_{pos}$** (Positive Patients Only) | 0.6071 $\pm$ 0.2244 | **0.6094 $\pm$ 0.2243** | 0.5864 $\pm$ 0.2578 |
| **$DSC_{all}$** (Complete Cohort with Negatives) | 0.1368 | 0.1363 | **0.1766** |
| **$IoU_{pos}$** (Jaccard Index) | 0.4726 | **0.4754** | 0.4603 |
| **Sensitivity** (Lesion Recall) | **0.6906** | 0.6824 | 0.5834 |
| **Specificity** (Healthy Sparing) | 0.9999 | 0.9999 | **0.9999** |
| **AUROC-ED** (Error Detection) $\uparrow$ | **0.9900** | 0.5000 | 0.9617 |
| **ESCE** (Calibration Error) $\downarrow$ | 0.0012 | 0.0012 | **0.0006** (50% reduction) |
| **Brier Score** $\downarrow$ | 0.0001 | 0.0001 | 0.0001 |
| **AURC** (Risk-Coverage) $\downarrow$ | 0.8686 | 0.8655 | **0.8531** (Best Pareto frontier) |
