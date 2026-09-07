### Table 3: Anatomical Subgroup Stratification Across Projection Views (EXP-09)

| Projection View | Cases ($N$) | Model | $DSC_{pos}$ | $DSC_{all}$ | AUROC-ED | Case Uncertainty |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **AP (Bedside / Portable)** | 837 | Deterministic Baseline | 0.6049 | 0.1279 | 0.9888 | 0.3444 |
| | | MC Dropout ($T=20$) | **0.6158** | 0.1266 | 0.5000 | 0.0000 |
| | | Deep Ensemble ($M=3$) | 0.5629 | **0.1551** | 0.9645 | 0.0349 |
| **PA (Upright / Standing)** | 1,298 | Deterministic Baseline | 0.6083 | 0.1425 | 0.9909 | 0.3380 |
| | | MC Dropout ($T=20$) | 0.6057 | 0.1426 | 0.5000 | 0.0000 |
| | | Deep Ensemble ($M=3$) | 0.5997 | **0.1905** | 0.9599 | 0.0360 |
