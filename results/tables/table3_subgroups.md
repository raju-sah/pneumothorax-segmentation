### Table 3: Anatomical Subgroup Stratification Across Projection Views (P10)

| Projection View | Cases ($N$) | Model | $DSC_{pos}$ | $DSC_{all}$ | Case Uncertainty |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **AP (Bedside / Portable)** | 837 | Deterministic Baseline | 0.2438 | 0.0943 | 0.3596 |
| | | MC Dropout ($T=20$)* | 0.0202 | 0.7939 | 0.0009 |
| | | Deep Ensemble ($M=3$) | 0.2838 | **0.3259** | 0.0143 |
| **PA (Upright / Standing)** | 1,298 | Deterministic Baseline | 0.3009 | 0.1221 | 0.3351 |
| | | MC Dropout ($T=20$)* | 0.0162 | 0.7603 | 0.0008 |
| | | Deep Ensemble ($M=3$) | 0.3339 | **0.3286** | 0.0117 |

\*Degenerate near-empty predictions. Per-subgroup AUROC-ED unavailable (pixel pools saved globally only); global values in Table 1.
