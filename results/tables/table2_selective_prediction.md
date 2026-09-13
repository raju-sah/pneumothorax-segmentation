### Table 2: Clinical Selective Prediction & Human Referral Simulation (P10, standard AURC)

| Referral Strategy | AURC $\downarrow$ | Retained Dice @ 100% | Retained Dice @ 90% | Retained Dice @ 80% | Retained Dice @ 70% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Referral Baseline** | 0.8888 | 0.1112 | 0.1112 | 0.1112 | 0.1112 |
| **Deterministic Entropy** | 0.7687 | 0.1112 | 0.1142 | 0.1176 | 0.1222 |
| **MC Dropout Variance*** | 0.1525 | 0.7735 | 0.7968 | 0.8122 | 0.8266 |
| **Deep Ensemble Mutual Information** | 0.7100 | **0.3275** | **0.3228** | **0.3061** | **0.2914** |

\*Degenerate (near-empty predictions). Random baselines (1 − mean DSC_all): det 0.8888, ens 0.6725, mc 0.2265 — ens MI ranking (0.7100) is worse than random.
