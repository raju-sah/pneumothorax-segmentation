# P1 Findings — Honest Recompute (seed 42/7, B=1000 case-level bootstraps)

Script: `scripts/p1_recompute.py` → `results/p1_bootstrap.json`. Case-level only (no pixel probs locally; ESCE/Brier/global-AUROC-ED **not** recomputed — still literals).

## 1. Bootstrap 95% CIs (new, report in paper Table 1)

| Metric | Det | MC | Ens |
|---|---|---|---|
| DSC_all | 0.1368 [0.1250, 0.1492] | 0.1363 [0.1253, 0.1482] | 0.1766 [0.1634, 0.1907] |
| DSC_pos | 0.6071 [0.5859, 0.6283] | 0.6094 [0.5882, 0.6293] | 0.5864 [0.5625, 0.6098] |
| IoU_pos | 0.4726 [0.4519, 0.4932] | 0.4754 [0.4552, 0.4977] | 0.4603 [0.4388, 0.4837] |
| Sens | 0.6906 [0.6642, 0.7164] | 0.6824 [0.6537, 0.7115] | 0.5834 [0.5532, 0.6139] |

Paired tests (per-case DSC_all): ens>det **p=1.4e-05** (Wilcoxon two-sided), Δ=+0.040 [0.030, 0.050] — significant. mc vs det p=0.72, Δ=-0.0004 — n.s.

## 2. AURC: headline does not survive standard definition

| Def | Det | MC | Ens | "Winner" |
|---|---|---|---|---|
| Old (rescaled [0.2,1]→[0,1]) | 0.8686 | 0.8655 | **0.8531** | ens |
| Standard (empirical k/n, no rescale) | 0.8715 | **0.8579** | 0.8596 | mc (n.s.) |

Paired bootstrap CIs on standard-AURC deltas all cross 0: ens−det −0.0121 [−0.0290, +0.0055]; mc−det −0.0086; ens−mc −0.0035. **No method significantly wins AURC.**
E-AURC (vs oracle): det 0.2547, mc 0.2404, ens 0.3269 — ensemble ranks worst relative to its own oracle (its DSC spread is larger).

Recommendation: switch paper+site to standard AURC (0.8715/0.8579/0.8596), report E-AURC, soften claim to "selective ranking comparable across methods; ensemble's gain is in retained DSC_all level (+0.04, significant), not ranking quality." Old rescaled numbers must go.

## 3. Retention values confirmed

Old hardcoded Table 2 retention dice match recompute exactly (100/90/80/70). Only the AURC column + interpretation change.

## 4. Still blocked (need GPU inference rerun with test images)

- ESCE/Brier literals; lesion-conditioned ECE; temp scaling (EXP-07).
- Global pixel-pooled AUROC-ED + empirical ROC (current Fig 3 synthetic).
- MC variance-collapse diagnosis (case_unc = 0.0).
