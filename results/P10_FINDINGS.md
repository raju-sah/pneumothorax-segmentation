# P10 Findings — verbatim-rule final run (best = val DSC_pos)

Kernel: `rajucode/p10-final-retrain-v2` (COMPLETE 2026-09-12). Protocol identical to P9
except checkpoint selection: argmax val DSC_pos (verbatim original rule) instead of
argmax val DSC_all. Artifacts: `results/p10_predictions.csv` (N=2135),
`results/p10_final_metrics.json`, `results/p10_*_trace.json`.

## Primary numbers (test, N=2135, 476 pos / 1659 neg)

| branch | DSC_all [95% CI] | DSC_pos [95% CI] | AURC_std | AUROC-ED (global) | ESCE | Brier |
|---|---|---|---|---|---|---|
| det (seed 42) | 0.1112 [0.1006, 0.1215] | 0.2803 [0.2638, 0.2969] | 0.7687 | 0.9936 | 0.000244 | 9.3e-05 |
| mc (T=20, seed 42) | 0.7735* [0.7563, 0.7910] | 0.0176 [0.0101, 0.0269] | 0.1525* | 0.9884 | 0.000150 | 8.3e-05 |
| ens (seeds 42-44) | 0.3275 [0.3086, 0.3459] | 0.3158 [0.2915, 0.3394] | 0.7100 | 0.9929 | 0.000254 | 9.5e-05 |

\*mc is degenerate: predicts near-empty everywhere (0.7735 ≈ 0.7770 empty baseline).
Its AURC "win" is meaningless — do not claim it.

- ens vs det DSC_all: Δ = +0.2163 [0.1989, 0.2352], Wilcoxon two-sided p = 2.3e-96.
- ESCE/Brier now computed from the run (not the old hardcoded literals).

## Headline result: checkpoint-selection variance dominates

Same seed 42, same code — only the selection rule changed:

| det seed 42 | P9 (best val_all) | P10 (best val_pos) |
|---|---|---|
| checkpoint picked | ep1 (val_pos 0.4678) | ep10 (val_pos 0.2675) |
| test DSC_pos | 0.6071 | 0.2803 |
| test DSC_all | 0.1368 | 0.1112 |

Test DSC_pos swings 0.61 → 0.28 from the rule change alone. Training at lr=1e-3 is
visibly unstable (val DSC_all oscillates 0.0–0.78 across epochs in every trace).
Any single-checkpoint claim is therefore fragile — reported as a limitation, with
multi-seed/multi-checkpoint averaging as required future work.

## Selective-prediction reversal vs P9

Random baselines (1 − mean DSC_all): det 0.8888, ens 0.6725, mc 0.2265.

- det entropy helps: 0.7687 < 0.8888. Retained DSC_all 0.1112 → 0.1222 @70%.
- ens mutual information HURTS: 0.7100 > 0.6725. Retained DSC_all 0.3275 → 0.2914
  @70%. The most-confident ens cases are worse than average (Fig 1: curve spikes
  to ~0.78 risk at 20% coverage) — three unstable seeds agree confidently on
  wrong (mostly empty) masks, so MI ranks backwards at the confident end.
- det is still the only branch where referral monotonically helps.

## MC dropout: functional but magnitude-collapsed

Hooks fire (all 2135 mc_var_mean > 0) but variance magnitude is ~3.2e-06 mean
(max 1.3e-03, p=0.2 dropout2d on decoder). Saturated sigmoid outputs crush the
stochastic spread, so T=20 passes are near-identical and the mean predicts
empty (DSC_pos 0.0176). AUROC-ED global 0.9884 looks good only because
near-zero variance still orders the few non-empty pixels correctly — the branch
is unusable as an uncertainty signal at this scale.

## Operating-point vs ranking split

Pixel-pooled AUROC-ED ≈ 0.99 on all branches while threshold-0.5 DSC is poor
(det 0.11): ranking of pixels is near-perfect, the fixed 0.5 operating point is
miscalibrated in scale (det fires FP blobs on ~94% of negatives: only 6.3% of
negatives score DSC=1.0). Threshold tuning / temperature scaling (EXP-07, never
run) is the obvious next step, now P2.

## Subgroups (P10, with bootstrap CIs)

| view | det all / pos | mc all / pos | ens all / pos | mean U_case det / ens |
|---|---|---|---|---|
| AP (837, 172 pos) | 0.0943 / 0.2438 | 0.7939* / 0.0202 | 0.3259 / 0.2838 | 0.3596 / 0.0143 |
| PA (1298, 304 pos) | 0.1221 / 0.3009 | 0.7603* / 0.0162 | 0.3286 / 0.3339 | 0.3351 / 0.0117 |

AP still harder than PA (higher det entropy, lower DSC). Per-subgroup AUROC-ED
not available (pixel pools only saved globally) — dropped from Table 3.
