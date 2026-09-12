# P2 post-mortem (2026-09-12) — SUPERSEDED BY P3, do not cite numbers

## MC root cause (real, affects paper EXP-04)
`smp.Unet` decoder blocks are custom `DecoderBlock` modules with explicit
`forward()` — `add_module("spatial_dropout", ...)` registers the Dropout2d
but `forward()` never calls it. Dropout was a **silent no-op in training AND
inference**. Original `predict_mc_dropout` also used `self.train()`, so BN ran
on batch stats (explains mc_dice 0.6094 vs det 0.6071). True MC variance = 0 →
`mc_case_unc = 0.0`, AUROC-ED exactly 0.5000. EXP-04 as reported is invalid;
P3 retrains MC with dropout applied via forward hooks (version-proof).

## P2's own MC = artifact of P2 (not evidence)
P3-builder's `PUNet` omitted dropout layers entirely → P2 `mc_var_mean = 0`
says nothing. Disregard P2 MC columns.

## P2 det/ens pixel numbers = preprocessing-mismatched, parked
P2 `load_case` skipped MONOCHROME1 inversion and used [0,1]+single-mean norm
instead of the training pipeline (uint8 + Albumentations 3ch Normalize), so
P2 dice (M1 DSC_all 0.2984) don't match the reported run (0.1368). Parked:
ESCE det 0.000832 / ens 0.000491, Brier 7.7e-05, global AUROC det 0.9808 /
ens 0.9720, temp T=0.75. P3 replicates the exact pipeline; use P3 numbers.

## Usable P2 outputs
- Empirical ROC methodology + fig3 pipeline (rerun in P3).
- Kernel ops lessons: kernelspec required; never upgrade torch (sm_70+ only);
  cu118 torch==2.3.1 for P100; smp==0.3.3 --no-deps + pretrainedmodels deps.
