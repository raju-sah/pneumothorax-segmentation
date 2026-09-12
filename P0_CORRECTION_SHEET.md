# P0 Correction Sheet — Source of Truth vs Paper/Site (MICCAI/MIDL path)

**Source of truth:** `results/test_predictions.csv` (2135 rows) + `data/processed/patient_splits.csv` (10675 rows) + `results/metrics_summary.json` (= CSV means, verified by recount 2026-09-12).
**Method:** recomputed column means from CSV; diffed against `paper/main.tex`, `docs/index.html`, `README.md`, `results/tables/`, `results/stdout_full.txt`.

## 1. Verified ground truth (recount)

| Item | Truth | Evidence |
|---|---|---|
| Test rows / patients | 2135 / 2135 (1 img/pt in this export) | `test_predictions.csv`, `patient_splits.csv` Fold=test |
| Pos / Neg | **476 / 1659 (22.3%)** | both CSVs agree |
| AP / PA | **837 / 1298** | `Counter(ViewPosition)` |
| Patient overlap across folds | **0** | pairwise PatientID disjoint |
| Total rows / unique patients | 10675 / 10675 | no multi-image patients in export |
| DSC_pos det/mc/ens | 0.6071 / 0.6094 / 0.5864 | CSV mean = JSON |
| DSC_all det/mc/ens | 0.1368 / 0.1363 / 0.1766 | CSV mean = JSON |
| IoU_pos det/mc/ens | **0.4726 / 0.4754 / 0.4603** | pos-only mean; CSV all-case mean differs (0.1068) — label tables IoU_pos |
| Sens det/mc/ens | **0.6906 / 0.6824 / 0.5834** | pos-only |
| Spec det/mc/ens | 0.99986 / 0.99988 / 0.99993 | all-case |
| AUROC-ED det/mc/ens | 0.9900 / 0.5000 / 0.9617 | mean-of-per-case (see §3 caveat) |
| AURC det/mc/ens/random | 0.8686 / 0.8655 / 0.8531 / **0.8648** | JSON + stdout:117 + table2 |
| AP pos 172 (det 0.6049 / mc 0.6158 / ens 0.5629), PA pos 304 (det 0.6083 / mc 0.6057 / ens 0.5997) | verified from CSV | matches JSON subgroups |

## 2. Corrections required (paper + site)

| # | Location | Wrong | Correct | Action |
|---|---|---|---|---|
| C1 | `paper/main.tex:62,99-101` + `docs/index.html:448` | 319 pos / 1816 neg (14.9%) | **476 / 1659 (22.3%)** | Replace all 3 spots + abstract-adjacent text |
| C2 | `paper/main.tex:187-189` + `docs/index.html:459-475` | IoU 0.4700/0.4728/0.4501, Sens 0.6974/0.6917/0.6279, Spec 0.9996/0.9996/0.9997 | IoU **0.4726/0.4754/0.4603**, Sens **0.6906/0.6824/0.5834**, Spec **0.99986/0.99988/0.99993** | Regenerate from JSON, 4-decimals |
| C3 | `paper/main.tex:218,226` + `docs/index.html:385,528` | Random AURC 0.8697 | **0.8648** (JSON:46, stdout:117, table2, README:122) | Replace 4 spots |
| C4 | `paper/main.tex:199` | "On positive patients (N=319)" | "(N=476)" | Same edit as C1 |
| C5 | Config/M reporting | M=5 (ensemble.yaml, MODEL_CARD, REPRODUCIBILITY) vs M=3 actual | **Pin M=3 (seeds 42,43,44)** everywhere until rerun | Edit `configs/ensemble.yaml:2-3`, MODEL_CARD, REPRODUCIBILITY, RESEARCH_PLAN |
| C6 | Epochs/LR | yaml 35 epochs lr 3e-4 vs run 10 epochs vs paper 1e-4+cosine | **State actual: 10 epochs (this run)** or rerun at 35 | Align yaml ↔ paper ↔ stdout |

Correct values already present (no change): `results/tables/table2_*`, `README.md:122`, `results/stdout_full.txt`, hero stats (AURC 0.8531, ESCE, DSC_all, AUROC-ED).

## 3. Methodology caveats (P1, not typo fixes)

- AUROC-ED 0.99/0.50/0.96 is **mean-of-per-case** with single-class→0.5 fallback; MC `case_unc`=0.0 (variance collapse). Needs global pixel-pooled recompute + seeded subsample + empirical ROC (current Fig3 curve synthetic).
- AURC uses non-standard [0.2,1]→[0,1] rescale; retention table hardcoded. Needs standard def + E-AURC.
- ESCE/Brier are literals in `compile_final_results.py:109,122,135`, not computed. Needs real compute + lesion-conditioned ECE.
- No CIs anywhere. Needs bootstrap/DeLong in P1.
- Reversed findings must be owned: Ens DSC_pos < Det (0.5864<0.6071, H1 p=0.9974); retention @80% < @100%. Reframe around DSC_all/calibration/selective.

## 4. Leakage note

Splits verified zero-overlap at PatientID level. Caveat: export has 1 image/patient, so grouping trivially holds; claim "Study Instance UID audit" in paper:93 overstates what code checks (SOP Instance UID join only, no dedup pass). Recommend softening to PatientID-grouped or adding dedup script.
