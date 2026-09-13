# Scientific Paper Submission Package & Checklist

This directory contains the finalized manuscript sources, figures, compiled PDF, and packaged arXiv bundle for:

> **Title:** Uncertainty-Aware Pneumothorax Segmentation and Selective Prediction in Chest Radiographs: Bridging Dense Uncertainty and Clinical Referral  
> **Author:** Raju Sah (`raju.sah.research@gmail.com`)  
> **Repository & Interactive Dashboard:** [https://github.com/raju-sah/pneumothorax-segmentation](https://github.com/raju-sah/pneumothorax-segmentation)

---

## Track 1: arXiv / medRxiv Submission (Immediate)

### 1. Upload Package
- **Pre-packaged Tarball:** `paper/arxiv_submission.tar.gz` (contains `main.tex`, `references.bib`, and all 6 publication-ready figures).
- **Direct Link:** Go to [arXiv Submit](https://arxiv.org/submit) or [medRxiv Submit](https://www.medrxiv.org/).

### 2. Submission Metadata Form

- **Title:**
  ```text
  Uncertainty-Aware Pneumothorax Segmentation and Selective Prediction in Chest Radiographs: Bridging Dense Uncertainty and Clinical Referral
  ```

- **Authors:**
  ```text
  Raju Sah
  ```

- **Primary Subject Classification:**
  - `eess.IV` (Electrical Engineering and Systems Science — Image and Video Processing)

- **Cross-List Categories (Recommended):**
  - `cs.CV` (Computer Science — Computer Vision and Pattern Recognition)
  - `cs.LG` (Computer Science — Machine Learning)
  - `q-bio.QM` (Quantitative Biology — Quantitative Methods)

- **Comments Field:**
  ```text
  12 pages, 5 figures, 4 tables. Verified zero patient-leakage benchmark on 2,135 holdout radiographs. Interactive dashboard and code: https://github.com/raju-sah/pneumothorax-segmentation
  ```

- **Abstract (Copy & Paste Ready):**
  ```text
  Automated segmentation of pneumothorax from chest radiographs offers immense clinical utility for emergency triage, yet uncalibrated deep neural networks exhibit severe vulnerability to thoracic confounders (e.g., skin folds, rib shadows) and catastrophic silent false negatives on subtle apical lesions. In high-stakes medical deployment, deep learning models must not only segment lesions accurately but also quantify their own predictive uncertainty to support selective prediction—autonomously reporting confident cases while routing ambiguous cases to expert radiologists. In this study, we establish an audited, patient-isolated benchmark on the SIIM-ACR Pneumothorax dataset (N=2,135 untouched holdout test patients with verified zero-leakage). We implement and rigorously compare a deterministic ResNet34 U-Net baseline against Monte Carlo Dropout (T=20) and Deep Ensembles (M=3). Our empirical findings reveal that Deep Ensembles retain substantially higher cohort segmentation quality (DSC_all = 0.3275 vs 0.1112, Delta = +0.216 [0.199, 0.235], Wilcoxon p = 2.3e-96) with directly computed spatial calibration errors (ESCE approx 2.5e-4). Selective-triage ranking helps the deterministic baseline (standard AURC = 0.7687 vs 0.8888 random) but uncertainty ranking via ensemble mutual information is worse than random (AURC = 0.7100 vs 0.6725): the most confident ensemble cases are worse than average. Pixel-pooled error discrimination is near-perfect on all branches (AUROC-ED approx 0.99), exposing a split between excellent ranking and a miscalibrated fixed 0.5 operating point. Calibrating the operating threshold to t* = 0.25 under validation temperature scaling (EXP-07) surges deterministic lesion Dice from 0.2803 to 0.4214 (+50.3% relative gain) and nearly triples pixel sensitivity (11.4% to 32.9%) without retraining. A same-seed rerun under a different checkpoint-selection rule swings deterministic DSC_pos from 0.61 to 0.28, so checkpoint-selection variance is reported as a first-order limitation. Subgroup stratification demonstrates that bedside portable anteroposterior (AP) radiographs exhibit systematically higher predictive uncertainty (0.3596 vs 0.3351) and lower dice similarity than upright posteroanterior (PA) examinations, highlighting clinical projection bias. By coupling dense epistemic uncertainty with a calibrated case-level referral filter, our framework safely automates high-confidence diagnoses while escalating ambiguous and occult cases, presenting a clinically viable blueprint for human-AI collaborative thoracic radiology.
  ```

---

## Track 2: Peer-Reviewed Venue (MICCAI UNSURE / MIDL)

### Target Venues
1. **UNSURE @ MICCAI (Primary Recommendation):**
   - *Scope:* Uncertainty for Safe Utilization of Machine Learning in Medical Imaging.
   - *Format:* Springer LNCS format (typically 8–10 pages + references).
   - *Why it fits:* UNSURE explicitly solicits critical diagnostic papers that expose where UQ methods (like MC Dropout or mutual information) fail in clinical practice.
2. **MIDL (Medical Imaging with Deep Learning):**
   - *Scope:* Core medical deep learning algorithms and evaluations.
   - *Format:* JMLR OpenReview format (8 pages full paper, or 3-page short paper).

### Strategic Framing for Reviewers
- **Do NOT claim SOTA overlap:** Frame the paper as an *audited diagnostic and reproducibility benchmark*.
- **Emphasize the 3 Core Diagnostic Insights:**
  1. *MC Dropout magnitude collapse:* Saturated sigmoid decoders crush predictive variance ($\sim 3 \times 10^{-6}$), rendering test-time dropout ineffective for dense segmentation triage.
  2. *The Checkpoint Selection Illusion:* The metric used for validation checkpoint selection (`val_all` vs `val_pos`) produces a $0.33$ Dice swing on holdout data.
  3. *Operating-Point Miscalibration:* Pixel error ranking is near-perfect ($\text{AUROC-ED} \approx 0.99$), yet threshold-$0.50$ models fail; post-hoc threshold tuning ($t^*=0.25$) fixes this scale distortion without retraining.
