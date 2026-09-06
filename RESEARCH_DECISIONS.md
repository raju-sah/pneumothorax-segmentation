# RESEARCH DECISIONS: Architecture Decision Records (ADRs)

This document formalizes the critical design trade-offs made during the research-design phase using the Architecture Decision Record (ADR) convention.

---

## ADR-01: Rejection of NIH ChestX-ray14 as External Segmentation Benchmark
- **Context**: Evaluating out-of-distribution (OOD) generalization is standard in medical machine learning. NIH ChestX-ray14 is frequently cited alongside SIIM-ACR.
- **Decision**: Strictly reject using NIH ChestX-ray14 as an external segmentation validation dataset.
- **Rationale**:
  1. SIIM-ACR was sampled directly from NIH ChestX-ray14 and re-annotated by SIIM/ACR/STR radiologists. Using NIH ChestX-ray14 creates severe patient-level overlap, producing artificial leakage.
  2. NIH ChestX-ray14 does not contain pixel-level segmentation masks; its labels were generated via text-mining with estimated 10–30% error rates.
- **Status**: Accepted.

---

## ADR-02: Enforcement of Patient-Grouped Stratified Splitting
- **Context**: In SIIM-ACR, multiple radiographs belong to the same individual (serial monitoring, AP and PA views on the same day).
- **Decision**: Partition the 12,047 Stage 2 images strictly by `PatientID`, ensuring zero patient overlap across training, validation, and test splits.
- **Rationale**: Random image-level splitting allows convolutional backbones to memorize patient-specific anatomical landmarks (rib morphology, cardiothoracic ratio, osteophytes), producing inflated test metrics that collapse on unseen patients.
- **Status**: Accepted.

---

## ADR-03: Selection of ResNet34 U-Net at 512×512 Resolution
- **Context**: Recent medical segmentation literature employs heavy Vision Transformers (TransUNet, Swin-UNet) or massive CNN backbones (EfficientNet-B7).
- **Decision**: Standardize on a ResNet34 U-Net at $512 \times 512$ resolution trained with $0.5 \cdot \text{BCE} + 0.5 \cdot \text{SoftDice}$.
- **Rationale**:
  1. ResNet34 (~21M parameters) has sufficient capacity to delineate fine pleural lines while enabling fast convergence and low memory consumption.
  2. $512 \times 512$ provides the optimal Pareto trade-off between spatial resolution (resolving thin apical pleural margins) and computational efficiency (enabling batch size 16 and 5 ensemble members on Kaggle 16GB GPUs).
  3. Avoids "architecture shopping" and focuses the scientific contribution squarely on uncertainty and selective prediction.
- **Status**: Accepted.

---

## ADR-04: Rejection of Probabilistic U-Net
- **Context**: Probabilistic U-Net (Kohl et al., 2018) is a recognized method for modeling ambiguous segmentations via a conditional VAE latent space.
- **Decision**: Omit Probabilistic U-Net in favor of Deep Ensembles ($M=5$) and MC Dropout ($T=20$).
- **Rationale**:
  1. Under severe class imbalance (~78% empty masks) and thin boundary lines, the cVAE latent space frequently suffers from posterior collapse or KL-vanishing.
  2. SIIM-ACR provides a single consensus mask per image rather than multi-rater annotations, making multi-modal latent space training ill-posed.
  3. Deep Ensembles empirically outperform variational approximations across calibration and OOD detection benchmarks without requiring fragile $\beta$-annealing schedules.
- **Status**: Accepted.

---

## ADR-05: Top-K Pixel Variance Aggregation for Selective Prediction
- **Context**: Selective prediction requires an image-level scalar uncertainty score to triage entire radiographs to human radiologists.
- **Decision**: Implement top-$K$ pixel variance ($K=500$) rather than whole-image mean uncertainty.
- **Rationale**: Pneumothorax lesions are localized, occupying $<5\%$ of total pixels in typical positive cases. Taking the mean across the entire image dilutes focal diagnostic uncertainty with hundreds of thousands of certain background pixels. Top-$K$ captures concentrated diagnostic hesitation.
- **Status**: Accepted.

---

## ADR-06: Offline Preprocessing to Compressed HDF5
- **Context**: Kaggle Notebooks impose a 30-hour weekly GPU quota and disk I/O bottlenecks when parsing 12,000+ individual DICOM files per epoch.
- **Decision**: Preprocess all images once offline (inverting `MONOCHROME1`, resizing to $512 \times 512$, normalizing to uint8) and store in a single compressed HDF5 archive.
- **Rationale**: Cuts per-epoch training time from ~12 minutes down to ~1.2 minutes, ensuring 5 ensemble models and 5 seeds can be trained within ~13 total GPU hours.
- **Status**: Accepted.
