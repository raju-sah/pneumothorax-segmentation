# RESEARCH LIMITATIONS & CLINICAL BOUNDARIES

To maintain the highest level of academic integrity, this document outlines the inherent methodological, data-driven, and clinical boundaries of this project.

---

## 1. Radiographic & Anatomical Limitations (2D Projection vs. 3D Pathology)

1. **Planar Projection Superimposition**:
   Standard chest radiography collapses a three-dimensional thoracic volume into a single two-dimensional planar projection. Pneumothorax pockets situated purely anteriorly or posteriorly in supine patients (deep sulcus sign) or masked behind dense retrocardiac and subdiaphragmatic zones may be invisible on 2D radiographs.
2. **Computed Tomography (CT) Disconnect**:
   Non-contrast thoracic CT is the clinical gold standard for quantifying pneumothorax volume. Any 2D radiograph segmentation algorithm represents an approximation of pleural detachment area, not volumetric lung collapse.

---

## 2. Dataset & Annotation Constraints

1. **Single Consensus Ground Truth**:
   While SIIM-ACR annotations were created by board-certified thoracic radiologists, the dataset provides only a single composite binary mask per image. True inter-observer variability (which requires multiple independent masks from different radiologists for the same image) cannot be directly isolated from aleatoric boundary ambiguity.
2. **Lack of Clinical Context**:
   Real-world radiologist interpretation relies heavily on clinical history (e.g., recent central venous catheter insertion, mechanical ventilation peak pressures, penetrating trauma, sudden onset pleuritic chest pain). The deep learning model operates strictly in a vacuum on isolated pixel arrays.
3. **Severe Class Imbalance**:
   With ~78% negative cases, any evaluation metric that does not explicitly disaggregate positive and negative cohorts will yield misleadingly optimistic performance estimates.

---

## 3. Modeling & Uncertainty Approximations

1. **Epistemic Approximations**:
   Neither MC Dropout nor Deep Ensembles provide exact Bayesian posterior posteriors. MC Dropout represents a variational approximation conditioned on an ad-hoc dropout rate ($p=0.2$); Deep Ensembles represent a discrete collection of $M=5$ point estimates that explore only a finite subset of modes in the loss surface.
2. **Idealized Human Referral Assumption**:
   Our selective prediction simulation assumes that cases referred to a human radiologist are diagnosed with 100% accuracy. In reality, human radiologists also exhibit diagnostic error, fatigue, and inter-reader variation, particularly on subtle apical pneumothoraces.

---

## 4. Operational & Licensing Constraints

1. **Compute Boundaries**:
   To ensure the project is feasible alongside a full-time job within a 30-hour weekly Kaggle GPU budget, hyperparameter searches are constrained to a parsimonious ResNet34 U-Net backbone. Massive Vision Transformers (e.g., Swin-UNet) and larger ensemble sizes ($M > 5$) are intentionally omitted.
2. **Research-Only Licensing**:
   The SIIM-ACR dataset is restricted by its data use agreement to academic research and non-commercial portfolio demonstrations. It cannot be packaged into a commercial medical diagnostic tool.
