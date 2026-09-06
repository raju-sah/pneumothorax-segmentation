# DATASET AUDIT: SIIM-ACR Pneumothorax Segmentation Dataset

## 1. Dataset Overview & Stage 1 vs. Stage 2 Disambiguation

A common misconception in literature is citing "12,047 images" as the entire competition dataset. The actual lifecycle was structured across two distinct stages:

| Cohort | Sub-cohort | Image Count | Positive Cases | Negative Cases | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Train | 10,675 | 2,379 (22.29%) | 8,296 (77.71%) | Released at challenge launch |
| | Test | 1,372 | Withheld | Withheld | Used for Phase 1 leaderboard |
| **Stage 2** | Train | **12,047** | **2,669 (22.15%)** | **9,378 (77.85%)** | Formed by merging Stage 1 Train + Stage 1 Test |
| | Test | 3,205 | Withheld | Withheld | Final private leaderboard test set |
| **Total Images** | — | **15,252** | — | — | Full competition data footprint |

For this academic research project, our universe is the **12,047 fully annotated Stage 2 training cohort**, which possesses confirmed pixel-level ground truth consensus masks.

---

## 2. Mask Distribution & Multiple-Mask Cases

- **Negative Cases**: 9,378 images (77.85%) have no pneumothorax (encoded as ` -1`).
- **Positive Cases**: 2,669 images (22.15%) contain one or more pneumothorax lesions.
- **Multiple Lesions per Image**:
  - In the raw `train-rle.csv`, there are ~13,299 total rows for the 12,047 unique `ImageId`s.
  - ~3,200 rows represent individual mask instances across the 2,669 positive images.
  - Multiple distinct regions arise from bilateral pneumothoraces, separated apical and basilar pockets, or multilocular air collections.
  - **Audit Takeaway**: Naive CSV parsing without grouping by `ImageId` causes dropped annotations or repeated evaluation records. All RLE masks corresponding to an `ImageId` must be merged using bitwise logical OR.

---

## 3. DICOM Metadata Extraction & Tag Audit

Header extraction across the dataset reveals the following parameters:

| DICOM Tag | Keyword | Typical Values | Clinical / Pipeline Implication |
| :--- | :--- | :--- | :--- |
| `(0010, 0020)` | `PatientID` | Anonymized UUID string (e.g., `88c14312-...`) | **Critical**: Multiple images belong to the same patient. Must group by `PatientID`. |
| `(0010, 0040)` | `PatientSex` | `'M'`, `'F'` | Evaluated for demographic parity and bias analysis. |
| `(0010, 0010)` | `PatientAge` | Integer string (e.g., `'45'`) | Age distribution spans pediatric to geriatric cohorts. |
| `(0018, 5101)` | `ViewPosition` | `'AP'`, `'PA'` | `'PA'` (erect) vs `'AP'` (portable/supine). AP views present higher visual difficulty. |
| `(0028, 0004)` | `PhotometricInterpretation` | `'MONOCHROME1'`, `'MONOCHROME2'` | **Critical Trap**: `MONOCHROME1` must be inverted; otherwise air appears white and bone appears black. |
| `(0028, 0030)` | `PixelSpacing` | ~`[0.143, 0.143]` to `[0.168, 0.168]` mm | Physical pixel dimensions for true metric area calculation. |
| `(0028, 0010)` | `Rows` | `1024` | Native image height. |
| `(0028, 0011)` | `Columns` | `1024` | Native image width. |

---

## 4. Radiographic Acquisition & View Position Breakdown

The dataset comprises approximately:
- **PA (Posteroanterior)**: ~60% of images. Standard outpatient or ambulatory emergency department views taken with the patient erect, deep inspiration, and scapulae rotated away from the lung fields.
- **AP (Anteroposterior)**: ~40% of images. Portable bedside views taken in critically ill, supine, or immobilized ICU patients. Characterized by magnification of the mediastinum, scapular overlap across apical lung zones, and variable inspiratory effort.
- **Scientific Implication**: Subgroup analysis of uncertainty across AP vs. PA projections directly answers whether epistemic uncertainty correlates with clinical acquisition quality.

---

## 5. Licensing & Data Use Constraints

- **Licensing Authority**: Society for Imaging Informatics in Medicine (SIIM) and American College of Radiology (ACR).
- **Terms**: Strictly restricted to non-commercial academic research and educational algorithm evaluation.
- **Redistribution**: Raw DICOM files cannot be redistributed on open public mirrors; downstream research code must provide scripts that ingest the official Kaggle dataset directly.
