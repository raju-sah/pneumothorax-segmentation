# DATA CARD: SIIM-ACR Pneumothorax Segmentation Dataset

Following the Datasheets for Datasets framework proposed by Gebru et al. (2021).

---

## 1. Motivation
- **Purpose**: Created for the 2019 SIIM-ACR Pneumothorax Segmentation Challenge to advance machine learning algorithms capable of detecting and localizing pneumothorax on chest radiographs to improve emergency triage.
- **Curators**: Curated collaboratively by the Society for Imaging Informatics in Medicine (SIIM), the American College of Radiology (ACR), and the Society of Thoracic Radiology (STR).

---

## 2. Dataset Composition
- **Instances**: Frontal chest radiographs in DICOM format and associated Run-Length Encoded (RLE) ground-truth masks in CSV format.
- **Counts**:
  - Stage 2 Training Cohort: **12,047 unique images** (9,378 negative [~77.85%], 2,669 positive [~22.15%]).
  - Stage 2 Test Cohort: **3,205 unique images**.
- **Metadata Fields**: `PatientID`, `PatientAge`, `PatientSex`, `ViewPosition` (AP vs. PA), `PhotometricInterpretation`, `PixelSpacing`.
- **Sensitivities / Identifiers**: All images were de-identified in compliance with HIPAA Safe Harbor guidelines; `PatientID` is a pseudonymized hash.

---

## 3. Collection Process & Radiologist Annotations
- **Source Population**: Sourced originally from the National Institutes of Health (NIH) Clinical Center ChestX-ray14 archive.
- **Annotation Methodology**: Radiographs were re-annotated from scratch by a panel of board-certified thoracic radiologists from the Society of Thoracic Radiology (STR). Radiologists delineated fine polygonal contours around all visible visceral pleural separations, which were subsequently converted into column-major Run-Length Encodings (RLE).

---

## 4. Preprocessing & Data Cleaning in this Project
- **Photometric Normalization**: Images with `MONOCHROME1` headers are inverted ($I_{\text{new}} = 255 - I$) to match the standard `MONOCHROME2` presentation.
- **Resolution**: Resampled from native $1024 \times 1024$ to $512 \times 512$ using bilinear interpolation for images and nearest-neighbor for masks.
- **Multi-Mask Union**: Images with multiple RLE rows are merged via bitwise logical OR.
- **Zero-Leakage Grouping**: Partitioned strictly by `PatientID` to guarantee that no individual patient contributes radiographs to multiple folds.

---

## 5. Uses & Distribution
- **Permitted Uses**: Non-commercial academic research, algorithmic benchmarking, and scientific educational portfolios.
- **Prohibited Uses**: Commercial diagnostic deployment, patient re-identification attempts.
- **Distribution Channel**: Hosted by Kaggle under competition slug `siim-acr-pneumothorax-segmentation`.
