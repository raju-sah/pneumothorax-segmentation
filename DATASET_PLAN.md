# DATASET PLAN: Data Sourcing, Storage, Preprocessing & Caching

## 1. Data Source & Access Protocol

The primary dataset is the official **SIIM-ACR Pneumothorax Segmentation Challenge Dataset** hosted on Kaggle.

### Acquisition Steps
- Competition slug: `siim-acr-pneumothorax-segmentation`
- Download command (Kaggle CLI):
  ```bash
  kaggle competitions download -c siim-acr-pneumothorax-segmentation
  ```
- Or access via direct Kaggle Notebook dataset mounting:
  `../input/siim-acr-pneumothorax-segmentation/`

---

## 2. Directory & Storage Architecture

### 2.1 Raw Data Directory Structure
```
data/
├── raw/
│   ├── dicom-images-train/           # Stage 2 training DICOM files (~12,047 files)
│   ├── dicom-images-test/            # Stage 2 test DICOM files (~3,205 files)
│   ├── train-rle.csv                 # RLE ground-truth annotations for train set
│   └── sample_submission.csv
```

### 2.2 Processed & Cached Data Directory Structure
To avoid parsing 12,000+ uncompressed DICOM files per epoch (which exhausts Kaggle disk I/O limits), data is pre-converted once into a compressed binary format:
```
data/
├── processed/
│   ├── metadata_master.csv           # Consolidated metadata extracted from all DICOM headers
│   ├── patient_splits.csv            # Master manifest with fold assignments (zero patient leakage)
│   └── cache_512x512/
│       ├── images_512.h5             # uint8 array of shape (N, 512, 512), gzip compressed
│       ├── masks_512.h5              # uint8 binary masks of shape (N, 512, 512), gzip compressed
│       └── index_mapping.json        # Maps ImageId -> index in HDF5 archive
```

---

## 3. Preprocessing Specification

### 3.1 DICOM Decoding & Pixel Standardization
Each radiograph is processed through the following deterministic pipeline:
1. **Reading**: Load DICOM file via `pydicom.dcmread(filepath)`.
2. **Photometric Interpretation Correction**:
   - Check `dataset.PhotometricInterpretation`.
   - If `'MONOCHROME1'`: Radiopaque bone appears black and air appears white.
     $$\text{Image}_{\text{corrected}} = \text{max\_pixel} - \text{pixel\_array}$$
   - If `'MONOCHROME2'`: Air is radiolucent (black) and bone is dense (white). No inversion.
3. **Rescaling / Windowing**:
   - Apply `RescaleIntercept` and `RescaleSlope` if present in the header.
   - Standardize dynamic range to $[0, 255]$ uint8 via min-max normalization.
4. **Spatial Resampling**:
   - Resample from native $1024 \times 1024$ to $512 \times 512$ using bilinear interpolation (`cv2.INTER_LINEAR`).

### 3.2 Annotation & RLE Parsing
- **Encoding Rule**: SIIM-ACR uses Fortran-order (column-major) 1-based indexing.
- **Sentinel Rule**: Values of ` -1` represent negative cases (no pneumothorax) $\rightarrow$ binary mask of all zeros.
- **Multi-Mask Aggregation**:
  ```python
  # Multiple rows with identical ImageId are aggregated via bitwise OR
  composite_mask = np.zeros((1024, 1024), dtype=np.uint8)
  for rle in rle_list:
      if rle.strip() != "-1":
          composite_mask |= rle2mask(rle, 1024, 1024)
  # Downsample mask to 512x512 via nearest-neighbor to prevent fractional labels
  mask_512 = cv2.resize(composite_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
  ```

---

## 4. Stratified Patient-Grouped Partition

1. Group all records by `PatientID` extracted from DICOM headers.
2. For each unique patient, assign:
   - Patient Label: `1` if any radiograph belonging to that patient has pneumothorax, else `0`.
   - Patient Dominant View: `AP` or `PA`.
3. Perform a 5-fold Stratified Group K-Fold split on the development cohort (80%, ~9,637 images) and reserve an untouched 20% holdout test cohort (~2,410 images).
4. Save the manifest to `data/processed/patient_splits.csv` with columns:
   `[ImageId, PatientID, ViewPosition, PatientAge, PatientSex, HasPneumothorax, MaskAreaFraction, SplitFold]`.
