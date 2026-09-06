"""Execute EXP-01 (Dataset Audit) and EXP-02 (Patient-Grouped Stratified Partitioning).

Generates:
1. data/processed/metadata_master.csv: Consolidated master dataset manifest.
2. data/processed/patient_splits.csv: Zero-leakage stratified grouped train/val/test splits.
"""

import os
import json
import pandas as pd
import numpy as np

from src.utils.seed import set_seed
from src.data.split import create_patient_grouped_splits


def build_master_manifest(
    meta_csv: str = "data/raw/sarmat_output/meta_train.csv",
    rle_csv: str = "data/raw/train-rle.csv",
    output_meta: str = "data/processed/metadata_master.csv",
    output_splits: str = "data/processed/patient_splits.csv",
    test_size: float = 0.20,
    n_splits: int = 5,
    seed: int = 42,
) -> None:
    set_seed(seed)
    os.makedirs(os.path.dirname(output_meta), exist_ok=True)

    print("Loading raw metadata and annotations...")
    df_meta = pd.read_csv(meta_csv)
    df_rle = pd.read_csv(rle_csv)

    df_meta["ImageId"] = df_meta["SOP Instance UID"].str.strip()
    df_rle["ImageId"] = df_rle["ImageId"].str.strip()

    # Aggregate RLE masks per ImageId
    print("Aggregating multi-mask instances...")
    rle_agg = (
        df_rle.groupby("ImageId")
        .agg(
            EncodedPixelsList=("EncodedPixels", list),
            MaskCount=("EncodedPixels", lambda s: sum(1 for x in s if str(x).strip() != "-1")),
            HasPneumothorax=("EncodedPixels", lambda s: int(any(str(x).strip() != "-1" for x in s))),
        )
        .reset_index()
    )

    # Standardize column names
    rename_cols = {
        "Patient ID": "PatientID",
        "View Position": "ViewPosition",
        "Patient's Age": "PatientAge",
        "Patient's Sex": "PatientSex",
        "Photometric Interpretation": "PhotometricInterpretation",
        "Pixel Spacing": "PixelSpacing",
        "Rows": "Rows",
        "Columns": "Columns",
    }

    df_clean_meta = df_meta[["ImageId"] + list(rename_cols.keys())].rename(columns=rename_cols)

    # Inner join on ImageId to retain validated annotated images
    df_master = pd.merge(df_clean_meta, rle_agg, on="ImageId", how="inner")

    # Serialize EncodedPixelsList to JSON string for CSV storage
    df_master["EncodedPixelsList"] = df_master["EncodedPixelsList"].apply(json.dumps)

    print(f"EXP-01 Audit: Master dataset contains {len(df_master)} unique images.")
    pos_count = df_master["HasPneumothorax"].sum()
    neg_count = len(df_master) - pos_count
    print(f"  -> Positive cases: {pos_count} ({pos_count / len(df_master) * 100:.2f}%)")
    print(f"  -> Negative cases: {neg_count} ({neg_count / len(df_master) * 100:.2f}%)")
    print(f"  -> Unique patients: {df_master['PatientID'].nunique()}")
    print(f"  -> View Position: {df_master['ViewPosition'].value_counts().to_dict()}")
    print(f"  -> Sex: {df_master['PatientSex'].value_counts().to_dict()}")

    df_master.to_csv(output_meta, index=False)
    print(f"Saved master metadata to: {output_meta}")

    # EXP-02: Patient-Grouped Stratified Partitioning
    print("\nExecuting EXP-02: Patient-Grouped Stratified Splitting...")
    df_splits = create_patient_grouped_splits(
        df=df_master,
        test_size=test_size,
        n_splits=n_splits,
        seed=seed,
    )

    # Verify fold distributions
    print("\nSplit Distribution Audit:")
    for fold in sorted(df_splits["Fold"].unique()):
        sub = df_splits[df_splits["Fold"] == fold]
        n_imgs = len(sub)
        n_pts = sub["PatientID"].nunique()
        prev = sub["HasPneumothorax"].mean() * 100
        pa_ratio = (sub["ViewPosition"] == "PA").mean() * 100
        print(
            f"  Fold {fold:>4}: {n_imgs:>5} images, {n_pts:>5} patients | "
            f"Pneumo: {prev:.2f}% | PA View: {pa_ratio:.2f}%"
        )

    df_splits.to_csv(output_splits, index=False)
    print(f"\nSaved master split manifest to: {output_splits}")


if __name__ == "__main__":
    build_master_manifest()
