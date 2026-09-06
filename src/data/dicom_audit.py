"""DICOM audit utility to extract tags and build the master dataset manifest."""

import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import pydicom
from tqdm import tqdm


def extract_dicom_metadata(dcm_path: str) -> Dict[str, Any]:
    """Extract standard demographic and acquisition tags from a DICOM header.

    Args:
        dcm_path: Path to the .dcm file.

    Returns:
        Dict with extracted metadata values.
    """
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    return {
        "ImageId": Path(dcm_path).stem,
        "PatientID": getattr(dcm, "PatientID", "UNKNOWN"),
        "PatientAge": getattr(dcm, "PatientAge", None),
        "PatientSex": getattr(dcm, "PatientSex", "U"),
        "ViewPosition": getattr(dcm, "ViewPosition", "PA"),
        "PhotometricInterpretation": getattr(dcm, "PhotometricInterpretation", "MONOCHROME2"),
        "Rows": getattr(dcm, "Rows", 1024),
        "Columns": getattr(dcm, "Columns", 1024),
    }


def audit_dicom_directory(
    dicom_dir: str,
    rle_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """Scan all DICOM files, extract metadata, aggregate RLE annotations, and save manifest.

    Args:
        dicom_dir: Directory containing DICOM files.
        rle_csv: Path to Kaggle train-rle.csv.
        output_csv: Path to write the consolidated manifest CSV.

    Returns:
        pd.DataFrame: Consolidated master manifest.
    """
    dcm_files = list(Path(dicom_dir).rglob("*.dcm"))
    records: List[Dict[str, Any]] = []

    for f in tqdm(dcm_files, desc="Auditing DICOMs"):
        try:
            meta = extract_dicom_metadata(str(f))
            meta["FilePath"] = str(f)
            records.append(meta)
        except Exception as e:
            print(f"Warning: Failed to read {f}: {e}")

    df_meta = pd.DataFrame(records)

    # Load annotations
    if os.path.exists(rle_csv):
        df_rle = pd.read_csv(rle_csv)
        # Handle multiple masks: group by ImageId
        rle_summary = (
            df_rle.groupby("ImageId")
            .agg(
                EncodedPixelsList=("EncodedPixels", list),
                HasPneumothorax=(
                    "EncodedPixels",
                    lambda x: int(any(str(r).strip() != "-1" for r in x)),
                ),
            )
            .reset_index()
        )
        df_master = pd.merge(df_meta, rle_summary, on="ImageId", how="left")
        df_master["HasPneumothorax"] = df_master["HasPneumothorax"].fillna(0).astype(int)
    else:
        df_master = df_meta
        df_master["HasPneumothorax"] = 0

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_master.to_csv(output_csv, index=False)
    return df_master
