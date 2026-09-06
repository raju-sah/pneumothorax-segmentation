"""Patient-Grouped Stratified Splitting to eliminate patient-level data leakage.

Ensures that:
1. Every unique PatientID resides in exactly ONE partition (train, val, or test).
2. The prevalence of pneumothorax-positive patients is balanced across partitions.
3. The distribution of view positions (AP vs. PA) is preserved across folds.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def create_patient_grouped_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign patient-grouped stratified fold identifiers to a dataset manifest.

    Args:
        df: Master DataFrame with columns ['ImageId', 'PatientID', 'HasPneumothorax', 'ViewPosition'].
        test_size: Proportion of unique patients assigned to holdout test set (default: 0.20).
        n_splits: Number of cross-validation folds for development set (default: 5).
        seed: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame: Copy of `df` with additional column 'Fold':
            - 'test' for holdout evaluation cases.
            - 0, 1, ..., n_splits-1 for cross-validation development cases.
    """
    df = df.copy()

    # Step 1: Create a patient-level summary dataframe
    patient_df = (
        df.groupby("PatientID")
        .agg(
            has_pneumo=("HasPneumothorax", "max"),
            view_position=("ViewPosition", lambda x: x.mode()[0] if len(x) > 0 else "PA"),
            image_count=("ImageId", "count"),
        )
        .reset_index()
    )

    # Composite stratification label: combination of pneumothorax presence and dominant view
    patient_df["strat_label"] = (
        patient_df["has_pneumo"].astype(str) + "_" + patient_df["view_position"].astype(str)
    )

    # Step 2: Separate holdout test cohort (patient-grouped)
    sgkf_test = StratifiedGroupKFold(
        n_splits=int(round(1.0 / test_size)),
        shuffle=True,
        random_state=seed,
    )

    # Take the first split as test holdout
    dev_idx, test_idx = next(
        sgkf_test.split(
            X=patient_df,
            y=patient_df["strat_label"],
            groups=patient_df["PatientID"],
        )
    )

    test_patients = set(patient_df.iloc[test_idx]["PatientID"])
    dev_patient_df = patient_df.iloc[dev_idx].reset_index(drop=True)

    # Step 3: Partition development cohort into k cross-validation folds
    sgkf_folds = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    dev_patient_df["dev_fold"] = -1
    for fold, (_, val_idx) in enumerate(
        sgkf_folds.split(
            X=dev_patient_df,
            y=dev_patient_df["strat_label"],
            groups=dev_patient_df["PatientID"],
        )
    ):
        dev_patient_df.loc[val_idx, "dev_fold"] = fold

    patient_fold_map = dict(zip(dev_patient_df["PatientID"], dev_patient_df["dev_fold"]))

    # Step 4: Map back to image-level dataframe
    def assign_fold(patient_id: str) -> str:
        if patient_id in test_patients:
            return "test"
        return str(patient_fold_map.get(patient_id, "train"))

    df["Fold"] = df["PatientID"].apply(assign_fold)

    # Verify zero patient leakage
    test_p = set(df[df["Fold"] == "test"]["PatientID"])
    dev_p = set(df[df["Fold"] != "test"]["PatientID"])
    assert len(test_p.intersection(dev_p)) == 0, "Catastrophic error: Patient leakage detected!"

    return df
