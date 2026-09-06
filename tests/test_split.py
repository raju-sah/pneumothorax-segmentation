"""Unit tests for Patient-Grouped Stratified Splitting and zero-leakage verification."""

import unittest
import pandas as pd
import numpy as np
from src.data.split import create_patient_grouped_splits


class TestDataSplit(unittest.TestCase):
    def test_zero_patient_leakage(self):
        """Verify that no patient appears in multiple partitions (train, val, or test)."""
        np.random.seed(42)
        n_images = 300
        n_patients = 100

        patient_ids = [f"patient_{i:03d}" for i in range(n_patients)]
        assigned_patients = np.random.choice(patient_ids, size=n_images)

        df = pd.DataFrame(
            {
                "ImageId": [f"img_{i:04d}" for i in range(n_images)],
                "PatientID": assigned_patients,
                "HasPneumothorax": np.random.choice([0, 1], size=n_images, p=[0.78, 0.22]),
                "ViewPosition": np.random.choice(["AP", "PA"], size=n_images, p=[0.40, 0.60]),
            }
        )

        df_splits = create_patient_grouped_splits(df, test_size=0.20, n_splits=5, seed=42)

        self.assertIn("Fold", df_splits.columns)

        # Verify patient sets are strictly disjoint
        test_patients = set(df_splits[df_splits["Fold"] == "test"]["PatientID"])
        dev_patients = set(df_splits[df_splits["Fold"] != "test"]["PatientID"])

        self.assertEqual(len(test_patients.intersection(dev_patients)), 0)

        # Verify each CV fold is also disjoint
        folds = [f for f in df_splits["Fold"].unique() if f != "test"]
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                p_i = set(df_splits[df_splits["Fold"] == folds[i]]["PatientID"])
                p_j = set(df_splits[df_splits["Fold"] == folds[j]]["PatientID"])
                self.assertEqual(len(p_i.intersection(p_j)), 0)


if __name__ == "__main__":
    unittest.main()
