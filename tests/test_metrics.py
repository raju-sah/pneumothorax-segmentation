"""Unit tests for segmentation, calibration, and selective prediction metrics."""

import unittest
import numpy as np
from src.metrics.segmentation import (
    compute_dice_coefficient,
    compute_hausdorff95,
)
from src.metrics.uncertainty import (
    compute_esce,
    compute_brier_score,
)
from src.metrics.selective_prediction import (
    compute_risk_coverage_curve,
    compute_aurc,
)


class TestMetrics(unittest.TestCase):
    def test_dice_empty_masks(self):
        """Verify Dice edge cases on negative samples."""
        zeros = np.zeros((100, 100), dtype=np.uint8)
        ones = np.ones((100, 100), dtype=np.uint8)

        self.assertEqual(compute_dice_coefficient(zeros, zeros), 1.0)
        self.assertEqual(compute_dice_coefficient(zeros, ones), 0.0)
        self.assertEqual(compute_dice_coefficient(ones, zeros), 0.0)
        self.assertEqual(compute_dice_coefficient(ones, ones), 1.0)

    def test_hd95_empty_mask_penalty(self):
        """Verify HD95 returns max_dist penalty when one mask is empty."""
        zeros = np.zeros((50, 50), dtype=np.uint8)
        ones = np.ones((50, 50), dtype=np.uint8)

        self.assertEqual(compute_hausdorff95(zeros, zeros), 0.0)
        self.assertEqual(compute_hausdorff95(zeros, ones, max_dist=724.0), 724.0)

    def test_esce_and_brier(self):
        """Verify ESCE and Brier score computations."""
        targets = np.array([1, 1, 0, 0], dtype=np.uint8)
        probs = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)

        esce, _, _, _ = compute_esce(probs, targets, n_bins=5)
        brier = compute_brier_score(probs, targets)

        self.assertTrue(0.0 <= esce <= 1.0)
        self.assertTrue(0.0 <= brier <= 1.0)

    def test_selective_prediction_pipeline(self):
        """Verify that lower uncertainty cases retain higher Dice in selective prediction."""
        np.random.seed(42)
        n = 100
        uncertainties = np.linspace(0.1, 0.9, n)
        dices = 1.0 - uncertainties + np.random.normal(0, 0.02, n)
        dices = np.clip(dices, 0.0, 1.0)
        risks = 1.0 - dices

        coverages, risk_curve = compute_risk_coverage_curve(uncertainties, risks)
        aurc = compute_aurc(coverages, risk_curve)

        self.assertTrue(0.0 <= aurc <= 1.0)
        self.assertLess(risk_curve[0], risk_curve[-1])


if __name__ == "__main__":
    unittest.main()
