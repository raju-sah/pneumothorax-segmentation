"""Unit tests for ResNet34 U-Net and MC Dropout behavior."""

import unittest
import torch
from src.models.unet import PneumothoraxUNet


class TestModels(unittest.TestCase):
    def test_unet_shapes(self):
        """Verify input-output tensor shapes through U-Net."""
        model = PneumothoraxUNet(dropout_rate=0.2, pretrained=False)
        x = torch.randn(2, 3, 64, 64)

        logits = model(x)
        self.assertEqual(logits.shape, (2, 1, 64, 64))

    def test_deterministic_prediction(self):
        """Verify deterministic prediction bounds and keys."""
        model = PneumothoraxUNet(dropout_rate=0.0, pretrained=False)
        x = torch.randn(2, 3, 64, 64)

        out = model.predict_deterministic(x)
        self.assertIn("prob", out)
        self.assertIn("entropy", out)
        self.assertEqual(out["prob"].shape, (2, 1, 64, 64))
        self.assertTrue((out["prob"] >= 0.0).all() and (out["prob"] <= 1.0).all())
        self.assertTrue((out["entropy"] >= 0.0).all() and (out["entropy"] <= 1.0).all())

    def test_mc_dropout_variance(self):
        """Verify MC Dropout produces valid variance and entropy tensors."""
        model = PneumothoraxUNet(dropout_rate=0.3, pretrained=False)
        x = torch.randn(2, 3, 64, 64)

        out = model.predict_mc_dropout(x, num_samples=5)
        self.assertIn("mean", out)
        self.assertIn("variance", out)
        self.assertIn("entropy", out)
        self.assertEqual(out["mean"].shape, (2, 1, 64, 64))
        self.assertEqual(out["variance"].shape, (2, 1, 64, 64))
        self.assertTrue((out["variance"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
