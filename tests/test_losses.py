"""Unit tests for combined BCE and Soft Dice loss."""

import unittest
import torch
from src.losses.combined import CombinedBCEDiceLoss, SoftDiceLoss


class TestLosses(unittest.TestCase):
    def test_soft_dice_loss_empty_masks(self):
        """Test Soft Dice Loss when both predictions and targets are empty."""
        dice_loss = SoftDiceLoss(smooth=1.0)
        logits = torch.full((2, 1, 64, 64), -10.0, dtype=torch.float32)
        targets = torch.zeros((2, 1, 64, 64), dtype=torch.float32)

        loss = dice_loss(logits, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertLess(loss.item(), 0.1)

    def test_combined_loss_backward(self):
        """Test that combined loss computes gradients without NaN."""
        loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1.0)
        logits = torch.randn((2, 1, 64, 64), requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        self.assertFalse(torch.isnan(loss))
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


if __name__ == "__main__":
    unittest.main()
