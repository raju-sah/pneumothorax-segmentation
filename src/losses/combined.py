"""Numerically stable Combined Binary Cross-Entropy and Soft Dice Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss with additive smoothing factor epsilon."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Soft Dice Loss.

        Args:
            logits: Pre-sigmoid logits of shape (B, 1, H, W).
            targets: Binary ground truth {0, 1} of shape (B, 1, H, W).

        Returns:
            Scalar loss tensor.
        """
        probs = torch.sigmoid(logits)

        # Flatten batch and spatial dimensions per sample or overall
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = (probs_flat.pow(2) + targets_flat.pow(2)).sum()

        dice_coeff = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_coeff


class CombinedBCEDiceLoss(nn.Module):
    """Balanced loss: 0.5 * BCEWithLogits + 0.5 * SoftDiceLoss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute balanced total loss.

        Args:
            logits: Pre-sigmoid logits of shape (B, 1, H, W).
            targets: Ground-truth binary masks of shape (B, 1, H, W).

        Returns:
            Scalar loss.
        """
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice
