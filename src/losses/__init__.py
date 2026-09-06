"""Loss functions for training segmentation networks."""

from src.losses.combined import CombinedBCEDiceLoss, SoftDiceLoss

__all__ = ["CombinedBCEDiceLoss", "SoftDiceLoss"]
