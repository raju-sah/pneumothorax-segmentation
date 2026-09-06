"""PyTorch Dataset implementation for SIIM-ACR pneumothorax radiographs and masks."""

from typing import Optional, Callable, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None
    ToTensorV2 = None


def get_training_transforms(image_size: int = 512) -> Any:
    """Standard, radiologically sound training data augmentations.

    Includes horizontal flipping, subtle rotation (+/- 10 deg), and contrast adjustments.
    Vertical flipping is strictly omitted as it violates anatomical cephalocaudal orientation.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for data transforms.")
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ]
    )


def get_validation_transforms(image_size: int = 512) -> Any:
    """Validation/testing transforms: resize and normalize only."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for data transforms.")
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ]
    )


class PneumothoraxDataset(Dataset):
    """Dataset for pneumothorax radiographs, supporting in-memory arrays or cached arrays."""

    def __init__(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        image_ids: Optional[np.ndarray] = None,
        transforms: Optional[Callable] = None,
    ):
        """Args:
            images: Array of shape (N, H, W) or (N, H, W, C) in uint8 [0, 255].
            masks: Binary ground-truth masks of shape (N, H, W) in uint8 {0, 1}.
            image_ids: Optional array of string ImageIds.
            transforms: Albumentations Compose pipeline.
        """
        self.images = images
        self.masks = masks
        self.image_ids = image_ids
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.images[idx]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Standardize 1-channel grayscale to 3-channel for pretrained backbones
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]

        if self.transforms is not None:
            if mask is not None:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"].unsqueeze(0).float()
            else:
                augmented = self.transforms(image=image)
                image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if mask is not None:
                mask = torch.from_numpy(mask).unsqueeze(0).float()

        item: Dict[str, Any] = {"image": image}
        if mask is not None:
            item["mask"] = mask
        if self.image_ids is not None:
            item["image_id"] = self.image_ids[idx]

        return item
