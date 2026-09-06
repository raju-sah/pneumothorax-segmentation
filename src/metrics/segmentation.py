"""Segmentation overlap and boundary distance metrics."""

from typing import Dict
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_dice_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    empty_score: float = 1.0,
) -> float:
    """Compute 2D Dice Similarity Coefficient with explicit empty-mask handling.

    Args:
        y_true: Binary ground-truth mask {0, 1}.
        y_pred: Binary predicted mask {0, 1}.
        empty_score: Score assigned when both true and pred are completely empty (default: 1.0).

    Returns:
        float: Dice score in [0.0, 1.0].
    """
    y_true_sum = np.sum(y_true)
    y_pred_sum = np.sum(y_pred)

    if y_true_sum == 0 and y_pred_sum == 0:
        return empty_score
    if y_true_sum == 0 or y_pred_sum == 0:
        return 0.0

    intersection = np.sum(y_true * y_pred)
    return float(2.0 * intersection / (y_true_sum + y_pred_sum))


def compute_hausdorff95(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spacing: tuple = (1.0, 1.0),
    max_dist: float = 724.0,
) -> float:
    """Compute the 95th-percentile Hausdorff Distance (HD95) between boundary contours.

    Args:
        y_true: Binary ground-truth mask.
        y_pred: Binary predicted mask.
        spacing: Pixel spacing in mm (default: (1.0, 1.0)).
        max_dist: Maximum penalty distance if one mask is empty (default: 724.0 for 512x512).

    Returns:
        float: HD95 value in physical units (or pixels).
    """
    y_true_sum = np.sum(y_true)
    y_pred_sum = np.sum(y_pred)

    if y_true_sum == 0 and y_pred_sum == 0:
        return 0.0
    if y_true_sum == 0 or y_pred_sum == 0:
        return max_dist

    # Extract boundary edges using distance transform
    dt_true = distance_transform_edt(1 - y_true, sampling=spacing)
    dt_pred = distance_transform_edt(1 - y_pred, sampling=spacing)

    # Distances from pred surface to true
    pred_border = (
        y_pred & ~np.pad(y_pred[1:, 1:], ((0, 1), (0, 1)), mode="constant")
        if np.any(y_pred)
        else y_pred
    )
    true_border = (
        y_true & ~np.pad(y_true[1:, 1:], ((0, 1), (0, 1)), mode="constant")
        if np.any(y_true)
        else y_true
    )

    dists_to_true = dt_true[pred_border > 0]
    dists_to_pred = dt_pred[true_border > 0]

    if len(dists_to_true) == 0 or len(dists_to_pred) == 0:
        return max_dist

    hd95 = max(
        np.percentile(dists_to_true, 95),
        np.percentile(dists_to_pred, 95),
    )
    return float(hd95)


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive segmentation evaluation suite for a single radiograph.

    Args:
        y_true: Binary ground truth mask {0, 1} of shape (H, W).
        y_pred_prob: Continuous predicted probability map in [0, 1] of shape (H, W).
        threshold: Decision threshold for binarization (default: 0.5).

    Returns:
        Dict with keys: ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'is_positive'].
    """
    y_pred = (y_pred_prob >= threshold).astype(np.uint8)
    y_true = (y_true > 0).astype(np.uint8)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    is_positive = float(np.sum(y_true) > 0)
    dice = compute_dice_coefficient(y_true, y_pred)
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else (1.0 if is_positive == 0 else 0.0)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if np.sum(y_pred) == 0 else 0.0)

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "is_positive": is_positive,
    }
