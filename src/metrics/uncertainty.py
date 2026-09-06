"""Uncertainty calibration, reliability, and error correlation metrics."""

from typing import Dict, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compute_esce(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Segmentation Calibration Error (ESCE) across binned probabilities.

    Args:
        probs: Continuous predicted probabilities in [0, 1] (flattened array).
        targets: Binary ground truth in {0, 1} (flattened array).
        n_bins: Number of equal-width probability bins (default: 10).

    Returns:
        Tuple:
            - esce (float): Weighted average calibration error.
            - bin_confs (np.ndarray): Mean confidence in each bin.
            - bin_accs (np.ndarray): Empirical accuracy in each bin.
            - bin_counts (np.ndarray): Number of pixels in each bin.
    """
    probs_flat = probs.flatten()
    targets_flat = targets.flatten()
    total_pixels = len(probs_flat)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confs = np.zeros(n_bins)
    bin_accs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    esce = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs_flat > low) & (probs_flat <= high) if i > 0 else (probs_flat >= low) & (probs_flat <= high)
        count = np.sum(mask)
        bin_counts[i] = count

        if count > 0:
            conf = np.mean(probs_flat[mask])
            acc = np.mean(targets_flat[mask])
            bin_confs[i] = conf
            bin_accs[i] = acc
            esce += (count / total_pixels) * np.abs(acc - conf)

    return float(esce), bin_confs, bin_accs, bin_counts


def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute the Brier Score (mean squared error of predicted probabilities).

    Args:
        probs: Predicted probability map.
        targets: Binary ground truth mask.

    Returns:
        float: Brier Score.
    """
    return float(np.mean((probs - targets) ** 2))


def compute_error_detection_auroc(
    uncertainty_map: np.ndarray,
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute the Area Under the ROC Curve for identifying prediction errors via uncertainty.

    Binary Error E = |targets - (probs >= threshold)|.

    Args:
        uncertainty_map: Continuous uncertainty map (e.g., variance or entropy).
        probs: Predicted probability map.
        targets: Ground truth mask.
        threshold: Decision threshold.

    Returns:
        float: AUROC for error detection. Returns 0.5 if error is uniform.
    """
    preds = (probs >= threshold).astype(np.uint8)
    error = np.abs(targets - preds).flatten()
    uncertainty = uncertainty_map.flatten()

    # If there are zero errors or all errors, AUROC is undefined -> return 0.5
    if np.all(error == 0) or np.all(error == 1):
        return 0.5

    # Subsample if pixel count is massive for speed
    if len(error) > 100_000:
        idx = np.random.choice(len(error), size=100_000, replace=False)
        error = error[idx]
        uncertainty = uncertainty[idx]

    try:
        return float(roc_auc_score(error, uncertainty))
    except Exception:
        return 0.5


def compute_uncertainty_error_correlation(
    uncertainty_map: np.ndarray,
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Spearman rank correlation between pixel uncertainty and binary error.

    Args:
        uncertainty_map: Continuous uncertainty map.
        probs: Predicted probability map.
        targets: Ground truth mask.
        threshold: Decision threshold.

    Returns:
        float: Spearman rank correlation coefficient rho.
    """
    preds = (probs >= threshold).astype(np.uint8)
    error = np.abs(targets - preds).flatten()
    uncertainty = uncertainty_map.flatten()

    if np.all(error == error[0]) or np.all(uncertainty == uncertainty[0]):
        return 0.0

    if len(error) > 50_000:
        idx = np.random.choice(len(error), size=50_000, replace=False)
        error = error[idx]
        uncertainty = uncertainty[idx]

    rho, _ = spearmanr(uncertainty, error)
    return float(rho) if not np.isnan(rho) else 0.0
