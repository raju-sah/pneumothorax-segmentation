"""Selective prediction and clinical triage referral simulation."""

from typing import List, Dict, Tuple, Optional
import numpy as np


def aggregate_case_uncertainty(
    uncertainty_map: np.ndarray,
    method: str = "top_k",
    k: int = 500,
) -> float:
    """Aggregate a 2D dense uncertainty map into a single case-level scalar score.

    Args:
        uncertainty_map: 2D array of pixel uncertainties (H, W).
        method: 'top_k' (default), 'mean', or 'max'.
        k: Number of highest-uncertainty pixels to average if method='top_k'.

    Returns:
        float: Scalar case uncertainty score.
    """
    flat = uncertainty_map.flatten()
    if method == "mean":
        return float(np.mean(flat))
    elif method == "max":
        return float(np.max(flat))
    elif method == "top_k":
        k = min(k, len(flat))
        # Partition to find top-k efficiently without full sort
        top_k_vals = np.partition(flat, -k)[-k:]
        return float(np.mean(top_k_vals))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_risk_coverage_curve(
    case_uncertainties: np.ndarray,
    case_risks: np.ndarray,
    coverage_steps: int = 50,
    min_coverage: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical Risk-Coverage Pareto curve by thresholding case uncertainties.

    Cases with the lowest uncertainty are retained first.

    Args:
        case_uncertainties: Array of shape (N,) containing scalar uncertainty for each radiograph.
        case_risks: Array of shape (N,) containing empirical error/risk (e.g., 1.0 - Dice).
        coverage_steps: Number of discrete coverage points to evaluate.
        min_coverage: Minimum coverage threshold (default: 0.20).

    Returns:
        Tuple:
            - coverages (np.ndarray): Array of retained coverage fractions from min_coverage to 1.0.
            - risks (np.ndarray): Mean risk of the retained cohort at each coverage level.
    """
    n_samples = len(case_uncertainties)
    sorted_indices = np.argsort(case_uncertainties)  # Ascending order (lowest uncertainty first)

    coverages = np.linspace(min_coverage, 1.0, coverage_steps)
    risks = np.zeros(coverage_steps)

    for i, cov in enumerate(coverages):
        retain_count = max(1, int(round(cov * n_samples)))
        retained_idx = sorted_indices[:retain_count]
        risks[i] = float(np.mean(case_risks[retained_idx]))

    return coverages, risks


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Compute the Area Under the Risk-Coverage Curve (AURC) via trapezoidal integration.

    Args:
        coverages: Array of coverage values in [0, 1].
        risks: Array of empirical risks corresponding to each coverage.

    Returns:
        float: AURC value (lower is better). Standard definition: trapezoid of
        risk over coverage directly (no rescaling of the coverage axis).
    """
    # ponytail: no coverage rescale; coverages already in [0, 1].
    try:
        from scipy.integrate import trapezoid
        return float(trapezoid(risks, coverages))
    except ImportError:
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return float(trapz_fn(risks, cov_norm))


def evaluate_selective_prediction(
    case_uncertainties: np.ndarray,
    case_dices: np.ndarray,
    target_coverages: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Calculate summary triage metrics at standard clinical referral thresholds.

    Args:
        case_uncertainties: Case-level uncertainty scores.
        case_dices: Case-level Dice scores.
        target_coverages: List of retained fractions to evaluate (default: [0.70, 0.80, 0.90, 1.00]).

    Returns:
        Dict with retained mean Dice at each coverage and overall AURC.
    """
    if target_coverages is None:
        target_coverages = [0.70, 0.80, 0.90, 1.00]

    case_risks = 1.0 - case_dices
    coverages, risks = compute_risk_coverage_curve(case_uncertainties, case_risks)
    aurc = compute_aurc(coverages, risks)

    results: Dict[str, float] = {"aurc": aurc}
    n_samples = len(case_uncertainties)
    sorted_indices = np.argsort(case_uncertainties)

    for cov in target_coverages:
        retain_count = max(1, int(round(cov * n_samples)))
        retained_idx = sorted_indices[:retain_count]
        retained_dice = float(np.mean(case_dices[retained_idx]))
        results[f"dice_at_coverage_{int(cov * 100)}"] = retained_dice

    return results
