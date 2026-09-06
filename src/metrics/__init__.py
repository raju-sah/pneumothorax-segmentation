"""Evaluation metrics for segmentation overlap, uncertainty calibration, and selective prediction."""

from src.metrics.segmentation import (
    compute_dice_coefficient,
    compute_hausdorff95,
    compute_segmentation_metrics,
)
from src.metrics.uncertainty import (
    compute_esce,
    compute_brier_score,
    compute_error_detection_auroc,
    compute_uncertainty_error_correlation,
)
from src.metrics.selective_prediction import (
    aggregate_case_uncertainty,
    compute_risk_coverage_curve,
    compute_aurc,
)

__all__ = [
    "compute_dice_coefficient",
    "compute_hausdorff95",
    "compute_segmentation_metrics",
    "compute_esce",
    "compute_brier_score",
    "compute_error_detection_auroc",
    "compute_uncertainty_error_correlation",
    "aggregate_case_uncertainty",
    "compute_risk_coverage_curve",
    "compute_aurc",
]
