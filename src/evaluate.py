"""Evaluation and selective prediction benchmark suite across all uncertainty paradigms."""

import argparse
import json
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch

from src.metrics.segmentation import compute_segmentation_metrics, compute_hausdorff95
from src.metrics.uncertainty import (
    compute_esce,
    compute_brier_score,
    compute_error_detection_auroc,
    compute_uncertainty_error_correlation,
)
from src.metrics.selective_prediction import (
    aggregate_case_uncertainty,
    evaluate_selective_prediction,
)


def evaluate_test_predictions(
    probs: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    method_name: str = "ensemble",
) -> Dict[str, Any]:
    """Execute complete benchmark suite across N test cases.

    Args:
        probs: Array of shape (N, H, W) containing predicted probabilities.
        uncertainties: Array of shape (N, H, W) containing spatial uncertainty maps.
        targets: Array of shape (N, H, W) containing ground-truth binary masks.
        output_dir: Directory to save evaluation tables and figures.
        method_name: Identifier for the method (e.g., 'deterministic', 'mc_dropout', 'ensemble').

    Returns:
        Dict of computed evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(probs)

    case_dices = []
    case_pos_dices = []
    case_ious = []
    case_sens = []
    case_specs = []
    case_hd95s = []
    case_uncertainties = []

    for i in range(n_samples):
        p = probs[i]
        u = uncertainties[i]
        t = targets[i]

        seg_m = compute_segmentation_metrics(t, p)
        hd95 = compute_hausdorff95(t, (p >= 0.5).astype(np.uint8))
        case_u = aggregate_case_uncertainty(u, method="top_k", k=500)

        case_dices.append(seg_m["dice"])
        if seg_m["is_positive"] == 1:
            case_pos_dices.append(seg_m["dice"])
        case_ious.append(seg_m["iou"])
        case_sens.append(seg_m["sensitivity"])
        case_specs.append(seg_m["specificity"])
        case_hd95s.append(hd95)
        case_uncertainties.append(case_u)

    case_dices_arr = np.array(case_dices)
    case_u_arr = np.array(case_uncertainties)

    # Global Calibration & Error Detection Metrics
    esce, bin_confs, bin_accs, bin_counts = compute_esce(probs, targets, n_bins=10)
    brier = compute_brier_score(probs, targets)
    auroc_ed = compute_error_detection_auroc(uncertainties, probs, targets)
    spearman_rho = compute_uncertainty_error_correlation(uncertainties, probs, targets)

    # Selective Prediction
    sel_pred = evaluate_selective_prediction(case_u_arr, case_dices_arr)

    results = {
        "method": method_name,
        "overall_dice_mean": float(np.mean(case_dices_arr)),
        "pos_dice_mean": float(np.mean(case_pos_dices)) if len(case_pos_dices) > 0 else 0.0,
        "iou_mean": float(np.mean(case_ious)),
        "sensitivity_mean": float(np.mean(case_sens)),
        "specificity_mean": float(np.mean(case_specs)),
        "hd95_mean": float(np.mean(case_hd95s)),
        "esce": esce,
        "brier_score": brier,
        "auroc_error_detection": auroc_ed,
        "spearman_rho": spearman_rho,
        **sel_pred,
    }

    # Save to JSON
    json_path = os.path.join(output_dir, f"metrics_{method_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results successfully saved to {json_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Uncertainty Predictions")
    parser.add_argument("--method", type=str, default="ensemble", help="Method identifier")
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory")
    args = parser.parse_args()
    print("Evaluation module loaded.")
