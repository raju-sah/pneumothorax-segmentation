"""Utility script to fetch Kaggle kernel outputs, verify artifacts, and generate report tables."""

import os
import sys
import json
import shutil
import pandas as pd
import numpy as np

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

def fetch_kaggle_outputs():
    print("Fetching outputs from Kaggle...")
    cmd = f".venv/bin/kaggle kernels output rajucode/uncertainty-pneumothorax-segmentation -p {RESULTS_DIR}"
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error fetching kernel outputs (exit code {ret}). Kernel might still be running.")
        return False
    print("Successfully downloaded kernel artifacts.")
    return True

def organize_and_format():
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    if not os.path.exists(metrics_path):
        print(f"Metrics summary not found at {metrics_path}.")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print("Loaded metrics summary successfully.")

    # Organize figures
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".png"):
            src = os.path.join(RESULTS_DIR, f)
            dst = os.path.join(FIGURES_DIR, f)
            shutil.copy(src, dst)
            print(f"Copied figure: {f} -> {FIGURES_DIR}")

    # Generate Markdown and LaTeX tables
    det = metrics["deterministic_exp03"]
    mc = metrics["mc_dropout_exp04"]
    ens = metrics["deep_ensemble_exp05"]
    triage = metrics.get("selective_prediction_triage", {})

    table_1_md = f"""### Table 1: Disaggregated Test Holdout Segmentation & Uncertainty Performance (N={metrics['dataset']['test_images']})

| Metric | Deterministic Baseline (EXP-03) | MC Dropout ($T=20$) (EXP-04) | Deep Ensemble ($M=3$) (EXP-05) |
| :--- | :---: | :---: | :---: |
| **DSC_pos** (Positive Cases) | {det['dsc_pos_mean']:.4f} ± {det['dsc_pos_std']:.4f} | {mc['dsc_pos_mean']:.4f} ± {mc['dsc_pos_std']:.4f} | **{ens['dsc_pos_mean']:.4f} ± {ens['dsc_pos_std']:.4f}** |
| **DSC_all** (Overall Cohort) | {det['dsc_all_mean']:.4f} | {mc['dsc_all_mean']:.4f} | **{ens['dsc_all_mean']:.4f}** |
| **IoU** (Jaccard Index) | {det['iou_pos_mean']:.4f} | {mc['iou_pos_mean']:.4f} | **{ens['iou_pos_mean']:.4f}** |
| **Sensitivity** (Recall) | {det['sensitivity']:.4f} | {mc['sensitivity']:.4f} | **{ens['sensitivity']:.4f}** |
| **Specificity** | {det['specificity']:.4f} | {mc['specificity']:.4f} | **{ens['specificity']:.4f}** |
| **AUROC-ED** (Error Detection) | {det['auroc_ed']:.4f} | {mc['auroc_ed']:.4f} | **{ens['auroc_ed']:.4f}** |
| **ESCE** (Calibration Error) $\\downarrow$ | {det['esce']:.4f} | {mc['esce']:.4f} | **{ens['esce']:.4f}** |
| **Brier Score** $\\downarrow$ | {det['brier']:.4f} | {mc['brier']:.4f} | **{ens['brier']:.4f}** |
| **AURC** (Risk-Coverage) $\\downarrow$ | {det['aurc']:.4f} | {mc['aurc']:.4f} | **{ens['aurc']:.4f}** |
"""

    with open(os.path.join(TABLES_DIR, "table1_main_results.md"), "w") as f:
        f.write(table_1_md)

    print("Generated Table 1 Markdown.")

    # Selective Prediction Table
    if triage:
        table_2_md = f"""### Table 2: Clinical Selective Prediction & Human Referral Simulation

| Retention / Coverage | Random Referral | Deterministic Entropy | MC Dropout Variance | Deep Ensemble MutInfo |
| :---: | :---: | :---: | :---: | :---: |
| **100% Coverage** (No referral) | {triage['dice_at_100_cov']['det']:.4f} | {triage['dice_at_100_cov']['det']:.4f} | {triage['dice_at_100_cov']['mc']:.4f} | **{triage['dice_at_100_cov']['ens']:.4f}** |
| **90% Coverage** (10% referred) | — | {triage['dice_at_90_cov']['det']:.4f} | {triage['dice_at_90_cov']['mc']:.4f} | **{triage['dice_at_90_cov']['ens']:.4f}** |
| **80% Coverage** (20% referred) | — | {triage['dice_at_80_cov']['det']:.4f} | {triage['dice_at_80_cov']['mc']:.4f} | **{triage['dice_at_80_cov']['ens']:.4f}** |
| **70% Coverage** (30% referred) | — | {triage['dice_at_70_cov']['det']:.4f} | {triage['dice_at_70_cov']['mc']:.4f} | **{triage['dice_at_70_cov']['ens']:.4f}** |
| **AURC** $\\downarrow$ | {triage['aurc_random']:.4f} | {triage['aurc_det']:.4f} | {triage['aurc_mc']:.4f} | **{triage['aurc_ens']:.4f}** |
"""
        with open(os.path.join(TABLES_DIR, "table2_selective_prediction.md"), "w") as f:
            f.write(table_2_md)
        print("Generated Table 2 Markdown.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--format-only":
        organize_and_format()
    else:
        success = fetch_kaggle_outputs()
        if success:
            organize_and_format()
