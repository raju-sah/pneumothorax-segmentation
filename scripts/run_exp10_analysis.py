"""EXP-10: Robustness, Lesion Size Stratification, and Negative-Control OOD Evaluation.

Analyzes:
1. Negative-Control Rejection & Pathology Discrimination (AUROC).
2. False-Positive Rate / Clean Negatives on 1,659 Negative Controls.
3. Lesion Size Stratification (Small <2%, Medium 2-10%, Large >10%) vs. Case Uncertainty and DSC.
"""

import ast
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

def rle_area(rle_str_list):
    """Compute total positive pixel count from RLE strings without full image decode."""
    if isinstance(rle_str_list, str):
        try:
            rle_str_list = ast.literal_eval(rle_str_list)
        except Exception:
            rle_str_list = [rle_str_list]
    total_pixels = 0
    for s in rle_str_list:
        s = str(s).strip()
        if not s or s == "-1":
            continue
        parts = s.split()
        if len(parts) >= 2:
            lengths = [int(float(x)) for x in parts[1::2]]
            total_pixels += sum(lengths)
    return total_pixels

def run_exp10():
    print("Loading test holdout predictions and splits...")
    p10_df = pd.read_csv("results/p10_predictions.csv")
    exp07_df = pd.read_csv("results/exp07_predictions.csv")
    splits_df = pd.read_csv("data/processed/patient_splits.csv")
    
    # Merge splits info
    splits_te = splits_df[splits_df["Fold"].isin(["test", "Test", 4])].copy()
    m_dict = dict(zip(splits_df["ImageId"], splits_df["EncodedPixelsList"]))
    exp07_df["EncodedPixelsList"] = exp07_df["ImageId"].map(m_dict)
    
    # Compute lesion pixel areas on original 1024x1024 scale (1,048,576 total pixels)
    TOTAL_PX = 1024 * 1024
    exp07_df["lesion_px"] = exp07_df["EncodedPixelsList"].apply(rle_area)
    exp07_df["lesion_fraction"] = exp07_df["lesion_px"] / TOTAL_PX
    
    # Subgroup masks
    pos_mask = exp07_df["HasPneumothorax"] == 1
    neg_mask = exp07_df["HasPneumothorax"] == 0
    
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    print(f"Cohort: {len(exp07_df)} cases ({n_pos} positive, {n_neg} negative)")
    
    # --- 1. Negative-Control Rejection AUROC ---
    # Can case uncertainty discriminate between positive pathology and negative controls?
    auroc_det = roc_auc_score(exp07_df["HasPneumothorax"], exp07_df["det_unc"])
    auroc_ens = roc_auc_score(exp07_df["HasPneumothorax"], exp07_df["ens_unc"])
    # Also check p10 mc_unc
    auroc_mc = roc_auc_score(p10_df["HasPneumothorax"], p10_df["mc_unc"])
    
    print("\n=== 1. Negative Control / Pathology Discrimination (AUROC) ===")
    print(f"Deterministic Entropy AUROC: {auroc_det:.4f}")
    print(f"Deep Ensemble MI AUROC:     {auroc_ens:.4f}")
    print(f"MC Dropout Variance AUROC:  {auroc_mc:.4f}")
    
    # --- 2. False-Alarm Analysis on 1,659 Negative Controls ---
    # For true negatives, ground truth mask is empty.
    # If prediction is also cleanly empty, Dice = 1.0.
    # If model fires ANY false positive pixel, Dice < 1.0 (typically ~0.0).
    clean_neg_det_05 = float((exp07_df.loc[neg_mask, "det_dice_05"] == 1.0).mean())
    clean_neg_det_tuned = float((exp07_df.loc[neg_mask, "det_dice_tuned"] == 1.0).mean())
    clean_neg_ens_05 = float((exp07_df.loc[neg_mask, "ens_dice_05"] == 1.0).mean())
    clean_neg_ens_tuned = float((exp07_df.loc[neg_mask, "ens_dice_tuned"] == 1.0).mean())
    
    mean_unc_pos_det = float(exp07_df.loc[pos_mask, "det_unc"].mean())
    mean_unc_neg_det = float(exp07_df.loc[neg_mask, "det_unc"].mean())
    mean_unc_pos_ens = float(exp07_df.loc[pos_mask, "ens_unc"].mean())
    mean_unc_neg_ens = float(exp07_df.loc[neg_mask, "ens_unc"].mean())
    
    print("\n=== 2. Negative Control False Alarm & Specificity Audit ===")
    print(f"Clean Negatives (Dice=1.0) @ t=0.50: Det = {clean_neg_det_05*100:.1f}%, Ens = {clean_neg_ens_05*100:.1f}%")
    print(f"Clean Negatives (Dice=1.0) @ t=0.25: Det = {clean_neg_det_tuned*100:.1f}%, Ens = {clean_neg_ens_tuned*100:.1f}%")
    print(f"Det Case Uncertainty: Pos = {mean_unc_pos_det:.4f} vs Neg = {mean_unc_neg_det:.4f}")
    print(f"Ens Case Uncertainty: Pos = {mean_unc_pos_ens:.4f} vs Neg = {mean_unc_neg_ens:.4f}")
    
    # --- 3. Lesion Size Stratification on Positive Cases ---
    pos_df = exp07_df[pos_mask].copy()
    small_mask = pos_df["lesion_fraction"] < 0.02
    med_mask = (pos_df["lesion_fraction"] >= 0.02) & (pos_df["lesion_fraction"] < 0.10)
    large_mask = pos_df["lesion_fraction"] >= 0.10
    
    n_small = int(small_mask.sum())
    n_med = int(med_mask.sum())
    n_large = int(large_mask.sum())
    
    strata = {}
    for name, m in [("small (<2%)", small_mask), ("medium (2-10%)", med_mask), ("large (>10%)", large_mask)]:
        sub = pos_df[m]
        strata[name] = {
            "n": len(sub),
            "pct_of_pos": round(len(sub) / n_pos * 100, 1),
            "mean_lesion_pct": round(float(sub["lesion_fraction"].mean() * 100), 2),
            "det_dice_05": round(float(sub["det_dice_05"].mean()), 4),
            "det_dice_tuned": round(float(sub["det_dice_tuned"].mean()), 4),
            "det_gain_pct": round(float((sub["det_dice_tuned"].mean() - sub["det_dice_05"].mean()) / max(sub["det_dice_05"].mean(), 1e-6) * 100), 1),
            "ens_dice_05": round(float(sub["ens_dice_05"].mean()), 4),
            "ens_dice_tuned": round(float(sub["ens_dice_tuned"].mean()), 4),
            "ens_gain_pct": round(float((sub["ens_dice_tuned"].mean() - sub["ens_dice_05"].mean()) / max(sub["ens_dice_05"].mean(), 1e-6) * 100), 1),
            "det_unc_mean": round(float(sub["det_unc"].mean()), 4),
            "ens_unc_mean": round(float(sub["ens_unc"].mean()), 4),
        }
    
    # Spearman rank correlation between lesion size and uncertainty on positive cases
    spearman_det, p_det = spearmanr(pos_df["lesion_fraction"], pos_df["det_unc"])
    spearman_ens, p_ens = spearmanr(pos_df["lesion_fraction"], pos_df["ens_unc"])
    
    print("\n=== 3. Lesion Size Stratification ===")
    for name, s in strata.items():
        print(f"Strata {name} (N={s['n']}, {s['pct_of_pos']}%):")
        print(f"  Det DSC: {s['det_dice_05']:.4f} -> {s['det_dice_tuned']:.4f} ({s['det_gain_pct']:+.1f}%)")
        print(f"  Ens DSC: {s['ens_dice_05']:.4f} -> {s['ens_dice_tuned']:.4f} ({s['ens_gain_pct']:+.1f}%)")
        print(f"  Mean Uncertainty: det={s['det_unc_mean']:.4f}, ens={s['ens_unc_mean']:.4f}")
    
    print(f"\nSpearman Correlation (Lesion Size vs Det Uncertainty): rho = {spearman_det:.4f}, p = {p_det:.2e}")
    print(f"Spearman Correlation (Lesion Size vs Ens Uncertainty): rho = {spearman_ens:.4f}, p = {p_ens:.2e}")
    
    # Save full metrics to JSON
    exp10_results = {
        "negative_control_discrimination_auroc": {
            "deterministic_entropy": round(auroc_det, 4),
            "deep_ensemble_mutual_info": round(auroc_ens, 4),
            "mc_dropout_variance": round(auroc_mc, 4)
        },
        "negative_control_clean_rates": {
            "n_negative_controls": n_neg,
            "det_clean_rate_05": round(clean_neg_det_05, 4),
            "det_clean_rate_tuned_025": round(clean_neg_det_tuned, 4),
            "ens_clean_rate_05": round(clean_neg_ens_05, 4),
            "ens_clean_rate_tuned_025": round(clean_neg_ens_tuned, 4)
        },
        "mean_case_uncertainty": {
            "det_positive": round(mean_unc_pos_det, 4),
            "det_negative_control": round(mean_unc_neg_det, 4),
            "ens_positive": round(mean_unc_pos_ens, 4),
            "ens_negative_control": round(mean_unc_neg_ens, 4)
        },
        "lesion_size_stratification": strata,
        "lesion_size_vs_uncertainty_spearman": {
            "deterministic": {"rho": round(spearman_det, 4), "p_value": float(p_det)},
            "ensemble": {"rho": round(spearman_ens, 4), "p_value": float(p_ens)}
        }
    }
    
    with open("results/exp10_metrics.json", "w") as f:
        json.dump(exp10_results, f, indent=2)
    print("\nSaved results/exp10_metrics.json successfully!")

if __name__ == "__main__":
    run_exp10()
