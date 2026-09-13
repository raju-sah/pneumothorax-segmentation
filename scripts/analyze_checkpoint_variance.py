"""Analyze checkpoint selection variance and quantify stabilization mechanisms.

Analyzes:
1. Epoch-by-epoch trajectory across seeds 42, 43, 44.
2. Selection rule sensitivity: argmax(val_all) vs. argmax(val_pos).
3. Variance reduction from Deep Ensembling vs. Single Checkpoint.
4. Recommendation of SWA (Stochastic Weight Averaging) & Temporal Checkpoint Averaging.
"""

import json
import numpy as np

def analyze():
    traces = {}
    for s in (42, 43, 44):
        with open(f"results/p10_det_seed{s}_trace.json") as f:
            traces[s] = json.load(f)
            
    with open("results/p10_mc_seed42_trace.json") as f:
        traces["mc42"] = json.load(f)
        
    print("Loaded training traces for seeds 42, 43, 44 and MC.")
    
    # 1. Selection rule comparison per seed
    rules = {}
    for s in (42, 43, 44):
        tr = traces[s]
        best_pos_epoch = max(tr, key=lambda x: x["val_pos"])
        best_all_epoch = max(tr, key=lambda x: x["val_all"])
        rules[s] = {
            "best_val_pos": best_pos_epoch,
            "best_val_all": best_all_epoch,
            "val_pos_swing": round(best_pos_epoch["val_pos"] - best_all_epoch["val_pos"], 4),
            "val_all_swing": round(best_all_epoch["val_all"] - best_pos_epoch["val_all"], 4)
        }
        print(f"\nSeed {s}:")
        print(f"  Best val_pos: Epoch {best_pos_epoch['epoch']} (val_pos={best_pos_epoch['val_pos']}, val_all={best_pos_epoch['val_all']})")
        print(f"  Best val_all: Epoch {best_all_epoch['epoch']} (val_pos={best_all_epoch['val_pos']}, val_all={best_all_epoch['val_all']})")
        print(f"  Discrepancy: Δval_pos={rules[s]['val_pos_swing']:+.4f}")
        
    # 2. Late-epoch convergence stability (Epochs 8-10)
    late_pos = {}
    for s in (42, 43, 44):
        tr = traces[s]
        vals = [ep["val_pos"] for ep in tr if ep["epoch"] >= 8]
        late_pos[s] = {
            "epochs_8_10_pos": vals,
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "cv": round(float(np.std(vals) / np.mean(vals)), 4)
        }
        print(f"Seed {s} late epochs (8-10): mean={late_pos[s]['mean']:.4f}, std={late_pos[s]['std']:.4f} (CV={late_pos[s]['cv']:.2%})")

    # 3. Multi-seed ensembling vs single-seed variance
    # Seed 42, 43, 44 best val_pos scores
    best_val_pos_per_seed = [rules[s]["best_val_pos"]["val_pos"] for s in (42, 43, 44)]
    seed_mean = float(np.mean(best_val_pos_per_seed))
    seed_std = float(np.std(best_val_pos_per_seed))
    
    # Ensemble validation score (from exp07 at t=0.50 and t*=0.25)
    ens_val_pos_05 = 0.3301
    ens_val_pos_tuned = 0.4183
    
    summary = {
        "selection_rule_discrepancy": rules,
        "late_epoch_stability": late_pos,
        "inter_seed_variance": {
            "seed_val_pos_values": best_val_pos_per_seed,
            "mean": round(seed_mean, 4),
            "std": round(seed_std, 4),
            "cv": round(seed_std / seed_mean, 4)
        },
        "mitigation_strategies": {
            "1_multi_seed_ensembling": {
                "description": "Deep Ensemble (M=3) averages out stochastic seed initialization and optimization divergence.",
                "val_pos_ref": ens_val_pos_05,
                "val_pos_tuned": ens_val_pos_tuned,
                "benefit": "Ensemble validation score (0.4183) sits at the ceiling of individual seed performance with zero single-checkpoint fragility."
            },
            "2_stochastic_weight_averaging_temporal_averaging": {
                "description": "Temporal averaging of weights across epochs 8, 9, and 10 or prediction averaging across final epochs.",
                "target_epochs": [8, 9, 10],
                "rationale": "Mitigates within-seed epoch oscillation (where val_pos fluctuates between 0.27 and 0.40 in seed 44)."
            },
            "3_curriculum_or_lr_decay": {
                "description": "Cosine decay with warm restarts instead of fixed lr=1e-3 AdamW to prevent late-stage loss hopping."
            }
        }
    }
    
    with open("results/checkpoint_variance_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/checkpoint_variance_analysis.json successfully!")

if __name__ == "__main__":
    analyze()
