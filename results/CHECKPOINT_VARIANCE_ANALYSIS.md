# Checkpoint Selection Variance & Temporal Ensembling Analysis

**Context:** In the P9 and P10 training runs, the choice of validation checkpoint selection rule produced a dramatic divergence in test overlap metrics (the same seed 42 scored test $\text{DSC}_{\text{pos}} = 0.6071$ under best-val-$\text{DSC}_{\text{all}}$ selection in P9, but $0.2803$ under verbatim best-val-$\text{DSC}_{\text{pos}}$ selection in P10).

This analysis dissects the epoch-by-epoch training traces across seeds 42, 43, and 44, models the underlying optimization dynamics, and evaluates multi-checkpoint stabilization strategies.

---

## 1. Selection Rule Pathology: The $\text{val\_all}$ Trap

Because $\sim 78\%$ of radiographs are true negatives without pneumothorax, an untrained model predicting completely empty masks receives $\text{DSC} = 1.0$ on $78\%$ of images, achieving $\text{DSC}_{\text{all}} \approx 0.7763$.

| Model Seed | Best $\text{val\_pos}$ Checkpoint | $\text{val\_pos}$ | $\text{val\_all}$ | Best $\text{val\_all}$ Checkpoint | $\text{val\_pos}$ | $\text{val\_all}$ | Selection Discrepancy ($\Delta \text{val\_pos}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | **Epoch 10** | **0.2675** | 0.1090 | Epoch 1 | 0.0000 | 0.7763 | **+0.2675** |
| **Seed 43** | **Epoch 10** | **0.2888** | 0.4042 | Epoch 2 | 0.0000 | 0.5509 | **+0.2888** |
| **Seed 44** | **Epoch 9** | **0.3984** | 0.2677 | Epoch 1 | 0.0000 | 0.7763 | **+0.3984** |

### Key Finding:
Across all three independent seeds, selecting checkpoints via `argmax(val_all)` systematically picks **Epoch 1 or 2**, where the network has learned nothing about pneumothorax and simply outputs an empty mask. Selecting by `argmax(val_pos)` correctly forces the network to pick late-stage trained weights (Epochs 9–10), but exposes the run to late-stage epoch oscillation.

---

## 2. Late-Epoch Intra-Seed Oscillation (Epochs 8–10)

Fixed-learning-rate AdamW ($\text{lr} = 10^{-3}$) induces substantial epoch-to-epoch oscillation around non-convex minima:

| Seed | Epoch 8 $\text{val\_pos}$ | Epoch 9 $\text{val\_pos}$ | Epoch 10 $\text{val\_pos}$ | 3-Epoch Mean | Standard Deviation | Coefficient of Variation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 0.2560 | 0.2452 | 0.2675 | 0.2562 | $\pm 0.0091$ | **3.5%** |
| **Seed 43** | 0.2165 | 0.2210 | 0.2888 | 0.2421 | $\pm 0.0331$ | **13.7%** |
| **Seed 44** | 0.2905 | 0.3984 | 0.2762 | 0.3217 | $\pm 0.0545$ | **17.0%** |

In Seed 44, validation performance swings by over **$0.12$ Dice** between Epoch 9 and Epoch 10. A single checkpoint selection from this noisy trajectory is fragile.

---

## 3. Stabilization & Mitigation Architecture

To completely eliminate single-checkpoint fragility in future clinical pipelines:

1. **Multi-Seed Deep Ensembling (Implemented in EXP-05 & EXP-07):**
   - Combining predictions from multiple independent seeds ($M=3$) averages out stochastic initialization and trajectory variance.
   - Ensembling pushes calibrated validation $\text{DSC}_{\text{pos}}$ to **$0.4183$** and test sensitivity to **$35.4\%$**, matching or exceeding the best single seed with zero single-checkpoint exposure.

2. **Stochastic Weight Averaging (SWA) / Temporal Checkpoint Averaging:**
   - Rather than picking an arbitrary peak epoch $e^* \in \{8, 9, 10\}$, average model weights across the final 3 epochs:
     $$\bar{\theta} = \frac{1}{K} \sum_{k=8}^{10} \theta_k$$
   - This smooths out the non-convex loss landscape, dampening the $17\%$ coefficient of variation observed in late-stage training.

3. **Curriculum Learning Rate Schedules:**
   - Replacing fixed $\text{lr} = 10^{-3}$ with a Cosine Annealing schedule with warm restarts ($\text{lr}_{\text{min}} = 10^{-6}$) ensures weights settle into broader, flatter local minima before checkpoint evaluation.

---

## 4. Artifacts Produced
- Analysis script: `scripts/analyze_checkpoint_variance.py`
- Trace metric data: `results/checkpoint_variance_analysis.json`
