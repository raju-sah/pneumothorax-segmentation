"""P1 honest recompute from results/test_predictions.csv (case-level only).

Seeded (rng 42). No pixel data available locally, so ESCE/Brier/global
AUROC-ED cannot be recomputed here — flagged, not fabricated.
Outputs results/p1_bootstrap.json + prints markdown rows.
"""
import csv
import json

import numpy as np
from scipy.stats import wilcoxon

RNG = np.random.default_rng(42)
B = 1000
METHODS = ["det", "mc", "ens"]
Q = [2.5, 97.5]


def load():
    with open("results/test_predictions.csv") as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return np.array([float(r[key]) for r in rows])


def boot_mean_ci(vals, n_boot=B):
    n = len(vals)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), [float(np.percentile(means, q)) for q in Q]


def aurc_standard(dice, unc):
    """Standard AURC: sort by unc asc, risk(c)=1-mean dice of top-c fraction,
    trapezoid over coverage in [0,1]. Empirical coverages k/n (no rescale)."""
    order = np.argsort(unc, kind="stable")
    d = dice[order]
    n = len(d)
    cum = np.cumsum(d) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    risk = 1.0 - cum
    return float(np.trapezoid(risk, cov))


def aurc_oracle(dice):
    d = np.sort(dice)[::-1]
    n = len(d)
    cum = np.cumsum(d) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    return float(np.trapezoid(1.0 - cum, cov))


def retained(dice, unc, cov):
    k = max(1, int(round(len(dice) * cov)))
    order = np.argsort(unc, kind="stable")
    return float(dice[order][:k].mean())


def main():
    rows = load()
    pos = [r for r in rows if float(r["HasPneumothorax"]) > 0]
    out = {"n": len(rows), "n_pos": len(pos), "seed": 42, "bootstraps": B,
           "note": "ESCE/Brier/global-AUROC-ED need pixel probs; not recomputed."}
    for m in METHODS:
        d_all = col(rows, f"{m}_dice")
        d_pos = col(pos, f"{m}_dice")
        iou = col(pos, f"{m}_iou")
        sens = col(pos, f"{m}_sens")
        spec = col(rows, f"{m}_spec")
        unc = col(rows, f"{m}_case_unc")
        a = aurc_standard(d_all, unc)
        ao = aurc_oracle(d_all)
        entry = {}
        for name, v in [("dsc_all", d_all), ("dsc_pos", d_pos),
                        ("iou_pos", iou), ("sens", sens), ("spec", spec)]:
            mean, ci = boot_mean_ci(v)
            entry[name] = {"mean": round(mean, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)]}
        entry["aurc_standard"] = round(a, 4)
        entry["aurc_oracle"] = round(ao, 4)
        entry["e_aurc"] = round(a - ao, 4)
        entry["retained"] = {f"cov_{int(c*100)}": round(retained(d_all, unc, c), 4)
                             for c in (1.0, 0.9, 0.8, 0.7)}
        out[m] = entry
    # paired tests on per-case dice (all cases), ens vs det / mc vs det
    dd = col(rows, "det_dice")
    for m in ("mc", "ens"):
        w = wilcoxon(col(rows, f"{m}_dice"), dd, alternative="two-sided")
        out[f"wilcoxon_{m}_vs_det_dsc_all"] = {"stat": float(w.statistic), "p": float(w.pvalue)}
    # paired delta CIs via bootstrap (same resample index for both)
    idx = RNG.integers(0, len(rows), size=(B, len(rows)))
    for m in ("mc", "ens"):
        deltas = col(rows, f"{m}_dice")[idx].mean(axis=1) - dd[idx].mean(axis=1)
        out[f"delta_{m}_minus_det_dsc_all"] = {
            "mean": round(float(deltas.mean()), 4),
            "ci95": [round(float(np.percentile(deltas, Q[0])), 4),
                     round(float(np.percentile(deltas, Q[1])), 4)]}
    with open("results/p1_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
