"""Regen Fig1 (standard risk-coverage) + risk_coverage_curves.csv + app.js curve values.

Reads results/test_predictions.csv only. Saves:
- results/figures/fig1_risk_coverage_curves.png
- docs/assets/fig1_risk_coverage_curves.png
- results/risk_coverage_curves.csv (standard empirical grid)
Prints JS risk arrays for docs/js/app.js.
"""
import csv

import numpy as np

METHODS = [("det", "Deterministic", "#f87171"),
           ("mc", "MC Dropout", "#fbbf24"),
           ("ens", "Deep Ensemble", "#38bdf8")]
COLORS = {"det": "#e74c3c", "mc": "#f39c12", "ens": "#2980b9"}


def main():
    with open("results/test_predictions.csv") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    data = {}
    for m, _, _ in METHODS:
        dice = np.array([float(r[f"{m}_dice"]) for r in rows])
        unc = np.array([float(r[f"{m}_case_unc"]) for r in rows])
        order = np.argsort(unc, kind="stable")
        d = dice[order]
        cum = np.cumsum(d) / np.arange(1, n + 1)
        cov = np.arange(1, n + 1) / n
        data[m] = (cov, 1.0 - cum)

    # standard CSV (empirical grid, decimated to 200 pts)
    idx = np.linspace(0, n - 1, 200).astype(int)
    with open("results/risk_coverage_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Coverage", "Risk_Random", "Risk_Deterministic",
                    "Risk_MCDropout", "Risk_DeepEnsemble"])
        base = 1.0 - np.array([float(r["det_dice"]) for r in rows]).mean()
        for i in idx:
            w.writerow([round(float(data["det"][0][i]), 6), round(base, 6)] +
                       [round(float(data[m][1][i]), 6) for m, _, _ in METHODS])

    # risk at fixed coverages for app.js
    print("=== app.js arrays (coverages 0.5..1.0) ===")
    covs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for m, label, _ in METHODS:
        cov, risk = data[m]
        vals = [float(np.interp(c, cov, risk)) for c in covs]
        print(f"{label}: [{', '.join(f'{v:.3f}' for v in vals)}]")

    # Fig1 PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    for m, label, _ in METHODS:
        cov, risk = data[m]
        a = float(np.trapezoid(risk, cov))
        ax.plot(cov, risk, color=COLORS[m], linewidth=2.0,
                label=f"{label} (AURC = {a:.4f})")
    ax.plot([0, 1], [base, base], "k--", linewidth=1.2,
            label=f"Random (AURC = {base:.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Autonomous Retention Coverage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Empirical Risk (1 - DSC_all)", fontsize=12, fontweight="bold")
    ax.set_title("Risk-Coverage Curves, standard def (N=2135)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    plt.tight_layout()
    for p in ("results/figures/fig1_risk_coverage_curves.png",
              "docs/assets/fig1_risk_coverage_curves.png"):
        fig.savefig(p, dpi=300)
    plt.close()
    print("Saved fig1 (both copies) + risk_coverage_curves.csv")


if __name__ == "__main__":
    main()
