"""Build EXP-07: temperature scaling + threshold tuning on P10 verbatim checkpoints.

No retraining. Loads det_seed{42,43,44}.pt from the p10-verbatim-checkpoints
dataset, fits per-branch temperature T on VAL (BCE, LBFGS, pixel stride 64),
sweeps the operating threshold on VAL (max val DSC_pos, verbatim spirit),
then evaluates on TEST at tuned point (+ 0.5 reference).

Branches: det (seed42), ens (mean of 3 member probs, P10 def). MC skipped:
degenerate in P10 (near-empty, var ~3e-6); tuning its threshold meaningless.
Uncertainty defs identical to P10 (unscaled probs) so AURC stays comparable.

Outputs: exp07_predictions.csv, exp07_metrics.json, exp07_threshold_curves.png
"""
import nbformat as nbf

SETUP = r'''
import os, subprocess
_q = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True)
_sm = _q.stdout.strip().split(".")
SM = (int(_sm[0]), int(_sm[1])) if len(_sm) == 2 and _sm[0].strip().isdigit() else (9, 0)
print("GPU SM:", SM)
if SM < (7, 0):
    subprocess.run(["pip", "install", "-q", "torch==2.3.1+cu118", "torchvision==0.18.1+cu118",
                    "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
os.system("pip install -q pydicom albumentations pretrainedmodels efficientnet_pytorch tqdm munch scikit-learn")
os.system("pip install -q --no-deps segmentation-models-pytorch")
import sys, time, math, glob, ast, shutil, json, random
import numpy as np, pandas as pd, pydicom, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
print("torch:", torch.__version__, "| smp:", smp.__version__, "| alb:", A.__version__)
device = torch.device("cuda")
seed_everything = lambda s=42: (random.seed(s), np.random.seed(s), torch.manual_seed(s),
                                torch.cuda.manual_seed_all(s))
'''

SPLITS = """
df_splits = pd.read_csv(glob.glob("/kaggle/input/**/patient_splits.csv", recursive=True)[0])
dmap0 = {}
for p in glob.glob("/kaggle/input/**/dicom-images-train/**/*.dcm", recursive=True):
    dmap0.setdefault(os.path.basename(p).replace(".dcm", ""), p)
df_splits["dcm_path"] = df_splits["ImageId"].map(dmap0)
print(len(df_splits), "missing:", int(df_splits["dcm_path"].isna().sum()))
"""

EXP07 = r'''
# ---- models (arch identical to P10) ----
class DetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    def forward(self, x): return self.model(x)

_pt_cands = glob.glob("/kaggle/input/**/*.pt", recursive=True)
print("Discovered .pt files:", _pt_cands)
wmap = {os.path.basename(p).replace(".pt", ""): p for p in _pt_cands}
print("Weights map keys:", sorted(wmap.keys()))
if not ({"det_seed42", "det_seed43", "det_seed44"} <= set(wmap)):
    print("ALL items in /kaggle/input:", glob.glob("/kaggle/input/**", recursive=True)[:100])
assert {"det_seed42", "det_seed43", "det_seed44"} <= set(wmap), "P10 weights missing"
def load_det(seed):
    m = DetNet().to(device)
    m.load_state_dict(torch.load(wmap[f"det_seed{seed}"], map_location=device)); m.eval()
    return m
nets = {s: load_det(s) for s in (42, 43, 44)}

F_ = lambda v: df_splits["Fold"].isin([v, int(v)])
va_df = df_splits[F_("0")].reset_index(drop=True)
te_df = df_splits[df_splits["Fold"].isin(["test", "Test"])].reset_index(drop=True)
print(len(va_df), len(te_df))
assert set(va_df["PatientID"]).isdisjoint(set(te_df["PatientID"]))
va_ld = DataLoader(SIIMPneumothoraxDataset(va_df, image_size=512, transforms=val_transforms),
                   batch_size=1, shuffle=False, num_workers=2)
te_ld = DataLoader(SIIMPneumothoraxDataset(te_df, image_size=512, transforms=val_transforms),
                   batch_size=1, shuffle=False, num_workers=2)

NV, S = len(va_df), 512
# cache val masks once (uint8)
vmask = np.memmap("/tmp/val_masks.dat", dtype=np.uint8, mode="w+", shape=(NV, S, S))
for j, b in enumerate(va_ld):
    vmask[j] = (b["mask"].numpy()[0, 0] > 0).astype(np.uint8)
    if (j + 1) % 400 == 0: print(f"masks {j+1}/{NV}", flush=True)
vmask.flush()
vpos = np.array([m.sum() > 0 for m in vmask])
print("val pos:", int(vpos.sum()))

def dice_at(p, y, t):
    pb = (p >= t)
    if y.sum() == 0 and pb.sum() == 0: return 1.0
    if y.sum() == 0 or pb.sum() == 0: return 0.0
    return float(2 * np.logical_and(y > 0, pb).sum() / (y.sum() + pb.sum()))

def stream_val_logits(member_seeds):
    """Single-seed val logits -> f16 memmap (reused file)."""
    assert len(member_seeds) == 1
    s = member_seeds[0]; net = nets[s]
    lg = np.memmap("/tmp/val_logits.dat", dtype=np.float16, mode="w+", shape=(NV, S, S))
    with torch.no_grad():
        for j, b in enumerate(va_ld):
            pr = torch.sigmoid(net(b["image"].to(device)))[0, 0].cpu().numpy().astype(np.float32)
            lg[j] = np.log(pr / np.clip(1 - pr, 1e-7, 1)).astype(np.float16)
            if (j + 1) % 400 == 0: print(f"val {s} {j+1}/{NV}", flush=True)
    lg.flush()
    return lg

def stream_val_meanprob():
    lg = np.memmap("/tmp/val_logits.dat", dtype=np.float16, mode="w+", shape=(NV, S, S))
    mean = np.zeros((NV, S, S), dtype=np.float32)
    for k, s in enumerate((42, 43, 44)):
        net = nets[s]
        with torch.no_grad():
            for j, b in enumerate(va_ld):
                pr = torch.sigmoid(net(b["image"].to(device)))[0, 0].cpu().numpy().astype(np.float32)
                mean[j] += pr / 3.0
                if (j + 1) % 400 == 0: print(f"val ens {s} {j+1}/{NV}", flush=True)
    eps = 1e-7
    lg[:] = np.log(np.clip(mean, eps, 1 - eps) / np.clip(1 - mean, eps, 1 - eps)).astype(np.float16)
    lg.flush()
    return lg

def fit_temperature(lg):
    z = lg[:].reshape(-1).astype(np.float32)[::64]
    y = vmask[:].reshape(-1).astype(np.float32)[::64]
    zt = torch.from_numpy(z); yt = torch.from_numpy(y)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], max_iter=25, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(zt / torch.exp(logT), yt)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(torch.exp(logT).item())
    print("T =", round(T, 4), flush=True)
    return T

def sweep_threshold(lg, T, tag):
    P = 1 / (1 + np.exp(-lg[:].astype(np.float32) / T))  # (NV,512,512) f32 = 1.8GB transient
    ths = [round(v, 2) for v in np.arange(0.05, 0.91, 0.05)]
    scored = {}
    for t in ths:
        dp = np.array([dice_at(P[j], vmask[j], t) for j in range(NV)])
        scored[t] = (float(dp[vpos].mean()), float(dp.mean()))
        print(f"{tag} t={t:.2f} val_pos={scored[t][0]:.4f} val_all={scored[t][1]:.4f}", flush=True)
    t0 = max(ths, key=lambda t: scored[t][0])
    for t in [round(t0 + d, 2) for d in (-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04) if 0.02 < t0 + d < 0.98]:
        dp = np.array([dice_at(P[j], vmask[j], t) for j in range(NV)])
        scored[t] = (float(dp[vpos].mean()), float(dp.mean()))
    best = max(scored, key=lambda t: scored[t][0])
    print(tag, "BEST t =", best, scored[best], flush=True)
    del P
    return best, scored

tuned = {}
curves = {}
# det branch
lg = stream_val_logits((42,))
T_det = fit_temperature(lg)
t_det, curves["det"] = sweep_threshold(lg, T_det, "det")
tuned["det"] = {"T": T_det, "t": t_det}
# ens branch
lg = stream_val_meanprob()
T_ens = fit_temperature(lg)
t_ens, curves["ens"] = sweep_threshold(lg, T_ens, "ens")
tuned["ens"] = {"T": T_ens, "t": t_ens}
print(json.dumps(tuned))

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
for tag, col in (("det", "#e74c3c"), ("ens", "#2980b9")):
    xs = sorted(curves[tag]); ax.plot(xs, [curves[tag][t][0] for t in xs], color=col, label=f"{tag} val DSC_pos")
    ax.plot(xs, [curves[tag][t][1] for t in xs], color=col, linestyle="--", label=f"{tag} val DSC_all")
    ax.axvline(tuned[tag]["t"], color=col, linestyle=":")
ax.set_xlabel("Threshold", fontweight="bold"); ax.set_ylabel("Val Dice", fontweight="bold")
ax.set_title("EXP-07 val threshold sweep (star = selected)", fontweight="bold"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("exp07_threshold_curves.png", dpi=300); plt.close()

# ---- test @ tuned + @ 0.5 ----
def entropy(p): return -(p * np.log2(p + 1e-8) + (1 - p) * np.log2(1 - p + 1e-8))
def topk(a, k=500):
    f = a.ravel(); return float(f[np.argpartition(f, -k)[-k:]].mean())
N = len(te_df); B = 10
acc = {k: {"n": 0, "bn": np.zeros(B), "bp": np.zeros(B), "by": np.zeros(B), "br": 0.0}
       for k in ("det", "det05", "ens", "ens05")}
conf = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in ("det", "det05", "ens", "ens05")}
def acc_up(a, p, y):
    b = np.clip((p * B).astype(int), 0, B - 1); a["n"] += y.size
    for i in range(B):
        m = b == i
        if m.any(): a["bn"][i] += m.sum(); a["bp"][i] += p[m].sum(); a["by"][i] += y[m].sum()
    a["br"] += float(((p - y) ** 2).sum())
def conf_up(c, pred, y):
    c["tp"] += int(np.logical_and(pred == 1, y == 1).sum()); c["fp"] += int(np.logical_and(pred == 1, y == 0).sum())
    c["fn"] += int(np.logical_and(pred == 0, y == 1).sum()); c["tn"] += int(np.logical_and(pred == 0, y == 0).sum())

fout = open("exp07_predictions.csv", "w")
fout.write("ImageId,PatientID,ViewPosition,HasPneumothorax,det_dice_tuned,det_dice_05,ens_dice_tuned,ens_dice_05,det_unc,ens_unc\n")
t0 = time.time()
for j, b in enumerate(te_ld):
    x = b["image"].to(device); yn = b["mask"].numpy()[0, 0]
    yb = (yn > 0).astype(np.uint8)
    with torch.no_grad():
        ps = [torch.sigmoid(nets[s](x))[0, 0].cpu().numpy() for s in (42, 43, 44)]
        em = np.stack(ps).mean(0)
        ee = entropy(em); me = np.stack([entropy(p) for p in ps]).mean(0)
        mi = np.clip(ee - me, 0, None); ed = entropy(ps[0])
    row = [te_df.iloc[j]["ImageId"], te_df.iloc[j]["PatientID"], te_df.iloc[j]["ViewPosition"],
           te_df.iloc[j]["HasPneumothorax"]]
    for key, prob, T, t in (("det", ps[0], T_det, t_det), ("ens", em, T_ens, t_ens)):
        eps = 1e-7
        pc = np.clip(prob, eps, 1 - eps)
        ps_ = 1 / (1 + np.exp(-np.log(pc / (1 - pc)) / T))
        for suf, tt in (("", t), ("05", 0.5)):
            kk = key if suf == "" else key + "05"
            acc_up(acc[kk], ps_, yb.astype(np.float32))
            conf_up(conf[kk], (ps_ >= tt).astype(np.uint8), yb)
        row.append(compute_segmentation_metrics(yb, ps_, threshold=t)["dice"])
        row.append(compute_segmentation_metrics(yb, ps_, threshold=0.5)["dice"])
    # row order: det_tuned, det_05, ens_tuned, ens_05
    fout.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]},{topk(ed)},{topk(mi)}\n")
    if (j + 1) % 400 == 0:
        fout.flush(); print(f"{j+1}/{N} ({(time.time()-t0)/60:.1f}m)", flush=True)
fout.close()

def esce_of(a):
    t = a["bn"].sum()
    return float((a["bn"] / t * np.abs(a["bp"] / np.maximum(a["bn"], 1) - a["by"] / np.maximum(a["bn"], 1))).sum())
OUT = {"tuned": tuned,
       "esce": {k: round(esce_of(acc[k]), 6) for k in acc},
       "brier": {k: round(acc[k]["br"] / acc[k]["n"], 6) for k in acc}}
for k in conf:
    c = conf[k]; tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    OUT[k + "_pix"] = {"iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0,
                       "sens": round(tp / (tp + fn), 4) if tp + fn else 1.0,
                       "spec": round(tn / (tn + fp), 4) if tn + fp else 1.0}
df = pd.read_csv("exp07_predictions.csv")
for c in ["det_dice_tuned", "det_dice_05", "ens_dice_tuned", "ens_dice_05"]:
    OUT[c + "_mean"] = round(float(df[c].mean()), 4)
    OUT[c + "_posmean"] = round(float(df[df["HasPneumothorax"] == 1][c].mean()), 4)
rng = np.random.default_rng(42); n = len(df)
for c in ["det_dice_tuned", "ens_dice_tuned"]:
    v = df[c].values; bs = v[rng.integers(0, n, (1000, n))].mean(1)
    OUT[c + "_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
OUT["wilcoxon_tuned"] = float(wilcoxon(df["ens_dice_tuned"], df["det_dice_tuned"], alternative="two-sided").pvalue)
for m, u in (("det_dice_tuned", "det_unc"), ("ens_dice_tuned", "ens_unc")):
    o = np.argsort(df[u].values, kind="stable"); d = df[m].values[o]
    cum = np.cumsum(d) / np.arange(1, n + 1); cov = np.arange(1, n + 1) / n
    OUT[m + "_aurc"] = round(float(np.trapezoid(1 - cum, cov)), 4)
    oo = np.argsort(-d, kind="stable"); dd = d[oo]
    cum_o = np.cumsum(dd) / np.arange(1, n + 1)
    OUT[m + "_eaurc"] = round(OUT[m + "_aurc"] - float(np.trapezoid(1 - cum_o, cov)), 4)
with open("exp07_metrics.json", "w") as f: json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
print("saved all")
'''

VERBATIM_CELLS = (3, 4, 5, 7)


def build(src="kaggle_runner/train_uncertainty_pneumothorax.ipynb",
          path="kaggle_runner/exp07/exp07_tuning.ipynb"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    srcnb = nbf.read(src, as_version=4)
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    cells = [nbf.v4.new_markdown_cell(
        "# EXP-07: temperature scaling + threshold tuning on P10 checkpoints\n"
        "No retraining. Per-branch T fit on VAL (BCE/LBFGS); operating threshold "
        "swept on VAL (max DSC_pos). Test reported at tuned point + 0.5 ref. "
        "det=seed42, ens=mean(42,43,44). MC skipped (P10-degenerate)."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(SPLITS)]
    for i in VERBATIM_CELLS:
        c = srcnb.cells[i]
        assert c.cell_type == "code", i
        cells.append(nbf.v4.new_code_cell(c.source))
    cells.append(nbf.v4.new_code_cell(EXP07))
    nb["cells"] = cells
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path, "cells:", len(cells))


if __name__ == "__main__":
    build()
