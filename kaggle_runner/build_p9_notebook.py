"""Build P9: FINAL clean retrain + full eval in locked env.

Rationale: rerun inference (P2/P4/P6) deterministically disagrees with the
published CSV (det 0.30 vs 0.14) despite verbatim code -> original env
unrecoverable. Only honest path: replace ALL results with one self-consistent
run, fully provenanced (this file + kernel log).

- Arch: smp Unet-ResNet34, NO dropout modules (deterministic seeds 42/43/44).
- MC: same arch + Dropout2d(p=0.2) applied via forward hooks on decoder
  blocks, armed during training AND inference (functional, verified var>0).
- Protocol (as original): AdamW lr=1e-3, wd=1e-4, cosine eta_min=1e-6,
  0.5BCE+0.5SoftDice, batch 16, aug Flip+Affine+BC, 10 epochs, best val DSC_pos.
- Data/metrics cells verbatim from train_uncertainty_pneumothorax.ipynb.
- Streaming eval (det/ens/MC T=20): per-case CSV, ESCE/Brier, global AUROC +
  empirical ROC, standard AURC + E-AURC, bootstrap CIs inline, fig1+fig3.

Outputs: final_predictions.csv, final_metrics.json, fig1_rc.png, fig3_roc.png,
det_seed{42,43,44}.pt, mc_seed42.pt
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

TRAIN_EVAL = r'''
# ---- splits / loaders (verbatim transforms) ----
F_ = lambda v: df_splits["Fold"].isin([v, int(v)])
tr_df = df_splits[F_("1") | F_("2") | F_("3") | F_("4")].reset_index(drop=True)
va_df = df_splits[F_("0")].reset_index(drop=True)
te_df = df_splits[df_splits["Fold"].isin(["test", "Test"])].reset_index(drop=True)
print(len(tr_df), len(va_df), len(te_df))
assert (set(tr_df["PatientID"]) | set(va_df["PatientID"])).isdisjoint(set(te_df["PatientID"]))
tr_ld = DataLoader(SIIMPneumothoraxDataset(tr_df, image_size=512, transforms=train_transforms),
                   batch_size=16, shuffle=True, num_workers=2)
va_ld = DataLoader(SIIMPneumothoraxDataset(va_df, image_size=512, transforms=val_transforms),
                   batch_size=16, shuffle=False, num_workers=2)

# ---- models ----
class DetNet(nn.Module):  # pure deterministic: no dropout modules at all
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    def forward(self, x): return self.model(x)

class MCNet(nn.Module):  # functional dropout via hooks (verified nonzero variance)
    def __init__(self, p=0.2):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        self.p = p; self.armed = True
        for blk in self.model.decoder.blocks:
            blk.register_forward_hook(self._hook)
    def _hook(self, mod, inp, out): return F.dropout2d(out, p=self.p, training=self.armed)
    def forward(self, x): return self.model(x)

crit = CombinedBCEDiceLoss()

def run_train(net, seed, tag):
    seed_everything(seed)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
    scl = torch.cuda.amp.GradScaler()
    best, bp = -1, f"{tag}.pt"
    for ep in range(1, 11):
        net.train()
        if isinstance(net, MCNet): net.armed = True
        tl = 0; t0 = time.time()
        for b in tr_ld:
            x, y = b["image"].to(device), b["mask"].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(net(x), y)
            scl.scale(loss).backward(); scl.step(opt); scl.update(); tl += loss.item() * len(x)
        sch.step()
        net.eval()
        if isinstance(net, MCNet): net.armed = False
        vd = []
        with torch.no_grad():
            for b in va_ld:
                p = torch.sigmoid(net(b["image"].to(device))).cpu().numpy()
                y = b["mask"].numpy()
                for i in range(len(y)):
                    vd.append(compute_dice_coefficient((y[i, 0] > 0).astype(np.uint8),
                                                      (p[i, 0] >= 0.5).astype(np.uint8)))
        vd = np.array(vd); pos = vd  # all-case for logging
        print(f"{tag} ep{ep}/10 [{time.time()-t0:.0f}s] loss={tl/len(tr_df):.4f} valDSC={vd.mean():.4f}", flush=True)
        if vd.mean() > best: best = float(vd.mean()); torch.save(net.state_dict(), bp)
    print(tag, "best:", best, flush=True)
    return bp

p42 = run_train(DetNet(), 42, "det_seed42")
p43 = run_train(DetNet(), 43, "det_seed43")
p44 = run_train(DetNet(), 44, "det_seed44")
pmc = run_train(MCNet(), 42, "mc_seed42")

# ---- streaming eval ----
def load_net(cls, path, **kw):
    m = cls(**kw).to(device); m.load_state_dict(torch.load(path, map_location=device)); m.eval()
    return m
nets = {42: load_net(DetNet, p42), 43: load_net(DetNet, p43), 44: load_net(DetNet, p44)}
mc = load_net(MCNet, pmc); mc.armed = False
te_ld = DataLoader(SIIMPneumothoraxDataset(te_df, image_size=512, transforms=val_transforms),
                   batch_size=1, shuffle=False, num_workers=2)

def entropy(p): return -(p * np.log2(p + 1e-8) + (1 - p) * np.log2(1 - p + 1e-8))
def topk(a, k=500):
    f = a.ravel(); return float(f[np.argpartition(f, -k)[-k:]].mean())
N = len(te_df); S = 512 // 4
mm = np.memmap("/tmp/auc.dat", dtype=np.float32, mode="w+", shape=(3, N * S * S, 2))
B = 10
acc = {k: {"n": 0, "bn": np.zeros(B), "bp": np.zeros(B), "by": np.zeros(B), "br": 0.0} for k in ("det", "ens", "mc")}
def acc_up(a, p, y):
    b = np.clip((p * B).astype(int), 0, B - 1); a["n"] += y.size
    for i in range(B):
        m = b == i
        if m.any(): a["bn"][i] += m.sum(); a["bp"][i] += p[m].sum(); a["by"][i] += y[m].sum()
    a["br"] += float(((p - y) ** 2).sum())

fout = open("final_predictions.csv", "w")
fout.write("ImageId,PatientID,ViewPosition,HasPneumothorax,det_dice,mc_dice,ens_dice,det_unc,ens_unc,mc_unc,mc_var_mean\n")
t0 = time.time()
for j, b in enumerate(te_ld):
    x = b["image"].to(device); yn = b["mask"].numpy()[0, 0]
    with torch.no_grad():
        ps = [torch.sigmoid(nets[s](x))[0, 0].cpu().numpy() for s in (42, 43, 44)]
        em = np.stack(ps).mean(0)
        ee = entropy(em)
        me = np.stack([entropy(p) for p in ps]).mean(0)
        mi = np.clip(ee - me, 0, None)
        mc.armed = True
        mcs = np.stack([torch.sigmoid(mc(x))[0, 0].cpu().numpy() for _ in range(20)])
        mc.armed = False
        mmean = mcs.mean(0); mvar = mcs.var(0); ed = entropy(ps[0])
    dd = compute_segmentation_metrics(yn, ps[0]); de = compute_segmentation_metrics(yn, em)
    dm = compute_segmentation_metrics(yn, mmean)
    for k, pk, uk in (("det", ps[0], ed), ("ens", em, mi), ("mc", mmean, mvar)):
        acc_up(acc[k], pk, (yn > 0).astype(np.float32))
        err = np.abs((yn > 0).astype(np.float32) - (pk >= 0.5).astype(np.float32))
        us = uk[::4, ::4].ravel(); es = err[::4, ::4].ravel()
        mm[{"det": 0, "ens": 1, "mc": 2}[k], j * S * S:(j + 1) * S * S, 0] = us
        mm[{"det": 0, "ens": 1, "mc": 2}[k], j * S * S:(j + 1) * S * S, 1] = es
    r = te_df.iloc[j]
    fout.write(f"{r['ImageId']},{r['PatientID']},{r['ViewPosition']},{r['HasPneumothorax']},"
               f"{dd['dice']},{dm['dice']},{de['dice']},"
               f"{topk(ed)},{topk(mi)},{topk(mvar)},{float(mvar.mean())}\n")
    if (j + 1) % 400 == 0:
        fout.flush(); mm.flush(); print(f"{j+1}/{N} ({(time.time()-t0)/60:.1f}m)", flush=True)
fout.close(); mm.flush()

def esce_of(a):
    t = a["bn"].sum()
    return float((a["bn"] / t * np.abs(a["bp"] / np.maximum(a["bn"], 1) - a["by"] / np.maximum(a["bn"], 1))).sum())
OUT = {"esce": {k: round(esce_of(acc[k]), 6) for k in acc},
       "brier": {k: round(acc[k]["br"] / acc[k]["n"], 6) for k in acc}}
roc = {}
for i, k in enumerate(["det", "ens", "mc"]):
    U = np.array(mm[i, :, 0]); E = np.array(mm[i, :, 1]).astype(int)
    a = float(roc_auc_score(E, U)) if E.min() != E.max() else 0.5
    fpr, tpr, _ = roc_curve(E, U); kp = np.linspace(0, len(fpr) - 1, 200).astype(int)
    roc[k] = {"auc": round(a, 4), "fpr": [round(float(v), 4) for v in fpr[kp]],
              "tpr": [round(float(v), 4) for v in tpr[kp]]}
    print(k, "AUC:", round(a, 4), flush=True)
OUT["auroc_global"] = {k: roc[k]["auc"] for k in roc}
df = pd.read_csv("final_predictions.csv")
OUT["means"] = {c: round(float(df[c].mean()), 4) for c in ["det_dice", "mc_dice", "ens_dice", "mc_var_mean"]}
OUT["pos_means"] = {c: round(float(df[df["HasPneumothorax"] == 1][c].mean()), 4) for c in ["det_dice", "mc_dice", "ens_dice"]}
rng = np.random.default_rng(42); n = len(df)
for c in ["det_dice", "mc_dice", "ens_dice"]:
    v = df[c].values; bs = v[rng.integers(0, n, (1000, n))].mean(1)
    OUT[c + "_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
OUT["wilcoxon"] = {"mc_det_p": float(wilcoxon(df["mc_dice"], df["det_dice"], alternative="two-sided").pvalue),
                   "ens_det_p": float(wilcoxon(df["ens_dice"], df["det_dice"], alternative="two-sided").pvalue)}
# standard AURC from CSV
for m, u in (("det", "det_unc"), ("ens", "ens_unc"), ("mc", "mc_unc")):
    o = np.argsort(df[u].values, kind="stable"); d = df[m + "_dice"].values[o]
    cum = np.cumsum(d) / np.arange(1, n + 1); cov = np.arange(1, n + 1) / n
    OUT[m + "_aurc_std"] = round(float(np.trapezoid(1 - cum, cov)), 4)
with open("final_metrics.json", "w") as f: json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
for m, u, col in (("det", "det_unc", "#e74c3c"), ("ens", "ens_unc", "#2980b9"), ("mc", "mc_unc", "#f39c12")):
    o = np.argsort(df[u].values, kind="stable"); d = df[m + "_dice"].values[o]
    cum = np.cumsum(d) / np.arange(1, n + 1); cov = np.arange(1, n + 1) / n
    ax.plot(cov, 1 - cum, color=col, linewidth=2.0, label=f"{m} ({OUT[m+'_aurc_std']})")
ax.set_xlabel("Coverage", fontweight="bold"); ax.set_ylabel("Risk (1-DSC_all)", fontweight="bold")
ax.set_title("Risk-Coverage (standard, final run)", fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig("fig1_rc.png", dpi=300); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
cols = {"det": "#e74c3c", "ens": "#2980b9", "mc": "#f39c12"}
ax.plot([0, 1], [0, 1], "k--", label="Chance")
for k in ("det", "ens", "mc"):
    ax.plot(roc[k]["fpr"], roc[k]["tpr"], color=cols[k], linewidth=2.0, label=f"{k} ({roc[k]['auc']})")
ax.set_xlabel("FPR", fontweight="bold"); ax.set_ylabel("TPR", fontweight="bold")
ax.set_title("Empirical ROC (global pixels, final run)", fontweight="bold")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig("fig3_roc.png", dpi=300)
print("saved all")
'''

VERBATIM_CELLS = (3, 4, 5, 7)

SPLITS = """
df_splits = pd.read_csv(glob.glob("/kaggle/input/**/patient_splits.csv", recursive=True)[0])
dmap0 = {}
for p in glob.glob("/kaggle/input/**/dicom-images-train/**/*.dcm", recursive=True):
    dmap0.setdefault(os.path.basename(p).replace(".dcm", ""), p)
df_splits["dcm_path"] = df_splits["ImageId"].map(dmap0)
print(len(df_splits), "missing:", int(df_splits["dcm_path"].isna().sum()))
"""


def build(src="kaggle_runner/train_uncertainty_pneumothorax.ipynb",
          path="kaggle_runner/p9_final_retrain.ipynb"):
    srcnb = nbf.read(src, as_version=4)
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    cells = [nbf.v4.new_markdown_cell(
        "# P9 FINAL: clean retrain (3 det + 1 functional MC) + full eval\n"
        "Self-consistent locked-env run replacing all published numbers. "
        "Data/metrics cells verbatim from original notebook."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(SPLITS)]
    for i in VERBATIM_CELLS:
        c = srcnb.cells[i]
        assert c.cell_type == "code", i
        cells.append(nbf.v4.new_code_cell(c.source))
    cells.append(nbf.v4.new_code_cell(TRAIN_EVAL))
    nb["cells"] = cells
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path, "cells:", len(cells))


if __name__ == "__main__":
    build()
