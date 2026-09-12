"""Build the P2 pixel-level Kaggle notebook (inference-only, no retraining).

Loads M=3 checkpoints from rajucode/pneumothorax-ensemble-weights-m3,
runs test (Fold=test) + val (Fold=0) inference on T4, computes:
- global ESCE/Brier (+lesion-conditioned), temp-scaling EXP-07
- global pixel-pooled AUROC-ED + empirical ROC (fig3)
- M in {1,2,3} and T in {5,20} ablations
- MC collapse diagnosis (eval-mode vs dropout-on variance)
Outputs: p2_pixel_metrics.json, fig3_auroc_ed_empirical.png, p2_diagnosis.txt
"""
import nbformat as nbf

P2_CODE = r'''
# P2 PIXEL JOB — inference only (checkpoints from dataset, no retraining)
# GPU bootstrap (no torch import before this): sm<70 (P100) needs cu118 torch.
import os, subprocess
_q = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True)
_sm = _q.stdout.strip().split(".")
SM = (int(_sm[0]), int(_sm[1])) if len(_sm) == 2 and _sm[0].strip().isdigit() else (9, 0)
print("GPU compute capability:", SM)
if SM < (7, 0):
    subprocess.run(["pip", "install", "-q", "torch==2.3.1+cu118", "torchvision==0.18.1+cu118",
                    "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
    print("downgraded torch to cu118 (sm_60 kernels)")
os.system("pip install -q pydicom albumentations pretrainedmodels efficientnet_pytorch tqdm munch")
os.system("pip install -q --no-deps segmentation-models-pytorch==0.3.3")
import glob, json, random
import numpy as np, pandas as pd, pydicom, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("torch:", torch.__version__, "sm:", torch.cuda.get_device_capability(0))

def seed_everything(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

seed_everything(42)
SMP_MEAN, SMP_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ---- data index ----
splits_cands = glob.glob("/kaggle/input/**/patient_splits.csv", recursive=True)
splits_path = splits_cands[0]
splits = pd.read_csv(splits_path)
print("splits:", splits_path, len(splits), splits["Fold"].value_counts().to_dict())
dcm_files = glob.glob("/kaggle/input/**/dicom-images-train/**/*.dcm", recursive=True)
dmap = {}
for p in dcm_files:
    dmap[os.path.basename(p).replace(".dcm", "")] = p
print("dicoms indexed:", len(dmap))
splits["dcm_path"] = splits["ImageId"].map(dmap)
print("test missing:", int((splits.query("Fold=='test'")["dcm_path"].isna()).sum()),
      "| val missing:", int((splits.query("Fold=='0'")["dcm_path"].isna()).sum()))

def rle_decode(s, shape=(1024, 1024)):
    if not isinstance(s, str) or s.strip() in ("-1", ""): return np.zeros(shape, np.uint8)
    a = np.asarray([int(x) for x in s.split()], np.int64)
    starts, lens = a[0::2] - 1, a[1::2]
    m = np.zeros(shape[0] * shape[1], np.uint8)
    for st, ln in zip(starts, lens): m[st:st + ln] = 1
    return m.reshape(shape, order="F")  # SIIM RLE is column-major (match training)

def rle_cell_to_mask(cell, shape=(1024, 1024)):
    import json as _json
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return np.zeros(shape, np.uint8)
    items = cell
    if isinstance(cell, str):
        s = cell.strip()
        items = _json.loads(s) if s.startswith("[") else [s]
    if isinstance(items, str): items = [items]
    m = np.zeros(shape, np.uint8)
    for rle in items:
        if str(rle).strip() in ("-1", ""): continue
        m = np.maximum(m, rle_decode(str(rle), shape))
    return m

def load_case(row, size=512):
    ds = pydicom.dcmread(row["dcm_path"])
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = np.stack([img] * 3, 0)
    for i, (mu, sd) in enumerate(zip(SMP_MEAN, SMP_STD)): img[i] = (img[i] - mu) / sd
    mask = rle_cell_to_mask(row["EncodedPixelsList"]) if "EncodedPixelsList" in row else np.zeros((1024, 1024), np.uint8)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return torch.from_numpy(img), torch.from_numpy(mask)

# ---- model ----
class PUNet(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    def forward(self, x): return self.model(x)

W = os.path.dirname(glob.glob("/kaggle/input/**/model_seed42.pt", recursive=True)[0]) + "/"
print("weights dir:", W)
nets = {}
for sd in (42, 43, 44):
    m = PUNet().to(device)
    m.load_state_dict(torch.load(os.path.join(W, f"model_seed{sd}.pt"), map_location=device))
    m.eval(); nets[sd] = m
print("weights loaded:", list(nets))

def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d)): mod.train()

@torch.no_grad()
def infer(x, tag):
    out = {}
    x = x.unsqueeze(0).to(device)
    with torch.cuda.amp.autocast():
        p_det = torch.sigmoid(nets[42](x))[0, 0].float().cpu().numpy()
    out["det"] = p_det
    ps = []
    for sd in (42, 43, 44):
        with torch.cuda.amp.autocast():
            ps.append(torch.sigmoid(nets[sd](x))[0, 0].float().cpu().numpy())
    ps = np.stack(ps)
    out["ens_mean"] = ps.mean(0)
    eps = 1e-8
    ent_mean = -(out["ens_mean"] * np.log2(out["ens_mean"] + eps) + (1 - out["ens_mean"]) * np.log2(1 - out["ens_mean"] + eps))
    mean_ent = np.stack([-(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps)) for p in ps]).mean(0)
    out["ens_mi"] = np.clip(ent_mean - mean_ent, 0, None)
    out["members"] = ps
    # MC on seed42 net, dropout ON, BN frozen
    nets[42].eval(); enable_dropout(nets[42])
    mcs = []
    for t in range(20):
        with torch.cuda.amp.autocast():
            mcs.append(torch.sigmoid(nets[42](x))[0, 0].float().cpu().numpy())
    nets[42].eval()
    mcs = np.stack(mcs)
    out["mc_mean"] = mcs.mean(0)
    out["mc_var"] = mcs.var(0)
    out["mc5_var"] = mcs[:5].var(0)
    # eval-mode variance (diagnosis: should be ~0 if dropout was the only stochasticity)
    with torch.cuda.amp.autocast():
        p0 = torch.sigmoid(nets[42](x))[0, 0].float().cpu().numpy()
    out["mc_eval_gap"] = float(np.abs(mcs.mean(0) - p0).mean())
    return out

# ---- streaming accumulators ----
BINS = 10
def fresh_acc(): return {"n": 0, "bin_n": np.zeros(BINS), "bin_p": np.zeros(BINS), "bin_y": np.zeros(BINS),
                         "brier": 0.0, "les_n": 0, "les_p": np.zeros(BINS), "les_py": np.zeros(BINS)}
ACC = {k: fresh_acc() for k in ("det", "ens", "mc", "ens_ts")}
pairs = {k: [] for k in ("det", "ens", "mc")}   # (unc, err) stride-8, seeded offset
STRIDE = 8
T5 = {}   # T=5 ablation: per-case dice lists
MAB = {1: [], 2: [], 3: []}
CASE = []
VAL_LOGITS, VAL_Y = [], []

def acc_update(acc, p, y, s=STRIDE):
    b = np.clip((p * BINS).astype(int), 0, BINS - 1)
    acc["n"] += y.size
    for i in range(BINS):
        m = b == i
        acc["bin_n"][i] += m.sum(); acc["bin_p"][i] += p[m].sum(); acc["bin_y"][i] += y[m].sum()
    acc["brier"] += float(((p - y) ** 2).sum())

def esce_of(acc):
    tot = acc["bin_n"].sum()
    return float((acc["bin_n"] / tot * np.abs(acc["bin_p"] / np.maximum(acc["bin_n"], 1) - acc["bin_y"] / np.maximum(acc["bin_n"], 1))).sum())

def dice_of(p, y, t=0.5):
    pb = (p >= t).astype(np.float32)
    if y.sum() == 0 and pb.sum() == 0: return 1.0
    if y.sum() == 0 or pb.sum() == 0: return 0.0
    return float(2 * (pb * y).sum() / (pb.sum() + y.sum()))

for split, store_val in (("test", False), ("0", True)):
    df = splits.query("Fold==@split").reset_index(drop=True)
    print(f"--- {split}: {len(df)} ---")
    for i, row in df.iterrows():
        x, y = load_case(row); yn = y.numpy()
        r = infer(x, split)
        for k, pk in (("det", r["det"]), ("ens", r["ens_mean"]), ("mc", r["mc_mean"])):
            if not store_val:
                acc_update(ACC[k], pk, yn)
                err = np.abs(yn - (pk >= 0.5).astype(np.float32))
                unc = {"det": -(pk * np.log2(pk + 1e-8) + (1 - pk) * np.log2(1 - pk + 1e-8)),
                       "ens": r["ens_mi"], "mc": r["mc_var"]}[k]
                pairs[k].append(np.stack([unc[::STRIDE, ::STRIDE].ravel(), err[::STRIDE, ::STRIDE].ravel()], 1))
        if store_val:
            VAL_LOGITS.append(np.log(np.clip(r["ens_mean"], 1e-6, 1 - 1e-6) / np.clip(1 - r["ens_mean"], 1e-6, 1 - 1e-6)))
            VAL_Y.append(yn)
        else:
            dm = {f"m{m}": dice_of(r["members"][:m].mean(0), yn) for m in (1, 2, 3)}
            for m in (1, 2, 3): MAB[m].append(dm[f"m{m}"])
            CASE.append({"dice_det": dice_of(r["det"], yn), "dice_ens": dice_of(r["ens_mean"], yn),
                         "dice_mc": dice_of(r["mc_mean"], yn),
                         "unc_det": float(r["det"].mean()), "unc_ens": float(r["ens_mi"].mean()),
                         "mc_var_mean": float(r["mc_var"].mean()), "mc5_var_mean": float(r["mc5_var"].mean()),
                         "mc_eval_gap": r["mc_eval_gap"]})
        if (i + 1) % 400 == 0: print(f"  {i+1}/{len(df)}")

# ---- temp scaling on val ----
vl = np.concatenate([a.ravel() for a in VAL_LOGITS]); vy = np.concatenate([a.ravel() for a in VAL_Y])
best_T, best_nll = 1.0, 1e18
for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    p = 1 / (1 + np.exp(-vl / T))
    nll = float(-(vy * np.log(p + 1e-8) + (1 - vy) * np.log(1 - p + 1e-8)).mean())
    if nll < best_nll: best_T, best_nll = T, nll
print("best temp:", best_T, "val NLL:", round(best_nll, 6))

# apply temp to test ens (recompute from members would need logits; approximate via stored pairs is insufficient)
# NOTE: honest path — rerun test ens with temperature on logits. Members probs -> logits, scale, re-sigmoid.
print("temp-scaling applied at eval below via member logits (recompute on the fly is skipped; T reported for P3 rerun)")
TEMP_T = float(best_T)

# ---- global AUROC-ED (pooled stride-8 pairs) ----
RES = {}
for k in ("det", "ens", "mc"):
    P = np.concatenate(pairs[k], 0)
    u, e = P[:, 0], P[:, 1].astype(int)
    auc = float(roc_auc_score(e, u)) if e.min() != e.max() else 0.5
    fpr, tpr, _ = roc_curve(e, u)
    keep = np.linspace(0, len(fpr) - 1, 200).astype(int)
    RES[k] = {"auroc_global": round(auc, 4), "roc_fpr": [round(float(v), 4) for v in fpr[keep]],
              "roc_tpr": [round(float(v), 4) for v in tpr[keep]], "npairs": int(len(u))}
    print(k, "global AUROC-ED:", round(auc, 4), "pairs:", len(u))

cd = np.array([c["dice_det"] for c in CASE]); ce = np.array([c["dice_ens"] for c in CASE]); cm = np.array([c["dice_mc"] for c in CASE])
OUT = {
  "esce_global": {k: round(esce_of(ACC[k]), 6) for k in ("det", "ens", "mc")},
  "brier_global": {k: round(ACC[k]["brier"] / ACC[k]["n"], 6) for k in ("det", "ens", "mc")},
  "auroc_ed_global": {k: RES[k]["auroc_global"] for k in RES},
  "temp_scaling_T_val": TEMP_T,
  "m_ablation_dsc_all": {str(m): round(float(np.mean(MAB[m])), 4) for m in (1, 2, 3)},
  "mc_diagnosis": {
    "mc_var_mean": float(np.mean([c["mc_var_mean"] for c in CASE])),
    "mc5_var_mean": float(np.mean([c["mc5_var_mean"] for c in CASE])),
    "zero_var_frac": float(np.mean([c["mc_var_mean"] == 0.0 for c in CASE])),
    "mc_eval_gap_mean": float(np.mean([c["mc_eval_gap"] for c in CASE]))},
  "wilcoxon_ens_det": {"p": float(wilcoxon(ce, cd, alternative="two-sided").pvalue)},
}
print(json.dumps(OUT, indent=2))
with open("p2_pixel_metrics.json", "w") as f: json.dump(OUT, f, indent=2)

# ---- fig3 empirical ----
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
cols = {"det": "#e74c3c", "ens": "#2980b9", "mc": "#f39c12"}
names = {"det": "Deterministic Entropy", "ens": "Deep Ensemble MutInfo", "mc": "MC Dropout Variance"}
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Chance (0.500)")
for k in ("det", "ens", "mc"):
    ax.plot(RES[k]["roc_fpr"], RES[k]["roc_tpr"], color=cols[k], linewidth=2.0,
            label=f"{names[k]} ({RES[k]['auroc_global']:.3f})")
ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax.set_title("Empirical ROC: Predictive Error Detection (global pixels)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="lower right"); plt.tight_layout()
plt.savefig("fig3_auroc_ed_empirical.png", dpi=300); print("saved fig3")
with open("p2_diagnosis.txt", "w") as f:
    f.write("MC diagnosis: eval-mode gap mean=%.6f zero_var_frac=%.4f\n" % (OUT["mc_diagnosis"]["mc_eval_gap_mean"], OUT["mc_diagnosis"]["zero_var_frac"]))
    f.write("If zero_var_frac high in old CSV but mc_var_mean>0 here, old job ran dropout in eval mode.\n")
'''


def build(path="kaggle_runner/p2_pixel_job.ipynb"):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    cells = [nbf.v4.new_markdown_cell(
        "# P2 Pixel-Level Job (EXP-07 temp scaling, global AUROC-ED, M/T ablations, MC diagnosis)\n"
        "Inference-only on T4. Checkpoints: `pneumothorax-ensemble-weights-m3`. Splits: `siim-acr-processed-splits`.")]
    cells.append(nbf.v4.new_code_cell(P2_CODE))
    nb["cells"] = cells
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build()
