"""Build P3 Kaggle notebook: retrain MC-Dropout CORRECTLY + full pixel eval.

Root cause fixed: smp DecoderBlock.forward ignores add_module'd dropout
(silent no-op) -> original EXP-04 variance exactly 0. P3 applies Dropout2d
via forward hooks (version-proof) during training AND inference.

Same protocol as original: AdamW lr=1e-3, cosine eta_min=1e-6, 10 epochs,
BCE+0.5Dice, batch 16, aug=Flip+Affine+BC, best val DSC_pos, seed 42.
Exact preprocessing replica (MONOCHROME1, uint8, Albumentations Normalize).

Outputs: p3_metrics.json, test_predictions_p3.csv, fig3 empirical, mc_fixed_seed42.pt
"""
import nbformat as nbf

P3_CODE = r'''
# P3 — CORRECT MC DROPOUT RETRAIN + FULL PIXEL EVAL (seed 42, 10 epochs)
import os, subprocess
_q = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True)
_sm = _q.stdout.strip().split(".")
SM = (int(_sm[0]), int(_sm[1])) if len(_sm) == 2 and _sm[0].strip().isdigit() else (9, 0)
print("GPU SM:", SM)
if SM < (7, 0):
    subprocess.run(["pip", "install", "-q", "torch==2.3.1+cu118", "torchvision==0.18.1+cu118",
                    "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
os.system("pip install -q pydicom albumentations pretrainedmodels efficientnet_pytorch tqdm munch")
os.system("pip install -q --no-deps segmentation-models-pytorch==0.3.3")
import glob, json, random, ast, time
import numpy as np, pandas as pd, pydicom, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

device = torch.device("cuda"); print(torch.cuda.get_device_name(0), torch.__version__)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---- data (exact replica incl MONOCHROME1 + uint8 + Albumentations) ----
splits = pd.read_csv(glob.glob("/kaggle/input/**/patient_splits.csv", recursive=True)[0])
dmap = {}
for p in glob.glob("/kaggle/input/**/dicom-images-train/**/*.dcm", recursive=True):
    dmap[os.path.basename(p).replace(".dcm", "")] = p
splits["dcm_path"] = splits["ImageId"].map(dmap)
print("missing:", int(splits["dcm_path"].isna().sum()))

def agg_rle(cell, shape):
    import json as _j
    items = _j.loads(cell) if isinstance(cell, str) and cell.strip().startswith("[") else [cell]
    m = np.zeros(shape, np.uint8)
    for rle in items:
        rle = str(rle).strip()
        if rle in ("-1", ""): continue
        a = np.asarray([int(float(x)) for x in rle.split()], np.int64)
        st, ln = a[0::2] - 1, a[1::2]
        f = np.zeros(shape[0] * shape[1], np.uint8)
        for s0, l0 in zip(st, ln): f[s0:s0 + l0] = 1
        m = np.maximum(m, f.reshape(shape, order="F"))
    return m

class SIIMD(Dataset):
    def __init__(self, df, tf): self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        d = pydicom.dcmread(r["dcm_path"]); img = d.pixel_array.astype(np.float32)
        if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1": img = img.max() - img
        mn, mx = img.min(), img.max()
        img = ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8) if mx > mn else np.zeros_like(img, np.uint8)
        img = np.repeat(img[..., None], 3, -1)
        mask = agg_rle(r["EncodedPixelsList"], img.shape[:2])
        a = self.tf(image=img, mask=mask)
        return {"image": a["image"], "mask": a["mask"].unsqueeze(0).float()}
tr_tf = A.Compose([A.Resize(512, 512), A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
va_tf = A.Compose([A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
F_ = lambda v: splits["Fold"].isin([v, int(v)]) if str(v).isdigit() else splits["Fold"].isin(["test", "Test"])
tr_df = splits[F_("1") | F_("2") | F_("3") | F_("4")].reset_index(drop=True)
va_df = splits[F_("0")].reset_index(drop=True)
te_df = splits[splits["Fold"].isin(["test", "Test"])].reset_index(drop=True)
print(len(tr_df), len(va_df), len(te_df))
tr_ld = DataLoader(SIIMD(tr_df, tr_tf), batch_size=16, shuffle=True, num_workers=2)
va_ld = DataLoader(SIIMD(va_df, va_tf), batch_size=16, shuffle=False, num_workers=2)
te_ld = DataLoader(SIIMD(te_df, va_tf), batch_size=8, shuffle=False, num_workers=2)

# ---- model with CORRECT dropout (forward hooks, always-on when armed) ----
class FixedMC(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        self.p = p; self.armed = False
        for blk in self.model.decoder.blocks:
            blk.register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        return F.dropout2d(out, p=self.p, training=self.armed)
    def forward(self, x): return self.model(x)

class PUNet(nn.Module):  # orig arch (dropout no-op) for det/ens inference
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    def forward(self, x): return self.model(x)

W = os.path.dirname(glob.glob("/kaggle/input/**/model_seed42.pt", recursive=True)[0]) + "/"
orig = {}
for sd in (42, 43, 44):
    m = PUNet().to(device)
    m.load_state_dict(torch.load(W + f"model_seed{sd}.pt", map_location=device)); m.eval(); orig[sd] = m
print("orig weights ok")

class DiceBCE(nn.Module):
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * targets).sum() + 1) / (p.sum() + targets.sum() + 1)
        return 0.5 * bce + 0.5 * dice

def dice_np(p, y, t=0.5):
    pb = (p >= t).astype(np.float32)
    if y.sum() == 0 and pb.sum() == 0: return 1.0
    if y.sum() == 0 or pb.sum() == 0: return 0.0
    return float(2 * (pb * y).sum() / (pb.sum() + y.sum()))

# ---- train fixed MC (seed 42) ----
seed_everything(42)
mc = FixedMC().to(device); mc.armed = True
opt = torch.optim.AdamW(mc.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-6)
scl = torch.cuda.amp.GradScaler()
crit = DiceBCE()
best, bp = -1, "mc_fixed_seed42.pt"
for ep in range(1, 11):
    mc.train(); mc.armed = True; tl = 0; t0 = time.time()
    for b in tr_ld:
        x, y = b["image"].to(device), b["mask"].to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = crit(mc(x), y)
        scl.scale(loss).backward(); scl.step(opt); scl.update(); tl += loss.item() * len(x)
    sch.step()
    mc.eval(); mc.armed = False; vd = []
    with torch.no_grad():
        for b in va_ld:
            p = torch.sigmoid(mc(b["image"].to(device))).cpu().numpy()
            y = b["mask"].numpy()
            for i in range(len(y)): vd.append(dice_np(p[i, 0], y[i, 0]))
    print(f"ep {ep}/10 [{time.time()-t0:.0f}s] loss={tl/len(tr_df):.4f} valDSC={np.mean(vd):.4f}", flush=True)
    if np.mean(vd) > best: best = float(np.mean(vd)); torch.save(mc.state_dict(), bp)
print("best val:", best)
mc.load_state_dict(torch.load(bp)); mc.eval()

# ---- inference test ----
@torch.no_grad()
def run_all():
    R = []
    for b in te_ld:
        x = b["image"].to(device); y = b["mask"].numpy()
        with torch.cuda.amp.autocast():
            pd_ = torch.sigmoid(orig[42](x)).cpu().numpy()
        ps = []
        for sd in (42, 43, 44):
            with torch.cuda.amp.autocast():
                ps.append(torch.sigmoid(orig[sd](x)).cpu().numpy())
        ps = np.stack([p[:, 0] for p in ps]); pd0 = pd_[:, 0]
        em = ps.mean(0)
        ee = -(em * np.log2(em + 1e-8) + (1 - em) * np.log2(1 - em + 1e-8))
        me = np.stack([-(p * np.log2(p + 1e-8) + (1 - p) * np.log2(1 - p + 1e-8)) for p in ps]).mean(0)
        mi = np.clip(ee - me, 0, None)
        mc.armed = True; mcs = []
        for t in range(20):
            with torch.cuda.amp.autocast():
                mcs.append(torch.sigmoid(mc(x))[:, 0].cpu().numpy())
        mc.armed = False
        mcs = np.stack(mcs); mm = mcs.mean(0); mv = mcs.var(0); mv5 = mcs[:5].var(0)
        for i in range(len(y)):
            R.append({"pd": pd0[i], "em": em[i], "mi": mi[i], "mm": mm[i], "mv": mv[i],
                      "mv5": mv5[i], "m1": ps[0][i], "m2": ps[:2].mean(0)[i], "y": y[i, 0]})
    return R
REC = run_all()
print("inferred:", len(REC))

# ---- metrics ----
B = 10
def esce_brier(p, y):
    b = np.clip((p * B).astype(int), 0, B - 1); n = y.size
    e = np.zeros(B); c = np.zeros(B)
    for i in range(B):
        m = b == i
        if m.sum(): e[i] = abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(e.sum()), float(((p - y) ** 2).mean())
OUT = {}
for k, pk in (("det", "pd"), ("ens", "em"), ("mc", "mm")):
    P = np.concatenate([r[pk].ravel() for r in REC]); Y = np.concatenate([r["y"].ravel() for r in REC])
    e, br = esce_brier(P, Y)
    OUT[k] = {"esce": round(e, 6), "brier": round(float(br), 6)}
    print(k, OUT[k])
# global AUROC + empirical ROC (stride-4 seeded)
rng = np.random.default_rng(0)
roc = {}
for k, pk, uk in (("det", "pd", None), ("ens", "em", "mi"), ("mc", "mm", "mv")):
    U, E = [], []
    for r in REC:
        p = r[pk]; y = r["y"]
        u = r[uk] if uk else -(p * np.log2(p + 1e-8) + (1 - p) * np.log2(1 - p + 1e-8))
        err = np.abs(y - (p >= 0.5).astype(np.float32))
        idx = rng.choice(u.size, size=u.size // 16, replace=False)
        U.append(u.ravel()[idx]); E.append(err.ravel()[idx])
    U = np.concatenate(U); E = np.concatenate(E).astype(int)
    a = float(roc_auc_score(E, U)); fpr, tpr, _ = roc_curve(E, U)
    kp = np.linspace(0, len(fpr) - 1, 200).astype(int)
    roc[k] = {"auc": round(a, 4), "fpr": [round(float(v), 4) for v in fpr[kp]],
              "tpr": [round(float(v), 4) for v in tpr[kp]]}
    print(k, "AUC:", round(a, 4))
OUT["auroc_global"] = {k: roc[k]["auc"] for k in roc}
# per-case rows
rows = []
for j, r in enumerate(REC):
    q = te_df.iloc[j]
    u_det = float((-(r["pd"] * np.log2(r["pd"] + 1e-8) + (1 - r["pd"]) * np.log2(1 - r["pd"] + 1e-8))).reshape(-1)[np.argpartition((-(r["pd"] * np.log2(r["pd"] + 1e-8) + (1 - r["pd"]) * np.log2(1 - r["pd"] + 1e-8))).ravel(), -500)[-500:]].mean())
    rows.append({"ImageId": q["ImageId"], "PatientID": q["PatientID"], "ViewPosition": q["ViewPosition"],
                 "HasPneumothorax": q["HasPneumothorax"],
                 "det_dice": dice_np(r["pd"], r["y"]), "mc_dice": dice_np(r["mm"], r["y"]),
                 "ens_dice": dice_np(r["em"], r["y"]),
                 "mc_var_mean": float(r["mv"].mean()), "mc5_var_mean": float(r["mv5"].mean()),
                 "m1_dice": dice_np(r["m1"], r["y"]), "m2_dice": dice_np(r["m2"], r["y"])})
pd.DataFrame(rows).to_csv("test_predictions_p3.csv", index=False)
mv_mean = float(np.mean([r["mv"].mean() for r in REC]))
OUT["mc_var_mean"] = mv_mean
print("MC var mean:", mv_mean, "(>0 proves dropout engaged)")
OUT["m_ablation"] = {str(m): round(float(np.mean([dice_np(r[{"1": "m1", "2": "m2", "3": "em"}[str(m)]], r["y"]) for r in REC])), 4) for m in (1, 2, 3)}
dd = np.array([r2["det_dice"] for r2 in rows]); de = np.array([r2["ens_dice"] for r2 in rows]); dm = np.array([r2["mc_dice"] for r2 in rows])
OUT["wilcoxon"] = {"ens_det_p": float(wilcoxon(de, dd, alternative="two-sided").pvalue),
                   "mc_det_p": float(wilcoxon(dm, dd, alternative="two-sided").pvalue)}
with open("p3_metrics.json", "w") as f: json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
cols = {"det": "#e74c3c", "ens": "#2980b9", "mc": "#f39c12"}
ax.plot([0, 1], [0, 1], "k--", label="Chance (0.500)")
for k in ("det", "ens", "mc"):
    ax.plot(roc[k]["fpr"], roc[k]["tpr"], color=cols[k], linewidth=2.0, label=f"{k} ({roc[k]['auc']:.3f})")
ax.set_xlabel("FPR", fontweight="bold"); ax.set_ylabel("TPR", fontweight="bold")
ax.set_title("Empirical ROC: Error Detection (global pixels, P3)", fontweight="bold")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig("fig3_auroc_ed_empirical.png", dpi=300)
print("saved fig3 + csv + json + weights")
'''


def build(path="kaggle_runner/p3_mc_retrain.ipynb"):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    nb["cells"] = [nbf.v4.new_markdown_cell(
        "# P3: Correct MC-Dropout Retrain + Full Pixel Eval\n"
        "Fixes silent no-op dropout (smp DecoderBlock.forward bypasses add_module). "
        "Dropout via forward hooks. Same protocol: AdamW 1e-3, cosine, 10ep, seed 42."),
        nbf.v4.new_code_cell(P3_CODE)]
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build()
