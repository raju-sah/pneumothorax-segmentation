"""Build P4 Kaggle notebook: streaming pixel eval with fixed MC weights.

No training. Streams test cases one-by-one (OOM-proof): per-case scalars to
CSV, ESCE/Brier accumulators, AUROC pairs to disk memmap.
Loads orig seeds 42-44 + mc_fixed_seed42 from weights dataset v2.
"""
import nbformat as nbf

P4_CODE = r'''
# P4 — STREAMING PIXEL EVAL (fixed MC weights, OOM-proof)
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
import glob, json, random, time
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
random.seed(42); np.random.seed(42); torch.manual_seed(42)

splits = pd.read_csv(glob.glob("/kaggle/input/**/patient_splits.csv", recursive=True)[0])
dmap = {}
for p in glob.glob("/kaggle/input/**/dicom-images-train/**/*.dcm", recursive=True):
    dmap[os.path.basename(p).replace(".dcm", "")] = p
splits["dcm_path"] = splits["ImageId"].map(dmap)

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

va_tf = A.Compose([A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

def load_xy(row):
    d = pydicom.dcmread(row["dcm_path"]); img = d.pixel_array.astype(np.float32)
    if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1": img = img.max() - img
    mn, mx = img.min(), img.max()
    img = ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8) if mx > mn else np.zeros_like(img, np.uint8)
    img = np.repeat(img[..., None], 3, -1)
    mask = agg_rle(row["EncodedPixelsList"], img.shape[:2])
    a = va_tf(image=img, mask=mask)
    return a["image"], a["mask"].float()

class PUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    def forward(self, x): return self.model(x)

class FixedMC(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.p = p; self.armed = False
        for blk in self.model.decoder.blocks:
            blk.register_forward_hook(self._hook)
    def _hook(self, mod, inp, out): return F.dropout2d(out, p=self.p, training=self.armed)
    def forward(self, x): return self.model(x)

W = os.path.dirname(glob.glob("/kaggle/input/**/mc_fixed_seed42.pt", recursive=True)[0]) + "/"
print("weights:", sorted(os.listdir(W)))
orig = {}
for sd in (42, 43, 44):
    m = PUNet().to(device)
    m.load_state_dict(torch.load(W + f"model_seed{sd}.pt", map_location=device)); m.eval(); orig[sd] = m
mc = FixedMC().to(device)
mc.load_state_dict(torch.load(W + "mc_fixed_seed42.pt", map_location=device)); mc.eval(); mc.armed = False
print("all weights loaded")

def dice_np(p, y, t=0.5):
    pb = (p >= t).astype(np.float32)
    if y.sum() == 0 and pb.sum() == 0: return 1.0
    if y.sum() == 0 or pb.sum() == 0: return 0.0
    return float(2 * (pb * y).sum() / (pb.sum() + y.sum()))

def entropy(p):
    return -(p * np.log2(p + 1e-8) + (1 - p) * np.log2(1 - p + 1e-8))

def topk_mean(u, k=500):
    f = u.ravel()
    return float(f[np.argpartition(f, -k)[-k:]].mean()) if f.size >= k else float(f.mean())

te_df = splits[splits["Fold"].isin(["test", "Test"])].reset_index(drop=True)
va_df = splits[splits["Fold"].isin(["0", 0])].reset_index(drop=True)
N = len(te_df); S = 512 // 4
mm = np.memmap("/tmp/auc_pairs.dat", dtype=np.float32, mode="w+", shape=(3, N * S * S, 2))
B = 10
acc = {k: {"n": 0, "bn": np.zeros(B), "bp": np.zeros(B), "by": np.zeros(B), "br": 0.0} for k in ("det", "ens", "mc")}
def acc_up(a, p, y):
    b = np.clip((p * B).astype(int), 0, B - 1); a["n"] += y.size
    for i in range(B):
        m = b == i
        if m.any(): a["bn"][i] += m.sum(); a["bp"][i] += p[m].sum(); a["by"][i] += y[m].sum()
    a["br"] += float(((p - y) ** 2).sum())

fout = open("test_predictions_p4.csv", "w")
fout.write("ImageId,PatientID,ViewPosition,HasPneumothorax,det_dice,mc_dice,ens_dice,m1_dice,m2_dice,det_unc,ens_unc,mc_unc,mc_var_mean\n")
t0 = time.time()
for j in range(N):
    row = te_df.iloc[j]
    x, y = load_xy(row); yn = y.numpy()[0]; xb = x.unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        pd_ = torch.sigmoid(orig[42](xb))[0, 0].cpu().numpy()
    ps = []
    with torch.no_grad():
        for sd in (42, 43, 44):
            with torch.cuda.amp.autocast():
                ps.append(torch.sigmoid(orig[sd](xb))[0, 0].cpu().numpy())
    em = np.stack(ps).mean(0)
    ee = entropy(em)
    me = np.stack([entropy(p) for p in ps]).mean(0)
    mi = np.clip(ee - me, 0, None)
    mc.armed = True; mcs = []
    with torch.no_grad():
        for t in range(20):
            with torch.cuda.amp.autocast():
                mcs.append(torch.sigmoid(mc(xb))[0, 0].cpu().numpy())
    mc.armed = False
    mcs = np.stack(mcs); mmean = mcs.mean(0); mvar = mcs.var(0)
    ed = entropy(pd_)
    for k, pk, uk in (("det", pd_, ed), ("ens", em, mi), ("mc", mmean, mvar)):
        acc_up(acc[k], pk, yn)
        err = np.abs(yn - (pk >= 0.5).astype(np.float32))
        us = uk[::4, ::4].ravel(); es = err[::4, ::4].ravel()
        mm[{"det": 0, "ens": 1, "mc": 2}[k], j * S * S:(j + 1) * S * S, 0] = us
        mm[{"det": 0, "ens": 1, "mc": 2}[k], j * S * S:(j + 1) * S * S, 1] = es
    fout.write(f"{row['ImageId']},{row['PatientID']},{row['ViewPosition']},{row['HasPneumothorax']},"
               f"{dice_np(pd_, yn)},{dice_np(mmean, yn)},{dice_np(em, yn)},"
               f"{dice_np(ps[0], yn)},{dice_np(np.stack(ps[:2]).mean(0), yn)},"
               f"{topk_mean(ed)},{topk_mean(mi)},{topk_mean(mvar)},{float(mvar.mean())}\n")
    if (j + 1) % 400 == 0:
        fout.flush(); mm.flush()
        print(f"{j+1}/{N} ({(time.time()-t0)/60:.1f} min)", flush=True)
fout.close(); mm.flush()
print("test loop done")

# temp scaling on val (ens mean logits)
vl, vy = [], []
with torch.no_grad():
    for j in range(len(va_df)):
        x, y = load_xy(va_df.iloc[j]); xb = x.unsqueeze(0).to(device)
        ps = []
        for sd in (42, 43, 44):
            with torch.cuda.amp.autocast():
                ps.append(torch.sigmoid(orig[sd](xb))[0, 0].cpu().numpy())
        em = np.stack(ps).mean(0)
        vl.append(np.log(np.clip(em, 1e-6, 1 - 1e-6) / np.clip(1 - em, 1e-6, 1 - 1e-6)).ravel())
        vy.append(y.numpy()[0].ravel())
vl = np.concatenate(vl); vy = np.concatenate(vy)
best_T, best_nll = 1.0, 1e18
for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    p = 1 / (1 + np.exp(-vl / T))
    nll = float(-(vy * np.log(p + 1e-8) + (1 - vy) * np.log(1 - p + 1e-8)).mean())
    if nll < best_nll: best_T, best_nll = T, nll
print("temp T:", best_T)

def esce_of(a):
    tot = a["bn"].sum()
    return float((a["bn"] / tot * np.abs(a["bp"] / np.maximum(a["bn"], 1) - a["by"] / np.maximum(a["bn"], 1))).sum())
OUT = {"esce": {k: round(esce_of(acc[k]), 6) for k in acc},
       "brier": {k: round(acc[k]["br"] / acc[k]["n"], 6) for k in acc},
       "temp_T_val": best_T}
roc = {}
for i, k in enumerate(["det", "ens", "mc"]):
    U = np.array(mm[i, :, 0]); E = np.array(mm[i, :, 1]).astype(int)
    a = float(roc_auc_score(E, U)) if E.min() != E.max() else 0.5
    fpr, tpr, _ = roc_curve(E, U); kp = np.linspace(0, len(fpr) - 1, 200).astype(int)
    roc[k] = {"auc": round(a, 4), "fpr": [round(float(v), 4) for v in fpr[kp]],
              "tpr": [round(float(v), 4) for v in tpr[kp]]}
    print(k, "AUC:", round(a, 4), flush=True)
OUT["auroc_global"] = {k: roc[k]["auc"] for k in roc}
df = pd.read_csv("test_predictions_p4.csv")
OUT["case_means"] = {c: round(float(df[c].mean()), 4) for c in
                     ["det_dice", "mc_dice", "ens_dice", "m1_dice", "m2_dice", "mc_var_mean"]}
OUT["wilcoxon"] = {"mc_det_p": float(wilcoxon(df["mc_dice"], df["det_dice"], alternative="two-sided").pvalue),
                   "ens_det_p": float(wilcoxon(df["ens_dice"], df["det_dice"], alternative="two-sided").pvalue)}
pos = df[df["HasPneumothorax"] == 1]
OUT["pos_means"] = {c: round(float(pos[c].mean()), 4) for c in ["det_dice", "mc_dice", "ens_dice"]}
with open("p4_metrics.json", "w") as f: json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
cols = {"det": "#e74c3c", "ens": "#2980b9", "mc": "#f39c12"}
ax.plot([0, 1], [0, 1], "k--", label="Chance (0.500)")
for k in ("det", "ens", "mc"):
    ax.plot(roc[k]["fpr"], roc[k]["tpr"], color=cols[k], linewidth=2.0, label=f"{k} ({roc[k]['auc']:.3f})")
ax.set_xlabel("FPR", fontweight="bold"); ax.set_ylabel("TPR", fontweight="bold")
ax.set_title("Empirical ROC: Error Detection (global pixels, P4)", fontweight="bold")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig("fig3_auroc_ed_empirical.png", dpi=300)
print("saved all")
'''


def build(path="kaggle_runner/p4_streaming_eval.ipynb"):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    nb["cells"] = [nbf.v4.new_markdown_cell(
        "# P4: Streaming Pixel Eval (fixed MC weights)\n"
        "OOM-proof: per-case streaming, memmap AUROC pairs. No training."),
        nbf.v4.new_code_cell(P4_CODE)]
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build()
