"""Build P6 notebook: verbatim runtime code from the original training notebook.

Eliminates replica drift: dataset/RLE/transforms/arch/ensemble/metrics cells
are inlined character-for-character from
kaggle_runner/train_uncertainty_pneumothorax.ipynb (cells 2,3,4,5,6,7).
Only setup (GPU-safe pip) and the final eval cell (streaming + fixed MC) are new.
"""
import nbformat as nbf

BOOT = r'''
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
import torch
print("torch:", torch.__version__)
import segmentation_models_pytorch as smp
print("smp:", smp.__version__)
import albumentations as A
print("albumentations:", A.__version__)
'''

# NOTE: keep in sync with streaming design of P4 (OOM-proof).
EVAL = r'''
# P6 EVAL — verbatim pipeline (dataset/transforms/metrics above) + fixed MC
import glob, json, time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

device = torch.device("cuda")
W = os.path.dirname(glob.glob("/kaggle/input/**/mc_fixed_seed42.pt", recursive=True)[0]) + "/"

# verbatim-arch nets (dropout no-op, exactly as trained) for det/ens
orig = {}
for sd in (42, 43, 44):
    m = PneumothoraxUNet(dropout_rate=0.2, pretrained=False).to(device)
    m.load_state_dict(torch.load(W + f"model_seed{sd}.pt", map_location=device)); m.eval(); orig[sd] = m
ens_model = DeepEnsemble([orig[42], orig[43], orig[44]])

# fixed MC net: same arch + hooks that ACTUALLY apply dropout
mc = PneumothoraxUNet(dropout_rate=0.2, pretrained=False).to(device)
mc.load_state_dict(torch.load(W + "mc_fixed_seed42.pt", map_location=device)); mc.eval()
for blk in mc.model.decoder.blocks:
    blk.register_forward_hook(lambda mod, inp, out: F.dropout2d(out, p=0.2, training=True))

test_df = df_splits[df_splits["Fold"].isin(["test", "Test"])].reset_index(drop=True)
val_df = df_splits[df_splits["Fold"].isin(["0", 0])].reset_index(drop=True)
test_loader = DataLoader(SIIMPneumothoraxDataset(test_df, image_size=512, transforms=val_transforms),
                         batch_size=1, shuffle=False, num_workers=2)
print("test:", len(test_df), "val:", len(val_df))

def topk(a, k=500):
    f = a.ravel()
    return float(f[np.argpartition(f, -k)[-k:]].mean())

N = len(test_df); S = 512 // 4
mm = np.memmap("/tmp/auc_pairs.dat", dtype=np.float32, mode="w+", shape=(3, N * S * S, 2))
B = 10
acc = {k: {"n": 0, "bn": np.zeros(B), "bp": np.zeros(B), "by": np.zeros(B), "br": 0.0} for k in ("det", "ens", "mc")}
def acc_up(a, p, y):
    b = np.clip((p * B).astype(int), 0, B - 1); a["n"] += y.size
    for i in range(B):
        m = b == i
        if m.any(): a["bn"][i] += m.sum(); a["bp"][i] += p[m].sum(); a["by"][i] += y[m].sum()
    a["br"] += float(((p - y) ** 2).sum())

fout = open("test_predictions_p6.csv", "w")
fout.write("ImageId,PatientID,ViewPosition,HasPneumothorax,det_dice,mc_dice,ens_dice,det_unc,ens_unc,mc_unc,mc_var_mean\n")
t0 = time.time(); run_det = 0.0
for j, batch in enumerate(test_loader):
    img = batch["image"].to(device); mask = batch["mask"].cpu().numpy()[0, 0]
    with torch.no_grad():
        det_out = orig[42].predict_deterministic(img)
        p_det = det_out["prob"].cpu().numpy()[0, 0]; ent_det = det_out["entropy"].cpu().numpy()[0, 0]
        ens_out = ens_model(img)
        p_ens = ens_out["mean"].cpu().numpy()[0, 0]; mi_ens = ens_out["mutual_information"].cpu().numpy()[0, 0]
        mcs = [torch.sigmoid(mc(img)).cpu().numpy()[0, 0] for _ in range(20)]
    mcs = np.stack(mcs); p_mc = mcs.mean(0); var_mc = mcs.var(0)
    dm = compute_segmentation_metrics(mask, p_mc); dd = compute_segmentation_metrics(mask, p_det)
    de = compute_segmentation_metrics(mask, p_ens)
    run_det += dd["dice"]
    for k, pk, uk in (("det", p_det, ent_det), ("ens", p_ens, mi_ens), ("mc", p_mc, var_mc)):
        acc_up(acc[k], pk, mask)
        err = np.abs(mask - (pk >= 0.5).astype(np.float32))
        us = uk[::4, ::4].ravel(); es = err[::4, ::4].ravel()
        mm[{"det": 0, "ens": 1, "mc": 2}[k], j * S * S:(j + 1) * S * S, 0] = us
        mm[{"det": 0, "ens": 1, "mc": 2}[k], j * S * S:(j + 1) * S * S, 1] = es
    r = test_df.iloc[j]
    fout.write(f"{r['ImageId']},{r['PatientID']},{r['ViewPosition']},{r['HasPneumothorax']},"
               f"{dd['dice']},{dm['dice']},{de['dice']},"
               f"{topk(ent_det)},{topk(mi_ens)},{topk(var_mc)},{float(var_mc.mean())}\n")
    if (j + 1) == 100:
        print(f"SANITY det_dice@100 = {run_det/100:.4f} (expect ~0.137)", flush=True)
    if (j + 1) % 400 == 0:
        fout.flush(); mm.flush(); print(f"{j+1}/{N} ({(time.time()-t0)/60:.1f} min)", flush=True)
fout.close(); mm.flush(); print("test loop done")

def esce_of(a):
    tot = a["bn"].sum()
    return float((a["bn"] / tot * np.abs(a["bp"] / np.maximum(a["bn"], 1) - a["by"] / np.maximum(a["bn"], 1))).sum())
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
df = pd.read_csv("test_predictions_p6.csv")
OUT["case_means"] = {c: round(float(df[c].mean()), 4) for c in ["det_dice", "mc_dice", "ens_dice", "mc_var_mean"]}
OUT["wilcoxon"] = {"mc_det_p": float(wilcoxon(df["mc_dice"], df["det_dice"], alternative="two-sided").pvalue),
                   "ens_det_p": float(wilcoxon(df["ens_dice"], df["det_dice"], alternative="two-sided").pvalue)}
pos = df[df["HasPneumothorax"] == 1]
OUT["pos_means"] = {c: round(float(pos[c].mean()), 4) for c in ["det_dice", "mc_dice", "ens_dice"]}
with open("p6_metrics.json", "w") as f: json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2))
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
cols = {"det": "#e74c3c", "ens": "#2980b9", "mc": "#f39c12"}
ax.plot([0, 1], [0, 1], "k--", label="Chance (0.500)")
for k in ("det", "ens", "mc"):
    ax.plot(roc[k]["fpr"], roc[k]["tpr"], color=cols[k], linewidth=2.0, label=f"{k} ({roc[k]['auc']:.3f})")
ax.set_xlabel("FPR", fontweight="bold"); ax.set_ylabel("TPR", fontweight="bold")
ax.set_title("Empirical ROC: Error Detection (global pixels, P6 verbatim pipeline)", fontweight="bold")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig("fig3_auroc_ed_empirical.png", dpi=300)
print("saved all")
'''

VERBATIM_CELLS = (2, 3, 4, 5, 6, 7)


def build(src="kaggle_runner/train_uncertainty_pneumothorax.ipynb",
          path="kaggle_runner/p6_verbatim_eval.ipynb"):
    srcnb = nbf.read(src, as_version=4)
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    cells = [nbf.v4.new_markdown_cell(
        "# P6: Verbatim-Pipeline Eval (fixed MC)\n"
        "Dataset/RLE/transforms/arch/ensemble/metrics cells inlined unchanged from "
        "the original training notebook. Setup + eval are new.")]
    cells.append(nbf.v4.new_code_cell(BOOT))
    for i in VERBATIM_CELLS:
        c = srcnb.cells[i]
        assert c.cell_type == "code", i
        cells.append(nbf.v4.new_code_cell(c.source))
    cells.append(nbf.v4.new_code_cell(EVAL))
    nb["cells"] = cells
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path, "cells:", len(cells))


if __name__ == "__main__":
    build()
