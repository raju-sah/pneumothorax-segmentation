"""Build P7 notebook: source-priority experiment (150 cases, det only)."""
import nbformat as nbf

P7 = r'''
import os, subprocess
_q = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True)
_sm = _q.stdout.strip().split(".")
SM = (int(_sm[0]), int(_sm[1])) if len(_sm) == 2 and _sm[0].strip().isdigit() else (9, 0)
if SM < (7, 0):
    subprocess.run(["pip", "install", "-q", "torch==2.3.1+cu118", "torchvision==0.18.1+cu118",
                    "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
os.system("pip install -q pydicom albumentations pretrainedmodels efficientnet_pytorch tqdm munch scikit-learn")
os.system("pip install -q --no-deps segmentation-models-pytorch")
import glob, json
import numpy as np, pandas as pd, pydicom
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional, Any
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
print("smp:", smp.__version__, "torch:", torch.__version__)
device = torch.device("cuda")

# 1. per-source pixel comparison for 3 test ids
splits = pd.read_csv(glob.glob("/kaggle/input/**/patient_splits.csv", recursive=True)[0])
te = splits[splits["Fold"].isin(["test", "Test"])].reset_index(drop=True)
for iid in te["ImageId"].iloc[[0, 1, 2]]:
    hits = glob.glob(f"/kaggle/input/**/{iid}.dcm", recursive=True)
    print("ID", iid[:20], "nhits", len(hits))
    for h in hits:
        d = pydicom.dcmread(h); a = d.pixel_array
        print("  ", h.split("/kaggle/input/")[1][:60], a.shape, a.dtype, int(a.min()), int(a.max()),
              getattr(d, "PhotometricInterpretation", "?"))

# 2. original priority indexer
def index_orig():
    m = {}
    for sp in ["/kaggle/input/siim-acr-pneumothorax-segmentation-data",
               "/kaggle/input/siim-acr-pneumothorax-segmentation", "/kaggle/input"]:
        if os.path.exists(sp):
            for root, _, files in os.walk(sp):
                for f in files:
                    if f.endswith(".dcm"):
                        m.setdefault(f[:-4], os.path.join(root, f))
    return m
dmap = index_orig()
print("indexed:", len(dmap), "sample:", list(dmap.values())[0].split("/kaggle/input/")[1][:70])
'''
# NOTE: dataset/model/metrics cells inlined verbatim below (same as P6).


def build(src="kaggle_runner/train_uncertainty_pneumothorax.ipynb",
          path="kaggle_runner/p7_source_test.ipynb"):
    import copy
    srcnb = nbf.read(src, as_version=4)
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    EVAL150 = r'''
# 150-case det inference with ORIGINAL priority mapping
import torch.nn.functional as F
te150 = te.iloc[:150].copy()
te150["dcm_path"] = te150["ImageId"].map(dmap)
print("mapped:", int(te150["dcm_path"].notna().sum()))
ld = DataLoader(SIIMPneumothoraxDataset(te150, image_size=512, transforms=val_transforms),
                batch_size=1, shuffle=False, num_workers=2)
W = os.path.dirname(glob.glob("/kaggle/input/**/model_seed42.pt", recursive=True)[0]) + "/"
net = PneumothoraxUNet(dropout_rate=0.2, pretrained=False).to(device)
net.load_state_dict(torch.load(W + "model_seed42.pt", map_location=device)); net.eval()
ds = []
with torch.no_grad():
    for b in ld:
        p = torch.sigmoid(net(b["image"].to(device))).cpu().numpy()[0, 0]
        m = b["mask"].numpy()[0, 0]
        ds.append(compute_segmentation_metrics(m, p)["dice"])
print("P7 det_dice@150 =", round(float(np.mean(ds)), 4), "(orig first-150 = 0.191)")
'''
    cells = [nbf.v4.new_markdown_cell("# P7: DICOM source-priority test (150 cases)"),
             nbf.v4.new_code_cell(P7)]
    for i in (3, 4, 5, 7):
        c = srcnb.cells[i]
        assert c.cell_type == "code", i
        cells.append(nbf.v4.new_code_cell(c.source))
    cells.append(nbf.v4.new_code_cell(EVAL150))
    nb["cells"] = cells
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build()
