"""Cache DICOM radiographs + RLE masks to compressed HDF5 (.h5) per fold.

Usage:
    python -m src.data.cache_hdf5 --split_csv data/processed/patient_splits.csv \\
        --metadata data/processed/metadata_master.csv --dicom_root data/raw/dicom-images-train \\
        --output_dir data/processed/cache_512x512/ [--size 512]

Reads the split manifest (ImageId, Fold, EncodedPixelsList as JSON list),
decodes each DICOM (per-image min-max, uint8), resizes to --size, ORs the
RLE mask list, and writes one compressed .h5 per Fold with
images (N,H,W uint8), masks (N,H,W uint8), image_ids (str).
"""
import argparse
import ast
import glob
import json
import os

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from src.utils.rle import rle_decode


def find_dicoms(dicom_root):
    paths = glob.glob(os.path.join(dicom_root, "**", "*.dcm"), recursive=True)
    return {os.path.basename(p).replace(".dcm", ""): p for p in paths}


def decode_mask_list(cell, shape=(1024, 1024)):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.zeros(shape, np.uint8)
    items = cell if isinstance(cell, list) else ast.literal_eval(str(cell))
    if isinstance(items, str):
        items = [items]
    m = np.zeros(shape, np.uint8)
    for rle in items:
        if str(rle).strip() in ("-1", ""):
            continue
        m = np.maximum(m, rle_decode(rle, shape))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--dicom_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    splits = pd.read_csv(args.split_csv)
    meta = pd.read_csv(args.metadata).set_index("ImageId")["EncodedPixelsList"].to_dict() if args.metadata else {}
    dmap = find_dicoms(args.dicom_root)
    print(f"splits={len(splits)} dicoms={len(dmap)}")
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import h5py
        use_h5 = True
    except ImportError:
        use_h5 = False

    for fold, grp in splits.groupby("Fold"):
        imgs, masks, ids, missing = [], [], [], 0
        for _, row in grp.iterrows():
            iid = str(row["ImageId"])
            p = dmap.get(iid)
            if p is None:
                missing += 1
                continue
            px = pydicom.dcmread(p).pixel_array.astype(np.float32)
            px = (px - px.min()) / (px.max() - px.min() + 1e-8)
            img = np.asarray(Image.fromarray((px * 255).astype(np.uint8)).resize(
                (args.size, args.size), Image.BILINEAR))
            cell = meta.get(iid, row.get("EncodedPixelsList", "-1"))
            m = decode_mask_list(cell)
            m = np.asarray(Image.fromarray(m).resize((args.size, args.size), Image.NEAREST),
                           np.uint8)
            imgs.append(img)
            masks.append(m)
            ids.append(iid)
        imgs = np.stack(imgs).astype(np.uint8)
        masks = np.stack(masks).astype(np.uint8)
        out = os.path.join(args.output_dir, f"fold_{fold}.h5" if use_h5 else f"fold_{fold}.npz")
        if use_h5:
            import h5py
            with h5py.File(out, "w") as f:
                f.create_dataset("images", data=imgs, compression="gzip")
                f.create_dataset("masks", data=masks, compression="gzip")
                f.create_dataset("image_ids", data=np.array(ids, dtype="S"))
        else:
            np.savez_compressed(out, images=imgs, masks=masks, image_ids=np.array(ids))
        print(f"fold={fold} n={len(ids)} missing={missing} -> {out}")


if __name__ == "__main__":
    main()
