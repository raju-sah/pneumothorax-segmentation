"""Training pipeline supporting deterministic, MC Dropout, and individual Ensemble runs.

Usage:
    python -m src.train --config configs/baseline.yaml [--seed 42]
    # ensemble: run once per seed -> checkpoints/model_seed{42,43,44}.pt
"""

import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.data.dataset import PneumothoraxDataset, get_training_transforms, get_validation_transforms
from src.models.unet import PneumothoraxUNet
from src.losses.combined import CombinedBCEDiceLoss
from src.metrics.segmentation import compute_dice_coefficient


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * len(images)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dices = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item() * len(images)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = masks.cpu().numpy()

        for p, t in zip(probs, targets):
            d = compute_dice_coefficient((t > 0).astype(np.uint8), (p >= 0.5).astype(np.uint8))
            dices.append(d)

    return running_loss / len(loader.dataset), float(np.mean(dices))


def load_fold_arrays(cache_dir, fold):
    """Load cached (images, masks) for one fold (.h5 preferred, .npz fallback)."""
    h5 = os.path.join(cache_dir, f"fold_{fold}.h5")
    npz = os.path.join(cache_dir, f"fold_{fold}.npz")
    if os.path.exists(h5):
        import h5py
        with h5py.File(h5, "r") as f:
            return np.asarray(f["images"]), np.asarray(f["masks"])
    z = np.load(npz)
    return z["images"], z["masks"]


def main():
    parser = argparse.ArgumentParser(description="Train Pneumothorax Segmentation Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    parser.add_argument("--train_folds", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--val_folds", nargs="+", default=["0"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    if "seeds" in config and args.seed is None:
        seed = config["seeds"][0]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing {config['experiment_name']} on device: {device} with seed: {seed}")

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    cache_dir = config["data"]["cache_dir"]
    tr_x = np.concatenate([load_fold_arrays(cache_dir, f)[0] for f in args.train_folds])
    tr_y = np.concatenate([load_fold_arrays(cache_dir, f)[1] for f in args.train_folds])
    va_x = np.concatenate([load_fold_arrays(cache_dir, f)[0] for f in args.val_folds])
    va_y = np.concatenate([load_fold_arrays(cache_dir, f)[1] for f in args.val_folds])

    train_loader = DataLoader(
        PneumothoraxDataset(tr_x, tr_y, transforms=get_training_transforms(config["data"]["image_size"])),
        batch_size=config["data"]["batch_size"], shuffle=True,
        num_workers=config["data"]["num_workers"])
    val_loader = DataLoader(
        PneumothoraxDataset(va_x, va_y, transforms=get_validation_transforms(config["data"]["image_size"])),
        batch_size=config["data"]["batch_size"], shuffle=False,
        num_workers=config["data"]["num_workers"])

    dropout_rate = config["model"].get("dropout_rate", 0.0)
    model = PneumothoraxUNet(
        dropout_rate=dropout_rate,
        pretrained=config["model"].get("pretrained", True),
    ).to(device)

    criterion = CombinedBCEDiceLoss(
        bce_weight=config["training"]["bce_weight"],
        dice_weight=config["training"]["dice_weight"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    use_amp = config["training"].get("mixed_precision", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_dice, best_path = -1.0, os.path.join(checkpoint_dir, f"model_seed{seed}.pt")
    for epoch in range(1, config["training"]["epochs"] + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va_loss, va_dice = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{config['training']['epochs']} "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_dice={va_dice:.4f}", flush=True)
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), best_path)
    print(f"Best val Dice={best_dice:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
