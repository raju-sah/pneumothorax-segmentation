"""Training pipeline supporting deterministic, MC Dropout, and individual Ensemble runs."""

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


def main():
    parser = argparse.ArgumentParser(description="Train Pneumothorax Segmentation Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing {config['experiment_name']} on device: {device} with seed: {seed}")

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    use_amp = config["training"].get("mixed_precision", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("Model initialized. Ready for dataset mounting.")


if __name__ == "__main__":
    main()
