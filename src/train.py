"""
Training pipeline for `ArtistStyleNet`.
Trains the hybrid CNN + Gram-matrix style architecture on the curated WikiArt dataset
using mixed precision (AMP), differential learning rates, cosine annealing, and label smoothing.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data_fetcher import fetch_and_prepare_dataset
from src.model import ArtistStyleNet, PaintingDataset, get_transforms


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_split(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds_top3 = logits.topk(k=min(3, logits.size(1)), dim=1).indices
        correct_top1 += (preds_top3[:, 0] == labels).sum().item()
        correct_top3 += (preds_top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "top1_acc": correct_top1 / max(total, 1),
        "top3_acc": correct_top3 / max(total, 1),
    }


def train_model(
    data_dir: str = "data",
    output_dir: str = "outputs",
    epochs: int = 12,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
):
    seed_everything(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Ensure dataset is fetched and prepared
    df = fetch_and_prepare_dataset(data_dir=data_dir, seed=seed)
    with open(Path(data_dir) / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_artist = {int(k): v for k, v in label_map["idx_to_artist"].items()}
    num_classes = len(idx_to_artist)

    train_tf, eval_tf = get_transforms(image_size=image_size)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = PaintingDataset(train_df, root_dir=".", transform=train_tf)
    val_ds = PaintingDataset(val_df, root_dir=".", transform=eval_tf)
    test_ds = PaintingDataset(test_df, root_dir=".", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Training on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"[Trainer] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Artists: {num_classes}")

    model = ArtistStyleNet(num_classes=num_classes, pretrained=True).to(device)
    model.freeze_early_layers(freeze_through_layer2=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    # Differential learning rates: lower LR for pretrained ResNet layers, higher LR for style/Gram heads
    backbone_params = (
        list(model.layer2.parameters())
        + list(model.layer3.parameters())
        + list(model.layer4.parameters())
    )
    head_params = (
        list(model.gram_proj.parameters())
        + list(model.semantic_proj.parameters())
        + list(model.style_head.parameters())
        + list(model.classifier.parameters())
    )

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": 1.8e-4, "weight_decay": 1e-3},
            {"params": head_params, "lr": 8e-4, "weight_decay": 1e-3},
        ]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    history = []
    best_val_acc = 0.0
    best_ckpt_path = out_path / "best_model.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Unfreeze layer2 after warmup epoch 2 for richer mid-level Gram style learning
        if epoch == 3:
            model.unfreeze_upper_backbone()

        model.train()
        running_loss = 0.0
        correct_top1 = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = correct_top1 / max(total, 1)

        val_metrics = evaluate_split(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_top1_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_top1_acc": round(val_metrics["top1_acc"], 4),
            "val_top3_acc": round(val_metrics["top3_acc"], 4),
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(epoch_record)

        marker = ""
        if val_metrics["top1_acc"] >= best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "idx_to_artist": idx_to_artist,
                    "image_size": image_size,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )
            marker = " *BEST*"

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Top-1: {val_metrics['top1_acc']*100:.1f}% "
            f"Top-3: {val_metrics['top3_acc']*100:.1f}%{marker}"
        )

    with open(out_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best checkpoint and evaluate on held-out Test set
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_split(model, test_loader, criterion, device)
    print(
        f"\n[Trainer] Best Model Held-Out Test Performance -> "
        f"Top-1 Accuracy: {test_metrics['top1_acc']*100:.2f}% | "
        f"Top-3 Accuracy: {test_metrics['top3_acc']*100:.2f}%"
    )
    return model


if __name__ == "__main__":
    train_model()

