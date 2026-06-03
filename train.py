"""
train.py — Transfer learning with EfficientNet-V2-M for CSE 144 final project.
Trains on 100-class image data downloaded via kagglehub.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


# ── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_DIR = "/kaggle/input/competitions/ucsc-cse-144-spring-2026-final-project/train"
CHECKPOINT = "/kaggle/working/best_model.pth"
NUM_CLASSES    = 100
BATCH_SIZE     = 16
NUM_WORKERS    = 4
IMG_SIZE       = 384            # EfficientNet-V2-M native input size

# Phase 1: backbone frozen
FREEZE_EPOCHS  = 10
FREEZE_LR      = 1e-3

# Phase 2: all layers unfrozen
UNFREEZE_EPOCHS = 80
UNFREEZE_LR     = 1e-4

MIXUP_ALPHA    = 0.0            # disabled — over-regularizes when combined with label smoothing + RandomErasing
VAL_SPLIT      = 0.2


# ── Transforms ───────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset helpers ───────────────────────────────────────────────────────────

class NumericalImageFolder(datasets.ImageFolder):
    """ImageFolder that sorts class folders numerically instead of alphabetically.

    Default ImageFolder sorts as strings: "0","1","10","100",...
    Numerical sort gives: "0","1","2",...,"99" — matching the true labels.
    """
    def find_classes(self, directory):
        classes = [
            d.name for d in os.scandir(directory) if d.is_dir()
        ]
        classes.sort(key=lambda x: int(x))
        class_to_idx = {cls: int(cls) for cls in classes}
        return classes, class_to_idx


def build_dataloaders(train_dir: str):
    full_dataset = NumericalImageFolder(train_dir, transform=train_transform)

    indices = list(range(len(full_dataset)))
    labels  = [full_dataset.targets[i] for i in indices]

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=labels,
    )

    val_dataset = NumericalImageFolder(train_dir, transform=val_transform)

    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(val_dataset,  val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples : {len(train_subset)}")
    print(f"Val   samples : {len(val_subset)}")
    print(f"Classes       : {len(full_dataset.classes)}")
    return train_loader, val_loader


# ── MixUp ─────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha: float):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train and MIXUP_ALPHA > 0:
                images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                outputs = model(images)
                loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            # accuracy tracked against the primary label
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(train_dir: str = TRAIN_DIR):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    train_loader, val_loader = build_dataloaders(train_dir)

    model     = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    # ── Phase 1: train head only ──────────────────────────────────────────────
    freeze_backbone(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FREEZE_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)

    print(f"{'='*60}")
    print(f"Phase 1 — frozen backbone ({FREEZE_EPOCHS} epochs, lr={FREEZE_LR})")
    print(f"{'='*60}")

    for epoch in range(1, FREEZE_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{FREEZE_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"           ↑ New best val acc {best_val_acc:.4f} — checkpoint saved")

    # ── Phase 2: fine-tune all layers ─────────────────────────────────────────
    torch.cuda.empty_cache()
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=UNFREEZE_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNFREEZE_EPOCHS)

    print(f"\n{'='*60}")
    print(f"Phase 2 — full fine-tune ({UNFREEZE_EPOCHS} epochs, lr={UNFREEZE_LR})")
    print(f"{'='*60}")

    for epoch in range(1, UNFREEZE_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{UNFREEZE_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"           ↑ New best val acc {best_val_acc:.4f} — checkpoint saved")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    train()
