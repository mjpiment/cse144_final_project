"""
inference.py — Load best checkpoint, run on test images with TTA, write submission.csv.

Expected layout (Kaggle):
    /kaggle/input/.../test/0.jpg, 1.jpg, ...
    /kaggle/working/best_model.pth  (produced by train.py)
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


# ── Config (must match train.py) ──────────────────────────────────────────────

TEST_DIR    = "/kaggle/input/competitions/ucsc-cse-144-spring-2026-final-project/test"
CHECKPOINT  = "/kaggle/working/best_model.pth"
OUTPUT_CSV  = "/kaggle/working/submission.csv"
NUM_CLASSES = 100
IMG_SIZE    = 480               # must match train.py

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── TTA transforms: original + 3 flips ───────────────────────────────────────

def _base(extra_ops):
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        *extra_ops,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

tta_transforms = [
    _base([]),                                                          # original
    _base([transforms.RandomHorizontalFlip(p=1.0)]),                   # h-flip
    _base([transforms.RandomVerticalFlip(p=1.0)]),                     # v-flip
    _base([transforms.RandomHorizontalFlip(p=1.0),
           transforms.RandomVerticalFlip(p=1.0)]),                     # both flips
]


# ── Model (mirrors train.py exactly) ─────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_m(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(test_dir: str = TEST_DIR, checkpoint: str = CHECKPOINT, output: str = OUTPUT_CSV):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = build_model(NUM_CLASSES)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"TTA passes: {len(tta_transforms)}")

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    rows = []
    with torch.no_grad():
        for fname in image_files:
            img = Image.open(os.path.join(test_dir, fname)).convert("RGB")

            # average softmax probabilities across all TTA passes
            probs = torch.zeros(NUM_CLASSES, device=device)
            for tfm in tta_transforms:
                tensor = tfm(img).unsqueeze(0).to(device)
                probs += F.softmax(model(tensor).squeeze(0), dim=0)

            pred = probs.argmax().item()
            rows.append((fname, pred))

    rows.sort(key=lambda r: int(os.path.splitext(r[0])[0]))
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    predict()
