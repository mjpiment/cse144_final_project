"""
inference.py — Load best checkpoint, run on test images, write submission.csv.

Expected layout:
    test/0.jpg, test/1.jpg, ..., test/999.jpg
    best_model.pth  (produced by train.py)
    sample_submission.csv  (template with ID and Label columns)
"""

import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ── Config (must match train.py) ──────────────────────────────────────────────

TEST_DIR    = "test"
CHECKPOINT  = "best_model.pth"
OUTPUT_CSV  = "submission.csv"
NUM_CLASSES = 100
IMG_SIZE    = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transform (no augmentation — deterministic) ───────────────────────────────

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Model (mirrors train.py exactly) ─────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)   # weights loaded from checkpoint
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
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

    # Load model
    model = build_model(NUM_CLASSES)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # Collect test image paths sorted by numeric ID
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort(key=lambda f: int(os.path.splitext(f)[0]))  # "42.jpg" → 42

    rows = []
    with torch.no_grad():
        for fname in image_files:
            img_id = int(os.path.splitext(fname)[0])
            img_path = os.path.join(test_dir, fname)

            image  = Image.open(img_path).convert("RGB")
            tensor = test_transform(image).unsqueeze(0).to(device)

            logits = model(tensor)
            pred   = logits.argmax(dim=1).item()
            rows.append((img_id, pred))

    # Write submission
    rows.sort(key=lambda r: r[0])   # ensure ascending ID order
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    predict()
