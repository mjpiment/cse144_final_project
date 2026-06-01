# CSE 144 Final Project — Transfer Learning on CIFAR-100 Subset

100-class image classification using EfficientNet-B0 fine-tuned on the [UCSC CSE 144 Spring 2026 Kaggle competition](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project) dataset.

## Repository Structure

```
├── train.py               # Training script (two-phase fine-tuning)
├── inference.py           # Inference script (generates submission.csv)
├── cse144_final.ipynb     # Exploratory notebook
├── requirements.txt       # Python dependencies
└── leaderboard.png        # Kaggle leaderboard screenshot (see below)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download the dataset from Kaggle and place it so the directory structure looks like:

```
final_project/
├── train/
│   ├── 0/   (10 images)
│   ├── 1/
│   └── ... (100 classes total)
└── test/
    ├── 0.jpg
    └── ... (1000 images)
```

## Training

```bash
python train.py
```

This runs two phases:
1. **Phase 1 (5 epochs)** — backbone frozen, only the classification head is trained (`lr=1e-3`)
2. **Phase 2 (15 epochs)** — full fine-tuning of all layers (`lr=1e-4`)

The best checkpoint (by validation accuracy) is saved to `best_model.pth`.

Reproducibility is ensured via a fixed seed (`SEED = 42`). Key hyperparameters are at the top of `train.py`.

## Inference

After training, generate `submission.csv` with:

```bash
python inference.py
```

Requires `best_model.pth` and the `test/` directory to be present. Outputs `submission.csv` formatted for Kaggle submission.

## Pretrained Model Weights

Trained weights (`best_model.pth`) are available on Google Drive:

> [Google Drive link — add after uploading weights]

## Kaggle Leaderboard

![Kaggle Leaderboard](leaderboard.png)
