# CSE 144 Final Project — Transfer Learning

100-class image classification using EfficientNet-B0 fine-tuned on the [UCSC CSE 144 Spring 2026 Kaggle competition](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project) dataset.

## Repository Structure

```
├── train.py                  # Two-phase fine-tuning script
├── inference.py              # Generates submission.csv from trained weights
├── cse144_final.ipynb        # Exploratory notebook
├── sample_submission.csv     # Submission format template
└── requirements.txt          # Python dependencies
```

Model weights are hosted on Google Drive (link below) and are not included in this repo.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project) and extract it into the project root so the structure looks like:

```
final_project/
├── train/
│   ├── 0/      (10 images per class)
│   ├── 1/
│   └── ...     (100 classes total)
└── test/
    ├── 0.jpg
    └── ...     (1036 images)
```

## Training

```bash
python train.py
```

Two-phase fine-tuning with a fixed seed (`SEED = 42`) for reproducibility:

| Phase | Epochs | LR | Backbone |
|-------|--------|-----|----------|
| 1 | 5 | 1e-3 | Frozen |
| 2 | 15 | 1e-4 | Unfrozen |

The best checkpoint by validation accuracy is saved to `best_model.pth`.

## Inference

```bash
python inference.py
```

Loads `best_model.pth` and writes `submission.csv` with predictions for all 1036 test images. The CSV format matches the Kaggle submission template (`ID` = filename, e.g. `0.jpg`).

## Pretrained Model Weights

Trained `best_model.pth` available on Google Drive:

> **[Add Google Drive link here after uploading]**

## Kaggle Leaderboard

![Kaggle Leaderboard](leaderboard.png)
