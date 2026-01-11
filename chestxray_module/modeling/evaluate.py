"""
Final evaluation script for the trained chest X-ray model.

Purpose:
- Evaluate the frozen best model on the TEST split
- Compute final metrics (AUROC, F1, Recall)
- Save confusion matrix and classification report

This script performs NO training.
Run once, after training is complete.

"""

from chestxray_module.dataset import load_split
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# =============================
# Configuration 
# =============================

SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_WORKERS = 0  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = f"{os.getcwd()}/models/best_model.pt"
OUT_DIR = Path(f"{os.getcwd()}/reports/best_model/eval_reports/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]


# =============================
# Reproducibility
# =============================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================
# Dataset
# =============================
 
test_dataset = load_split("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# =============================
# Model
# =============================

model = models.densenet121(weights=None)
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, NUM_CLASSES)
)


model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =============================
# Evaluation loop
# =============================
all_targets = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(DEVICE)
        labels = batch["class"].to(DEVICE)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_targets.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_targets = np.concatenate(all_targets)
all_probs = np.concatenate(all_probs)
all_preds = np.argmax(all_probs, axis=1)

# =============================
# Metrics
# =============================
# could call helper functions from training script, but need to restructure train.py to allow that
auroc = roc_auc_score(
    all_targets,
    all_probs,
    multi_class="ovr",
    average="macro",
)

f1 = f1_score(all_targets, all_preds, average="macro")
recall_per_class = recall_score(all_targets, all_preds, average=None)

metrics = {
    "auroc_macro": float(auroc),
    "f1_macro": float(f1),
    "recall_normal": float(recall_per_class[0]),
    "recall_pneumonia": float(recall_per_class[1]),
    "recall_tuberculosis": float(recall_per_class[2]),
}

rows = []

for key, value in metrics.items():
    # Case 1: scalar metric
    if np.isscalar(value):
        rows.append({
            "metric": key,
            "class": "all",
            "value": value
        })

    # Case 2: per-class metric (list / array)
    else:
        for i, v in enumerate(value):
            rows.append({
                "metric": key,
                "class": CLASS_NAMES[i],
                "value": float(v)
            })

df_metrics = pd.DataFrame(rows)

print("Final TEST evaluation:")
print(f"  AUROC (macro): {auroc:.4f}")
print(f"  F1 (macro):    {f1:.4f}")
print("  Recall per class:")
for cls, r in zip(CLASS_NAMES, recall_per_class):
    print(f"    {cls}: {r:.4f}")


df_metrics.to_csv(OUT_DIR / "metrics_report.csv", index=False)
print(f"\nmetrics_report.csv saved to: {OUT_DIR.resolve()}")


cm = confusion_matrix(all_targets, all_preds)
import pandas as pd

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES],
)

cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")
print(f"\nconfusion matrix.csv saved to: {OUT_DIR.resolve()}")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Test Confusion Matrix")
plt.tight_layout()
fig_dir = OUT_DIR / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "confusion_matrix.png", dpi=300)
plt.close()
print(f"\nconfusion matrix heatmap saved to: {fig_dir.resolve()}")


report_dict = classification_report(
    all_targets,
    all_preds,
    target_names=CLASS_NAMES,
    digits=4,
    output_dict=True,
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(OUT_DIR / "classification_report.csv")
print(f"\n classification report.csv saved to: {OUT_DIR.resolve()}")


# =============================