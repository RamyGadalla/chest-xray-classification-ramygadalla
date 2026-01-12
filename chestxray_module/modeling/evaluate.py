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
from sklearn.metrics import recall_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.calibration import calibration_curve, CalibrationDisplay

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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
# Calibration curve
# =============================

all_probs = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(DEVICE)

        # Ensure correct dtype for the model (float)
        if images.dtype != torch.float32:
            images = images.float()

        # labels must be numeric long tensor 
        labels = batch["class"].to(DEVICE).long()

        logits = model(images)                     
        probs = torch.softmax(logits, dim=1)       

        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)       
all_targets = np.concatenate(all_targets, axis=0)   



fig, ax = plt.subplots(figsize=(6, 6))

# Perfect calibration line
ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

for class_idx, class_name in enumerate(CLASS_NAMES):
    # One-vs-rest labels
    y_true_bin = (all_targets == class_idx).astype(int)
    y_prob_bin = all_probs[:, class_idx]

    CalibrationDisplay.from_predictions(
        y_true_bin,
        y_prob_bin,
        n_bins=5,
        strategy="uniform",
        name=f"{class_name} (OvR)",
        ax=ax,
    )

ax.set_title("Reliability Diagram (Test data) - One-vs-Rest")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(fig_dir / "calibration_curve.png")



# =============================
# Threshold analysis - Sensitivity and Specificity
# =============================
PN_IDX = 1
y_true = (all_targets == PN_IDX).astype(int)
y_prob = all_probs[:, PN_IDX]

thresholds = np.linspace(0.0, 1.0, 101)

sensitivities = []
specificities = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    sensitivity = recall_score(y_true, y_pred)  # TP / (TP + FN)
    specificity = recall_score(y_true, y_pred, pos_label=0)  # TN / (TN + FP)

    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)   

axes[0].plot(thresholds, sensitivities, label="Sensitivity (Recall)")
axes[0].plot(thresholds, specificities, label="Specificity")
axes[0].set_xlabel("Probability Threshold")
axes[0].set_ylabel("Score")
axes[0].set_title("Threshold Analysis – Pneumonia")
axes[0].legend()



TB_IDX = 2
y_true = (all_targets == TB_IDX).astype(int)
y_prob = all_probs[:, TB_IDX]

thresholds = np.linspace(0.0, 1.0, 101)

sensitivities = []
specificities = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    sensitivity = recall_score(y_true, y_pred)  # TP / (TP + FN)
    specificity = recall_score(y_true, y_pred, pos_label=0)  # TN / (TN + FP)

    sensitivities.append(sensitivity)
    specificities.append(specificity)


axes[1].plot(thresholds, sensitivities, label="Sensitivity (Recall)")
axes[1].plot(thresholds, specificities, label="Specificity")
axes[1].set_xlabel("Probability Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title("Threshold Analysis – Tuberculosis (Test)")
axes[1].legend()

plt.savefig(fig_dir / "threshold_Recall_specificity_curve.png")


# =============================
# Grad-CAM Visualizations
# =============================

# Last conv layer for DenseNet
target_layers = [model.features.denseblock4]

cam = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers
)


# ----------------------------
# Collect predictions
# ----------------------------
records = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(DEVICE)
        labels = batch["class"].to(DEVICE)
        paths  = batch["path"]

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        for i in range(images.size(0)):
            records.append({
                "image": images[i],
                "true": labels[i].item(),
                "pred": preds[i].item(),
                "prob": probs[i].cpu().numpy(),
                "path": paths[i],
            })


# ----------------------------
# Helper to pick examples
# ----------------------------
def pick_case(true_cls, pred_cls):
    for r in records:
        if r["true"] == true_cls and r["pred"] == pred_cls:
            return r
    return None

CASES = {
    "TB_TP": pick_case(2, 2),
    "TB_FP": pick_case(0, 2),
    "TB_FN": pick_case(2, 0),
    "PNA_TP": pick_case(1, 1),
    "PNA_FP": pick_case(0, 1),
    "PNA_FN": pick_case(1, 0),
}

# ----------------------------
# Run Grad-CAM++
# ----------------------------
for name, case in CASES.items():
    if case is None:
        print(f"[WARN] {name} not found")
        continue

    image = case["image"].unsqueeze(0)
    target_class = case["pred"]

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(image, targets=targets)[0]

    # Convert tensor → numpy image
    img = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    cam_image = np.rot90(cam_image, k=-1)
    cam_image = np.fliplr(cam_image)
    
    original_img = Image.open(case["path"]).convert("RGB")
        
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_img)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(cam_image)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    
    TITLE_MAP = {
    "TB_TP":  "Tuberculosis – True Positive",
    "TB_FP":  "Tuberculosis – False Positive",
    "TB_FN":  "Tuberculosis – False Negative",
    "PNA_TP": "Pneumonia – True Positive",
    "PNA_FP": "Pneumonia – False Positive",
    "PNA_FN": "Pneumonia – False Negative",
}

    fig.suptitle(TITLE_MAP.get(name, name))
    plt.tight_layout()

    # ----------------------------
    # SAVE figure
    # ----------------------------
    #out_path = os.path.join(OUT_DIR, f"{name}_gradcam.png")
 
    gradcam_dir = OUT_DIR / "GradcamMaps"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(gradcam_dir / f"{name}_gradcam.png", dpi=200, bbox_inches="tight")
    plt.close()

print(f"Gradcam SAVED to {gradcam_dir}")
    
