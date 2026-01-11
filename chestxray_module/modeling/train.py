"""
Training script for multi-class chest X-ray classification
Classes: Normal / Pneumonia / TB

Key features:
- Pretrained DenseNet
- MONAI augmentation (train only)
- Weighted CrossEntropyLoss
- AdamW with differential learning rates
- ReduceLROnPlateau scheduler
- Early stopping on validation Macro AUROC
- Checkpointing best model
"""
from chestxray_module.dataset import load_split
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import mlflow
import mlflow.pytorch


# # -----------------------------
# # CONFIGURATION (keep explicit)
# # -----------------------------

# need to parse these arguments for "make train" command if have time later.
NUM_CLASSES = 3
BATCH_SIZE = 32
MAX_EPOCHS = 50

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = f"{os.getcwd()}/models/best_model.pt"
REPORT_PATH = f"{os.getcwd()}/reports/"
CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]

print(f"CHECKPOINT_PATH is {CHECKPOINT_PATH}")
print(f"REPORT_PATH is {REPORT_PATH}")


# -----------------------------
# REPRODUCIBILITY
# -----------------------------

SEED = 42

random.seed(SEED)              
np.random.seed(SEED)         
torch.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # -----------------------------
# # DATA LOADERS
# # -----------------------------



train_dataset = load_split('train')
val_dataset = load_split('val')

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,   # pickling issue with more than 0 workers
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# -----------------------------
# MODEL
# -----------------------------

# Load pretrained DenseNet
model = models.densenet121(weights="IMAGENET1K_V1")

# Replace classifier head for multi-class output
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),           # regularization
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

# -----------------------------
# LOSS FUNCTION
# -----------------------------
# pneumonia gets higher weight because it is under-represented compared to TB and has high cost of misclassification. 
# TB less weight as it is over-represented. normal gets the least weight. 
class_weights = torch.tensor([0.5, 2.0, 1.0], device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -----------------------------
# OPTIMIZER (DIFFERENTIAL LR)
# -----------------------------

optimizer = torch.optim.AdamW(
    [
        {"params": model.features.parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ],
    weight_decay=WEIGHT_DECAY,
)



# -----------------------------
# SCHEDULER
# -----------------------------

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",           # we monitor AUROC (higher is better)
    factor=0.1,
    patience=2,
    #verbose=True,
)

# -----------------------------
# METRIC HELPER FUNCTION
# -----------------------------

def compute_metrics(y_true, y_probs):
    """
    Computes Macro AUROC and Macro F1.
    y_true: (N,)
    y_probs: (N, C)
    """
    auroc = roc_auc_score(
        y_true,
        y_probs,
        multi_class="ovr",
        average="macro"
    )

    y_pred = np.argmax(y_probs, axis=1)
    f1 = f1_score(y_true, y_pred, average="macro")
    
    recall_per_class = recall_score(
    y_true,
    y_pred,
    average=None
    )

    return auroc, f1, recall_per_class



# -----------------------------
# TRAIN / VALIDATION LOOPS
# -----------------------------

def train_one_epoch(model, loader):
    model.train()

    running_loss = 0.0
    all_targets, all_probs = [], []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["class"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)
        all_targets.append(labels.cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    auroc, f1, recall_per_class = compute_metrics(all_targets, all_probs)
    avg_loss = running_loss / len(loader.dataset)

    return avg_loss, auroc, f1, recall_per_class, all_targets, all_probs


@torch.no_grad()
def validate_one_epoch(model, loader):
    model.eval()

    running_loss = 0.0
    all_targets, all_probs = [], []

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["class"].to(DEVICE)
        
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)
        all_targets.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    auroc, f1, recall_per_class = compute_metrics(all_targets, all_probs)
    avg_loss = running_loss / len(loader.dataset)

    return avg_loss, auroc, f1, recall_per_class, all_targets, all_probs


# -----------------------------
# TRAINING LOOP (WITH EARLY STOPPING)
# -----------------------------

# Enable MLflow autologging ONCE (before training)
mlflow.autolog(
    log_models=False  #    log the best model manually at this point
)

best_val_auroc = -np.inf
epochs_without_improvement = 0

with mlflow.start_run(run_name="DenseNet_Multiclass_ChestXray"):

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss, train_auroc, train_f1, train_recall_per_class, _, _ = train_one_epoch(model, train_loader)
        val_loss,   val_auroc,   val_f1,    val_recall_per_class , val_targets, val_probs  = validate_one_epoch(model, val_loader)

        
        mlflow.log_metrics(
            {
                "train_auroc": train_auroc,
                "val_auroc": val_auroc,
                "train_f1": train_f1,
                "val_f1": val_f1
            },
            step=epoch,
        )

        # Per-class recall logging
        for i, class_name in enumerate(CLASS_NAMES):
            mlflow.log_metric(
                f"train_recall_{class_name}",
                train_recall_per_class[i],
                step=epoch,
            )
            mlflow.log_metric(
                f"val_recall_{class_name}",
                val_recall_per_class[i],
                step=epoch,
            )
            
        print(
          f"Epoch {epoch:03d} | "
          f"Train AUROC: {train_auroc:.4f} | "
          f"Val AUROC: {val_auroc:.4f} | "
          f"Train F1: {train_f1:.4f} | "
          f"Val F1: {val_f1:.4f} | "
          f"Val Recall "
          f"Normal={val_recall_per_class[0]:.3f}, "
          f"Pneumonia={val_recall_per_class[1]:.3f}, "
          f"TB={val_recall_per_class[2]:.3f}"
        )

            
        # ---- Scheduler ----
        scheduler.step(val_auroc)

        # Best model selection -> PRIMARY METRIC = val AUROC
        if val_auroc > best_val_auroc + MIN_DELTA:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0

            # Save best model on disk
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, CHECKPOINT_PATH)

            # Log BEST model 
            mlflow.pytorch.log_model(
                model,
                artifact_path="best_model",
            )

            # Log best score
            mlflow.log_metric("best_val_auroc", best_val_auroc)

            print("  ✓ New best model saved")

        else:
            epochs_without_improvement += 1
            print(f"  ✗ No improvement ({epochs_without_improvement})")

        # ---- Early stopping ----
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break


# -----------------------------
# LOAD BEST MODEL
# -----------------------------

model.load_state_dict(torch.load(CHECKPOINT_PATH))
print(f"Best validation AUROC: {best_val_auroc:.4f}")


# -----------------------------
# Reports and figures
# -----------------------------

model.eval()

all_targets = []
all_preds = []

with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(DEVICE)
        labels = batch["class"].to(DEVICE)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

all_targets = np.concatenate(all_targets)
all_preds = np.concatenate(all_preds)

# Heatmap - confusion matrix
cm = confusion_matrix(all_targets, all_preds)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES],
)

cm_df.to_csv(REPORT_PATH + "confusion_matrix.csv")
print(f"Confusion Matrix saved in {os.getcwd()}/reports. \n", cm_df)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Oranges",
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Validation Confusion Matrix")
#plt.show()
plt.savefig("reports/figures/confusion_matrix_heatmap.png", dpi=300)
print(f"Confusion matrix heatmap saved in {os.getcwd()}/reports/figures")

report_dict = classification_report(
    all_targets,
    all_preds,
    target_names=CLASS_NAMES,
    digits=4,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(REPORT_PATH + "metrics_report.csv")
print(f"Classification Report saved in {os.getcwd()}/reports. \n", report_df)






































