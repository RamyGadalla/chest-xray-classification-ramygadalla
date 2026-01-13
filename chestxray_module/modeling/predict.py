import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from chestxray_module.dataset import data_load, transform
import os
import pandas as pd

# =========================
# Argument parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Chest X-ray inference")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to an image or a directory of images",
    )
    
    parser.add_argument(
    "--output_path",
    type=str,
    default=f"{os.getcwd()}",
    help="Path to save predictions CSV (default: predictions.csv)",
    )
    
    return parser.parse_args()

# Function to enable script to handle single-image and batch inference.
# Could be extended to accept other image formats
def resolve_image_paths(input_path):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    # Case 1: directory 
    if input_path.is_dir():
        return input_path

    # Case 2: single image 
    if input_path.is_file():
        return input_path.parent


# =========================
# Defaults (reviewer-friendly)
# =========================
DEFAULT_CHECKPOINT = f"{os.getcwd()}/models/best_model.pt"
DEFAULT_BATCH_SIZE = 32
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Model builder
# =========================
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, NUM_CLASSES)
)
    return model

# =========================
# Load trained weights
# =========================
def load_model(checkpoint_path: str) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model()
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


# =========================
# Inference / prediction
# =========================
@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader):
    all_preds = []
    all_probs = []
    all_paths = []

    for batch in dataloader:
        images = batch["image"].to(DEVICE)
        paths = batch["path"]

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_paths.extend(paths)

    return all_preds, all_probs, all_paths

# =========================
# Output
# =========================
IDX_TO_CLASS = {
    0: "normal",
    1: "pneumonia",
    2: "tuberculosis",
}

def save_predictions_csv(preds, probs, paths, out_path="predictions.csv"):
    records = []

    for p, prob, path in zip(preds, probs, paths):
        record = {
            "path": str(path),
            "predicted_class": int(p),
            "predicted_class": IDX_TO_CLASS[int(p)],
        }

        # add per-class probabilities
        for i, v in enumerate(prob):
            record[f"prob_{IDX_TO_CLASS[i]}"] = float(v)

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    print(f"[INFO] Predictions saved to: {out_path}")
    
    
# =========================
# Dataset class for easy access with labels
# =========================  
class adjust(Dataset):
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.paths = base_dataset.data  # original image paths

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image = self.base_dataset[idx]  # tensor, unchanged
        path = Path(self.paths[idx])

        return {
            "image": image,
            "path": str(path),

        }
    


# =========================
# Main entry point
# =========================
def main():
    print("[INFO] predict.py started")
    args = parse_args()

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading model from: {args.checkpoint}")

    model = load_model(args.checkpoint)
    
    image_paths = resolve_image_paths(args.input_path)
    
    raw_data = data_load(data_dir=str(image_paths), recursive=False, inspect=False)
    transformed_data = transform(raw_data, "test") # test split transformation. no augmentation.
    adjusted_data = adjust(transformed_data)
    
    input_data = DataLoader(
                    adjusted_data,
                    batch_size = args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    )

    preds, probs, paths = predict(model, input_data)
    save_predictions_csv(preds, probs, paths, out_path=f"{args.output_path}/prediction.csv")
    
    # print the prediction if single image#
    if len(preds) == 1:
      pred_id = int(preds[0])
      pred_label = IDX_TO_CLASS[pred_id]
      prob_vec = probs[0]

      print("\n===== Prediction =====")
      print(f"Image: {paths[0]}")
      print(f"Predicted class: {pred_label}")
      print("Probabilities:")
      for i, p in enumerate(prob_vec):
        print(f"  {IDX_TO_CLASS[i]}: {p:.4f}")
    print("==================\n")
    
    print(f"[DONE] Inference complete on {len(preds)} images /n")
    print(f"Output saved {args.output_path} ")

if __name__ == "__main__":
    main()
