
"""
extract_reference_stats.py

Extract reference stats (training dataset) on pixel and feature level.
These stats will be used as a reference to monitor data drift and model performance during inference.

Input:
- model (pytorch)
- dataset

Used by:
- CLI (make predict)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from chestxray_module.dataset import data_load, transform
from chestxray_module.modeling.predict import adjust
import torch.nn as nn
from torchvision import models

# -----------------------------
#        Variables
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REFERENCE_DIR = Path("references")
REFERENCE_DIR.mkdir(exist_ok=True)
NUM_CLASSES = 3 


# -----------------------------
#      Feature extractor
# -----------------------------
def get_feature_extractor(model):
    """
    Removes classifier head, returns penultimate features.
    """
    model.eval()
    return torch.nn.Sequential(
        model.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
    ).to(DEVICE)


# -----------------------------
#   Stats builder function
# -----------------------------
def build_reference_stats(model, train_dataset):

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
)
    feature_extractor = get_feature_extractor(model)

    # ---- accumulators ----
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    hist_bins = 256
    pixel_hist = np.zeros((3, hist_bins))

    features = []
    image_means = []
    image_stds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building reference stats"):
            images = batch["image"].to(DEVICE)  # (B, C, H, W)

            # ---------------------
            # Pixel statistics
            # ---------------------
            pixel_sum += images.sum(dim=[0, 2, 3])
            pixel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
            pixel_count += images.numel() / images.shape[1]

            imgs_np = images.cpu().numpy()
            for c in range(3):        # per chan
                pixel_hist[c] += np.histogram(
                    imgs_np[:, c, :, :].ravel(),
                    bins=hist_bins,
                    range=(-3, 3),
                )[0]
            
            batch_means = imgs_np.mean(axis=(1, 2, 3))
            batch_stds  = imgs_np.std(axis=(1, 2, 3))

            image_means.extend(batch_means)
            image_stds.extend(batch_stds)
            
            # ---------------------
            # Feature statistics
            # ---------------------
            feats = feature_extractor(images)
            features.append(feats.cpu().numpy())

    # ---- finalize pixel stats ----
    pixel_mean = pixel_sum / pixel_count
    pixel_var  = pixel_sq_sum / pixel_count - pixel_mean ** 2
    pixel_std  = torch.sqrt(pixel_var)
    
    pixel_mean = pixel_mean.cpu().numpy()
    pixel_std  = pixel_std.cpu().numpy()


    # ---- finalize features ----
    features = np.concatenate(features, axis=0)
    feature_centroid = features.mean(axis=0, keepdims=True)
    feature_cov = np.cov(features, rowvar=False)

    # -----------------------------
    #  Save artifacts
    # -----------------------------
    np.savez(
        REFERENCE_DIR / "pixel_stats.npz",
        image_means=np.array(image_means),
        image_stds=np.array(image_stds),
        channel_mean=pixel_mean,
        channel_std=pixel_std,
    )

    np.savez(
        REFERENCE_DIR / "pixel_histograms.npz",
        hist=pixel_hist,
    )

    np.save(REFERENCE_DIR / "feature_centroid.npy", feature_centroid)
    np.save(REFERENCE_DIR / "feature_cov.npy", feature_cov)

    print("\n[INFO] Reference statistics saved to ./reference/")
    print(" - pixel_stats.npz")
    print(" - pixel_histograms.npz")
    print(" - feature_centroid.npy")
    print(" - feature_cov.npy")



if __name__ == "__main__":
    
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(in_features, NUM_CLASSES)
    )
    
    model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    image_paths = "data/interim/cleaned_data/train"
    raw_data = data_load(data_dir=image_paths, recursive=True, inspect=False)
    transformed_data = transform(raw_data, "test") # Test-type split transformation. No augmentation.
    train_dataset = adjust(transformed_data)
    
    build_reference_stats(model, train_dataset)
