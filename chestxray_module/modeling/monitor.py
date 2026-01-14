import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from chestxray_module.dataset import data_load, transform
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_distances


def collect_pixel_stats(dataloader):
    print("Collecting pixel stats for data monitoring \n")
    means, stds = [], []

    for batch in tqdm(dataloader, desc="Collecting pixel stats"):
        imgs = batch["image"]  # (B, C, H, W)
        imgs = imgs.cpu().numpy()

        batch_means = imgs.mean(axis=(1, 2, 3))
        batch_stds  = imgs.std(axis=(1, 2, 3))

        means.extend(batch_means)
        stds.extend(batch_stds)

    return np.array(means), np.array(stds)



def extract_features_torch(model, dataloader):
    print("Collecting feature stats for data monitoring \n")
    model.eval()
    features = []
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            imgs = batch["image"].to(DEVICE)

            x = model.features(imgs)
            x = torch.relu(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)  # (B, F)

            features.append(x.cpu().numpy())

    return np.vstack(features)



def prediction_entropy(probs):
    eps = 1e-8
    return -np.sum(probs * np.log(probs + eps))

def needs_human_review(prob, cos_dist,
                       conf_thresh=0.6,
                       entropy_thresh=1.0,
                       cos_thresh=0.2):
    confidence = np.max(prob)
    entropy = prediction_entropy(prob)

    reasons = []

    if confidence < conf_thresh:
        reasons.append("low_confidence")

    if entropy > entropy_thresh:
        reasons.append("high_entropy")

    if cos_dist > cos_thresh:
        reasons.append("feature_drift")

    return len(reasons) > 0, reasons





