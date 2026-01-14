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
    print("Collecting pixel stats for data monitoring /n")
    means, stds = [], []

    for batch in tqdm(dataloader, desc="Collecting pixel stats"):
        imgs = batch["image"]  # (B, C, H, W)
        imgs = imgs.cpu().numpy()

        batch_means = imgs.mean(axis=(1, 2, 3))
        batch_stds  = imgs.std(axis=(1, 2, 3))

        means.extend(batch_means)
        stds.extend(batch_stds)

    return np.array(means), np.array(stds)



def extract_features(model, dataloader):
    model.eval()
    features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            imgs = batch["image"].to(DEVICE)

            x = model.features(imgs)
            x = torch.relu(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)  # (B, F)

            features.append(x.cpu().numpy())

    return np.vstack(features)





