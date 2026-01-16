"""
dataset.py

Contains helper functions to handle data importing, cleaning, and processing.

Input:
pre-processed data

Used by:
- CLI (make train, make predict and make eval)

IMPORTANT: needs to be ran from the project/repo root.
"""

from pathlib import Path
from torch.utils.data import Dataset
from monai.transforms import Compose, Transform, LoadImage, EnsureChannelFirst, Lambda, RepeatChannel, Resize, SpatialPad, ScaleIntensity, NormalizeIntensity, RandRotate, RandAffine, RandFlip, RandGaussianNoise
from monai.data import Dataset
import pandas as pd
import shutil
import random
import sys
import torch
import cv2
from torch.utils.data import Subset
import numpy as np

def clean_data():
    """
    Removes duplicates and low quality images.
    
    Returns: cleaned data no duplicate images.
    """
    # csv_path contains the paths to all files needs to be delete. Decision were made during EDA. Please see 01_ExploratoryDataAnalysis.
    csv_path = Path("data/raw/paths_to_delete.csv")
    source_dir = Path("data/raw/unzipped_raw_data")
    cleaned_dir = Path("data/interim/cleaned_data")

    # --- Abort if already cleaned ---
    if cleaned_dir.exists():
        print(f"[INFO] {cleaned_dir} already exists.")
        return

    # --- Copy raw data to interim data folder---
    print("Copying raw/unzipped_raw_data → data/interim/cleaned_data ...")
    shutil.copytree(source_dir, cleaned_dir)

    # --- Load paths to delete ---
    df = pd.read_csv(csv_path)

    deleted = 0
    
    # --- Delete files from copied data only ---
    deleted = 0
    for p in df["path"]:
      fp = Path(p.replace("../data/raw/unzipped_raw_data", "data/interim/cleaned_data"))
      if fp.exists():
          fp.unlink()
          deleted += 1

    print(f"Deleted {deleted} files from interim/cleaned_data.")
    
    return clean_data

def data_load(data_dir, recursive=False, inspect=True, n_samples=3):
    """
    Create a MONAI Dataset, ensure channels come first, and optionally inspect sample properties.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing image files.
    inspect : bool
        Whether to print sample dataset information and print information as sanity check.
    n_samples : int
        Number of samples to inspect. samples are chosen randomly.
    recursive: bool
        whether to fetch images in data_dir recursively 

    Returns
    -------
    dataset : monai.data.Dataset
        Lazy-loading MONAI dataset.    
    """
    data_dir = Path(data_dir)
    if recursive:
        image_paths = [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in data_dir.rglob(ext)]
        # image_paths = list(data_dir.rglob("*.jpg"))
    else:
        image_paths = [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in data_dir.glob(ext)] 
        # image_paths = list(data_dir.glob("*.jpg"))

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
    ])

    dataset = Dataset(
        data=image_paths,
        transform=transforms
    )

    # 🔍 Lightweight inspection
    if inspect and len(dataset) > 0:
        print(f"Dataset size: {len(dataset)} images")

        sample_indices = random.sample(
            range(len(dataset)),
            min(n_samples, len(dataset))
        )

        for idx in sample_indices:
            img = dataset[idx]
            print(
                f"Sample {idx}: "
                f"shape={tuple(img.shape)}, "
                f"dtype={img.dtype}, "
                f"min={float(img.min()):.2f}, "
                f"max={float(img.max()):.2f}"
            )

    return dataset

class GCLAHE(Transform):
    """
    Create a Global-CLAHE transformation for image enhancement more suited for medical images.
    Improves local contrast while preserving anatomical detail in medical images.
    GCLACHE is not native to MONAI. So OpenCv is used.
    """
    
    def __init__(self, tile_grid_size=(8, 8), max_clip_limit=3.0):
        self.tile_grid_size = tile_grid_size
        self.max_clip_limit = max_clip_limit

    def __call__(self, img_tensor):
        #  (C, H, W) tensor to (H, W, C) numpy for ease of channel processing now. Will reverse it back later
        img_np = img_tensor.detach().cpu().numpy()
        
        processed_channels = []
        # Process each of the 3 channels individually
        for c in range(img_np.shape[0]):
            # 1. Scale to uint8 for OpenCV
            channel = (img_np[c] * 255).astype(np.uint8)
            
            # 2. Global Reference (GEI)
            gei = cv2.equalizeHist(channel)
            
            # 3. Local Enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=self.max_clip_limit, tileGridSize=self.tile_grid_size)
            lei = clahe.apply(channel)
            
            # 4. G-CLAHE Blending: Maintain global brightness consistency
            # 50/50 blend is common to prevent local artifacts
            g_clahe_channel = cv2.addWeighted(lei, 0.5, gei, 0.5, 0)
            
            processed_channels.append(g_clahe_channel.astype(np.float32) / 255.0)
            
        # back to (3, H, W)
        return torch.from_numpy(np.stack(processed_channels))
    
def transform(dataset, type):
    """
    Return a new transformed Dataset
     a- convert to grayscale by averaging channels since some images have incononsistent values across channels. e.g dominating blue
     b- repeat channels to have 3 channels again.
     c- resize to 224x224
     d- pad to ensure 224x224
     e- scale intensity to [0,1]
     f- apply G-CLAHE Transform class
     g- normalize intensity with ImageNet stats

    Parameters
    ----------
    dataset : monai.data.Dataset
        Existing dataset (paths will be reused).
        
    Type:  str
       If "train" MONAI augmentation is applied

    Returns
    -------
    dataset_transformed : monai.data.Dataset
        New dataset after applying the transformations.
    """
    image_paths = dataset.data  
    
    to_gray = Lambda(lambda x: x.mean(dim=0, keepdim=True))

    base_transforms = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        to_gray,
        RepeatChannel(3),
        Resize((224, 224), mode='bilinear'),
        SpatialPad(spatial_size=(224, 224), method="symmetric"),
        ScaleIntensity(),               # [0,255] → [0,1]
        GCLAHE(tile_grid_size=(8, 8), max_clip_limit=3.0),
        NormalizeIntensity(
        subtrahend =[0.485, 0.456, 0.406],
        divisor=[0.229, 0.224, 0.225],
        channel_wise=True,
        ),
        
    ]
    
    train_augments = [
        
        RandRotate(
            range_x=10 * 3.1416 / 180,   # ±10 degrees
            prob=0.5,
        ),
        RandAffine(
            translate_range=(10, 10),
            scale_range=(0.1, 0.1),
            prob=0.5,
        ),
        RandFlip(
            spatial_axis=1,
            prob=0.5,
        ),
        RandGaussianNoise(
            mean=0.0,
            std=0.01,
            prob=0.3,
        ),
    ]
    
    train_transforms = base_transforms + train_augments
    
    if type == "train":
        dataset_transformed = Dataset(
            data=image_paths,
            transform=train_transforms)
        print(f"Transformation + augmentation done successfully on training data.")
    else: 
        dataset_transformed = Dataset(
        data=image_paths,
        transform=base_transforms
        )
        print(f"Transformation done successfully.")
    
    return dataset_transformed

CLASS_TO_IDX = {
    "normal": 0,
    "pneumonia": 1,
    "tuberculosis": 2,
}

class add_split_class(Dataset):
    
    """
    A custom Dataset wrapper that adds 'class' and 'split' metadata based on file paths without modifying image tensors or transforms.

    """
    
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
            "class": torch.tensor(CLASS_TO_IDX[path.parent.name], dtype=torch.long),    # normal / pneumonia / tuberculosis
            "split": path.parent.parent.name,      # train / val / test
        }
    

def get_split(dataset, split_name):
    """
    Return a subset of the dataset corresponding to the given split.
    """
    indices = [
        i for i, p in enumerate(dataset.paths)
        if Path(p).parent.parent.name == split_name
    ]
    return Subset(dataset, indices)

#Specific for this project/repo
def load_split(split):
    """
    Wrapper function to load and transform a specific split ('train', 'val', or 'test').
    """
    ## This order could be more efficient I will look into it later ##
    clean_data()
    raw_data = data_load(data_dir='data/interim/cleaned_data', recursive=True, inspect=False)
    transformed_data = transform(raw_data, split)
    labeled_data = add_split_class(transformed_data)
    split_data = get_split(labeled_data, split)
    
    print(f"{len(split_data)} images in {split}_data.")
    return split_data


