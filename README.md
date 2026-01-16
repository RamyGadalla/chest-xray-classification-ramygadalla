# Deep-learning based multi-class classification model for chest X-ray images

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Description

This repository contains an end-to-end chest X-ray classification pipeline (Normal / Pneumonia / Tuberculosis), including training, evaluation, explainability, inference, deployment, and monitoring utilities.

## Dateset

Dataset URL: https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset

This dataset merges multiple public chest X-ray sources into one classification set with three classes:

1-Normal  
2-Pneumonia  
3-Tuberculosis 


## Quick Start

```
git clone https://github.com/<your-username>/chestxray.git
cd chestxray
``` 
If Kaggle API token is configured on your system, the dataset can be downloaded using:
```
python download_data.py  # download dataset in data/raw/unzipped_raw_data
```
Run inference in Docker container from repo root directory
```
docker build -t chestxray:latest

sudo docker run --rm \
	  -v "$(pwd):/app/chestxray" \
	  -w /app/chestxray \
	  chestxray:latest \
	  make predict \
	  INPUT="" \      # Input images path
	  OUTPUT=""       # output destination path
```
## Project Organization

```
├── chestxray_module                    
│   ├── dataset.py
│   ├── download_data.py
│   ├── extract_reference_stats.py
│   ├── lung_segment_model.py
│   └── modeling                       
├── data
├── Dockerfile
├── environment.yml
├── Makefile
├── models                              
│   ├── best_model.onnx
│   ├── best_model.onnx.data
│   └── best_model.pt
├── notebooks                            
│   ├── 01_ExploratoryDataAnaylsis.ipynb
│   ├── 02_Preprocess.ipynb
│   ├── evaluate_dev.ipynb
│   ├── monitoring.ipynb
│   ├── monitor.py
│   ├── predict.ipynb
│   ├── segmentation_dev.ipynb
│   ├── training_dev.ipynb
│   └── unet-6v.pt
├── pyproject.toml
├── README.md
├── references
│   ├── feature_centroid.npy
│   ├── feature_cov.npy
│   ├── pixel_histograms.npz
│   └── pixel_stats.npz
├── reports
│   ├── best_model
│   ├── inference
│   └── model_with_duplicates_images
└── requirements.txt
```






--------

