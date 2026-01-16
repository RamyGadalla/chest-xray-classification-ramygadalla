# Deep-learning based multi-class classification model for chest X-ray images

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Description

This repository contains an end-to-end chest X-ray classification pipeline (Normal / Pneumonia / Tuberculosis), including training, evaluation, explainability, inference, deployment, and monitoring utilities.

## Dateset

Dataset URL: https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset


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
	  INPUT="" \      # Input images
	  OUTPUT=""       # output destination
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




```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         chestxray_module and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── chestxray_module   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes chestxray_module a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

