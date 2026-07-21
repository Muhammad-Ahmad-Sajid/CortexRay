# CortexRay: Clinical Fracture Detection Platform

CortexRay is an advanced, AI-powered medical imaging platform designed specifically for the detection, classification, and prognostic evaluation of bone fractures in musculoskeletal radiographs (X-rays).

## Project Overview

Musculoskeletal fractures are among the most common injuries requiring medical attention. CortexRay was built to assist radiologists and emergency room physicians by providing a fast, highly accurate, and explainable AI second opinion.

The platform uses a two-stage deep learning pipeline:
1. **Stage 1 (Pretraining):** A ResNet50 model is pretrained on a massive corpus of musculoskeletal radiographs to learn generalized bone features and anomalies.
2. **Stage 2 (Fine-tuning):** The model is then fine-tuned on a targeted dataset of annotated bone fractures to precisely classify X-rays into `fractured` or `not_fractured` categories.

Beyond binary classification, CortexRay includes a **Prognosis Engine** which estimates the severity of the fracture based on confidence scores and region data, generating actionable clinical recommendations. It also integrates **Grad-CAM**, an explainable AI (XAI) technique that produces heatmaps over the original X-ray, highlighting the exact visual regions the AI focused on to make its diagnosis.

## Datasets

CortexRay relies on two major publicly available medical datasets for training its deep learning models:

1. **MURA (Musculoskeletal Radiographs) Dataset**
   - *Description:* A massive dataset of over 40,000 X-ray images of the elbow, finger, forearm, hand, humerus, shoulder, and wrist, provided by the Stanford ML Group. It is used in Stage 1 to teach the model to distinguish between normal and abnormal bone structures.
   - *Link:* [Stanford MURA Dataset](https://stanfordmlgroup.github.io/competitions/mura/)

2. **FracAtlas Dataset**
   - *Description:* A highly curated dataset designed specifically for fracture classification, localization, and segmentation. It contains detailed annotations for thousands of X-rays, which CortexRay uses during Stage 2 fine-tuning to perfect its fracture detection capabilities.
   - *Link:* [FracAtlas Dataset on Figshare](https://figshare.com/articles/dataset/FracAtlas_A_Dataset_for_Fracture_Classification_Localization_and_Segmentation_of_Musculoskeletal_Radiographs/22353277)

## Features Included in this Repository

- **Frontend Application:** Contains the vanilla JavaScript and HTML/CSS web interface for CortexRay.
- **Model Training Pipelines:** Scripts for both Stage 1 and Stage 2 training (`train_stage1.py`, `train_stage2.py`).
- **Data Preparation:** Scripts to parse and organize the MURA and FracAtlas datasets (`prepare_mura.py`, `prepare_fracatlas.py`).
- **Inference Engine:** The core `inference.py` script that loads the trained model and evaluates new patient X-rays.
- **Prognosis & XAI:** The `prognosis_engine.py` generates clinical recommendations, while `gradcam.py` produces visual heatmaps.
- **API & Database:** Fully configured FastAPI backend (`main.py`, `database.py`) with PostgreSQL schemas and models for storing patient records securely.

## Getting Started

1. Install the requirements: `pip install -r requirements.txt`
2. Configure your database in `.env`
3. Download the MURA and FracAtlas datasets using the links above and place them in the appropriate data directories.
4. Run the data preparation scripts.
5. Train the models sequentially (Stage 1, then Stage 2).
6. Start the API server: `uvicorn main:app --reload`
