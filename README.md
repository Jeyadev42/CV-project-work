# Manufacturing Anomaly Detection & Classification

This repository contains code for an anomaly detection and classification system for manufacturing images. The project uses a ResNet-based feature extractor for anomaly detection (via Nearest Neighbors) and a custom convolutional classifier to classify objects into one of the following classes:

- **bottle**
- **capsul**
- **carpet**
- **leather**
- **hazlenut**

The system is designed to work with the MVTec AD dataset (and can be adapted for custom datasets) and includes a Streamlit application for interactive inference.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)

## Overview

This project implements two main components:

1. **Anomaly Detection**  
   A ResNet18-based feature extraction pipeline is trained on good samples from various categories (e.g., bottle and leather) to learn the feature distribution. At test time, the features from a test image are compared to the training features using a Nearest Neighbors model to generate an anomaly score and heatmap.

2. **Classification**  
   A lightweight convolutional classifier is used to predict the category of an input image. The classifier distinguishes between five classes: *bottle, capsul, carpet, leather, hazlenut*.

The project also includes a [Streamlit](https://streamlit.io/) app to interactively upload images and display:
- The predicted category.
- Anomaly heatmaps for selected categories.
- An anomaly score metric.

## Features

- **ResNet-based Feature Extraction** for anomaly detection.
- **Nearest Neighbors Model** to compute patch-wise anomaly distances.
- **Custom CNN Classifier** that categorizes images into five classes.
- **Streamlit Application** for quick demos and interactive testing.
- **Model Saving/Loading** with PyTorch and Joblib for reproducibility.

## Project Structure

```plaintext
CV-PROJECT/
├── Input/                                # Raw input image folders (for training classifier)
│   ├── bottle/
│   ├── capsule/
│   ├── carpet/
│   ├── hazelnut/
│   └── leather/
├── Models/                               # Jupyter notebooks for model experimentation
│   ├── Alternate_test.ipynb
│   └── Model_generator.ipynb
├── Models_dump/                          # Trained models and precomputed features
│   ├── autoencoder_structural.pth        # Autoencoder model for structure-based detection
│   ├── model_bottle.pth                  # Autoencoder model for bottle
│   ├── model_carpet_patch.pth            # Patch-based autoencoder model for carpet
│   ├── product_classifier.pth            # CNN classifier for 5 categories
│   ├── resnet_bottle_nn_model.joblib      # NearestNeighbors model for bottle
│   ├── resnet_bottle_train_features.npy   # ResNet features from "bottle" good training set
│   ├── resnet_leather_nn_model.joblib     # NearestNeighbors model for leather
│   ├── resnet_leather_train_features.npy  # ResNet features from "leather" good training set
│   └── resnet18_imagenet.pth               # ImageNet-pretrained ResNet18 weights
├── Output/                               # Evaluation or summary reports
│   └── anomaly_summary.csv
├── Streamlit_app/                        # Streamlit app code and components
│   ├── __pycache__/                      # Python cache files (auto-generated)
│   ├── app.py                            # Main Streamlit app UI
│   └── model.py                          # Supporting model functions used in Streamlit
├── .gitattributes                        # Git file for handling line endings or diff behavior
└── requirements.txt                      # Python dependencies
