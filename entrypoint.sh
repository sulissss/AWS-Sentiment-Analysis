#!/bin/bash
set -e # Abort immediately if any stage fails

echo ">>> [1/3] Step 1: Preprocessing & Exploratory Data Analysis..."
python pipeline/1_preprocessing_eda.py

echo ">>> [2/3] Step 2: Feature Engineering & Extraction..."
python pipeline/2_feature_engineering.py

echo ">>> [3/3] Step 3: Model Training, Evaluation & MLflow Logging..."
python pipeline/3_model_training.py

echo ">>> [SUCCESS] Pipeline execution finished! Metrics and artifacts are live on MLflow."