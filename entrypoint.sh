#!/bin/bash
set -e

# Build baseline artifacts if not present
if [ ! -f "artifacts/models/linear_svc_model.joblib" ]; then
    echo ">>> Running initial end-to-end training pipeline..."
    python pipeline/execute_pipeline.py
fi

echo ">>> Launching Flask API..."
exec python app.py