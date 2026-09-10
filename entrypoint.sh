#!/bin/bash
set -e

export PYTHONPATH=/app

# Run initial pipeline if baseline model does not exist
if [ ! -f "artifacts/models/linear_svc_model.joblib" ]; then
    echo ">>> Running initial end-to-end training pipeline..."
    python -m pipeline.execute_pipeline
fi

echo ">>> Launching Flask API..."
exec python app.py