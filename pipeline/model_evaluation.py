import io
import os
import tarfile
import zipfile
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from scipy.sparse import csr_matrix

from pipeline._1_preprocessing_eda import RedditTextCleaner
from pipeline._2_feature_engineering import RedditFeaturePipeline


class RedditSentimentPredictor:
    """Handles inference by loading models from AWS S3 / MLflow and applying feature pipelines."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        models_dir: str = "artifacts/models",
        s3_bucket: str = "mlflow-bucket-s1808",
        model_s3_key: Optional[str] = None
    ):
        self.artifacts_dir = artifacts_dir
        self.models_dir = models_dir
        self.s3_bucket = s3_bucket
        self.model_s3_key = model_s3_key

        self.cleaner = RedditTextCleaner(keep_emojis=True)
        self.feature_pipeline = RedditFeaturePipeline()
        self.model: Optional[Any] = None

        self.load_transformers()
        self.load_model()

    def load_transformers(self):
        """Loads fitted TF-IDF vectorizer and metadata scaler."""
        tfidf_path = os.path.join(self.artifacts_dir, "tfidf_vectorizer.joblib")
        scaler_path = os.path.join(self.artifacts_dir, "metadata_scaler.joblib")

        if not os.path.exists(tfidf_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Transformers not found in '{self.artifacts_dir}'. Run training first."
            )

        self.feature_pipeline.tfidf = joblib.load(tfidf_path)
        self.feature_pipeline.structural_extractor.scaler = joblib.load(scaler_path)

    def _load_from_mlflow_registry(self) -> bool:
        """Attempts loading the champion model registered in MLflow."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            return False

        try:
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = "models:/reddit-sentiment-classifier@champion"
            print(f"Attempting to fetch model from MLflow Registry: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
            print("Successfully loaded model from MLflow Registry.")
            return True
        except Exception as err:
            print(f"MLflow Registry load failed: {err}")
            return False

    def _load_from_s3_direct(self) -> bool:
        """Downloads the model artifact directly from the MLflow AWS S3 bucket."""
        try:
            s3_client = boto3.client("s3")

            # If explicit key is not provided, look for the latest joblib/model in bucket
            key = self.model_s3_key
            if not key:
                response = s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix="models/")
                contents = response.get("Contents", [])
                if not contents:
                    return False
                # Sort to get latest modified artifact
                sorted_objs = sorted(contents, key=lambda x: x["LastModified"], reverse=True)
                key = sorted_objs[0]["Key"]

            print(f"Downloading model artifact from s3://{self.s3_bucket}/{key}...")
            buffer = io.BytesIO()
            s3_client.download_fileobj(self.s3_bucket, key, buffer)
            buffer.seek(0)

            # Support both direct joblib dumps or archived MLflow model folders
            if key.endswith(".joblib"):
                self.model = joblib.load(buffer)
            else:
                extract_dir = os.path.join(self.models_dir, "s3_downloaded")
                os.makedirs(extract_dir, exist_ok=True)
                try:
                    with tarfile.open(fileobj=buffer, mode="r:*") as tar:
                        tar.extractall(extract_dir)
                    self.model = mlflow.sklearn.load_model(extract_dir)
                except tarfile.ReadError:
                    buffer.seek(0)
                    with zipfile.ZipFile(buffer) as z:
                        z.extractall(extract_dir)
                    self.model = mlflow.sklearn.load_model(extract_dir)

            print("Successfully loaded model from AWS S3.")
            return True
        except (ClientError, Exception) as err:
            print(f"AWS S3 load failed: {err}")
            return False

    def _load_from_local_fallback(self) -> bool:
        """Loads model from the local artifacts directory."""
        local_path = os.path.join(self.models_dir, "linear_svc_model.joblib")
        if os.path.exists(local_path):
            print(f"Loading fallback model from local path: {local_path}")
            self.model = joblib.load(local_path)
            return True
        return False

    def load_model(self):
        """Orchestrates model resolution hierarchy: MLflow Registry -> S3 -> Local disk."""
        if self._load_from_mlflow_registry():
            return
        if self._load_from_s3_direct():
            return
        if self._load_from_local_fallback():
            return
        raise RuntimeError("Could not load model from MLflow, S3, or local storage.")

    def predict(self, raw_comments: List[str]) -> List[Dict[str, Any]]:
        """Cleans input, extracts features, and returns labeled predictions."""
        if not raw_comments:
            return []

        if self.model is None:
            self.load_model()

        # 1. Clean using 1_preprocessing_eda.py logic
        clean_comments = [self.cleaner.clean(text) for text in raw_comments]

        # 2. Extract features using 2_feature_engineering.py
        raw_s = pd.Series(raw_comments).fillna("").astype(str)
        clean_s = pd.Series([c if c.strip() else "neutral" for c in clean_comments])

        # Features
        X_tfidf = self.feature_pipeline.tfidf.transform(clean_s)
        meta = self.feature_pipeline.structural_extractor.extract_metadata(raw_s, clean_s)
        X_meta_scaled = self.feature_pipeline.structural_extractor.transform(meta)
        X_features = pd.concat if False else None  # sparse stack
        from scipy.sparse import hstack
        X_final = hstack([X_tfidf, csr_matrix(X_meta_scaled)]).tocsr()

        # 3. Model prediction
        preds = self.model.predict(X_final)

        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        return [
            {
                "comment": raw,
                "sentiment_code": int(pred),
                "sentiment_label": label_map.get(int(pred), "Unknown")
            }
            for raw, pred in zip(raw_comments, preds)
        ]


if __name__ == "__main__":
    sample_texts = [
        "This project is fantastic and works like a charm! :smile:",
        "Completely useless. It crashed immediately.",
        "It is average, does the job."
    ]
    predictor = RedditSentimentPredictor()
    results = predictor.predict(sample_texts)
    for res in results:
        print(f"[{res['sentiment_label']}] -> {res['comment']}")