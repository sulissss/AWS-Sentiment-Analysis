import json
import os
from typing import Any, Dict, Optional, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from dotenv import load_dotenv

load_dotenv()

class ModelEvaluator:
    """Evaluates classification performance and saves diagnostic plots and reports."""

    def __init__(self, output_dir: str = "artifacts/evaluation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="white")

    def evaluate(
        self,
        model_name: str,
        model: Any,
        X_test: csr_matrix,
        y_test: pd.Series
    ) -> Tuple[Dict[str, float], str, str]:
        """Calculates classification metrics, logs to MLflow, and exports artifacts."""
        y_pred = model.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        clf_report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
        clf_report_text = classification_report(y_test, y_pred, digits=4)

        print("\n" + "=" * 60)
        print(f"EVALUATION REPORT: {model_name}")
        print("=" * 60)
        print(f"Macro F1-Score:    {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}\n")
        print(clf_report_text)

        # 1. Save classification report text artifact
        report_path = os.path.join(
            self.output_dir, f"{model_name.lower().replace(' ', '_')}_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(clf_report_text)

        # 2. Confusion Matrix
        labels = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()

        cm_path = os.path.join(
            self.output_dir, f"{model_name.lower().replace(' ', '_')}_cm.png"
        )
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix plot: '{cm_path}'")

        # Compile flat metrics dict for MLflow
        metrics = {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }
        for label, scores in clf_report_dict.items():
            if isinstance(scores, dict):
                metrics[f"class_{label}_f1"] = float(scores["f1-score"])
                metrics[f"class_{label}_precision"] = float(scores["precision"])
                metrics[f"class_{label}_recall"] = float(scores["recall"])

        return metrics, cm_path, report_path


class RedditSVCTrainer:
    """Handles ingestion, training, MLflow tracking, and persistence for LinearSVC."""

    def __init__(
        self,
        experiment_name: str = "mlflow-sentiment-experiment",
        artifacts_dir: str = "artifacts",
        models_dir: str = "artifacts/models",
        random_state: int = 42
    ):
        self.experiment_name = experiment_name
        self.artifacts_dir = artifacts_dir
        self.models_dir = models_dir
        self.random_state = random_state
        self.evaluator = ModelEvaluator(output_dir=os.path.join(artifacts_dir, "evaluation"))

        os.makedirs(self.models_dir, exist_ok=True)

        self.model = LinearSVC(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state
        )
        self.metrics: Optional[Dict[str, float]] = None
        self.active_run_id: Optional[str] = None

    def setup_mlflow(self):
        """Initializes or connects to an existing MLflow experiment."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def load_datasets(self) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
        """Loads serialized feature matrices and labels from previous stage."""
        train_path = os.path.join(self.artifacts_dir, "train_data.joblib")
        test_path = os.path.join(self.artifacts_dir, "test_data.joblib")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Feature matrices not found in '{self.artifacts_dir}'. "
                f"Ensure feature_engineering.py has been executed."
            )

        print(f"Loading datasets from '{self.artifacts_dir}'...")
        X_train, y_train = joblib.load(train_path)
        X_test, y_test = joblib.load(test_path)

        print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self):
        """Fits LinearSVC and tracks parameters, metrics, artifacts, and model in an MLflow run."""
        self.setup_mlflow()
        X_train, X_test, y_train, y_test = self.load_datasets()

        with mlflow.start_run(run_name="LinearSVC-Baseline") as run:
            self.active_run_id = run.info.run_id
            print("\n" + "#" * 60)
            print(f"TRAINING: Linear Support Vector Classifier (LinearSVC) [Run ID: {self.active_run_id}]")
            print("#" * 60)

            # 1. Log Hyperparameters and Dataset Metadata
            params = self.model.get_params()
            params["train_rows"] = X_train.shape[0]
            params["num_features"] = X_train.shape[1]
            mlflow.log_params(params)

            # 2. Fit Model
            self.model.fit(X_train, y_train)

            # 3. Evaluate and Log Metrics
            self.metrics, cm_path, report_path = self.evaluator.evaluate(
                "Linear SVC", self.model, X_test, y_test
            )
            mlflow.log_metrics(self.metrics)

            # 4. Log Evaluation Plots & Text Files
            mlflow.log_artifact(cm_path, artifact_path="evaluation_charts")
            mlflow.log_artifact(report_path, artifact_path="evaluation_reports")

            # 5. Log Model with Signature
            sample_input = X_test[:5]
            sample_pred = self.model.predict(sample_input)
            signature = infer_signature(sample_input, sample_pred)

            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature
            )

            print("\nMLflow run completed and logged successfully.")

    def register_model_in_registry(
        self,
        registered_model_name: str = "reddit-sentiment-classifier",
        alias: Optional[str] = "champion"
    ):
        """
        Registers the model from the current MLflow run into the MLflow Model Registry.
        Optionally sets an alias (e.g., 'champion' or 'staging') to the registered version.
        """
        if self.active_run_id is None:
            raise ValueError("No active run found. Execute train_and_evaluate() first.")

        model_uri = f"runs:/{self.active_run_id}/model"
        print(f"\nRegistering model from URI '{model_uri}' to Model Registry as '{registered_model_name}'...")

        # Register the model version in the Model Registry
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )
        print(f"Successfully registered '{registered_model_name}' (Version: {model_version.version})")

        # Set alias (champion/staging) so downstream services can pull by alias
        if alias:
            client = MlflowClient()
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=alias,
                version=model_version.version
            )
            print(f"Assigned alias '@{alias}' to version {model_version.version}")

    def save_model(self):
        """Saves local joblib artifacts alongside MLflow for local pipelines."""
        if self.metrics is None:
            raise ValueError("Model has not been trained. Run train_and_evaluate() first.")

        model_file = os.path.join(self.models_dir, "linear_svc_model.joblib")
        joblib.dump(self.model, model_file)

        metadata = {
            "model_type": "LinearSVC",
            "hyperparameters": self.model.get_params(),
            "evaluation_metrics": self.metrics
        }
        metadata_file = os.path.join(self.models_dir, "linear_svc_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Local model backup saved to: '{model_file}'")
        print(f"Local metadata written to:   '{metadata_file}'")


# =====================================================================
# Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    trainer = RedditSVCTrainer(
        experiment_name="mlflow-sentiment-experiment",
        artifacts_dir="artifacts",
        models_dir="artifacts/models",
        random_state=42
    )

    trainer.train_and_evaluate()
    trainer.register_model_in_registry(
        registered_model_name="reddit-sentiment-classifier",
        alias="champion"
    )
    trainer.save_model()