import os
import joblib
import pandas as pd

from pipeline._1_preprocessing_eda import RedditSentimentPipeline
from pipeline._2_feature_engineering import RedditFeaturePipeline
from pipeline._3_model_training import RedditSVCTrainer


def execute_full_pipeline(
    raw_data_path: str = "data/Reddit_Data.csv",
    cleaned_data_path: str = "data/cleaned_data.csv",
    artifacts_dir: str = "artifacts",
    models_dir: str = "artifacts/models",
    experiment_name: str = "mlflow-sentiment-experiment"
):
    """Executes Stage 1, Stage 2, and Stage 3 sequentially."""
    print("=" * 70)
    print("STARTING FULL ML TRAINING PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Stage 1: Ingestion, Cleaning & EDA
    # -------------------------------------------------------------
    print("\n>>> [1/3] Running Preprocessing & EDA...")
    eda_pipeline = RedditSentimentPipeline(
        raw_text_col="clean_comment",
        target_col="category",
        keep_emojis=True
    )
    (
        eda_pipeline
        .load_data(raw_data_path)
        .preprocess()
        .export(cleaned_data_path)
    )

    # -------------------------------------------------------------
    # Stage 2: Feature Engineering
    # -------------------------------------------------------------
    print("\n>>> [2/3] Running Feature Engineering...")
    df_clean = pd.read_csv(cleaned_data_path)
    feat_pipeline = RedditFeaturePipeline(
        max_tfidf_features=5000,
        ngram_range=(1, 2),
        test_size=0.2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = feat_pipeline.run(
        df=df_clean,
        raw_text_col="clean_comment",
        clean_text_col="clean_clean_comment",
        target_col="category"
    )

    os.makedirs(artifacts_dir, exist_ok=True)
    feat_pipeline.save_artifacts(artifacts_dir)
    joblib.dump((X_train, y_train), os.path.join(artifacts_dir, "train_data.joblib"))
    joblib.dump((X_test, y_test), os.path.join(artifacts_dir, "test_data.joblib"))

    # -------------------------------------------------------------
    # Stage 3: Model Training, Evaluation & MLflow Sync
    # -------------------------------------------------------------
    print("\n>>> [3/3] Training Model, Logging to S3 & Registering to MLflow...")
    trainer = RedditSVCTrainer(
        experiment_name=experiment_name,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        random_state=42
    )
    trainer.train_and_evaluate()
    trainer.register_model_in_registry(
        registered_model_name="reddit-sentiment-classifier",
        alias="champion"
    )
    trainer.save_model()

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE: New model is live on MLflow / S3")
    print("=" * 70)


if __name__ == "__main__":
    execute_full_pipeline()