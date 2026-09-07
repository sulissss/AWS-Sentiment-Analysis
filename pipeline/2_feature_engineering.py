import os
from typing import Tuple, List, Optional
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class StructuralFeatureExtractor:
    """Extracts linguistic and structural properties from text."""

    def __init__(self):
        self._ensure_nltk_resources()
        self.vader = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()

    @staticmethod
    def _ensure_nltk_resources():
        """Ensure VADER lexicon is present."""
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

    def extract_metadata(self, raw_series: pd.Series, clean_series: pd.Series) -> pd.DataFrame:
        """Computes statistical and lexical features from both raw and cleaned text."""
        features = pd.DataFrame(index=raw_series.index)

        raw_str = raw_series.fillna("").astype(str)
        clean_str = clean_series.fillna("").astype(str)

        # 1. Structural ratios from raw text (captures emotion lost during cleaning)
        char_len = raw_str.str.len().replace(0, 1)
        features["upper_case_ratio"] = raw_str.apply(lambda x: sum(1 for c in x if c.isupper())) / char_len
        features["exclamation_count"] = raw_str.apply(lambda x: x.count("!"))
        features["question_count"] = raw_str.apply(lambda x: x.count("?"))

        # 2. Token length metrics from clean text
        features["clean_word_count"] = clean_str.str.split().str.len()
        features["clean_char_count"] = clean_str.str.len()
        features["avg_word_len"] = features["clean_char_count"] / features["clean_word_count"].replace(0, 1)

        # 3. Rule-based VADER sentiment polarities
        vader_scores = clean_str.apply(self.vader.polarity_scores)
        features["vader_neg"] = vader_scores.apply(lambda d: d["neg"])
        features["vader_neu"] = vader_scores.apply(lambda d: d["neu"])
        features["vader_pos"] = vader_scores.apply(lambda d: d["pos"])
        features["vader_compound"] = vader_scores.apply(lambda d: d["compound"])

        return features

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scales numeric features on training set."""
        return self.scaler.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Applies fitted scale parameters to test/unseen set."""
        return self.scaler.transform(X)


class RedditFeaturePipeline:
    """Orchestrates TF-IDF extraction, structural feature scaling, and feature matrix assembly."""

    def __init__(
        self,
        max_tfidf_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.test_size = test_size
        self.random_state = random_state

        # Feature engines
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.structural_extractor = StructuralFeatureExtractor()

    def run(
        self,
        df: pd.DataFrame,
        raw_text_col: str,
        clean_text_col: str,
        target_col: str
    ) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
        """
        Splits data, extracts combined features, and prevents data leakage 
        by fitting scalers and vectorizers strictly on training data.
        """
        # Ensure proper data types
        df[raw_text_col] = df[raw_text_col].fillna("").astype(str)
        df[clean_text_col] = df[clean_text_col].fillna("").astype(str)

        # Stratified train/test split
        print(f"Splitting data ({1 - self.test_size:.0%} train, {self.test_size:.0%} test)...")
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[target_col]
        )

        y_train = train_df[target_col].reset_index(drop=True)
        y_test = test_df[target_col].reset_index(drop=True)

        print("1. Extracting TF-IDF text representations...")
        X_train_tfidf = self.tfidf.fit_transform(train_df[clean_text_col])
        X_test_tfidf = self.tfidf.transform(test_df[clean_text_col])

        print("2. Extracting structural and lexical (VADER) properties...")
        train_meta = self.structural_extractor.extract_metadata(train_df[raw_text_col], train_df[clean_text_col])
        test_meta = self.structural_extractor.extract_metadata(test_df[raw_text_col], test_df[clean_text_col])

        print("3. Normalizing continuous numerical features...")
        X_train_meta_scaled = self.structural_extractor.fit_transform(train_meta)
        X_test_meta_scaled = self.structural_extractor.transform(test_meta)

        print("4. Stacking sparse matrices into final training inputs...")
        X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_meta_scaled)]).tocsr()
        X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_meta_scaled)]).tocsr()

        print(f"Feature engineering complete.")
        print(f"Train matrix shape: {X_train_final.shape}")
        print(f"Test matrix shape:  {X_test_final.shape}")

        return X_train_final, X_test_final, y_train, y_test

    def save_artifacts(self, artifact_dir: str = "artifacts"):
        """Saves fitted transformers for model serving in production."""
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.tfidf, os.path.join(artifact_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.structural_extractor.scaler, os.path.join(artifact_dir, "metadata_scaler.joblib"))
        print(f"Transformers exported to directory: '{artifact_dir}/'")


# =====================================================================
# Execution Demonstration
# =====================================================================
if __name__ == "__main__":
    input_file = "../data/cleaned_data.csv"
    output_dir = "artifacts"

    # 1. Load the cleaned CSV
    print(f"Loading cleaned dataset from {input_file}...")
    df_clean = pd.read_csv(input_file)

    # 2. Configure and run feature engineering
    pipeline = RedditFeaturePipeline(
        max_tfidf_features=5000,
        ngram_range=(1, 2),
        test_size=0.2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = pipeline.run(
        df=df_clean,
        raw_text_col="clean_comment",
        clean_text_col="clean_clean_comment",
        target_col="category"
    )

    # 3. Save feature matrices and fitted transformers for the training script
    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_artifacts(output_dir)

    joblib.dump((X_train, y_train), os.path.join(output_dir, "train_data.joblib"))
    joblib.dump((X_test, y_test), os.path.join(output_dir, "test_data.joblib"))
    print(f"Training and testing datasets stored in '{output_dir}/'")