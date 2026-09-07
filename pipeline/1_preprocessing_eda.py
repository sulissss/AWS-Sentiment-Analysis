import os
import re
from typing import List, Optional
import emoji
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


class RedditTextCleaner:
    """Handles text cleaning and normalization specific to Reddit comment/post conventions."""

    def __init__(self, keep_emojis: bool = True):
        self.keep_emojis = keep_emojis
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _ensure_nltk_resources():
        """Silently verify or download required NLTK corpora."""
        for resource in ["stopwords", "wordnet"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    def clean(self, text: any) -> str:
        """Applies a full cleaning regex pipeline to a single string."""
        if not isinstance(text, str):
            return ""

        # Remove deleted/removed placeholders
        if text.strip() in ["[deleted]", "[removed]"]:
            return ""

        # Demojize before regex stripping (converts :smile: so sentiment tools catch it)
        if self.keep_emojis:
            text = emoji.demojize(text, delimiters=(" ", " "))

        # Reddit-specific markdown & formatting
        text = re.sub(r"\[.*?\]\(.*?\)", " ", text)        # Markdown hyperlinks [text](url)
        text = re.sub(r"http\S+|www\S+", " ", text)         # Plain URLs
        text = re.sub(r"\b[ru]/[A-Za-z0-9_]+\b", " ", text) # Subreddit & user tags
        text = re.sub(r"&[a-z]+;", " ", text)              # HTML entities (&gt;, &amp;)
        text = re.sub(r"<.*?>", " ", text)                 # HTML tags

        # Keep alphanumeric tokens and downcase
        text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()

        # Lemmatize and prune stop words
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in text.split()
            if token not in self.stop_words and len(token) > 2
        ]

        return " ".join(tokens)


class RedditEDAAnalyzer:
    """Produces statistical and visual distributions from preprocessed Reddit data."""

    def __init__(self, data: pd.DataFrame, text_col: str, target_col: str, output_dir: str = "pipeline/eda_plots"):
        self.df = data
        self.text_col = text_col
        self.target_col = target_col
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_class_distribution(self):
        """Displays target class frequency and saves figure."""
        print("\n" + "=" * 50)
        print("TARGET CLASS DISTRIBUTION")
        print("=" * 50)
        distribution = self.df[self.target_col].value_counts(normalize=True) * 100
        print(distribution.round(2).to_string())

        plt.figure(figsize=(7, 4))
        # Fixed the FutureWarning by assigning hue=self.target_col and legend=False
        sns.countplot(
            data=self.df,
            x=self.target_col,
            hue=self.target_col,
            legend=False,
            order=self.df[self.target_col].value_counts().index,
            palette="viridis"
        )
        plt.title("Sentiment Class Balance")
        plt.xlabel("Class")
        plt.ylabel("Observations")
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, "class_distribution.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_length_distributions(self):
        """Plots character and clean word count distributions across sentiment classes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(
            data=self.df,
            x=self.target_col,
            y="clean_word_count",
            hue=self.target_col,
            legend=False,
            palette="Set2",
            showfliers=False,
            ax=axes[0]
        )
        axes[0].set_title("Clean Word Count per Sentiment")
        axes[0].set_ylabel("Word Count")

        sns.kdeplot(
            data=self.df,
            x="char_count",
            hue=self.target_col,
            common_norm=False,
            palette="Set2",
            ax=axes[1]
        )
        axes[1].set_title("Raw Character Density")
        axes[1].set_xlim(0, self.df["char_count"].quantile(0.95))

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "length_distributions.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_wordclouds(self, max_classes: int = 3):
        """Renders lexical word clouds for each unique sentiment category."""
        unique_targets = self.df[self.target_col].dropna().unique()[:max_classes]
        fig, axes = plt.subplots(1, len(unique_targets), figsize=(6 * len(unique_targets), 4))

        if len(unique_targets) == 1:
            axes = [axes]

        for idx, sentiment in enumerate(unique_targets):
            subset = self.df[self.df[self.target_col] == sentiment]
            corpus = " ".join(subset[self.text_col].dropna())

            if not corpus.strip():
                continue

            wc = WordCloud(
                width=500,
                height=350,
                background_color="white",
                max_words=75,
                colormap="Dark2"
            ).generate(corpus)

            axes[idx].imshow(wc, interpolation="bilinear")
            axes[idx].set_title(f"Vocabulary: {sentiment}")
            axes[idx].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "wordclouds.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_metadata_correlation(self, score_col: str = "score"):
        """Correlates post/comment scores with target sentiment if the metric exists."""
        if score_col in self.df.columns:
            plt.figure(figsize=(7, 4))
            sns.barplot(
                data=self.df,
                x=self.target_col,
                y=score_col,
                hue=self.target_col,
                legend=False,
                estimator=np.median,
                errorbar=None,
                palette="mako"
            )
            plt.title(f"Median Upvotes ({score_col}) by Sentiment")
            plt.ylabel("Median Upvotes")
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, "metadata_correlation.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved: {save_path}")
            

class RedditSentimentPipeline:
    """Orchestrator class coordinating end-to-end ingestion, cleaning, EDA, and export."""

    def __init__(
        self,
        raw_text_col: str = "text",
        target_col: str = "sentiment",
        keep_emojis: bool = True
    ):
        self.raw_text_col = raw_text_col
        self.clean_text_col = f"clean_{raw_text_col}"
        self.target_col = target_col
        self.cleaner = RedditTextCleaner(keep_emojis=keep_emojis)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, file_path_or_df) -> "RedditSentimentPipeline":
        """Loads data from a CSV file path or accepts a pre-existing DataFrame."""
        if isinstance(file_path_or_df, str):
            if not os.path.exists(file_path_or_df):
                raise FileNotFoundError(f"Source file not found: {file_path_or_df}")
            self.df = pd.read_csv(file_path_or_df)
        elif isinstance(file_path_or_df, pd.DataFrame):
            self.df = file_path_or_df.copy()
        else:
            raise TypeError("Input must be a valid file path string or pandas DataFrame.")
        
        print(f"Data ingested successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        return self

    def preprocess(self) -> "RedditSentimentPipeline":
        """Executes text normalization, token removal, and metadata feature calculation."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.raw_text_col not in self.df.columns:
            raise KeyError(f"Expected text column '{self.raw_text_col}' was not found in dataset.")

        # 1. Fill NaNs immediately and enforce standard string type
        self.df[self.raw_text_col] = self.df[self.raw_text_col].fillna("").astype(str)

        print("Executing text normalization...")
        self.df[self.clean_text_col] = self.df[self.raw_text_col].apply(self.cleaner.clean)

        # 2. Use native vectorised string operations (avoids len() on float)
        self.df["char_count"] = self.df[self.raw_text_col].str.len()
        self.df["word_count"] = self.df[self.raw_text_col].str.split().str.len().fillna(0).astype(int)
        self.df["clean_word_count"] = self.df[self.clean_text_col].str.split().str.len().fillna(0).astype(int)

        # 3. Drop empty records generated by cleaning
        initial_count = len(self.df)
        self.df = self.df[self.df[self.clean_text_col].str.strip() != ""].reset_index(drop=True)
        dropped_count = initial_count - len(self.df)
        print(f"Preprocessing complete. Dropped {dropped_count} blank/removed rows.")
        return self

    def run_eda(self, score_col: str = "score"):
        """Invokes the exploratory visualization suite."""
        if self.df is None:
            raise ValueError("Data not loaded or processed.")
        
        analyzer = RedditEDAAnalyzer(self.df, self.clean_text_col, self.target_col)
        analyzer.plot_class_distribution()
        analyzer.plot_length_distributions()
        analyzer.plot_wordclouds()
        analyzer.plot_metadata_correlation(score_col=score_col)

    def export(self, output_path: str, columns: Optional[List[str]] = None):
        """Exports the cleaned dataset to CSV."""
        if self.df is None:
            raise ValueError("No data available to export.")
        
        selected_cols = columns or [self.raw_text_col, self.clean_text_col, self.target_col, "clean_word_count"]
        available_cols = [c for c in selected_cols if c in self.df.columns]
        
        self.df[available_cols].to_csv(output_path, index=False)
        print(f"Processed dataset successfully saved to: {output_path}")


# =====================================================================
# Execution Demonstration (Runs automatically when executed as main)
# =====================================================================
if __name__ == "__main__":

    csv_path = 'data/Reddit_Data.csv'

    # 1. Instantiate and run pipeline
    pipeline = RedditSentimentPipeline(
        raw_text_col="clean_comment",
        target_col="category",
        keep_emojis=True
    )

    (
        pipeline
        .load_data(csv_path)
        .preprocess()
    )

    # 2. Trigger EDA charts
    pipeline.run_eda(score_col="score")

    # 3. Save clean data
    pipeline.export("data/cleaned_data.csv")