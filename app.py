import os
import threading

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
import pandas as pd

from pipeline.model_evaluation import RedditSentimentPredictor
from pipeline.execute_pipeline import execute_full_pipeline

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_API_SECRET_KEY")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Reddit_Data.csv")

# Initialize inference predictor (handles S3/MLflow/local model loading)
predictor = None
try:
    predictor = RedditSentimentPredictor()
except Exception as e:
    print(f"[Warning] Initial predictor initialization delayed: {e}")

# Global state to prevent concurrent retraining runs
training_state = {
    "is_training": False,
    "last_status": "Idle",
    "error": None
}


def _retraining_worker():
    global training_state, predictor
    training_state["is_training"] = True
    training_state["last_status"] = "Retraining pipeline executing..."
    training_state["error"] = None

    try:
        execute_full_pipeline()
        # Reload predictor with latest model and vectorizers
        predictor = RedditSentimentPredictor()
        training_state["last_status"] = "Completed successfully. Latest model loaded!"
    except Exception as exc:
        training_state["error"] = str(exc)
        training_state["last_status"] = "Training failed."
        print(f"Error during execution: {exc}")
    finally:
        training_state["is_training"] = False


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", training_state=training_state)


@app.route("/predict", methods=["POST"])
def predict():
    global predictor
    try:
        # Check JSON payload vs HTML form submission
        data = request.get_json(silent=True)
        if data and "comments" in data:
            raw_input = data.get("comments", [])
        else:
            text_block = request.form.get("comments_text", "")
            raw_input = [line.strip() for line in text_block.split("\n") if line.strip()]

        if not raw_input:
            return jsonify({"error": "No input comments supplied."}), 400

        if predictor is None:
            predictor = RedditSentimentPredictor()

        predictions = predictor.predict(raw_input)

        if request.is_json:
            return jsonify({"results": predictions})
        return render_template("index.html", predictions=predictions, training_state=training_state)

    except Exception as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 500
        flash(f"Prediction failed: {str(exc)}", "danger")
        return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        flash("No file was uploaded.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".csv"):
        flash("Invalid upload. A .csv file is required.", "danger")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)

        # Exact Header Validation Check: only 'clean_comment' and 'category'
        expected_columns = {"clean_comment", "category"}
        uploaded_columns = set(df.columns.str.strip())
        if uploaded_columns != expected_columns:
            flash(
                f"Validation Error: Columns must be strictly {list(expected_columns)}. "
                f"Received: {list(df.columns)}", "danger"
            )
            return redirect(url_for("index"))

        # Category Validation: strictly values {-1, 0, 1}
        valid_targets = {-1, 0, 1, -1.0, 0.0, 1.0}
        df["category"] = pd.to_numeric(df["category"], errors="coerce")
        if df["category"].isna().any() or not set(df["category"].unique()).issubset(valid_targets):
            flash("Validation Error: 'category' must strictly contain only values -1, 0, or 1.", "danger")
            return redirect(url_for("index"))

        df["category"] = df["category"].astype(int)
        df["clean_comment"] = df["clean_comment"].astype(str)

        # Append valid samples to master data file
        file_exists = os.path.exists(DATA_PATH)
        df.to_csv(DATA_PATH, mode="a", header=not file_exists, index=False)

        flash(f"Validated and appended {len(df)} samples into {DATA_PATH}.", "success")

    except Exception as exc:
        flash(f"File processing failed: {str(exc)}", "danger")

    return redirect(url_for("index"))


@app.route("/train", methods=["POST"])
def train():
    global training_state
    if training_state["is_training"]:
        flash("Retraining is already active in the background.", "warning")
        return redirect(url_for("index"))

    worker = threading.Thread(target=_retraining_worker, daemon=True)
    worker.start()

    flash("Training pipeline launched in background. Monitoring progress...", "info")
    return redirect(url_for("index"))


@app.route("/training-status", methods=["GET"])
def training_status():
    return jsonify(training_state)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)