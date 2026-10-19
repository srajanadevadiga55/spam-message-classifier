"""
Spam Message Classifier
- Downloads a public SMS spam dataset
- Splits data into train/test sets before fitting vectorizers
- Compares Multinomial Naive Bayes and Logistic Regression
- Reports precision, recall, F1-score, confusion matrix, and sample predictions
- Saves the best pipeline for interactive testing
"""

from pathlib import Path
import io
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_PATH = ROOT / "SMSSpamCollection.tsv"
MODEL_PATH = ROOT / "spam_classifier_pipeline.joblib"


def load_dataset():
    """Load the public SMS Spam Collection TSV file."""
    if not DATA_PATH.exists():
        print("Downloading the public SMS spam dataset...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        DATA_PATH.write_bytes(response.content)

    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df = df.dropna(subset=["label", "message"]).drop_duplicates()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def main():
    try:
        df = load_dataset()
    except Exception as exc:
        print("\nCould not download/load the dataset.")
        print("Check your internet connection and try again.")
        print(f"Details: {exc}")
        sys.exit(1)

    print("\n=== DATASET SUMMARY ===")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Duplicate rows removed during cleaning: see code's drop_duplicates() step")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"]
    )

    models = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("model", MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("model", LogisticRegression(max_iter=1000, class_weight=None))
        ])
    }

    rows = []
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        fitted_models[name] = pipeline
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "spam_precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
            "spam_recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
            "spam_f1_score": f1_score(y_test, predictions, pos_label=1, zero_division=0)
        })

    results = pd.DataFrame(rows).sort_values(
        by=["spam_precision", "spam_f1_score"], ascending=False
    )
    results.to_csv(OUTPUTS / "classification_metrics.csv", index=False)

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Prioritize spam precision to reduce legitimate messages incorrectly flagged as spam.
    best_name = results.iloc[0]["model"]
    best_model = fitted_models[best_name]
    y_pred = best_model.predict(X_test)

    print(f"\nSelected model (highest spam precision, tie-break by F1): {best_name}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["ham (legitimate)", "spam"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted ham", "Predicted spam"],
        yticklabels=["Actual ham", "Actual spam"]
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=200)
    plt.close()

    # Store example predictions as a small reviewable log.
    examples = pd.DataFrame({
        "message": X_test.reset_index(drop=True).head(20),
        "actual_label": y_test.reset_index(drop=True).head(20).map({0: "ham", 1: "spam"}),
        "predicted_label": pd.Series(best_model.predict(X_test.head(20))).map({0: "ham", 1: "spam"})
    })
    examples.to_csv(OUTPUTS / "sample_prediction_logs.csv", index=False)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved metrics: outputs/classification_metrics.csv")
    print(f"Saved confusion matrix: outputs/confusion_matrix.png")
    print(f"Saved sample logs: outputs/sample_prediction_logs.csv")

    print("\n=== TRY YOUR OWN MESSAGE ===")
    print("Type a message to classify, or type 'quit' to stop.")
    while True:
        try:
            message = input("\nMessage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if message.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not message:
            print("Please enter a non-empty message.")
            continue
        prediction = int(best_model.predict([message])[0])
        label = "SPAM" if prediction == 1 else "HAM (likely legitimate)"
        print(f"Prediction: {label}")
        print("Note: This is a machine-learning prediction, not a guarantee.")


if __name__ == "__main__":
    main()
