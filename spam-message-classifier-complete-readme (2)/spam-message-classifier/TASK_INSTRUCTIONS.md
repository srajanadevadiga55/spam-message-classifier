# Task 2 Instructions — Spam Message Classifier

## Objective
Train a text classifier to identify spam and legitimate (ham) messages. Evaluate it using precision, recall, and F1-score, and demonstrate sample predictions.

## Required features
1. **Text preprocessing:** tokenization/vectorization, lowercasing, stop-word removal, and TF-IDF or Count Vectorization.
2. **Train and compare models:** for example, Multinomial Naive Bayes versus Logistic Regression.
3. **Evaluate carefully:** report precision, recall, F1-score, and a confusion matrix.
4. **Interactive prediction:** let a user type a custom message and receive a prediction.

## Required tools / recommended stack
- Python
- Scikit-learn
- NLTK or regular expressions (optional for additional preprocessing)
- Pandas
- Matplotlib / Seaborn

## Required deliverables
- Complete model training script or notebook.
- Saved vectorizer and model pipeline (for this project, saved using Joblib).
- README.md describing the project, setup, classification metrics table, and sample prediction logs.
- Demo or screenshots where applicable.

## Acceptance criteria
- Aim for high spam precision to reduce false positives (legitimate messages incorrectly flagged as spam).
- Evaluate on held-out test data. Do not fit the vectorizer on the test data or use test data during training.
- Interactive testing must demonstrate inference on new user-entered messages.

## How to complete this project
1. Open the project folder in VS Code.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Run `python spam_classifier.py`.
4. Allow the script to download the public SMS dataset on the first run.
5. Review the real metrics printed in the terminal and saved to `outputs/classification_metrics.csv`.
6. Check `outputs/confusion_matrix.png` and `outputs/sample_prediction_logs.csv`.
7. Test at least two messages you type yourself.
8. Take a screenshot showing the terminal and predictions.
9. Push the source code, README, requirements, and output evidence to GitHub.
10. Fill `REPORT_TEMPLATE.md` using actual run results; do not make up metric values.

## Important
- Use free tools only.
- Dataset download requires an internet connection the first time.
- A machine-learning prediction is not a guarantee. The project is a learning demo, not a production spam filter.
- Review the repository's dataset source and cite it in the README.
