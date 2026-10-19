# Spam Message Classifier — Results Report

> Fill this report after running `python spam_classifier.py`. Copy the real values from `outputs/classification_metrics.csv`; do not guess.

## Objective
Build a text classifier that identifies SMS messages as spam or ham and compare two algorithms.

## Data preparation
- Dataset source: public SMS TSV dataset linked in README.
- Rows after cleaning: **[fill from terminal output]**
- Cleaning: removed missing labels/messages, duplicate rows, invalid labels, and empty messages.
- Split: 80% training / 20% test, stratified, random state 42.

## Text preprocessing and features
TF-IDF converts messages into numeric features. Text is lowercased and English stop words are removed. The vectorizer is fitted within the training pipeline to avoid learning vocabulary from the test set.

## Model comparison

Paste the table from `outputs/classification_metrics.csv` here.

| Model | Accuracy | Spam precision | Spam recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | [fill] | [fill] | [fill] | [fill] |
| Logistic Regression | [fill] | [fill] | [fill] | [fill] |

## Confusion matrix
Insert `outputs/confusion_matrix.png`.

Explain the false-positive count: legitimate messages predicted as spam. This is important because the task asks to minimize false positives.

## Sample predictions
Include examples from `outputs/sample_prediction_logs.csv` and at least two custom messages tested in the terminal.

## Conclusion
State which model performed better on the held-out test set, include its actual precision/recall/F1-score, discuss at least one limitation, and avoid claiming perfect accuracy.
