# Spam Message Classifier — Results Report

## Objective

Build a machine learning model that identifies SMS messages as spam or ham (legitimate) and compare two classification algorithms.

## Data Preparation

- **Dataset:** Public SMS Spam Collection dataset.
- **Rows after cleaning:** 5,572 *(confirm this against your terminal output before submitting).*
- **Cleaning:** Removed missing labels or messages, duplicate rows, invalid labels, and empty messages.
- **Data split:** 80% training and 20% testing, using stratification and random state 42.

## Text Preprocessing and Features

TF-IDF converts text messages into numerical features for machine learning. The preprocessing lowercases text and removes English stop words. The vectorizer is fitted within each training pipeline to prevent data leakage from the test set.

## Model Comparison

| Model | Accuracy | Spam precision | Spam recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.71% | 100.00% | 74.05% | 85.09% |
| Logistic Regression | 95.65% | 97.78% | 67.18% | 79.64% |

Multinomial Naive Bayes performed better across all four reported metrics on this test set.

## Confusion Matrix

For Multinomial Naive Bayes:

- **True Negatives (TN):** 903
- **False Positives (FP):** 0
- **False Negatives (FN):** 34
- **True Positives (TP):** 97

The model correctly classified 903 legitimate messages and 97 spam messages. It incorrectly classified zero legitimate messages as spam, while 34 spam messages were classified as legitimate.

The confusion matrix visualization is available at `outputs/confusion_matrix.png`.

## Sample Predictions

The following custom messages were tested interactively:

| Message | Prediction |
|---|---|
| Congratulations! You have won a free prize. Click here now! | SPAM |
| Hey, are we meeting for lunch today? | HAM (likely legitimate) |

The actual predictions and additional examples are recorded in `outputs/sample_prediction_logs.csv`. The table above contains custom test examples; retain these labels only if they match your terminal output.

## Conclusion

Multinomial Naive Bayes achieved the best results among the two tested models, with 96.71% accuracy, 100.00% spam precision, 74.05% spam recall, and an 85.09% spam F1-score on the held-out test set.

The model produced zero false positives in this particular test set, but it missed 34 actual spam messages. This shows that high precision does not guarantee that every spam message will be detected.

One limitation is that the dataset consists of SMS messages, so the model may perform differently on emails, social media messages, or newly emerging scam messages.

The project demonstrates how TF-IDF and machine learning can be used for basic spam classification. It is intended for educational purposes and is not a production-ready spam filter.
