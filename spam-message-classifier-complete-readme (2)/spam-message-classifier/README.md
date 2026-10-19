# Spam Message Classifier Using Machine Learning

## 1. Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) to classify SMS messages as **Spam** or **Ham (legitimate message)**.

It compares two machine learning algorithms, Multinomial Naive Bayes and Logistic Regression, using TF-IDF text vectorization. The models are evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## 2. Objectives

- Clean and preprocess SMS text data.
- Convert text into numerical features using TF-IDF.
- Train and compare two machine learning models.
- Evaluate model performance using standard classification metrics.
- Predict whether a new SMS message is spam or legitimate.
- Save the trained model pipeline for reuse.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Matplotlib and Seaborn
- Joblib

## 4. Dataset

The project uses a publicly available SMS Spam Collection dataset containing labeled SMS messages.

**Dataset source:** https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

The labels are:
- **Spam:** Unwanted or potentially fraudulent messages.
- **Ham:** Legitimate messages.

The script downloads the dataset automatically if it is not available locally. An internet connection is required for the initial download.

## 5. Project Workflow

1. Load the SMS dataset.
2. Clean the text and remove invalid or duplicate records.
3. Split the dataset into training and testing sets.
4. Convert text into TF-IDF features within each model pipeline.
5. Train Multinomial Naive Bayes and Logistic Regression.
6. Evaluate and compare model performance.
7. Save the selected trained pipeline using Joblib.
8. Predict spam or ham for custom user-entered messages.

The evaluation uses a stratified 80/20 train-test split. The vectorizer is fitted on training data to avoid test-data leakage.

## 6. Model Performance

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.71% | 100.00% | 74.05% | 85.09% |
| Logistic Regression | 95.65% | 97.78% | 67.18% | 79.64% |

**Best-performing model:** Multinomial Naive Bayes achieved the highest scores across all four listed metrics in this evaluation.

Spam precision measures how often messages predicted as spam were actually spam. Spam recall measures how many actual spam messages were detected.

## 7. Confusion Matrix

For the selected Multinomial Naive Bayes model:

- True Negatives (TN): 903
- False Positives (FP): 0
- False Negatives (FN): 34
- True Positives (TP): 97

The model correctly classified 903 legitimate messages and 97 spam messages in the test set. It incorrectly marked zero legitimate messages as spam, while 34 spam messages were classified as legitimate.

See `outputs/confusion_matrix.png` for the visualization.

## 8. Sample Predictions

| Input Message | Expected/Observed Category |
|---|---|
| Congratulations! You have won a free prize. Click here now! | Spam |
| Hey, are we meeting for lunch today? | Ham |

These are example messages tested interactively. The output shown by your actual program should be used as the final evidence.

## 9. Project Structure

```text
spam-message-classifier/
├── spam_classifier.py
├── requirements.txt
├── README.md
├── REPORT_TEMPLATE.md
├── TASK_INSTRUCTIONS.md
├── spam_classifier_pipeline.joblib
└── outputs/
    ├── classification_metrics.csv
    ├── confusion_matrix.png
    └── sample_prediction_logs.csv
```

## 10. How to Run the Project

**Step 1: Install Python**

Install Python 3 and check it in your terminal:

```bash
python --version
```

**Step 2: Install dependencies**

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

**Step 3: Run the classifier**

```bash
python spam_classifier.py
```

**Step 4: Test custom messages**

Enter messages when prompted. Try both a suspicious promotional message and an ordinary personal message. Type `quit` to exit.

The program trains and evaluates the models, creates output files, and saves the selected model pipeline.

## 11. Limitations

- The dataset mainly contains SMS messages and may not represent emails or modern online scams.
- The model can make incorrect predictions.
- High spam precision on this test set does not guarantee the same performance on new messages.
- The project is intended for learning and demonstration, not as a production-ready spam filter.

## 12. Conclusion

This project demonstrates how machine learning and NLP can be used to classify SMS messages. Both Multinomial Naive Bayes and Logistic Regression were evaluated, with Multinomial Naive Bayes performing better on the reported test metrics. The saved pipeline enables predictions on new messages.

## 13. Author

**Srajana Devadiga**

GitHub: https://github.com/srajanadevadiga55/spam-message-classifier
