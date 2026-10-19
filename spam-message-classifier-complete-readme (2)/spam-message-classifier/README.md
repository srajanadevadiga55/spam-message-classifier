# Spam Message Classifier

A Python Natural Language Processing (NLP) project that classifies SMS messages as **spam** or **ham (likely legitimate)**. It compares Multinomial Naive Bayes and Logistic Regression, evaluates them on a held-out test set, saves a reusable model pipeline, and supports interactive message predictions.

## Project objectives

- Prepare and clean an SMS text dataset.
- Convert text into numerical features using TF-IDF.
- Train and compare two machine-learning classifiers.
- Evaluate results using accuracy, spam precision, spam recall, spam F1-score, and a confusion matrix.
- Test new messages entered by the user.
- Save reproducible outputs for review and GitHub submission.

## Project structure

```text
spam-message-classifier/
├── spam_classifier.py
├── requirements.txt
├── README.md
├── TASK_INSTRUCTIONS.md
├── REPORT_TEMPLATE.md
├── .gitignore
└── outputs/
    ├── classification_metrics.csv       # generated when the script runs
    ├── confusion_matrix.png             # generated when the script runs
    └── sample_prediction_logs.csv       # generated when the script runs
```

The trained model pipeline (`spam_classifier_pipeline.joblib`) and downloaded dataset (`SMSSpamCollection.tsv`) are created when the program runs; they are not expected to exist before the first run.

## Technologies used

- Python
- Pandas
- scikit-learn
- TF-IDF text vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Matplotlib and Seaborn
- Joblib

## Dataset

The script downloads the public SMS dataset from:

https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

The dataset has a message label (`ham` or `spam`) and message text. The script removes rows with missing labels or messages, duplicate rows, unsupported labels, and empty messages. An internet connection is required for the first run.

Please retain the dataset source link when reusing this dataset and check the source repository for applicable terms.

## Setup and run

### 1. Install Python

Use a supported Python 3 version and confirm that it is available in your terminal:

```bash
python --version
```

### 2. Open this folder in VS Code

Open the `spam-message-classifier` folder, then choose **Terminal → New Terminal**.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the classifier

```bash
python spam_classifier.py
```

The first run downloads the dataset, trains both models, prints evaluation results, creates output files, saves the selected pipeline, and prompts you to enter custom messages. Type `quit` to exit interactive testing.

## Model evaluation

The script uses an 80/20 stratified train/test split with a fixed random seed. Each TF-IDF vectorizer is part of its model pipeline, so its vocabulary is fitted only on training messages. The test set is held out for evaluation.

The model selection rule prioritizes **spam precision**, using spam F1-score as a tie-breaker. This is intended to reduce legitimate messages incorrectly marked as spam. It does not guarantee that every message will be classified correctly.

### Actual results

**Run `python spam_classifier.py` first.** The program writes the measured scores to `outputs/classification_metrics.csv`. Copy those real values into this table before final submission; do not invent results.

| Model | Accuracy | Spam precision | Spam recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | To be filled from run | To be filled from run | To be filled from run | To be filled from run |
| Logistic Regression | To be filled from run | To be filled from run | To be filled from run | To be filled from run |

### Evaluation outputs

After a successful run, the following files are generated:

- `outputs/classification_metrics.csv` — comparison of both models.
- `outputs/confusion_matrix.png` — confusion matrix for the selected model.
- `outputs/sample_prediction_logs.csv` — a small sample of held-out test predictions.
- `spam_classifier_pipeline.joblib` — saved fitted TF-IDF and classifier pipeline.

### Metrics explained

- **Accuracy:** fraction of all test messages classified correctly.
- **Spam precision:** of the messages predicted as spam, the fraction that actually are spam.
- **Spam recall:** of all actual spam messages, the fraction detected by the model.
- **Spam F1-score:** harmonic mean of spam precision and spam recall.
- **Confusion matrix:** shows correct and incorrect predictions for each class.

## Sample interactive testing

When prompted with `Message:`, test at least two messages of your own. For example:

1. `Congratulations! You have won a prize. Call now to claim.`
2. `Hey, are we still meeting at 5 pm?`

Record the actual predictions from your run in your report or screenshots. Predictions can vary by training data and model.

## Results and interpretation

After running the project, discuss:
1. Which model achieved the highest spam precision?
2. What were its spam recall and F1-score?
3. How many legitimate messages were incorrectly classified as spam, according to the confusion matrix?
4. What limitations did you notice?

Use measured results rather than assumptions. High precision is useful for reducing false positives, but the project should not be described as perfect or production-ready.

## Limitations

- This is a learning project, not a production spam filter.
- The dataset contains SMS messages; performance may not transfer to emails, social media posts, or newly emerging scams.
- A classifier can make mistakes, so its predictions should not be treated as guarantees.

## Submission checklist

- [ ] Run the script successfully.
- [ ] Review `outputs/classification_metrics.csv`.
- [ ] Review and include `outputs/confusion_matrix.png`.
- [ ] Review `outputs/sample_prediction_logs.csv`.
- [ ] Test at least two custom messages and capture a screenshot of the terminal.
- [ ] Fill the metrics table and results discussion above using the real output.
- [ ] Complete `REPORT_TEMPLATE.md` using the measured results.
- [ ] Upload the Python script, README, requirements, instructions, report, and generated evidence to GitHub.
- [ ] Check that the repository opens correctly and the README is displayed on the repository home page.
