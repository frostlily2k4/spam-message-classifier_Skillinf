# Spam Message Classifier

## Project Overview
This project uses machine learning to classify SMS messages as spam or non-spam.

## Objective
To train a text-classification model and evaluate its precision, recall, F1-score, and classification errors using unseen test messages.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Machine Learning Techniques
- TF-IDF text feature extraction
- Complement Naive Bayes classification

## Dataset
The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

After cleaning and removing duplicate labeled messages, the dataset contains 5,158 messages.

## Methodology
1. Load and clean labeled SMS messages.
2. Convert text into numerical TF-IDF features.
3. Split the data into training and testing sets.
4. Train the Complement Naive Bayes model.
5. Predict labels for unseen test messages.
6. Evaluate precision, recall, F1-score, and accuracy.
7. Examine the confusion matrix and misclassified messages.

## Results
Test messages: 1,032

- Accuracy: approximately 97%
- Spam precision: 0.90
- Spam recall: 0.88
- Spam F1-score: 0.89

Confusion matrix, with actual labels as rows and predicted labels as columns:

| | Predicted Non-spam | Predicted Spam |
|---|---:|---:|
| Actual Non-spam | 891 | 13 |
| Actual Spam | 15 | 113 |

## Limitations
The model can make incorrect predictions. A false positive may cause a legitimate message to be flagged as spam, while a false negative may allow spam to go undetected. Performance on new messages may differ from the test results.

## How to Run
Install the dependencies:

`python -m pip install pandas scikit-learn matplotlib`

Run:

`python spam_classifier.py`

## Disclaimer
This project is for educational purposes and should not be treated as a perfect spam filter.

## Author
Jothi A