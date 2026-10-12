
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# 1. Load the labeled SMS dataset
file_path = Path("SMSSpamCollection/SMSSpamCollection")

if not file_path.exists():
    raise FileNotFoundError(
        "Place SMSSpamCollection in the project folder."
    )

data = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# 2. Clean and prepare the data
data = data.dropna()
data["label"] = data["label"].str.strip().str.lower()
data["message"] = data["message"].astype(str).str.strip()
data = data[
    data["label"].isin(["ham", "spam"])
    & data["message"].ne("")
]
data = data.drop_duplicates(subset=["label", "message"])

# Save a clean copy
data.to_csv("spam_dataset.csv", index=False)

print("DATASET OVERVIEW")
print("=" * 45)
print("Total messages:", len(data))
print("\nLabel distribution:")
print(data["label"].value_counts())

# 3. Separate messages and labels
X = data["message"]
y = data["label"]

# 4. Split into training and unseen test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Build the text-classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", ComplementNB())
])

# 6. Train the model
model.fit(X_train, y_train)

# 7. Evaluate on unseen test messages
y_pred = model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print("=" * 45)
print(classification_report(
    y_test,
    y_pred,
    labels=["ham", "spam"],
    target_names=["Non-spam", "Spam"],
    zero_division=0
))

# 8. Display the confusion matrix
cm = confusion_matrix(
    y_test, y_pred, labels=["ham", "spam"]
)

print("CONFUSION MATRIX")
print("Rows = actual, columns = predicted")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-spam", "Spam"]
)
display.plot()
plt.title("Spam Message Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 9. Show misclassified messages
errors = pd.DataFrame({
    "Message": X_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

errors = errors[errors["Actual"] != errors["Predicted"]]

print("\nMISCLASSIFIED TEST MESSAGES")
print("=" * 45)

if errors.empty:
    print("No misclassified messages in this test split.")
else:
    print(errors.to_string(index=False, max_rows=20))

errors.to_csv("misclassified_messages.csv", index=False)

# 10. Test custom messages
custom_messages = [
    "Congratulations! You won a free prize. Claim it now!",
    "Hi, are we meeting at the library this afternoon?",
    "URGENT! Claim your cash reward by clicking this link.",
    "Please send me the class notes when you have time."
]

print("\nCUSTOM MESSAGE PREDICTIONS")
print("=" * 45)

for message, prediction in zip(
    custom_messages, model.predict(custom_messages)
):
    result = "SPAM" if prediction == "spam" else "NON-SPAM"
    print(f"\nMessage: {message}")
    print(f"Prediction: {result}")

print("\nNote: Predictions can be wrong.")
print("Do not treat this classifier as a perfect spam filter.")
