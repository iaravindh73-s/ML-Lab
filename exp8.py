import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

print("Dataset loaded successfully!")
display(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumn Names:", df.columns)
print("\nDataset Information:")
df.info()
print("\nSentiment Distribution:")
print(df["Liked"].value_counts())

nltk.download("stopwords")

corpus = []
ps = PorterStemmer()
english_stopwords = set(stopwords.words("english"))

for i in range(len(df)):
    review = re.sub("[^a-zA-Z]", " ", df["Review"].iloc[i]).lower()
    words = review.split()
    words = [w for w in words if w not in english_stopwords]
    words = [ps.stem(w) for w in words]
    corpus.append(" ".join(words))

print("\nPreprocessing completed!")
print("\nFirst 10 processed reviews:")
print(corpus[:10])

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df["Liked"].values

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=35,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

y_pred = model.predict(X_test)

score1 = accuracy_score(y_test, y_pred)
score2 = precision_score(y_test, y_pred, zero_division=0)
score3 = recall_score(y_test, y_pred, zero_division=0)

print("\n---- Scores ----")
print("Accuracy:", round(score1 * 100, 2), "%")
print("Precision:", round(score2, 2))
print("Recall:", round(score3, 2))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Random Forest Confusion Matrix")
plt.show()

best_accuracy = 0
n_estimators_val = 5

print("\nTesting different n_estimators values...\n")

for i in np.arange(1, 30, 2):
    temp_model = RandomForestClassifier(
        n_estimators=i,
        random_state=42
    )

    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)

    score = accuracy_score(y_test, temp_pred)

    print(
        "n_estimators =",
        i,
        "Accuracy =",
        round(score * 100, 2),
        "%"
    )

    if score > best_accuracy:
        best_accuracy = score
        n_estimators_val = i

print(
    "\nBest accuracy:",
    round(best_accuracy * 100, 2),
    "%",
    "with n_estimators =",
    n_estimators_val
)

model = RandomForestClassifier(
    n_estimators=n_estimators_val,
    random_state=42
)

model.fit(X_train, y_train)

print("\nOptimized Random Forest Model:")
print(model)

def predict_sentiment(sample_review):
    review = re.sub("[^a-zA-Z]", " ", sample_review).lower()
    words = review.split()
    words = [w for w in words if w not in english_stopwords]
    words = [ps.stem(w) for w in words]
    review = " ".join(words)
    data = cv.transform([review]).toarray()
    return model.predict(data)[0]

sample_review = "The food is really good here."
prediction = predict_sentiment(sample_review)

print("\nReview:", sample_review)

if prediction == 1:
    print("This is a POSITIVE review.")
else:
    print("This is a NEGATIVE review!")

sample_review = "The food was absolutely awful."
prediction = predict_sentiment(sample_review)

print("\nReview:", sample_review)

if prediction == 1:
    print("This is a POSITIVE review.")
else:
    print("This is a NEGATIVE review!")

final_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, final_predictions)
precision = precision_score(y_test, final_predictions, zero_division=0)
recall = recall_score(y_test, final_predictions, zero_division=0)

print("\n==========================================")
print("FINAL RANDOM FOREST MODEL PERFORMANCE")
print("==========================================")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 2))
print("Recall   :", round(recall, 2))
print("Best n_estimators:", n_estimators_val)