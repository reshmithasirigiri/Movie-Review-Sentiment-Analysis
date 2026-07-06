
# Movie Review Sentiment Analysis
# Google Colab / Jupyter Ready

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
nltk.download("stopwords")

df = pd.read_csv("IMDB Dataset.csv")

print("First 5 Rows")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Count:")
print(df["sentiment"].value_counts())

# -----------------------------
# Sentiment Distribution
# -----------------------------
plt.figure(figsize=(6,4))
df["sentiment"].value_counts().plot(kind="bar", color=["red","green"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# Encode Labels
# -----------------------------
df["sentiment"] = df["sentiment"].map({
    "positive":1,
    "negative":0
})

# -----------------------------
# Text Cleaning
# -----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"<.*?>"," ",text)
    text = re.sub(r"[^a-zA-Z]"," ",text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

print("\nCleaning reviews...")
df["clean_review"] = df["review"].apply(clean_text)

print(df[["review","clean_review"]].head())

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# -----------------------------
# Train Model
# -----------------------------
print("\nTraining Logistic Regression...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test,y_pred)

print(f"\nAccuracy : {accuracy*100:.2f}%")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report\n")
print(classification_report(y_test,y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test,y_pred)

print("\nConfusion Matrix\n")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative","Positive"],
    yticklabels=["Negative","Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# Test Custom Reviews
# -----------------------------
while True:
    review = input("\nEnter Movie Review (type exit to stop): ")

    if review.lower() == "exit":
        break

    cleaned = clean_text(review)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Prediction : Positive 😊")
    else:
        print("Prediction : Negative 😞")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("\nFiles Saved Successfully!")
print("model.pkl")
print("vectorizer.pkl")
