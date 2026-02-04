import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Always resolve paths from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df_true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
df_fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))

df_true["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_true, df_fake]).sample(frac=1, random_state=42)
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1,2)
)

X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.33, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model + vectorizer
joblib.dump(model, os.path.join(MODELS_DIR, "logistic_model.pkl"))
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

print("✅ Model and TF-IDF vectorizer saved in models/")
