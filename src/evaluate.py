import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Resolve project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# Load model & vectorizer
model_path = os.path.join(MODELS_DIR, "logistic_model.pkl")
vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model not found. Run src/train.py first.")

model = joblib.load(model_path)
tfidf = joblib.load(vectorizer_path)

# Load data
df_true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
df_fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))

df_true["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_true, df_fake])
df["content"] = df["title"] + " " + df["text"]

X = tfidf.transform(df["content"])
y = df["label"]

# Predictions
y_pred = model.predict(X)

print(classification_report(y, y_pred))

# Save metrics
results = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Precision": [precision_score(y, y_pred)],
    "Recall": [recall_score(y, y_pred)],
    "F1-Score": [f1_score(y, y_pred)]
})

results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

# Confusion matrix
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
plt.close()

print("✅ Evaluation complete. Reports saved in reports/")
