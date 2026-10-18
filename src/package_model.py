"""Train the final model on all data and save the full pipeline."""
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import FEATURES, build_preprocessor, load_data

X, y = load_data()
pipe = Pipeline(
    [
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(C=1.0, max_iter=1000)),
    ]
)
pipe.fit(X, y)

Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/heart_model.joblib")
print("Saved models/heart_model.joblib")

# Reload and predict on the first row to prove the file works on its own
loaded = joblib.load("models/heart_model.joblib")
print("Features:", FEATURES)
print("Sample probability of disease:", loaded.predict_proba(X.iloc[[0]])[0, 1])
