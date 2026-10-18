"""Compare two models with 5-fold cross-validation."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from features import build_preprocessor, load_data

X, y = load_data()

models = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42
    ),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = ["accuracy", "precision", "recall", "roc_auc"]

for name, model in models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=metrics)
    print(name)
    for m in metrics:
        vals = scores["test_" + m]
        print(f"  {m}: {vals.mean():.3f} +/- {vals.std():.3f}")
