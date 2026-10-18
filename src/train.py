"""Train models with cross-validation and log every run to MLflow."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from features import build_preprocessor, load_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart-disease")

X, y = load_data()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
METRICS = ["accuracy", "precision", "recall", "roc_auc"]

CONFIGS = []
for c in [0.1, 1.0, 10.0]:
    CONFIGS.append(("logistic_regression", {"C": c}))
for depth in [3, 5, 8]:
    for n in [100, 200]:
        CONFIGS.append(("random_forest", {"max_depth": depth, "n_estimators": n}))


def make_model(name, params):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, **params)
    return RandomForestClassifier(random_state=42, **params)


for name, params in CONFIGS:
    pipe = Pipeline([("prep", build_preprocessor()), ("model", make_model(name, params))])
    run_name = name + "-" + "-".join(f"{k}={v}" for k, v in params.items())

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", name)
        mlflow.log_params(params)
        mlflow.log_param("cv_folds", 5)

        scores = cross_validate(pipe, X, y, cv=cv, scoring=METRICS)
        for m in METRICS:
            vals = scores["test_" + m]
            mlflow.log_metric(f"cv_{m}_mean", vals.mean())
            mlflow.log_metric(f"cv_{m}_std", vals.std())

        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(y, proba, ax=ax)
        ax.set_title("ROC curve (cross-validated)")
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y, proba > 0.5, ax=ax)
        ax.set_title("Confusion matrix (cross-validated)")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.log_artifact("data/processed/heart_clean.csv")
        print(
            f"{run_name}: roc_auc={scores['test_roc_auc'].mean():.3f} "
            f"recall={scores['test_recall'].mean():.3f}"
        )
