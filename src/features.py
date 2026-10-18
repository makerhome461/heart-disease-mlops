"""Feature definitions and the preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
BINARY = ["sex", "fbs", "exang"]
FEATURES = NUMERIC + CATEGORICAL + BINARY
TARGET = "target"


def load_data(path="data/processed/heart_clean.csv"):
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ]
    )
