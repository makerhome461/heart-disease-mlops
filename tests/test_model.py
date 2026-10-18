"""Tests for the preprocessing + model pipeline."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import build_preprocessor, load_data


def make_pipeline():
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def test_pipeline_probabilities_are_valid():
    X, y = load_data()
    pipe = make_pipeline().fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_beats_chance():
    X, y = load_data()
    pipe = make_pipeline().fit(X, y)
    assert pipe.score(X, y) > 0.7
