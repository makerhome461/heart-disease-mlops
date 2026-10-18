"""Tests for data loading and the cleaned dataset."""
from features import FEATURES, load_data


def test_features_and_target_shapes():
    X, y = load_data()
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)
    assert len(X) > 250


def test_no_missing_values():
    X, y = load_data()
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_target_is_binary():
    _, y = load_data()
    assert set(y.unique()) == {0, 1}
