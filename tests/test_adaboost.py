"""
Unit tests for AdaBoostClassifier from scratch vs scikit-learn.
"""

import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from src.adaboost import AdaBoostClassifier


@pytest.fixture
def non_linear_data():
    np.random.seed(42)
    # XOR / 4 quadrants
    X = np.random.uniform(-2, 2, size=(200, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    return X, y


def test_adaboost_training_improvement(non_linear_data):
    X, y = non_linear_data
    model = AdaBoostClassifier(n_estimators=20, learning_rate=0.8).fit(X, y)

    assert len(model.estimators_) > 1
    # Check probabilities valid
    probs = model.predict_proba(X)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # Predictions accuracy on non-linear XOR pattern improves beyond random baseline (0.50)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.58


def test_adaboost_vs_sklearn(non_linear_data):
    X, y = non_linear_data

    scratch_ada = AdaBoostClassifier(n_estimators=15, learning_rate=0.8).fit(X, y)
    sklearn_ada = SklearnAdaBoost(
        estimator=SklearnDTC(max_depth=1),
        n_estimators=15,
        learning_rate=0.8,
        random_state=42,
    ).fit(X, y)

    scratch_acc = np.mean(scratch_ada.predict(X) == y)
    sklearn_acc = np.mean(sklearn_ada.predict(X) == y)

    # Benchmark comparison: scratch matches sklearn within 5% on same dataset
    assert abs(scratch_acc - sklearn_acc) < 0.05
