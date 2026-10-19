"""
Unit tests for DecisionTreeClassifier from scratch vs scikit-learn.
"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from src.decision_tree import DecisionTreeClassifier


@pytest.fixture
def classification_data():
    np.random.seed(42)
    # 2-class dataset
    X1 = np.random.normal(loc=[-2.0, -2.0], scale=0.8, size=(100, 2))
    y1 = np.zeros(100, dtype=int)
    X2 = np.random.normal(loc=[2.0, 2.0], scale=0.8, size=(100, 2))
    y2 = np.ones(100, dtype=int)

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    return X, y


def test_decision_tree_separable_data(classification_data):
    X, y = classification_data
    dt = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
    dt.fit(X, y)

    preds = dt.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95

    probs = dt.predict_proba(X)
    assert probs.shape == (200, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_decision_tree_rules_export(classification_data):
    X, y = classification_data
    dt = DecisionTreeClassifier(max_depth=2).fit(X, y)
    rules_text = dt.export_text(
        feature_names=["account_status", "loan_duration"],
        class_names=["Good", "Bad"],
    )

    assert "Decision Tree Rules:" in rules_text
    assert "|--- if account_status" in rules_text or "|--- if loan_duration" in rules_text
    assert "[Leaf]" in rules_text


def test_decision_tree_vs_sklearn(classification_data):
    X, y = classification_data

    scratch_dt = DecisionTreeClassifier(max_depth=3).fit(X, y)
    sklearn_dt = SklearnDTC(criterion="gini", max_depth=3, random_state=42).fit(X, y)

    scratch_acc = np.mean(scratch_dt.predict(X) == y)
    sklearn_acc = np.mean(sklearn_dt.predict(X) == y)

    # Both achieve comparable classification performance (within 5% accuracy)
    assert abs(scratch_acc - sklearn_acc) < 0.05
