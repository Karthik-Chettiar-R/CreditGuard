"""
Unit tests for LinearSVM and KernelSVM from scratch vs scikit-learn.
"""

import numpy as np
import pytest
from sklearn.svm import LinearSVC, SVC
from src.svm import LinearSVM, KernelSVM


@pytest.fixture
def linearly_separable_data():
    np.random.seed(42)
    X1 = np.random.normal(loc=[-2.5, -2.5], scale=0.6, size=(60, 2))
    y1 = np.zeros(60, dtype=int)
    X2 = np.random.normal(loc=[2.5, 2.5], scale=0.6, size=(60, 2))
    y2 = np.ones(60, dtype=int)

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    return X, y


def test_linear_svm_subgradient_descent(linearly_separable_data):
    X, y = linearly_separable_data

    svm = LinearSVM(C=1.0, learning_rate=0.01, epochs=200, random_state=42)
    svm.fit(X, y)

    # Assert loss history was recorded and decreased
    losses = svm.loss_history_
    assert len(losses) == 200
    assert losses[-1] < losses[0]

    # Perfect or near-perfect classification on linearly separable data
    preds = svm.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95

    # Check calibrated probabilities
    probs = svm.predict_proba(X)
    assert probs.shape == (120, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_linear_svm_vs_sklearn(linearly_separable_data):
    X, y = linearly_separable_data

    scratch_svm = LinearSVM(C=1.0, learning_rate=0.01, epochs=250, random_state=42).fit(X, y)
    sklearn_svm = LinearSVC(C=1.0, random_state=42, dual="auto").fit(X, y)

    scratch_acc = np.mean(scratch_svm.predict(X) == y)
    sklearn_acc = np.mean(sklearn_svm.predict(X) == y)

    # Both achieve >95% accuracy on linearly separable data
    assert scratch_acc >= 0.95
    assert sklearn_acc >= 0.95


def test_kernel_svm_rbf_smo(linearly_separable_data):
    X, y = linearly_separable_data

    kernel_svm = KernelSVM(C=1.0, gamma=0.5, max_iter=30, random_state=42)
    kernel_svm.fit(X, y)

    preds = kernel_svm.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95
