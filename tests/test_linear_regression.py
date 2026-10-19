"""
Unit tests for SimpleLinearRegression and MultipleLinearRegression from scratch vs scikit-learn.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from src.linear_regression import SimpleLinearRegression, MultipleLinearRegression


@pytest.fixture
def synthetic_regression_data():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    true_coef = np.array([2.5, -1.8, 0.5, 3.2, -0.9])
    true_intercept = 4.2
    y = X @ true_coef + true_intercept + np.random.normal(0, 0.1, size=200)
    return X, y


def test_simple_linear_regression(synthetic_regression_data):
    X, y = synthetic_regression_data
    x_1d = X[:, 0]

    scratch_model = SimpleLinearRegression().fit(x_1d, y)
    sklearn_model = SklearnLinearRegression().fit(x_1d.reshape(-1, 1), y)

    # Compare slope and intercept within tight tolerance
    assert np.isclose(scratch_model.coef_, sklearn_model.coef_[0], atol=1e-5)
    assert np.isclose(scratch_model.intercept_, sklearn_model.intercept_, atol=1e-5)

    # Compare predictions and R^2 score
    preds_scratch = scratch_model.predict(x_1d)
    preds_sklearn = sklearn_model.predict(x_1d.reshape(-1, 1))
    assert np.allclose(preds_scratch, preds_sklearn, atol=1e-5)
    assert np.isclose(scratch_model.score(x_1d, y), sklearn_model.score(x_1d.reshape(-1, 1), y), atol=1e-5)


def test_multiple_linear_regression_normal_equation(synthetic_regression_data):
    X, y = synthetic_regression_data

    scratch_model = MultipleLinearRegression(method="normal", l2_reg=0.0).fit(X, y)
    sklearn_model = SklearnLinearRegression().fit(X, y)

    # Coefficients and intercept match sklearn
    assert np.allclose(scratch_model.coef_, sklearn_model.coef_, atol=1e-4)
    assert np.isclose(scratch_model.intercept_, sklearn_model.intercept_, atol=1e-4)

    # Predictions match
    preds_scratch = scratch_model.predict(X)
    preds_sklearn = sklearn_model.predict(X)
    assert np.allclose(preds_scratch, preds_sklearn, atol=1e-4)
    assert scratch_model.score(X, y) > 0.95


def test_multiple_linear_regression_gradient_descent(synthetic_regression_data):
    X, y = synthetic_regression_data

    scratch_gd = MultipleLinearRegression(
        method="gradient_descent", learning_rate=0.05, max_iter=2000, l2_reg=0.0
    ).fit(X, y)

    # Assert loss history decreases monotonically
    losses = scratch_gd.loss_history_
    assert len(losses) > 1
    assert losses[-1] < losses[0]

    # Model achieves high R^2 close to analytical solution
    r2 = scratch_gd.score(X, y)
    assert r2 > 0.90
