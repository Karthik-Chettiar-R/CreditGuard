"""
AdaBoost Classifier implemented from scratch using NumPy with Gini-based decision stumps.
Supports sample reweighting, estimator weight calculation, margin-based probabilities, and error tracking.
"""

from typing import List, Tuple, Optional
import numpy as np
from src.decision_tree import DecisionTreeClassifier


class DecisionStump:
    """
    Fast Gini-based Decision Stump (depth-1 decision tree) for AdaBoost.
    """

    def __init__(self):
        self.feature: int = 0
        self.threshold: float = 0.0
        self.polarity: int = 1  # 1 means <= thresh is -1, > thresh is +1; -1 means inverted
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y_pm1: np.ndarray, sample_weight: np.ndarray) -> "DecisionStump":
        """
        Fit decision stump to minimize weighted classification error.
        y_pm1: binary labels in {-1, +1}
        sample_weight: normalized sample weights summing to 1
        """
        n_samples, n_features = X.shape
        min_error = float("inf")

        for feat in range(n_features):
            col = X[:, feat]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) > 40:
                percentiles = np.linspace(1, 99, 39)
                thresholds = np.unique(np.percentile(col, percentiles))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                for polarity in [1, -1]:
                    # Predictions in {-1, +1}
                    if polarity == 1:
                        preds = np.where(col <= thresh, -1, 1)
                    else:
                        preds = np.where(col <= thresh, 1, -1)

                    error = np.sum(sample_weight[y_pm1 != preds])
                    if error < min_error:
                        min_error = error
                        self.feature = feat
                        self.threshold = thresh
                        self.polarity = polarity

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels in {-1, +1}."""
        col = X[:, self.feature]
        if self.polarity == 1:
            return np.where(col <= self.threshold, -1, 1)
        else:
            return np.where(col <= self.threshold, 1, -1)


class AdaBoostClassifier:
    """
    AdaBoost (Adaptive Boosting) Classifier using Gini-based decision stumps.

    Parameters:
    -----------
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight shrinkage applied to each classifier at each boosting iteration.
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_: List[DecisionStump] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        self.classes_: np.ndarray = np.array([0, 1])
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """
        Fit AdaBoost model.
        X: shape (n_samples, n_features)
        y: binary labels in {0, 1}
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n_samples = len(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostClassifier currently supports binary classification.")

        # Map {0, 1} to {-1, +1}
        y_pm1 = np.where(y == self.classes_[0], -1, 1)

        # Initialize uniform sample weights
        weights = np.ones(n_samples, dtype=float) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        for m in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y_pm1, weights)
            preds = stump.predict(X)

            # Misclassifications
            misclassified = (y_pm1 != preds)
            error = np.sum(weights[misclassified])

            # Numerical stability guard
            error = np.clip(error, 1e-10, 1.0 - 1e-10)

            # If error >= 0.5, weak learner is no better than random guess
            if error >= 0.5:
                # If it's the very first estimator, keep it with minimal weight
                if m == 0:
                    alpha = 1e-4
                    self.estimators_.append(stump)
                    self.estimator_weights_.append(alpha)
                    self.estimator_errors_.append(error)
                break

            # Estimator weight alpha: 0.5 * ln((1 - error) / error)
            alpha = 0.5 * np.log((1.0 - error) / error) * self.learning_rate

            # Exponential weight update: w_i * exp(-alpha * y_i * h(x_i))
            weights = weights * np.exp(-alpha * y_pm1 * preds)

            # Renormalize
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous ensemble decision score:
        F(x) = sum_{m=1}^M alpha_m * h_m(x)
        """
        if not self.is_fitted:
            raise ValueError("AdaBoostClassifier has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        score = np.zeros(X.shape[0], dtype=float)
        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            score += alpha * stump.predict(X)
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using logistic sigmoid calibration of decision score.
        P(y=1|x) = 1 / (1 + exp(-2 * F(x)))
        """
        score = self.decision_function(X)
        prob_class1 = 1.0 / (1.0 + np.exp(-2.0 * np.clip(score, -20.0, 20.0)))
        prob_class0 = 1.0 - prob_class1
        return np.column_stack([prob_class0, prob_class1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {0, 1}."""
        score = self.decision_function(X)
        y_pm1 = np.where(score >= 0.0, 1, -1)
        return np.where(y_pm1 == -1, self.classes_[0], self.classes_[1])
