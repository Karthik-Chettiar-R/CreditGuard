"""
Linear Regression models implemented from scratch using NumPy.
Includes SimpleLinearRegression and MultipleLinearRegression (Normal Equation & Gradient Descent).
"""

from typing import Optional, List
import numpy as np


class SimpleLinearRegression:
    """
    Univariate Simple Linear Regression using closed-form analytical formulas:
        slope (beta_1) = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        intercept (beta_0) = y_mean - beta_1 * x_mean
    """

    def __init__(self):
        self.coef_: float = 0.0
        self.intercept_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleLinearRegression":
        """
        Fit univariate regression model.
        x: 1D array of shape (N,) or 2D array of shape (N, 1)
        y: 1D array of shape (N,)
        """
        x_flat = np.asarray(x, dtype=float).ravel()
        y_flat = np.asarray(y, dtype=float).ravel()

        if len(x_flat) != len(y_flat):
            raise ValueError("x and y must have the same number of samples.")

        x_mean = np.mean(x_flat)
        y_mean = np.mean(y_flat)

        denom = np.sum((x_flat - x_mean) ** 2)
        if denom == 0.0:
            self.coef_ = 0.0
            self.intercept_ = y_mean
        else:
            self.coef_ = np.sum((x_flat - x_mean) * (y_flat - y_mean)) / denom
            self.intercept_ = y_mean - self.coef_ * x_mean

        self.is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        x_flat = np.asarray(x, dtype=float).ravel()
        return self.intercept_ + self.coef_ * x_flat

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 coefficient of determination."""
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(x)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    def rmse(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(x)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class MultipleLinearRegression:
    """
    Multiple Linear Regression implemented from scratch using NumPy.
    Supports:
    - method='normal': Exact closed-form solution via Normal Equation (X^T X)^(-1) X^T y.
    - method='gradient_descent': Batch Gradient Descent with learning rate and loss tracking.
    """

    def __init__(
        self,
        method: str = "normal",
        learning_rate: float = 0.01,
        max_iter: int = 2000,
        tol: float = 1e-7,
        fit_intercept: bool = True,
        l2_reg: float = 1e-8,
    ):
        if method not in ("normal", "gradient_descent"):
            raise ValueError("method must be either 'normal' or 'gradient_descent'.")
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.l2_reg = l2_reg

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.loss_history_: List[float] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultipleLinearRegression":
        """
        Fit multiple linear regression model.
        X: shape (N, D)
        y: shape (N,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        if n_samples != len(y):
            raise ValueError("X and y sample lengths must match.")

        if self.fit_intercept:
            X_design = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_design = X

        if self.method == "normal":
            # Normal equation: (X^T X + lambda I)^(-1) X^T y
            XtX = X_design.T @ X_design
            if self.l2_reg > 0:
                reg_matrix = self.l2_reg * np.eye(XtX.shape[0])
                if self.fit_intercept:
                    reg_matrix[0, 0] = 0.0  # Do not regularize intercept
                XtX = XtX + reg_matrix

            try:
                beta = np.linalg.solve(XtX, X_design.T @ y)
            except np.linalg.LinAlgError:
                # Use Moore-Penrose pseudoinverse if singular
                beta = np.linalg.pinv(XtX) @ (X_design.T @ y)

            if self.fit_intercept:
                self.intercept_ = float(beta[0])
                self.coef_ = beta[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = beta

            # Record final loss
            y_pred = X_design @ beta
            mse = float(np.mean((y - y_pred) ** 2))
            self.loss_history_ = [mse]

        elif self.method == "gradient_descent":
            # Feature standardization inside gradient descent for numerical stability if unscaled
            beta = np.zeros(X_design.shape[1], dtype=float)
            self.loss_history_ = []

            for iteration in range(self.max_iter):
                preds = X_design @ beta
                error = preds - y
                mse = float(np.mean(error ** 2))
                self.loss_history_.append(mse)

                # Gradient: (1 / n) * X^T (X * beta - y)
                grad = (X_design.T @ error) / n_samples
                if self.l2_reg > 0:
                    reg_term = (self.l2_reg / n_samples) * beta
                    if self.fit_intercept:
                        reg_term[0] = 0.0
                    grad += reg_term

                grad_norm = np.linalg.norm(grad)
                if grad_norm < self.tol:
                    break

                beta -= self.learning_rate * grad

            if self.fit_intercept:
                self.intercept_ = float(beta[0])
                self.coef_ = beta[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = beta

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return (X @ self.coef_) + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 coefficient of determination."""
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute residuals (y_true - y_pred)."""
        y_true = np.asarray(y, dtype=float).ravel()
        return y_true - self.predict(X)
