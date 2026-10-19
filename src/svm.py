"""
Support Vector Machine (SVM) models implemented from scratch using NumPy.
Includes:
1. LinearSVM: Soft-margin linear SVM trained by sub-gradient descent on hinge loss.
2. KernelSVM: Non-linear RBF kernel SVM trained via Simplified Sequential Minimal Optimization (SMO).
"""

from typing import Optional, List
import numpy as np


class LinearSVM:
    """
    Linear Soft-Margin Support Vector Machine trained via Sub-Gradient Descent.

    Objective:
        min_{w, b} 0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i * (w^T x_i + b))

    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter. Higher C means smaller margin, fewer margin violations.
    learning_rate : float, default=0.001
        Initial learning rate for sub-gradient descent.
    epochs : int, default=500
        Number of passes over the training dataset.
    batch_size : int, default=64
        Mini-batch size for subgradient descent.
    decay : float, default=0.001
        Learning rate decay factor: lr_t = lr_0 / (1 + decay * epoch).
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.001,
        epochs: int = 500,
        batch_size: int = 64,
        decay: float = 0.001,
        random_state: Optional[int] = 42,
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay = decay
        self.random_state = random_state

        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.loss_history_: List[float] = []
        self.classes_: np.ndarray = np.array([0, 1])
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        """
        Fit linear SVM using mini-batch subgradient descent on soft-margin hinge loss.
        """
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LinearSVM supports binary classification.")

        # Map {0, 1} to {-1, +1}
        y_pm1 = np.where(y == self.classes_[0], -1, 1).astype(float)

        # Initialize weights
        self.w_ = rng.normal(0, 0.01, size=n_features)
        self.b_ = 0.0
        self.loss_history_ = []

        for epoch in range(self.epochs):
            lr = self.learning_rate / (1.0 + self.decay * epoch)
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y_pm1[batch_idx]
                m = len(batch_idx)

                # Margins: y_i * (w^T x_i + b)
                margins = y_batch * (X_batch @ self.w_ + self.b_)

                # Indicator of margin violation: margin < 1.0
                misclassified = margins < 1.0

                # Subgradients
                # grad_w = (1/n_samples) * w - (C / m) * sum_{i: viol} y_i x_i
                grad_w = self.w_ / n_samples
                grad_b = 0.0

                if np.any(misclassified):
                    X_viol = X_batch[misclassified]
                    y_viol = y_batch[misclassified]
                    grad_w -= (self.C / m) * (y_viol @ X_viol)
                    grad_b -= (self.C / m) * np.sum(y_viol)

                # Update parameters
                self.w_ -= lr * grad_w
                self.b_ -= lr * grad_b

            # Compute epoch total loss
            total_margins = y_pm1 * (X @ self.w_ + self.b_)
            hinge_loss = np.sum(np.maximum(0.0, 1.0 - total_margins))
            reg_loss = 0.5 * np.sum(self.w_ ** 2)
            total_loss = float(reg_loss + self.C * hinge_loss)
            self.loss_history_.append(total_loss)

        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate signed distance to hyperplane: w^T x + b."""
        if not self.is_fitted:
            raise ValueError("LinearSVM has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return X @ self.w_ + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability via logistic sigmoid of decision function."""
        dist = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(dist, -15.0, 15.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        dist = self.decision_function(X)
        pred_pm1 = np.where(dist >= 0.0, 1, -1)
        return np.where(pred_pm1 == -1, self.classes_[0], self.classes_[1])


class KernelSVM:
    """
    RBF Kernel Support Vector Machine trained via Simplified Sequential Minimal Optimization (SMO).

    Dual problem:
        max_alpha sum_i alpha_i - 0.5 * sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
        s.t. 0 <= alpha_i <= C, sum_i alpha_i y_i = 0

    Kernel:
        K(x, x') = exp(-gamma * ||x - x'||^2)
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: Optional[float] = None,
        max_iter: int = 50,
        tol: float = 1e-3,
        random_state: Optional[int] = 42,
    ):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.alphas_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.support_vector_indices_: Optional[np.ndarray] = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.is_fitted: bool = False

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF Gaussian kernel matrix between X1 and X2."""
        gamma = self.gamma if self.gamma is not None else 1.0 / X1.shape[1]
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        sq_dist = np.maximum(0.0, X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T))
        return np.exp(-gamma * sq_dist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelSVM":
        """Fit RBF Kernel SVM using Simplified SMO."""
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("KernelSVM supports binary classification.")

        # Map {0, 1} to {-1, +1}
        y_pm1 = np.where(y == self.classes_[0], -1, 1).astype(float)
        self.gamma = self.gamma if self.gamma is not None else 1.0 / n_features

        # Precompute kernel matrix K
        K = self._rbf_kernel(X, X)

        alphas = np.zeros(n_samples, dtype=float)
        b = 0.0
        passes = 0

        while passes < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                # Prediction f(x_i) = sum_j alpha_j y_j K(x_j, x_i) + b
                f_xi = np.sum(alphas * y_pm1 * K[:, i]) + b
                E_i = f_xi - y_pm1[i]

                # Check KKT conditions
                if (y_pm1[i] * E_i < -self.tol and alphas[i] < self.C) or (
                    y_pm1[i] * E_i > self.tol and alphas[i] > 0
                ):
                    # Randomly pick j != i
                    j = rng.choice([idx for idx in range(n_samples) if idx != i])
                    f_xj = np.sum(alphas * y_pm1 * K[:, j]) + b
                    E_j = f_xj - y_pm1[j]

                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    # Compute clipping bounds L and H
                    if y_pm1[i] != y_pm1[j]:
                        L = max(0.0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0.0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])

                    if abs(L - H) < 1e-7:
                        continue

                    # eta = 2 * K(i, j) - K(i, i) - K(j, j)
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alphas[j] = alpha_j_old - (y_pm1[j] * (E_i - E_j)) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alphas[i] = alpha_i_old + y_pm1[i] * y_pm1[j] * (alpha_j_old - alphas[j])

                    # Compute threshold b
                    b1 = b - E_i - y_pm1[i] * (alphas[i] - alpha_i_old) * K[i, i] - y_pm1[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y_pm1[i] * (alphas[i] - alpha_i_old) * K[i, j] - y_pm1[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # Extract support vectors
        sv_mask = alphas > 1e-5
        self.support_vector_indices_ = np.where(sv_mask)[0]
        self.X_train_ = X[sv_mask]
        self.y_train_ = y_pm1[sv_mask]
        self.alphas_ = alphas[sv_mask]
        self.b_ = b
        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate continuous decision score for test points X."""
        if not self.is_fitted:
            raise ValueError("KernelSVM has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        K_test = self._rbf_kernel(X, self.X_train_)
        return (K_test @ (self.alphas_ * self.y_train_)) + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        score = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(score, -15.0, 15.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {0, 1}."""
        score = self.decision_function(X)
        pred_pm1 = np.where(score >= 0.0, 1, -1)
        return np.where(pred_pm1 == -1, self.classes_[0], self.classes_[1])
