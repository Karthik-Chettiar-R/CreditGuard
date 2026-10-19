"""
Principal Component Analysis (PCA) implemented from scratch using NumPy
via sample covariance matrix and eigendecomposition.
"""

from typing import Optional
import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) using Covariance Matrix + Eigendecomposition.

    Parameters:
    -----------
    n_components : int or None, default=None
        Number of principal components to keep. If None, all components are kept.

    Attributes:
    -----------
    components_ : np.ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance.
    explained_variance_ : np.ndarray of shape (n_components,)
        The variance explained by each of the selected components.
    explained_variance_ratio_ : np.ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    cumulative_explained_variance_ratio_ : np.ndarray of shape (n_components,)
        Cumulative explained variance ratio.
    mean_ : np.ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Fit PCA model on data matrix X.
        X: shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        k = self.n_components if self.n_components is not None else n_features
        k = min(k, n_samples, n_features)

        # 1. Compute empirical mean and center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute sample covariance matrix: (X_c^T X_c) / (N - 1)
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # 3. Eigendecomposition of symmetric covariance matrix
        # np.linalg.eigh is numerically stable for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = np.maximum(eigenvalues[sort_indices], 0.0)
        sorted_eigenvectors = eigenvectors[:, sort_indices]

        # Total variance across all components
        total_variance = np.sum(sorted_eigenvalues)
        if total_variance == 0.0:
            total_variance = 1e-12

        # 5. Select top k components
        self.explained_variance_ = sorted_eigenvalues[:k]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # Transpose so rows are component vectors: shape (k, n_features)
        components = sorted_eigenvectors[:, :k].T

        # Deterministic sign convention (similar to sklearn's svd_flip):
        # Force the component with maximum absolute value to be positive
        for i in range(components.shape[0]):
            max_abs_idx = np.argmax(np.abs(components[i]))
            if components[i, max_abs_idx] < 0:
                components[i] *= -1.0

        self.components_ = components
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.
        X: shape (n_samples, n_features)
        Returns:
            X_transformed: shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        X_transformed: shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet.")
        X_transformed = np.asarray(X_transformed, dtype=float)
        return (X_transformed @ self.components_) + self.mean_
