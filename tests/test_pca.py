"""
Unit tests for PCA from scratch vs scikit-learn.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from src.pca import PCA


@pytest.fixture
def sample_data():
    np.random.seed(42)
    # Generate correlated synthetic dataset
    mean = [1.0, 2.0, -1.0, 0.5]
    cov = [
        [3.0, 1.5, 0.2, 0.1],
        [1.5, 2.0, 0.8, 0.3],
        [0.2, 0.8, 1.5, 0.4],
        [0.1, 0.3, 0.4, 0.8],
    ]
    return np.random.multivariate_normal(mean, cov, size=200)


def test_pca_explained_variance_ratio(sample_data):
    X = sample_data
    n_components = 3

    scratch_pca = PCA(n_components=n_components).fit(X)
    sklearn_pca = SklearnPCA(n_components=n_components).fit(X)

    # Explained variance ratios match sklearn within 1e-5
    assert np.allclose(
        scratch_pca.explained_variance_ratio_,
        sklearn_pca.explained_variance_ratio_,
        atol=1e-5,
    )

    # Cumulative variance ratio matches
    assert np.allclose(
        scratch_pca.cumulative_explained_variance_ratio_,
        np.cumsum(sklearn_pca.explained_variance_ratio_),
        atol=1e-5,
    )


def test_pca_transform_and_reconstruction(sample_data):
    X = sample_data
    n_components = 2

    scratch_pca = PCA(n_components=n_components).fit(X)
    sklearn_pca = SklearnPCA(n_components=n_components).fit(X)

    X_trans_scratch = scratch_pca.transform(X)
    X_trans_sklearn = sklearn_pca.transform(X)

    # Check projection magnitude matches (up to component sign convention)
    assert np.allclose(np.abs(X_trans_scratch), np.abs(X_trans_sklearn), atol=1e-4)

    # Check inverse transform reconstructs centered data with low residual
    X_recon = scratch_pca.inverse_transform(X_trans_scratch)
    reconstruction_error = np.mean((X - X_recon) ** 2)
    # Low reconstruction error with 2 components out of 4
    assert reconstruction_error < 0.5
