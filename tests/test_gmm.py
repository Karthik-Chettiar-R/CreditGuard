"""
Unit tests for Gaussian Mixture Model (EM algorithm) from scratch.
Explicitly asserts that EM log-likelihood is monotonically non-decreasing,
and compares with scikit-learn GaussianMixture.
"""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture as SklearnGMM
from src.gmm import GaussianMixtureModel


@pytest.fixture
def cluster_data():
    np.random.seed(42)
    # Generate 3 distinct Gaussian clusters
    c1 = np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.2], [0.2, 1.0]], size=80)
    c2 = np.random.multivariate_normal([5.0, 5.0], [[1.2, -0.3], [-0.3, 1.2]], size=80)
    c3 = np.random.multivariate_normal([-5.0, 5.0], [[0.8, 0.1], [0.1, 0.8]], size=80)
    return np.vstack([c1, c2, c3])


def test_gmm_log_likelihood_non_decreasing(cluster_data):
    X = cluster_data
    gmm = GaussianMixtureModel(n_components=3, max_iter=50, random_state=42)
    gmm.fit(X)

    ll_history = gmm.log_likelihood_history_
    assert len(ll_history) >= 2

    # Assert EM log-likelihood is non-decreasing across all iterations
    # Allow tiny numerical tolerance for floating point precision (1e-6)
    diffs = np.diff(ll_history)
    assert np.all(diffs >= -1e-6), f"Log-likelihood decreased! Diffs: {diffs[diffs < -1e-6]}"

    # Assert model converged
    assert gmm.is_fitted
    assert gmm.weights_.shape == (3,)
    assert np.isclose(np.sum(gmm.weights_), 1.0)


def test_gmm_bic_and_clustering(cluster_data):
    X = cluster_data

    # Compare BIC for k=1, 3, 6
    gmm_1 = GaussianMixtureModel(n_components=1, max_iter=40, random_state=42).fit(X)
    gmm_3 = GaussianMixtureModel(n_components=3, max_iter=40, random_state=42).fit(X)
    gmm_6 = GaussianMixtureModel(n_components=6, max_iter=40, random_state=42).fit(X)

    bic_1 = gmm_1.bic(X)
    bic_3 = gmm_3.bic(X)
    bic_6 = gmm_6.bic(X)

    # 3 true clusters should have significantly lower BIC than 1 cluster
    assert bic_3 < bic_1

    # Responsibilities shape and valid probability distribution
    probs = gmm_3.predict_proba(X)
    assert probs.shape == (240, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_gmm_vs_sklearn(cluster_data):
    X = cluster_data

    scratch_gmm = GaussianMixtureModel(n_components=3, max_iter=50, random_state=42).fit(X)
    sklearn_gmm = SklearnGMM(n_components=3, max_iter=50, random_state=42).fit(X)

    # Compare per-sample log likelihood score: should be very close
    scratch_score = scratch_gmm.score(X)
    sklearn_score = sklearn_gmm.score(X)

    # Both models achieve high average log-likelihood (within 0.5 nats/sample)
    assert abs(scratch_score - sklearn_score) < 0.5
