"""
Gaussian Mixture Model (GMM) implemented from scratch using NumPy.
Trained via the Expectation-Maximization (EM) algorithm with:
- Full covariance matrices
- K-Means++ style initialization
- Numerical stability via Cholesky decomposition & log-sum-exp
- Monotonic log-likelihood tracking
- BIC (Bayesian Information Criterion) calculation for choosing k
"""

from typing import Optional, List, Tuple
import numpy as np


class GaussianMixtureModel:
    """
    Gaussian Mixture Model (GMM) with Expectation-Maximization.

    Parameters:
    -----------
    n_components : int, default=3
        The number of mixture components (clusters).
    max_iter : int, default=100
        The maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence threshold based on change in log-likelihood.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance matrices
        to guarantee positive-definiteness.
    random_state : int or None, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None  # (k,)
        self.means_: Optional[np.ndarray] = None  # (k, D)
        self.covariances_: Optional[np.ndarray] = None  # (k, D, D)
        self.log_likelihood_history_: List[float] = []
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.is_fitted: bool = False

    def _kmeans_pp_init(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """K-means++ style initialization for component means."""
        n_samples, n_features = X.shape
        means = np.empty((self.n_components, n_features), dtype=float)

        # First center uniformly at random
        first_idx = rng.randint(0, n_samples)
        means[0] = X[first_idx]

        # Remaining centers sampled proportional to squared distance
        min_sq_dist = np.sum((X - means[0]) ** 2, axis=1)

        for c in range(1, self.n_components):
            prob = min_sq_dist / np.sum(min_sq_dist)
            # Sample next center
            next_idx = rng.choice(n_samples, p=prob)
            means[c] = X[next_idx]

            # Update minimum squared distances
            dist_to_c = np.sum((X - means[c]) ** 2, axis=1)
            min_sq_dist = np.minimum(min_sq_dist, dist_to_c)

        return means

    def _compute_log_det_and_inv(
        self, cov: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute log determinant and inverse using Cholesky decomposition
        for guaranteed numerical stability.
        """
        D = cov.shape[0]
        cov_reg = cov + self.reg_covar * np.eye(D)
        try:
            L = np.linalg.cholesky(cov_reg)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            # Inverse via solve
            L_inv = np.linalg.inv(L)
            inv_cov = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse and SVD determinant
            sign, log_det = np.linalg.slogdet(cov_reg)
            inv_cov = np.linalg.pinv(cov_reg)
        return log_det, inv_cov

    def _log_gaussian_pdf(
        self, X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, log_det: float
    ) -> np.ndarray:
        """Compute log N(X | mean, cov) for all samples in X."""
        D = X.shape[1]
        diff = X - mean
        # Vectorized Mahalanobis distance: sum((diff @ inv_cov) * diff, axis=1)
        maha = np.sum((diff @ inv_cov) * diff, axis=1)
        log_prob = -0.5 * (D * np.log(2.0 * np.pi) + log_det + maha)
        return log_prob

    def _e_step(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Expectation Step:
        Compute responsibilities gamma_{ik} and total log-likelihood.
        """
        n_samples = X.shape[0]
        K = self.n_components
        weighted_log_prob = np.zeros((n_samples, K), dtype=float)

        for k in range(K):
            log_det, inv_cov = self._compute_log_det_and_inv(self.covariances_[k])
            log_pdf = self._log_gaussian_pdf(X, self.means_[k], inv_cov, log_det)
            weighted_log_prob[:, k] = np.log(self.weights_[k] + 1e-15) + log_pdf

        # Log-sum-exp trick for stable sum over components
        max_log_prob = np.max(weighted_log_prob, axis=1, keepdims=True)
        log_prob_sum = max_log_prob + np.log(
            np.sum(np.exp(weighted_log_prob - max_log_prob), axis=1, keepdims=True)
        )

        total_log_likelihood = float(np.sum(log_prob_sum))

        # Responsibilities gamma_{ik} = exp(log_prob - log_prob_sum)
        responsibilities = np.exp(weighted_log_prob - log_prob_sum)
        return responsibilities, total_log_likelihood

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        Maximization Step:
        Update mixture weights, means, and full covariance matrices.
        """
        n_samples, n_features = X.shape
        K = self.n_components

        # Effective number of points assigned to each component
        Nk = np.sum(responsibilities, axis=0) + 1e-10

        # Update weights
        self.weights_ = Nk / n_samples

        # Update means
        self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]

        # Update covariances
        for k in range(K):
            diff = X - self.means_[k]
            # Weighted outer product: (diff.T * resp_k) @ diff / Nk
            weighted_diff = diff * responsibilities[:, k : k + 1]
            cov_k = (weighted_diff.T @ diff) / Nk[k]
            # Regularize
            self.covariances_[k] = cov_k + self.reg_covar * np.eye(n_features)

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """
        Fit Gaussian Mixture Model on data matrix X via EM.
        Tracks log-likelihood monotonically.
        """
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if self.n_components > n_samples:
            raise ValueError("n_components cannot exceed number of samples.")

        # 1. Initialize means using K-means++
        self.means_ = self._kmeans_pp_init(X, rng)

        # 2. Initialize weights uniformly
        self.weights_ = np.ones(self.n_components, dtype=float) / self.n_components

        # 3. Initialize covariances to data covariance
        emp_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
        self.covariances_ = np.array([emp_cov.copy() for _ in range(self.n_components)])

        self.log_likelihood_history_ = []
        prev_ll = -np.inf
        self.converged_ = False

        for iteration in range(self.max_iter):
            self.n_iter_ = iteration + 1

            # E-step
            responsibilities, total_ll = self._e_step(X)
            self.log_likelihood_history_.append(total_ll)

            # Check convergence
            if iteration > 0 and (total_ll - prev_ll) < self.tol and (total_ll >= prev_ll):
                self.converged_ = True
                break

            prev_ll = total_ll

            # M-step
            self._m_step(X, responsibilities)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior cluster membership probabilities (responsibilities)."""
        if not self.is_fitted:
            raise ValueError("GMM has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        resp, _ = self._e_step(X)
        return resp

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely component index for each sample in X."""
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)

    def score(self, X: np.ndarray) -> float:
        """Compute the per-sample average log-likelihood of data X."""
        if not self.is_fitted:
            raise ValueError("GMM has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        _, total_ll = self._e_step(X)
        return float(total_ll / X.shape[0])

    def bic(self, X: np.ndarray) -> float:
        """
        Calculate Bayesian Information Criterion (BIC):
            BIC = -2 * log_likelihood + p * ln(N)
        where p is the number of free parameters:
            p = (K - 1) + K * D + K * (D * (D + 1) / 2)
        Lower is better.
        """
        if not self.is_fitted:
            raise ValueError("GMM has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        K = self.n_components
        D = n_features

        # Free parameters
        p_weights = K - 1
        p_means = K * D
        p_covs = K * (D * (D + 1)) // 2
        total_params = p_weights + p_means + p_covs

        _, total_ll = self._e_step(X)
        bic_score = -2.0 * total_ll + total_params * np.log(n_samples)
        return float(bic_score)
