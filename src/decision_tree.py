"""
Decision Tree Classifier implemented from scratch using NumPy with Gini index.
Supports max_depth, min_samples_split, sample weights (for AdaBoost), and human-readable rule printing.
"""

from typing import Optional, List, Union
import numpy as np


class DecisionTreeNode:
    """A single node in the Decision Tree."""

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        value: Optional[int] = None,
        probs: Optional[np.ndarray] = None,
        gini: float = 0.0,
        n_samples: int = 0,
        weighted_n_samples: float = 0.0,
        is_leaf: bool = False,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probs = probs
        self.gini = gini
        self.n_samples = n_samples
        self.weighted_n_samples = weighted_n_samples
        self.is_leaf = is_leaf


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using Gini Impurity.

    Parameters:
    -----------
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_impurity_decrease : float, default=0.0
        A split will be considered if it decreases impurity by at least this amount.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root_: Optional[DecisionTreeNode] = None
        self.n_classes_: int = 2
        self.classes_: np.ndarray = np.array([0, 1])
        self.is_fitted: bool = False

    def _gini(self, y: np.ndarray, sample_weight: np.ndarray) -> float:
        """Calculate Gini impurity with optional sample weights."""
        total_weight = np.sum(sample_weight)
        if total_weight <= 0.0:
            return 0.0
        gini = 1.0
        for c in self.classes_:
            p = np.sum(sample_weight[y == c]) / total_weight
            gini -= p ** 2
        return max(0.0, float(gini))

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> tuple:
        """Find the feature and threshold that maximizes weighted Gini gain."""
        n_samples, n_features = X.shape
        total_weight = np.sum(sample_weight)
        parent_gini = self._gini(y, sample_weight)

        best_gain = -1.0
        best_feat = None
        best_thresh = None

        for feat in range(n_features):
            col_values = X[:, feat]
            unique_vals = np.unique(col_values)
            if len(unique_vals) <= 1:
                continue

            # Candidate thresholds (midpoints)
            # If unique values are many, sample percentiles to maintain high efficiency
            if len(unique_vals) > 40:
                percentiles = np.linspace(1, 99, 39)
                thresholds = np.unique(np.percentile(col_values, percentiles))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = col_values <= thresh
                right_mask = ~left_mask

                w_left = np.sum(sample_weight[left_mask])
                w_right = np.sum(sample_weight[right_mask])

                if w_left == 0.0 or w_right == 0.0:
                    continue

                gini_left = self._gini(y[left_mask], sample_weight[left_mask])
                gini_right = self._gini(y[right_mask], sample_weight[right_mask])

                gain = parent_gini - ((w_left / total_weight) * gini_left + (w_right / total_weight) * gini_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, depth: int
    ) -> DecisionTreeNode:
        """Recursively build decision tree."""
        n_samples = len(y)
        total_weight = np.sum(sample_weight)
        gini = self._gini(y, sample_weight)

        # Compute class distribution
        probs = np.zeros(self.n_classes_, dtype=float)
        for i, c in enumerate(self.classes_):
            if total_weight > 0:
                probs[i] = np.sum(sample_weight[y == c]) / total_weight
            else:
                probs[i] = 1.0 / self.n_classes_

        majority_val = self.classes_[np.argmax(probs)]

        # Stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or (n_samples < self.min_samples_split)
            or (gini == 0.0)
            or (n_samples <= 1)
        ):
            return DecisionTreeNode(
                value=majority_val,
                probs=probs,
                gini=gini,
                n_samples=n_samples,
                weighted_n_samples=total_weight,
                is_leaf=True,
            )

        best_feat, best_thresh, best_gain = self._best_split(X, y, sample_weight)

        if best_feat is None or best_gain <= self.min_impurity_decrease:
            return DecisionTreeNode(
                value=majority_val,
                probs=probs,
                gini=gini,
                n_samples=n_samples,
                weighted_n_samples=total_weight,
                is_leaf=True,
            )

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(
            X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1
        )

        return DecisionTreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=majority_val,
            probs=probs,
            gini=gini,
            n_samples=n_samples,
            weighted_n_samples=total_weight,
            is_leaf=False,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DecisionTreeClassifier":
        """
        Build a decision tree classifier from training set (X, y).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n_samples = len(y)

        self.classes_ = np.unique(y)
        self.classes_.sort()
        self.n_classes_ = len(self.classes_)

        if sample_weight is None:
            weights = np.ones(n_samples, dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float).copy()
            weights = weights / np.sum(weights) * n_samples

        self.root_ = self._build_tree(X, y, weights, depth=0)
        self.is_fitted = True
        return self

    def _predict_row(self, node: DecisionTreeNode, row: np.ndarray) -> tuple:
        """Traverse tree for a single sample and return (value, probs)."""
        curr = node
        while not curr.is_leaf:
            if row[curr.feature] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.value, curr.probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if not self.is_fitted:
            raise ValueError("Classifier has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        probs = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i in range(X.shape[0]):
            _, prob = self._predict_row(self.root_, X[i])
            probs[i] = prob
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def export_text(
        self,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a human-readable string representation of the decision tree rules.
        """
        if not self.is_fitted:
            raise ValueError("Classifier has not been fitted yet.")

        if class_names is None:
            class_names = [f"Class {c}" for c in self.classes_]

        lines = []

        def _recurse(node: DecisionTreeNode, depth: int):
            indent = "|   " * depth
            if node.is_leaf:
                class_label = class_names[node.value] if node.value < len(class_names) else f"Class {node.value}"
                p_str = ", ".join(f"{p:.2f}" for p in node.probs)
                lines.append(
                    f"{indent}|--- [Leaf] Predict: {class_label} (probs=[{p_str}], samples={node.n_samples})"
                )
                return

            feat_name = (
                feature_names[node.feature]
                if feature_names is not None and node.feature < len(feature_names)
                else f"feature_{node.feature}"
            )
            thresh = node.threshold

            lines.append(f"{indent}|--- if {feat_name} <= {thresh:.3f}:")
            _recurse(node.left, depth + 1)
            lines.append(f"{indent}|--- else ({feat_name} > {thresh:.3f}):")
            _recurse(node.right, depth + 1)

        lines.append("Decision Tree Rules:")
        _recurse(self.root_, 0)
        return "\n".join(lines)

    def print_tree(
        self,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Print readable if/else rules to stdout."""
        print(self.export_text(feature_names=feature_names, class_names=class_names))
