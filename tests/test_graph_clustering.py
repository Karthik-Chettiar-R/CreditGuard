"""
Unit tests for Divisive Graph Clustering with Girvan-Newman edge betweenness.
"""

import numpy as np
import pytest
from src.graph_clustering import DivisiveGraphClustering


@pytest.fixture
def well_separated_clusters():
    np.random.seed(42)
    # 2 small clusters for fast testing
    c1 = np.random.normal(loc=[-4.0, 0.0], scale=0.4, size=(25, 2))
    c2 = np.random.normal(loc=[4.0, 0.0], scale=0.4, size=(25, 2))
    return np.vstack([c1, c2])


def test_divisive_graph_clustering(well_separated_clusters):
    X = well_separated_clusters
    dgc = DivisiveGraphClustering(n_clusters=2, n_neighbors=5, criterion="n_clusters")
    dgc.fit(X)

    # Check fitted state and labels
    assert dgc.is_fitted
    assert len(dgc.labels_) == 50
    assert len(np.unique(dgc.labels_)) == 2

    # Check that the two groups [0:25] and [25:50] are separated into different clusters
    c1_labels = dgc.labels_[:25]
    c2_labels = dgc.labels_[25:]
    assert len(np.unique(c1_labels)) == 1
    assert len(np.unique(c2_labels)) == 1
    assert c1_labels[0] != c2_labels[0]

    # Check modularity history recorded
    assert len(dgc.modularity_history_) > 0
    assert dgc.best_modularity_ > 0.0

    # Test out-of-sample prediction
    X_new = np.array([[-3.8, 0.1], [4.1, -0.2]])
    preds = dgc.predict(X_new)
    assert preds[0] == c1_labels[0]
    assert preds[1] == c2_labels[0]
