"""
Divisive Graph Clustering implemented from scratch using NumPy.
Builds a kNN similarity graph on PCA-reduced data and applies the Girvan-Newman
algorithm (edge betweenness centrality via BFS Brandes algorithm) to partition
the graph top-down until a target number of clusters or maximum modularity is reached.
"""

from typing import Optional, List, Tuple, Dict, Set
from collections import deque
import numpy as np


class DivisiveGraphClustering:
    """
    Divisive Graph Clustering using k-Nearest Neighbors graph and Girvan-Newman
    edge betweenness partitioning.

    Parameters:
    -----------
    n_clusters : int, default=3
        Target number of communities/clusters.
    n_neighbors : int, default=10
        Number of nearest neighbors to connect in the similarity graph.
    criterion : str, default='n_clusters'
        Stopping criterion: 'n_clusters' or 'max_modularity'.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_neighbors: int = 10,
        criterion: str = "n_clusters",
    ):
        if criterion not in ("n_clusters", "max_modularity"):
            raise ValueError("criterion must be 'n_clusters' or 'max_modularity'.")
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.criterion = criterion

        self.labels_: Optional[np.ndarray] = None
        self.modularity_history_: List[float] = []
        self.n_communities_history_: List[int] = []
        self.X_train_: Optional[np.ndarray] = None
        self.best_modularity_: float = -1.0
        self.is_fitted: bool = False

    def _build_knn_graph(self, X: np.ndarray) -> Dict[int, Set[int]]:
        """Construct symmetric k-NN adjacency list."""
        n_samples = X.shape[0]
        adj: Dict[int, Set[int]] = {i: set() for i in range(n_samples)}

        # Pairwise Euclidean distances
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        sq_dists = np.maximum(0.0, X_sq + X_sq.T - 2.0 * (X @ X.T))

        for i in range(n_samples):
            # Sort nearest neighbors (excluding self at index 0)
            nearest = np.argsort(sq_dists[i])[1 : self.n_neighbors + 1]
            for neighbor in nearest:
                adj[i].add(int(neighbor))
                adj[int(neighbor)].add(i)

        return adj

    def _find_connected_components(self, adj: Dict[int, Set[int]]) -> List[List[int]]:
        """Find connected components in the graph via BFS."""
        visited = set()
        components = []

        for node in adj:
            if node not in visited:
                comp = []
                queue = deque([node])
                visited.add(node)
                while queue:
                    curr = queue.popleft()
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(comp)

        return components

    def _edge_betweenness(
        self, adj: Dict[int, Set[int]]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute edge betweenness centrality for all edges using Brandes BFS algorithm.
        """
        edge_betweenness: Dict[Tuple[int, int], float] = {}
        nodes = list(adj.keys())

        for s in nodes:
            # BFS state
            S = []  # Stack of nodes in order of non-decreasing distance
            P: Dict[int, List[int]] = {w: [] for w in nodes}  # Predecessors
            sigma: Dict[int, float] = {w: 0.0 for w in nodes}  # Number of shortest paths
            sigma[s] = 1.0
            dist: Dict[int, int] = {w: -1 for w in nodes}
            dist[s] = 0

            queue = deque([s])
            while queue:
                v = queue.popleft()
                S.append(v)
                for w in adj[v]:
                    # w found for the first time?
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    # Shortest path to w via v?
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            # Accumulation of dependencies back from the furthest nodes
            delta: Dict[int, float] = {w: 0.0 for w in nodes}
            while S:
                w = S.pop()
                for v in P[w]:
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    edge = (min(v, w), max(v, w))
                    edge_betweenness[edge] = edge_betweenness.get(edge, 0.0) + c
                    delta[v] += c

        # Symmetrize (halve because every pair counted twice for undirected graphs)
        for edge in edge_betweenness:
            edge_betweenness[edge] /= 2.0

        return edge_betweenness

    def _compute_modularity(
        self,
        components: List[List[int]],
        initial_adj: Dict[int, Set[int]],
        initial_m: int,
        initial_degrees: Dict[int, int],
    ) -> float:
        """
        Compute Newman-Girvan modularity Q for the given partition:
            Q = (1 / 2m) sum_c sum_{i,j in c} [ A_{ij} - (k_i * k_j) / (2m) ]
        """
        if initial_m == 0:
            return 0.0

        two_m = 2.0 * initial_m
        Q = 0.0

        for comp in components:
            comp_set = set(comp)
            for i in comp:
                k_i = initial_degrees[i]
                for j in comp:
                    k_j = initial_degrees[j]
                    A_ij = 1.0 if j in initial_adj[i] else 0.0
                    Q += A_ij - (k_i * k_j) / two_m

        return float(Q / two_m)

    def fit(self, X: np.ndarray) -> "DivisiveGraphClustering":
        """
        Fit divisive graph clustering on PCA-reduced data X.
        """
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        self.X_train_ = X

        # 1. Build initial kNN graph
        adj = self._build_knn_graph(X)

        # Record initial topology for modularity calculation
        initial_adj = {i: set(adj[i]) for i in adj}
        initial_degrees = {i: len(adj[i]) for i in adj}
        initial_m = sum(len(adj[i]) for i in adj) // 2

        # Initial components and modularity
        components = self._find_connected_components(adj)
        initial_q = self._compute_modularity(
            components, initial_adj, initial_m, initial_degrees
        )

        self.modularity_history_ = [initial_q]
        self.n_communities_history_ = [len(components)]

        best_components = components
        best_modularity = initial_q

        # Divisive loop
        max_divisions = initial_m
        step = 0

        while step < max_divisions:
            step += 1

            # Termination condition
            current_n_clusters = len(components)
            if self.criterion == "n_clusters" and current_n_clusters >= self.n_clusters:
                best_components = components
                best_modularity = self.modularity_history_[-1]
                break

            # Compute edge betweenness
            betweenness = self._edge_betweenness(adj)
            if not betweenness:
                break

            # Find edge(s) with maximum betweenness
            max_eb = max(betweenness.values())
            # Remove all edges with maximum betweenness (or top edge)
            edges_to_remove = [e for e, eb in betweenness.items() if eb >= max_eb - 1e-9]

            for u, v in edges_to_remove:
                if v in adj[u]:
                    adj[u].remove(v)
                if u in adj[v]:
                    adj[v].remove(u)

            # Re-evaluate components
            new_components = self._find_connected_components(adj)
            new_q = self._compute_modularity(
                new_components, initial_adj, initial_m, initial_degrees
            )

            components = new_components
            self.modularity_history_.append(new_q)
            self.n_communities_history_.append(len(components))

            if new_q > best_modularity:
                best_modularity = new_q
                if self.criterion == "max_modularity":
                    best_components = components

            if self.criterion == "n_clusters" and len(components) >= self.n_clusters:
                best_components = components
                best_modularity = new_q
                break

        # Assign cluster labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, comp in enumerate(best_components[: self.n_clusters]):
            for node in comp:
                labels[node] = cluster_id

        # If there are residual small components beyond n_clusters, merge them with nearest cluster
        if len(best_components) > self.n_clusters:
            # For each extra component, find mean center and assign to closest primary cluster
            centroids = [
                np.mean(X[c], axis=0) for c in best_components[: self.n_clusters]
            ]
            for extra_comp in best_components[self.n_clusters :]:
                for node in extra_comp:
                    dists = [np.linalg.norm(X[node] - c) for c in centroids]
                    labels[node] = int(np.argmin(dists))

        self.labels_ = labels
        self.best_modularity_ = best_modularity
        self.is_fitted = True
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Assign new samples to clusters based on Euclidean nearest neighbor in training set.
        """
        if not self.is_fitted:
            raise ValueError("Graph clustering has not been fitted yet.")
        X_new = np.asarray(X_new, dtype=float)
        # Compute nearest neighbor in X_train_
        preds = np.zeros(X_new.shape[0], dtype=int)
        for i, pt in enumerate(X_new):
            dists = np.sum((self.X_train_ - pt) ** 2, axis=1)
            nearest_idx = int(np.argmin(dists))
            preds[i] = self.labels_[nearest_idx]
        return preds
