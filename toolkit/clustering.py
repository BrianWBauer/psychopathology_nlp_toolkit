"""
Clustering Algorithms
======================
HDBSCAN, hierarchical (agglomerative), and k-means clustering for
identifying discrete groups in embedding space.

Key recommendations from the manuscript:
    - HDBSCAN: Use with BERTopic (Goal 1); automatically determines k
    - Hierarchical (Ward's): Use for taxonomic analysis (Goal 2)
    - k-means: Use when k is known a priori
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    verbose: bool = True,
) -> dict:
    """
    Cluster embeddings using HDBSCAN.

    Used in Goal 1 (BERTopic pipeline). Automatically determines
    the number of clusters and assigns noise label (-1).

    Parameters
    ----------
    embeddings : np.ndarray
        Typically UMAP-reduced embeddings, shape (n_docs, n_components).
    min_cluster_size : int
        Minimum number of documents to form a topic.
        - Small datasets (<5k): 10-15
        - Medium datasets (5-50k): 15-30
        - Large datasets (>50k): 30-100
    min_samples : int, optional
        Controls cluster density. Defaults to min_cluster_size.
        Lower values = more inclusive clusters; higher = stricter.
    metric : str
        Distance metric. 'euclidean' for UMAP-reduced data.

    Returns
    -------
    dict with keys:
        'labels' : np.ndarray of cluster assignments (-1 = noise)
        'n_clusters' : int
        'noise_ratio' : float, proportion assigned to noise
        'cluster_sizes' : pd.Series
        'probabilities' : np.ndarray, membership probabilities
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("Install hdbscan: pip install hdbscan")

    if min_samples is None:
        min_samples = min_cluster_size

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    probs = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).mean()
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    if verbose:
        print(f"HDBSCAN: {n_clusters} clusters found")
        print(f"  Noise ratio: {noise_ratio:.1%}")
        print(f"  Cluster sizes: {dict(cluster_sizes[cluster_sizes.index != -1])}")
        if noise_ratio > 0.30:
            print(
                "  WARNING: >30% noise. Consider reducing min_cluster_size "
                "or min_samples."
            )
        elif noise_ratio < 0.03:
            print(
                "  WARNING: <3% noise. Clusters may be too inclusive. "
                "Consider increasing min_cluster_size."
            )

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "cluster_sizes": cluster_sizes,
        "probabilities": probs,
    }


def cluster_hierarchical(
    embedding_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = "ward",
    metric: str = "euclidean",
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> dict:
    """
    Hierarchical agglomerative clustering on centroid embeddings.

    Used in Goal 2 for taxonomic analysis (e.g., how subreddits group
    into higher-order categories).

    Parameters
    ----------
    embedding_matrix : np.ndarray
        Shape (n_groups, embedding_dim). Typically centroid embeddings.
    labels : list of str, optional
        Group labels for the dendrogram.
    method : str
        Linkage method. 'ward' recommended (minimizes within-cluster variance).
    metric : str
        Distance metric. Use 'euclidean' with Ward's method.
    n_clusters : int, optional
        Cut dendrogram at this many clusters.
    distance_threshold : float, optional
        Cut dendrogram at this distance.

    Returns
    -------
    dict with keys:
        'linkage_matrix' : np.ndarray for scipy dendrogram
        'labels' : list of str, group labels
        'cluster_assignments' : np.ndarray (if n_clusters or threshold given)
        'distances' : np.ndarray, pairwise distance matrix
    """
    if labels is None:
        labels = [f"group_{i}" for i in range(embedding_matrix.shape[0])]

    # Compute linkage
    Z = linkage(embedding_matrix, method=method, metric=metric)

    result = {
        "linkage_matrix": Z,
        "labels": labels,
    }

    # Compute pairwise distances
    dists = pdist(embedding_matrix, metric=metric)
    result["distances"] = squareform(dists)

    # Optional: cut into flat clusters
    if n_clusters is not None:
        assignments = fcluster(Z, t=n_clusters, criterion="maxclust")
        result["cluster_assignments"] = assignments
        cluster_map = {}
        for label, cluster in zip(labels, assignments):
            cluster_map.setdefault(cluster, []).append(label)
        result["cluster_map"] = cluster_map
        print(f"Hierarchical clustering: {n_clusters} clusters")
        for c, members in sorted(cluster_map.items()):
            print(f"  Cluster {c}: {members}")
    elif distance_threshold is not None:
        assignments = fcluster(Z, t=distance_threshold, criterion="distance")
        result["cluster_assignments"] = assignments

    return result


def evaluate_cluster_solutions(
    embeddings: np.ndarray,
    k_range: range = range(2, 11),
    method: str = "kmeans",
) -> pd.DataFrame:
    """
    Evaluate multiple cluster solutions using silhouette scores.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings (original or UMAP-reduced).
    k_range : range
        Range of cluster counts to evaluate.
    method : str
        'kmeans' or 'hierarchical'.

    Returns
    -------
    pd.DataFrame
        Columns: k, silhouette_score, inertia (kmeans only).
    """
    from sklearn.cluster import KMeans

    results = []
    for k in k_range:
        if method == "kmeans":
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            inertia = km.inertia_
        else:
            Z = linkage(embeddings, method="ward")
            labels = fcluster(Z, t=k, criterion="maxclust")
            inertia = None

        sil = silhouette_score(embeddings, labels)
        results.append({"k": k, "silhouette_score": sil, "inertia": inertia})

    return pd.DataFrame(results)
