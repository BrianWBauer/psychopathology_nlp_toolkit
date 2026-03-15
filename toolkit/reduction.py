"""
Dimensionality Reduction
=========================
UMAP, SVD, and PCA for reducing high-dimensional embeddings.

Key recommendations from the manuscript:
    - UMAP: Use for BERTopic / clustering pipelines (nonlinear, preserves local structure)
    - SVD/PCA: Use for dimensional analysis and factor analysis (linear, preserves global structure)
    - Always use cosine distance metric for embedding data
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.decomposition import PCA, TruncatedSVD
import warnings


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP.

    Default parameters follow BERTopic conventions. The manuscript recommends
    iterative tuning starting from these defaults.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_documents, embedding_dim).
    n_components : int
        Output dimensionality. Default 5 works for most datasets up to ~100k.
        Use 10-15 for >100k documents.
    n_neighbors : int
        Controls local vs. global structure trade-off.
        - Lower (5-10): finer-grained, more specific topics
        - Higher (30-50): broader, more general topics
        - Default 15: good starting point
    min_dist : float
        Controls cluster compactness. 0.0 for clustering (default).
        Use 0.1-0.3 for visualization.
    metric : str
        Distance metric. Always use 'cosine' for embeddings.
    random_state : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_documents, n_components).
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")

    if verbose:
        print(
            f"UMAP: {embeddings.shape[1]}d → {n_components}d "
            f"(n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})"
        )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)

    if verbose:
        print(f"  Output shape: {reduced.shape}")
    return reduced


def reduce_svd(
    embedding_matrix: np.ndarray,
    n_components: Optional[int] = None,
    center: bool = True,
    labels: Optional[list] = None,
) -> dict:
    """
    Reduce embeddings using SVD (or PCA when centered).

    Used in Goal 2 for dimensional analysis of centroid embeddings.

    Parameters
    ----------
    embedding_matrix : np.ndarray
        Shape (n_groups, embedding_dim). Typically centroid embeddings.
    n_components : int, optional
        Number of components to extract. Defaults to min(n_groups, 20).
    center : bool
        Mean-center the matrix before decomposition (equivalent to PCA).
    labels : list, optional
        Group labels for interpretability output.

    Returns
    -------
    dict with keys:
        'scores' : np.ndarray, shape (n_groups, n_components)
            Group scores on each dimension.
        'components' : np.ndarray, shape (n_components, embedding_dim)
            Principal axes in embedding space.
        'explained_variance' : np.ndarray
            Variance explained by each component.
        'explained_variance_ratio' : np.ndarray
            Proportion of variance explained by each component.
        'cumulative_variance' : np.ndarray
            Cumulative proportion of variance explained.
        'scores_df' : pd.DataFrame
            Labeled scores for easy inspection.
    """
    if n_components is None:
        n_components = min(embedding_matrix.shape[0], 20)

    matrix = embedding_matrix.copy()
    if center:
        matrix = matrix - matrix.mean(axis=0)

    # Use PCA for centered data (more numerically stable)
    if center:
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(matrix)
        components = pca.components_
        explained_var = pca.explained_variance_
        explained_ratio = pca.explained_variance_ratio_
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        scores = svd.fit_transform(matrix)
        components = svd.components_
        explained_var = svd.explained_variance_
        explained_ratio = svd.explained_variance_ratio_

    cumulative = np.cumsum(explained_ratio)

    # Build labeled DataFrame
    if labels is None:
        labels = [f"group_{i}" for i in range(embedding_matrix.shape[0])]

    col_names = [f"Dim{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=labels, columns=col_names)

    result = {
        "scores": scores,
        "components": components,
        "explained_variance": explained_var,
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance": cumulative,
        "scores_df": scores_df,
    }

    print(f"SVD: extracted {n_components} components")
    print(f"  Top 5 variance explained: {explained_ratio[:5].round(3)}")
    print(f"  Cumulative at 5 components: {cumulative[min(4, len(cumulative)-1)]:.3f}")

    return result


def get_extreme_groups(
    scores_df: pd.DataFrame,
    dimension: int = 1,
    n_extreme: int = 5,
) -> pd.DataFrame:
    """
    Extract groups with the most extreme scores on a given SVD dimension.

    Used for interpreting what each dimension represents (Goal 2).

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output from reduce_svd()['scores_df'].
    dimension : int
        Which dimension (1-indexed).
    n_extreme : int
        Number of groups at each pole.

    Returns
    -------
    pd.DataFrame
        Groups at positive and negative extremes with scores.
    """
    col = f"Dim{dimension}"
    sorted_df = scores_df[[col]].sort_values(col)

    negative_pole = sorted_df.head(n_extreme).copy()
    negative_pole["pole"] = "negative"
    positive_pole = sorted_df.tail(n_extreme).copy()
    positive_pole["pole"] = "positive"

    result = pd.concat([negative_pole, positive_pole])
    result["rank"] = range(1, len(result) + 1)
    return result
