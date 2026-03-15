"""
Visualization
==============
Publication-quality plots for embedding analyses: scree plots, dendrograms,
similarity heatmaps, UMAP projections, and factor loading displays.

All functions return matplotlib Figure objects for further customization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from scipy.cluster.hierarchy import dendrogram


# ── Style Defaults ───────────────────────────────────────────────────────────

def set_publication_style():
    """Set matplotlib defaults for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 6),
    })


# ── Scree Plots ──────────────────────────────────────────────────────────────


def plot_scree(
    eigenvalues: np.ndarray,
    simulated_eigenvalues: Optional[np.ndarray] = None,
    max_components: int = 15,
    title: str = "Scree Plot with Parallel Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scree plot with optional parallel analysis threshold line.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Observed eigenvalues from EFA or SVD.
    simulated_eigenvalues : np.ndarray, optional
        Simulated eigenvalue thresholds from parallel_analysis().
    max_components : int
        Number of components to display.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = min(max_components, len(eigenvalues))
    x = range(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, eigenvalues[:n], "bo-", label="Observed eigenvalues", linewidth=2)

    if simulated_eigenvalues is not None:
        ax.plot(
            x, simulated_eigenvalues[:n], "r--",
            label="Parallel analysis threshold (95th %ile)",
            linewidth=1.5,
        )

    ax.set_xlabel("Factor Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(list(x))
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_variance_explained(
    explained_variance_ratio: np.ndarray,
    max_components: int = 15,
    title: str = "Variance Explained by SVD Components",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar + cumulative line chart for SVD variance explained.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        From reduce_svd()['explained_variance_ratio'].
    """
    n = min(max_components, len(explained_variance_ratio))
    x = range(1, n + 1)
    cumulative = np.cumsum(explained_variance_ratio[:n])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(x, explained_variance_ratio[:n], alpha=0.7, label="Individual")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Proportion of Variance")
    ax1.set_xticks(list(x))

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, "ro-", label="Cumulative", linewidth=2)
    ax2.set_ylabel("Cumulative Proportion")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Dendrograms ──────────────────────────────────────────────────────────────


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str],
    title: str = "Hierarchical Clustering Dendrogram",
    orientation: str = "right",
    color_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Publication-quality dendrogram.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        From cluster_hierarchical()['linkage_matrix'].
    labels : list of str
        Leaf labels.
    orientation : str
        'right', 'left', 'top', 'bottom'.
    color_threshold : float, optional
        Distance at which to color branches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation=orientation,
        leaf_font_size=10,
        color_threshold=color_threshold,
        ax=ax,
    )

    ax.set_title(title)
    if orientation in ["right", "left"]:
        ax.set_xlabel("Distance")
    else:
        ax.set_ylabel("Distance")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Heatmaps ─────────────────────────────────────────────────────────────────


def plot_similarity_heatmap(
    similarity_matrix: pd.DataFrame,
    title: str = "Cosine Similarity Matrix",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 10),
    annotate: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of cosine similarity matrix.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square similarity matrix.
    annotate : bool
        Show values in cells (disable for large matrices).
    """
    fig, ax = plt.subplots(figsize=figsize)

    fmt = ".2f" if similarity_matrix.shape[0] <= 20 else ""
    sns.heatmap(
        similarity_matrix,
        annot=annotate and similarity_matrix.shape[0] <= 20,
        fmt=fmt,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_loading_heatmap(
    loadings: pd.DataFrame,
    title: str = "Factor Loadings",
    threshold: float = 0.40,
    cmap: str = "RdBu_r",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of factor loadings with salient loadings highlighted.

    Parameters
    ----------
    loadings : pd.DataFrame
        Factor loadings from run_efa().
    threshold : float
        Loadings above this (absolute) are considered salient.
    """
    if figsize is None:
        h = max(6, len(loadings) * 0.35)
        w = max(4, len(loadings.columns) * 1.5 + 2)
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )

    # Bold salient loadings
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            if abs(loadings.iloc[i, j]) >= threshold:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1, fill=False, edgecolor="black", linewidth=2
                    )
                )

    ax.set_title(title)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── UMAP Projection Plots ───────────────────────────────────────────────────


def plot_umap_clusters(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP Cluster Visualization",
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.3,
    noise_color: str = "lightgray",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2D scatter plot of UMAP-reduced embeddings colored by cluster.

    Parameters
    ----------
    reduced_embeddings : np.ndarray
        Shape (n_docs, >=2). First 2 dimensions used.
    labels : np.ndarray
        Cluster assignments (-1 = noise).
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=noise_color,
                alpha=0.1,
                s=5,
                label="Noise",
            )
        else:
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=[colors[i]],
                alpha=alpha,
                s=10,
                label=f"Cluster {label}",
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if len(unique_labels) <= 15:
        ax.legend(markerscale=3, loc="best")
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Bass-ackwards Visualization ──────────────────────────────────────────────


def plot_bass_ackwards(
    congruence_matrices: Dict[Tuple[int, int], pd.DataFrame],
    congruence_threshold: float = 0.80,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the bass-ackwards factor hierarchy as a flow diagram.

    Shows how factors at level k map to factors at level k+1 based
    on Tucker's congruence coefficients.

    Parameters
    ----------
    congruence_matrices : dict
        Output from bass_ackwards()['congruence_matrices'].
    congruence_threshold : float
        Minimum congruence to draw a connection.
    """
    fig, ax = plt.subplots(figsize=figsize)

    levels = sorted(set([k for pair in congruence_matrices.keys() for k in pair]))
    max_k = max(levels)

    # Position factors vertically per level
    positions = {}
    for level in levels:
        n_factors = level
        for i in range(n_factors):
            y = (i - (n_factors - 1) / 2) * 1.5
            positions[(level, i)] = (level * 3, y)

    # Draw nodes
    for (level, factor_idx), (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.4, color="steelblue", alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, f"F{factor_idx+1}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    # Draw connections based on congruence
    for (k, k1), cong_df in congruence_matrices.items():
        for i in range(cong_df.shape[0]):
            for j in range(cong_df.shape[1]):
                cong = abs(cong_df.iloc[i, j])
                if cong >= congruence_threshold:
                    x1, y1 = positions[(k, i)]
                    x2, y2 = positions[(k1, j)]
                    ax.annotate(
                        "",
                        xy=(x2 - 0.4, y2),
                        xytext=(x1 + 0.4, y1),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=1 + 2 * cong,
                            color="gray",
                            alpha=0.6 + 0.4 * cong,
                        ),
                    )
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y + 0.2, f"{cong:.2f}",
                            fontsize=7, ha="center", color="gray")

    # Level labels
    for level in levels:
        x = level * 3
        ax.text(x, max_k * 0.8 + 1, f"{level}-Factor", ha="center",
                fontsize=11, fontweight="bold")

    ax.set_xlim(min(levels) * 3 - 1, max(levels) * 3 + 1)
    ax.set_ylim(-max_k, max_k + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Bass-Ackwards Factor Hierarchy", fontsize=14, pad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
