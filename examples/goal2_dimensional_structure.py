"""
Goal 2: Revealing Dimensional Structure and Relationships
==========================================================
SVD + Hierarchical clustering pipeline from the manuscript.

Pipeline: Embeddings → Centroid Aggregation → SVD → Hierarchical Clustering
          → ProtoDash + LLM Interpretation

This example mirrors the manuscript's analysis of mental health
discourse on Reddit (Bauer et al., 2024), examining dimensional
structure and taxonomic relationships across 30 subreddits.

Usage:
    python goal2_dimensional_structure.py --input data.csv \
        --text_col text --group_col subreddit
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def run_dimensional_structure_pipeline(
    texts: list,
    group_labels: list,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    n_svd_components: int = 10,
    n_hierarchical_clusters: int = 3,
    n_prototypes: int = 5,
    use_llm: bool = False,
    llm_provider: str = "anthropic",
    llm_api_key: str = None,
    output_dir: str = "output/goal2",
):
    """
    Run the full Goal 2 pipeline.

    Parameters
    ----------
    texts : list of str
        All documents (e.g., Reddit posts).
    group_labels : list of str
        Group label per document (e.g., subreddit name).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from toolkit.embeddings import generate_embeddings, load_model, compute_centroids, centroids_to_matrix
    from toolkit.reduction import reduce_svd, get_extreme_groups
    from toolkit.clustering import cluster_hierarchical
    from toolkit.interpretation import (
        select_prototypes_protodash_for_groups,
        select_prototypes_for_groups,
        interpret_with_llm,
    )
    from toolkit.visualization import (
        set_publication_style,
        plot_scree,
        plot_variance_explained,
        plot_dendrogram,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    print("=" * 60)
    print("GOAL 2: DIMENSIONAL STRUCTURE PIPELINE")
    print("=" * 60)

    group_labels = np.asarray(group_labels)
    unique_groups = np.unique(group_labels)
    print(f"\n{len(texts)} documents across {len(unique_groups)} groups")
    for g in unique_groups:
        n = (group_labels == g).sum()
        print(f"  {g}: {n:,} documents")

    # ── Step 1: Generate Embeddings ──────────────────────────────────────
    print("\n--- Step 1: Embedding Generation ---")
    model = load_model(embedding_model_name)
    embeddings = generate_embeddings(texts, model=model)

    # ── Step 2: Compute Centroids ────────────────────────────────────────
    print("\n--- Step 2: Centroid Aggregation ---")
    centroids = compute_centroids(embeddings, group_labels)
    labels, centroid_matrix = centroids_to_matrix(centroids)
    print(f"  Centroid matrix shape: {centroid_matrix.shape}")

    # ── Step 3: SVD Dimensional Analysis ─────────────────────────────────
    print("\n--- Step 3: SVD Dimensional Analysis ---")
    svd_result = reduce_svd(
        centroid_matrix,
        n_components=n_svd_components,
        center=True,
        labels=labels,
    )

    # Interpret dimensions via extreme groups
    print("\n  Dimension Poles:")
    for dim in range(1, min(4, n_svd_components + 1)):
        extremes = get_extreme_groups(svd_result["scores_df"], dimension=dim, n_extreme=3)
        neg = extremes[extremes["pole"] == "negative"].index.tolist()
        pos = extremes[extremes["pole"] == "positive"].index.tolist()
        print(f"    Dim {dim}: [{', '.join(neg)}] ←→ [{', '.join(pos)}]")

    # Scree plot
    plot_variance_explained(
        svd_result["explained_variance_ratio"],
        title="SVD Variance Explained",
        save_path=str(output_path / "svd_scree.png"),
    )

    # ── Step 4: Hierarchical Clustering ──────────────────────────────────
    print("\n--- Step 4: Hierarchical Clustering ---")
    hc_result = cluster_hierarchical(
        centroid_matrix,
        labels=labels,
        method="ward",
        n_clusters=n_hierarchical_clusters,
    )

    # Dendrogram
    plot_dendrogram(
        hc_result["linkage_matrix"],
        labels=labels,
        title="Hierarchical Clustering of Groups",
        save_path=str(output_path / "dendrogram.png"),
    )

    # ── Step 5: Prototype Selection ──────────────────────────────────────
    print("\n--- Step 5: Prototype Selection (ProtoDash) ---")
    try:
        prototypes = select_prototypes_protodash_for_groups(
            embeddings, texts, group_labels, n_prototypes=n_prototypes
        )
    except ImportError:
        print("  aix360 not installed; falling back to cosine-centroid selection.")
        print("  Install aix360 for manuscript-compliant ProtoDash: pip install aix360")
        prototypes = select_prototypes_for_groups(
            embeddings, texts, group_labels, n_prototypes=n_prototypes
        )

    # Print sample prototypes for first 3 groups
    for group in list(prototypes.keys())[:3]:
        print(f"\n  {group} prototypes:")
        score_col = "importance_weight" if "importance_weight" in prototypes[group].columns else "similarity"
        for _, row in prototypes[group].head(2).iterrows():
            print(f"    [{row[score_col]:.3f}] {row['text'][:100]}...")

    # ── Step 6: LLM Interpretation (optional) ────────────────────────────
    dim_interpretations = {}
    if use_llm:
        print("\n--- Step 6: LLM Interpretation of SVD Dimensions ---")
        for dim in range(1, min(4, n_svd_components + 1)):
            extremes = get_extreme_groups(svd_result["scores_df"], dimension=dim, n_extreme=3)

            prompt_parts = [
                f"Below are groups at the extreme poles of Dimension {dim} from "
                f"an SVD analysis of mental health language data.",
                f"",
                f"Please describe what this dimension represents.",
                f"",
            ]

            for pole in ["negative", "positive"]:
                pole_groups = extremes[extremes["pole"] == pole].index.tolist()
                prompt_parts.append(f"{'NEGATIVE' if pole == 'negative' else 'POSITIVE'} pole:")
                for g in pole_groups:
                    if g in prototypes:
                        prompt_parts.append(f"\n  {g}:")
                        for _, row in prototypes[g].head(2).iterrows():
                            prompt_parts.append(f"    - {row['text'][:200]}")
                prompt_parts.append("")

            interpretation = interpret_with_llm(
                "\n".join(prompt_parts),
                provider=llm_provider,
                api_key=llm_api_key,
            )
            dim_interpretations[dim] = interpretation
            print(f"\n  Dim {dim}: {interpretation[:200]}...")

    # ── Save Results ─────────────────────────────────────────────────────
    svd_result["scores_df"].to_csv(output_path / "svd_scores.csv")

    if "cluster_assignments" in hc_result:
        cluster_df = pd.DataFrame({
            "group": labels,
            "cluster": hc_result["cluster_assignments"],
        })
        cluster_df.to_csv(output_path / "cluster_assignments.csv", index=False)

    print(f"\nResults saved to {output_path}/")
    print("=" * 60)

    return {
        "embeddings": embeddings,
        "centroids": centroids,
        "svd_result": svd_result,
        "hc_result": hc_result,
        "prototypes": prototypes,
        "dim_interpretations": dim_interpretations,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Goal 2: Dimensional Structure Pipeline"
    )
    parser.add_argument("--input", required=True, help="CSV with text data")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--group_col", required=True, help="Column for group labels")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--n_svd", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--output_dir", default="output/goal2")
    parser.add_argument("--use_llm", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    texts = df[args.text_col].dropna().tolist()
    groups = df[args.group_col].dropna().tolist()

    run_dimensional_structure_pipeline(
        texts=texts,
        group_labels=groups,
        embedding_model_name=args.model,
        n_svd_components=args.n_svd,
        n_hierarchical_clusters=args.n_clusters,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
    )
