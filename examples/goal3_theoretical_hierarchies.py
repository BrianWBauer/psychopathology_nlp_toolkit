"""
Goal 3: Testing Theoretical Hierarchies
=========================================
EFA + Bass-ackwards pipeline from the manuscript.

Pipeline: Embeddings → Centroids → Cosine Similarity Matrix → EFA
          → Bass-ackwards → ProtoDash + LLM Interpretation

This example mirrors the manuscript's analysis validating the HiTOP
model using natural language from Reddit (Ringwald et al., under review).

Usage:
    python goal3_theoretical_hierarchies.py --input data.csv \
        --text_col text --group_col subreddit --max_factors 6
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def run_theoretical_hierarchy_pipeline(
    texts: list,
    group_labels: list,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    # EFA parameters
    max_factors: int = 6,
    efa_method: str = "minres",
    efa_rotation: str = "promax",
    loading_threshold: float = 0.40,
    # Prototype selection
    n_prototypes: int = 5,
    # LLM
    use_llm: bool = False,
    llm_provider: str = "anthropic",
    llm_api_key: str = None,
    # Output
    output_dir: str = "output/goal3",
):
    """
    Run the full Goal 3 pipeline.

    Parameters
    ----------
    texts : list of str
        All documents.
    group_labels : list of str
        Group/indicator label for each document (e.g., subreddit name).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from toolkit.embeddings import (
        generate_embeddings, load_model,
        compute_centroids, centroids_to_matrix,
    )
    from toolkit.similarity import compute_similarity_matrix, check_matrix_properties
    from toolkit.factor_analysis import (
        run_efa, parallel_analysis, bass_ackwards, get_salient_loadings,
    )
    from toolkit.interpretation import (
        select_prototypes_protodash_for_groups,
        select_prototypes_for_groups,
        interpret_factors_with_llm,
    )
    from toolkit.visualization import (
        set_publication_style,
        plot_scree,
        plot_similarity_heatmap,
        plot_loading_heatmap,
        plot_bass_ackwards,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    print("=" * 60)
    print("GOAL 3: TESTING THEORETICAL HIERARCHIES")
    print("=" * 60)

    group_labels = np.asarray(group_labels)
    unique_groups = np.unique(group_labels)
    print(f"\n{len(texts)} documents across {len(unique_groups)} indicators")

    # ── Step 1: Generate Embeddings ──────────────────────────────────────
    print("\n--- Step 1: Embedding Generation ---")
    model = load_model(embedding_model_name)
    embeddings = generate_embeddings(texts, model=model)

    # ── Step 2: Compute Centroids ────────────────────────────────────────
    print("\n--- Step 2: Centroid Aggregation ---")
    centroids = compute_centroids(embeddings, group_labels)
    labels, centroid_matrix = centroids_to_matrix(centroids)
    print(f"  {len(labels)} indicator centroids, {centroid_matrix.shape[1]}d")

    # ── Step 3: Cosine Similarity Matrix ─────────────────────────────────
    print("\n--- Step 3: Cosine Similarity Matrix ---")
    sim_matrix = compute_similarity_matrix(
        centroid_matrix, labels=labels, center=True
    )

    # Diagnostics
    diagnostics = check_matrix_properties(sim_matrix)

    # Heatmap
    plot_similarity_heatmap(
        sim_matrix,
        title="Cosine Similarity Matrix (Centered)",
        save_path=str(output_path / "similarity_matrix.png"),
    )

    # ── Step 4: Parallel Analysis ────────────────────────────────────────
    print("\n--- Step 4: Parallel Analysis ---")
    pa_result = parallel_analysis(
        sim_matrix, method=efa_method
    )

    n_factors_suggested = pa_result["n_factors_suggested"]
    print(f"  Suggested factors: {n_factors_suggested}")

    # Scree plot with parallel analysis
    plot_scree(
        pa_result["observed_eigenvalues"],
        simulated_eigenvalues=pa_result["simulated_eigenvalues"],
        title="Scree Plot with Parallel Analysis",
        save_path=str(output_path / "parallel_analysis_scree.png"),
    )

    # ── Step 5: EFA ──────────────────────────────────────────────────────
    print(f"\n--- Step 5: EFA ({n_factors_suggested}-factor solution) ---")
    efa_result = run_efa(
        sim_matrix,
        n_factors=n_factors_suggested,
        method=efa_method,
        rotation=efa_rotation,
    )

    # Display loadings
    print(f"\n  Factor Loadings (threshold = {loading_threshold}):")
    salient = get_salient_loadings(efa_result["loadings"], threshold=loading_threshold)

    # Loading heatmap
    plot_loading_heatmap(
        efa_result["loadings"],
        title=f"Factor Loadings ({n_factors_suggested}-Factor Solution)",
        threshold=loading_threshold,
        save_path=str(output_path / "loadings_heatmap.png"),
    )

    # Factor correlations (if oblique rotation)
    if "factor_correlations" in efa_result:
        print(f"\n  Factor Correlations:")
        print(efa_result["factor_correlations"].round(3).to_string())

    # ── Step 6: Bass-ackwards ────────────────────────────────────────────
    print(f"\n--- Step 6: Bass-ackwards (1 to {max_factors} factors) ---")
    ba_result = bass_ackwards(
        sim_matrix,
        max_factors=max_factors,
        method=efa_method,
        rotation=efa_rotation,
    )

    # Bass-ackwards visualization
    if ba_result["congruence_matrices"]:
        plot_bass_ackwards(
            ba_result["congruence_matrices"],
            save_path=str(output_path / "bass_ackwards.png"),
        )

    # ── Step 7: Prototype Selection ──────────────────────────────────────
    print("\n--- Step 7: Prototype Selection (ProtoDash) ---")
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

    # ── Step 8: LLM Factor Interpretation (optional) ─────────────────────
    factor_interpretations = {}
    if use_llm:
        print("\n--- Step 8: LLM Factor Interpretation ---")
        for factor_name in efa_result["loadings"].columns:
            interpretation = interpret_factors_with_llm(
                efa_result["loadings"],
                prototypes,
                factor_name,
                threshold=loading_threshold,
                provider=llm_provider,
                api_key=llm_api_key,
            )
            factor_interpretations[factor_name] = interpretation
            print(f"\n  {factor_name}: {interpretation[:200]}...")

    # ── Save Results ─────────────────────────────────────────────────────
    sim_matrix.to_csv(output_path / "similarity_matrix.csv")
    efa_result["loadings"].to_csv(output_path / "factor_loadings.csv")
    efa_result["variance_explained"].to_csv(output_path / "variance_explained.csv")
    efa_result["communalities"].to_csv(output_path / "communalities.csv")

    if "factor_correlations" in efa_result:
        efa_result["factor_correlations"].to_csv(
            output_path / "factor_correlations.csv"
        )

    # Save bass-ackwards congruence matrices
    for (k, k1), cong in ba_result["congruence_matrices"].items():
        cong.to_csv(output_path / f"congruence_{k}to{k1}.csv")

    # Save parallel analysis
    pa_result["comparison_df"].to_csv(output_path / "parallel_analysis.csv", index=False)

    print(f"\nResults saved to {output_path}/")
    print("=" * 60)

    return {
        "embeddings": embeddings,
        "centroids": centroids,
        "similarity_matrix": sim_matrix,
        "parallel_analysis": pa_result,
        "efa_result": efa_result,
        "bass_ackwards": ba_result,
        "prototypes": prototypes,
        "factor_interpretations": factor_interpretations,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Goal 3: Testing Theoretical Hierarchies"
    )
    parser.add_argument("--input", required=True, help="CSV with text data")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--group_col", required=True, help="Column for indicator labels")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--max_factors", type=int, default=6)
    parser.add_argument("--efa_method", default="minres")
    parser.add_argument("--efa_rotation", default="promax")
    parser.add_argument("--output_dir", default="output/goal3")
    parser.add_argument("--use_llm", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    texts = df[args.text_col].dropna().tolist()
    groups = df[args.group_col].dropna().tolist()

    run_theoretical_hierarchy_pipeline(
        texts=texts,
        group_labels=groups,
        embedding_model_name=args.model,
        max_factors=args.max_factors,
        efa_method=args.efa_method,
        efa_rotation=args.efa_rotation,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
    )
