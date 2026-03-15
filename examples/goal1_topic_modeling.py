"""
Goal 1: Identifying Emergent Themes / Topic Modeling
=====================================================
Complete BERTopic pipeline from the manuscript worked example.

Pipeline: Embeddings → UMAP → HDBSCAN → c-TF-IDF → ProtoDash + LLM

This example mirrors the manuscript's analysis of open-ended survey
responses about perceptions of the youth mental health crisis
(Sappenfield et al., in prep).

Usage:
    python goal1_topic_modeling.py --input data.csv --text_col response
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Pipeline ─────────────────────────────────────────────────────────────────


def run_topic_modeling_pipeline(
    texts: list,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    # UMAP parameters (manuscript defaults)
    umap_n_components: int = 5,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = 15,
    hdbscan_min_samples: int = None,
    # Interpretation
    n_prototypes: int = 5,
    n_keywords: int = 10,
    # LLM interpretation (optional)
    use_llm: bool = False,
    llm_provider: str = "anthropic",
    llm_api_key: str = None,
    # Output
    output_dir: str = "output/goal1",
):
    """
    Run the full Goal 1 pipeline.

    Parameters
    ----------
    texts : list of str
        Input documents (e.g., open-ended survey responses).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from toolkit.embeddings import generate_embeddings, load_model, check_text_lengths
    from toolkit.reduction import reduce_umap
    from toolkit.clustering import cluster_hdbscan
    from toolkit.interpretation import (
        select_prototypes_protodash,
        select_prototypes,
        compute_ctfidf,
        format_prototypes_for_llm,
        interpret_with_llm,
    )
    from toolkit.visualization import (
        set_publication_style,
        plot_umap_clusters,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    # ── Step 0: Preprocessing ────────────────────────────────────────────
    print("=" * 60)
    print("GOAL 1: TOPIC MODELING PIPELINE")
    print("=" * 60)

    # Remove duplicates and short responses
    original_n = len(texts)
    texts = [t for t in texts if isinstance(t, str) and len(t.split()) >= 5]
    texts = list(set(texts))  # Remove duplicates
    print(f"\nPreprocessing: {original_n} → {len(texts)} texts "
          f"(removed {original_n - len(texts)} duplicates/short)")

    # ── Step 1: Generate Embeddings ──────────────────────────────────────
    print("\n--- Step 1: Embedding Generation ---")
    model = load_model(embedding_model_name)
    length_stats = check_text_lengths(texts, model)
    print(length_stats.to_string(index=False))

    embeddings = generate_embeddings(texts, model=model)

    # ── Step 2: UMAP Dimensionality Reduction ────────────────────────────
    print("\n--- Step 2: UMAP Dimensionality Reduction ---")
    reduced = reduce_umap(
        embeddings,
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
    )

    # Also compute 2D reduction for visualization
    reduced_2d = reduce_umap(
        embeddings,
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=0.1,  # Slightly higher for visualization
        verbose=False,
    )

    # ── Step 3: HDBSCAN Clustering ───────────────────────────────────────
    print("\n--- Step 3: HDBSCAN Clustering ---")
    cluster_result = cluster_hdbscan(
        reduced,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
    )

    labels = cluster_result["labels"]
    print(f"\nNoise ratio interpretation:")
    noise = cluster_result["noise_ratio"]
    if noise < 0.05:
        print("  Very low noise — clusters may be too inclusive")
    elif noise < 0.15:
        print("  Good — strong thematic coherence")
    elif noise < 0.30:
        print("  Moderate — some responses don't fit clear themes")
    else:
        print("  High — consider adjusting parameters")

    # ── Step 4: c-TF-IDF Keywords ────────────────────────────────────────
    print("\n--- Step 4: c-TF-IDF Keyword Extraction ---")
    keywords = compute_ctfidf(
        texts, labels, n_keywords=n_keywords
    )

    # ── Step 5: Prototype Selection ──────────────────────────────────────
    print("\n--- Step 5: Prototype Selection (ProtoDash) ---")
    try:
        prototypes = select_prototypes_protodash(
            embeddings, texts, labels, n_prototypes=n_prototypes
        )
    except ImportError:
        print("  aix360 not installed; falling back to cosine-centroid selection.")
        print("  Install aix360 for manuscript-compliant ProtoDash: pip install aix360")
        prototypes = select_prototypes(
            embeddings, texts, labels, n_prototypes=n_prototypes
        )

    # Print prototypes
    for cluster_id, proto_df in prototypes.items():
        print(f"\n  Cluster {cluster_id}:")
        score_col = "importance_weight" if "importance_weight" in proto_df.columns else "similarity"
        for _, row in proto_df.iterrows():
            print(f"    [{row[score_col]:.3f}] {row['text'][:100]}...")

    # ── Step 6: LLM Interpretation (optional) ────────────────────────────
    interpretations = {}
    if use_llm:
        print("\n--- Step 6: LLM Interpretation ---")
        for cluster_id in prototypes:
            prompt = format_prototypes_for_llm(prototypes, cluster_id, keywords)
            interpretation = interpret_with_llm(
                prompt, provider=llm_provider, api_key=llm_api_key
            )
            interpretations[cluster_id] = interpretation
            print(f"\n  Cluster {cluster_id}: {interpretation[:200]}...")

    # ── Visualization ────────────────────────────────────────────────────
    print("\n--- Generating Visualizations ---")
    fig = plot_umap_clusters(
        reduced_2d, labels,
        title="Topic Model: UMAP Projection with HDBSCAN Clusters",
        save_path=str(output_path / "umap_clusters.png"),
    )

    # ── Save Results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "text": texts,
        "cluster": labels,
    })
    results_df.to_csv(output_path / "cluster_assignments.csv", index=False)

    # Save keywords
    kw_rows = []
    for cluster_id, kws in keywords.items():
        for kw, score in kws:
            kw_rows.append({"cluster": cluster_id, "keyword": kw, "score": score})
    pd.DataFrame(kw_rows).to_csv(output_path / "keywords.csv", index=False)

    print(f"\nResults saved to {output_path}/")
    print("=" * 60)

    return {
        "embeddings": embeddings,
        "reduced": reduced,
        "cluster_result": cluster_result,
        "keywords": keywords,
        "prototypes": prototypes,
        "interpretations": interpretations,
    }


# ── BERTopic Convenience Wrapper ─────────────────────────────────────────────


def run_bertopic_native(
    texts: list,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    hdbscan_min_cluster_size: int = 15,
    n_keywords: int = 10,
) -> "BERTopic":
    """
    Run BERTopic directly (alternative to the modular pipeline above).

    This uses BERTopic's built-in pipeline with custom parameters.

    Returns
    -------
    BERTopic model
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Configure each component
    embedding_model = SentenceTransformer(embedding_model_name)

    umap_model = UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=n_keywords,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts)

    print(f"\nBERTopic results:")
    print(f"  Topics found: {len(set(topics)) - (1 if -1 in topics else 0)}")
    print(f"  Noise ratio: {(np.array(topics) == -1).mean():.1%}")
    print(f"\nTopic overview:")
    print(topic_model.get_topic_info())

    return topic_model


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goal 1: Topic Modeling Pipeline")
    parser.add_argument("--input", required=True, help="Path to CSV with text data")
    parser.add_argument("--text_col", default="text", help="Column name for text")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--min_cluster_size", type=int, default=15)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--output_dir", default="output/goal1")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_provider", default="anthropic")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    texts = df[args.text_col].dropna().tolist()

    run_topic_modeling_pipeline(
        texts=texts,
        embedding_model_name=args.model,
        hdbscan_min_cluster_size=args.min_cluster_size,
        umap_n_neighbors=args.n_neighbors,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
    )
