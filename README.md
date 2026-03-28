# Psychopathology NLP Toolkit

Companion code for:

> Bauer, B. W., et al. "Transformer-Based Text Embeddings for Identifying Higher-Order Constructs in Psychopathology: A Practical Tutorial." *Journal of Psychopathology and Clinical Science*.

A modular, reproducible Python toolkit for analyzing mental health language data using transformer-based embeddings combined with dimensionality reduction, clustering, factor analysis, and explainable AI.

## Overview

This toolkit implements three analysis pipelines described in the manuscript, each targeting a different research goal. The pipelines share modular components that can be mixed and matched:

```
Raw Text
  │
  ▼
┌─────────────────────────────┐
│  Embedding Generation       │  sentence-transformers (BERT, RoBERTa, etc.)
│  (toolkit/embeddings.py)    │
└──────────┬──────────────────┘
           │
     ┌─────┴──────────────────────────────┐
     │                                    │
     ▼                                    ▼
 Individual                          Centroid
 Embeddings                         Aggregation
     │                                    │
     ▼                              ┌─────┴─────┐
┌────────────┐                      │           │
│ Goal 1     │                      ▼           ▼
│ BERTopic   │               ┌──────────┐ ┌──────────┐
│            │               │ Goal 2   │ │ Goal 3   │
│ UMAP       │               │ SVD/PCA  │ │ Cosine   │
│ HDBSCAN    │               │ Hierarch.│ │ Sim Mtx  │
│ c-TF-IDF   │               │ Cluster  │ │ EFA      │
└────────────┘               └──────────┘ │ Bass-    │
                                          │ ackwards │
                                          └──────────┘
                                    │           │
                                    └─────┬─────┘
                                          │
                                          ▼
                              ┌────────────────────┐
                              │  Interpretation     │
                              │  ProtoDash          │
                              │  LLM Interpretation │
                              └────────────────────┘
```

## Installation

```bash
git clone https://github.com/<your-repo>/psychopathology-nlp-toolkit.git
cd psychopathology-nlp-toolkit
pip install -r requirements.txt
```

**GPU recommended** for embedding generation on large datasets (>10k documents). CPU is fine for small datasets and all downstream analyses.

## Modules

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `toolkit/embeddings.py` | Embedding generation, model selection, centroid aggregation | `generate_embeddings()`, `compute_centroids()`, `check_text_lengths()` |
| `toolkit/reduction.py` | Dimensionality reduction | `reduce_umap()`, `reduce_svd()`, `get_extreme_groups()` |
| `toolkit/clustering.py` | Clustering algorithms | `cluster_hdbscan()`, `cluster_hierarchical()`, `evaluate_cluster_solutions()` |
| `toolkit/similarity.py` | Similarity matrices | `compute_similarity_matrix()`, `check_matrix_properties()` |
| `toolkit/factor_analysis.py` | EFA, parallel analysis, bass-ackwards | `run_efa()`, `parallel_analysis()`, `bass_ackwards()` |
| `toolkit/interpretation.py` | Prototype selection, c-TF-IDF, LLM interpretation | `select_prototypes_protodash()`, `compute_ctfidf()`, `interpret_with_llm()` |
| `toolkit/classification.py` | Supervised classification on embeddings | `train_embedding_classifier()`, `predict_unlabeled()`, `evaluate_label_quality()` |
| `toolkit/visualization.py` | Publication-quality plots | `plot_scree()`, `plot_dendrogram()`, `plot_loading_heatmap()` |

## Quick Start

### Goal 1: Topic Modeling (BERTopic)

```python
from toolkit.embeddings import generate_embeddings
from toolkit.reduction import reduce_umap
from toolkit.clustering import cluster_hdbscan
from toolkit.interpretation import select_prototypes_protodash, compute_ctfidf

# Generate embeddings
embeddings = generate_embeddings(texts, model_name="all-MiniLM-L6-v2")

# UMAP + HDBSCAN
reduced = reduce_umap(embeddings, n_neighbors=15, n_components=5)
result = cluster_hdbscan(reduced, min_cluster_size=15)

# Interpret
keywords = compute_ctfidf(texts, result["labels"])
prototypes = select_prototypes_protodash(embeddings, texts, result["labels"], n_prototypes=5)
```

Or use BERTopic directly:

```python
from examples.goal1_topic_modeling import run_bertopic_native
topic_model = run_bertopic_native(texts, hdbscan_min_cluster_size=15)
```

### Goal 2: Dimensional Structure

```python
from toolkit.embeddings import generate_embeddings, compute_centroids, centroids_to_matrix
from toolkit.reduction import reduce_svd, get_extreme_groups
from toolkit.clustering import cluster_hierarchical

embeddings = generate_embeddings(texts, model_name="all-MiniLM-L6-v2")
centroids = compute_centroids(embeddings, group_labels)
labels, matrix = centroids_to_matrix(centroids)

# SVD
svd = reduce_svd(matrix, n_components=10, center=True, labels=labels)
extremes = get_extreme_groups(svd["scores_df"], dimension=1, n_extreme=5)

# Hierarchical clustering
hc = cluster_hierarchical(matrix, labels=labels, method="ward", n_clusters=3)
```

### Goal 3: Testing Theoretical Hierarchies

```python
from toolkit.similarity import compute_similarity_matrix
from toolkit.factor_analysis import run_efa, parallel_analysis, bass_ackwards

# Cosine similarity matrix (centered)
sim = compute_similarity_matrix(matrix, labels=labels, center=True)

# Determine number of factors
pa = parallel_analysis(sim)

# EFA
efa = run_efa(sim, n_factors=pa["n_factors_suggested"], method="minres", rotation="promax")

# Bass-ackwards hierarchy
ba = bass_ackwards(sim, max_factors=6)
```

### Goal 4: Supervised Classification (Human Labels + Embeddings)

```python
from toolkit.embeddings import generate_embeddings
from toolkit.classification import train_embedding_classifier, predict_unlabeled

# Embed labeled subset (e.g., 300 manually coded posts)
labeled_emb = generate_embeddings(labeled_texts, model_name="all-MiniLM-L6-v2")

# Train with cross-validation
result = train_embedding_classifier(labeled_emb, labels, cv_folds=5)

# Scale to full corpus
unlabeled_emb = generate_embeddings(unlabeled_texts, model_name="all-MiniLM-L6-v2")
predictions = predict_unlabeled(result["model"], unlabeled_emb, texts=unlabeled_texts)
```

## Full Pipeline Scripts

Each goal has a complete worked example with CLI:

```bash
# Goal 1: Topic Modeling
python examples/goal1_topic_modeling.py \
    --input data.csv --text_col response \
    --min_cluster_size 15 --output_dir output/goal1

# Goal 2: Dimensional Structure
python examples/goal2_dimensional_structure.py \
    --input data.csv --text_col text --group_col subreddit \
    --n_svd 10 --n_clusters 3 --output_dir output/goal2

# Goal 3: Testing Hierarchies
python examples/goal3_theoretical_hierarchies.py \
    --input data.csv --text_col text --group_col subreddit \
    --max_factors 6 --output_dir output/goal3

# Goal 4: Supervised Classification
python examples/goal4_supervised_classification.py \
    --labeled_data annotated.csv --label_col target \
    --unlabeled_data full_corpus.csv --text_col text \
    --output_dir output/goal4
```

Add `--use_llm` to enable LLM-based interpretation (requires API key as environment variable).

## Embedding Model Selection

| Use Case | Recommended Model | Max Tokens | Dimensions |
|----------|-------------------|------------|------------|
| Short survey responses | `all-MiniLM-L6-v2` | 256 | 384 |
| General purpose | `all-mpnet-base-v2` | 384 | 768 |
| Long clinical text | `roberta-base` | 512 | 768 |
| Clinical notes | `Bio_ClinicalBERT` | 512 | 768 |
| Mental health text | `mental-bert-base-uncased` | 512 | 768 |

For comprehensive model benchmarks, see the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## Modularity Note

The pipelines are presented as worked examples, not fixed recipes. The components at each step are interchangeable. For instance, Goal 3 could use PCA on individual post embeddings instead of EFA on aggregated centroid embeddings, or clustering of centroid embeddings instead of EFA. Goal 4's supervised approach can be combined with any of the unsupervised pipelines (e.g., use cluster assignments from Goal 1 as training labels, or use classifier predictions as inputs to Goal 2's dimensional analysis). Researchers should select the method at each step that best matches their specific research question.

## Prototype Selection: ProtoDash vs. Cosine-Centroid

The toolkit provides two prototype selection methods:

- **`select_prototypes_protodash()`** — The manuscript-described method. Uses the AIX360 ProtoDash algorithm (Gurumoorthy et al., 2019) with importance-weighted MMD minimization. Produces diverse, non-redundant prototypes. Requires `pip install aix360` (Apache 2.0).
- **`select_prototypes()`** — Lightweight fallback selecting points nearest the cluster centroid. Does not perform importance weighting or redundancy penalization. Use only when aix360 is unavailable.

## Citation

```bibtex
@article{bauer2026transformer,
  title={Transformer-Based Text Embeddings for Identifying Higher-Order 
         Constructs in Psychopathology: A Practical Tutorial},
  author={Bauer, Brian W., Sappenfield, C., Follet, L., Cecchi, G., & Norel, R},
  journal={Journal of Psychopathology and Clinical Science},
  year={2026}
}
```

## License

MIT
