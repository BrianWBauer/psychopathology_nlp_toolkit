"""
Psychopathology NLP Toolkit
============================
A modular, reproducible toolkit for analyzing mental health language data
using transformer-based embeddings.

Companion code for:
    Bauer et al. "Transformer-Based Text Embeddings for Identifying
    Higher-Order Constructs in Psychopathology: A Practical Tutorial"

Modules
-------
embeddings : Embedding generation, model selection, and centroid aggregation
reduction  : Dimensionality reduction (UMAP, SVD, PCA)
clustering : Clustering algorithms (HDBSCAN, hierarchical, k-means)
similarity : Cosine similarity matrices and distance computations
factor_analysis : EFA, bass-ackwards, parallel analysis
interpretation : ProtoDash prototype selection, c-TF-IDF, LLM interpretation
classification : Supervised classification on labeled embeddings
visualization : Scree plots, dendrograms, heatmaps, embedding projections
"""

__version__ = "0.1.0"
