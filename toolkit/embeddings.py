"""
Embedding Generation, Model Selection, and Centroid Aggregation
================================================================
Functions for generating transformer-based text embeddings, selecting
appropriate models, and computing centroid (mean) embeddings for groups.

Recommended models by text length (see manuscript Table 1):
    - Short text (<128 tokens): all-MiniLM-L6-v2
    - Medium text (128-512 tokens): all-mpnet-base-v2, all-MiniLM-L12-v2
    - Long text / clinical notes: roberta-base, roberta-large
    - Domain-specific: MentalBERT, ClinicalBERT

All models are available via HuggingFace: https://huggingface.co/models
For embedding model benchmarks, see MTEB leaderboard:
    https://huggingface.co/spaces/mteb/leaderboard
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
from sentence_transformers import SentenceTransformer


# ── Model Selection ──────────────────────────────────────────────────────────

RECOMMENDED_MODELS = {
    "short": {
        "name": "all-MiniLM-L6-v2",
        "max_tokens": 256,
        "dimensions": 384,
        "notes": "Fast, good for short survey responses and social media posts",
    },
    "medium": {
        "name": "all-mpnet-base-v2",
        "max_tokens": 384,
        "dimensions": 768,
        "notes": "Best general-purpose sentence embedding model",
    },
    "long": {
        "name": "roberta-base",
        "max_tokens": 512,
        "dimensions": 768,
        "notes": "Good for longer documents; requires mean pooling",
    },
    "clinical": {
        "name": "emilyalsentzer/Bio_ClinicalBERT",
        "max_tokens": 512,
        "dimensions": 768,
        "notes": "Pre-trained on clinical notes (MIMIC-III); use for PHI-compliant pipelines",
    },
    "mental_health": {
        "name": "mental/mental-bert-base-uncased",
        "max_tokens": 512,
        "dimensions": 768,
        "notes": "Pre-trained on mental health Reddit data",
    },
}


def list_recommended_models() -> pd.DataFrame:
    """Return a DataFrame of recommended models with metadata."""
    rows = []
    for category, info in RECOMMENDED_MODELS.items():
        rows.append(
            {
                "category": category,
                "model_name": info["name"],
                "max_tokens": info["max_tokens"],
                "dimensions": info["dimensions"],
                "notes": info["notes"],
            }
        )
    return pd.DataFrame(rows)


def load_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> SentenceTransformer:
    """
    Load a sentence-transformer model from HuggingFace.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. See RECOMMENDED_MODELS for suggestions.
    device : str, optional
        'cuda', 'cpu', or None (auto-detect).

    Returns
    -------
    SentenceTransformer
        Loaded model ready for encoding.
    """
    model = SentenceTransformer(model_name, device=device)
    print(f"Loaded model: {model_name}")
    print(f"  Embedding dimensions: {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length: {model.max_seq_length}")
    return model


# ── Embedding Generation ─────────────────────────────────────────────────────


def generate_embeddings(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Parameters
    ----------
    texts : list of str
        Input documents/sentences.
    model : SentenceTransformer, optional
        Pre-loaded model. If None, loads model_name.
    model_name : str
        Model to load if model is None.
    batch_size : int
        Encoding batch size. Reduce if GPU OOM.
    show_progress : bool
        Show progress bar during encoding.
    normalize : bool
        L2-normalize embeddings (recommended for cosine similarity).

    Returns
    -------
    np.ndarray
        Shape (n_texts, embedding_dim).
    """
    if model is None:
        model = load_model(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
    )

    print(f"Generated embeddings: {embeddings.shape}")
    return embeddings


# ── Centroid Aggregation ─────────────────────────────────────────────────────


def compute_centroids(
    embeddings: np.ndarray,
    labels: Union[List[str], np.ndarray, pd.Series],
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute mean centroid embeddings for each group/label.

    Used in Goals 2 and 3 to aggregate individual post embeddings
    to the subreddit or construct level.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_documents, embedding_dim).
    labels : array-like
        Group label for each document (e.g., subreddit name).
    normalize : bool
        L2-normalize centroids after averaging.

    Returns
    -------
    dict
        {label: centroid_vector} mapping.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        mask = labels == label
        centroid = embeddings[mask].mean(axis=0)
        if normalize:
            centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid

    print(f"Computed {len(centroids)} centroids from {len(embeddings)} documents")
    return centroids


def centroids_to_matrix(
    centroids: Dict[str, np.ndarray],
) -> tuple:
    """
    Convert centroid dict to a labeled matrix.

    Returns
    -------
    labels : list of str
        Ordered group labels.
    matrix : np.ndarray
        Shape (n_groups, embedding_dim).
    """
    labels = sorted(centroids.keys())
    matrix = np.vstack([centroids[label] for label in labels])
    return labels, matrix


def check_text_lengths(
    texts: List[str],
    model: SentenceTransformer,
    warn_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Check what proportion of texts exceed the model's max token length.

    Parameters
    ----------
    texts : list of str
        Input texts.
    model : SentenceTransformer
        Model to check against.
    warn_threshold : float
        Warn if this proportion of texts are truncated.

    Returns
    -------
    pd.DataFrame
        Summary statistics of token lengths.
    """
    tokenizer = model.tokenizer
    lengths = [len(tokenizer.encode(t)) for t in texts]
    lengths = np.array(lengths)
    max_len = model.max_seq_length

    truncated = (lengths > max_len).mean()
    stats = {
        "n_texts": len(texts),
        "mean_tokens": lengths.mean(),
        "median_tokens": np.median(lengths),
        "max_tokens": lengths.max(),
        "model_max_tokens": max_len,
        "pct_truncated": truncated * 100,
    }

    if truncated > warn_threshold:
        print(
            f"WARNING: {truncated:.1%} of texts exceed model max length "
            f"({max_len} tokens). Consider a model with longer context."
        )

    return pd.DataFrame([stats])
