"""
Interpretation / Explainable AI
=================================
Prototype selection, c-TF-IDF keyword extraction, and LLM-based
interpretation of clusters and factors.

Prototype selection methods:
    - ProtoDash (select_prototypes_protodash): Importance-weighted MMD
      minimization via AIX360. This is the manuscript-described method.
      Requires: pip install aix360
    - Cosine-centroid fallback (select_prototypes): Lightweight alternative
      selecting points nearest the cluster centroid. Does NOT match the
      manuscript's ProtoDash description (no importance weighting, no
      redundancy penalization).

Other recommendations:
    - c-TF-IDF: Class-based TF-IDF per Grootendorst (2022) for keyword extraction
    - LLM interpretation: Set temperature=0 for reproducible outputs;
      use iterative prompting of prototypical examples
    - n_gram_range=(1,3) to capture compound clinical terms
      (e.g., "panic attack", "emotional dysregulation")
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


# ── ProtoDash / Prototype Selection ──────────────────────────────────────────


def select_prototypes(
    embeddings: np.ndarray,
    texts: List[str],
    labels: np.ndarray,
    n_prototypes: int = 5,
    method: str = "cosine_centroid",
) -> Dict[int, pd.DataFrame]:
    """
    Select representative texts using simplified centroid/medoid proximity.

    WARNING: This is NOT the ProtoDash algorithm described in the manuscript.
    This method selects points nearest to the cluster centroid or medoid,
    which is redundancy-agnostic (selected prototypes may be near-duplicates).

    For the manuscript-compliant importance-weighted algorithm that minimizes
    MMD and maximizes coverage, use select_prototypes_protodash() instead.

    This function is retained as a lightweight fallback when aix360 is not
    installed or when speed is prioritized over distributional coverage.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_documents, embedding_dim).
    texts : list of str
        Original text for each document.
    labels : np.ndarray
        Cluster/group assignment for each document.
    n_prototypes : int
        Number of prototypical examples per cluster.
    method : str
        'cosine_centroid': Select texts closest to cluster centroid.
        'medoid': Select text that minimizes mean distance to all others.

    Returns
    -------
    dict
        {cluster_label: DataFrame with columns [text, similarity, index]}
    """
    unique_labels = sorted(set(labels))
    prototypes = {}

    for label in unique_labels:
        if label == -1:  # Skip noise cluster
            continue

        mask = np.where(np.asarray(labels) == label)[0]
        cluster_emb = embeddings[mask]
        cluster_texts = [texts[i] for i in mask]

        if method == "cosine_centroid":
            centroid = cluster_emb.mean(axis=0, keepdims=True)
            sims = cosine_similarity(cluster_emb, centroid).flatten()
            top_idx = np.argsort(sims)[-n_prototypes:][::-1]
        elif method == "medoid":
            sim_matrix = cosine_similarity(cluster_emb)
            mean_sims = sim_matrix.mean(axis=1)
            top_idx = np.argsort(mean_sims)[-n_prototypes:][::-1]
            sims = mean_sims
        else:
            raise ValueError(f"Unknown method: {method}")

        proto_df = pd.DataFrame(
            {
                "text": [cluster_texts[i] for i in top_idx],
                "similarity": sims[top_idx],
                "original_index": mask[top_idx],
            }
        )
        prototypes[label] = proto_df

    print(f"Selected {n_prototypes} prototypes for {len(prototypes)} clusters")
    return prototypes


def select_prototypes_for_groups(
    embeddings: np.ndarray,
    texts: List[str],
    group_labels: np.ndarray,
    n_prototypes: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Select prototypical texts for named groups (e.g., subreddits).

    Convenience wrapper for Goal 2/3 where groups are named strings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_documents, embedding_dim).
    texts : list of str
        Original texts.
    group_labels : array-like
        Group name for each document (e.g., subreddit name).

    Returns
    -------
    dict
        {group_name: DataFrame with columns [text, similarity, index]}
    """
    group_labels = np.asarray(group_labels)
    unique_groups = np.unique(group_labels)
    prototypes = {}

    for group in unique_groups:
        mask = np.where(group_labels == group)[0]
        group_emb = embeddings[mask]
        group_texts = [texts[i] for i in mask]

        centroid = group_emb.mean(axis=0, keepdims=True)
        sims = cosine_similarity(group_emb, centroid).flatten()
        top_idx = np.argsort(sims)[-n_prototypes:][::-1]

        prototypes[group] = pd.DataFrame(
            {
                "text": [group_texts[i] for i in top_idx],
                "similarity": sims[top_idx],
                "original_index": mask[top_idx],
            }
        )

    return prototypes


# ── ProtoDash (AIX360 implementation) ─────────────────────────────────────────


def select_prototypes_protodash(
    embeddings: np.ndarray,
    texts: List[str],
    labels: np.ndarray,
    n_prototypes: int = 5,
) -> Dict[int, pd.DataFrame]:
    """
    Select prototypical texts using the AIX360 ProtoDash algorithm.

    Unlike the simplified cosine-centroid method in select_prototypes(),
    ProtoDash minimizes Maximum Mean Discrepancy (MMD) between the selected
    prototypes and the full cluster distribution using non-negative least
    squares. This produces:
        - Importance weights quantifying each prototype's contribution
        - Diversity-maximizing selections (penalizes redundancy)
        - Better coverage of the cluster's distributional shape

    See: Gurumoorthy et al. (2019). Efficient data representation by
    selecting prototypes with importance weights. AISTATS.

    Requires: pip install aix360

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_documents, embedding_dim).
    texts : list of str
        Original text for each document.
    labels : np.ndarray
        Cluster/group assignment for each document.
    n_prototypes : int
        Number of prototypical examples per cluster.

    Returns
    -------
    dict
        {cluster_label: DataFrame with columns
         [text, importance_weight, original_index]}
    """
    try:
        from aix360.algorithms.protodash import ProtodashExplainer
    except ImportError:
        raise ImportError(
            "ProtoDash requires the AIX360 library.\n"
            "Install with: pip install aix360\n"
            "License: Apache 2.0 (https://github.com/Trusted-AI/AIX360)"
        )

    unique_labels = sorted(set(labels))
    prototypes = {}

    for label in unique_labels:
        if label == -1:
            continue

        mask = np.where(np.asarray(labels) == label)[0]
        cluster_emb = embeddings[mask]
        cluster_texts = [texts[i] for i in mask]

        m = min(n_prototypes, len(cluster_emb))

        explainer = ProtodashExplainer()
        # explain(X_target, X_candidate, m):
        #   X_target = distribution to explain (full cluster)
        #   X_candidate = candidate set to select from (same cluster)
        #   m = number of prototypes
        weights, indices, _ = explainer.explain(
            cluster_emb,
            cluster_emb,
            m=m,
        )

        proto_df = pd.DataFrame(
            {
                "text": [cluster_texts[i] for i in indices],
                "importance_weight": weights,
                "original_index": mask[indices],
            }
        )
        proto_df = proto_df.sort_values("importance_weight", ascending=False)
        prototypes[label] = proto_df

    print(
        f"ProtoDash: selected {n_prototypes} prototypes "
        f"for {len(prototypes)} clusters (MMD-minimized)"
    )
    return prototypes


def select_prototypes_protodash_for_groups(
    embeddings: np.ndarray,
    texts: List[str],
    group_labels: np.ndarray,
    n_prototypes: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    ProtoDash prototype selection for named groups (e.g., subreddits).

    Convenience wrapper for Goal 2/3. Uses importance-weighted MMD
    minimization rather than simple centroid proximity.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_documents, embedding_dim).
    texts : list of str
        Original texts.
    group_labels : array-like
        Group name for each document.
    n_prototypes : int
        Number of prototypes per group.

    Returns
    -------
    dict
        {group_name: DataFrame with columns
         [text, importance_weight, original_index]}
    """
    try:
        from aix360.algorithms.protodash import ProtodashExplainer
    except ImportError:
        raise ImportError(
            "ProtoDash requires the AIX360 library.\n"
            "Install with: pip install aix360\n"
            "License: Apache 2.0 (https://github.com/Trusted-AI/AIX360)"
        )

    group_labels = np.asarray(group_labels)
    unique_groups = np.unique(group_labels)
    prototypes = {}

    explainer = ProtodashExplainer()

    for group in unique_groups:
        mask = np.where(group_labels == group)[0]
        group_emb = embeddings[mask]
        group_texts = [texts[i] for i in mask]

        m = min(n_prototypes, len(group_emb))
        weights, indices, _ = explainer.explain(
            group_emb,
            group_emb,
            m=m,
        )

        proto_df = pd.DataFrame(
            {
                "text": [group_texts[i] for i in indices],
                "importance_weight": weights,
                "original_index": mask[indices],
            }
        )
        proto_df = proto_df.sort_values("importance_weight", ascending=False)
        prototypes[group] = proto_df

    print(
        f"ProtoDash: selected {n_prototypes} prototypes "
        f"for {len(prototypes)} groups (MMD-minimized)"
    )
    return prototypes


# ── c-TF-IDF ─────────────────────────────────────────────────────────────────


def compute_ctfidf(
    texts: List[str],
    labels: np.ndarray,
    n_keywords: int = 10,
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 1,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute class-based TF-IDF (c-TF-IDF) for topic/cluster interpretation.

    Implements the c-TF-IDF formula from Grootendorst (2022):

        c-TF-IDF_{t,c} = (tf_{t,c} / w_c) × log(1 + A / tf_t)

    where:
        tf_{t,c} = frequency of term t in class c
        w_c      = total number of words in class c
        A        = average number of words per class across all classes
        tf_t     = frequency of term t across all classes

    Standard TF-IDF penalizes words appearing in many documents;
    c-TF-IDF instead normalizes by class size to prevent large
    clusters from dominating keyword extraction.

    Parameters
    ----------
    texts : list of str
        All documents.
    labels : np.ndarray
        Cluster assignment for each document.
    n_keywords : int
        Number of top keywords per cluster.
    ngram_range : tuple
        Range of n-grams. Default (1, 3) includes unigrams, bigrams,
        and trigrams per manuscript recommendation for capturing compound
        clinical terms (e.g., "panic attack", "emotional dysregulation").
    min_df : int
        Minimum number of cluster-documents a term must appear in.
        Default 1 (consistent with BERTopic). With few clusters (<10),
        higher values will aggressively filter cluster-specific terms.

    Returns
    -------
    dict
        {cluster_label: [(keyword, score), ...]}
    """
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # Step 1: Concatenate all texts within each cluster into class documents
    cluster_docs = {}
    for label in unique_labels:
        mask = np.asarray(labels) == label
        cluster_docs[label] = " ".join([t for t, m in zip(texts, mask) if m])

    corpus_labels = list(cluster_docs.keys())
    corpus_texts = [cluster_docs[label] for label in corpus_labels]

    # Step 2: Get raw term counts per class using CountVectorizer
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=10000,
        stop_words="english",
    )
    try:
        count_matrix = vectorizer.fit_transform(corpus_texts)
    except ValueError:
        # min_df too high for number of cluster-documents; fall back to 1
        print(
            f"  WARNING: min_df={min_df} pruned all terms "
            f"(only {len(corpus_texts)} cluster-documents). "
            f"Falling back to min_df=1."
        )
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=1,
            max_features=10000,
            stop_words="english",
        )
        count_matrix = vectorizer.fit_transform(corpus_texts)
    feature_names = vectorizer.get_feature_names_out()

    # count_matrix: shape (n_classes, n_features)
    # Convert to dense for easier manipulation
    tf_per_class = count_matrix.toarray().astype(float)  # (n_classes, n_features)

    # Step 3: Compute c-TF-IDF per Grootendorst (2022)
    # w_c: total words in each class
    w_c = tf_per_class.sum(axis=1, keepdims=True)  # (n_classes, 1)
    w_c[w_c == 0] = 1  # prevent division by zero

    # Normalized term frequency within class
    tf_norm = tf_per_class / w_c  # (n_classes, n_features)

    # tf_t: total frequency of term t across all classes
    tf_global = tf_per_class.sum(axis=0)  # (n_features,)
    tf_global[tf_global == 0] = 1  # prevent division by zero

    # A: average number of words per class
    A = w_c.mean()

    # c-TF-IDF = (tf_{t,c} / w_c) * log(1 + A / tf_t)
    idf = np.log(1 + A / tf_global)  # (n_features,)
    ctfidf = tf_norm * idf[np.newaxis, :]  # (n_classes, n_features)

    # Step 4: Extract top keywords per class
    keywords = {}
    for i, label in enumerate(corpus_labels):
        scores = ctfidf[i]
        top_idx = np.argsort(scores)[-n_keywords:][::-1]
        keywords[label] = [
            (feature_names[j], float(scores[j])) for j in top_idx
        ]

    print(f"c-TF-IDF: extracted {n_keywords} keywords for {len(keywords)} clusters")
    for label, kws in keywords.items():
        kw_str = ", ".join([f"{k}({s:.4f})" for k, s in kws[:5]])
        print(f"  Cluster {label}: {kw_str}...")

    return keywords


# ── LLM Interpretation ───────────────────────────────────────────────────────


def format_prototypes_for_llm(
    prototypes: Dict,
    cluster_label: int,
    keywords: Optional[Dict] = None,
) -> str:
    """
    Format prototypical texts and keywords into a prompt for LLM interpretation.

    Parameters
    ----------
    prototypes : dict
        Output from select_prototypes().
    cluster_label : int or str
        Which cluster to format.
    keywords : dict, optional
        Output from compute_ctfidf().

    Returns
    -------
    str
        Formatted prompt text.
    """
    prompt_parts = [
        f"Below are representative texts from a cluster of mental health discussions.",
        f"Please provide:",
        f"1. A concise label (2-5 words) for this cluster",
        f"2. A 2-3 sentence description of the main theme",
        f"3. Key distinguishing features compared to other mental health topics",
        f"",
    ]

    if keywords and cluster_label in keywords:
        kw_str = ", ".join([k for k, s in keywords[cluster_label][:10]])
        prompt_parts.append(f"Top keywords: {kw_str}")
        prompt_parts.append("")

    prompt_parts.append("Representative texts:")
    if cluster_label in prototypes:
        for i, row in prototypes[cluster_label].iterrows():
            text = row["text"][:500]  # Truncate for context window
            prompt_parts.append(f"\n--- Example {i+1} ---")
            prompt_parts.append(text)

    return "\n".join(prompt_parts)


def interpret_with_llm(
    prompt: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Send a cluster interpretation prompt to an LLM.

    Temperature defaults to 0.0 for reproducible outputs, per manuscript
    recommendation: "setting temperature to zero to provide reproducible
    outputs" (see Understanding Embeddings > Interpretation Techniques).

    Parameters
    ----------
    prompt : str
        Formatted prompt from format_prototypes_for_llm().
    provider : str
        'anthropic' or 'openai'.
    model : str, optional
        Model name. Defaults to 'claude-sonnet-4-20250514' for Anthropic
        or 'gpt-4' for OpenAI. Verify that the model string is currently
        active with your provider before running.
    api_key : str, optional
        API key. Falls back to ANTHROPIC_API_KEY or OPENAI_API_KEY
        environment variables respectively.
    temperature : float
        Sampling temperature. Default 0.0 for deterministic/reproducible
        outputs. The manuscript explicitly prescribes temperature=0 for
        all LLM interpretation calls.

    Returns
    -------
    str
        LLM interpretation text.
    """
    if provider == "anthropic":
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            if model is None:
                model = "claude-sonnet-4-20250514"
            response = client.messages.create(
                model=model,
                max_tokens=500,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    elif provider == "openai":
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            if model is None:
                model = "gpt-4"
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")


def interpret_factors_with_llm(
    loadings: pd.DataFrame,
    prototypes: Dict[str, pd.DataFrame],
    factor_name: str,
    threshold: float = 0.40,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Interpret an EFA factor by sending high-loading indicators' prototypical
    texts to an LLM.

    Parameters
    ----------
    loadings : pd.DataFrame
        Factor loadings from run_efa().
    prototypes : dict
        {group_name: DataFrame} from select_prototypes_for_groups().
    factor_name : str
        Column name in loadings (e.g., 'Factor1').
    threshold : float
        Minimum loading to include an indicator.
    provider : str
        LLM provider.
    model : str, optional
        Model name (see interpret_with_llm for defaults).
    api_key : str, optional
        API key.
    temperature : float
        Sampling temperature. Default 0.0 for reproducible outputs.

    Returns
    -------
    str
        LLM interpretation of the factor.
    """
    salient = loadings[loadings[factor_name].abs() >= threshold]
    salient = salient.sort_values(factor_name, ascending=False)

    prompt_parts = [
        f"Below are groups that load highly on a factor from a factor analysis "
        f"of mental health language data. Each group is shown with its loading "
        f"value and representative texts.",
        f"",
        f"Please provide:",
        f"1. A concise label for this factor (2-5 words)",
        f"2. A description of what this factor captures",
        f"3. How the groups relate to each other on this dimension",
        f"",
    ]

    for indicator, row in salient.iterrows():
        loading = row[factor_name]
        prompt_parts.append(f"=== {indicator} (loading: {loading:.3f}) ===")
        if indicator in prototypes:
            for _, proto_row in prototypes[indicator].head(3).iterrows():
                text = proto_row["text"][:300]
                prompt_parts.append(f"  - {text}")
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    return interpret_with_llm(
        prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
