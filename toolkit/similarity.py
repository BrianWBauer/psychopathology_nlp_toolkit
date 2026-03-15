"""
Cosine Similarity Matrices
============================
Compute and manipulate cosine similarity matrices for use in
factor analysis and dimensional mapping.

Key recommendations:
    - Center embeddings to grand mean before computing similarity (Goal 3)
    - Use cosine similarity (not Euclidean distance) for embedding comparisons
    - Inspect the similarity matrix for anomalous values before factor analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(
    embedding_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    center: bool = True,
) -> pd.DataFrame:
    """
    Compute a cosine similarity matrix from centroid embeddings.

    Used in Goal 3 as input to EFA.

    Parameters
    ----------
    embedding_matrix : np.ndarray
        Shape (n_groups, embedding_dim).
    labels : list of str, optional
        Row/column labels.
    center : bool
        Mean-center embeddings before computing similarity.
        Recommended for EFA applications (removes shared variance
        due to overall mean).

    Returns
    -------
    pd.DataFrame
        n_groups × n_groups similarity matrix.
    """
    matrix = embedding_matrix.copy()
    if center:
        matrix = matrix - matrix.mean(axis=0)

    sim = cosine_similarity(matrix)

    if labels is None:
        labels = [f"group_{i}" for i in range(len(matrix))]

    sim_df = pd.DataFrame(sim, index=labels, columns=labels)

    # Diagnostics
    off_diag = sim[np.triu_indices_from(sim, k=1)]
    print(f"Similarity matrix: {sim.shape[0]} × {sim.shape[1]}")
    print(f"  Off-diagonal range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
    print(f"  Off-diagonal mean: {off_diag.mean():.3f}")
    print(f"  Off-diagonal SD: {off_diag.std():.3f}")

    return sim_df


def similarity_to_distance(sim_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert cosine similarity to cosine distance (1 - similarity)."""
    return 1 - sim_matrix


def _compute_kmo_from_correlation(corr_matrix: np.ndarray) -> tuple:
    """
    Compute KMO statistic directly from a correlation/similarity matrix.

    This avoids the bug in factor_analyzer.calculate_kmo, which expects
    a raw data matrix (observations × variables) and internally computes
    correlations. Passing a pre-computed similarity matrix to that function
    produces a correlation-of-correlations, yielding meaningless output.

    KMO is computed from partial correlations derived from the inverse of
    the similarity matrix (anti-image correlation matrix).

    KMO_j = Σ_{i≠j} r²_ij / [Σ_{i≠j} r²_ij + Σ_{i≠j} q²_ij]
    KMO_overall = ΣΣ_{i≠j} r²_ij / [ΣΣ_{i≠j} r²_ij + ΣΣ_{i≠j} q²_ij]

    where r_ij are elements of the similarity matrix and q_ij are partial
    correlations from the anti-image correlation matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Square similarity/correlation matrix.

    Returns
    -------
    kmo_per_variable : np.ndarray
        KMO value for each variable/indicator.
    kmo_overall : float
        Overall KMO statistic.
    """
    n = corr_matrix.shape[0]

    # Compute inverse (anti-image covariance matrix)
    try:
        corr_inv = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for singular matrices
        corr_inv = np.linalg.pinv(corr_matrix)

    # Convert anti-image covariance to anti-image correlation
    # q_ij = -S_inv_ij / sqrt(S_inv_ii * S_inv_jj)
    diag = np.diag(corr_inv)
    diag_sqrt = np.sqrt(np.abs(diag))
    diag_sqrt[diag_sqrt < 1e-10] = 1e-10  # prevent division by zero
    anti_image_corr = -corr_inv / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(anti_image_corr, 1.0)

    # Compute KMO per variable and overall (off-diagonal elements only)
    r_sq = corr_matrix ** 2
    q_sq = anti_image_corr ** 2
    np.fill_diagonal(r_sq, 0)
    np.fill_diagonal(q_sq, 0)

    r_sq_sum = r_sq.sum(axis=1)
    q_sq_sum = q_sq.sum(axis=1)

    # Per-variable KMO
    denom = r_sq_sum + q_sq_sum
    kmo_per_variable = np.where(denom > 1e-10, r_sq_sum / denom, 0.0)

    # Overall KMO
    total_r_sq = r_sq.sum()
    total_q_sq = q_sq.sum()
    total_denom = total_r_sq + total_q_sq
    kmo_overall = total_r_sq / total_denom if total_denom > 1e-10 else 0.0

    return kmo_per_variable, kmo_overall


def check_matrix_properties(
    sim_matrix: pd.DataFrame,
    min_acceptable_kmo: float = 0.60,
) -> dict:
    """
    Check if a similarity matrix is suitable for factor analysis.

    Evaluates: positive semi-definiteness, condition number, determinant,
    eigenvalue distribution, and KMO (computed directly from the similarity
    matrix via its inverse, not via the factor_analyzer package).

    Note on KMO and Bartlett's:
        Standard implementations (e.g., factor_analyzer.calculate_kmo and
        calculate_bartlett_sphericity) expect a raw data matrix
        (observations × variables) and internally compute correlations.
        Passing a pre-computed similarity matrix to those functions produces
        a correlation-of-correlations, yielding meaningless results.

        KMO is computed here directly from the similarity matrix inverse
        to obtain partial correlations. Bartlett's test is omitted because
        its chi-square approximation requires a subject-level sample size N,
        which has no clear analogue for centroid-derived similarity matrices.

    Parameters
    ----------
    sim_matrix : pd.DataFrame
        Square similarity matrix (treated as correlation-like input).
    min_acceptable_kmo : float
        Minimum KMO value for adequate factorability (0.60 is 'mediocre';
        Kaiser, 1974).

    Returns
    -------
    dict with diagnostic results.
    """
    matrix = sim_matrix.values
    n = matrix.shape[0]
    labels = sim_matrix.index.tolist()

    # ── Positive semi-definiteness ───────────────────────────────────────
    eigenvalues = np.linalg.eigvalsh(matrix)
    is_psd = bool(np.all(eigenvalues >= -1e-10))

    # ── Condition number ─────────────────────────────────────────────────
    # High values indicate multicollinearity. Rules of thumb:
    #   <30: acceptable; 30-1000: moderate concern; >1000: severe
    cond_number = float(np.linalg.cond(matrix))

    # ── Determinant ──────────────────────────────────────────────────────
    # Near-zero flags multicollinearity / near-singularity.
    det = float(np.linalg.det(matrix))

    # ── Eigenvalue distribution ──────────────────────────────────────────
    sorted_eigs = np.sort(eigenvalues)[::-1]
    n_positive = int(np.sum(sorted_eigs > 1e-10))
    cumulative_var = np.cumsum(sorted_eigs / sorted_eigs.sum())

    # ── KMO (custom, computed from matrix inverse) ───────────────────────
    kmo_per_variable = None
    kmo_overall = None
    kmo_adequate = None
    try:
        kmo_pv, kmo_overall = _compute_kmo_from_correlation(matrix)
        kmo_adequate = kmo_overall >= min_acceptable_kmo
        kmo_per_variable = pd.Series(kmo_pv, index=labels, name="KMO")
    except Exception as e:
        print(f"  KMO computation failed (matrix may be singular): {e}")

    # ── Assemble results ─────────────────────────────────────────────────
    result = {
        "is_positive_semidefinite": is_psd,
        "min_eigenvalue": float(eigenvalues.min()),
        "condition_number": cond_number,
        "determinant": det,
        "n_positive_eigenvalues": n_positive,
        "eigenvalues": sorted_eigs,
        "cumulative_variance": cumulative_var,
        "kmo_overall": float(kmo_overall) if kmo_overall is not None else None,
        "kmo_adequate": kmo_adequate,
        "kmo_per_variable": kmo_per_variable,
    }

    # ── Print report ─────────────────────────────────────────────────────
    print("Matrix diagnostics:")
    print(f"  Positive semi-definite: {is_psd}")
    if not is_psd:
        print(f"    Min eigenvalue: {eigenvalues.min():.6f}")
        print("    WARNING: Matrix is not PSD. EFA may produce Heywood cases.")
    print(f"  Condition number: {cond_number:.1f}", end="")
    if cond_number > 1000:
        print(" [SEVERE - near-singular]")
    elif cond_number > 30:
        print(" [elevated - possible multicollinearity]")
    else:
        print(" [acceptable]")
    print(f"  Determinant: {det:.6f}", end="")
    if abs(det) < 1e-5:
        print(" [near-zero - multicollinearity concern]")
    else:
        print()
    print(f"  Positive eigenvalues: {n_positive}/{n}")
    print(f"  Variance in first 3 components: {cumulative_var[min(2, n-1)]:.3f}")
    if kmo_overall is not None:
        print(f"  KMO overall: {kmo_overall:.3f}", end="")
        if kmo_overall >= 0.80:
            print(" [meritorious]")
        elif kmo_overall >= 0.70:
            print(" [middling]")
        elif kmo_overall >= 0.60:
            print(" [mediocre]")
        elif kmo_overall >= 0.50:
            print(" [miserable]")
        else:
            print(" [unacceptable - factorability questionable]")
        # Flag low per-variable KMOs
        if kmo_per_variable is not None:
            low_kmo = kmo_per_variable[kmo_per_variable < min_acceptable_kmo]
            if len(low_kmo) > 0:
                print(f"  Low KMO indicators ({len(low_kmo)}):")
                for idx, val in low_kmo.items():
                    print(f"    {idx}: {val:.3f}")
    print(
        f"  Bartlett's test: omitted (requires subject-level N; not applicable "
        f"to centroid-derived similarity matrices)"
    )

    return result
