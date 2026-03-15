"""
Factor Analysis
================
Exploratory Factor Analysis (EFA), parallel analysis, and the
bass-ackwards procedure for testing hierarchical construct structure.

Key recommendations from the manuscript:
    - Use ULS (unweighted least squares) estimation for similarity matrices
      (ML assumes multivariate normality; cosine similarities may violate this)
    - Use Promax rotation (oblique) to allow correlated factors
    - Use parallel analysis + scree + interpretability for factor retention
    - Bass-ackwards: extract 1-factor through k-factor solutions and trace splits
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import warnings

# ── Compatibility shim ───────────────────────────────────────────────────────
# factor_analyzer 0.5.1 uses sklearn's deprecated `force_all_finite` parameter
# (renamed to `ensure_all_finite` in sklearn ≥1.6). This shim patches the
# import so factor_analyzer works with current sklearn versions.
try:
    import factor_analyzer.factor_analyzer as _fam
    import sklearn.utils.validation as _val

    _original_check = _val.check_array

    def _patched_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _original_check(*args, **kwargs)

    _fam.check_array = _patched_check_array
except (ImportError, AttributeError):
    pass  # If either library changes internals, skip silently

from factor_analyzer import FactorAnalyzer


def run_efa(
    similarity_matrix: pd.DataFrame,
    n_factors: int,
    method: str = "minres",
    rotation: str = "promax",
    is_corr_matrix: bool = True,
) -> dict:
    """
    Run Exploratory Factor Analysis on a similarity/correlation matrix.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square matrix (n_indicators × n_indicators).
    n_factors : int
        Number of factors to extract.
    method : str
        Extraction method. 'minres' (ULS) recommended for NLP similarity matrices.
        Options: 'minres', 'ml', 'principal'.
    rotation : str
        Rotation method. 'promax' (oblique) recommended to allow correlated factors.
        Options: 'promax', 'varimax', 'oblimin', None.
    is_corr_matrix : bool
        If True, input is treated as a correlation/similarity matrix.

    Returns
    -------
    dict with keys:
        'loadings' : pd.DataFrame, factor loadings matrix
        'factor_correlations' : pd.DataFrame (if oblique rotation)
        'communalities' : pd.Series
        'eigenvalues' : np.ndarray
        'variance_explained' : pd.DataFrame
        'model' : FactorAnalyzer object
    """
    labels = similarity_matrix.index.tolist()

    fa = FactorAnalyzer(
        n_factors=n_factors,
        method=method,
        rotation=rotation,
        is_corr_matrix=is_corr_matrix,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa.fit(similarity_matrix)

    # Loadings
    loadings = pd.DataFrame(
        fa.loadings_,
        index=labels,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )

    # Communalities
    communalities = pd.Series(
        fa.get_communalities(), index=labels, name="communality"
    )

    # Eigenvalues
    eigenvalues = fa.get_eigenvalues()

    # Variance explained
    var_exp = fa.get_factor_variance()
    variance_df = pd.DataFrame(
        var_exp,
        index=["SS_Loadings", "Proportion_Variance", "Cumulative_Variance"],
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )

    result = {
        "loadings": loadings,
        "communalities": communalities,
        "eigenvalues_original": eigenvalues[0],
        "eigenvalues_common": eigenvalues[1],
        "variance_explained": variance_df,
        "model": fa,
    }

    # Factor correlations (oblique rotations only)
    if rotation in ["promax", "oblimin"]:
        phi = fa.phi_
        if phi is not None:
            factor_corr = pd.DataFrame(
                phi,
                index=[f"Factor{i+1}" for i in range(n_factors)],
                columns=[f"Factor{i+1}" for i in range(n_factors)],
            )
            result["factor_correlations"] = factor_corr

    # Heywood case check
    heywood = communalities[communalities > 1.0]
    if len(heywood) > 0:
        print(
            f"WARNING: Heywood case detected. Indicators with communality > 1.0: "
            f"{heywood.index.tolist()}"
        )

    print(f"EFA: {n_factors} factors, method={method}, rotation={rotation}")
    print(f"  Cumulative variance: {variance_df.iloc[2, -1]:.3f}")

    return result


def parallel_analysis(
    similarity_matrix: pd.DataFrame,
    n_simulations: int = 1000,
    percentile: int = 95,
    method: str = "minres",
    random_state: int = 42,
) -> dict:
    """
    Horn's parallel analysis for determining the number of factors to retain.

    Generates random correlation matrices of the same size and compares
    eigenvalues to the observed matrix.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square matrix.
    n_simulations : int
        Number of random matrices to generate.
    percentile : int
        Percentile of simulated eigenvalues to use as threshold.
    method : str
        Factor extraction method.

    Returns
    -------
    dict with keys:
        'n_factors_suggested' : int
        'observed_eigenvalues' : np.ndarray
        'simulated_eigenvalues' : np.ndarray (percentile thresholds)
        'comparison_df' : pd.DataFrame
    """
    np.random.seed(random_state)
    n = similarity_matrix.shape[0]

    # Observed eigenvalues
    fa = FactorAnalyzer(n_factors=1, method=method, is_corr_matrix=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa.fit(similarity_matrix)
    observed = fa.get_eigenvalues()[0]

    # Simulated eigenvalues
    sim_eigenvalues = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        random_data = np.random.normal(size=(max(n * 10, 100), n))
        random_corr = np.corrcoef(random_data.T)
        sim_eigenvalues[i] = np.linalg.eigvalsh(random_corr)[::-1]

    threshold = np.percentile(sim_eigenvalues, percentile, axis=0)

    # How many factors to retain
    n_factors = int(np.sum(observed > threshold))

    comparison = pd.DataFrame(
        {
            "component": range(1, n + 1),
            "observed_eigenvalue": observed,
            f"simulated_p{percentile}": threshold,
            "retain": observed > threshold,
        }
    )

    print(f"Parallel analysis: retain {n_factors} factors")
    print(comparison[comparison["retain"]].to_string(index=False))

    return {
        "n_factors_suggested": n_factors,
        "observed_eigenvalues": observed,
        "simulated_eigenvalues": threshold,
        "comparison_df": comparison,
    }


def bass_ackwards(
    similarity_matrix: pd.DataFrame,
    max_factors: int = 6,
    method: str = "minres",
    rotation: str = "promax",
) -> dict:
    """
    Bass-ackwards procedure (Goldberg, 2006) for examining hierarchical
    factor structure.

    Extracts solutions from 1 through max_factors and computes factor
    congruence coefficients between adjacent levels to trace how factors
    split at each level.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square similarity matrix.
    max_factors : int
        Maximum number of factors to extract.
    method : str
        EFA extraction method.
    rotation : str
        EFA rotation method.

    Returns
    -------
    dict with keys:
        'solutions' : dict of {n_factors: EFA result dict}
        'congruence_matrices' : dict of {(k, k+1): pd.DataFrame}
            Tucker's congruence coefficients between k and k+1 factor solutions.
        'split_summary' : list of str
            Human-readable description of factor splits at each level.
    """

    def _tucker_congruence(loadings_a: np.ndarray, loadings_b: np.ndarray) -> np.ndarray:
        """Compute Tucker's congruence coefficient matrix."""
        # Columns of A vs columns of B
        n_a = loadings_a.shape[1]
        n_b = loadings_b.shape[1]
        congruence = np.zeros((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                a = loadings_a[:, i]
                b = loadings_b[:, j]
                num = np.dot(a, b)
                denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
                congruence[i, j] = num / denom if denom > 0 else 0
        return congruence

    solutions = {}
    congruence_matrices = {}
    split_summary = []

    for k in range(1, max_factors + 1):
        try:
            solutions[k] = run_efa(
                similarity_matrix,
                n_factors=k,
                method=method,
                rotation=rotation if k > 1 else None,
            )
        except Exception as e:
            print(f"  Could not extract {k}-factor solution: {e}")
            break

    # Compute congruence between adjacent solutions
    for k in range(1, max_factors):
        if k not in solutions or (k + 1) not in solutions:
            continue

        loadings_k = solutions[k]["loadings"].values
        loadings_k1 = solutions[k + 1]["loadings"].values

        cong = _tucker_congruence(loadings_k, loadings_k1)

        cong_df = pd.DataFrame(
            cong,
            index=[f"{k}F_Factor{i+1}" for i in range(k)],
            columns=[f"{k+1}F_Factor{j+1}" for j in range(k + 1)],
        )
        congruence_matrices[(k, k + 1)] = cong_df

        # Identify splits: for each factor in k, find its highest-congruence
        # match(es) in k+1
        for i in range(k):
            matches = np.where(np.abs(cong[i]) > 0.80)[0]
            if len(matches) > 1:
                match_labels = [f"Factor{m+1}" for m in matches]
                split_summary.append(
                    f"  {k}→{k+1}F: Factor{i+1} splits into {match_labels}"
                )
            elif len(matches) == 1:
                split_summary.append(
                    f"  {k}→{k+1}F: Factor{i+1} → Factor{matches[0]+1} "
                    f"(congruence={cong[i, matches[0]]:.3f})"
                )

    print("\nBass-ackwards split summary:")
    for line in split_summary:
        print(line)

    return {
        "solutions": solutions,
        "congruence_matrices": congruence_matrices,
        "split_summary": split_summary,
    }


def get_salient_loadings(
    loadings: pd.DataFrame,
    threshold: float = 0.40,
) -> dict:
    """
    Extract indicators with salient loadings (>threshold) on each factor.

    Parameters
    ----------
    loadings : pd.DataFrame
        Factor loadings matrix from run_efa().
    threshold : float
        Minimum absolute loading to be considered salient.

    Returns
    -------
    dict
        {factor_name: DataFrame of salient indicators sorted by loading}
    """
    result = {}
    for col in loadings.columns:
        salient = loadings[loadings[col].abs() >= threshold][[col]]
        salient = salient.sort_values(col, ascending=False)
        result[col] = salient
        print(f"{col}: {len(salient)} salient indicators (|loading| >= {threshold})")
        for idx, row in salient.iterrows():
            print(f"  {idx}: {row[col]:.3f}")
    return result
