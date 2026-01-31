"""Multiple comparison correction methods for statistical analysis.

This module provides functions for correcting p-values to control for
multiple comparisons:
- FDR (False Discovery Rate): Benjamini-Hochberg and Benjamini-Yekutieli
- Bonferroni: Family-wise error rate control
- Tmax: Maximum statistic correction from permutation distributions

All functions are designed for MEG/EEG data with spatial dimensions.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def apply_fdr_correction(
    pvals: np.ndarray,
    alpha: float = 0.05,
    method: str = "bh",
) -> np.ndarray:
    """Apply False Discovery Rate (FDR) correction to p-values.

    Implements Benjamini-Hochberg (BH) and Benjamini-Yekutieli (BY) FDR
    correction methods. BH assumes independent or positively correlated tests,
    while BY is more conservative and handles arbitrary dependence.

    Args:
        pvals: P-values, any shape.
        alpha: Significance threshold. Defaults to 0.05.
        method: FDR method ('bh' or 'by'). Defaults to 'bh'.

    Returns:
        Corrected p-values (q-values), same shape as input.

    Examples:
        >>> corrected = apply_fdr_correction(pvals, alpha=0.05, method='bh')
        >>> significant = corrected < 0.05

    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
        Journal of the Royal Statistical Society Series B, 57, 289-300.

        Benjamini, Y., & Yekutieli, D. (2001). The control of the false
        discovery rate in multiple testing under dependency.
        Annals of Statistics, 29, 1165-1188.
    """
    original_shape = pvals.shape
    pvals_flat = pvals.flatten()

    # Remove NaN values
    valid_mask = ~np.isnan(pvals_flat)
    pvals_valid = pvals_flat[valid_mask]

    if len(pvals_valid) == 0:
        logger.warning("No valid p-values found for FDR correction")
        return pvals

    # Number of tests
    m = len(pvals_valid)

    # Sort p-values and get indices
    sort_idx = np.argsort(pvals_valid)
    pvals_sorted = pvals_valid[sort_idx]

    # Create reverse index to restore original order
    reverse_idx = np.argsort(sort_idx)

    # Benjamini-Hochberg threshold
    if method == "bh":
        # BH: assumes independence or positive dependence
        threshold = (np.arange(1, m + 1) / m) * alpha

    elif method == "by":
        # BY: handles arbitrary dependence (more conservative)
        c_m = np.sum(1.0 / np.arange(1, m + 1))  # Harmonic sum
        threshold = (np.arange(1, m + 1) / (m * c_m)) * alpha

    else:
        raise ValueError(f"Unknown FDR method: {method}. Use 'bh' or 'by'.")

    # Find largest i where p(i) <= threshold(i)
    # This is the critical value for rejection
    below_threshold = pvals_sorted <= threshold
    if np.any(below_threshold):
        # Find the largest index where condition holds
        critical_idx = np.where(below_threshold)[0][-1]
        critical_pval = pvals_sorted[critical_idx]
    else:
        critical_pval = 0.0  # No rejections

    # Compute q-values (adjusted p-values)
    # q(i) = min(p(i) * m / i, 1)
    qvals_sorted = np.minimum(
        pvals_sorted * m / np.arange(1, m + 1),
        1.0
    )

    # Enforce monotonicity (q-values should not decrease)
    for i in range(m - 2, -1, -1):
        qvals_sorted[i] = min(qvals_sorted[i], qvals_sorted[i + 1])

    # Restore original order
    qvals_valid = qvals_sorted[reverse_idx]

    # Put back into full array with NaNs
    qvals_flat = np.full(len(pvals_flat), np.nan)
    qvals_flat[valid_mask] = qvals_valid

    # Reshape to original shape
    qvals = qvals_flat.reshape(original_shape)

    n_significant = np.sum(qvals < alpha)
    logger.debug(
        f"FDR-{method.upper()}: {n_significant}/{m} significant tests "
        f"at alpha={alpha} (critical p={critical_pval:.4f})"
    )

    return qvals


def apply_bonferroni_correction(
    pvals: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply Bonferroni correction to p-values.

    Bonferroni correction controls the family-wise error rate (FWER) by
    adjusting the significance threshold to alpha/m, where m is the number
    of tests. This is equivalent to multiplying p-values by m.

    Args:
        pvals: P-values, any shape.
        alpha: Significance threshold. Defaults to 0.05.

    Returns:
        Corrected p-values, same shape as input.

    Examples:
        >>> corrected = apply_bonferroni_correction(pvals, alpha=0.05)
        >>> significant = corrected < 0.05

    References:
        Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo
        delle probabilità. Pubblicazioni del R Istituto Superiore di Scienze
        Economiche e Commerciali di Firenze, 8, 3-62.
    """
    original_shape = pvals.shape
    pvals_flat = pvals.flatten()

    # Count valid (non-NaN) p-values
    valid_mask = ~np.isnan(pvals_flat)
    m = np.sum(valid_mask)

    if m == 0:
        logger.warning("No valid p-values found for Bonferroni correction")
        return pvals

    # Bonferroni correction: multiply p-values by number of tests
    corrected = pvals * m

    # Clip to [0, 1] range
    corrected = np.clip(corrected, 0, 1)

    n_significant = np.sum(corrected < alpha)
    logger.debug(
        f"Bonferroni: {n_significant}/{m} significant tests at alpha={alpha} "
        f"(corrected threshold={alpha/m:.6f})"
    )

    return corrected


def apply_holm_bonferroni_correction(
    pvals: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply Holm-Bonferroni (step-down) correction to p-values.

    Holm-Bonferroni is a more powerful alternative to Bonferroni that still
    controls FWER. It uses a step-down procedure where the threshold depends
    on the rank of each p-value.

    Args:
        pvals: P-values, any shape.
        alpha: Significance threshold. Defaults to 0.05.

    Returns:
        Corrected p-values, same shape as input.

    Examples:
        >>> corrected = apply_holm_bonferroni_correction(pvals, alpha=0.05)
        >>> significant = corrected < 0.05

    References:
        Holm, S. (1979). A simple sequentially rejective multiple test
        procedure. Scandinavian Journal of Statistics, 6, 65-70.
    """
    original_shape = pvals.shape
    pvals_flat = pvals.flatten()

    # Remove NaN values
    valid_mask = ~np.isnan(pvals_flat)
    pvals_valid = pvals_flat[valid_mask]

    if len(pvals_valid) == 0:
        logger.warning("No valid p-values found for Holm-Bonferroni correction")
        return pvals

    m = len(pvals_valid)

    # Sort p-values and get indices
    sort_idx = np.argsort(pvals_valid)
    pvals_sorted = pvals_valid[sort_idx]
    reverse_idx = np.argsort(sort_idx)

    # Compute Holm-Bonferroni adjusted p-values
    # For rank i (0-indexed), multiply by (m - i)
    multipliers = m - np.arange(m)
    adjusted_sorted = pvals_sorted * multipliers

    # Enforce monotonicity (adjusted p-values should not decrease)
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    # Clip to [0, 1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    # Restore original order
    adjusted_valid = adjusted_sorted[reverse_idx]

    # Put back into full array with NaNs
    adjusted_flat = np.full(len(pvals_flat), np.nan)
    adjusted_flat[valid_mask] = adjusted_valid

    # Reshape to original shape
    adjusted = adjusted_flat.reshape(original_shape)

    n_significant = np.sum(adjusted < alpha)
    logger.debug(
        f"Holm-Bonferroni: {n_significant}/{m} significant tests at alpha={alpha}"
    )

    return adjusted


def apply_tmax_correction(
    pvals: np.ndarray,
    perm_scores: np.ndarray,
    observed_scores: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply maximum statistic (tmax) correction using permutation distribution.

    Tmax correction controls FWER by comparing each test to the maximum
    statistic across all tests in the permutation distribution. This is more
    powerful than Bonferroni when tests are correlated.

    Args:
        pvals: Observed p-values, shape (n_features, n_spatial).
        perm_scores: Permutation scores, shape (n_features, n_perms, n_spatial).
        observed_scores: Observed scores, shape (n_features, n_spatial).
        alpha: Significance threshold. Defaults to 0.05.

    Returns:
        Corrected p-values, shape (n_features, n_spatial).

    Examples:
        >>> corrected = apply_tmax_correction(
        ...     pvals, perm_scores, observed_scores, alpha=0.05
        ... )

    References:
        Blair, R. C., & Karniski, W. (1993). An alternative method for
        significance testing of waveform difference potentials.
        Psychophysiology, 30, 518-524.
    """
    from code.utils.statistics import apply_tmax

    # Use existing tmax implementation
    all_results = {
        "scores": observed_scores,
        "perm_scores": perm_scores,
    }

    corrected = apply_tmax(all_results)

    n_significant = np.sum(corrected < alpha)
    n_total = np.sum(~np.isnan(corrected))

    logger.debug(
        f"Tmax: {n_significant}/{n_total} significant tests at alpha={alpha}"
    )

    return corrected


def cluster_based_permutation_correction(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05,
    n_permutations: int = 10000,
    threshold: Optional[float] = None,
    adjacency: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply cluster-based permutation correction (placeholder).

    Cluster-based permutation testing identifies clusters of adjacent
    significant tests and evaluates their significance using permutation
    testing. This is particularly useful for spatially or temporally
    structured data.

    Note: This is a placeholder for future implementation using MNE-Python's
    cluster-based permutation testing.

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels, shape (n_trials,).
        groups: Subject indices, shape (n_trials,).
        alpha: Significance threshold. Defaults to 0.05.
        n_permutations: Number of permutations. Defaults to 10000.
        threshold: Cluster-forming threshold. Defaults to None (uses t-distribution).
        adjacency: Spatial adjacency matrix. Defaults to None.

    Returns:
        Tuple containing:
        - cluster_pvals: P-values for each cluster
        - cluster_labels: Cluster assignment for each test

    Examples:
        >>> cluster_pvals, labels = cluster_based_permutation_correction(
        ...     X, y, groups, n_permutations=10000
        ... )
    """
    logger.warning(
        "Cluster-based permutation correction not yet implemented. "
        "Consider using MNE-Python's cluster_level.permutation_cluster_test()"
    )

    # Placeholder: return uncorrected p-values
    from code.utils.statistics import subject_contrast

    _, _, pvals = subject_contrast(X, y)

    return pvals, np.zeros_like(pvals)


def compare_correction_methods(
    pvals: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Compare different multiple comparison correction methods.

    Applies all available correction methods and reports the number of
    significant tests for each.

    Args:
        pvals: P-values, any shape.
        alpha: Significance threshold. Defaults to 0.05.

    Returns:
        Dictionary mapping method name to corrected p-values and counts.

    Examples:
        >>> results = compare_correction_methods(pvals, alpha=0.05)
        >>> for method, info in results.items():
        ...     print(f"{method}: {info['n_significant']} significant")
    """
    results = {}

    # Uncorrected
    n_uncorrected = np.sum(pvals < alpha)
    results["uncorrected"] = {
        "pvals": pvals,
        "n_significant": n_uncorrected,
    }

    # FDR (BH)
    fdr_bh = apply_fdr_correction(pvals, alpha, method="bh")
    results["fdr_bh"] = {
        "pvals": fdr_bh,
        "n_significant": np.sum(fdr_bh < alpha),
    }

    # FDR (BY)
    fdr_by = apply_fdr_correction(pvals, alpha, method="by")
    results["fdr_by"] = {
        "pvals": fdr_by,
        "n_significant": np.sum(fdr_by < alpha),
    }

    # Bonferroni
    bonferroni = apply_bonferroni_correction(pvals, alpha)
    results["bonferroni"] = {
        "pvals": bonferroni,
        "n_significant": np.sum(bonferroni < alpha),
    }

    # Holm-Bonferroni
    holm = apply_holm_bonferroni_correction(pvals, alpha)
    results["holm_bonferroni"] = {
        "pvals": holm,
        "n_significant": np.sum(holm < alpha),
    }

    # Log comparison
    logger.info("Correction method comparison:")
    for method, info in results.items():
        logger.info(f"  {method}: {info['n_significant']} significant tests")

    return results
