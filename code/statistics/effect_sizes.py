"""Effect size computations for statistical analysis.

This module provides functions for computing various effect size measures:
- Cohen's d: Standardized mean difference
- Hedges' g: Bias-corrected Cohen's d for small samples
- Eta-squared: Proportion of variance explained

All functions are designed for MEG/EEG data with shape (n_features, n_trials, n_spatial).
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_cohens_d(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Compute Cohen's d effect size between two conditions.

    Cohen's d represents the standardized mean difference between two groups,
    defined as: d = (mean1 - mean2) / pooled_std

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0 and 1), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).

    Returns:
        Cohen's d values, shape (n_features, n_spatial).

    Examples:
        >>> d = compute_cohens_d(X, y, groups)
        >>> print(f"Mean Cohen's d: {np.nanmean(d):.3f}")
    """
    n_features = X.shape[0]
    n_spatial = X.shape[2]

    # Transpose for easier indexing: (n_trials, n_features, n_spatial)
    X_t = X.transpose(1, 0, 2)

    # Split by condition
    X_cond0 = X_t[y == 0]  # IN condition
    X_cond1 = X_t[y == 1]  # OUT condition

    # Compute means for each condition
    mean_0 = np.nanmean(X_cond0, axis=0)  # (n_features, n_spatial)
    mean_1 = np.nanmean(X_cond1, axis=0)  # (n_features, n_spatial)

    # Compute standard deviations
    std_0 = np.nanstd(X_cond0, axis=0, ddof=1)
    std_1 = np.nanstd(X_cond1, axis=0, ddof=1)

    # Compute sample sizes
    n_0 = len(X_cond0)
    n_1 = len(X_cond1)

    # Compute pooled standard deviation
    pooled_std = np.sqrt(
        ((n_0 - 1) * std_0**2 + (n_1 - 1) * std_1**2) / (n_0 + n_1 - 2)
    )

    # Avoid division by zero
    pooled_std[pooled_std == 0] = np.nan

    # Compute Cohen's d
    cohens_d = (mean_1 - mean_0) / pooled_std

    logger.debug(
        f"Cohen's d computed: mean={np.nanmean(cohens_d):.3f}, "
        f"median={np.nanmedian(cohens_d):.3f}, "
        f"range=[{np.nanmin(cohens_d):.3f}, {np.nanmax(cohens_d):.3f}]"
    )

    return cohens_d


def compute_hedges_g(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Compute Hedges' g effect size (bias-corrected Cohen's d).

    Hedges' g applies a correction factor for small sample sizes to Cohen's d.
    The correction factor is: J = 1 - 3/(4*df - 1), where df = n1 + n2 - 2

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0 and 1), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).

    Returns:
        Hedges' g values, shape (n_features, n_spatial).

    Examples:
        >>> g = compute_hedges_g(X, y, groups)
        >>> print(f"Mean Hedges' g: {np.nanmean(g):.3f}")
    """
    # First compute Cohen's d
    cohens_d = compute_cohens_d(X, y, groups)

    # Compute sample sizes
    n_0 = np.sum(y == 0)
    n_1 = np.sum(y == 1)

    # Degrees of freedom
    df = n_0 + n_1 - 2

    # Correction factor (J)
    J = 1 - 3 / (4 * df - 1)

    # Apply correction
    hedges_g = cohens_d * J

    logger.debug(
        f"Hedges' g computed (correction factor J={J:.4f}): "
        f"mean={np.nanmean(hedges_g):.3f}, "
        f"median={np.nanmedian(hedges_g):.3f}"
    )

    return hedges_g


def compute_eta_squared(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Compute eta-squared effect size (proportion of variance explained).

    Eta-squared represents the proportion of total variance that is attributable
    to the effect (condition difference). It ranges from 0 to 1.

    For between-subjects design:
    η² = SS_between / SS_total

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0 and 1), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).

    Returns:
        Eta-squared values, shape (n_features, n_spatial).

    Examples:
        >>> eta2 = compute_eta_squared(X, y, groups)
        >>> print(f"Mean eta-squared: {np.nanmean(eta2):.3f}")
    """
    n_features = X.shape[0]
    n_spatial = X.shape[2]

    # Transpose for easier indexing: (n_trials, n_features, n_spatial)
    X_t = X.transpose(1, 0, 2)

    # Split by condition
    X_cond0 = X_t[y == 0]  # IN condition
    X_cond1 = X_t[y == 1]  # OUT condition

    # Grand mean (across all trials)
    grand_mean = np.nanmean(X_t, axis=0)  # (n_features, n_spatial)

    # Condition means
    mean_0 = np.nanmean(X_cond0, axis=0)
    mean_1 = np.nanmean(X_cond1, axis=0)

    # Sample sizes
    n_0 = len(X_cond0)
    n_1 = len(X_cond1)

    # Sum of squares between groups (SS_between)
    SS_between = n_0 * (mean_0 - grand_mean) ** 2 + n_1 * (mean_1 - grand_mean) ** 2

    # Sum of squares total (SS_total)
    SS_total = np.nansum((X_t - grand_mean) ** 2, axis=0)

    # Avoid division by zero
    SS_total[SS_total == 0] = np.nan

    # Eta-squared
    eta_squared = SS_between / SS_total

    # Clip to valid range [0, 1]
    eta_squared = np.clip(eta_squared, 0, 1)

    logger.debug(
        f"Eta-squared computed: mean={np.nanmean(eta_squared):.3f}, "
        f"median={np.nanmedian(eta_squared):.3f}, "
        f"range=[{np.nanmin(eta_squared):.3f}, {np.nanmax(eta_squared):.3f}]"
    )

    return eta_squared


def compute_partial_eta_squared(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Compute partial eta-squared effect size.

    Partial eta-squared controls for other factors by removing their variance
    from the denominator:
    η²_p = SS_effect / (SS_effect + SS_error)

    For a simple two-group comparison, partial eta-squared equals regular
    eta-squared.

    Args:
        X: Feature data, shape (n_features, n_trials, n_spatial).
        y: Labels (0 and 1), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).

    Returns:
        Partial eta-squared values, shape (n_features, n_spatial).

    Examples:
        >>> partial_eta2 = compute_partial_eta_squared(X, y, groups)
        >>> print(f"Mean partial eta-squared: {np.nanmean(partial_eta2):.3f}")
    """
    # For two-group comparison, partial eta-squared = eta-squared
    # This is a placeholder for future extension to more complex designs
    return compute_eta_squared(X, y, groups)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size magnitude.

    Standard interpretation guidelines (Cohen, 1988):
    - Small: |d| = 0.2
    - Medium: |d| = 0.5
    - Large: |d| = 0.8

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.

    Examples:
        >>> interpretation = interpret_cohens_d(0.6)
        >>> print(interpretation)  # "medium"
    """
    abs_d = abs(d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_eta_squared(eta2: float) -> str:
    """Interpret eta-squared effect size magnitude.

    Standard interpretation guidelines (Cohen, 1988):
    - Small: η² = 0.01
    - Medium: η² = 0.06
    - Large: η² = 0.14

    Args:
        eta2: Eta-squared value.

    Returns:
        Interpretation string.

    Examples:
        >>> interpretation = interpret_eta_squared(0.08)
        >>> print(interpretation)  # "medium"
    """
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"
