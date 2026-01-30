"""Statistical analysis utilities for saflow.

This module provides utilities for:
- Subject-level averaging and contrasts
- T-tests and permutation tests
- Classification with cross-validation
- P-value computation and correction

All functions use logging and are designed for MEG/EEG data analysis.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score

logger = logging.getLogger(__name__)


def subject_average(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute subject-level averages for each condition.

    Averages trials within each subject and condition to create subject-level
    data for group statistics.

    Args:
        X: Feature data, shape (n_features, n_trials, n_channels).
        y: Labels, shape (n_trials,).
        groups: Subject indices, shape (n_trials,).

    Returns:
        Tuple containing:
        - new_X: Subject-averaged data, shape (n_subjects*n_conditions, n_features, n_channels)
        - new_y: Condition labels for averaged data

    Examples:
        >>> X_avg, y_avg = subject_average(X, y, groups)
        >>> print(f"Averaged to {len(y_avg)} subject-condition pairs")
    """
    new_X = []
    new_y = []

    for subj in np.unique(groups):
        for cond in np.unique(y):
            # Select trials for this subject and condition
            mask = (groups == subj) & (y == cond)
            subj_cond_data = X.transpose(1, 0, 2)[mask]

            if len(subj_cond_data) > 0:
                # Use median to be robust to outliers
                new_X.append(np.nanmedian(subj_cond_data, axis=0))
                new_y.append(cond)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    logger.debug(
        f"Averaged {X.shape[1]} trials to {len(new_y)} subject-condition pairs "
        f"({len(np.unique(groups))} subjects, {len(np.unique(y))} conditions)"
    )

    return new_X, new_y


def simple_contrast(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute subject-averages and contrasts between conditions.

    Performs subject-level averaging, computes normalized contrast (A-B)/B,
    and runs independent samples t-tests.

    Args:
        X: Feature data, shape (n_features, n_trials, n_channels).
        y: Labels (0 and 1), shape (n_trials,).
        groups: Subject indices, shape (n_trials,).

    Returns:
        Tuple containing:
        - X_contrast: Normalized contrast (A-B)/B, shape (n_features, n_channels)
        - tvals: T-values from t-test, shape (n_features, n_channels)
        - pvals: P-values from t-test, shape (n_features, n_channels)

    Examples:
        >>> contrast, tvals, pvals = simple_contrast(X, y, groups)
        >>> sig_mask = pvals < 0.05
        >>> print(f"Significant features: {np.sum(sig_mask)}")
    """
    n_features = X.shape[0]

    # Subject-level averaging
    X_avg, y_avg = subject_average(X, y, groups)

    # Average each condition separately across subjects
    X_avg_by_cond = []
    for cond in np.unique(y_avg):
        X_avg_by_cond.append(np.nanmean(X_avg[y_avg == cond], axis=0))

    # Compute normalized contrast (A - B)/B
    X_contrast = (X_avg_by_cond[0] - X_avg_by_cond[1]) / X_avg_by_cond[1]

    # Split conditions for t-test
    X_condA = X_avg[y_avg == 0]
    X_condB = X_avg[y_avg == 1]

    # Compute t-test for each feature
    tvals = []
    pvals = []
    for feature_idx in range(n_features):
        t, p = stats.ttest_ind(
            X_condB[:, feature_idx, :], X_condA[:, feature_idx, :], axis=0
        )
        tvals.append(t)
        pvals.append(p)

    tvals = np.array(tvals)
    pvals = np.array(pvals)

    logger.info(
        f"Computed contrast and t-tests for {n_features} features, "
        f"{X_contrast.shape[1]} channels"
    )

    return X_contrast, tvals, pvals


def subject_contrast(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute within-subject contrasts between conditions.

    Computes normalized contrast (A-B)/B and runs paired t-tests.

    Args:
        X: Feature data, shape (n_features, n_trials, n_channels).
        y: Labels (0 and 1), shape (n_trials,).

    Returns:
        Tuple containing:
        - X_contrast: Normalized contrast (A-B)/B, shape (n_features, n_channels)
        - tvals: T-values from paired t-test, shape (n_features, n_channels)
        - pvals: P-values from paired t-test, shape (n_features, n_channels)

    Examples:
        >>> contrast, tvals, pvals = subject_contrast(X, y)
    """
    n_features = X.shape[0]
    X = X.transpose(1, 0, 2)

    # Average each condition separately
    X_avg_by_cond = []
    for cond in np.unique(y):
        X_avg_by_cond.append(np.nanmean(X[y == cond], axis=0))

    # Compute normalized contrast (A - B)/B
    X_contrast = (X_avg_by_cond[0] - X_avg_by_cond[1]) / X_avg_by_cond[1]

    # Split conditions for paired t-test
    X_condA = X[y == 0]
    X_condB = X[y == 1]

    # Compute paired t-test for each feature
    tvals = []
    pvals = []
    for feature_idx in range(n_features):
        t, p = stats.ttest_rel(
            X_condB[:, feature_idx, :], X_condA[:, feature_idx, :], axis=0
        )
        tvals.append(t)
        pvals.append(p)

    tvals = np.array(tvals)
    pvals = np.array(pvals)

    logger.info(
        f"Computed within-subject contrast and paired t-tests for {n_features} features"
    )

    return X_contrast, tvals, pvals


def mask_pvals(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Create significance mask from p-values.

    Args:
        pvals: Array of p-values.
        alpha: Significance threshold.

    Returns:
        Boolean mask with True for significant values (pval < alpha).

    Examples:
        >>> sig_mask = mask_pvals(pvals, alpha=0.05)
        >>> print(f"Significant: {np.sum(sig_mask)}")
    """
    return pvals < alpha


def compute_pval(score: float, perm_scores: np.ndarray) -> float:
    """Compute p-value from permutation distribution.

    Args:
        score: Observed score.
        perm_scores: Permutation distribution of scores.

    Returns:
        P-value (proportion of permutations >= observed score).

    Examples:
        >>> pval = compute_pval(0.75, perm_scores)
    """
    n_perm = len(perm_scores)
    pvalue = (np.sum(perm_scores >= score) + 1.0) / (n_perm + 1)
    return pvalue


def apply_tmax(all_results: dict) -> np.ndarray:
    """Apply maximum statistic correction to classification results.

    Computes corrected p-values using the maximum statistic from permutation
    distributions to control for multiple comparisons.

    Args:
        all_results: Dictionary with 'scores' and 'perm_scores' keys.
            'scores': shape (n_features, n_channels)
            'perm_scores': shape (n_features, n_perms, n_channels) or similar

    Returns:
        Corrected p-values, shape (n_features, n_channels).

    Examples:
        >>> tmax_pvals = apply_tmax(classification_results)
    """
    da = all_results["scores"]
    da_perms = all_results["perm_scores"]

    # Reshape permutations to (n_features, n_total_perms)
    if da_perms.ndim > 2:
        da_perms = da_perms.reshape(da_perms.shape[0], -1)

    tmax_pvals = np.empty_like(da)

    for x in range(da.shape[0]):
        for y in range(da.shape[1]):
            pval = compute_pval(da[x, y], da_perms[x, :])
            tmax_pvals[x, y] = pval

    logger.info(
        f"Applied tmax correction: {np.sum(tmax_pvals < 0.05)} significant tests "
        f"at alpha=0.05"
    )

    return tmax_pvals


def singlefeat_classif(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf: Optional[object] = None,
    cv: Optional[object] = None,
    n_perms: int = 1,
) -> dict:
    """Single-feature classification with permutation testing.

    Performs univariate classification for each feature and channel separately,
    with permutation testing to assess significance.

    Args:
        X: Feature data, shape (n_features, n_trials, n_channels).
        y: Labels, shape (n_trials,).
        groups: Subject indices for cross-validation, shape (n_trials,).
        clf: Scikit-learn classifier. Defaults to LinearDiscriminantAnalysis().
        cv: Cross-validation splitter. Defaults to LeaveOneGroupOut().
        n_perms: Number of permutations for testing. Defaults to 1.

    Returns:
        Dictionary containing:
        - 'scores': Classification scores, shape (n_features, n_channels)
        - 'perm_scores': Permutation scores, shape (n_features, n_perms, n_channels)
        - 'pvals': P-values, shape (n_features, n_channels)

    Examples:
        >>> results = singlefeat_classif(X, y, groups, n_perms=1000)
        >>> print(f"Best score: {np.max(results['scores']):.3f}")
    """
    if clf is None:
        clf = LinearDiscriminantAnalysis()
    if cv is None:
        cv = LeaveOneGroupOut()

    all_scores, all_perm_scores, all_pvals = [], [], []

    for freq_idx in range(X.shape[0]):
        scores, perm_scores, pvals = [], [], []

        for chan_idx in range(X.shape[-1]):
            X_sf = X[freq_idx, :, chan_idx]

            score, permutation_scores, pvalue = permutation_test_score(
                clf,
                X=X_sf.reshape(-1, 1),
                y=y,
                groups=groups,
                cv=cv,
                n_permutations=n_perms,
                scoring="roc_auc",
                n_jobs=-1,
            )

            scores.append(score)
            perm_scores.append(permutation_scores)
            pvals.append(pvalue)

            logger.debug(
                f"Feature {freq_idx}, chan {chan_idx}: score={score:.3f}, "
                f"pvalue={pvalue:.4f}"
            )

        all_scores.append(scores)
        all_perm_scores.append(perm_scores)
        all_pvals.append(pvals)

    all_scores = np.array(all_scores)
    all_perm_scores = np.array(all_perm_scores)
    all_pvals = np.array(all_pvals)

    all_results = {
        "scores": all_scores,
        "perm_scores": all_perm_scores,
        "pvals": all_pvals,
    }

    logger.info(
        f"Completed single-feature classification: {X.shape[0]} features, "
        f"{X.shape[2]} channels, {n_perms} permutations"
    )

    return all_results
