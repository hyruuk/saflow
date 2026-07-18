"""Leakage-safe multifeature decoding and attribution.

This module contains the scientific core of the corrected multifeature
analysis.  It deliberately has no filesystem or SLURM dependencies so the
complete inferential procedure can be exercised with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRIMARY_OUTPUTS = (
    "joint",
    "standalone-feature",
    "feature-contribution",
    "region-contribution",
)


@dataclass(frozen=True)
class NestedRidgeConfig:
    """Configuration for nested ridge-logistic decoding.

    Args:
        c_grid: Positive inverse-regularization values considered in inner CV.
        seed: Seed used for estimators and synchronized permutations.
        inner_splits: Maximum number of subject-grouped inner folds.
    """

    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    seed: int = 42
    inner_splits: int = 5

    def __post_init__(self) -> None:
        if not self.c_grid or any(value <= 0 for value in self.c_grid):
            raise ValueError("c_grid must contain positive values")
        if self.inner_splits < 2:
            raise ValueError("inner_splits must be at least 2")


def permute_labels_within_subject(
    labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return one label permutation that preserves every subject's labels."""
    permuted = np.asarray(labels).copy()
    for subject in np.unique(groups):
        selected = groups == subject
        permuted[selected] = rng.permutation(permuted[selected])
    return permuted


def _make_estimator(c_value: float, seed: int) -> Pipeline:
    """Build a training-fitted imputation, scaling, and ridge-logistic pipeline."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5_000,
                    penalty="l2",
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _auc(estimator: Pipeline, features: np.ndarray, labels: np.ndarray) -> float:
    """Score an estimator, returning NaN when a fold has only one class."""
    if np.unique(labels).size != 2:
        return float("nan")
    probabilities = estimator.predict_proba(features)[:, 1]
    return float(roc_auc_score(labels, probabilities))


def _select_c(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: NestedRidgeConfig,
) -> float:
    """Select C using subject-grouped CV on an outer fold's training data."""
    subjects = np.unique(groups)
    n_splits = min(config.inner_splits, subjects.size)
    if n_splits < 2:
        raise ValueError("Nested CV requires at least two training subjects")
    splitter = GroupKFold(n_splits=n_splits)
    means: list[float] = []
    for c_value in config.c_grid:
        scores: list[float] = []
        for train, validation in splitter.split(features, labels, groups):
            estimator = _make_estimator(c_value, config.seed)
            estimator.fit(features[train], labels[train])
            scores.append(_auc(estimator, features[validation], labels[validation]))
        means.append(float(np.nanmean(scores)))
    if np.isnan(means).all():
        raise ValueError("No inner fold contained both classes")
    return config.c_grid[int(np.nanargmax(means))]


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Compute the prespecified primary and secondary classification metrics."""
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(true_positive / sensitivity_denominator),
        "specificity": float(true_negative / specificity_denominator),
        "confusion_matrix": matrix,
    }


def _shuffle_block(
    features: np.ndarray,
    columns: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Jointly permute selected columns across held-out trials."""
    shuffled = features.copy()
    shuffled[:, columns] = shuffled[rng.permutation(len(shuffled))][:, columns]
    return shuffled


def fit_nested_loso(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: NestedRidgeConfig,
    contribution_blocks: Sequence[np.ndarray] = (),
    contribution_seed: int | None = None,
) -> dict:
    """Run nested LOSO and optional held-out grouped permutation attribution.

    Preprocessing is fitted exclusively on each outer training fold.  Each
    contribution block is permuted jointly in the held-out subject, and its
    value is the decrease in that subject's ROC AUC.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(groups):
        raise ValueError("features must be 2D and align with labels and groups")
    if np.unique(groups).size < 4:
        raise ValueError("Nested LOSO requires at least four subjects")
    probabilities = np.full(len(labels), np.nan)
    selected_cs: list[float] = []
    subject_metrics: list[dict] = []
    contributions: list[np.ndarray] = []
    rng = np.random.default_rng(config.seed if contribution_seed is None else contribution_seed)
    for train, test in LeaveOneGroupOut().split(features, labels, groups):
        selected_c = _select_c(features[train], labels[train], groups[train], config)
        estimator = _make_estimator(selected_c, config.seed)
        estimator.fit(features[train], labels[train])
        fold_probabilities = estimator.predict_proba(features[test])[:, 1]
        probabilities[test] = fold_probabilities
        metrics = _classification_metrics(labels[test], fold_probabilities)
        metrics["subject"] = str(groups[test][0])
        metrics["selected_c"] = selected_c
        subject_metrics.append(metrics)
        selected_cs.append(selected_c)
        baseline = metrics["roc_auc"]
        fold_contributions = []
        for columns in contribution_blocks:
            shuffled = _shuffle_block(features[test], np.asarray(columns), rng)
            fold_contributions.append(baseline - _auc(estimator, shuffled, labels[test]))
        contributions.append(np.asarray(fold_contributions, dtype=float))
    result = {
        "metrics": _classification_metrics(labels, probabilities),
        "subject_metrics": subject_metrics,
        "probabilities": probabilities,
        "selected_c": np.asarray(selected_cs),
    }
    if contribution_blocks:
        result["subject_contributions"] = np.stack(contributions)
    return result


def _feature_columns(n_regions: int, n_features: int) -> list[np.ndarray]:
    """Return flattened column indices for each feature across all regions."""
    return [np.arange(feature, n_regions * n_features, n_features) for feature in range(n_features)]


def _region_columns(n_regions: int, n_features: int) -> list[np.ndarray]:
    """Return flattened column indices for each region across all features."""
    return [np.arange(region * n_features, (region + 1) * n_features) for region in range(n_regions)]


def _confidence_interval(values: np.ndarray) -> np.ndarray:
    """Return a normal-approximation 95% interval over held-out subjects."""
    mean = np.nanmean(values, axis=0)
    if values.shape[0] < 2:
        return np.stack([mean, mean], axis=-1)
    error = 1.96 * np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return np.stack([mean - error, mean + error], axis=-1)


def max_statistic_pvalues(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Compute synchronized one-sided family-wise max-statistic p-values."""
    observed = np.asarray(observed)
    null = np.asarray(null)
    if null.ndim != 2 or null.shape[1] != observed.size:
        raise ValueError("null must have shape (permutations, tests)")
    maxima = np.nanmax(null, axis=1)
    return (np.sum(maxima[:, None] >= observed[None, :], axis=0) + 1) / (len(null) + 1)


def run_primary_analysis(
    tensor: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: NestedRidgeConfig,
    n_permutations: int = 0,
) -> dict:
    """Compute joint, standalone, and grouped held-out attribution outputs."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError("tensor must have shape (trials, regions, features)")
    n_trials, n_regions, n_features = tensor.shape
    flattened = tensor.reshape(n_trials, -1)
    feature_blocks = _feature_columns(n_regions, n_features)
    region_blocks = _region_columns(n_regions, n_features)
    joint = fit_nested_loso(
        flattened,
        labels,
        groups,
        config,
        contribution_blocks=feature_blocks + region_blocks,
    )
    subject_contributions = joint.pop("subject_contributions")
    feature_values = subject_contributions[:, :n_features]
    region_values = subject_contributions[:, n_features:]
    standalone = [
        fit_nested_loso(flattened[:, columns], labels, groups, config)
        for columns in feature_blocks
    ]
    result = {
        "joint": joint,
        "standalone-feature": standalone,
        "feature-contribution": {
            "subject_values": feature_values,
            "mean_delta_auc": np.nanmean(feature_values, axis=0),
            "ci95": _confidence_interval(feature_values),
        },
        "region-contribution": {
            "subject_values": region_values,
            "mean_delta_auc": np.nanmean(region_values, axis=0),
            "ci95": _confidence_interval(region_values),
        },
    }
    if n_permutations:
        feature_null = np.empty((n_permutations, n_features))
        region_null = np.empty((n_permutations, n_regions))
        rng = np.random.default_rng(config.seed)
        for permutation in range(n_permutations):
            permuted = permute_labels_within_subject(labels, groups, rng)
            null_result = fit_nested_loso(
                flattened,
                permuted,
                groups,
                config,
                contribution_blocks=feature_blocks + region_blocks,
                contribution_seed=config.seed + permutation + 1,
            )
            values = null_result["subject_contributions"]
            feature_null[permutation] = np.nanmean(values[:, :n_features], axis=0)
            region_null[permutation] = np.nanmean(values[:, n_features:], axis=0)
        result["feature-contribution"]["null_mean_delta_auc"] = feature_null
        result["feature-contribution"]["pvalue_maxstat"] = max_statistic_pvalues(
            result["feature-contribution"]["mean_delta_auc"], feature_null
        )
        result["region-contribution"]["null_mean_delta_auc"] = region_null
        result["region-contribution"]["pvalue_maxstat"] = max_statistic_pvalues(
            result["region-contribution"]["mean_delta_auc"], region_null
        )
    return result


def validate_alignment(
    alignment_keys: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    groups: Sequence[np.ndarray],
    spatial_names: Sequence[Iterable[str]],
) -> None:
    """Fail when any combined feature has a different sample or spatial axis."""
    if not alignment_keys:
        raise ValueError("At least one feature is required")
    references = (alignment_keys[0], labels[0], groups[0], list(spatial_names[0]))
    names = ("alignment key", "labels", "groups", "spatial names")
    for feature_index in range(1, len(alignment_keys)):
        candidates = (
            alignment_keys[feature_index],
            labels[feature_index],
            groups[feature_index],
            list(spatial_names[feature_index]),
        )
        for name, reference, candidate in zip(names, references, candidates):
            if not np.array_equal(reference, candidate):
                raise ValueError(f"Feature {feature_index} has nonmatching {name}")
