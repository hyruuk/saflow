"""Leakage-safe nested LOSO decoding and circular-shift inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DecodingConfig:
    """Nested ridge-logistic settings."""

    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    inner_splits: int = 5
    seed: int = 42


def _estimator(c_value: float, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value, l1_ratio=0.0, solver="liblinear", class_weight="balanced",
            max_iter=5_000, random_state=seed,
        )),
    ])


def _score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float("nan") if np.unique(labels).size < 2 else float(roc_auc_score(labels, probabilities))


def _select_c(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
              config: DecodingConfig) -> float:
    n_splits = min(config.inner_splits, np.unique(groups).size)
    if n_splits < 2:
        raise ValueError("inner tuning requires at least two training subjects")
    splitter = GroupKFold(n_splits=n_splits)
    means = []
    for c_value in config.c_grid:
        fold_scores = []
        for train, validation in splitter.split(features, labels, groups):
            model = _estimator(c_value, config.seed).fit(features[train], labels[train])
            fold_scores.append(_score(labels[validation], model.predict_proba(features[validation])[:, 1]))
        means.append(np.nanmean(fold_scores))
    if np.isnan(means).all():
        raise ValueError("inner folds contain no evaluable validation labels")
    return config.c_grid[int(np.nanargmax(means))]


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = probabilities >= 0.5
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "roc_auc": _score(labels, probabilities),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "confusion_matrix": matrix,
    }


def _ci(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return np.asarray([np.nan, np.nan])
    error = 1.96 * np.std(finite, ddof=1) / np.sqrt(finite.size)
    return np.asarray([np.mean(finite) - error, np.mean(finite) + error])


def fit_nested_loso(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                    config: DecodingConfig) -> dict[str, object]:
    """Fit nested LOSO with training-only preprocessing and balanced weights."""
    feature_values = np.asarray(features, dtype=float)
    label_values = np.asarray(labels, dtype=int)
    subject_groups = np.asarray(groups)
    if feature_values.ndim != 2 or not (len(feature_values) == len(label_values) == len(subject_groups)):
        raise ValueError("features, labels, and groups must align")
    probabilities = np.full(len(label_values), np.nan)
    subject_metrics = []
    selected_c = []
    for train, test in LeaveOneGroupOut().split(feature_values, label_values, subject_groups):
        c_value = _select_c(feature_values[train], label_values[train], subject_groups[train], config)
        model = _estimator(c_value, config.seed).fit(feature_values[train], label_values[train])
        probabilities[test] = model.predict_proba(feature_values[test])[:, 1]
        metrics = _metrics(label_values[test], probabilities[test])
        metrics.update({"subject": str(subject_groups[test][0]), "selected_c": c_value})
        subject_metrics.append(metrics)
        selected_c.append(c_value)
    summary = _metrics(label_values, probabilities)
    for name in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity"):
        summary[f"{name}_subject_mean"] = float(np.nanmean([item[name] for item in subject_metrics]))
        summary[f"{name}_ci95"] = _ci(np.asarray([item[name] for item in subject_metrics]))
    return {"metrics": summary, "subject_metrics": subject_metrics,
            "probabilities": probabilities, "selected_c": np.asarray(selected_c)}


def synchronized_max_pvalues(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Apply a synchronized maximum statistic across a decoding family."""
    observed_values = np.asarray(observed, dtype=float)
    null_values = np.asarray(null, dtype=float)
    if null_values.ndim != 2 or null_values.shape[1:] != (observed_values.size,):
        raise ValueError("null must be permutations x family tests")
    maxima = np.nanmax(null_values, axis=1)
    return (1 + np.sum(maxima[:, None] >= observed_values.ravel()[None, :], axis=0)) / (len(maxima) + 1)


def run_circular_null(
    feature_sets: Sequence[np.ndarray], groups: np.ndarray, n_permutations: int,
    rebuild_labels: Callable[[np.random.Generator], np.ndarray], config: DecodingConfig,
) -> np.ndarray:
    """Repeat the entire nested procedure for synchronized rebuilt labels.

    ``rebuild_labels`` owns run-wise circular shifts and is called once per
    permutation, ensuring its resulting labels are shared by every parcel and
    feature in the family.
    """
    null = np.empty((n_permutations, len(feature_sets)))
    rng = np.random.default_rng(config.seed)
    for permutation in range(n_permutations):
        labels = rebuild_labels(rng)
        for test_index, features in enumerate(feature_sets):
            null[permutation, test_index] = fit_nested_loso(
                features, labels, groups, config
            )["metrics"]["roc_auc"]
    return null
