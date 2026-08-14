"""Fast fixed-ridge Schaefer-400 state decoding primitives."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FixedRidgeConfig:
    """Configuration for prespecified ridge state decoding.

    Args:
        alpha: Positive L2 penalty fixed before observed-label analysis.
        tolerance: Solver convergence tolerance.
    """

    alpha: float = 1.0
    tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.tolerance <= 0:
            raise ValueError("alpha and tolerance must be positive")


def make_fixed_ridge(config: FixedRidgeConfig) -> Pipeline:
    """Build a training-fitted imputation, scaling, and ridge pipeline."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "classifier",
                RidgeClassifier(
                    alpha=config.alpha,
                    class_weight="balanced",
                    solver="lsqr",
                    tol=config.tolerance,
                ),
            ),
        ]
    )


def fit_held_out_subject(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    held_out_subject: str,
    config: FixedRidgeConfig,
) -> dict[str, object]:
    """Fit on every other subject and score one held-out subject."""
    feature_values = np.asarray(features, dtype=float)
    label_values = np.asarray(labels, dtype=int)
    subject_values = np.asarray(subjects).astype(str)
    test = subject_values == str(held_out_subject)
    train = ~test
    if not test.any() or np.unique(label_values[train]).size != 2:
        raise ValueError("held-out split lacks data or training classes")
    started = perf_counter()
    estimator = make_fixed_ridge(config)
    estimator.fit(feature_values[train], label_values[train])
    scores = estimator.decision_function(feature_values[test])
    elapsed = perf_counter() - started
    auc = (
        float(roc_auc_score(label_values[test], scores))
        if np.unique(label_values[test]).size == 2
        else float("nan")
    )
    return {
        "held_out_subject": str(held_out_subject),
        "labels": label_values[test],
        "scores": np.asarray(scores, dtype=float),
        "auc": auc,
        "elapsed_seconds": elapsed,
        "train_count": int(train.sum()),
        "test_count": int(test.sum()),
    }


def fit_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    config: FixedRidgeConfig,
) -> tuple[Pipeline, np.ndarray, float]:
    """Fit one prespecified split and return estimator, scores, and AUC."""
    feature_values = np.asarray(features, dtype=float)
    label_values = np.asarray(labels, dtype=int)
    if not train.any() or not test.any() or np.unique(label_values[train]).size != 2:
        raise ValueError("train/test split lacks observations or training classes")
    estimator = make_fixed_ridge(config)
    estimator.fit(feature_values[train], label_values[train])
    scores = np.asarray(estimator.decision_function(feature_values[test]), dtype=float)
    auc = (
        float(roc_auc_score(label_values[test], scores))
        if np.unique(label_values[test]).size == 2
        else float("nan")
    )
    return estimator, scores, auc


def compute_grouped_reliance(
    estimator: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    blocks: list[np.ndarray],
    exchangeability_groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Measure held-out AUC decrease after within-group block shuffling."""
    if repeats < 1:
        raise ValueError("reliance repeats must be positive")
    feature_values = np.asarray(features, dtype=float)
    label_values = np.asarray(labels, dtype=int)
    groups = np.asarray(exchangeability_groups)
    transformed = estimator[:-1].transform(feature_values)
    classifier = estimator[-1]
    weights = np.asarray(classifier.coef_, dtype=float).reshape(-1)
    baseline_scores = np.asarray(classifier.decision_function(transformed), dtype=float)
    baseline = float(roc_auc_score(label_values, baseline_scores))
    generator = np.random.default_rng(seed)
    reliance = np.full((len(blocks), repeats), np.nan)
    for block_index, columns in enumerate(blocks):
        contribution = transformed[:, columns] @ weights[columns]
        for repeat in range(repeats):
            shuffled_contribution = contribution.copy()
            for group in np.unique(groups):
                selected = np.flatnonzero(groups == group)
                shuffled_contribution[selected] = contribution[
                    generator.permutation(selected)
                ]
            shuffled_scores = baseline_scores - contribution + shuffled_contribution
            shuffled_auc = roc_auc_score(label_values, shuffled_scores)
            reliance[block_index, repeat] = baseline - shuffled_auc
    return reliance


def pool_held_out_folds(folds: list[dict[str, object]]) -> dict[str, object]:
    """Combine disjoint held-out predictions into the primary pooled AUC."""
    if not folds:
        raise ValueError("at least one held-out fold is required")
    labels = np.concatenate([np.asarray(fold["labels"]) for fold in folds])
    scores = np.concatenate([np.asarray(fold["scores"]) for fold in folds])
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "subject_auc": np.asarray([fold["auc"] for fold in folds], dtype=float),
        "subject_order": np.asarray(
            [str(fold["held_out_subject"]) for fold in folds]
        ),
        "labels": labels,
        "scores": scores,
        "elapsed_seconds": np.asarray(
            [fold["elapsed_seconds"] for fold in folds], dtype=float
        ),
    }
