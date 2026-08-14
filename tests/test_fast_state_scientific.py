"""Tests for fixed-ridge Schaefer-400 state decoding."""

import numpy as np

from code.classification.fast_state_scientific import (
    FixedRidgeConfig,
    fit_held_out_subject,
    pool_held_out_folds,
)


def test_fixed_ridge_generalizes_and_pools_disjoint_subjects():
    generator = np.random.default_rng(4)
    subjects = np.repeat(np.asarray(["01", "02", "03", "04"]), 30)
    labels = np.tile(np.repeat([0, 1], 15), 4)
    features = generator.normal(size=(len(labels), 12))
    features[:, :3] += labels[:, None] * 1.5
    folds = [
        fit_held_out_subject(
            features, labels, subjects, subject, FixedRidgeConfig(alpha=1.0)
        )
        for subject in np.unique(subjects)
    ]
    pooled = pool_held_out_folds(folds)
    assert pooled["roc_auc"] > 0.8
    assert pooled["labels"].shape == labels.shape
    assert pooled["subject_order"].tolist() == ["01", "02", "03", "04"]


def test_fixed_ridge_rejects_nonpositive_penalty():
    try:
        FixedRidgeConfig(alpha=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive ridge penalty was accepted")
