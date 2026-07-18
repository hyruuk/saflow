"""Integration tests for the corrected multifeature analysis workflow."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from code.classification.multifeature_provenance import (
    canonical_config_hash,
    create_analysis_id,
    export_analysis,
    initialize_analysis_directory,
    inventory_legacy_results,
    validate_result,
)
from code.classification.multifeature_scientific import (
    NestedRidgeConfig,
    max_statistic_pvalues,
    permute_labels_within_subject,
    run_primary_analysis,
    validate_alignment,
)


@pytest.fixture
def synthetic_multifeature() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return four subjects, three features, five regions, and known signal."""
    generator = np.random.default_rng(19)
    groups = np.repeat(np.arange(4), 20)
    labels = np.tile(np.repeat([0, 1], 10), 4)
    tensor = generator.normal(size=(80, 5, 3))
    tensor[:, 0, 0] += 3.0 * labels
    tensor[3, 2, 1] = np.nan
    tensor[44, 4, 2] = np.nan
    return tensor, labels, groups


def test_primary_analysis_is_nested_deterministic_and_complete(synthetic_multifeature):
    tensor, labels, groups = synthetic_multifeature
    config = NestedRidgeConfig(c_grid=(0.1, 1.0), seed=7, inner_splits=3)
    first = run_primary_analysis(tensor, labels, groups, config, n_permutations=9)
    second = run_primary_analysis(tensor, labels, groups, config, n_permutations=9)
    assert first["joint"]["metrics"]["roc_auc"] > 0.7
    assert first["feature-contribution"]["subject_values"].shape == (4, 3)
    assert first["region-contribution"]["subject_values"].shape == (4, 5)
    assert len(first["standalone-feature"]) == 3
    np.testing.assert_array_equal(
        first["feature-contribution"]["null_mean_delta_auc"],
        second["feature-contribution"]["null_mean_delta_auc"],
    )
    assert first["joint"]["metrics"]["confusion_matrix"].shape == (2, 2)


def test_permutations_are_within_subject(synthetic_multifeature):
    _, labels, groups = synthetic_multifeature
    permuted = permute_labels_within_subject(labels, groups, np.random.default_rng(2))
    for subject in np.unique(groups):
        np.testing.assert_array_equal(
            np.sort(labels[groups == subject]), np.sort(permuted[groups == subject])
        )


def test_alignment_rejects_trial_and_spatial_order():
    key = np.array(["a", "b"])
    labels = np.array([0, 1])
    groups = np.array([0, 0])
    validate_alignment([key, key], [labels, labels], [groups, groups], [["x"], ["x"]])
    with pytest.raises(ValueError, match="alignment key"):
        validate_alignment([key, key[::-1]], [labels, labels], [groups, groups], [["x"], ["x"]])
    with pytest.raises(ValueError, match="spatial names"):
        validate_alignment([key, key], [labels, labels], [groups, groups], [["x"], ["y"]])


def test_max_statistic_uses_synchronized_family_maximum():
    observed = np.array([0.2, 0.4])
    null = np.array([[0.1, 0.3], [0.5, 0.0], [0.2, 0.1]])
    np.testing.assert_array_equal(max_statistic_pvalues(observed, null), [1.0, 0.5])


def test_analysis_directory_validation_export_and_legacy_inventory(tmp_path: Path):
    config = {"analysis": {"seed": 42}, "paths": {"logs": "logs"}}
    analysis_id = create_analysis_id(
        config, Path.cwd(), datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    )
    analysis_dir = initialize_analysis_directory(
        tmp_path / "processed", analysis_id, config, {"axis": "joint"}, Path.cwd()
    )
    np.savez_compressed(
        analysis_dir / "scores.npz",
        observed=np.array(0.8),
        perm_scores=np.arange(9),
        feature_names=np.array(["a", "b"]),
        spatial_names=np.array(["r1"]),
    )
    metadata = {
        "analysis_id": analysis_id,
        "config_hash": canonical_config_hash(config),
        "seed": 42,
        "n_permutations": 9,
    }
    (analysis_dir / "scores.json").write_text(__import__("json").dumps(metadata))
    valid, reason = validate_result(
        analysis_dir / "scores.npz", analysis_dir / "scores.json",
        analysis_id=analysis_id, config_hash=canonical_config_hash(config),
        seed=42, n_permutations=9,
        required_arrays={"observed": (), "perm_scores": (9,)},
    )
    assert valid, reason
    (analysis_dir / "subject_features").mkdir()
    np.save(analysis_dir / "subject_features" / "private.npy", np.ones(2))
    export_dir = export_analysis(analysis_dir, tmp_path / "export")
    assert (export_dir / "scores.npz").exists()
    assert not (export_dir / "subject_features").exists()
    legacy = tmp_path / "old_combined-24_scores.npz"
    legacy.write_bytes(b"legacy")
    manifest = inventory_legacy_results(tmp_path, tmp_path / "legacy.json")
    assert legacy.exists()
    assert '"moved": false' in manifest.read_text()
