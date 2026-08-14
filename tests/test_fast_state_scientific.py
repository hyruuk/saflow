"""Tests for fixed-ridge Schaefer-400 state decoding."""

import argparse
import json

import numpy as np

from code.analysis.contracts import MULTIFEATURE_FEATURES
from code.analysis.fast_state_workflow import aggregate, run_batch
from code.analysis.fast_state_submit import build_scripts
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


def test_multifeature_contract_excludes_fit_quality():
    assert "fooof_r_squared" not in MULTIFEATURE_FEATURES
    assert len(MULTIFEATURE_FEATURES) == 9


def test_fast_workflow_checkpoints_and_aggregates(tmp_path):
    generator = np.random.default_rng(8)
    subjects = np.repeat(np.asarray(["01", "02", "03", "04"]), 24)
    states = np.tile(np.repeat([-1, 1], 12), 4)
    tensor = generator.normal(size=(len(states), 5, 9))
    tensor[:, 0, 0] += (states == 1) * 1.5
    analysis_directory = tmp_path / "main"
    directory = analysis_directory / "fast_state"
    directory.mkdir(parents=True)
    (analysis_directory / "provenance.json").write_text(
        json.dumps({"analysis_id": "analysis-test"})
    )
    np.save(directory / "features.npy", tensor.astype(np.float32))
    np.save(directory / "subjects.npy", subjects)
    np.save(directory / "observed_states.npy", states)
    np.save(directory / "permuted_states.npy", np.stack([states, -states]))
    (directory / "metadata.json").write_text(json.dumps({
        "alpha": 1.0,
        "tolerance": 1e-4,
        "n_permutations": 2,
    }))
    common = {
        "analysis_root": str(tmp_path),
        "config": "config.yaml",
        "stage_local": False,
        "batch_index": 0,
        "permutations_per_job": 2,
        "skip_valid": True,
    }
    run_batch(argparse.Namespace(**common, observed=True))
    run_batch(argparse.Namespace(**common, observed=False))
    output = aggregate(argparse.Namespace(**common))
    with np.load(output) as inference:
        assert inference["null_auc"].shape == (2,)
        assert 0 < float(inference["p_value"]) <= 1


def test_fast_state_submission_builds_main_dependency_scripts(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    (main / "provenance.json").write_text(
        json.dumps({"analysis_id": "analysis-test"})
    )
    config = {
        "paths": {"venv": "env", "logs": str(tmp_path / "logs")},
        "computing": {"slurm": {"account": "def-pbellec"}},
        "analysis_workflow": {"node_resources": {}},
    }
    arguments = argparse.Namespace(
        analysis_root=str(tmp_path),
        config="config.yaml",
        n_permutations=1000,
        permutations_per_job=10,
        array_throttle=25,
        alpha=1.0,
        tolerance=1e-4,
        account="def-pbellec",
    )
    scripts = build_scripts(arguments, config)
    assert "#SBATCH --array=0-99%25" in scripts["permutations"]
    assert "#SBATCH --account=def-pbellec" in scripts["prepare"]
    assert "--analysis-id" not in "".join(scripts.values())
    assert "--stage-local" in scripts["permutations"]
