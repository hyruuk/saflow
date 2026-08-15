"""Tests for fixed-ridge Schaefer-400 state decoding."""

import argparse
import json

import numpy as np

from code.analysis.contracts import MULTIFEATURE_FEATURES
from code.analysis.fast_state_workflow import aggregate, run_batch
from code.analysis.fast_state_submit import build_scripts
from code.analysis.state_multifeature_submit import build_scripts as build_state_scripts
from code.analysis.state_multifeature_workflow import (
    aggregate as aggregate_state,
    run_population,
    run_reliance,
    run_within,
)
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
    analysis_root = tmp_path / "data" / "processed" / "panel_analysis"
    main = analysis_root / "main"
    main.mkdir(parents=True)
    (main / "provenance.json").write_text(
        json.dumps({"analysis_id": "analysis-test"})
    )
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "venv": "env",
            "logs": str(tmp_path / "logs"),
        },
        "computing": {"slurm": {"account": "def-pbellec"}},
        "analysis_workflow": {
            "processed_directory": "panel_analysis",
            "node_resources": {},
        },
    }
    arguments = argparse.Namespace(
        analysis_root=None,
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


def test_state_multifeature_stages_and_aggregation(tmp_path):
    analysis = tmp_path / "main"
    directory = analysis / "multifeature_state"
    prepared = directory / "prepared"
    prepared.mkdir(parents=True)
    (analysis / "provenance.json").write_text(
        json.dumps({"analysis_id": "analysis-test"})
    )
    generator = np.random.default_rng(21)
    subjects = np.repeat(np.asarray(["01", "02", "03", "04"]), 24)
    runs = np.tile(np.repeat(["02", "03"], 12), 4)
    states = np.tile(np.repeat([-1, 1], 6), 8)
    features = generator.normal(size=(len(states), 7, 3)).astype(np.float32)
    features[:, 0, 0] += (states == 1) * 1.5
    np.save(prepared / "features.npy", features)
    np.save(prepared / "subjects.npy", subjects)
    np.save(prepared / "runs.npy", runs)
    np.save(prepared / "observed_states.npy", states)
    np.save(prepared / "permuted_states.npy", np.stack([states, -states]))
    np.save(prepared / "network_assignments.npy", np.asarray([
        "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"
    ]))
    metadata = {
        "analysis_id": "analysis-test",
        "shape": list(features.shape),
        "feature_order": ["f1", "f2", "f3"],
        "network_order": ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"],
        "subject_order": ["01", "02", "03", "04"],
        "n_permutations": 2,
        "alpha": 1.0,
        "tolerance": 1e-4,
        "reliance_repeats": 1,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    common = {
        "analysis_root": str(tmp_path), "config": "config.yaml",
        "stage_local": False, "skip_valid": True,
    }
    run_population(argparse.Namespace(**common, observed=True, batch_index=0, permutations_per_job=2))
    run_population(argparse.Namespace(**common, observed=False, batch_index=0, permutations_per_job=2))
    for subject_index in range(4):
        run_within(argparse.Namespace(**common, observed=True, cell_index=subject_index, permutations_per_job=1))
        for cell_index in (subject_index * 2, subject_index * 2 + 1):
            run_within(argparse.Namespace(**common, observed=False, cell_index=cell_index, permutations_per_job=1))
        for regime in ("population", "within_subject"):
            run_reliance(argparse.Namespace(**common, regime=regime, cell_index=subject_index))
    output = aggregate_state(argparse.Namespace(**common, sign_flip_permutations=19))
    with np.load(output) as result:
        assert result["within_subject_auc"].shape == (4,)
        assert result["population_feature_reliance"].shape == (4, 3)
        assert result["within_subject_cell_reliance"].shape == (4, 21)
    panel_bundle = analysis / "multifeature_decoding" / "observed.npz"
    assert panel_bundle.exists()
    with np.load(panel_bundle) as result:
        assert result["population_auc"].shape == ()
        assert result["population_cell_reliance"].shape == (4, 21)
        assert result["feature_order"].tolist() == ["f1", "f2", "f3"]


def test_state_multifeature_submission_fans_out_independent_branches(tmp_path):
    main = tmp_path / "data" / "processed" / "panel_analysis" / "main"
    main.mkdir(parents=True)
    (main / "provenance.json").write_text(json.dumps({"analysis_id": "analysis-test"}))
    config = {
        "paths": {"data_root": str(tmp_path / "data"), "venv": "env", "logs": str(tmp_path / "logs")},
        "analysis_workflow": {"processed_directory": "panel_analysis", "node_resources": {}},
        "computing": {"slurm": {"account": "def-pbellec"}},
        "bids": {"subjects": ["01", "02", "03", "04"]},
    }
    arguments = argparse.Namespace(
        analysis_root=None, config="config.yaml", n_permutations=1000,
        permutations_per_job=10, within_permutations_per_job=100,
        reliance_repeats=20, sign_flip_permutations=10000,
        array_throttle=25, subject_throttle=4, alpha=1.0,
        tolerance=1e-4, account="def-pbellec",
    )
    scripts = build_state_scripts(arguments, config)
    assert set(scripts) == {
        "prepare", "population_observed", "population_permutations",
        "population_reliance", "within_observed", "within_permutations",
        "within_reliance", "aggregate",
    }
    assert "#SBATCH --array=0-99%25" in scripts["population_permutations"]
    assert "#SBATCH --array=0-39%25" in scripts["within_permutations"]
