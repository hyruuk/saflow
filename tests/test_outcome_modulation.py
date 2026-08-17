"""Tests for Correct-versus-Lapse participant-level modulation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from code.analysis.outcome_modulation import (
    compute_balanced_contrasts,
    compute_outcome_modulation,
    compute_subject_contrasts,
)
from code.analysis.result_io import write_result_bundle
from code.visualization.correct_lapse_panel import load_correct_lapse_bundle


def _imbalanced_inputs() -> tuple[np.ndarray, ...]:
    records = []
    for subject, correct_n, lapse_n, correct_value, lapse_value in (
        ("01", 20, 4, 1.0, 3.0),
        ("02", 8, 4, 2.0, 5.0),
    ):
        records.extend(
            (subject, "01", "IN", "correct_omission", correct_value) for _ in range(correct_n)
        )
        records.extend(
            (subject, "01", "IN", "commission_error", lapse_value) for _ in range(lapse_n)
        )
    subjects, runs, states, outcomes, scalar = map(np.asarray, zip(*records))
    values = scalar.astype(float)[:, None, None]
    return values, states, outcomes, subjects, runs


def test_primary_subject_contrasts_do_not_weight_the_majority_cell():
    values, states, outcomes, subjects, _ = _imbalanced_inputs()
    result = compute_subject_contrasts(values, states, outcomes, subjects, "IN", minimum_windows=4)
    assert result.subjects.tolist() == ["01", "02"]
    assert result.counts.tolist() == [[20, 4], [8, 4]]
    assert result.differences[:, 0, 0].tolist() == [2.0, 3.0]


def test_balanced_contrasts_match_counts_within_run():
    values, states, outcomes, subjects, runs = _imbalanced_inputs()
    result = compute_balanced_contrasts(
        values,
        states,
        outcomes,
        subjects,
        runs,
        "IN",
        4,
        np.random.default_rng(42),
    )
    assert result.counts.tolist() == [[4, 4], [4, 4]]
    assert result.differences[:, 0, 0].tolist() == [2.0, 3.0]


def test_renderer_requires_complete_network_and_parcel_results(tmp_path: Path):
    state = {
        "subject_n": 3,
        "parcel_t_values": np.zeros((9, 400)),
        "parcel_p_cluster_fwer": np.ones((9, 400)),
        "network_t_values": np.zeros((7, 9)),
        "network_p_fwer": np.ones((7, 9)),
        "balanced": {"direction_stability": np.ones((9, 400))},
    }
    result = {
        "IN": state,
        "OUT": state,
        "parcel_order": np.asarray([f"p{i}" for i in range(400)]),
        "minimum_windows_per_cell": 5,
        "balanced_repetitions": 10,
    }
    write_result_bundle(tmp_path, result, {"analysis_id": "test"})
    loaded, provenance = load_correct_lapse_bundle(tmp_path)
    assert loaded["IN"]["parcel_t_values"].shape == (9, 400)
    assert provenance["analysis_id"] == "test"


def test_complete_analysis_returns_network_parcel_and_balanced_maps(monkeypatch):
    rng = np.random.default_rng(3)
    cells = [
        (state, outcome)
        for state in ("IN", "OUT")
        for outcome in ("correct_omission", "commission_error")
    ]
    records = [
        (subject, run, state, outcome)
        for subject in ("01", "02", "03")
        for run in ("01", "02")
        for state, outcome in cells
        for _ in range(3)
    ]
    subjects, runs, states, outcomes = map(np.asarray, zip(*records))
    inputs = SimpleNamespace(
        feature_tensor=rng.normal(size=(len(records), 400, 9)),
        states=states,
        outcomes=outcomes,
        subjects=subjects,
        runs=runs,
        parcel_order=tuple(f"parcel-{index}" for index in range(400)),
    )
    monkeypatch.setattr(
        "code.analysis.outcome_modulation._parcel_adjacency",
        lambda parcel_order, config: [[] for _ in parcel_order],
    )
    network_order = np.asarray(
        [
            "Visual",
            "Somatomotor",
            "Dorsal Attention",
            "Ventral Attention",
            "Limbic",
            "Control",
            "Default Mode",
        ]
    )
    monkeypatch.setattr(
        "code.analysis.outcome_modulation._network_assignments",
        lambda parcel_order: np.resize(network_order, len(parcel_order)),
    )
    result = compute_outcome_modulation(
        inputs,
        {},
        minimum_windows=5,
        permutations=3,
        balanced_repetitions=2,
        seed=42,
    )
    assert result["IN"]["parcel_t_values"].shape == (9, 400)
    assert result["OUT"]["network_t_values"].shape == (7, 9)
    assert result["IN"]["balanced"]["direction_stability"].shape == (9, 400)
