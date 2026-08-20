"""Tests for Correct-versus-Lapse participant-level modulation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from code.analysis.outcome_modulation import (
    WEIGHTING_ORDER,
    compute_balanced_contrasts,
    compute_outcome_modulation,
    compute_subject_contrasts,
    participant_weights,
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


def test_equal_run_pooling_averages_runs_rather_than_windows():
    """A run holding most windows must not dominate the participant mean."""
    records = []
    for run, correct_n, correct_value in (("01", 9, 1.0), ("02", 1, 9.0)):
        records.extend(("01", run, "IN", "correct_omission", correct_value) for _ in range(correct_n))
        records.extend(("01", run, "IN", "commission_error", 0.0) for _ in range(2))
    for run in ("01", "02"):
        records.extend(("02", run, "IN", "correct_omission", 2.0) for _ in range(3))
        records.extend(("02", run, "IN", "commission_error", 0.0) for _ in range(2))
    subjects, runs, states, outcomes, scalar = map(np.asarray, zip(*records))
    values = scalar.astype(float)[:, None, None]
    pooled = compute_subject_contrasts(
        values, states, outcomes, subjects, "IN", 2, runs=runs, pooling="window"
    )
    by_run = compute_subject_contrasts(
        values, states, outcomes, subjects, "IN", 2, runs=runs, pooling="run"
    )
    # subject 01: windows pool to (9*1 + 1*9)/10 = 1.8; run means average to (1 + 9)/2 = 5.0
    assert pooled.differences[0, 0, 0] == -1.8
    assert by_run.differences[0, 0, 0] == -5.0
    assert by_run.run_counts.tolist() == [[2, 2], [2, 2]]
    # eligibility is shared, so both poolings retain the same cohort
    assert pooled.subjects.tolist() == by_run.subjects.tolist()


def test_equal_window_weights_track_effective_window_count():
    values, states, outcomes, subjects, _ = _imbalanced_inputs()
    contrasts = compute_subject_contrasts(
        values, states, outcomes, subjects, "IN", minimum_windows=4
    )
    assert participant_weights(contrasts, "equal_subject") is None
    assert participant_weights(contrasts, "equal_run") is None
    weights = participant_weights(contrasts, "equal_window")
    # harmonic effective n: (20*4)/24 and (8*4)/12
    assert weights.tolist() == [80 / 24, 32 / 12]


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


def _variant(scale: float) -> dict:
    return {
        "subject_n": 3,
        "parcel_t_values": np.full((9, 400), scale),
        "parcel_p_cluster_fwer": np.ones((9, 400)),
        "network_t_values": np.full((7, 9), scale),
        "network_p_fwer": np.ones((7, 9)),
    }


def _bundle_result() -> dict:
    state = {
        "balanced": {"direction_stability": np.ones((9, 400))},
        **{name: _variant(index + 1.0) for index, name in enumerate(WEIGHTING_ORDER)},
    }
    return {
        "IN": state,
        "OUT": state,
        "parcel_order": np.asarray([f"p{i}" for i in range(400)]),
        "minimum_windows_per_cell": 2,
        "balanced_repetitions": 10,
    }


def test_renderer_selects_one_weighting_and_keeps_shared_sensitivity(tmp_path: Path):
    write_result_bundle(tmp_path, _bundle_result(), {"analysis_id": "test"})
    for index, weighting in enumerate(WEIGHTING_ORDER):
        loaded, provenance = load_correct_lapse_bundle(tmp_path, weighting)
        assert loaded["IN"]["parcel_t_values"].shape == (9, 400)
        assert np.all(loaded["IN"]["parcel_t_values"] == index + 1.0)
        assert loaded["IN"]["balanced"]["direction_stability"].shape == (9, 400)
        assert loaded["weighting"] == weighting
        assert provenance["analysis_id"] == "test"


def test_renderer_rejects_an_unknown_weighting(tmp_path: Path):
    write_result_bundle(tmp_path, _bundle_result(), {"analysis_id": "test"})
    with pytest.raises(ValueError, match="unknown participant weighting"):
        load_correct_lapse_bundle(tmp_path, "equal_epoch")


def test_renderer_reads_bundles_written_before_weighting_variants(tmp_path: Path):
    state = {**_variant(1.0), "balanced": {"direction_stability": np.ones((9, 400))}}
    legacy = {
        "IN": state,
        "OUT": state,
        "parcel_order": np.asarray([f"p{i}" for i in range(400)]),
        "minimum_windows_per_cell": 5,
        "balanced_repetitions": 10,
    }
    write_result_bundle(tmp_path, legacy, {"analysis_id": "legacy"})
    loaded, _ = load_correct_lapse_bundle(tmp_path)
    assert loaded["IN"]["parcel_t_values"].shape == (9, 400)
    with pytest.raises(ValueError, match="predates weighting variants"):
        load_correct_lapse_bundle(tmp_path, "equal_run")


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
    assert result["IN"]["balanced"]["direction_stability"].shape == (9, 400)
    assert result["weighting_order"] == WEIGHTING_ORDER
    for weighting in WEIGHTING_ORDER:
        for state in ("IN", "OUT"):
            variant = result[state][weighting]
            assert variant["parcel_t_values"].shape == (9, 400)
            assert variant["network_t_values"].shape == (7, 9)
            # every variant shares one eligible cohort
            assert variant["subject_order"].tolist() == ["01", "02", "03"]
    assert result["IN"]["equal_subject"]["effective_subject_n"] == 3.0
    assert result["IN"]["equal_window"]["effective_subject_n"] <= 3.0
