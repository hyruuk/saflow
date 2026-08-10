"""Tests for resumable panel-analysis inference infrastructure."""

from pathlib import Path

import numpy as np
import pytest

from code.analysis.chunks import (
    aggregate_chunks,
    build_chunk_specs,
    derive_chunk_seed,
    write_chunk,
)
from code.analysis.networks import (
    CELL_ORDER,
    combine_run_fisher_z,
    compute_factorial_contrasts,
    fisher_z_correlation,
    require_complete_cells,
    synchronized_sign_flip_test,
)
from code.analysis.labels import permute_outcomes_within_run_state
from code.analysis.decoding import DecodingConfig
from code.analysis.workers import (
    compute_feature_modulation_statistics,
    compute_multifeature_models,
    compute_network_modulation,
    compute_network_coupling,
    compute_mixed_effects_sensitivity,
)
from code.analysis.synthetic_phase_c import run_synthetic_phase_c
from code.analysis.permutations import (
    DecodingModelInput,
    correct_decoding_families,
    run_decoding_permutation_chunk,
)
from code.classification.multifeature_scientific import NestedRidgeConfig


def _specs():
    return build_chunk_specs(
        analysis_id="analysis-20260102T030405Z-gabc-c123456789abc",
        endpoint="multifeature_decoding",
        family="features",
        n_permutations=9,
        chunk_size=4,
        config_hash="123456789abc",
        git_commit="abc",
        feature_order=("exponent", "offset"),
    )


def test_chunk_seeds_and_intervals_are_deterministic():
    first = _specs()
    second = _specs()
    assert first == second
    assert [(item.start, item.stop) for item in first] == [(0, 4), (4, 8), (8, 9)]
    assert first[0].seed == derive_chunk_seed(
        first[0].analysis_id, "multifeature_decoding", "features", 0
    )


def test_chunk_aggregation_rejects_missing_overlap_and_incompatibility(tmp_path: Path):
    specs = _specs()
    paths = []
    for spec in specs:
        path = tmp_path / f"chunk-{spec.chunk_index}.npz"
        write_chunk(path, spec, np.full((spec.stop - spec.start, 2), spec.chunk_index))
        paths.append(path)
    values, manifest = aggregate_chunks(paths[::-1], specs)
    assert values.shape == (9, 2)
    assert manifest["permutation_interval"] == [0, 9]
    with pytest.raises(ValueError, match="missing or duplicate"):
        aggregate_chunks(paths[:-1], specs)
    incompatible = list(specs)
    incompatible[1] = incompatible[1].__class__(
        **{**incompatible[1].__dict__, "feature_order": ("offset", "exponent")}
    )
    with pytest.raises(ValueError, match="incompatible chunk"):
        aggregate_chunks(paths, incompatible)
    with pytest.raises(FileExistsError, match="immutable"):
        write_chunk(paths[0], specs[0], np.ones((4, 2)))


def test_factorial_contrast_signs_match_primary_definition():
    # IN correct, IN lapse, OUT correct, OUT lapse
    cells = np.asarray([[[1.0], [3.0], [2.0], [7.0]]])
    contrasts = compute_factorial_contrasts(cells)
    assert contrasts["interaction"].item() == 3.0  # (7-2) - (3-1)
    assert contrasts["lapse_vs_correct_within_IN"].item() == 2.0
    assert contrasts["lapse_vs_correct_within_OUT"].item() == 5.0
    assert contrasts["OUT_vs_IN_within_correct"].item() == 1.0
    assert contrasts["OUT_vs_IN_within_lapse"].item() == 4.0


def test_complete_case_report_is_deterministic():
    counts = np.asarray([[5, 5, 5, 5], [5, 4, 8, 7]])
    mask, report = require_complete_cells(counts, minimum=5)
    np.testing.assert_array_equal(mask, [True, False])
    assert list(report[0]["counts"]) == list(CELL_ORDER)
    assert report[1]["reason"] == "requires >= 5 in every cell"


def test_fisher_z_and_run_weighting():
    first = np.arange(12, dtype=float)
    second = first * 2
    assert fisher_z_correlation(first, second) > 7
    combined = combine_run_fisher_z([0.2, 0.8], [13, 23])
    assert combined == pytest.approx((0.2 * 10 + 0.8 * 20) / 30)


def test_synchronized_network_family_is_deterministic():
    generator = np.random.default_rng(9)
    contrasts = {
        "interaction": generator.normal(0.5, 0.2, size=(8, 3)),
        "simple": generator.normal(0.0, 0.2, size=(8, 3)),
    }
    first = synchronized_sign_flip_test(contrasts, n_permutations=19, seed=4)
    second = synchronized_sign_flip_test(contrasts, n_permutations=19, seed=4)
    np.testing.assert_array_equal(first["null_max_abs_t"], second["null_max_abs_t"])
    assert first["p_values_fwer"].shape == (2, 3)


def test_panel1_uses_subject_paired_out_minus_in_and_per_feature_fdr():
    inside = np.zeros((6, 2, 4))
    outside = inside.copy()
    outside[:, 0, 0] = np.arange(1, 7)
    result = compute_feature_modulation_statistics(
        inside, outside, feature_order=("raw_theta", "exponent")
    )
    assert result["contrast"] == "OUT_minus_IN"
    assert result["effect_size_dz"][0, 0] > 1
    assert result["significant_fdr"][0, 0]


def _decoding_fixture():
    generator = np.random.default_rng(11)
    subjects = np.repeat(np.arange(4), 24)
    states = np.tile(np.repeat(["IN", "OUT"], 12), 4)
    outcomes = np.tile(
        np.asarray(["correct_omission"] * 6 + ["commission_error"] * 6)[:12],
        8,
    )
    tensor = generator.normal(size=(len(subjects), 3, 2))
    tensor[:, 0, 0] += (states == "OUT") * 2
    tensor[:, 1, 1] += (outcomes == "commission_error") * 2
    return tensor, states, outcomes, subjects


def test_panel2_fits_three_prespecified_models_without_leakage():
    tensor, states, outcomes, subjects = _decoding_fixture()
    result = compute_multifeature_models(
        tensor,
        states,
        outcomes,
        subjects,
        feature_order=("exponent", "theta"),
        parcel_order=("p1", "p2", "p3"),
        config=DecodingConfig(c_grid=(0.1, 1.0), inner_splits=3, seed=3),
    )
    assert tuple(result["models"]) == (
        "state",
        "lapse_within_IN",
        "lapse_within_OUT",
    )
    assert result["models"]["state"]["joint"]["metrics"]["roc_auc"] > 0.8


def test_panel3_aggregates_complete_subjects_and_all_contrasts():
    generator = np.random.default_rng(5)
    subjects = np.repeat(np.arange(3), 4 * 5)
    cells = np.tile(np.repeat(CELL_ORDER, 5), 3)
    assignments = np.asarray(
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
    values = generator.normal(size=(len(subjects), 7, 2))
    values[cells == "OUT_commission_error"] += 1.0
    result = compute_network_modulation(
        values,
        cells,
        subjects,
        assignments,
        minimum_windows=5,
        n_permutations=19,
        seed=7,
    )
    assert result["complete_case"].all()
    assert tuple(result["contrasts"]) == (
        "interaction",
        "lapse_vs_correct_within_IN",
        "lapse_vs_correct_within_OUT",
        "OUT_vs_IN_within_correct",
        "OUT_vs_IN_within_lapse",
    )


def test_network_coupling_is_run_centered_pooled_and_complete_case():
    generator = np.random.default_rng(18)
    subjects = np.repeat(np.arange(3), 4 * 12)
    cells = np.tile(np.repeat(CELL_ORDER, 12), 3)
    runs = np.tile(np.repeat(["02", "03"], 24), 3)
    assignments = np.asarray(
        ["Default Mode", "Default Mode", "Dorsal Attention", "Dorsal Attention"]
    )
    values = generator.normal(size=(len(subjects), 4, 2))
    values[:, 2:] = values[:, :2] + generator.normal(0, 0.1, size=values[:, 2:].shape)
    result = compute_network_coupling(
        values,
        cells,
        subjects,
        runs,
        assignments,
        minimum_windows=10,
        n_permutations=19,
        seed=4,
    )
    assert result["complete_case"].all()
    assert result["estimator"] == "within-run-centered pooled Pearson correlation"
    assert np.nanmean(result["fisher_z"]) > 1
    assert result["inference"]["p_values_fwer"].shape == (5, 2)


def test_network_coupling_removes_run_offsets_before_pooling():
    within_run = np.asarray([-2.0, -1.0, 1.0, 2.0])
    default = np.tile(np.r_[within_run, within_run + 100.0], 4)
    attention = np.tile(np.r_[-within_run, -within_run + 100.0], 4)
    cells = np.repeat(CELL_ORDER, 8)
    runs = np.tile(np.repeat(["02", "03"], 4), 4)
    values = np.tile(np.stack([default, attention], axis=1)[:, :, None], (2, 1, 1))
    result = compute_network_coupling(
        values,
        np.tile(cells, 2),
        np.repeat(["04", "05"], len(cells)),
        np.tile(runs, 2),
        np.asarray(["Default Mode", "Dorsal Attention"]),
        minimum_windows=5,
        n_permutations=3,
        seed=4,
    )
    assert np.all(result["fisher_z"] < 0)


def test_all_available_mixed_effects_is_explicitly_secondary():
    generator = np.random.default_rng(23)
    subjects = np.repeat(np.arange(6), 40)
    cells = np.tile(np.repeat(CELL_ORDER, 10), 6)
    values = generator.normal(size=(len(subjects), 1))
    values[:, 0] += np.char.startswith(cells.astype(str), "OUT") * 0.4
    result = compute_mixed_effects_sensitivity(
        values, cells, subjects, feature_order=("exponent",)
    )
    assert result["analysis_role"] == "secondary_all_available_sensitivity"
    assert result["features"]["exponent"]["n_subjects"] == 6
    assert "state:lapse" in result["features"]["exponent"]["p_values"]


def test_lapse_null_preserves_counts_in_each_subject_run_state():
    outcomes = np.asarray([1, 1, 2, 2, 1, 2, 1, 2])
    subjects = np.asarray(["04"] * 4 + ["05"] * 4)
    runs = np.asarray(["02"] * 8)
    states = np.asarray([-1] * 4 + [1] * 4)
    permuted = permute_outcomes_within_run_state(
        outcomes, subjects, runs, states, np.random.default_rng(8)
    )
    for subject, state in (("04", -1), ("05", 1)):
        selector = (subjects == subject) & (states == state)
        np.testing.assert_array_equal(
            np.sort(outcomes[selector]), np.sort(permuted[selector])
        )


def test_synthetic_phase_c_runs_all_panels_and_writes_immutable_chunks(
    tmp_path: Path,
):
    manifest = run_synthetic_phase_c(tmp_path / "analysis", seed=13)
    assert '"status": "complete"' in manifest.read_text()
    for panel in ("feature_modulation", "multifeature_decoding", "network_dynamics"):
        assert (tmp_path / "analysis" / panel / "observed.npz").exists()
        assert (tmp_path / "analysis" / panel / "observed.json").exists()
    assert len(list((tmp_path / "analysis" / "multifeature_decoding" / "chunks").glob("*.npz"))) == 3


def test_synchronized_decoding_chunk_and_three_family_correction():
    generator = np.random.default_rng(31)
    groups = np.repeat(np.arange(4), 12)
    labels = np.tile(np.repeat([0, 1], 6), 4)
    tensor = generator.normal(size=(48, 2, 1))
    inputs = {
        name: DecodingModelInput(tensor, labels, groups)
        for name in ("state", "lapse_within_IN", "lapse_within_OUT")
    }

    def permute(rng, values):
        shuffled = values.copy()
        for subject in np.unique(groups):
            selector = groups == subject
            shuffled[selector] = rng.permutation(shuffled[selector])
        return shuffled

    generators = {name: permute for name in inputs}
    null = run_decoding_permutation_chunk(
        inputs,
        generators,
        model_order=tuple(inputs),
        config=NestedRidgeConfig(c_grid=(1.0,), inner_splits=3, seed=2),
        n_permutations=1,
        seed=9,
    )
    assert null["joint_auc"].shape == (1, 3)
    observed = {name: values[0] for name, values in null.items()}
    corrected = correct_decoding_families(observed, null)
    assert corrected["joint_auc_p_fwer"].shape == (3,)
    assert corrected["feature_family_p_fwer"].shape == (6,)
    assert corrected["parcel_family_p_fwer"].shape == (3, 2)
