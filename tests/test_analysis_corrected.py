"""Deterministic contracts for the corrected panel-analysis workflow."""

from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from code.analysis.alignment import (
    build_alignment_keys,
    require_exact_alignment,
    validate_schaefer_400,
)
from code.analysis.contracts import (
    FEATURE_MODULATION_FEATURES,
    CORRECTED_FEATURES,
    PANEL_SPECS,
    CANONICAL_BAND_KEYS,
    canonical_band_key,
    schema_catalog,
)
from code.analysis.execution_plan import build_execution_plan
from code.analysis.inference import synchronized_cluster_mass_test
from code.analysis.labels import (
    LABEL_MID,
    OUTCOME_COMMISSION_ERROR,
    OUTCOME_CORRECT_OMISSION,
    build_corrected_window_labels,
    draw_circular_shifts,
    filter_vtc_reflect,
    label_matched_rare_outcomes,
    reconstruct_opposite_free_labels,
    reconstruct_strict_labels,
    summarize_label_overlap,
    valid_circular_offsets,
)
from code.analysis.real_inputs import _band_reduce
from code.analysis.preflight import inspect_inputs
from code.analysis.provenance import (
    active_analysis_id,
    create_analysis_id,
    initialize,
    resolve_analysis_directory,
)
from code.analysis.aggregate_runner import (
    _fit_periodic_spectra,
    _subject_selected_spectra,
)
from code.analysis.real_inputs import RunLabelContext
from code.analysis.observed_runner import _parcel_adjacency, _subject_state_means
from code.statistics.run_group_statistics import _resolve_fsaverage_subjects_dir
from code.visualization.panel1_bundle import (
    PANEL_NAMES,
    _select_weighting,
    _spectral_rows,
    _write_slides,
)
from code.visualization.stats_classif_panel import _plot_brain


def test_reflected_filter_is_boundary_safe():
    np.testing.assert_allclose(filter_vtc_reflect(np.ones(20), 9), 1.0)
    edge_impulse = filter_vtc_reflect(np.r_[1.0, np.zeros(19)], 9)
    assert edge_impulse[0] > edge_impulse[-1]
    assert edge_impulse.sum() == pytest.approx(1.0)


def test_strict_eight_trial_labels():
    values = np.arange(40.0)
    indices = np.asarray([np.arange(8), np.arange(16, 24), np.arange(32, 40)])
    np.testing.assert_array_equal(
        reconstruct_strict_labels(values, indices), [-1, 0, 1]
    )
    with pytest.raises(ValueError, match="shape"):
        reconstruct_strict_labels(values, indices[:, :7])


def test_opposite_free_labels_allow_mid_but_reject_opposites_and_bad_trials():
    values = np.arange(40.0)
    indices = np.asarray(
        [
            [0, 1, 10, 11, 12, 13, 14, 15],
            [24, 25, 26, 27, 28, 29, 30, 31],
            [0, 1, 20, 21, 22, 23, 38, 39],
            [0, 1, 10, 11, 12, 13, 14, 15],
        ]
    )
    bad = np.zeros((4, 8), dtype=bool)
    bad[3, 2] = True
    labels = reconstruct_opposite_free_labels(values, indices, bad)
    np.testing.assert_array_equal(labels, [-1, 1, 0, 0])


def test_matched_anchor_outcome_and_any_bad_rejection():
    outcomes = np.asarray(["correct_commission"] * 24, dtype=object)
    outcomes[7] = "correct_omission"
    outcomes[23] = "commission_error"
    indices = np.asarray([np.arange(8), np.arange(8, 16), np.arange(16, 24)])
    matched = label_matched_rare_outcomes(outcomes, indices)
    np.testing.assert_array_equal(
        matched, [OUTCOME_CORRECT_OMISSION, 0, OUTCOME_COMMISSION_ERROR]
    )
    bad = np.zeros((3, 8), dtype=bool)
    bad[2, 0] = True
    vtc = np.repeat([0.0, 1.0, 2.0], 8)
    labels = build_corrected_window_labels(vtc, indices, bad, outcomes)
    assert labels["state"][2] == LABEL_MID
    assert labels["cell"][0] == "IN_correct_omission"
    assert labels["cell"][2] == ""


def test_circular_offsets_and_seeds_are_deterministic():
    offsets = valid_circular_offsets(80, 24)
    assert np.all(np.minimum(offsets, 80 - offsets) > 24)
    first = draw_circular_shifts([80, 90], 24, np.random.default_rng(3))
    second = draw_circular_shifts([80, 90], 24, np.random.default_rng(3))
    np.testing.assert_array_equal(first, second)


def test_real_input_band_reduction_uses_only_canonical_bands():
    frequencies = np.arange(2.0, 121.0)
    psd = np.broadcast_to(frequencies, (2, 3, len(frequencies)))
    reduced = _band_reduce(psd, frequencies)
    assert reduced.shape == (2, 3, 7)
    assert reduced[0, 0, 0] == pytest.approx(np.mean(np.arange(4.0, 8.0)))
    assert np.all(reduced[..., 0] > 2.0)


def test_alignment_and_schaefer_guards():
    keys = build_alignment_keys(["01"], ["02"], [1.25], np.arange(8)[None])
    require_exact_alignment(keys, keys.copy())
    with pytest.raises(ValueError, match="mismatch"):
        require_exact_alignment(keys, np.asarray(["different"]))
    names = [f"7Networks_LH_Parcel_{index:03d}" for index in range(400)]
    validate_schaefer_400(names)
    names[-1] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        validate_schaefer_400(names)


def test_synchronized_cluster_family_is_deterministic():
    values = np.random.default_rng(4).normal(size=(10, 2, 4))
    values[:, 0, :2] += 2
    adjacency = [[1], [0, 2], [1, 3], [2]]
    first = synchronized_cluster_mass_test(
        values, adjacency, n_permutations=19, cluster_threshold=2, seed=8
    )
    second = synchronized_cluster_mass_test(
        values, adjacency, n_permutations=19, cluster_threshold=2, seed=8
    )
    np.testing.assert_array_equal(
        first["null_max_cluster_mass"], second["null_max_cluster_mass"]
    )


def test_label_qc_and_immutable_id(tmp_path: Path):
    summary = summarize_label_overlap(np.asarray([-1, 0, 1]), np.asarray([-1, 1, 0]))
    assert summary["retained"] == 1 and summary["gained"] == 1 and summary["lost"] == 1
    config = {"paths": {"data_root": str(tmp_path)}}
    analysis_id = create_analysis_id(
        config, Path.cwd(), datetime(2026, 1, 2, tzinfo=timezone.utc)
    )
    initialize(tmp_path, analysis_id, config, {}, Path.cwd())
    with pytest.raises(FileExistsError, match="immutable"):
        initialize(tmp_path, analysis_id, config, {}, Path.cwd())


def test_active_main_analysis_reuses_metadata_id_and_force_replaces(tmp_path: Path):
    config = {"paths": {"data_root": str(tmp_path)}}
    first_id = create_analysis_id(
        config, Path.cwd(), datetime(2026, 1, 2, tzinfo=timezone.utc)
    )
    active = initialize(
        tmp_path, first_id, config, {}, Path.cwd(), active=True
    )
    assert active == tmp_path / "main"
    assert resolve_analysis_directory(tmp_path) == active
    assert active_analysis_id(tmp_path) == first_id
    with pytest.raises(FileExistsError, match="already exists"):
        initialize(tmp_path, first_id, config, {}, Path.cwd(), active=True)
    second_id = create_analysis_id(
        config, Path.cwd(), datetime(2026, 1, 3, tzinfo=timezone.utc)
    )
    initialize(
        tmp_path, second_id, config, {}, Path.cwd(), active=True, force=True
    )
    assert active_analysis_id(tmp_path) == second_id


def test_selected_subject_spectra_average_runs_then_parcels():
    contexts = tuple(
        RunLabelContext(
            start=index, stop=index + 1, subject=subject, run=run,
            vtc=np.empty(0), contributing_indices=np.empty((0, 8), dtype=int),
            contributing_bad_flags=np.empty((0, 8), dtype=bool),
        )
        for index, (subject, run) in enumerate(
            (("04", "02"), ("04", "03"), ("05", "02"))
        )
    )
    inputs = type(
        "Inputs", (),
        {"run_label_contexts": contexts, "states": np.asarray(["IN"] * 3)},
    )()
    values = np.asarray([
        [[1.0, 3.0], [3.0, 5.0]],
        [[3.0, 5.0], [5.0, 7.0]],
        [[9.0, 11.0], [11.0, 13.0]],
    ])
    observed = _subject_selected_spectra(values, np.asarray([0, 1]), inputs)
    np.testing.assert_array_equal(observed, [[3.0, 5.0], [10.0, 12.0]])


def test_feature_modulation_retains_equal_window_and_equal_run_weighting():
    values = np.asarray([[0.0], [10.0], [20.0], [20.0], [20.0], [30.0]])
    states = np.asarray(["IN", "IN", "IN", "IN", "IN", "OUT"])
    subjects = np.asarray(["04"] * 6)
    runs = np.asarray(["02", "03", "03", "03", "03", "02"])
    # Add a second paired subject so the production guard is satisfied.
    values = np.concatenate([values, values + 1])
    states = np.tile(states, 2)
    subjects = np.concatenate([subjects, np.asarray(["05"] * 6)])
    runs = np.tile(runs, 2)
    window_in, _, _ = _subject_state_means(
        values, states, subjects, runs, weighting="equal_window"
    )
    run_in, _, _ = _subject_state_means(
        values, states, subjects, runs, weighting="equal_run"
    )
    assert window_in[0, 0] == pytest.approx(14.0)
    assert run_in[0, 0] == pytest.approx(8.75)


def test_panel1_spectral_pairs_share_main_and_difference_y_axes():
    frequency = np.linspace(2.0, 120.0, 20)
    arrays = {"frequency": frequency}
    for prefix in ("", "aperiodic_", "corrected_", "periodic_"):
        baseline = -np.log10(frequency)
        arrays[f"subject_{prefix}spectrum_in"] = np.stack(
            [baseline, baseline + 0.01]
        )
        arrays[f"subject_{prefix}spectrum_out"] = np.stack(
            [baseline - 0.1, baseline - 0.09]
        )
        arrays[f"{prefix}spectrum_in"] = baseline
        arrays[f"{prefix}spectrum_out"] = baseline - 0.1
    figure = plt.figure()
    groups = _spectral_rows(figure, figure.add_gridspec(6, 16), arrays)
    for left, right in ((PANEL_NAMES[2], PANEL_NAMES[3]),
                        (PANEL_NAMES[4], PANEL_NAMES[5])):
        assert groups[left][0].get_shared_y_axes().joined(
            groups[left][0], groups[right][0]
        )
        assert groups[left][1].get_shared_y_axes().joined(
            groups[left][1], groups[right][1]
        )
    for name in PANEL_NAMES[2:6]:
        condition_lines = groups[name][0].lines[:2]
        assert condition_lines[0].get_linestyle() == "--"
        assert condition_lines[1].get_linestyle() == "-"
    plt.close(figure)


def test_panel1_defaults_to_equal_window_and_can_select_equal_run():
    arrays = {
        "raw_psd_modulation": np.asarray([1.0]),
        "raw_psd_modulation_equal_run": np.asarray([2.0]),
    }
    metadata = {"summary": {"available_weightings": ["equal_window", "equal_run"]}}
    assert _select_weighting(arrays, metadata, "equal_window")[
        "raw_psd_modulation"
    ] == 1.0
    assert _select_weighting(arrays, metadata, "equal_run")[
        "raw_psd_modulation"
    ] == 2.0


def test_panel1_periodic_spectra_are_modeled_peak_fits(monkeypatch):
    class FakeModel:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, frequencies, powers, *, freq_range):
            self.frequencies = frequencies
            self.powers = powers
            self.frequency_range = freq_range

    monkeypatch.setattr(
        "code.analysis.aggregate_runner.load_spectral_model",
        lambda: FakeModel,
    )
    monkeypatch.setattr(
        "code.analysis.aggregate_runner.get_peak_fit",
        lambda model: np.asarray([0.1, 0.3, 0.1]),
    )
    modeled = _fit_periodic_spectra(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.log10(np.asarray([[2.0, 3.0, 4.0, 5.0, 6.0]])),
        {"features": {"fooof": [{"freq_range": [2.0, 4.0]}]}},
    )
    np.testing.assert_allclose(modeled[0, 1:4], [0.1, 0.3, 0.1])
    assert np.isnan(modeled[0, [0, 4]]).all()


def test_brain_raster_cache_reuses_identical_map(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "code.visualization.plot_surface.roi_to_surface",
        lambda values, names, atlas: (values, values),
    )

    def fake_render(*args, **kwargs):
        calls.append((args[2], args[3]))
        return np.full((12, 16, 3), 127, dtype=np.uint8)

    monkeypatch.setattr(
        "code.visualization.plot_surface.render_inflated_view", fake_render
    )
    figure, axes = plt.subplots(1, 2)
    arguments = (
        np.asarray([1.0, 2.0]), np.asarray([True, False]),
        ["parcel-lh", "parcel-rh"], "schaefer_400", {}, -3.0, 3.0, "RdBu_r",
    )
    _plot_brain(axes[0], *arguments, cache_directory=tmp_path)
    _plot_brain(axes[1], *arguments, cache_directory=tmp_path)
    assert len(calls) == 4
    assert len(list(tmp_path.glob("*.png"))) == 1
    plt.close(figure)


def test_panel1_slide_exports_are_native_and_combine_spectral_progression(tmp_path):
    frequency = np.linspace(2.0, 120.0, 20)
    arrays = {
        "frequency": frequency,
        "raw_psd_modulation": np.ones((7, 4)),
        "raw_psd_auc": np.full((7, 4), 0.52),
        "fooof_modulation": np.ones((3, 4)),
        "fooof_auc": np.full((3, 4), 0.52),
        "corrected_psd_modulation": np.ones((7, 4)),
        "corrected_psd_auc": np.full((7, 4), 0.52),
        "raw_psd_p_fdr": np.ones((7, 4)),
        "fooof_p_fdr": np.ones((3, 4)),
        "corrected_psd_p_fdr": np.ones((7, 4)),
        "decoding_p_tmax": np.ones((17, 4)),
    }
    baseline = -np.log10(frequency)
    for prefix in ("", "aperiodic_", "corrected_", "periodic_"):
        arrays[f"{prefix}spectrum_in"] = baseline
        arrays[f"{prefix}spectrum_out"] = baseline - 0.1
    groups = {}
    for name, count in (
        (PANEL_NAMES[0], 7), (PANEL_NAMES[1], 7),
        (PANEL_NAMES[6], 2), (PANEL_NAMES[7], 2),
        (PANEL_NAMES[8], 7), (PANEL_NAMES[9], 7),
    ):
        figure, axes = plt.subplots(1, count)
        for axis in np.atleast_1d(axes):
            axis.imshow(np.ones((20, 30, 3)))
        groups[name] = np.atleast_1d(axes).tolist()
        plt.close(figure)
    outputs = _write_slides(
        arrays, groups, tmp_path,
        {"weighting": "equal_window", "map_correction": "FDR"},
    )
    assert len(outputs) == 7
    assert outputs[2].name == "03_C-F_spectral_decomposition.png"
    assert Image.open(outputs[0]).size == (2560, 1440)


def test_band_and_feature_contract_excludes_delta():
    assert CANONICAL_BAND_KEYS == (
        "theta",
        "alpha",
        "lobeta",
        "hibeta",
        "gamma1",
        "gamma2",
        "gamma3",
    )
    assert canonical_band_key("low_beta") == "lobeta"
    assert not any(
        "delta" in feature
        for feature in (*FEATURE_MODULATION_FEATURES, *CORRECTED_FEATURES)
    )
    with pytest.raises(ValueError, match="not a canonical"):
        canonical_band_key("delta")


def test_fsaverage_resolver_prefers_configured_shared_copy(tmp_path: Path, monkeypatch):
    subjects_dir = tmp_path / "fs_subjects"
    (subjects_dir / "fsaverage" / "surf").mkdir(parents=True)
    (subjects_dir / "fsaverage" / "label").mkdir()

    monkeypatch.setitem(sys.modules, "mne", None)
    resolved = _resolve_fsaverage_subjects_dir(
        {"paths": {"freesurfer_subjects_dir": str(subjects_dir)}}
    )
    assert resolved == subjects_dir


def test_fsaverage_resolver_fails_before_network_for_invalid_config(tmp_path: Path):
    subjects_dir = tmp_path / "missing_fs_subjects"
    with pytest.raises(FileNotFoundError, match="must contain surf/ and label/"):
        _resolve_fsaverage_subjects_dir(
            {"paths": {"freesurfer_subjects_dir": str(subjects_dir)}}
        )


def test_panel1_adjacency_passes_project_config(monkeypatch):
    captured = {}

    class FakeRow:
        indices = np.asarray([0])

    class FakeAdjacency:
        shape = (1, 1)

        def getrow(self, index):
            return FakeRow()

    def fake_builder(space, parcel_names, config):
        captured.update(config)
        return FakeAdjacency(), parcel_names

    monkeypatch.setattr("code.analysis.observed_runner.build_atlas_adjacency", fake_builder)
    config = {"paths": {"freesurfer_subjects_dir": "/shared/fs_subjects"}}
    assert _parcel_adjacency(("parcel",), config) == [[0]]
    assert captured == config


def test_panel_and_schema_contracts_are_complete():
    assert set(PANEL_SPECS) == {"panel1", "panel2", "panel3"}
    assert (
        PANEL_SPECS["panel1"]["composite_filename"] == "panel1_feature_modulation.png"
    )
    assert (
        PANEL_SPECS["panel2"]["composite_filename"]
        == "panel2_multifeature_decoding.png"
    )
    assert PANEL_SPECS["panel3"]["composite_filename"] == "panel3_network_dynamics.png"
    assert set(schema_catalog()) == {
        "labels",
        "maps",
        "decoding",
        "factorial_networks",
        "coupling",
        "compact_export",
        "figure",
        "dag_manifest",
    }


def test_dry_run_dag_has_aligned_arrays_and_validator_barriers():
    manifest = build_execution_plan(
        "analysis-20260102T000000Z-gunknown-c123456789abc",
        ["04", "05"],
        ["02", "03"],
    )
    assert manifest["array_cells"] == [
        {"index": 0, "subject": "04", "run": "02"},
        {"index": 1, "subject": "04", "run": "03"},
        {"index": 2, "subject": "05", "run": "02"},
        {"index": 3, "subject": "05", "run": "03"},
    ]
    edges = {
        (edge["upstream"], edge["downstream"], edge["dependency"])
        for edge in manifest["edges"]
    }
    assert ("run_preprocessing", "run_source", "aftercorr") in edges
    assert ("run_source", "run_features", "aftercorr") in edges
    assert ("run_features", "schaefer_400_feature_validator", "afterany") in edges
    assert (
        "schaefer_400_feature_validator",
        "feature_modulation_statistics",
        "afterok",
    ) in edges
    assert ("feature_modulation_validator", "analysis_export", "afterok") in edges
    assert ("panel_generation", "analysis_audit", "afterok") in edges


def test_preflight_reads_corrected_events_and_exact_window_metadata(tmp_path: Path):
    config = {
        "paths": {
            "data_root": str(tmp_path),
            "bids": str(tmp_path / "bids"),
        },
        "bids": {"subjects": ["04"], "task_runs": ["02"]},
    }
    event_dir = tmp_path / "bids" / "sub-04" / "meg"
    event_dir.mkdir(parents=True)
    outcomes = np.asarray(["correct_commission"] * 24, dtype=object)
    outcomes[7] = "correct_omission"
    outcomes[23] = "commission_error"
    import pandas as pd

    pd.DataFrame(
        {
            "onset": np.arange(24.0),
            "trial_type": [
                "Rare" if value != "correct_commission" else "Freq"
                for value in outcomes
            ],
            "trial_idx": np.arange(24),
            "VTC_raw": np.repeat([0.0, 1.0, 2.0], 8),
            "VTC_filtered": np.repeat([0.0, 1.0, 2.0], 8),
            "task": outcomes,
            "VTC_filter_method": "gaussian_reflect",
            "VTC_filter_version": "1.0.0",
        }
    ).to_csv(event_dir / "sub-04_task-gradCPT_run-02_events.tsv", sep="\t", index=False)
    features_root = tmp_path / "features"
    config["paths"]["features"] = str(features_root)
    indices = np.asarray([np.arange(8), np.arange(8, 16), np.arange(16, 24)])
    metadata = {
        "onset": [7.0, 15.0, 23.0],
        "anchor_epoch_index": [7, 15, 23],
        "included_epoch_indices": list(indices),
        "included_bad_ar2": list(np.zeros((3, 8), dtype=bool)),
    }
    names = np.asarray([f"7Networks_LH_Parcel_{index:03d}" for index in range(400)])
    files = {
        "welch_psds_schaefer_400": "sub-04_ses-recording_task-gradCPT_run-02_space-schaefer_400_desc-welchw8_psds.npz",
        "fooof_schaefer_400": "sub-04_ses-recording_task-gradCPT_run-02_space-schaefer_400_desc-fooofw8.npz",
        "welch_psds_corrected_schaefer_400": "sub-04_ses-recording_task-gradCPT_run-02_space-schaefer_400_desc-welch-corrw8_psds.npz",
    }
    for directory, filename in files.items():
        feature_dir = features_root / directory / "sub-04"
        feature_dir.mkdir(parents=True)
        np.savez_compressed(
            feature_dir / filename, trial_metadata=metadata, ch_names=names
        )

    report = inspect_inputs(config, ["04"], ["02"])

    assert report["status"] == "passed"
    assert report["recordings"][0]["cell_counts"] == {
        "IN_correct_omission": 1,
        "IN_commission_error": 0,
        "OUT_correct_omission": 0,
        "OUT_commission_error": 1,
    }
    assert not report["recordings"][0]["modulation_eligible"]
    assert not report["recordings"][0]["coupling_eligible"]
