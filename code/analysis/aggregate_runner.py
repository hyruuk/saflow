"""Aggregate complete immutable partials into scientific result bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from code.analysis.chunks import derive_chunk_seed
from code.analysis.contracts import (
    CORRECTED_FEATURES,
    FEATURE_MODULATION_FEATURES,
    MULTIFEATURE_FEATURES,
)
from code.analysis.real_inputs import AnalysisInputs, load_real_inputs
from code.analysis.observed_runner import _network_assignments
from code.analysis.permutations import correct_decoding_families
from code.analysis.networks import (
    CONTRAST_WEIGHTS,
    compute_factorial_contrasts,
    synchronized_sign_flip_test,
)
from code.analysis.workers import (
    compute_all_network_pair_coupling,
    compute_mixed_effects_sensitivity,
)
from code.analysis.result_io import read_result_bundle
from code.analysis.provenance import resolve_analysis_directory
from code.utils.config import load_config
from code.utils.specparam_compat import get_peak_fit, load_spectral_model


def aggregate_analysis(args: argparse.Namespace) -> Path:
    """Aggregate one analysis only when every expected partial is compatible."""
    config = load_config(args.config)
    analysis_dir = _analysis_directory(config, args.analysis_root, args.analysis_id)
    if args.analysis == "feature_modulation":
        arrays, summary = _aggregate_feature_modulation(config, analysis_dir, args)
    elif args.analysis == "multifeature_decoding":
        arrays, summary = _aggregate_multifeature_decoding(config, analysis_dir, args.analysis_id)
    elif args.analysis == "network_dynamics":
        arrays, summary = _aggregate_network_dynamics(config, analysis_dir, args)
    else:
        raise ValueError(f"unknown analysis: {args.analysis}")
    return _write_observed(analysis_dir / args.analysis, arrays, summary, analysis_dir)


def _aggregate_feature_modulation(
    config: dict[str, Any], analysis_dir: Path, args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Combine paired maps, synchronized decoding chunks, and spectra."""
    statistics_bundles = [
        read_result_bundle(
            analysis_dir / "feature_modulation" / "partials" / "statistics" / feature
        )
        for feature in FEATURE_MODULATION_FEATURES
    ]
    _require_bundle_provenance(statistics_bundles, analysis_dir)
    statistics_by_weighting = {
        weighting: [bundle["result"][weighting] for bundle in statistics_bundles]
        for weighting in ("equal_window", "equal_run")
    }
    statistics = statistics_by_weighting["equal_window"]
    analysis_workflow = config.get("analysis_workflow", {})
    total = int(analysis_workflow.get("map_permutations", 10_000))
    size = int(analysis_workflow.get("map_chunk_size", 250))
    chunk_count = total // size
    observed = []
    corrected = []
    confusion = []
    for feature in FEATURE_MODULATION_FEATURES:
        chunks = [
            read_result_bundle(
                analysis_dir
                / "feature_modulation"
                / "partials"
                / "decoding"
                / feature
                / f"chunk-{index:04d}"
            )
            for index in range(chunk_count)
        ]
        _require_bundle_provenance(chunks, analysis_dir)
        _validate_feature_modulation_chunks(chunks, args.analysis_id, feature, total, size)
        results = [chunk["result"] for chunk in chunks]
        reference = np.asarray(results[0]["observed"])
        if any(not np.array_equal(reference, result["observed"], equal_nan=True)
               for result in results[1:]):
            raise ValueError(f"observed decoding differs across {feature} chunks")
        null = np.concatenate([np.asarray(result["perm_scores"]) for result in results])
        maximum = np.nanmax(null, axis=1)
        p_values = (
            1 + np.sum(maximum[:, None] >= reference[None, :], axis=0)
        ) / (len(null) + 1)
        observed.append(reference)
        corrected.append(p_values)
        confusion.append(np.asarray(results[0]["confusion_matrices"]))
    t_values = np.stack(
        [np.asarray(result["t_values"]).reshape(-1) for result in statistics]
    )
    statistics_p_cluster = np.stack(
        [np.asarray(result["p_values_cluster"]).reshape(-1) for result in statistics]
    )
    statistics_p_fdr = np.stack(
        [np.asarray(result["p_values_fdr"]).reshape(-1) for result in statistics]
    )
    mean_difference = np.stack(
        [np.asarray(result["mean_difference"]).reshape(-1) for result in statistics]
    )
    effect_size = np.stack(
        [np.asarray(result["effect_size_dz"]).reshape(-1) for result in statistics]
    )
    p_uncorrected = np.stack(
        [
            np.asarray(result["p_values_uncorrected"]).reshape(-1)
            for result in statistics
        ]
    )
    auc = np.stack(observed)
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    inputs = load_real_inputs(config, subjects, runs, include_spectra=True)
    exponent_p = np.asarray(statistics[7]["p_values_cluster"]).reshape(-1)
    exponent_t = np.asarray(statistics[7]["t_values"]).reshape(-1)
    selected_parcels = np.flatnonzero(exponent_p < 0.05)
    selection_rule = "cluster-FWER-significant FOOOF exponent parcels"
    if selected_parcels.size == 0:
        selected_parcels = np.asarray([int(np.nanargmax(np.abs(exponent_t)))])
        selection_rule = "fallback maximum-|t| FOOOF exponent parcel"
    equal_run_statistics = statistics_by_weighting["equal_run"]
    equal_run_exponent_p = np.asarray(
        equal_run_statistics[7]["p_values_cluster"]
    ).reshape(-1)
    equal_run_exponent_t = np.asarray(equal_run_statistics[7]["t_values"]).reshape(-1)
    equal_run_selected = np.flatnonzero(equal_run_exponent_p < 0.05)
    equal_run_selection_rule = "cluster-FWER-significant FOOOF exponent parcels"
    if equal_run_selected.size == 0:
        equal_run_selected = np.asarray([int(np.nanargmax(np.abs(equal_run_exponent_t)))])
        equal_run_selection_rule = "fallback maximum-|t| FOOOF exponent parcel"
    log_frequency = np.log10(inputs.frequencies)
    aperiodic_in = (
        inputs.fooof_state_in[:, selected_parcels, 1, None]
        - inputs.fooof_state_in[:, selected_parcels, 0, None] * log_frequency
    )
    aperiodic_out = (
        inputs.fooof_state_out[:, selected_parcels, 1, None]
        - inputs.fooof_state_out[:, selected_parcels, 0, None] * log_frequency
    )
    subject_raw_in = _subject_selected_spectra(
        inputs.raw_spectrum_in, selected_parcels, inputs, state="IN",
        weighting="equal_window",
    )
    subject_raw_out = _subject_selected_spectra(
        inputs.raw_spectrum_out, selected_parcels, inputs, state="OUT",
        weighting="equal_window",
    )
    subject_aperiodic_in = _subject_selected_spectra(
        aperiodic_in, np.arange(selected_parcels.size), inputs, state="IN",
        weighting="equal_window",
    )
    subject_aperiodic_out = _subject_selected_spectra(
        aperiodic_out, np.arange(selected_parcels.size), inputs, state="OUT",
        weighting="equal_window",
    )
    subject_corrected_in = _subject_selected_spectra(
        inputs.corrected_spectrum_in, selected_parcels, inputs, state="IN",
        weighting="equal_window",
    )
    subject_corrected_out = _subject_selected_spectra(
        inputs.corrected_spectrum_out, selected_parcels, inputs, state="OUT",
        weighting="equal_window",
    )
    equal_run_aperiodic_in = (
        inputs.fooof_state_in[:, equal_run_selected, 1, None]
        - inputs.fooof_state_in[:, equal_run_selected, 0, None] * log_frequency
    )
    equal_run_aperiodic_out = (
        inputs.fooof_state_out[:, equal_run_selected, 1, None]
        - inputs.fooof_state_out[:, equal_run_selected, 0, None] * log_frequency
    )
    equal_run_spectra = _selected_spectral_variants(
        inputs, equal_run_selected, equal_run_aperiodic_in,
        equal_run_aperiodic_out, "equal_run", config
    )
    subject_periodic_in = _fit_periodic_spectra(
        inputs.frequencies, subject_raw_in, config
    )
    subject_periodic_out = _fit_periodic_spectra(
        inputs.frequencies, subject_raw_out, config
    )
    arrays = {
        "raw_psd_modulation": t_values[:7],
        "raw_psd_auc": auc[:7],
        "raw_psd_p_cluster": statistics_p_cluster[:7],
        "raw_psd_p_fdr": statistics_p_fdr[:7],
        "frequency": inputs.frequencies,
        "spectrum_in": np.nanmean(subject_raw_in, axis=0),
        "spectrum_out": np.nanmean(subject_raw_out, axis=0),
        "aperiodic_spectrum_in": np.nanmean(subject_aperiodic_in, axis=0),
        "aperiodic_spectrum_out": np.nanmean(subject_aperiodic_out, axis=0),
        "corrected_spectrum_in": np.nanmean(subject_corrected_in, axis=0),
        "corrected_spectrum_out": np.nanmean(subject_corrected_out, axis=0),
        "periodic_spectrum_in": np.nanmean(subject_periodic_in, axis=0),
        "periodic_spectrum_out": np.nanmean(subject_periodic_out, axis=0),
        "subject_spectrum_in": subject_raw_in,
        "subject_spectrum_out": subject_raw_out,
        "subject_aperiodic_spectrum_in": subject_aperiodic_in,
        "subject_aperiodic_spectrum_out": subject_aperiodic_out,
        "subject_corrected_spectrum_in": subject_corrected_in,
        "subject_corrected_spectrum_out": subject_corrected_out,
        "subject_periodic_spectrum_in": subject_periodic_in,
        "subject_periodic_spectrum_out": subject_periodic_out,
        "spectral_subject_order": np.unique(inputs.subjects),
        "fooof_modulation": t_values[7:9],
        "fooof_auc": auc[7:9],
        "fooof_p_cluster": statistics_p_cluster[7:9],
        "fooof_p_fdr": statistics_p_fdr[7:9],
        "corrected_psd_modulation": t_values[9:],
        "corrected_psd_auc": auc[9:],
        "corrected_psd_p_cluster": statistics_p_cluster[9:],
        "corrected_psd_p_fdr": statistics_p_fdr[9:],
        "decoding_p_tmax": np.stack(corrected),
        "confusion_matrices": np.stack(confusion),
        "parcel_order": np.asarray(inputs.parcel_order),
        "mean_difference": mean_difference,
        "effect_size_dz": effect_size,
        "p_values_uncorrected": p_uncorrected,
        "p_values_cluster": statistics_p_cluster,
        "p_values_fdr": statistics_p_fdr,
        "subject_n": np.stack(
            [np.asarray(result["subject_n"]).reshape(-1) for result in statistics]
        ),
        "window_counts_by_feature": np.stack(
            [np.asarray(result["window_counts"]) for result in statistics]
        ),
    }
    arrays.update(_weighting_map_arrays(statistics_by_weighting["equal_run"], "equal_run"))
    arrays.update({f"{name}_equal_run": value for name, value in equal_run_spectra.items()})
    summary = {
        "feature_order": list(FEATURE_MODULATION_FEATURES),
        "subject_n_by_feature": [
            int(np.nanmax(result["subject_n"])) for result in statistics
        ],
        "primary_weighting": "equal_window",
        "available_weightings": ["equal_window", "equal_run"],
        "map_correction": "cluster-mass permutation FWER within each feature",
        "sensitivity_correction": "Benjamini-Hochberg FDR within each feature",
        "decoding_correction": "shared-permutation maximum across 400 parcels",
        "decoding_permutations": total,
        "spectral_selection_rule": selection_rule,
        "spectral_selection_parcel_indices": selected_parcels.tolist(),
        "spectral_selection_rule_equal_run": equal_run_selection_rule,
        "spectral_selection_parcel_indices_equal_run": equal_run_selected.tolist(),
        "periodic_spectrum_definition": (
            "participant-level summed Gaussian peak fit from specparam/FOOOF"
        ),
    }
    return arrays, summary


def _subject_average_recordings(
    values: np.ndarray, inputs: AnalysisInputs, *, state: str,
    weighting: str,
) -> np.ndarray:
    """Average recordings within subject using the requested state weighting."""
    if weighting not in {"equal_run", "equal_window"}:
        raise ValueError(f"unknown spectral weighting: {weighting}")
    recording_subjects = np.asarray(
        [context.subject for context in inputs.run_label_contexts]
    )
    target = state.upper()
    counts = np.asarray([
        np.sum(inputs.states[context.start:context.stop] == target)
        for context in inputs.run_label_contexts
    ])
    return np.stack(
        [
            np.average(
                values[(recording_subjects == subject) & (counts > 0)], axis=0,
                weights=(None if weighting == "equal_run" else
                         counts[(recording_subjects == subject) & (counts > 0)]),
            )
            for subject in np.unique(recording_subjects)
        ]
    )


def _subject_selected_spectra(
    values: np.ndarray,
    selected_parcels: np.ndarray,
    inputs: AnalysisInputs,
    *,
    state: str = "IN",
    weighting: str = "equal_run",
) -> np.ndarray:
    """Return one selected-parcel mean spectrum per subject."""
    subject_recordings = _subject_average_recordings(
        values[:, selected_parcels], inputs, state=state, weighting=weighting
    )
    return np.nanmean(subject_recordings, axis=1)


def _selected_spectral_variants(
    inputs: AnalysisInputs, selected: np.ndarray, aperiodic_in: np.ndarray,
    aperiodic_out: np.ndarray, weighting: str, config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build one complete selected-spectrum payload for a weighting rule."""
    raw_in = _subject_selected_spectra(
        inputs.raw_spectrum_in, selected, inputs, state="IN", weighting=weighting
    )
    raw_out = _subject_selected_spectra(
        inputs.raw_spectrum_out, selected, inputs, state="OUT", weighting=weighting
    )
    ap_in = _subject_selected_spectra(
        aperiodic_in, np.arange(selected.size), inputs,
        state="IN", weighting=weighting,
    )
    ap_out = _subject_selected_spectra(
        aperiodic_out, np.arange(selected.size), inputs,
        state="OUT", weighting=weighting,
    )
    corrected_in = _subject_selected_spectra(
        inputs.corrected_spectrum_in, selected, inputs,
        state="IN", weighting=weighting,
    )
    corrected_out = _subject_selected_spectra(
        inputs.corrected_spectrum_out, selected, inputs,
        state="OUT", weighting=weighting,
    )
    return _spectral_payload(
        raw_in, raw_out, ap_in, ap_out, corrected_in, corrected_out,
        inputs.frequencies, config,
    )


def _spectral_payload(
    raw_in: np.ndarray, raw_out: np.ndarray, ap_in: np.ndarray,
    ap_out: np.ndarray, corrected_in: np.ndarray, corrected_out: np.ndarray,
    frequencies: np.ndarray, config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return empirical and modeled spectra for one weighting rule."""
    periodic_in = _fit_periodic_spectra(frequencies, raw_in, config)
    periodic_out = _fit_periodic_spectra(frequencies, raw_out, config)
    return {
        "spectrum_in": np.nanmean(raw_in, axis=0),
        "spectrum_out": np.nanmean(raw_out, axis=0),
        "aperiodic_spectrum_in": np.nanmean(ap_in, axis=0),
        "aperiodic_spectrum_out": np.nanmean(ap_out, axis=0),
        "corrected_spectrum_in": np.nanmean(corrected_in, axis=0),
        "corrected_spectrum_out": np.nanmean(corrected_out, axis=0),
        "periodic_spectrum_in": np.nanmean(periodic_in, axis=0),
        "periodic_spectrum_out": np.nanmean(periodic_out, axis=0),
        "subject_spectrum_in": raw_in,
        "subject_spectrum_out": raw_out,
        "subject_aperiodic_spectrum_in": ap_in,
        "subject_aperiodic_spectrum_out": ap_out,
        "subject_corrected_spectrum_in": corrected_in,
        "subject_corrected_spectrum_out": corrected_out,
        "subject_periodic_spectrum_in": periodic_in,
        "subject_periodic_spectrum_out": periodic_out,
    }


def _fit_periodic_spectra(
    frequencies: np.ndarray,
    log_spectra: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Fit and align summed Gaussian peak models for participant spectra."""
    fooof_config = config.get("features", {}).get("fooof", [{}])
    parameters = fooof_config[0] if isinstance(fooof_config, list) else fooof_config
    frequency_range = parameters.get("freq_range", [2.0, 40.0])
    model_class = load_spectral_model()
    modeled = np.full(np.asarray(log_spectra).shape, np.nan, dtype=float)
    fit_mask = (
        (frequencies >= float(frequency_range[0]))
        & (frequencies <= float(frequency_range[1]))
    )
    for subject_index, log_spectrum in enumerate(log_spectra):
        model = model_class(
            peak_width_limits=parameters.get("peak_width_limits", [1, 8]),
            max_n_peaks=parameters.get("max_n_peaks", 4),
            min_peak_height=parameters.get("min_peak_height", 0.10),
            peak_threshold=parameters.get("peak_threshold", 2.0),
            aperiodic_mode=parameters.get("aperiodic_mode", "fixed"),
            verbose=False,
        )
        model.fit(
            frequencies, np.power(10.0, log_spectrum),
            freq_range=frequency_range,
        )
        peak_fit = get_peak_fit(model)
        if peak_fit.shape != (int(fit_mask.sum()),):
            raise ValueError("FOOOF peak model does not align with its fit range")
        modeled[subject_index, fit_mask] = peak_fit
    return modeled


def _weighting_map_arrays(
    statistics: list[dict[str, Any]], suffix: str
) -> dict[str, np.ndarray]:
    """Flatten one weighting variant into explicitly suffixed map arrays."""
    t_values = np.stack([np.asarray(item["t_values"]).reshape(-1) for item in statistics])
    cluster = np.stack([np.asarray(item["p_values_cluster"]).reshape(-1) for item in statistics])
    fdr = np.stack([np.asarray(item["p_values_fdr"]).reshape(-1) for item in statistics])
    effect = np.stack([np.asarray(item["effect_size_dz"]).reshape(-1) for item in statistics])
    return {
        f"raw_psd_modulation_{suffix}": t_values[:7],
        f"fooof_modulation_{suffix}": t_values[7:9],
        f"corrected_psd_modulation_{suffix}": t_values[9:],
        f"raw_psd_p_cluster_{suffix}": cluster[:7],
        f"fooof_p_cluster_{suffix}": cluster[7:9],
        f"corrected_psd_p_cluster_{suffix}": cluster[9:],
        f"p_values_cluster_{suffix}": cluster,
        f"p_values_fdr_{suffix}": fdr,
        f"raw_psd_p_fdr_{suffix}": fdr[:7],
        f"fooof_p_fdr_{suffix}": fdr[7:9],
        f"corrected_psd_p_fdr_{suffix}": fdr[9:],
        f"effect_size_dz_{suffix}": effect,
    }


def _validate_feature_modulation_chunks(
    chunks: list[dict[str, Any]],
    analysis_id: str,
    feature: str,
    total: int,
    size: int,
) -> None:
    """Reject gaps, wrong seeds, and incompatible feature-modulation analysis chunks."""
    intervals = []
    for index, chunk in enumerate(chunks):
        provenance = chunk["provenance"]
        result = chunk["result"]
        interval = np.asarray(result["permutation_interval"]).tolist()
        expected = [index * size, min((index + 1) * size, total)]
        if interval != expected:
            raise ValueError(f"wrong interval for {feature} chunk {index}")
        if provenance["analysis_id"] != analysis_id or provenance["feature"] != feature:
            raise ValueError(f"incompatible provenance for {feature} chunk {index}")
        seed = derive_chunk_seed(analysis_id, "feature_modulation", "decoding", index)
        if int(result["seed"]) != seed or provenance["seed"] != seed:
            raise ValueError(f"wrong seed for {feature} chunk {index}")
        intervals.append(interval)
    if intervals[0][0] != 0 or intervals[-1][1] != total:
        raise ValueError(f"incomplete permutation coverage for {feature}")
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError(f"gap or overlap in {feature} chunks")


def _aggregate_multifeature_decoding(
    config: dict[str, Any],
    analysis_dir: Path,
    analysis_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Combine the three independently fitted observed decoding models."""
    model_order = ("state", "lapse_within_IN", "lapse_within_OUT")
    model_bundles = [
        read_result_bundle(
            analysis_dir / "multifeature_decoding" / "partials" / "observed" / model
        )
        for model in model_order
    ]
    _require_bundle_provenance(model_bundles, analysis_dir)
    models = [bundle["result"] for bundle in model_bundles]
    arrays = {
        "joint_auc": np.asarray(
            [model["joint"]["metrics"]["roc_auc"] for model in models]
        ),
        "feature_contribution": np.stack(
            [model["feature-contribution"]["mean_delta_auc"] for model in models]
        ),
        "parcel_contribution": np.stack(
            [model["region-contribution"]["mean_delta_auc"] for model in models]
        ),
        "standalone_feature_auc": np.stack(
            [
                [
                    feature["metrics"]["roc_auc"]
                    for feature in model["standalone-feature"]
                ]
                for model in models
            ]
        ),
        "parcel_order": np.asarray(models[0]["parcel_order"]),
    }
    for model_name, model in zip(model_order, models):
        arrays[f"held_out_probabilities_{model_name}"] = np.asarray(
            model["joint"]["probabilities"]
        )
        arrays[f"selected_c_{model_name}"] = np.asarray(
            model["joint"]["selected_c"]
        )
        arrays[f"feature_subject_reliance_{model_name}"] = np.asarray(
            model["feature-contribution"]["subject_values"]
        )
        arrays[f"feature_reliance_ci95_{model_name}"] = np.asarray(
            model["feature-contribution"]["ci95"]
        )
        arrays[f"parcel_subject_reliance_{model_name}"] = np.asarray(
            model["region-contribution"]["subject_values"]
        )
        arrays[f"parcel_reliance_ci95_{model_name}"] = np.asarray(
            model["region-contribution"]["ci95"]
        )
        arrays[f"subject_auc_{model_name}"] = np.asarray(
            [
                metrics["roc_auc"]
                for metrics in model["joint"]["subject_metrics"]
            ]
        )
        arrays[f"subject_balanced_accuracy_{model_name}"] = np.asarray(
            [
                metrics["balanced_accuracy"]
                for metrics in model["joint"]["subject_metrics"]
            ]
        )
    analysis_workflow = config.get("analysis_workflow", {})
    total = int(analysis_workflow.get("decoding_permutations", 1_000))
    size = int(analysis_workflow.get("decoding_chunk_size", 25))
    chunks = [
        read_result_bundle(
            analysis_dir
            / "multifeature_decoding"
            / "partials"
            / "permutations"
            / f"chunk-{index:04d}"
        )
        for index in range(total // size)
    ]
    _require_bundle_provenance(chunks, analysis_dir)
    failures = np.concatenate(
        [np.asarray(chunk["result"]["class_failure"]) for chunk in chunks]
    )
    null = _aggregate_multifeature_null(chunks, analysis_id, total, size)
    synchronized_valid = ~np.any(failures, axis=1)
    if synchronized_valid.sum() < max(19, int(0.9 * total)):
        raise RuntimeError("too many class-failed synchronized multifeature-decoding analysis permutations")
    valid_null = {key: value[synchronized_valid] for key, value in null.items()}
    arrays.update(correct_decoding_families(arrays, valid_null))
    return arrays, {
        "model_order": list(model_order),
        "feature_order": list(MULTIFEATURE_FEATURES),
        "permutation_inference": "synchronized max-statistic families",
        "decoding_permutations": total,
        "effective_synchronized_permutations": int(synchronized_valid.sum()),
        "class_failure_count_by_model": failures.sum(axis=0).astype(int).tolist(),
    }


def _aggregate_multifeature_null(
    chunks: list[dict[str, Any]],
    analysis_id: str,
    total: int,
    size: int,
) -> dict[str, np.ndarray]:
    """Validate and concatenate synchronized multifeature-decoding analysis null intervals."""
    keys = (
        "joint_auc",
        "standalone_feature_auc",
        "feature_contribution",
        "parcel_contribution",
    )
    for index, chunk in enumerate(chunks):
        result = chunk["result"]
        provenance = chunk["provenance"]
        expected = [index * size, min((index + 1) * size, total)]
        if np.asarray(result["permutation_interval"]).tolist() != expected:
            raise ValueError(f"wrong multifeature-decoding analysis interval for chunk {index}")
        seed = derive_chunk_seed(
            analysis_id, "multifeature_decoding", "decoding_families", index
        )
        if int(result["seed"]) != seed or provenance["seed"] != seed:
            raise ValueError(f"wrong multifeature-decoding analysis seed for chunk {index}")
        if provenance["analysis_id"] != analysis_id:
            raise ValueError(f"wrong multifeature-decoding analysis analysis ID for chunk {index}")
    return {
        key: np.concatenate([np.asarray(chunk["result"][key]) for chunk in chunks])
        for key in keys
    }


def _aggregate_network_dynamics(
    config: dict[str, Any],
    analysis_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Combine nine feature-wise modulation and coupling results."""
    analysis_workflow = config.get("analysis_workflow", {})
    permutations = int(analysis_workflow.get("map_permutations", 10_000))
    base_seed = int(analysis_workflow.get("random_seed", 42))
    modulation_bundles = [
        read_result_bundle(
            analysis_dir / "network_dynamics" / "partials" / "modulation" / feature
        )
        for feature in CORRECTED_FEATURES
    ]
    coupling_bundles = [
        read_result_bundle(
            analysis_dir / "network_dynamics" / "partials" / "coupling" / feature
        )
        for feature in CORRECTED_FEATURES
    ]
    _require_bundle_provenance(modulation_bundles + coupling_bundles, analysis_dir)
    modulation = [bundle["result"] for bundle in modulation_bundles]
    coupling = [bundle["result"] for bundle in coupling_bundles]
    network_cells = np.concatenate(
        [result["network_cell_means"] for result in modulation], axis=-1
    )
    coupling_cells = np.concatenate(
        [result["fisher_z"] for result in coupling], axis=-1
    )
    modulation_complete = np.all(
        np.stack([result["complete_case"] for result in modulation]), axis=0
    )
    coupling_complete = np.all(
        np.stack([result["complete_case"] for result in coupling]), axis=0
    )
    modulation_contrasts = compute_factorial_contrasts(
        network_cells[modulation_complete]
    )
    coupling_contrasts = compute_factorial_contrasts(
        coupling_cells[coupling_complete]
    )
    fooof_inference = _network_family_inference(
        modulation_contrasts,
        slice(0, 2),
        permutations=permutations,
        seed=base_seed,
    )
    corrected_inference = _network_family_inference(
        modulation_contrasts,
        slice(2, 9),
        permutations=permutations,
        seed=base_seed + 1,
    )
    coupling_inference = _network_family_inference(
        coupling_contrasts,
        slice(0, 9),
        permutations=permutations,
        seed=base_seed + 2,
    )
    subjects = args.subjects.split() if args.subjects else config["bids"]["subjects"]
    runs = args.runs.split() if args.runs else config["bids"]["task_runs"]
    real_inputs = load_real_inputs(config, subjects, runs)
    mixed_coefficients, mixed_p_values, mixed_converged = _mixed_effects_arrays(
        real_inputs
    )
    all_pair_coupling = compute_all_network_pair_coupling(
        real_inputs.feature_tensor,
        real_inputs.cells,
        real_inputs.subjects,
        real_inputs.runs,
        _network_assignments(real_inputs.parcel_order),
    )
    arrays = {
        "network_cell_means": network_cells,
        "interaction": modulation_contrasts["interaction"],
        "modulation_contrasts": np.stack(
            [modulation_contrasts[name] for name in CONTRAST_WEIGHTS], axis=1
        ),
        "coupling": coupling_cells,
        "coupling_interaction": coupling_contrasts["interaction"],
        "coupling_contrasts": np.stack(
            [coupling_contrasts[name] for name in CONTRAST_WEIGHTS], axis=1
        ),
        "fooof_t_values": fooof_inference["t_values"],
        "fooof_p_fwer": fooof_inference["p_values_fwer"],
        "corrected_psd_t_values": corrected_inference["t_values"],
        "corrected_psd_p_fwer": corrected_inference["p_values_fwer"],
        "coupling_t_values": coupling_inference["t_values"],
        "coupling_p_fwer": coupling_inference["p_values_fwer"],
        "mixed_effects_coefficients": mixed_coefficients,
        "mixed_effects_p_values": mixed_p_values,
        "mixed_effects_converged": mixed_converged,
        "all_network_pair_fisher_z": all_pair_coupling["fisher_z"],
        "all_network_pair_counts": all_pair_coupling["cell_counts"],
    }
    return arrays, {
        "feature_order": list(CORRECTED_FEATURES),
        "primary": "state-by-outcome interaction",
        "simple_effects_always_reported": True,
        "contrast_order": list(CONTRAST_WEIGHTS),
        "modulation_complete_subject_n": int(modulation_complete.sum()),
        "coupling_complete_subject_n": int(coupling_complete.sum()),
        "correction": (
            "synchronized sign-flip max-|t|; FOOOF and corrected-PSD "
            "modulation families separate; DMN-DAN coupling one family"
        ),
        "mixed_effects_role": "secondary_all_available_sensitivity",
        "exploratory_network_pairs": [
            list(pair) for pair in all_pair_coupling["network_pairs"]
        ],
    }


def _network_family_inference(
    contrasts: dict[str, np.ndarray],
    feature_slice: slice,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Correct one prespecified network-dynamics analysis feature family synchronously."""
    flattened = {
        name: np.asarray(values)[..., feature_slice].reshape(len(values), -1)
        for name, values in contrasts.items()
    }
    return synchronized_sign_flip_test(
        flattened, n_permutations=permutations, seed=seed
    )


def _mixed_effects_arrays(
    inputs: AnalysisInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit secondary all-available models by Yeo network and feature."""
    assignments = _network_assignments(inputs.parcel_order)
    networks = (
        "Visual",
        "Somatomotor",
        "Dorsal Attention",
        "Ventral Attention",
        "Limbic",
        "Control",
        "Default Mode",
    )
    terms = ("Intercept", "state", "lapse", "state:lapse")
    coefficients = np.full((7, len(CORRECTED_FEATURES), len(terms)), np.nan)
    p_values = np.full_like(coefficients, np.nan)
    converged = np.zeros((7, len(CORRECTED_FEATURES)), dtype=bool)
    for network_index, network in enumerate(networks):
        values = np.nanmean(
            inputs.feature_tensor[:, assignments == network], axis=1
        )
        try:
            result = compute_mixed_effects_sensitivity(
                values,
                inputs.cells,
                inputs.subjects,
                feature_order=CORRECTED_FEATURES,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        for feature_index, feature in enumerate(CORRECTED_FEATURES):
            fitted = result["features"][feature]
            coefficients[network_index, feature_index] = [
                fitted["coefficients"][term] for term in terms
            ]
            p_values[network_index, feature_index] = [
                fitted["p_values"][term] for term in terms
            ]
            converged[network_index, feature_index] = fitted["converged"]
    return coefficients, p_values, converged


def _write_observed(
    directory: Path,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    analysis_dir: Path,
) -> Path:
    """Atomically write a render-ready real observed bundle."""
    archive = directory / "observed.npz"
    metadata = directory / "observed.json"
    if archive.exists() or metadata.exists():
        raise FileExistsError(f"immutable observed bundle exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(archive)
    analysis = json.loads((analysis_dir / "provenance.json").read_text())
    input_inventory = _collect_partial_inputs(directory)
    metadata.write_text(
        json.dumps(
            {
                "provenance": {
                    "analysis_id": analysis["analysis_id"],
                    "data_mode": "real",
                    "git": analysis["git"],
                    "config_hash": analysis["config_hash"],
                    "inputs": input_inventory or analysis.get("inputs", []),
                    "software": analysis.get("software", {}),
                },
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return directory


def _collect_partial_inputs(directory: Path) -> list[dict[str, Any]]:
    """Union stable input inventories from all validated panel partials."""
    by_path: dict[str, dict[str, Any]] = {}
    for metadata_path in (directory / "partials").rglob("observed.json"):
        metadata = json.loads(metadata_path.read_text())
        for item in metadata.get("provenance", {}).get("inputs", []):
            if isinstance(item, dict) and item.get("path"):
                by_path[item["path"]] = item
    return [by_path[path] for path in sorted(by_path)]


def _require_bundle_provenance(
    bundles: list[dict[str, Any]], analysis_dir: Path
) -> None:
    """Reject partials from a different analysis, Git state, or configuration."""
    analysis = json.loads((analysis_dir / "provenance.json").read_text())
    expected_commit = analysis["git"]["commit"]
    expected_analysis_id = analysis["analysis_id"]
    expected_config = analysis["config_hash"]
    for bundle in bundles:
        provenance = bundle["provenance"]
        commit = provenance.get("git", {}).get("commit")
        if (
            provenance.get("analysis_id") != expected_analysis_id
            or provenance.get("config_hash") != expected_config
            or commit != expected_commit
        ):
            raise ValueError("partial bundle has incompatible provenance")


def _analysis_directory(
    config: dict[str, Any], override: str | None, analysis_id: str
) -> Path:
    """Resolve an initialized immutable analysis directory."""
    root = (
        Path(override)
        if override
        else Path(config["paths"]["data_root"])
        / "processed"
        / config.get("analysis_workflow", {}).get("processed_directory", "analysis_workflow")
    )
    directory = resolve_analysis_directory(root, analysis_id)
    if not directory.exists():
        raise FileNotFoundError(f"analysis not found: {directory}")
    return directory


def main() -> None:
    """Aggregate one scientific analysis from scheduler arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument(
        "--analysis",
        required=True,
        choices=("feature_modulation", "multifeature_decoding", "network_dynamics"),
    )
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    aggregate_analysis(parser.parse_args())


if __name__ == "__main__":
    main()
