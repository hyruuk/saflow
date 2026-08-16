"""Validate one complete scientific result bundle before downstream rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.analysis.contracts import (
    FEATURE_MODULATION_RENDER_ARRAYS,
    FEATURE_MODULATION_SUBJECT_SPECTRA,
)
from code.utils.config import load_config
from code.analysis.provenance import resolve_analysis_directory

REQUIRED_ARRAYS = {
    "feature_modulation": set(FEATURE_MODULATION_RENDER_ARRAYS)
    | set(FEATURE_MODULATION_SUBJECT_SPECTRA)
    | {
        "raw_psd_p_fdr",
        "raw_psd_p_cluster",
        "fooof_p_fdr",
        "fooof_p_cluster",
        "corrected_psd_p_fdr",
        "corrected_psd_p_cluster",
        "decoding_p_tmax",
        "confusion_matrices",
        "effect_size_dz",
        "p_values_uncorrected",
        "p_values_fdr",
        "p_values_cluster",
        "raw_psd_modulation_equal_run",
        "fooof_modulation_equal_run",
        "corrected_psd_modulation_equal_run",
        "raw_psd_p_cluster_equal_run",
        "fooof_p_cluster_equal_run",
        "corrected_psd_p_cluster_equal_run",
        "subject_spectrum_in_equal_run",
        "subject_spectrum_out_equal_run",
        "subject_aperiodic_spectrum_in_equal_run",
        "subject_aperiodic_spectrum_out_equal_run",
        "subject_corrected_spectrum_in_equal_run",
        "subject_corrected_spectrum_out_equal_run",
        "subject_periodic_spectrum_in_equal_run",
        "subject_periodic_spectrum_out_equal_run",
        "spectrum_in_equal_run",
        "spectrum_out_equal_run",
        "aperiodic_spectrum_in_equal_run",
        "aperiodic_spectrum_out_equal_run",
        "corrected_spectrum_in_equal_run",
        "corrected_spectrum_out_equal_run",
        "periodic_spectrum_in_equal_run",
        "periodic_spectrum_out_equal_run",
        "raw_psd_p_fdr_equal_run",
        "fooof_p_fdr_equal_run",
        "corrected_psd_p_fdr_equal_run",
    },
    "multifeature_decoding": {
        "population_auc",
        "population_subject_auc",
        "population_null",
        "population_p",
        "within_subject_auc",
        "within_run_auc",
        "within_group_null_mean_auc",
        "within_group_p",
        "subject_order",
        "population_feature_reliance",
        "population_feature_reliance_p_fwer",
        "population_network_reliance",
        "population_network_reliance_p_fwer",
        "population_cell_reliance",
        "population_cell_reliance_p_fwer",
        "feature_order",
        "network_order",
    },
    "network_dynamics": {
        "network_cell_means",
        "interaction",
        "modulation_contrasts",
        "coupling",
        "coupling_interaction",
        "coupling_contrasts",
        "fooof_p_fwer",
        "corrected_psd_p_fwer",
        "coupling_p_fwer",
        "mixed_effects_coefficients",
        "all_network_pair_fisher_z",
    },
}


def validate_analysis_result(
    config_path: str,
    analysis_id: str,
    analysis_root: str | None,
    analysis: str,
) -> None:
    """Reject missing, synthetic, empty, or schema-incomplete result bundles."""
    if analysis not in REQUIRED_ARRAYS:
        raise ValueError(f"unknown analysis: {analysis}")
    config = load_config(config_path)
    root = (
        Path(analysis_root)
        if analysis_root
        else Path(config["paths"]["data_root"])
        / "processed"
        / config.get("analysis_workflow", {}).get("processed_directory", "analysis_workflow")
    )
    directory = resolve_analysis_directory(root, analysis_id) / analysis
    metadata = json.loads((directory / "observed.json").read_text())
    provenance = metadata.get("provenance", {})
    if provenance.get("analysis_id") != analysis_id:
        raise ValueError(f"{analysis} bundle has the wrong analysis ID")
    if provenance.get("data_mode") != "real":
        raise ValueError(f"{analysis} validator requires real data")
    with np.load(directory / "observed.npz", allow_pickle=False) as archive:
        missing = sorted(REQUIRED_ARRAYS[analysis] - set(archive.files))
        if missing:
            raise ValueError(f"{analysis} bundle lacks arrays: {missing}")
        empty = sorted(
            name for name in REQUIRED_ARRAYS[analysis] if archive[name].size == 0
        )
        if empty:
            raise ValueError(f"{analysis} bundle has empty arrays: {empty}")


def main() -> None:
    """Validate a real scientific result bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--analysis", required=True, choices=tuple(REQUIRED_ARRAYS))
    args = parser.parse_args()
    validate_analysis_result(
        args.config, args.analysis_id, args.analysis_root, args.analysis
    )


if __name__ == "__main__":
    main()
