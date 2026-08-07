"""Validate one complete real panel bundle before downstream rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.figure3.contracts import PANEL1_RENDER_ARRAYS
from code.utils.config import load_config

REQUIRED_ARRAYS = {
    "panel1": set(PANEL1_RENDER_ARRAYS)
    | {
        "raw_psd_p_fdr",
        "fooof_p_fdr",
        "corrected_psd_p_fdr",
        "decoding_p_tmax",
        "confusion_matrices",
        "effect_size_dz",
        "p_values_uncorrected",
        "p_values_fdr",
    },
    "panel2": {
        "joint_auc",
        "standalone_feature_auc",
        "feature_contribution",
        "parcel_contribution",
        "joint_auc_p_fwer",
        "feature_family_p_fwer",
        "parcel_family_p_fwer",
        "held_out_probabilities_state",
        "held_out_probabilities_lapse_within_IN",
        "held_out_probabilities_lapse_within_OUT",
    },
    "panel3": {
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


def validate_panel(
    config_path: str,
    analysis_id: str,
    analysis_root: str | None,
    panel: str,
) -> None:
    """Reject missing, synthetic, empty, or schema-incomplete panel bundles."""
    if panel not in REQUIRED_ARRAYS:
        raise ValueError(f"unknown panel: {panel}")
    config = load_config(config_path)
    root = (
        Path(analysis_root)
        if analysis_root
        else Path(config["paths"]["data_root"])
        / "processed"
        / config.get("figure3", {}).get("processed_directory", "figure3")
    )
    directory = root / analysis_id / panel
    metadata = json.loads((directory / "observed.json").read_text())
    provenance = metadata.get("provenance", {})
    if provenance.get("analysis_id") != analysis_id:
        raise ValueError(f"{panel} bundle has the wrong analysis ID")
    if provenance.get("data_mode") != "real":
        raise ValueError(f"{panel} validator requires real data")
    with np.load(directory / "observed.npz", allow_pickle=False) as archive:
        missing = sorted(REQUIRED_ARRAYS[panel] - set(archive.files))
        if missing:
            raise ValueError(f"{panel} bundle lacks arrays: {missing}")
        empty = sorted(name for name in REQUIRED_ARRAYS[panel] if archive[name].size == 0)
        if empty:
            raise ValueError(f"{panel} bundle has empty arrays: {empty}")


def main() -> None:
    """Validate a real panel bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--panel", required=True)
    args = parser.parse_args()
    validate_panel(args.config, args.analysis_id, args.analysis_root, args.panel)


if __name__ == "__main__":
    main()
