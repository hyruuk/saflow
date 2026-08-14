"""Stable scientific and artifact contracts for the panel-analysis workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "1.2.0"


@dataclass(frozen=True)
class FrequencyBand:
    """Describe one canonical frequency band."""

    display_name: str
    key: str
    low_hz: float
    high_hz: float


CANONICAL_BANDS = (
    FrequencyBand("Theta", "theta", 4.0, 8.0),
    FrequencyBand("Alpha", "alpha", 8.0, 12.0),
    FrequencyBand("Low Beta", "lobeta", 12.0, 20.0),
    FrequencyBand("High Beta", "hibeta", 20.0, 30.0),
    FrequencyBand("Gamma 1", "gamma1", 30.0, 60.0),
    FrequencyBand("Gamma 2", "gamma2", 60.0, 90.0),
    FrequencyBand("Gamma 3", "gamma3", 90.0, 120.0),
)
CANONICAL_BAND_KEYS = tuple(band.key for band in CANONICAL_BANDS)
BAND_ALIASES = {
    "low_beta": "lobeta",
    "high_beta": "hibeta",
    "low_gamma": "gamma1",
    "high_gamma": "gamma2",
}
FOOOF_FEATURES = ("fooof_exponent", "fooof_offset")
CORRECTED_PSD_FEATURES = tuple(f"psd_corrected_{key}" for key in CANONICAL_BAND_KEYS)
FEATURE_MODULATION_FEATURES = (
    *(f"psd_{key}" for key in CANONICAL_BAND_KEYS),
    *FOOOF_FEATURES,
    *CORRECTED_PSD_FEATURES,
)
CORRECTED_FEATURES = (*FOOOF_FEATURES, *CORRECTED_PSD_FEATURES)
MULTIFEATURE_FEATURES = tuple(
    feature for feature in CORRECTED_FEATURES if feature != "fooof_r_squared"
)
FEATURE_DISPLAY_NAMES = {
    "fooof_exponent": "FOOOF exponent",
    "fooof_offset": "FOOOF offset",
    **{
        f"psd_corrected_{band.key}": f"Corrected {band.display_name}"
        for band in CANONICAL_BANDS
    },
}
PANEL_COMPONENTS = {
    "panel1": (
        "A_raw_PSD_modulation",
        "B_raw_PSD_decoding",
        "C_raw_spectrum",
        "D_aperiodic_spectrum",
        "E_corrected_spectrum",
        "F_periodic_spectrum",
        "G_FOOOF_modulation",
        "H_FOOOF_decoding",
        "I_corrected_PSD_modulation",
        "J_corrected_PSD_decoding",
    ),
    "panel2": (
        "A_model_performance",
        "B_standalone_features",
        "C_feature_reliance",
        "D_state_parcels",
        "E_lapse_in_parcels",
        "F_lapse_out_parcels",
    ),
    "panel3": (
        "A_four_cell_overview",
        "B_interaction",
        "C_simple_effects",
        "D_network_summary",
        "E_dmn_dan_coupling",
        "F_coupling_contrasts",
    ),
}
PANEL1_SLIDE_COMPONENTS = (
    "A_raw_PSD_modulation",
    "B_raw_PSD_decoding",
    "C-F_spectral_decomposition",
    "G_FOOOF_modulation",
    "H_FOOOF_decoding",
    "I_corrected_PSD_modulation",
    "J_corrected_PSD_decoding",
)
FEATURE_MODULATION_RENDER_ARRAYS = (
    "raw_psd_modulation",
    "raw_psd_auc",
    "frequency",
    "spectrum_in",
    "spectrum_out",
    "aperiodic_spectrum_in",
    "aperiodic_spectrum_out",
    "corrected_spectrum_in",
    "corrected_spectrum_out",
    "periodic_spectrum_in",
    "periodic_spectrum_out",
    "fooof_modulation",
    "fooof_auc",
    "corrected_psd_modulation",
    "corrected_psd_auc",
)

FEATURE_MODULATION_SUBJECT_SPECTRA = (
    "subject_spectrum_in",
    "subject_spectrum_out",
    "subject_aperiodic_spectrum_in",
    "subject_aperiodic_spectrum_out",
    "subject_corrected_spectrum_in",
    "subject_corrected_spectrum_out",
    "subject_periodic_spectrum_in",
    "subject_periodic_spectrum_out",
    "spectral_subject_order",
)

PANEL_SPECS = {
    "panel1": {
        "composite_filename": "panel1_feature_modulation.png",
        "composite_directory": "manuscript",
        "slide_directory": "panel1_feature_modulation",
        "layout": "A-J feature-modulation narrative; exponent/offset maps and widened C-F spectra",
        "features": FEATURE_MODULATION_FEATURES,
    },
    "panel2": {
        "composite_filename": "panel2_multifeature_decoding.png",
        "composite_directory": "manuscript",
        "slide_directory": "panel2_multifeature_decoding",
        "layout": "three-model performance, feature reliance, and parcel reliance",
        "features": MULTIFEATURE_FEATURES,
    },
    "panel3": {
        "composite_filename": "panel3_network_dynamics.png",
        "composite_directory": "manuscript",
        "slide_directory": "panel3_network_dynamics",
        "layout": "four-cell modulation, contrasts, and DMN-DAN coupling",
        "features": CORRECTED_FEATURES,
    },
}
PANEL_ANALYSES = {
    "panel1": "feature_modulation",
    "panel2": "multifeature_decoding",
    "panel3": "network_dynamics",
}

RESULT_SCHEMA_NAMES = (
    "labels",
    "maps",
    "decoding",
    "factorial_networks",
    "coupling",
    "compact_export",
    "figure",
    "dag_manifest",
)


def canonical_band_key(key: str) -> str:
    """Return a canonical compatibility key and reject noncanonical bands."""
    canonical = BAND_ALIASES.get(key, key)
    if canonical not in CANONICAL_BAND_KEYS:
        raise ValueError(f"{key!r} is not a canonical band")
    return canonical


def frequency_band_manifest() -> list[dict[str, Any]]:
    """Return the serializable ordered band manifest."""
    return [asdict(band) for band in CANONICAL_BANDS]


def schema_catalog() -> dict[str, dict[str, Any]]:
    """Return minimal versioned schemas shared by real and synthetic bundles."""
    common = {
        "schema_version": SCHEMA_VERSION,
        "required_provenance": [
            "analysis_id",
            "data_mode",
            "git",
            "config_hash",
            "inputs",
            "software",
        ],
    }
    return {
        name: {
            **common,
            "schema_name": name,
            "required": _required_fields(name),
        }
        for name in RESULT_SCHEMA_NAMES
    }


def _required_fields(name: str) -> list[str]:
    """Return required payload fields for one schema."""
    fields = {
        "labels": ["alignment_keys", "trial_indices", "state", "outcome", "bad_any"],
        "maps": ["feature_order", "parcel_order", "contrasts", "statistics"],
        "decoding": ["models", "held_out_probabilities", "metrics", "contributions"],
        "factorial_networks": ["network_order", "cells", "contrasts", "complete_case"],
        "coupling": ["network_pairs", "cells", "fisher_z", "contrasts"],
        "compact_export": ["panels", "tables", "render_arrays"],
        "figure": ["panel", "path", "dpi", "data_mode", "render_parameters"],
        "dag_manifest": ["analysis_id", "nodes", "edges", "array_cells", "provenance"],
    }
    return fields[name]
