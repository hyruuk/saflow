"""Tests for the outcome-stratified state-modulation panel."""

import numpy as np
import pytest

from code.visualization.outcome_state_panel import outcome_matrices, validate_outcome_arrays


def _arrays() -> dict[str, np.ndarray]:
    contrasts = np.ones((13, 5, 7, 9))
    contrasts[:, 3] *= np.arange(1, 14)[:, None, None]
    contrasts[:, 4] *= -np.arange(1, 14)[:, None, None]
    return {
        "modulation_contrasts": contrasts,
        "fooof_t_values": np.arange(70).reshape(5, 14),
        "fooof_p_fwer": np.full((5, 14), 0.04),
        "corrected_psd_t_values": np.arange(245).reshape(5, 49),
        "corrected_psd_p_fwer": np.full((5, 49), 0.06),
        "parcel_order": np.asarray([f"parcel-{index}" for index in range(400)]),
    }


def _metadata() -> dict:
    return {
        "summary": {
            "contrast_order": [
                "interaction",
                "lapse_vs_correct_within_IN",
                "lapse_vs_correct_within_OUT",
                "OUT_vs_IN_within_correct",
                "OUT_vs_IN_within_lapse",
            ]
        }
    }


def test_outcome_matrices_select_prespecified_simple_effects():
    arrays = _arrays()
    validate_outcome_arrays(arrays, _metadata())
    correct_t, correct_p = outcome_matrices(arrays, "correct_omission")
    lapse_t, lapse_p = outcome_matrices(arrays, "commission_error")
    assert correct_t.shape == correct_p.shape == (7, 9)
    assert lapse_t.shape == lapse_p.shape == (7, 9)
    assert correct_t[0, 0] == arrays["fooof_t_values"].reshape(5, 7, 2)[3, 0, 0]
    assert lapse_t[0, 0] == arrays["fooof_t_values"].reshape(5, 7, 2)[4, 0, 0]


def test_outcome_panel_rejects_wrong_contrast_order():
    metadata = _metadata()
    metadata["summary"]["contrast_order"][4] = "wrong"
    with pytest.raises(ValueError, match="contrast order"):
        validate_outcome_arrays(_arrays(), metadata)
