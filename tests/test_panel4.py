"""Panel 4 network-attribution analysis tests."""

import numpy as np

from code.analysis.panel4 import compute_modulation_inference, compute_state_means


def test_state_means_pool_outcomes_by_window_count():
    cell_means = np.asarray([[[[1.0]], [[3.0]], [[5.0]], [[9.0]]]])
    cell_counts = np.asarray([[3, 1, 1, 3]])
    result = compute_state_means(cell_means, cell_counts)
    np.testing.assert_allclose(result[:, :, 0, 0], [[1.5, 8.0]])


def test_state_means_allow_missing_outcome_cell():
    cell_means = np.asarray([[[[2.0]], [[np.nan]], [[4.0]], [[6.0]]]])
    cell_counts = np.asarray([[5, 0, 2, 2]])
    result = compute_state_means(cell_means, cell_counts)
    np.testing.assert_allclose(result[:, :, 0, 0], [[2.0, 5.0]])


def test_modulation_inference_preserves_network_feature_shapes():
    differences = np.random.default_rng(4).normal(size=(12, 7, 9))
    result = compute_modulation_inference(differences, permutations=19, seed=8)
    assert result["fooof_t_values"].shape == (7, 2)
    assert result["fooof_p_fwer"].shape == (7, 2)
    assert result["corrected_psd_t_values"].shape == (7, 7)
    assert result["corrected_psd_p_fwer"].shape == (7, 7)
