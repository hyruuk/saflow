"""Observed scientific workers for the three paper panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

from code.classification.multifeature_scientific import (
    NestedRidgeConfig,
    run_primary_analysis,
)
from code.paper_panels.decoding import DecodingConfig
from code.paper_panels.networks import (
    CELL_ORDER,
    YEO7_ORDER,
    compute_factorial_contrasts,
    combine_run_fisher_z,
    fisher_z_correlation,
    require_complete_cells,
    synchronized_sign_flip_test,
)

PANEL2_MODELS = ("state", "lapse_within_IN", "lapse_within_OUT")


def compute_panel1_statistics(
    in_values: np.ndarray,
    out_values: np.ndarray,
    *,
    feature_order: Sequence[str],
    alpha: float = 0.05,
) -> dict[str, object]:
    """Compute subject-paired Panel 1 maps with per-feature BH correction."""
    inside = np.asarray(in_values, dtype=float)
    outside = np.asarray(out_values, dtype=float)
    if inside.shape != outside.shape or inside.ndim != 3:
        raise ValueError("Panel 1 values must align as subjects x features x parcels")
    if inside.shape[1] != len(feature_order):
        raise ValueError("feature order does not match Panel 1 tensor")
    differences = outside - inside
    tests = ttest_1samp(differences, 0.0, axis=0, nan_policy="omit")
    corrected = np.ones_like(tests.pvalue)
    rejected = np.zeros_like(tests.pvalue, dtype=bool)
    for feature_index in range(len(feature_order)):
        valid = np.isfinite(tests.pvalue[feature_index])
        if valid.any():
            reject, p_values, _, _ = multipletests(
                tests.pvalue[feature_index, valid], alpha=alpha, method="fdr_bh"
            )
            corrected[feature_index, valid] = p_values
            rejected[feature_index, valid] = reject
    standard_deviation = np.nanstd(differences, axis=0, ddof=1)
    effect_size = np.divide(
        np.nanmean(differences, axis=0),
        standard_deviation,
        out=np.full_like(standard_deviation, np.nan),
        where=standard_deviation > 0,
    )
    return {
        "contrast": "OUT_minus_IN",
        "feature_order": tuple(feature_order),
        "subject_n": np.sum(np.isfinite(differences), axis=0),
        "mean_difference": np.nanmean(differences, axis=0),
        "effect_size_dz": effect_size,
        "t_values": tests.statistic,
        "p_values_uncorrected": tests.pvalue,
        "p_values_fdr": corrected,
        "significant_fdr": rejected,
    }


def compute_panel2_models(
    feature_tensor: np.ndarray,
    states: np.ndarray,
    outcomes: np.ndarray,
    subjects: np.ndarray,
    *,
    feature_order: Sequence[str],
    parcel_order: Sequence[str],
    config: DecodingConfig,
) -> dict[str, object]:
    """Fit the three prespecified nested Panel 2 models."""
    tensor = np.asarray(feature_tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError("feature tensor must be windows x parcels x features")
    if tensor.shape[1:] != (len(parcel_order), len(feature_order)):
        raise ValueError("feature or parcel order does not match tensor")
    state_values = np.asarray(states)
    outcome_values = np.asarray(outcomes)
    subject_values = np.asarray(subjects)
    models = {
        name: compute_panel2_model(
            tensor,
            state_values,
            outcome_values,
            subject_values,
            model=name,
            config=config,
        )
        for name in PANEL2_MODELS
    }
    return {
        "models": models,
        "model_order": PANEL2_MODELS,
        "feature_order": tuple(feature_order),
        "parcel_order": tuple(parcel_order),
        "decoding_config": asdict(config),
    }


def compute_panel2_model(
    feature_tensor: np.ndarray,
    states: np.ndarray,
    outcomes: np.ndarray,
    subjects: np.ndarray,
    *,
    model: str,
    config: DecodingConfig,
) -> dict[str, object]:
    """Fit one prespecified Panel 2 model for independent array execution."""
    if model not in PANEL2_MODELS:
        raise ValueError(f"unknown Panel 2 model: {model}")
    tensor = np.asarray(feature_tensor, dtype=float)
    state_values = np.asarray(states)
    outcome_values = np.asarray(outcomes)
    subject_values = np.asarray(subjects)
    if not (
        len(tensor)
        == len(state_values)
        == len(outcome_values)
        == len(subject_values)
    ):
        raise ValueError("Panel 2 model inputs must align")
    selector = _panel2_selector(model, state_values, outcome_values)
    labels = (
        (state_values[selector] == "OUT").astype(int)
        if model == "state"
        else (outcome_values[selector] == "commission_error").astype(int)
    )
    if np.unique(labels).size < 2:
        raise ValueError(f"{model} lacks both outcome classes")
    nested_config = NestedRidgeConfig(
        c_grid=config.c_grid,
        inner_splits=config.inner_splits,
        seed=config.seed,
    )
    return {
        "selector_count": int(selector.sum()),
        "class_counts": np.bincount(labels, minlength=2),
        **run_primary_analysis(
            tensor[selector],
            labels,
            subject_values[selector],
            nested_config,
            n_permutations=0,
        ),
    }


def _panel2_selector(
    model: str, states: np.ndarray, outcomes: np.ndarray
) -> np.ndarray:
    """Return the prespecified observation selector for one decoding model."""
    rare = np.isin(outcomes, ("correct_omission", "commission_error"))
    if model == "state":
        return np.isin(states, ("IN", "OUT"))
    if model == "lapse_within_IN":
        return (states == "IN") & rare
    return (states == "OUT") & rare


def compute_panel3_modulation(
    parcel_values: np.ndarray,
    cell_labels: np.ndarray,
    subjects: np.ndarray,
    network_assignments: Sequence[str],
    *,
    minimum_windows: int,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Aggregate Yeo-7 cells and test interaction plus all simple effects."""
    values = np.asarray(parcel_values, dtype=float)
    labels = np.asarray(cell_labels)
    subject_values = np.asarray(subjects)
    assignments = np.asarray(network_assignments)
    if values.ndim != 3 or values.shape[1] != len(assignments):
        raise ValueError("values must be windows x parcels x features")
    if set(assignments) != set(YEO7_ORDER):
        raise ValueError("parcel assignments must cover exactly the Yeo-7 networks")
    unique_subjects = np.unique(subject_values)
    network_cells = np.full(
        (len(unique_subjects), len(CELL_ORDER), len(YEO7_ORDER), values.shape[2]),
        np.nan,
    )
    counts = np.zeros((len(unique_subjects), len(CELL_ORDER)), dtype=int)
    for subject_index, subject in enumerate(unique_subjects):
        for cell_index, cell in enumerate(CELL_ORDER):
            selector = (subject_values == subject) & (labels == cell)
            counts[subject_index, cell_index] = selector.sum()
            for network_index, network in enumerate(YEO7_ORDER):
                network_cells[subject_index, cell_index, network_index] = np.nanmean(
                    values[selector][:, assignments == network], axis=(0, 1)
                )
    complete, report = require_complete_cells(counts, minimum_windows)
    if complete.sum() < 2:
        raise ValueError("Panel 3 primary modulation requires at least two complete subjects")
    contrasts = compute_factorial_contrasts(network_cells[complete])
    flattened = {
        name: contrast.reshape(complete.sum(), -1)
        for name, contrast in contrasts.items()
    }
    inference = synchronized_sign_flip_test(flattened, n_permutations, seed)
    return {
        "network_order": YEO7_ORDER,
        "cell_order": CELL_ORDER,
        "subject_order": unique_subjects,
        "cell_counts": counts,
        "complete_case": complete,
        "exclusion_report": report,
        "network_cell_means": network_cells,
        "contrasts": contrasts,
        "inference": inference,
    }


def compute_panel3_coupling(
    parcel_values: np.ndarray,
    cell_labels: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    network_assignments: Sequence[str],
    *,
    minimum_windows: int,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Estimate weighted within-cell DMN–DAN Fisher-z coupling."""
    values = np.asarray(parcel_values, dtype=float)
    labels = np.asarray(cell_labels)
    subject_values = np.asarray(subjects)
    run_values = np.asarray(runs)
    assignments = np.asarray(network_assignments)
    if values.ndim != 3 or values.shape[1] != len(assignments):
        raise ValueError("values must be windows x parcels x features")
    default_mask = assignments == "Default Mode"
    attention_mask = assignments == "Dorsal Attention"
    if not default_mask.any() or not attention_mask.any():
        raise ValueError("DMN and DAN parcels are required for coupling")
    unique_subjects = np.unique(subject_values)
    coupling = np.full(
        (len(unique_subjects), len(CELL_ORDER), values.shape[2]), np.nan
    )
    counts = np.zeros((len(unique_subjects), len(CELL_ORDER)), dtype=int)
    for subject_index, subject in enumerate(unique_subjects):
        for cell_index, cell in enumerate(CELL_ORDER):
            run_estimates: list[list[float]] = []
            run_counts = []
            subject_cell = (subject_values == subject) & (labels == cell)
            counts[subject_index, cell_index] = subject_cell.sum()
            for run in np.unique(run_values[subject_values == subject]):
                selector = subject_cell & (run_values == run)
                count = int(selector.sum())
                if count < 3:
                    continue
                default = np.nanmean(values[selector][:, default_mask], axis=1)
                attention = np.nanmean(
                    values[selector][:, attention_mask], axis=1
                )
                run_estimates.append(
                    [
                        fisher_z_correlation(default[:, index], attention[:, index])
                        for index in range(values.shape[2])
                    ]
                )
                run_counts.append(count)
            if run_estimates:
                estimates = np.asarray(run_estimates)
                coupling[subject_index, cell_index] = [
                    combine_run_fisher_z(estimates[:, index], run_counts)
                    for index in range(values.shape[2])
                ]
    complete, report = require_complete_cells(counts, minimum_windows)
    complete &= np.all(np.isfinite(coupling), axis=(1, 2))
    for index, included in enumerate(complete):
        report[index]["included"] = bool(included)
        if not included and report[index]["reason"] is None:
            report[index]["reason"] = "non-estimable within-run DMN-DAN association"
    if complete.sum() < 2:
        raise ValueError("Panel 3 coupling requires at least two complete subjects")
    contrasts = compute_factorial_contrasts(coupling[complete])
    inference = synchronized_sign_flip_test(
        contrasts, n_permutations=n_permutations, seed=seed
    )
    return {
        "network_pair": ("Default Mode", "Dorsal Attention"),
        "cell_order": CELL_ORDER,
        "subject_order": unique_subjects,
        "cell_counts": counts,
        "complete_case": complete,
        "exclusion_report": report,
        "fisher_z": coupling,
        "contrasts": contrasts,
        "inference": inference,
    }


def compute_all_network_pair_coupling(
    parcel_values: np.ndarray,
    cell_labels: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    network_assignments: Sequence[str],
) -> dict[str, object]:
    """Export exploratory Fisher-z coupling for every unordered Yeo-7 pair."""
    values = np.asarray(parcel_values, dtype=float)
    labels = np.asarray(cell_labels)
    subject_values = np.asarray(subjects)
    run_values = np.asarray(runs)
    assignments = np.asarray(network_assignments)
    pairs = tuple(combinations(YEO7_ORDER, 2))
    unique_subjects = np.unique(subject_values)
    estimates = np.full(
        (
            len(unique_subjects),
            len(CELL_ORDER),
            len(pairs),
            values.shape[2],
        ),
        np.nan,
    )
    counts = np.zeros((len(unique_subjects), len(CELL_ORDER)), dtype=int)
    for subject_index, subject in enumerate(unique_subjects):
        for cell_index, cell in enumerate(CELL_ORDER):
            subject_cell = (subject_values == subject) & (labels == cell)
            counts[subject_index, cell_index] = int(subject_cell.sum())
            for pair_index, (first_network, second_network) in enumerate(pairs):
                run_estimates = []
                run_counts = []
                for run in np.unique(run_values[subject_values == subject]):
                    selector = subject_cell & (run_values == run)
                    count = int(selector.sum())
                    if count < 3:
                        continue
                    first = np.nanmean(
                        values[selector][:, assignments == first_network], axis=1
                    )
                    second = np.nanmean(
                        values[selector][:, assignments == second_network], axis=1
                    )
                    run_estimates.append(
                        [
                            fisher_z_correlation(
                                first[:, feature_index],
                                second[:, feature_index],
                            )
                            for feature_index in range(values.shape[2])
                        ]
                    )
                    run_counts.append(count)
                if run_estimates:
                    run_array = np.asarray(run_estimates)
                    estimates[subject_index, cell_index, pair_index] = [
                        combine_run_fisher_z(
                            run_array[:, feature_index], run_counts
                        )
                        for feature_index in range(values.shape[2])
                    ]
    return {
        "analysis_role": "exploratory_all_yeo_network_pairs",
        "network_pairs": pairs,
        "subject_order": unique_subjects,
        "cell_order": CELL_ORDER,
        "cell_counts": counts,
        "fisher_z": estimates,
    }


def compute_grouped_reliance(
    joint_auc: float, shuffled_auc: Mapping[str, Sequence[float]]
) -> dict[str, np.ndarray]:
    """Convert held-out grouped-shuffle scores into predictive reliance."""
    names = tuple(shuffled_auc)
    values = np.asarray([shuffled_auc[name] for name in names], dtype=float)
    return {
        "group_order": np.asarray(names),
        "subject_delta_auc": float(joint_auc) - values,
        "mean_delta_auc": np.nanmean(float(joint_auc) - values, axis=1),
    }


def compute_mixed_effects_sensitivity(
    values: np.ndarray,
    cell_labels: np.ndarray,
    subjects: np.ndarray,
    *,
    feature_order: Sequence[str],
) -> dict[str, object]:
    """Fit all-available random-intercept state × outcome sensitivity models."""
    observations = np.asarray(values, dtype=float)
    cells = np.asarray(cell_labels)
    subject_values = np.asarray(subjects)
    if observations.ndim != 2 or observations.shape[1] != len(feature_order):
        raise ValueError("values must be windows x features")
    valid_cells = np.isin(cells, CELL_ORDER)
    state = np.where(np.char.startswith(cells.astype(str), "OUT"), 1.0, 0.0)
    lapse = np.where(np.char.endswith(cells.astype(str), "commission_error"), 1.0, 0.0)
    models = {}
    for index, feature in enumerate(feature_order):
        frame = pd.DataFrame(
            {
                "value": observations[:, index],
                "state": state,
                "lapse": lapse,
                "subject": subject_values,
                "valid_cell": valid_cells,
            }
        )
        frame = frame[frame["valid_cell"] & np.isfinite(frame["value"])]
        if frame["subject"].nunique() < 2:
            raise ValueError("mixed-effects sensitivity requires at least two subjects")
        fitted = mixedlm(
            "value ~ state * lapse", frame, groups=frame["subject"]
        ).fit(reml=False, method="lbfgs", disp=False)
        terms = ("Intercept", "state", "lapse", "state:lapse")
        models[feature] = {
            "n_observations": int(len(frame)),
            "n_subjects": int(frame["subject"].nunique()),
            "coefficients": {term: float(fitted.params[term]) for term in terms},
            "p_values": {term: float(fitted.pvalues[term]) for term in terms},
            "converged": bool(fitted.converged),
        }
    return {
        "analysis_role": "secondary_all_available_sensitivity",
        "model": "random subject intercept; state * outcome fixed effects",
        "features": models,
    }
