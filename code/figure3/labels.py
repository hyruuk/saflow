"""Boundary-safe VTC filtering, strict windows, shifts, and label QC."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from code.utils.behavioral import filter_vtc_gaussian_reflect

LABEL_IN = -1
LABEL_MID = 0
LABEL_OUT = 1
OUTCOME_OTHER = 0
OUTCOME_CORRECT_OMISSION = 1
OUTCOME_COMMISSION_ERROR = 2


def filter_vtc_reflect(vtc_raw: np.ndarray, fwhm: float) -> np.ndarray:
    """Gaussian-filter one run using reflected, never zero-padded, boundaries.

    Args:
        vtc_raw: One-dimensional raw VTC values for a single run.
        fwhm: Gaussian full width at half maximum, in trials.

    Returns:
        Filtered VTC with the same shape as ``vtc_raw``.
    """
    return filter_vtc_gaussian_reflect(vtc_raw, fwhm)


def classify_vtc(vtc: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Classify a run as IN/MID/OUT from run-specific percentiles."""
    values = np.asarray(vtc, dtype=float)
    low, high = np.percentile(values, bounds)
    labels = np.full(values.shape, LABEL_MID, dtype=np.int8)
    labels[values <= low] = LABEL_IN
    labels[values >= high] = LABEL_OUT
    return labels


def reconstruct_strict_labels(
    vtc: np.ndarray,
    contributing_epoch_indices: np.ndarray,
    *,
    bounds: tuple[float, float] = (25.0, 75.0),
    window_size: int = 8,
) -> np.ndarray:
    """Rebuild strict window labels from exactly ``window_size`` trial indices.

    A window is IN or OUT only when every constituent trial has that label;
    mixed windows are MID. Indices are validated but need not be contiguous,
    allowing the neural artifact to define its precise contributing epochs.
    """
    indices = np.asarray(contributing_epoch_indices)
    if indices.ndim != 2 or indices.shape[1] != window_size:
        raise ValueError(f"contributing_epoch_indices must have shape (n, {window_size})")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("contributing epoch indices must be integers")
    values = np.asarray(vtc, dtype=float)
    if indices.size and (indices.min() < 0 or indices.max() >= values.size):
        raise ValueError("contributing epoch index is outside the behavioral run")
    trial_labels = classify_vtc(values, bounds)
    constituents = trial_labels[indices]
    labels = np.full(indices.shape[0], LABEL_MID, dtype=np.int8)
    labels[np.all(constituents == LABEL_IN, axis=1)] = LABEL_IN
    labels[np.all(constituents == LABEL_OUT, axis=1)] = LABEL_OUT
    return labels


def reject_bad_windows(
    labels: np.ndarray, contributing_bad_flags: np.ndarray, *, window_size: int = 8
) -> np.ndarray:
    """Set a state label to MID when any contributing trial is bad."""
    states = np.asarray(labels, dtype=np.int8).copy()
    bad = np.asarray(contributing_bad_flags, dtype=bool)
    if bad.shape != (len(states), window_size):
        raise ValueError(f"contributing_bad_flags must have shape (n, {window_size})")
    states[np.any(bad, axis=1)] = LABEL_MID
    return states


def label_matched_rare_outcomes(
    trial_outcomes: Sequence[str], contributing_epoch_indices: np.ndarray
) -> np.ndarray:
    """Label windows from the anchor/final rare-target outcome only."""
    outcomes = np.asarray(trial_outcomes, dtype=str)
    indices = np.asarray(contributing_epoch_indices)
    if indices.ndim != 2 or indices.shape[1] != 8:
        raise ValueError("contributing_epoch_indices must have shape (n, 8)")
    if indices.size and (indices.min() < 0 or indices.max() >= len(outcomes)):
        raise ValueError("contributing epoch index is outside trial outcomes")
    anchor = outcomes[indices[:, -1]]
    labels = np.full(len(indices), OUTCOME_OTHER, dtype=np.int8)
    labels[anchor == "correct_omission"] = OUTCOME_CORRECT_OMISSION
    labels[anchor == "commission_error"] = OUTCOME_COMMISSION_ERROR
    return labels


def four_cell_labels(states: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Return the prespecified state × matched-outcome cell for each window."""
    state_values = np.asarray(states, dtype=np.int8)
    outcome_values = np.asarray(outcomes, dtype=np.int8)
    if state_values.shape != outcome_values.shape:
        raise ValueError("states and outcomes must align")
    cells = np.full(len(state_values), "", dtype="<U28")
    mapping = {
        (LABEL_IN, OUTCOME_CORRECT_OMISSION): "IN_correct_omission",
        (LABEL_IN, OUTCOME_COMMISSION_ERROR): "IN_commission_error",
        (LABEL_OUT, OUTCOME_CORRECT_OMISSION): "OUT_correct_omission",
        (LABEL_OUT, OUTCOME_COMMISSION_ERROR): "OUT_commission_error",
    }
    for key, name in mapping.items():
        cells[(state_values == key[0]) & (outcome_values == key[1])] = name
    return cells


def build_corrected_window_labels(
    vtc: np.ndarray,
    contributing_epoch_indices: np.ndarray,
    contributing_bad_flags: np.ndarray,
    trial_outcomes: Sequence[str],
    *,
    bounds: tuple[float, float] = (25.0, 75.0),
) -> dict[str, np.ndarray]:
    """Build strict states, matched outcomes, and four-cell labels."""
    states = reconstruct_strict_labels(vtc, contributing_epoch_indices, bounds=bounds)
    states = reject_bad_windows(states, contributing_bad_flags)
    outcomes = label_matched_rare_outcomes(trial_outcomes, contributing_epoch_indices)
    return {
        "state": states,
        "outcome": outcomes,
        "cell": four_cell_labels(states, outcomes),
        "bad_any": np.any(np.asarray(contributing_bad_flags, dtype=bool), axis=1),
    }


def valid_circular_offsets(n_trials: int, minimum_offset: int = 24) -> np.ndarray:
    """Return offsets whose circular distance from zero exceeds the minimum."""
    if n_trials <= 0 or minimum_offset < 0:
        raise ValueError("n_trials must be positive and minimum_offset non-negative")
    offsets = np.arange(1, n_trials)
    distance = np.minimum(offsets, n_trials - offsets)
    valid = offsets[distance > minimum_offset]
    if valid.size == 0:
        raise ValueError("run is too short for the requested minimum circular offset")
    return valid


def draw_circular_shifts(
    run_lengths: Sequence[int], minimum_offset: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one valid deterministic shift for every run."""
    return np.asarray([
        rng.choice(valid_circular_offsets(length, minimum_offset))
        for length in run_lengths
    ], dtype=int)


def shift_and_rebuild_labels(
    vtc: np.ndarray,
    indices: np.ndarray,
    offset: int,
    *,
    bounds: tuple[float, float] = (25.0, 75.0),
) -> np.ndarray:
    """Circularly shift VTC within one run and rebuild strict labels."""
    return reconstruct_strict_labels(np.roll(vtc, offset), indices, bounds=bounds)


def permute_outcomes_within_run_state(
    outcomes: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    states: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute rare outcomes within subject/run/state, preserving class counts."""
    values = np.asarray(outcomes).copy()
    subject_values = np.asarray(subjects)
    run_values = np.asarray(runs)
    state_values = np.asarray(states)
    if not (
        values.shape
        == subject_values.shape
        == run_values.shape
        == state_values.shape
    ):
        raise ValueError("outcomes, subjects, runs, and states must align")
    for subject in np.unique(subject_values):
        for run in np.unique(run_values[subject_values == subject]):
            for state in (LABEL_IN, LABEL_OUT):
                selector = (
                    (subject_values == subject)
                    & (run_values == run)
                    & (state_values == state)
                    & np.isin(
                        values,
                        (OUTCOME_CORRECT_OMISSION, OUTCOME_COMMISSION_ERROR),
                    )
                )
                values[selector] = rng.permutation(values[selector])
    return values


def summarize_label_overlap(old: np.ndarray, corrected: np.ndarray) -> dict[str, object]:
    """Summarize old/corrected counts, transitions, and retained windows."""
    old_labels = np.asarray(old, dtype=int)
    new_labels = np.asarray(corrected, dtype=int)
    if old_labels.shape != new_labels.shape:
        raise ValueError("old and corrected labels must have identical shape")
    order = np.asarray([LABEL_IN, LABEL_MID, LABEL_OUT])
    transition = np.asarray([
        [np.sum((old_labels == source) & (new_labels == target)) for target in order]
        for source in order
    ])
    old_selected = old_labels != LABEL_MID
    new_selected = new_labels != LABEL_MID
    union = np.sum(old_selected | new_selected)
    return {
        "label_order": ["IN", "MID", "OUT"],
        "old_counts": [int(np.sum(old_labels == label)) for label in order],
        "corrected_counts": [int(np.sum(new_labels == label)) for label in order],
        "transition_matrix": transition.tolist(),
        "retained": int(np.sum(old_selected & new_selected)),
        "gained": int(np.sum(~old_selected & new_selected)),
        "lost": int(np.sum(old_selected & ~new_selected)),
        "jaccard": float(np.sum(old_selected & new_selected) / union) if union else 1.0,
    }
