"""Exact behavioral/neural alignment and Schaefer-400 validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_alignment_keys(
    subjects: Sequence[str], runs: Sequence[str], onsets: Sequence[float],
    epoch_indices: np.ndarray,
) -> np.ndarray:
    """Create stable keys containing subject, run, onset, and epoch indices."""
    indices = np.asarray(epoch_indices)
    n_rows = len(subjects)
    if indices.ndim != 2 or not (len(runs) == len(onsets) == n_rows == len(indices)):
        raise ValueError("alignment fields must have the same number of rows")
    return np.asarray([
        f"{subject}|{run}|{float(onset):.9f}|{','.join(map(str, row))}"
        for subject, run, onset, row in zip(subjects, runs, onsets, indices)
    ])


def require_exact_alignment(reference: np.ndarray, candidate: np.ndarray) -> None:
    """Reject missing, reordered, or differing behavioral/neural windows."""
    expected = np.asarray(reference)
    observed = np.asarray(candidate)
    if expected.shape != observed.shape:
        raise ValueError(f"alignment shape mismatch: {expected.shape} != {observed.shape}")
    mismatch = np.flatnonzero(expected != observed)
    if mismatch.size:
        raise ValueError(f"alignment mismatch at row {int(mismatch[0])}")


def validate_schaefer_400(names: Sequence[str]) -> None:
    """Require exactly 400 unique, ordered, non-medial-wall parcel labels."""
    labels = [str(name).strip() for name in names]
    if len(labels) != 400 or len(set(labels)) != 400:
        raise ValueError("Schaefer-400 requires exactly 400 unique ordered parcels")
    forbidden = ("unknown", "medialwall", "medial_wall", "medial wall")
    if any(any(token in label.lower() for token in forbidden) for label in labels):
        raise ValueError("Schaefer-400 may not contain unknown or medial-wall labels")
