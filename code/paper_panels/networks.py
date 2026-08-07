"""Factorial Yeo-network modulation and coupling utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

CELL_ORDER = (
    "IN_correct_omission",
    "IN_commission_error",
    "OUT_correct_omission",
    "OUT_commission_error",
)
CONTRAST_WEIGHTS = {
    "interaction": np.asarray([1.0, -1.0, -1.0, 1.0]),
    "lapse_vs_correct_within_IN": np.asarray([-1.0, 1.0, 0.0, 0.0]),
    "lapse_vs_correct_within_OUT": np.asarray([0.0, 0.0, -1.0, 1.0]),
    "OUT_vs_IN_within_correct": np.asarray([-1.0, 0.0, 1.0, 0.0]),
    "OUT_vs_IN_within_lapse": np.asarray([0.0, -1.0, 0.0, 1.0]),
}
YEO7_ORDER = (
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Ventral Attention",
    "Limbic",
    "Control",
    "Default Mode",
)


def compute_factorial_contrasts(cell_values: np.ndarray) -> dict[str, np.ndarray]:
    """Apply prespecified contrasts to values ordered by ``CELL_ORDER``."""
    values = np.asarray(cell_values, dtype=float)
    if values.ndim < 2 or values.shape[1] != len(CELL_ORDER):
        raise ValueError("cell values must be subjects x four cells x ...")
    return {
        name: np.tensordot(values, weights, axes=(1, 0))
        for name, weights in CONTRAST_WEIGHTS.items()
    }


def require_complete_cells(
    counts: np.ndarray, minimum: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return deterministic complete-case mask and exclusion report."""
    values = np.asarray(counts, dtype=int)
    if values.ndim != 2 or values.shape[1] != len(CELL_ORDER):
        raise ValueError("counts must be subjects x four cells")
    mask = np.all(values >= minimum, axis=1)
    report = [
        {
            "subject_index": index,
            "included": bool(mask[index]),
            "counts": dict(zip(CELL_ORDER, row.tolist())),
            "reason": None if mask[index] else f"requires >= {minimum} in every cell",
        }
        for index, row in enumerate(values)
    ]
    return mask, report


def fisher_z_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return a clipped Fisher-z Pearson association."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    correlation = np.corrcoef(left[valid], right[valid])[0, 1]
    return float(np.arctanh(np.clip(correlation, -0.999999, 0.999999)))


def combine_run_fisher_z(
    estimates: Sequence[float], observation_counts: Sequence[int]
) -> float:
    """Combine run-level Fisher-z estimates using ``n - 3`` precision weights."""
    values = np.asarray(estimates, dtype=float)
    counts = np.asarray(observation_counts, dtype=int)
    valid = np.isfinite(values) & (counts >= 4)
    if not valid.any():
        return float("nan")
    weights = counts[valid] - 3
    return float(np.average(values[valid], weights=weights))


def synchronized_sign_flip_test(
    contrasts: Mapping[str, np.ndarray], n_permutations: int, seed: int
) -> dict[str, object]:
    """Test all contrasts/locations with one synchronized max-|t| family."""
    names = tuple(contrasts)
    arrays = [np.asarray(contrasts[name], dtype=float) for name in names]
    if not arrays or any(array.ndim != 2 for array in arrays):
        raise ValueError("contrasts must contain subject x test arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("contrast arrays must align")
    stack = np.stack(arrays, axis=1)
    observed = _one_sample_t(stack)
    generator = np.random.default_rng(seed)
    null_max = np.empty(n_permutations)
    for index in range(n_permutations):
        signs = generator.choice((-1.0, 1.0), size=(stack.shape[0], 1, 1))
        null_max[index] = np.nanmax(np.abs(_one_sample_t(stack * signs)))
    corrected = (
        1
        + np.sum(
            null_max[:, None, None] >= np.abs(observed)[None, :, :], axis=0
        )
    ) / (n_permutations + 1)
    return {
        "contrast_order": names,
        "t_values": observed,
        "p_values_fwer": corrected,
        "null_max_abs_t": null_max,
    }


def _one_sample_t(values: np.ndarray) -> np.ndarray:
    """Compute NaN-aware one-sample t statistics."""
    count = np.sum(np.isfinite(values), axis=0)
    mean = np.nanmean(values, axis=0)
    error = np.nanstd(values, axis=0, ddof=1) / np.sqrt(count)
    return np.divide(mean, error, out=np.zeros_like(mean), where=error > 0)
