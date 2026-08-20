"""Synchronized family-wise statistics for Saflow analysis maps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _participant_weights(values: np.ndarray, weights: Sequence[float] | None) -> np.ndarray:
    """Broadcast optional participant weights across a map array's trailing axes."""
    count = values.shape[0]
    if weights is None:
        resolved = np.ones(count, dtype=float)
    else:
        resolved = np.asarray(weights, dtype=float)
        if resolved.shape != (count,):
            raise ValueError("weights must supply one positive value per participant")
        if not np.all(np.isfinite(resolved)) or np.any(resolved <= 0):
            raise ValueError("participant weights must be finite and positive")
    return resolved.reshape((count,) + (1,) * (values.ndim - 1))


def _weighted_moments(
    values: np.ndarray, weights: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return NaN-aware weighted mean, variance, and squared standard-error scale."""
    weight = _participant_weights(values, weights)
    finite = np.isfinite(values)
    spread = np.where(finite, weight, 0.0)
    total = spread.sum(axis=0)
    squared = np.square(spread).sum(axis=0)
    zeros = np.zeros_like(total)
    mean = np.divide(
        (spread * np.where(finite, values, 0.0)).sum(axis=0),
        total,
        out=zeros.copy(),
        where=total > 0,
    )
    effective = total - np.divide(squared, total, out=zeros.copy(), where=total > 0)
    variance = np.divide(
        (spread * np.where(finite, np.square(values - mean), 0.0)).sum(axis=0),
        effective,
        out=zeros.copy(),
        where=effective > 0,
    )
    scale = np.divide(squared, np.square(total), out=zeros.copy(), where=total > 0)
    return mean, variance, scale


def weighted_one_sample_t(
    differences: np.ndarray, weights: Sequence[float] | None = None
) -> np.ndarray:
    """Compute NaN-aware one-sample t statistics with optional participant weights.

    Equal weights reproduce the unweighted paired statistic exactly. Unequal
    weights use reliability-weighted mean and variance, so a participant's
    influence tracks the precision of their own contrast estimate.
    """
    values = np.asarray(differences, dtype=float)
    mean, variance, scale = _weighted_moments(values, weights)
    error = np.sqrt(variance * scale)
    return np.divide(mean, error, out=np.zeros_like(mean), where=error > 0)


def paired_effect_size(
    differences: np.ndarray, weights: Sequence[float] | None = None
) -> np.ndarray:
    """Compute paired Cohen's dz at each map location."""
    values = np.asarray(differences, dtype=float)
    if weights is None:
        return np.nanmean(values, axis=0) / np.nanstd(values, axis=0, ddof=1)
    mean, variance, _ = _weighted_moments(values, weights)
    deviation = np.sqrt(variance)
    return np.divide(mean, deviation, out=np.zeros_like(mean), where=deviation > 0)


def _clusters(mask: np.ndarray, adjacency: Sequence[Sequence[int]]) -> list[np.ndarray]:
    remaining = set(np.flatnonzero(mask).tolist())
    clusters: list[np.ndarray] = []
    while remaining:
        stack = [remaining.pop()]
        cluster = []
        while stack:
            node = stack.pop()
            cluster.append(node)
            neighbors = set(adjacency[node]) & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        clusters.append(np.asarray(cluster, dtype=int))
    return clusters


def synchronized_cluster_mass_test(
    differences: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    *,
    n_permutations: int,
    cluster_threshold: float,
    seed: int,
    weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Correct all maps in one family with synchronized sign flips and max mass.

    Args:
        differences: Paired differences shaped ``(subjects, maps, locations)``.
        adjacency: Neighbor indices for each location.
        n_permutations: Number of synchronized subject sign-flip permutations.
        cluster_threshold: Absolute t threshold forming clusters.
        seed: Deterministic random seed.
        weights: Optional per-participant weights held fixed across sign flips.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 3 or len(adjacency) != values.shape[2]:
        raise ValueError("differences must be subjects x maps x locations matching adjacency")
    if n_permutations < 1 or cluster_threshold <= 0:
        raise ValueError("n_permutations and cluster_threshold must be positive")
    observed = weighted_one_sample_t(values, weights)
    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_permutations)
    for permutation in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(values.shape[0], 1, 1))
        permuted = weighted_one_sample_t(values * signs, weights)
        masses = [
            np.sum(np.abs(permuted[map_index, cluster]))
            for map_index in range(values.shape[1])
            for cluster in _clusters(np.abs(permuted[map_index]) >= cluster_threshold, adjacency)
        ]
        null_max[permutation] = max(masses, default=0.0)
    corrected = np.ones_like(observed, dtype=float)
    for map_index in range(values.shape[1]):
        for cluster in _clusters(np.abs(observed[map_index]) >= cluster_threshold, adjacency):
            mass = np.sum(np.abs(observed[map_index, cluster]))
            corrected[map_index, cluster] = (1 + np.sum(null_max >= mass)) / (n_permutations + 1)
    return {
        "t_values": observed,
        "effect_size_dz": paired_effect_size(values, weights),
        "p_values_fwer": corrected,
        "null_max_cluster_mass": null_max,
    }
