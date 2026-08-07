"""Synchronized family-wise statistics for Panel analysis maps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import ttest_1samp


def paired_effect_size(differences: np.ndarray) -> np.ndarray:
    """Compute paired Cohen's dz at each map location."""
    values = np.asarray(differences, dtype=float)
    return np.nanmean(values, axis=0) / np.nanstd(values, axis=0, ddof=1)


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
) -> dict[str, np.ndarray]:
    """Correct all maps in one family with synchronized sign flips and max mass.

    Args:
        differences: Paired differences shaped ``(subjects, maps, locations)``.
        adjacency: Neighbor indices for each location.
        n_permutations: Number of synchronized subject sign-flip permutations.
        cluster_threshold: Absolute t threshold forming clusters.
        seed: Deterministic random seed.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 3 or len(adjacency) != values.shape[2]:
        raise ValueError("differences must be subjects x maps x locations matching adjacency")
    if n_permutations < 1 or cluster_threshold <= 0:
        raise ValueError("n_permutations and cluster_threshold must be positive")
    observed = ttest_1samp(values, 0.0, axis=0, nan_policy="omit").statistic
    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_permutations)
    for permutation in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(values.shape[0], 1, 1))
        permuted = ttest_1samp(values * signs, 0.0, axis=0, nan_policy="omit").statistic
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
        "effect_size_dz": paired_effect_size(values),
        "p_values_fwer": corrected,
        "null_max_cluster_mass": null_max,
    }
