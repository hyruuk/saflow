"""Feature extraction utilities for saflow.

This module provides utilities for:
- Trial classification based on VTC (IN/OUT/MID zones)
- Feature extraction helpers

Author: Claude (Anthropic)
Date: 2026-01-31
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def classify_trials_from_vtc(
    vtc_filtered: np.ndarray,
    inout_bounds: Tuple[int, int] = (25, 75),
) -> Dict[str, np.ndarray]:
    """Classify trials into IN/OUT/MID zones based on VTC percentiles.

    Trials are classified based on their VTC (Variability Time Course) values:
    - IN zone: VTC < lower percentile (stable performance)
    - OUT zone: VTC >= upper percentile (variable performance)
    - MID zone: Between lower and upper percentiles (excluded from comparisons)

    Args:
        vtc_filtered: Filtered VTC array (one value per trial).
        inout_bounds: Tuple of (lower_percentile, upper_percentile).
            Default: (25, 75) for quartile split.

    Returns:
        Dictionary containing:
        - 'IN_idx': Array of trial indices in IN zone
        - 'OUT_idx': Array of trial indices in OUT zone
        - 'MID_idx': Array of trial indices in MID zone (if bounds differ)
        - 'IN_mask': Boolean array marking IN trials
        - 'OUT_mask': Boolean array marking OUT trials
        - 'zone_labels': Array of zone labels ('IN', 'OUT', 'MID') per trial

    Examples:
        >>> zones = classify_trials_from_vtc(vtc_filtered, inout_bounds=(25, 75))
        >>> print(f"IN trials: {len(zones['IN_idx'])}")
        >>> print(f"OUT trials: {len(zones['OUT_idx'])}")
    """
    lower_pct, upper_pct = inout_bounds

    # Compute percentile thresholds
    lower_thresh = np.nanpercentile(vtc_filtered, lower_pct)
    upper_thresh = np.nanpercentile(vtc_filtered, upper_pct)

    logger.debug(
        f"VTC thresholds: lower ({lower_pct}th pct) = {lower_thresh:.3f}, "
        f"upper ({upper_pct}th pct) = {upper_thresh:.3f}"
    )

    # Classify trials
    n_trials = len(vtc_filtered)
    zone_labels = np.array(["MID"] * n_trials, dtype=object)

    # IN zone: VTC < lower threshold
    in_mask = vtc_filtered < lower_thresh
    zone_labels[in_mask] = "IN"

    # OUT zone: VTC >= upper threshold
    out_mask = vtc_filtered >= upper_thresh
    zone_labels[out_mask] = "OUT"

    # MID zone: between thresholds (remains "MID")
    mid_mask = ~in_mask & ~out_mask

    # Get indices
    in_idx = np.where(in_mask)[0]
    out_idx = np.where(out_mask)[0]
    mid_idx = np.where(mid_mask)[0]

    logger.info(
        f"Trial classification (bounds={inout_bounds}): "
        f"IN={len(in_idx)}, OUT={len(out_idx)}, MID={len(mid_idx)}"
    )

    return {
        "IN_idx": in_idx,
        "OUT_idx": out_idx,
        "MID_idx": mid_idx,
        "IN_mask": in_mask,
        "OUT_mask": out_mask,
        "MID_mask": mid_mask,
        "zone_labels": zone_labels,
        "lower_thresh": lower_thresh,
        "upper_thresh": upper_thresh,
    }
