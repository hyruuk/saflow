"""Feature extraction utilities for saflow.

This module provides utilities for:
- Trial classification based on VTC (IN/OUT/MID zones)
- Data segmentation for epoched analysis
- Feature extraction helpers

Author: Claude (Anthropic)
Date: 2026-01-31
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_bad_trial_mask(
    onsets: np.ndarray,
    tmin: float,
    tmax: float,
    annotations,
    bad_prefix: str = "BAD_",
) -> np.ndarray:
    """Flag trials whose epoch window overlaps any BAD_* annotation.

    A trial spans [onset + tmin, onset + tmax] in seconds. A BAD_* annotation
    spans [annot.onset, annot.onset + annot.duration]. The trial is flagged
    if the two intervals overlap at all (matches mne.Epochs reject_by_annotation
    behaviour: events landing inside a BAD region are dropped).

    Parameters
    ----------
    onsets : np.ndarray
        Trial event onsets in seconds, shape (n_trials,).
    tmin, tmax : float
        Epoch window relative to onset (seconds).
    annotations : mne.Annotations or None
        Annotations from the cleaned raw recording. If None or empty, every
        trial is considered good.
    bad_prefix : str
        Annotations whose description starts with this prefix are treated as
        bad. Defaults to ``"BAD_"`` (catches BAD_AR2 and any other BAD_*).

    Returns
    -------
    bad_mask : np.ndarray
        Boolean array of length n_trials, True where the trial overlaps a
        BAD_* annotation.
    """
    n = len(onsets)
    bad_mask = np.zeros(n, dtype=bool)
    if annotations is None or len(annotations) == 0:
        return bad_mask

    descs = np.asarray(annotations.description)
    is_bad = np.array([d.startswith(bad_prefix) for d in descs], dtype=bool)
    if not is_bad.any():
        return bad_mask

    bad_starts = np.asarray(annotations.onset)[is_bad]
    bad_ends = bad_starts + np.asarray(annotations.duration)[is_bad]

    trial_starts = onsets + tmin
    trial_ends = onsets + tmax

    # Overlap test: trial_end > bad_start AND trial_start < bad_end
    # Vectorise across (n_trials, n_bad).
    overlaps = (trial_ends[:, None] > bad_starts[None, :]) & (
        trial_starts[:, None] < bad_ends[None, :]
    )
    bad_mask = overlaps.any(axis=1)
    return bad_mask


def segment_spatial_temporal_data(
    data: np.ndarray,
    events_df: pd.DataFrame,
    sfreq: float,
    tmin: float = 0.426,
    tmax: float = 1.278,
    annotations=None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Segment continuous data based on events.

    Works for any analysis space (sensor/source/atlas).

    Parameters
    ----------
    data : np.ndarray
        Continuous data array, shape (n_spatial, n_times)
    events_df : pd.DataFrame
        Events dataframe with 'onset', 'trial_type' columns
    sfreq : float
        Sampling frequency in Hz
    tmin : float
        Start time of epoch relative to event onset, in seconds
    tmax : float
        End time of epoch relative to event onset, in seconds
    annotations : mne.Annotations or None
        Annotations from the cleaned raw recording. If provided, each trial
        is tagged with a ``bad_ar2`` boolean in the returned metadata
        indicating whether its epoch window overlaps a BAD_* annotation.
        Trials are kept either way — comparative analyses are responsible
        for filtering on ``bad_ar2`` at load time.

    Returns
    -------
    segmented_array : np.ndarray
        Segmented data, shape (n_trials, n_spatial, n_samples_per_trial)
    trial_metadata : pd.DataFrame
        Metadata for each segmented trial. Includes a ``bad_ar2`` column
        when ``annotations`` is supplied (always False when not).
    """
    logger.info(f"Segmenting continuous data (tmin={tmin}, tmax={tmax})")

    # Compute time samples
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)

    # Filter to stimulus trials only
    stim_trials = events_df[events_df['trial_type'].isin(['Freq', 'Rare'])].copy()
    logger.info(f"Found {len(stim_trials)} stimulus trials")

    segmented_array = []
    trial_metadata_list = []

    for idx, trial in stim_trials.iterrows():
        # Get event onset in samples
        onset_sample = int(trial['onset'] * sfreq)

        # Check boundaries
        if onset_sample + tmax_samples >= data.shape[1]:
            logger.debug(f"Skipping trial {idx}: extends beyond data")
            continue
        if onset_sample + tmin_samples < 0:
            logger.debug(f"Skipping trial {idx}: starts before data")
            continue

        # Extract segment
        segment = data[:, onset_sample + tmin_samples : onset_sample + tmax_samples]
        segmented_array.append(segment)
        trial_metadata_list.append(trial)

    segmented_array = np.array(segmented_array)
    trial_metadata = pd.DataFrame(trial_metadata_list).reset_index(drop=True)

    # Tag bad trials (BAD_* annotations from autoreject second pass).
    onsets = trial_metadata["onset"].to_numpy(dtype=float) if len(trial_metadata) else np.empty(0)
    bad_mask = compute_bad_trial_mask(
        onsets=onsets,
        tmin=tmin,
        tmax=tmax,
        annotations=annotations,
    )
    trial_metadata["bad_ar2"] = bad_mask

    logger.info(
        f"Segmented {len(segmented_array)} trials from continuous data "
        f"({int(bad_mask.sum())} flagged bad_ar2)"
    )
    logger.debug(f"Segmented array shape: {segmented_array.shape}")

    return segmented_array, trial_metadata


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
