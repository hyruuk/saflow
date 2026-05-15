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
    n_events_window: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Segment continuous data based on events.

    Works for any analysis space (sensor/source/atlas). Supports a sliding
    multi-trial window mode (``n_events_window > 1``) that matches the
    cc_saflow analysis: each segment ends at the current event and spans
    ``n_events_window`` consecutive trials (stride = 1 trial).

    Parameters
    ----------
    data : np.ndarray
        Continuous data array, shape (n_spatial, n_times)
    events_df : pd.DataFrame
        Events dataframe with 'onset', 'trial_type' columns
    sfreq : float
        Sampling frequency in Hz
    tmin : float
        Start time of one trial relative to event onset, in seconds
    tmax : float
        End time of one trial relative to event onset, in seconds
    annotations : mne.Annotations or None
        Annotations from the cleaned raw recording. Each trial is tagged
        with a ``bad_ar2`` boolean indicating whether its single-trial epoch
        overlaps a BAD_* annotation. In windowed mode the per-trial flags
        are aggregated to ``window_any_bad`` (ANY of N trials bad), and
        ``bad_ar2`` is set to this window-level value so downstream filters
        still work.
    n_events_window : int
        Number of consecutive stim trials per epoch. 1 = single-trial mode
        (default, original behaviour). 8 = cc_saflow's sliding-window mode.

    Returns
    -------
    segmented_array : np.ndarray
        Segmented data, shape (n_epochs, n_spatial, n_samples_per_epoch),
        where n_samples_per_epoch = (tmax-tmin) * sfreq * n_events_window.
    trial_metadata : pd.DataFrame
        Metadata for each segmented epoch. Always includes ``bad_ar2``.
        When ``n_events_window > 1`` also includes ``included_VTC``,
        ``included_task``, ``included_bad_ar2`` (length-N arrays) and
        aggregates ``window_vtc_mean``, ``window_any_bad``, and per-task
        counts ``window_n_cc``/``co``/``ce``/``oe``.
    """
    logger.info(
        f"Segmenting continuous data (tmin={tmin}, tmax={tmax}, "
        f"n_events_window={n_events_window})"
    )

    # Compute time samples for one trial
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)
    epoch_length = tmax_samples - tmin_samples
    # In windowed mode, the slice extends backward by (N-1) trial-lengths
    win_tmin_samples = tmin_samples - epoch_length * (n_events_window - 1)

    # Filter to stimulus trials only (reset index for positional iteration)
    stim_trials = (
        events_df[events_df["trial_type"].isin(["Freq", "Rare"])]
        .reset_index(drop=True)
        .copy()
    )
    logger.info(f"Found {len(stim_trials)} stimulus trials")

    # Pre-compute bad_ar1 / bad_ar2 for ALL stim trials so windowed mode can
    # aggregate. The masks are matched by annotation prefix: AR2 is the
    # post-ICA verdict (BAD_AR2), AR1 the pre-ICA pass (BAD_AR1). BAD_AR1
    # annotations are only present on raws preprocessed after that change;
    # when absent the AR1 mask is simply all-False.
    all_onsets = stim_trials["onset"].to_numpy(dtype=float) if len(stim_trials) else np.empty(0)
    per_trial_bad = compute_bad_trial_mask(
        onsets=all_onsets,
        tmin=tmin,
        tmax=tmax,
        annotations=annotations,
        bad_prefix="BAD_AR2",
    )
    per_trial_bad_ar1 = compute_bad_trial_mask(
        onsets=all_onsets,
        tmin=tmin,
        tmax=tmax,
        annotations=annotations,
        bad_prefix="BAD_AR1",
    )
    stim_trials["bad_ar2"] = per_trial_bad
    stim_trials["bad_ar1"] = per_trial_bad_ar1

    segmented_array = []
    metadata_rows = []

    for i, trial in stim_trials.iterrows():
        # Skip trials that don't have enough history for the window
        if i + 1 < n_events_window:
            continue

        # Get event onset in samples (anchor = current trial)
        onset_sample = int(trial["onset"] * sfreq)

        # Boundary checks (window extends backwards by win_tmin_samples)
        if onset_sample + tmax_samples >= data.shape[1]:
            logger.debug(f"Skipping trial {i}: extends beyond data")
            continue
        if onset_sample + win_tmin_samples < 0:
            logger.debug(f"Skipping trial {i}: window starts before data")
            continue

        # Extract segment
        segment = data[:, onset_sample + win_tmin_samples : onset_sample + tmax_samples]
        segmented_array.append(segment)

        row = dict(trial)  # anchor trial's events row
        if n_events_window > 1:
            first_idx = i - n_events_window + 1
            included = stim_trials.iloc[first_idx : i + 1]
            inc_vtc = (
                included["VTC_filtered"].to_numpy(dtype=float)
                if "VTC_filtered" in included.columns
                else np.full(n_events_window, np.nan)
            )
            inc_task = (
                included["task"].to_numpy()
                if "task" in included.columns
                else np.full(n_events_window, "", dtype=object)
            )
            inc_bad = included["bad_ar2"].to_numpy(dtype=bool)
            inc_bad_ar1 = included["bad_ar1"].to_numpy(dtype=bool)
            row["included_VTC"] = inc_vtc
            row["included_task"] = inc_task
            row["included_bad_ar2"] = inc_bad
            row["included_bad_ar1"] = inc_bad_ar1
            row["window_vtc_mean"] = (
                float(np.nanmean(inc_vtc)) if not np.all(np.isnan(inc_vtc)) else np.nan
            )
            row["window_any_bad"] = bool(inc_bad.any())
            row["window_any_bad_ar1"] = bool(inc_bad_ar1.any())
            row["window_n_cc"] = int(np.sum(inc_task == "correct_commission"))
            row["window_n_co"] = int(np.sum(inc_task == "correct_omission"))
            row["window_n_ce"] = int(np.sum(inc_task == "commission_error"))
            row["window_n_oe"] = int(np.sum(inc_task == "omission_error"))
            # AR flags → window-level (ANY of N trials bad) so downstream
            # drop_bad_trials still works on a per-window basis.
            row["bad_ar2"] = row["window_any_bad"]
            row["bad_ar1"] = row["window_any_bad_ar1"]

        metadata_rows.append(row)

    segmented_array = np.array(segmented_array)
    trial_metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)

    # Single-trial mode: bad_ar1/bad_ar2 columns already populated for
    # surviving trials (threaded through stim_trials into each row dict).
    if "bad_ar2" not in trial_metadata.columns:
        trial_metadata["bad_ar2"] = False
    if "bad_ar1" not in trial_metadata.columns:
        trial_metadata["bad_ar1"] = False

    n_bad = int(trial_metadata["bad_ar2"].sum()) if len(trial_metadata) else 0
    logger.info(
        f"Segmented {len(segmented_array)} epoch(s) from continuous data "
        f"(n_events_window={n_events_window}, {n_bad} flagged bad_ar2)"
    )
    logger.debug(f"Segmented array shape: {segmented_array.shape}")

    return segmented_array, trial_metadata


# Trial-type filter modes mirror cc_saflow/saflow/data.py:326-357 (`select_epoch`).
# Each mode operates on an "included_task" array (length n_events_window, or
# length 1 for single-trial mode). Returns whether the epoch is retained.
TRIAL_TYPE_MODES = (
    "alltrials",
    "correct",
    "rare",
    "lapse",
    "correct_commission",
)
CORRECT_TASKS = ("correct_commission", "correct_omission")
RARE_TASKS = ("correct_omission", "commission_error")


def select_window(included_task: np.ndarray, type_how: str = "alltrials") -> bool:
    """Decide whether an epoch survives the trial-type filter.

    Mirrors ``select_epoch`` in cc_saflow/saflow/data.py:326-357.

    Modes:
        - ``alltrials``: keep all epochs (no filter)
        - ``correct``: all constituent trials are correct_commission OR
          correct_omission (i.e., no errors in the window)
        - ``rare``: at least one constituent trial is correct_omission OR
          commission_error (i.e., window contains a rare-stim outcome)
        - ``lapse``: at least one constituent trial is commission_error
        - ``correct_commission``: all constituent trials are correct_commission
          (the strictest mode; matches saflow's previous default)

    Parameters
    ----------
    included_task : np.ndarray
        Length-N array of task labels for the constituent trials. Works for
        N=1 (single-trial mode) and N=8 (windowed mode).
    type_how : str
        One of ``TRIAL_TYPE_MODES``.

    Returns
    -------
    bool
        True if the epoch is retained.
    """
    if type_how == "alltrials":
        return True
    inc = np.asarray(included_task)
    if type_how == "correct":
        return bool(np.all(np.isin(inc, CORRECT_TASKS)))
    if type_how == "rare":
        return bool(np.any(np.isin(inc, RARE_TASKS)))
    if type_how == "lapse":
        return bool(np.any(inc == "commission_error"))
    if type_how == "correct_commission":
        return bool(np.all(inc == "correct_commission"))
    raise ValueError(
        f"Unknown trial-type filter '{type_how}'. "
        f"Expected one of {TRIAL_TYPE_MODES}."
    )


def select_window_mask(
    included_task_per_epoch,
    task_per_epoch=None,
    type_how: str = "alltrials",
) -> np.ndarray:
    """Vectorised ``select_window``: returns a boolean keep-mask per epoch.

    Designed to be called from feature loaders that hold metadata either as
    a per-epoch ``included_task`` (windowed mode, length-N array per epoch)
    or as a per-epoch scalar ``task`` (single-trial mode). When
    ``included_task_per_epoch`` is None or empty, falls back to wrapping
    ``task_per_epoch`` as a length-1 included_task.

    Parameters
    ----------
    included_task_per_epoch : sequence of np.ndarray, or None
        For each epoch, the array of constituent trial task labels.
        ``len(seq) == n_epochs``; ``len(seq[i]) == n_events_window``.
    task_per_epoch : np.ndarray, optional
        Fallback per-epoch scalar task labels (single-trial mode).
    type_how : str
        See ``select_window``.

    Returns
    -------
    np.ndarray
        Boolean mask of length ``n_epochs``.
    """
    if type_how == "alltrials":
        if included_task_per_epoch is not None:
            n = len(included_task_per_epoch)
        elif task_per_epoch is not None:
            n = len(task_per_epoch)
        else:
            return np.zeros(0, dtype=bool)
        return np.ones(n, dtype=bool)

    if included_task_per_epoch is None or len(included_task_per_epoch) == 0:
        if task_per_epoch is None:
            return np.zeros(0, dtype=bool)
        # Wrap scalar task labels as length-1 included_task per epoch
        included_task_per_epoch = [np.array([t]) for t in task_per_epoch]

    return np.array(
        [select_window(inc, type_how) for inc in included_task_per_epoch],
        dtype=bool,
    )


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
