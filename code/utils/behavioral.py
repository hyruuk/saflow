"""Behavioral analysis utilities for saflow.

This module provides utilities for:
- VTC (Variability of Reaction Times) computation
- Behavioral data loading and cleaning
- Performance metrics calculation
- Signal detection theory (SDT) analysis

All functions use logging and avoid hardcoded paths.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from scipy.stats import norm

logger = logging.getLogger(__name__)


def interpolate_RT(RT_raw: np.ndarray) -> np.ndarray:
    """Interpolate missing reaction times from nearest RTs.

    Replaces zero values (missing RTs) with the average of the two nearest
    non-zero reaction times.

    Args:
        RT_raw: Raw reaction times as floats, with 0 for missing RT.

    Returns:
        Array with zeros replaced by interpolated values.

    Examples:
        >>> RTs = np.array([0.5, 0.0, 0.6, 0.0, 0.0, 0.7])
        >>> RTs_interp = interpolate_RT(RTs)
    """
    RT_array = RT_raw.copy()

    for idx, val in enumerate(RT_array):
        if val == 0:
            idx_next_val = 1
            try:
                # Find next non-zero value
                while RT_array[idx + idx_next_val] == 0:
                    idx_next_val += 1

                if idx == 0:
                    # If first value is zero, use next non-zero
                    RT_array[idx] = RT_array[idx + idx_next_val]
                else:
                    # Use average of nearest non-zero values
                    RT_array[idx] = (
                        RT_array[idx - 1] + RT_array[idx + idx_next_val]
                    ) / 2.0

            except IndexError:
                # If end of file reached, use last non-zero
                RT_array[idx] = RT_array[idx - 1]

    return RT_array


def compute_VTC(
    RT_array: np.ndarray,
    subj_mean: float,
    subj_std: float,
) -> np.ndarray:
    """Compute raw (unfiltered) VTC from reaction times.

    VTC (Variability of reaction Times) is computed as the absolute z-score
    of reaction times relative to subject-level statistics.

    Args:
        RT_array: Array of reaction times after interpolation.
        subj_mean: Mean reaction time of subject across all runs.
        subj_std: Standard deviation of reaction times across all runs.

    Returns:
        Array containing VTC values (absolute z-scores).

    Examples:
        >>> VTC = compute_VTC(RTs, mean_RT, std_RT)
        >>> print(f"Mean VTC: {np.mean(VTC):.2f}")
    """
    return np.abs((RT_array - subj_mean) / subj_std)


def fwhm2sigma(fwhm: float) -> float:
    """Convert full width at half maximum to Gaussian sigma.

    Args:
        fwhm: Full width at half maximum.

    Returns:
        Corresponding Gaussian standard deviation (sigma).

    Examples:
        >>> sigma = fwhm2sigma(9.0)
    """
    return fwhm / np.sqrt(8 * np.log(2))


def clean_comerr(
    df_response: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Clean commission errors from behavioral data.

    Identifies correct and incorrect responses for rare and frequent stimuli.

    Args:
        df_response: DataFrame with behavioral responses.
            Expected columns: [0]=stimulus type, [1]=response, [4]=RT.

    Returns:
        Tuple containing:
        - cleaned_df: Cleaned DataFrame
        - performance_dict: Dictionary with indices for each response type:
          'commission_error', 'correct_omission', 'omission_error', 'correct_commission'

    Examples:
        >>> cleaned_df, perf_dict = clean_comerr(responses)
        >>> print(f"Commission errors: {len(perf_dict['commission_error'])}")
    """
    cleaned_df = df_response.copy()
    correct_omission_idx = []
    commission_error_idx = []
    correct_commission_idx = []
    omission_error_idx = []

    for idx_line, line in enumerate(cleaned_df.iterrows()):
        # Rare stim (1.0) with response = commission error
        if line[1][0] == 1.0 and line[1][1] != 0.0:
            cleaned_df.loc[idx_line, 4] = 0.0  # Set RT to 0
            commission_error_idx.append(idx_line)
        # Rare stim (1.0) without response = correct omission
        elif line[1][0] == 1.0 and line[1][1] == 0.0:
            correct_omission_idx.append(idx_line)
        # Freq stim (2.0) with response = correct commission
        elif line[1][0] == 2.0 and line[1][1] != 0.0:
            correct_commission_idx.append(idx_line)
        # Freq stim (2.0) without response = omission error
        elif line[1][0] == 2.0 and line[1][1] == 0.0:
            omission_error_idx.append(idx_line)

    performance_dict = {
        "commission_error": commission_error_idx,
        "correct_omission": correct_omission_idx,
        "omission_error": omission_error_idx,
        "correct_commission": correct_commission_idx,
    }

    logger.debug(
        f"Performance: {len(commission_error_idx)} commission errors, "
        f"{len(correct_omission_idx)} correct omissions, "
        f"{len(omission_error_idx)} omission errors, "
        f"{len(correct_commission_idx)} correct commissions"
    )

    return cleaned_df, performance_dict


def get_VTC_from_file(
    subject: str,
    run: str,
    files_list: List[str],
    logs_dir: Union[str, Path],
    cpt_blocs: Optional[List[str]] = None,
    filt_type: str = "gaussian",
    filt_config: Optional[Dict] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ma.MaskedArray,
    np.ma.MaskedArray,
    Dict[str, List[int]],
    pd.DataFrame,
    np.ndarray,
]:
    """Compute VTC from behavioral logfiles for a specific run.

    Loads logfiles for a subject, computes VTC (Variability of Reaction Times),
    and applies smoothing filter based on config parameters.

    Note: In the new config-driven architecture, zone classifications (IN/OUT/MID)
    are NOT computed here. They are computed on-demand during feature extraction
    using classify_trials_from_vtc() with bounds from config['analysis']['inout_bounds'].

    Run Matching:
        Behavioral files are matched to MEG runs by parsing the run number from
        the filename. MEG run N corresponds to behavioral run N-1:
        - MEG run 02 -> behavioral run 1
        - MEG run 03 -> behavioral run 2
        - etc.

        Only valid behavioral runs (1-6) are used. Run 0 (practice) and files with
        unexpected run numbers (typos like "47") are ignored with a warning.

    Args:
        subject: Subject ID (e.g., "04").
        run: Run number (e.g., "02").
        files_list: List of logfile names.
        logs_dir: Directory containing logfiles.
        cpt_blocs: DEPRECATED - no longer used. Run matching is now done by parsing
            the run number from the behavioral filename.
        filt_type: Filter type ('gaussian' or 'butterworth'). Defaults to 'gaussian'.
        filt_config: Dictionary with filter parameters from config['behavioral']['vtc']['filter'].
            For gaussian: {'gaussian_fwhm': 9}
            For butterworth: {'butterworth_order': 3, 'butterworth_cutoff': 0.05}

    Returns:
        Tuple containing:
        - IN_idx: Empty array (deprecated, for backward compatibility)
        - OUT_idx: Empty array (deprecated, for backward compatibility)
        - VTC_raw: Raw VTC array
        - VTC_filtered: Smoothed VTC array
        - IN_mask: Empty masked array (deprecated, for backward compatibility)
        - OUT_mask: Empty masked array (deprecated, for backward compatibility)
        - performance_dict: Dictionary with trial type indices
        - df_response_out: Response DataFrame for the run
        - RT_to_VTC: RT array used for VTC computation

    Examples:
        >>> filter_config = {'type': 'gaussian', 'gaussian_fwhm': 9}
        >>> _, _, VTC_raw, VTC_filt, _, _, perf, df, RT = get_VTC_from_file(
        ...     "04", "02", logfiles, logs_dir, filt_config=filter_config
        ... )
    """
    logs_dir = Path(logs_dir)

    # cpt_blocs is deprecated - run matching is now done by filename parsing
    if cpt_blocs is not None:
        logger.debug("cpt_blocs parameter is deprecated and ignored")

    if filt_config is None:
        # Default filter config (Gaussian with FWHM=9)
        filt_config = {"gaussian_fwhm": 9}

    # Find logfiles belonging to subject and map by behavioral run number
    # Behavioral runs 1-6 correspond to MEG runs 02-07 (cpt_blocs 2-7)
    valid_behav_runs = {"1", "2", "3", "4", "5", "6"}
    subject_logfiles = {}  # behav_run -> filepath

    for logfile in sorted(files_list):
        parts = logfile.split("_")
        if len(parts) >= 4 and parts[2] == subject:
            behav_run = parts[3]
            # Only include valid behavioral runs (1-6), skip run 0 and typos
            if behav_run in valid_behav_runs:
                if behav_run not in subject_logfiles:
                    subject_logfiles[behav_run] = logs_dir / logfile
                else:
                    logger.warning(
                        f"Duplicate behavioral file for sub-{subject} run {behav_run}, "
                        f"using first: {subject_logfiles[behav_run].name}"
                    )
            elif behav_run != "0":
                logger.warning(
                    f"Ignoring behavioral file with unexpected run number: {logfile}"
                )

    if not subject_logfiles:
        logger.error(f"No valid logfiles found for subject {subject}")
        raise FileNotFoundError(f"No valid logfiles found for subject {subject}")

    # Load and clean RT arrays from all valid runs
    RT_arrays = []
    RT_to_VTC = None
    performance_dict = None
    df_response_out = None

    # MEG run (from cpt_blocs) to behavioral run mapping: MEG run N -> behav run N-1
    # e.g., MEG run 02 (cpt_bloc "2") -> behavioral run "1"
    target_behav_run = str(int(run) - 1) if run.isdigit() else None

    for behav_run in sorted(subject_logfiles.keys()):
        logfile = subject_logfiles[behav_run]
        try:
            data = loadmat(logfile)
            df_response = pd.DataFrame(data["response"])

            # Replace commission errors by 0
            df_clean, perf_dict = clean_comerr(df_response)
            RT_raw = np.asarray(df_clean.loc[:, 4])
            RT_raw = np.array([x if x != 0 else np.nan for x in RT_raw])
            RT_arrays.append(RT_raw)

            # Match by actual run number, not index
            if behav_run == target_behav_run:
                RT_to_VTC = RT_raw
                performance_dict = perf_dict.copy()
                df_response_out = df_response
                logger.debug(f"Matched behavioral run {behav_run} to MEG run {run}")

        except Exception as e:
            logger.error(f"Failed to load logfile {logfile}: {e}")
            continue

    if RT_to_VTC is None:
        available_runs = sorted(subject_logfiles.keys())
        logger.error(
            f"Behavioral run {target_behav_run} (for MEG run {run}) not found for "
            f"subject {subject}. Available behavioral runs: {available_runs}"
        )
        raise ValueError(
            f"Behavioral run {target_behav_run} not found for subject {subject}. "
            f"MEG run {run} requires behavioral run {target_behav_run}."
        )

    # Compute mean and std across runs
    allruns_RT_array = np.concatenate(RT_arrays)
    subj_mean = np.nanmean(allruns_RT_array)
    subj_std = np.nanstd(allruns_RT_array)

    logger.debug(
        f"Subject {subject}: mean RT = {subj_mean:.3f}s, std = {subj_std:.3f}s"
    )

    # Compute VTC
    VTC_raw = compute_VTC(RT_to_VTC, subj_mean, subj_std)
    VTC_raw[np.isnan(VTC_raw)] = 0
    VTC_interpolated = interpolate_RT(VTC_raw)

    # Apply filter using config parameters
    if filt_type == "gaussian":
        fwhm = filt_config.get("gaussian_fwhm", 9)
        logger.debug(f"Applying Gaussian filter with FWHM={fwhm}")
        # scipy >= 1.10: gaussian moved to signal.windows
        try:
            filt = signal.windows.gaussian(len(VTC_interpolated), fwhm2sigma(fwhm))
        except AttributeError:
            # scipy < 1.10: use signal.gaussian
            filt = signal.gaussian(len(VTC_interpolated), fwhm2sigma(fwhm))
        VTC_filtered = np.convolve(VTC_interpolated, filt, "same") / sum(filt)
    elif filt_type == "butterworth":
        order = filt_config.get("butterworth_order", 3)
        cutoff = filt_config.get("butterworth_cutoff", 0.05)
        logger.debug(f"Applying Butterworth filter (order={order}, cutoff={cutoff})")
        b, a = signal.butter(order, cutoff)
        VTC_filtered = signal.filtfilt(b, a, VTC_interpolated)
    else:
        logger.warning(f"Unknown filter type: {filt_type}, using Gaussian with FWHM=9")
        filt = signal.gaussian(len(VTC_interpolated), fwhm2sigma(9))
        VTC_filtered = np.convolve(VTC_interpolated, filt, "same") / sum(filt)

    logger.info(
        f"Subject {subject}, run {run}: Computed VTC_raw and VTC_filtered "
        f"(filter: {filt_type})"
    )

    # Return empty arrays for deprecated IN/OUT outputs (for backward compatibility)
    # Zone classification is now done on-demand during feature extraction
    IN_idx = np.array([], dtype=int)
    OUT_idx = np.array([], dtype=int)
    IN_mask = np.ma.masked_array(np.array([]))
    OUT_mask = np.ma.masked_array(np.array([]))

    return (
        IN_idx,
        OUT_idx,
        VTC_raw,
        VTC_filtered,
        IN_mask,
        OUT_mask,
        performance_dict,
        df_response_out,
        RT_to_VTC,
    )


def SDT(
    hits: int,
    misses: int,
    fas: int,
    crs: int,
) -> Dict[str, float]:
    """Compute signal detection theory (SDT) measures.

    Calculates d-prime, beta, criterion c, and A' from hit and false alarm rates.

    Args:
        hits: Number of hits (correct detections).
        misses: Number of misses (missed detections).
        fas: Number of false alarms.
        crs: Number of correct rejections.

    Returns:
        Dictionary containing:
        - 'd': d-prime (sensitivity)
        - 'beta': Response bias (beta)
        - 'c': Criterion (c)
        - 'Ad': A-prime (non-parametric sensitivity)

    Examples:
        >>> sdt_measures = SDT(hits=80, misses=20, fas=10, crs=90)
        >>> print(f"d-prime: {sdt_measures['d']:.2f}")
    """
    Z = norm.ppf

    # Floors and ceilings are replaced by half hits and half FAs
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1:
        fa_rate = 1 - half_fa
    if fa_rate == 0:
        fa_rate = half_fa

    # Compute measures
    out = {}
    out["d"] = Z(hit_rate) - Z(fa_rate)
    out["beta"] = math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2)
    out["c"] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out["Ad"] = norm.cdf(out["d"] / math.sqrt(2))

    return out


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
        - 'MID_mask': Boolean array marking MID trials
        - 'zone_labels': Array of zone labels ('IN', 'OUT', 'MID') per trial
        - 'lower_thresh': Lower VTC threshold value
        - 'upper_thresh': Upper VTC threshold value

    Examples:
        >>> vtc = np.array([0.5, 1.2, 0.8, 2.1, 0.6, 1.8, 0.9, 2.5])
        >>> zones = classify_trials_from_vtc(vtc, inout_bounds=(25, 75))
        >>> print(f"IN trials: {len(zones['IN_idx'])}")
        IN trials: 2
        >>> print(f"OUT trials: {len(zones['OUT_idx'])}")
        OUT trials: 2
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
