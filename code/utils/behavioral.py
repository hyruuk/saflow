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
    inout_bounds: Optional[List[int]] = None,
    filt_cutoff: float = 0.05,
    filt_type: str = "gaussian",
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
    applies smoothing filter, and classifies trials into IN/OUT zones.

    Args:
        subject: Subject ID (e.g., "04").
        run: Run number (e.g., "02").
        files_list: List of logfile names.
        logs_dir: Directory containing logfiles.
        cpt_blocs: List of CPT block numbers. Defaults to ['2','3','4','5','6','7'].
        inout_bounds: [lower, upper] percentile bounds for IN/OUT split.
            Defaults to [25, 75].
        filt_cutoff: Cutoff frequency for Butterworth filter. Defaults to 0.05.
        filt_type: Filter type ('gaussian' or 'butterworth'). Defaults to 'gaussian'.

    Returns:
        Tuple containing:
        - IN_idx: Indices of IN zone trials
        - OUT_idx: Indices of OUT zone trials
        - VTC_raw: Raw VTC array
        - VTC_filtered: Smoothed VTC array
        - IN_mask: Masked array for IN zone
        - OUT_mask: Masked array for OUT zone
        - performance_dict: Dictionary with trial type indices
        - df_response_out: Response DataFrame for the run
        - RT_to_VTC: RT array used for VTC computation

    Examples:
        >>> IN_idx, OUT_idx, VTC_raw, VTC_filt, _, _, perf, df, RT = get_VTC_from_file(
        ...     "04", "02", logfiles, logs_dir
        ... )
        >>> print(f"IN: {len(IN_idx)}, OUT: {len(OUT_idx)}")
    """
    logs_dir = Path(logs_dir)

    if cpt_blocs is None:
        cpt_blocs = ["2", "3", "4", "5", "6", "7"]
    if inout_bounds is None:
        inout_bounds = [25, 75]

    # Find logfiles belonging to subject
    subject_logfiles = []
    for logfile in sorted(files_list):
        parts = logfile.split("_")
        if len(parts) >= 4 and parts[2] == subject and parts[3] != "0":
            subject_logfiles.append(logs_dir / logfile)

    if not subject_logfiles:
        logger.error(f"No logfiles found for subject {subject}")
        raise FileNotFoundError(f"No logfiles found for subject {subject}")

    # Load and clean RT arrays
    RT_arrays = []
    RT_to_VTC = None
    performance_dict = None
    df_response_out = None

    for idx_file, logfile in enumerate(subject_logfiles):
        try:
            data = loadmat(logfile)
            df_response = pd.DataFrame(data["response"])

            # Replace commission errors by 0
            df_clean, perf_dict = clean_comerr(df_response)
            RT_raw = np.asarray(df_clean.loc[:, 4])
            RT_raw = np.array([x if x != 0 else np.nan for x in RT_raw])
            RT_arrays.append(RT_raw)

            if int(cpt_blocs[idx_file]) == int(run):
                RT_to_VTC = RT_raw
                performance_dict = perf_dict.copy()
                df_response_out = df_response

        except Exception as e:
            logger.error(f"Failed to load logfile {logfile}: {e}")
            continue

    if RT_to_VTC is None:
        logger.error(f"Run {run} not found in logfiles for subject {subject}")
        raise ValueError(f"Run {run} not found")

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

    # Apply filter
    if filt_type == "gaussian":
        # scipy >= 1.10: gaussian moved to signal.windows
        try:
            filt = signal.windows.gaussian(len(VTC_interpolated), fwhm2sigma(9))
        except AttributeError:
            # scipy < 1.10: use signal.gaussian
            filt = signal.gaussian(len(VTC_interpolated), fwhm2sigma(9))
        VTC_filtered = np.convolve(VTC_interpolated, filt, "same") / sum(filt)
    elif filt_type == "butterworth":
        b, a = signal.butter(3, filt_cutoff)
        VTC_filtered = signal.filtfilt(b, a, VTC_interpolated)
    else:
        logger.warning(f"Unknown filter type: {filt_type}, using gaussian")
        filt = signal.gaussian(len(VTC_interpolated), fwhm2sigma(9))
        VTC_filtered = np.convolve(VTC_interpolated, filt, "same") / sum(filt)

    # Create IN/OUT masks
    IN_threshold = np.quantile(VTC_filtered, inout_bounds[0] / 100)
    OUT_threshold = np.quantile(VTC_filtered, inout_bounds[1] / 100)

    IN_mask = np.ma.masked_where(VTC_filtered >= IN_threshold, VTC_filtered)
    OUT_mask = np.ma.masked_where(VTC_filtered < OUT_threshold, VTC_filtered)

    IN_idx = np.where(IN_mask.mask == False)[0]
    OUT_idx = np.where(OUT_mask.mask == False)[0]

    logger.info(
        f"Subject {subject}, run {run}: {len(IN_idx)} IN trials, "
        f"{len(OUT_idx)} OUT trials (bounds: {inout_bounds})"
    )

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
