"""BIDS conversion utilities for saflow.

This module provides helper functions for converting raw MEG data to BIDS format.
Functions handle:
- Filename parsing (subject, run, task extraction)
- MEG data loading and channel renaming
- Event detection from MEG data
- Behavioral data enrichment (VTC, RT, task performance, IN/OUT zones)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

logger = logging.getLogger(__name__)


def parse_info_from_name(fname: str) -> Tuple[str, str, str]:
    """Parse subject, run, and task from CTF filename.

    Extracts BIDS identifiers from CTF dataset filename following the naming
    convention used in the gradCPT study.

    Args:
        fname: CTF dataset filename (e.g., 'SA04_01.ds').

    Returns:
        Tuple containing:
        - subject: Subject ID (e.g., '04')
        - run: Run number (e.g., '01')
        - task: Task name ('rest' for runs 01/08, 'gradCPT' otherwise)

    Examples:
        >>> subject, run, task = parse_info_from_name('SA04_02.ds')
        >>> print(f"sub-{subject} task-{task} run-{run}")
        sub-04 task-gradCPT run-02
    """
    subject = fname.split("SA")[1][:2]
    run = fname.split("_")[-1][:2]

    # Runs 01 and 08 are resting state
    task = "rest" if run in ["01", "08"] else "gradCPT"

    logger.debug(f"Parsed {fname}: sub-{subject}, run-{run}, task-{task}")
    return subject, run, task


def load_meg_recording(
    ds_path: Path,
    bids_root: Path,
) -> Tuple[mne.io.Raw, BIDSPath, str]:
    """Load CTF MEG recording and create BIDS path.

    Loads raw MEG data, renames EOG/ECG channels to BIDS-compliant names,
    and creates corresponding BIDSPath object.

    Args:
        ds_path: Path to CTF .ds file.
        bids_root: BIDS dataset root directory.

    Returns:
        Tuple containing:
        - raw: MNE Raw object with renamed channels
        - bidspath: BIDSPath object for this recording
        - task: Task name ('gradCPT' or 'rest')

    Examples:
        >>> raw, bidspath, task = load_meg_recording(
        ...     Path('/data/raw/SA04_02.ds'),
        ...     Path('/data/bids')
        ... )
    """
    fname = ds_path.name
    subject, run, task = parse_info_from_name(fname)

    logger.info(f"Loading recording: sub-{subject}, run-{run}, task-{task}")

    # Create BIDS path
    bidspath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        root=str(bids_root),
    )

    # Load raw data
    raw = mne.io.read_raw_ctf(str(ds_path), verbose=False)
    raw.info["line_freq"] = 60  # North America power line frequency

    # Rename auxiliary channels to BIDS-compliant names
    mne.rename_channels(
        raw.info,
        {
            "EEG057": "vEOG",
            "EEG058": "hEOG",
            "EEG059": "ECG",
        },
    )

    raw.set_channel_types(
        {
            "ECG": "ecg",
            "hEOG": "eog",
            "vEOG": "eog",
        }
    )

    logger.debug(f"Loaded {raw.n_times} samples, {len(raw.ch_names)} channels")
    return raw, bidspath, task


def detect_events(raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int]]:
    """Detect events from MEG trigger channel.

    Finds stimulus and response events in the MEG data for the gradCPT task.

    Args:
        raw: MNE Raw object.

    Returns:
        Tuple containing:
        - events: MNE events array, shape (n_events, 3)
        - event_id: Dictionary mapping event names to codes

    Examples:
        >>> events, event_id = detect_events(raw)
        >>> print(f"Found {len(events)} events")
    """
    event_id = {
        "Freq": 21,        # Frequent stimulus
        "Rare": 31,        # Rare stimulus
        "Resp": 99,        # Button response
        "BlocStart": 10,   # Block start marker
    }

    try:
        events = mne.find_events(raw, verbose=False)
    except ValueError:
        # If events are too close, increase min_duration
        min_dur = 2 / raw.info["sfreq"]
        events = mne.find_events(raw, min_duration=min_dur, verbose=False)
        logger.debug(f"Used min_duration={min_dur:.4f}s for event detection")

    logger.info(f"Detected {len(events)} events")
    return events, event_id


def add_trial_indices(events_df: pd.DataFrame) -> pd.DataFrame:
    """Add trial index column to BIDS events dataframe.

    Trial indices are assigned only to stimulus events (Freq, Rare), not
    to response or block markers.

    Args:
        events_df: BIDS events dataframe.

    Returns:
        Events dataframe with 'trial_idx' column added.

    Examples:
        >>> events_df = add_trial_indices(events_df)
        >>> print(events_df[['trial_type', 'trial_idx']].head())
    """
    trial_idx = 0
    trial_indices = []

    for event in events_df.itertuples():
        if event.trial_type in ["Freq", "Rare"]:
            trial_indices.append(trial_idx)
            trial_idx += 1
        else:
            trial_indices.append(-1)  # Use -1 for non-stimulus events instead of NaN

    events_df["trial_idx"] = np.array(trial_indices).astype(int)
    logger.debug(f"Added trial indices: {trial_idx} trials")
    return events_df


def find_trial_type(type_dict: Dict[str, List[int]], trial_idx: int) -> Optional[str]:
    """Find trial type for given trial index.

    Args:
        type_dict: Dictionary mapping trial types to lists of indices.
        trial_idx: Trial index to look up.

    Returns:
        Trial type string, or None if not found.

    Examples:
        >>> perf_dict = {'commission_error': [0, 5], 'correct_omission': [1, 2]}
        >>> find_trial_type(perf_dict, 0)
        'commission_error'
    """
    for trial_type, idx_list in type_dict.items():
        if trial_idx in idx_list:
            return trial_type
    return None


def add_behavioral_info(
    events_df: pd.DataFrame,
    vtc_raw: np.ndarray,
    vtc_filtered: np.ndarray,
    rt_values: np.ndarray,
    performance_dict: Dict[str, List[int]],
) -> pd.DataFrame:
    """Add behavioral data columns to events dataframe.

    Enriches the events dataframe with VTC (raw and filtered), RT, and
    task performance information from behavioral logfiles.

    Args:
        events_df: BIDS events dataframe with trial_idx column.
        vtc_raw: Raw VTC values for each trial.
        vtc_filtered: Filtered VTC values for each trial.
        rt_values: Reaction times for each trial.
        performance_dict: Dictionary with trial type indices.

    Returns:
        Events dataframe with VTC_raw, VTC_filtered, RT, and task columns added.

    Examples:
        >>> events_df = add_behavioral_info(events_df, VTC_raw, VTC_filt, RT, perf_dict)
        >>> print(events_df[['trial_type', 'VTC_raw', 'VTC_filtered', 'RT', 'task']].head())
    """
    vtc_raw_list = []
    vtc_filtered_list = []
    rt_list = []
    task_list = []

    for event in events_df.itertuples():
        if event.trial_type in ["Freq", "Rare"]:
            trial_idx = int(event.trial_idx)
            vtc_raw_list.append(vtc_raw[trial_idx])
            vtc_filtered_list.append(vtc_filtered[trial_idx])
            rt_list.append(rt_values[trial_idx])
            task_list.append(find_trial_type(performance_dict, trial_idx))
        else:
            # Non-stimulus events (responses, block markers)
            vtc_raw_list.append("n/a")
            vtc_filtered_list.append("n/a")
            rt_list.append(0)
            task_list.append("n/a")

    events_df["VTC_raw"] = vtc_raw_list
    events_df["VTC_filtered"] = vtc_filtered_list
    events_df["RT"] = rt_list
    events_df["task"] = task_list

    logger.debug("Added behavioral info: VTC_raw, VTC_filtered, RT, task performance")
    return events_df


def add_inout_zones(
    events_df: pd.DataFrame,
    subject: str,
    run: str,
    files_list: List[str],
    logs_dir: Path,
) -> pd.DataFrame:
    """Add IN/OUT zone classifications to events dataframe.

    **DEPRECATED**: This function is no longer used in the config-driven architecture.
    Zone classifications are now computed on-demand during feature extraction using
    `classify_trials_from_vtc()` from `code/features/utils.py` with bounds from
    `config['analysis']['inout_bounds']`.

    This function is kept for backward compatibility only.

    Computes IN/OUT classifications for three different percentile bounds
    (50/50, 25/75, 10/90) based on VTC.

    Args:
        events_df: BIDS events dataframe with trial_idx column.
        subject: Subject ID.
        run: Run number.
        files_list: List of behavioral logfiles.
        logs_dir: Directory containing behavioral logfiles.

    Returns:
        Events dataframe with INOUT_50_50, INOUT_25_75, INOUT_10_90 columns.

    Examples:
        >>> events_df = add_inout_zones(events_df, '04', '02', logfiles, logs_dir)
        >>> print(events_df[['trial_idx', 'INOUT_25_75']].head())
    """
    logger.warning(
        "add_inout_zones() is deprecated. Use classify_trials_from_vtc() instead "
        "for on-demand zone classification during feature extraction."
    )
    from code.utils.behavioral import get_VTC_from_file

    # Compute IN/OUT for different percentile bounds
    bounds_list = [[50, 50], [25, 75], [10, 90]]

    for bounds in bounds_list:
        logger.debug(f"Computing IN/OUT zones for bounds {bounds}")

        (
            IN_idx,
            OUT_idx,
            _,  # VTC_raw (not needed here)
            _,  # VTC_filtered (not needed)
            _,  # IN_mask (not needed)
            _,  # OUT_mask (not needed)
            _,  # performance_dict (not needed)
            _,  # df_response (not needed)
            _,  # RT_to_VTC (not needed)
        ) = get_VTC_from_file(
            subject=subject,
            run=run,
            files_list=files_list,
            logs_dir=logs_dir,
            cpt_blocs=["2", "3", "4", "5", "6", "7"],
            inout_bounds=bounds,
            filt_cutoff=0.05,
            filt_type="gaussian",
        )

        inout_dict = {"IN": IN_idx, "OUT": OUT_idx}
        inout_list = []

        for event in events_df.itertuples():
            if event.trial_type in ["Freq", "Rare"]:
                zone = find_trial_type(inout_dict, event.trial_idx)
                inout_list.append(zone if zone else "MID")
            else:
                inout_list.append("n/a")

        column_name = f"INOUT_{bounds[0]}_{bounds[1]}"
        events_df[column_name] = inout_list

        n_in = sum(1 for x in inout_list if x == "IN")
        n_out = sum(1 for x in inout_list if x == "OUT")
        logger.info(f"{column_name}: {n_in} IN, {n_out} OUT")

    return events_df
