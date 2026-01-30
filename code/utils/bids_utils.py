"""Core BIDS utilities and I/O helpers for saflow.

This module provides utilities for:
- BIDS path construction and naming
- MEG/EEG channel information extraction
- Source-level data segmentation
- Basic array utilities

All functions use configuration-based paths and logging.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import mne_bids
import numpy as np
import pandas as pd
import pickle
from mne_bids import BIDSPath

logger = logging.getLogger(__name__)


def create_fnames(
    subject: str,
    run: str,
    bids_root: Union[str, Path],
    task: str = "gradCPT",
) -> Dict[str, BIDSPath]:
    """Create BIDS file paths for different processing stages.

    Constructs BIDSPath objects for raw, preprocessed, morphed sources, PSD,
    LZC, and Welch power spectral density data.

    Args:
        subject: Subject ID (e.g., "04", "05").
        run: Run number (e.g., "02", "03").
        bids_root: Root directory of BIDS dataset.
        task: Task name. Defaults to "gradCPT".

    Returns:
        Dictionary containing BIDSPath objects for different processing stages:
        - 'raw': Raw MEG data path
        - 'preproc': Preprocessed data path
        - 'morph': Morphed source reconstruction path
        - 'psd': Power spectral density path
        - 'lzc': Lempel-Ziv complexity path
        - 'welch': Welch PSD path

    Examples:
        >>> fnames = create_fnames("04", "02", "/data/bids")
        >>> raw_path = fnames['raw']
        >>> preproc_path = fnames['preproc']
    """
    bids_root = Path(bids_root)

    # Raw data path
    raw_bidspath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        suffix="meg",
        root=str(bids_root),
    )

    # Preprocessed data path
    preproc_bidspath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        suffix="meg",
        processing="clean",
        root=str(bids_root / "derivatives" / "preprocessed"),
    )

    # Morphed source reconstruction path
    morph_bidspath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        processing="clean",
        description="morphed",
        root=str(bids_root / "derivatives" / "morphed_sources"),
    )

    # PSD data path
    psd_bidspath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        description="idx",
        root=str(bids_root / "derivatives" / "psd"),
    )
    psd_bidspath.mkdir(exist_ok=True)

    # Welch PSD path
    welch_bidspath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        suffix="meg",
        root=str(bids_root / "derivatives" / "welch"),
    )
    welch_bidspath.mkdir(exist_ok=True)

    # Lempel-Ziv complexity path
    lzc_bidspath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        description="idx",
        root=str(bids_root / "derivatives" / "lzc"),
    )
    lzc_bidspath.mkdir(exist_ok=True)

    return {
        "raw": raw_bidspath,
        "preproc": preproc_bidspath,
        "morph": morph_bidspath,
        "psd": psd_bidspath,
        "lzc": lzc_bidspath,
        "welch": welch_bidspath,
    }


def get_meg_picks_and_info(
    subject: str, run: str, bids_root: Union[str, Path], task: str = "gradCPT"
) -> Tuple[np.ndarray, mne.Info]:
    """Get MEG channel picks and info object from raw data.

    Args:
        subject: Subject ID (e.g., "04", "05").
        run: Run number (e.g., "02", "03").
        bids_root: Root directory of BIDS dataset.
        task: Task name. Defaults to "gradCPT".

    Returns:
        Tuple containing:
        - picks: Array of MEG channel indices
        - info: MNE Info object with channel information

    Raises:
        FileNotFoundError: If the BIDS file does not exist.

    Examples:
        >>> picks, info = get_meg_picks_and_info("04", "02", "/data/bids")
        >>> print(f"Found {len(picks)} MEG channels")
    """
    bids_root = Path(bids_root)

    rawpath = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        datatype="meg",
        suffix="meg",
        root=str(bids_root),
    )

    try:
        raw = mne_bids.read_raw_bids(rawpath, verbose=False)
    except FileNotFoundError as e:
        logger.error(f"Failed to read BIDS file for sub-{subject} run-{run}: {e}")
        raise

    picks = mne.pick_types(
        raw.info, meg=True, ref_meg=False, eeg=False, eog=False
    )

    raw_mag = raw.copy().pick_types(
        meg=True, ref_meg=False, eeg=False, eog=False
    )

    logger.debug(
        f"Extracted {len(picks)} MEG channels for sub-{subject} run-{run}"
    )

    return picks, raw_mag.info


def segment_sourcelevel(
    data_array: np.ndarray,
    filepaths: Dict[str, BIDSPath],
    sfreq: float = 600.0,
    tmin: float = 0.426,
    tmax: float = 1.278,
    n_events_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Segment source-level data array based on events from preprocessed MEG.

    This function takes a continuous source-level data array and segments it
    into epochs based on stimulus events. It also loads autoreject logs to
    identify bad epochs and extracts behavioral metadata for each segment.

    Args:
        data_array: Source-level data array to segment, shape (n_channels, n_samples).
        filepaths: Dictionary of BIDSPath objects (must contain 'preproc' and 'raw').
        sfreq: Sampling frequency in Hz. Defaults to 600.0.
        tmin: Start time of segment relative to event onset, in seconds. Defaults to 0.426.
        tmax: End time of segment relative to event onset, in seconds. Defaults to 1.278.
        n_events_window: Number of events to include in each window. Defaults to 1.

    Returns:
        Tuple containing:
        - segmented_array: Segmented data, shape (n_events, n_channels, n_samples)
        - events_idx: Indices of events that were segmented
        - events_dict: List of dictionaries with event metadata (VTC, RT, task, INOUT, etc.)

    Raises:
        KeyError: If required keys ('preproc', 'raw') are missing from filepaths.
        FileNotFoundError: If AutoReject log file is not found.

    Examples:
        >>> fnames = create_fnames("04", "02", "/data/bids")
        >>> data = np.random.rand(270, 360000)  # Example source data
        >>> segments, idx, metadata = segment_sourcelevel(data, fnames)
        >>> print(f"Created {len(segments)} segments")
    """
    # Load preprocessed data and events
    try:
        preproc = mne_bids.read_raw_bids(bids_path=filepaths["preproc"], verbose=False)
    except KeyError:
        logger.error("'preproc' key not found in filepaths dictionary")
        raise
    except FileNotFoundError as e:
        logger.error(f"Preprocessed file not found: {e}")
        raise

    events, event_id = mne.events_from_annotations(preproc, verbose=False)

    # Load full events file
    try:
        events_full_path = str(filepaths["raw"].fpath).replace("_meg.ds", "_events.tsv")
        events_full = pd.read_csv(events_full_path, sep="\t")
    except FileNotFoundError as e:
        logger.error(f"Events TSV file not found: {events_full_path}")
        raise

    # Compute time samples
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)
    epoch_length = tmax_samples - tmin_samples
    tmin_samples = tmin_samples - int(epoch_length * (n_events_window - 1))

    # Load AutoReject log
    arlog_fname = (
        str(
            filepaths["preproc"]
            .copy()
            .update(description="ARlog", processing=None)
            .fpath
        )
        + ".pkl"
    )

    try:
        with open(arlog_fname, "rb") as f:
            ARlog = pickle.load(f)
        bad_epochs = ARlog.bad_epochs
    except FileNotFoundError:
        logger.warning(f"AutoReject log not found: {arlog_fname}. Assuming no bad epochs.")
        bad_epochs = np.zeros(len(events), dtype=bool)

    # Segment array
    segmented_array = []
    events_idx = []
    events_dict = []
    stim_events_list = []

    for idx, event in enumerate(events):
        # Only process stimulus events (1 = rare, 2 = frequent)
        if event[2] in [1, 2]:
            stim_events_list.append(idx)

            # Check if segment is within bounds
            if event[0] + tmax_samples < data_array.shape[1]:
                if event[0] + tmin_samples > 0:
                    if len(stim_events_list) >= n_events_window:
                        # Extract segment
                        segmented_array.append(
                            data_array[
                                :, event[0] + tmin_samples : event[0] + tmax_samples
                            ]
                        )
                        events_idx.append(idx)

                        # Fill event metadata dictionary
                        included_events = stim_events_list[-n_events_window:]
                        event_dict = {
                            "event_idx": idx,
                            "t0_sample": event[0],
                            "VTC": events_full.loc[idx, "VTC"],
                            "task": events_full.loc[idx, "task"],
                            "RT": events_full.loc[idx, "RT"],
                            "INOUT": events_full.loc[idx, "INOUT_50_50"],
                            "INOUT_2575": events_full.loc[idx, "INOUT_25_75"],
                            "INOUT_1090": events_full.loc[idx, "INOUT_10_90"],
                            "bad_epoch": bad_epochs[idx],
                            "included_bad_epochs": bad_epochs[included_events],
                            "included_events_idx": included_events,
                            "included_VTC": events_full.loc[included_events, "VTC"].values,
                            "included_task": events_full.loc[included_events, "task"].values,
                            "included_RT": events_full.loc[included_events, "RT"].values,
                            "included_INOUT": events_full.loc[
                                included_events, "INOUT_50_50"
                            ].values,
                            "included_INOUT_2575": events_full.loc[
                                included_events, "INOUT_25_75"
                            ].values,
                            "included_INOUT_1090": events_full.loc[
                                included_events, "INOUT_10_90"
                            ].values,
                        }
                        events_dict.append(event_dict)

    segmented_array = np.array(segmented_array)
    events_idx = np.array(events_idx)

    logger.info(
        f"Segmented {len(segmented_array)} events from {data_array.shape[1]} samples"
    )

    return segmented_array, events_idx, events_dict


def create_pval_mask(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Create boolean mask for significant p-values.

    Args:
        pvals: Array of p-values.
        alpha: Significance threshold. Defaults to 0.05.

    Returns:
        Boolean mask with True for significant values (pval <= alpha).

    Examples:
        >>> pvals = np.array([0.01, 0.1, 0.03, 0.5])
        >>> mask = create_pval_mask(pvals, alpha=0.05)
        >>> print(mask)  # [True, False, True, False]
    """
    mask = np.zeros(len(pvals), dtype=bool)
    for i, pval in enumerate(pvals):
        if pval <= alpha:
            mask[i] = True

    return mask
