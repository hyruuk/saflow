"""Preprocessing utilities for saflow.

This module provides helper functions for the preprocessing pipeline:
- BIDS path creation for preprocessed data
- Noise covariance computation from empty-room recordings
- Filtering and epoching utilities
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

logger = logging.getLogger(__name__)


def create_preprocessing_paths(
    subject: str,
    run: str,
    bids_root: Path,
    derivatives_root: Path,
) -> Dict[str, BIDSPath]:
    """Create BIDSPath objects for preprocessing inputs and outputs.

    Args:
        subject: Subject ID.
        run: Run number.
        bids_root: BIDS dataset root directory.
        derivatives_root: Derivatives directory for processed data.

    Returns:
        Dictionary with BIDSPath objects:
        - 'raw': Input raw BIDS data
        - 'preproc': Output preprocessed continuous data
        - 'epoch_ica': Output epochs (ICA only, no AR)
        - 'epoch_ica_ar': Output epochs (ICA + AR)
        - 'ARlog_first': First AutoReject log (for ICA fitting)
        - 'ARlog_second': Second AutoReject log (post-ICA bad epoch detection)
        - 'report': Output HTML report

    Examples:
        >>> paths = create_preprocessing_paths('04', '02', bids_root, deriv_root)
        >>> raw = read_raw_bids(paths['raw'])
    """
    raw_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        extension=".fif",
        root=str(bids_root),
    )

    preproc_dir = derivatives_root / "preprocessed"
    preproc_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        processing="clean",
        root=str(preproc_dir),
    )
    preproc_bidspath.mkdir(exist_ok=True)

    ARlog_first_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        description="ARlog1",
        root=str(preproc_dir),
    )

    ARlog_second_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        description="ARlog2",
        root=str(preproc_dir),
    )

    report_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        description="report",
        root=str(preproc_dir),
    )

    ica_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        suffix="meg",
        description="ica",
        root=str(preproc_dir),
    )

    epoch_dir = derivatives_root / "epochs"

    # ICA only epochs
    epoch_ica_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="ica",
        root=str(epoch_dir),
    )
    epoch_ica_bidspath.mkdir(exist_ok=True)

    # ICA + AR epochs
    epoch_ica_ar_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="icaar",
        root=str(epoch_dir),
    )
    epoch_ica_ar_bidspath.mkdir(exist_ok=True)

    # AR2-interpolated epochs
    epoch_ar2_interp_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing="ar2interp",
        root=str(epoch_dir),
    )
    epoch_ar2_interp_bidspath.mkdir(exist_ok=True)

    return {
        "raw": raw_bidspath,
        "preproc": preproc_bidspath,
        "epoch_ica": epoch_ica_bidspath,
        "epoch_ica_ar": epoch_ica_ar_bidspath,
        "epoch_ar2_interp": epoch_ar2_interp_bidspath,
        "ARlog_first": ARlog_first_bidspath,
        "ARlog_second": ARlog_second_bidspath,
        "report": report_bidspath,
        "ica": ica_bidspath,
    }


def compute_or_load_noise_cov(
    subject: str,
    bids_root: Path,
    derivatives_root: Path,
) -> mne.Covariance:
    """Compute or load noise covariance from subject's empty-room recording.

    Computes noise covariance matrix from the empty-room recording in the
    subject's BIDS folder if not already cached, otherwise loads from file.

    The empty-room recording should be at:
        bids_root/sub-{subject}/meg/sub-{subject}_task-noise_meg.fif

    The noise covariance is saved at:
        derivatives_root/noise_covariance/sub-{subject}/meg/sub-{subject}_task-noise_cov.fif

    Args:
        subject: Subject ID (e.g., '04').
        bids_root: BIDS dataset root directory.
        derivatives_root: Derivatives directory for cached covariance.

    Returns:
        Noise covariance matrix.

    Examples:
        >>> noise_cov = compute_or_load_noise_cov('04', bids_root, deriv_root)
    """
    noise_cov_dir = derivatives_root / "noise_covariance"

    # Noise covariance file path (under subject directory)
    noise_cov_file = (
        noise_cov_dir
        / f"sub-{subject}"
        / "meg"
        / f"sub-{subject}_task-noise_cov.fif"
    )

    # Check if we already have the noise covariance computed
    if noise_cov_file.exists():
        logger.info(f"Loading cached noise covariance from {noise_cov_file}")
        noise_cov = mne.read_cov(str(noise_cov_file))
    else:
        logger.info(f"Computing noise covariance for sub-{subject}")

        # Empty-room recording in subject's BIDS folder
        er_fif_path = (
            bids_root
            / f"sub-{subject}"
            / "meg"
            / f"sub-{subject}_task-noise_meg.fif"
        )

        if not er_fif_path.exists():
            raise FileNotFoundError(
                f"Empty-room recording not found at: {er_fif_path}\n"
                f"Please ensure the empty-room recording is in the subject's BIDS folder.\n"
                f"Run BIDS conversion to copy the noise recording to the subject folder."
            )

        logger.info(f"Loading empty-room recording from: {er_fif_path}")
        er_raw = mne.io.read_raw_fif(str(er_fif_path), preload=True)

        logger.info("Computing noise covariance (shrunk + empirical methods)...")
        noise_cov = mne.compute_raw_covariance(
            er_raw, method=["shrunk", "empirical"], rank=None, verbose=False
        )

        # Save noise covariance
        noise_cov_file.parent.mkdir(parents=True, exist_ok=True)
        noise_cov.save(str(noise_cov_file), overwrite=True)
        logger.info(f"Saved noise covariance to {noise_cov_file}")

    return noise_cov


def apply_filtering(
    raw: mne.io.Raw,
    low_cutoff: float,
    high_cutoff: float,
    notch_freqs: list,
    picks: list,
) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """Apply bandpass and notch filtering to raw data.

    Creates two filtered copies:
    1. Main filtered data (low_cutoff to high_cutoff Hz)
    2. Additional 1 Hz highpass for ICA/AutoReject

    Args:
        raw: Raw MEG data.
        low_cutoff: Lower frequency cutoff (Hz).
        high_cutoff: Upper frequency cutoff (Hz).
        notch_freqs: Frequencies for notch filtering (Hz).
        picks: Channel picks for filtering.

    Returns:
        Tuple of (filtered_raw, filtered_raw_for_ica).

    Examples:
        >>> preproc, raw_filt = apply_filtering(raw, 0.1, 200, [60, 120], picks)
    """
    logger.info(
        f"Applying bandpass filter: {low_cutoff}-{high_cutoff} Hz, "
        f"notch at {notch_freqs} Hz"
    )

    # Main filtered data
    preproc = raw.copy().filter(
        low_cutoff, high_cutoff, picks=picks, fir_design="firwin"
    )

    preproc.notch_filter(
        notch_freqs,
        picks=picks,
        filter_length="auto",
        phase="zero",
        fir_design="firwin",
    )

    # Filtered copy for ICA/AR (1 Hz highpass)
    raw_filt = preproc.copy().filter(1, None)

    logger.debug("Filtering complete")
    return preproc, raw_filt


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    tmin: float,
    tmax: float,
    picks: list,
    reject: dict = None,
) -> mne.Epochs:
    """Create epochs from continuous data.

    Args:
        raw: Raw MEG data.
        events: MNE events array.
        event_id: Event ID dictionary.
        tmin: Start time before event (s).
        tmax: End time after event (s).
        picks: Channel picks.
        reject: Rejection criteria dictionary (optional).

    Returns:
        Epochs object.

    Examples:
        >>> epochs = create_epochs(raw, events, event_id, -0.2, 0.8, picks)
    """
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject=reject,
        picks=picks,
        preload=True,
    )

    logger.debug(f"Created {len(epochs)} epochs")
    return epochs


def detect_bad_epochs_threshold(
    epochs: mne.Epochs,
    reject_threshold: dict = None,
    flat_threshold: dict = None,
) -> Tuple[np.ndarray, dict]:
    """Detect bad epochs using amplitude thresholds.

    Simple threshold-based rejection for comparison with AutoReject.
    Uses peak-to-peak amplitude to identify epochs with extreme values.

    Args:
        epochs: Epochs object to check.
        reject_threshold: Max peak-to-peak amplitude (e.g., {'mag': 4000e-15}).
                          If None, uses default {'mag': 4000e-15}.
        flat_threshold: Min peak-to-peak amplitude (e.g., {'mag': 1e-15}).
                        If None, uses default {'mag': 1e-15}.

    Returns:
        Tuple of:
        - Boolean array marking bad epochs (True = bad)
        - Dictionary with detection statistics

    Examples:
        >>> bad_mask, stats = detect_bad_epochs_threshold(epochs)
        >>> print(f"Found {stats['n_bad']} bad epochs ({stats['pct_bad']:.1f}%)")
    """
    if reject_threshold is None:
        reject_threshold = {"mag": 5e-12}  # 5000 fT (5 pT) - reasonable for CTF MEG
    if flat_threshold is None:
        flat_threshold = {"mag": 1e-14}  # 10 fT

    logger.info(
        f"Running threshold-based epoch detection: "
        f"reject={reject_threshold}, flat={flat_threshold}"
    )

    n_epochs = len(epochs)
    bad_mask = np.zeros(n_epochs, dtype=bool)

    # Get data for magnetometers
    picks_mag = mne.pick_types(epochs.info, meg="mag")
    data = epochs.get_data(picks=picks_mag)  # (n_epochs, n_channels, n_times)

    # Compute peak-to-peak amplitude per epoch per channel
    ptp = np.ptp(data, axis=2)  # (n_epochs, n_channels)

    # Check reject threshold (too high amplitude)
    if "mag" in reject_threshold:
        reject_val = reject_threshold["mag"]
        bad_reject = np.any(ptp > reject_val, axis=1)
        logger.debug(f"Epochs exceeding reject threshold: {np.sum(bad_reject)}")
        bad_mask |= bad_reject

    # Check flat threshold (too low amplitude)
    if "mag" in flat_threshold:
        flat_val = flat_threshold["mag"]
        bad_flat = np.any(ptp < flat_val, axis=1)
        logger.debug(f"Epochs below flat threshold: {np.sum(bad_flat)}")
        bad_mask |= bad_flat

    n_bad = int(np.sum(bad_mask))
    pct_bad = 100 * n_bad / n_epochs if n_epochs > 0 else 0.0

    stats = {
        "n_total": n_epochs,
        "n_bad": n_bad,
        "n_good": n_epochs - n_bad,
        "pct_bad": pct_bad,
        "reject_threshold": reject_threshold,
        "flat_threshold": flat_threshold,
        "n_bad_reject": int(np.sum(bad_reject)) if "mag" in reject_threshold else 0,
        "n_bad_flat": int(np.sum(bad_flat)) if "mag" in flat_threshold else 0,
    }

    logger.info(
        f"Threshold detection: {n_bad}/{n_epochs} bad epochs "
        f"({pct_bad:.1f}%)"
    )

    return bad_mask, stats


def filter_events_by_type(
    events: np.ndarray,
    event_id: dict,
    include_types: List[str],
) -> Tuple[np.ndarray, dict]:
    """Filter events and event_id to only include specified trial types.

    Args:
        events: MNE events array (n_events, 3).
        event_id: Event ID dictionary mapping name -> int.
        include_types: List of event type names to keep (e.g. ["Freq", "Rare"]).

    Returns:
        Tuple of (filtered_events, filtered_event_id).
    """
    filtered_event_id = {k: v for k, v in event_id.items() if k in include_types}
    if not filtered_event_id:
        logger.warning(
            f"No matching event types found. Requested: {include_types}, "
            f"Available: {list(event_id.keys())}"
        )
        return events, event_id

    keep_ids = set(filtered_event_id.values())
    mask = np.isin(events[:, 2], list(keep_ids))
    filtered_events = events[mask]

    logger.info(
        f"Filtered events: {len(events)} -> {len(filtered_events)} "
        f"(keeping {include_types})"
    )
    return filtered_events, filtered_event_id


def compute_isi_statistics(events_stim: np.ndarray, sfreq: float) -> dict:
    """Compute inter-stimulus interval statistics.

    Args:
        events_stim: MNE events array for stimulus events only.
        sfreq: Sampling frequency in Hz.

    Returns:
        Dict with ISI stats: mean, std, min, max, median (all in seconds).
    """
    if len(events_stim) < 2:
        return {"mean": None, "std": None, "min": None, "max": None, "median": None}

    samples = events_stim[:, 0]
    isi_samples = np.diff(samples)
    isi_sec = isi_samples / sfreq

    stats = {
        "mean": float(np.mean(isi_sec)),
        "std": float(np.std(isi_sec)),
        "min": float(np.min(isi_sec)),
        "max": float(np.max(isi_sec)),
        "median": float(np.median(isi_sec)),
        "n_intervals": int(len(isi_sec)),
    }
    logger.info(
        f"ISI stats: mean={stats['mean']:.3f}s, std={stats['std']:.3f}s, "
        f"min={stats['min']:.3f}s, max={stats['max']:.3f}s"
    )
    return stats


def compute_event_counts(events_df: pd.DataFrame) -> dict:
    """Compute per-trial_type event counts from events dataframe.

    Args:
        events_df: Pandas DataFrame with 'trial_type' column.

    Returns:
        Dict mapping trial_type to count, plus 'total'.
    """
    col = "trial_type" if "trial_type" in events_df.columns else "value"
    counts = events_df[col].value_counts().to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}
    counts["total"] = int(len(events_df))
    logger.info(f"Event counts: {counts}")
    return counts


def detect_bad_epochs_data_driven(
    epochs: mne.Epochs,
    mad_multiplier: float = 4.0,
    flat_mad_multiplier: float = 4.0,
) -> Tuple[np.ndarray, dict]:
    """Detect bad epochs using data-driven thresholds (median PTP + N * MAD).

    Args:
        epochs: Epochs object to check.
        mad_multiplier: Number of MADs above median for reject threshold.
        flat_mad_multiplier: Number of MADs below median for flat threshold.

    Returns:
        Tuple of (bad_mask, stats) where stats includes computed thresholds.
    """
    n_epochs = len(epochs)
    picks_mag = mne.pick_types(epochs.info, meg="mag")
    data = epochs.get_data(picks=picks_mag)  # (n_epochs, n_channels, n_times)

    # Max PTP across channels per epoch
    ptp_per_epoch = np.max(np.ptp(data, axis=2), axis=1)  # (n_epochs,)

    median_ptp = float(np.median(ptp_per_epoch))
    mad_ptp = float(np.median(np.abs(ptp_per_epoch - median_ptp)))

    reject_threshold = median_ptp + mad_multiplier * mad_ptp
    flat_threshold = max(median_ptp - flat_mad_multiplier * mad_ptp, 0.0)

    bad_reject = ptp_per_epoch > reject_threshold
    bad_flat = ptp_per_epoch < flat_threshold
    bad_mask = bad_reject | bad_flat

    n_bad = int(np.sum(bad_mask))
    pct_bad = 100 * n_bad / n_epochs if n_epochs > 0 else 0.0

    stats = {
        "n_total": n_epochs,
        "n_bad": n_bad,
        "n_good": n_epochs - n_bad,
        "pct_bad": pct_bad,
        "reject_threshold": {"mag": float(reject_threshold)},
        "flat_threshold": {"mag": float(flat_threshold)},
        "n_bad_reject": int(np.sum(bad_reject)),
        "n_bad_flat": int(np.sum(bad_flat)),
        "mode": "data_driven",
        "mad_multiplier": mad_multiplier,
        "flat_mad_multiplier": flat_mad_multiplier,
        "median_ptp": median_ptp,
        "mad_ptp": mad_ptp,
        "ptp_percentiles": {
            "p5": float(np.percentile(ptp_per_epoch, 5)),
            "p25": float(np.percentile(ptp_per_epoch, 25)),
            "p50": median_ptp,
            "p75": float(np.percentile(ptp_per_epoch, 75)),
            "p95": float(np.percentile(ptp_per_epoch, 95)),
        },
    }

    logger.info(
        f"Data-driven threshold detection: {n_bad}/{n_epochs} bad epochs "
        f"({pct_bad:.1f}%) [reject={reject_threshold*1e15:.0f} fT, "
        f"flat={flat_threshold*1e15:.0f} fT]"
    )
    return bad_mask, stats


def pre_ar2_outlier_filter(
    epochs: mne.Epochs,
    ptp_multiplier: float = 10.0,
) -> Tuple[np.ndarray, dict]:
    """Identify extreme outlier epochs before AR2 (safety filter).

    Flags epochs with max PTP > ptp_multiplier * median PTP.

    Args:
        epochs: Epochs to filter.
        ptp_multiplier: Multiplier over median PTP to flag as outlier.

    Returns:
        Tuple of (outlier_mask, stats).
    """
    picks_mag = mne.pick_types(epochs.info, meg="mag")
    data = epochs.get_data(picks=picks_mag)
    ptp_per_epoch = np.max(np.ptp(data, axis=2), axis=1)

    median_ptp = float(np.median(ptp_per_epoch))
    threshold = ptp_multiplier * median_ptp
    outlier_mask = ptp_per_epoch > threshold

    n_outliers = int(np.sum(outlier_mask))

    stats = {
        "n_outliers": n_outliers,
        "n_total": len(epochs),
        "ptp_multiplier": ptp_multiplier,
        "median_ptp": median_ptp,
        "threshold": float(threshold),
    }

    logger.info(
        f"Pre-AR2 filter: {n_outliers}/{len(epochs)} outlier epochs "
        f"(>{ptp_multiplier}x median PTP = {threshold*1e15:.0f} fT)"
    )
    return outlier_mask, stats


def add_bad_epoch_annotations(
    raw: mne.io.Raw,
    epochs: mne.Epochs,
    bad_mask: np.ndarray,
    description: str = "BAD_AR2",
) -> mne.io.Raw:
    """Add BAD annotations to Raw for epochs marked as bad.

    Args:
        raw: Raw object to annotate.
        epochs: Epochs object (for timing information).
        bad_mask: Boolean array (n_epochs,) where True = bad.
        description: Annotation description string.

    Returns:
        Raw object with added annotations.
    """
    if not np.any(bad_mask):
        logger.info(f"No bad epochs to annotate as {description}")
        return raw

    sfreq = raw.info["sfreq"]
    bad_indices = np.where(bad_mask)[0]

    onsets = []
    durations = []
    epoch_duration = epochs.tmax - epochs.tmin

    for idx in bad_indices:
        # Get the event time for this epoch
        event_sample = epochs.events[idx, 0]
        epoch_onset = event_sample / sfreq + epochs.tmin
        onsets.append(epoch_onset)
        durations.append(epoch_duration)

    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=[description] * len(onsets),
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(raw.annotations + annotations)
    logger.info(f"Added {len(onsets)} {description} annotations to Raw")
    return raw
