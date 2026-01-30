"""Preprocessing utilities for saflow.

This module provides helper functions for the preprocessing pipeline:
- BIDS path creation for preprocessed data
- Noise covariance computation from empty-room recordings
- Filtering and epoching utilities
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import mne
import numpy as np
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
        - 'ARlog_first': First AutoReject log (for ICA)
        - 'ARlog_second': Second AutoReject log (with interpolation)
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
        extension=".ds",
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

    return {
        "raw": raw_bidspath,
        "preproc": preproc_bidspath,
        "epoch_ica": epoch_ica_bidspath,
        "epoch_ica_ar": epoch_ica_ar_bidspath,
        "ARlog_first": ARlog_first_bidspath,
        "ARlog_second": ARlog_second_bidspath,
        "report": report_bidspath,
    }


def compute_or_load_noise_cov(
    er_date: str,
    bids_root: Path,
    derivatives_root: Path,
) -> mne.Covariance:
    """Compute or load noise covariance from empty-room recording.

    Computes noise covariance matrix from empty-room recording if not
    already cached, otherwise loads from file.

    Args:
        er_date: Empty-room recording date (YYYYMMDD format).
        bids_root: BIDS dataset root directory.
        derivatives_root: Derivatives directory for cached covariance.

    Returns:
        Noise covariance matrix.

    Examples:
        >>> noise_cov = compute_or_load_noise_cov('20190615', bids_root, deriv_root)
    """
    noise_cov_dir = derivatives_root / "noise_cov"
    noise_cov_dir.mkdir(parents=True, exist_ok=True)

    noise_cov_bidspath = BIDSPath(
        subject="emptyroom",
        session=er_date,
        task="noise",
        datatype="meg",
        processing="noisecov",
        root=str(noise_cov_dir),
    )

    noise_cov_file = Path(str(noise_cov_bidspath.fpath) + ".fif")

    if not noise_cov_file.exists():
        logger.info(f"Computing noise covariance for {er_date}")

        er_raw = read_raw_bids(
            BIDSPath(
                subject="emptyroom",
                session=er_date,
                task="noise",
                datatype="meg",
                root=str(bids_root),
            )
        )

        noise_cov = mne.compute_raw_covariance(
            er_raw, method=["shrunk", "empirical"], rank=None, verbose=False
        )

        noise_cov.save(str(noise_cov_file), overwrite=True)
        logger.info(f"Saved noise covariance to {noise_cov_file}")
    else:
        logger.debug(f"Loading cached noise covariance from {noise_cov_file}")
        noise_cov = mne.read_cov(str(noise_cov_file))

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
