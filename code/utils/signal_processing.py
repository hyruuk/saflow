"""MEG/EEG preprocessing and analysis utilities for saflow.

This module provides utilities for:
- Frequency band power computation
- Event detection and annotation
- Epoch trimming and quality control
- Channel selection

These are reusable utilities; main preprocessing pipelines are in code/preprocessing/.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

logger = logging.getLogger(__name__)

# Default frequency bands (can be overridden by config)
DEFAULT_FREQS = [
    [2, 4],    # delta
    [4, 8],    # theta
    [8, 12],   # alpha
    [12, 20],  # low beta
    [20, 30],  # high beta
    [30, 60],  # gamma1
    [60, 90],  # gamma2
    [90, 120], # gamma3
]


def average_bands(
    psd: np.ndarray,
    freq_bins: np.ndarray,
    bands: Optional[List[List[float]]] = None,
) -> np.ndarray:
    """Compute average power in frequency bands from PSD.

    Averages power spectral density values within predefined frequency bands
    (delta, theta, alpha, beta, gamma).

    Args:
        psd: Power spectral density array, shape (n_freqs,).
        freq_bins: Frequency values corresponding to PSD, shape (n_freqs,).
        bands: List of [low, high] frequency ranges for each band.
            Defaults to standard 8 bands if None. Load from config for
            project-specific bands.

    Returns:
        Array of band power values, shape (n_bands,).

    Examples:
        >>> psd = np.random.rand(100)
        >>> freqs = np.linspace(2, 120, 100)
        >>> band_power = average_bands(psd, freqs)
        >>> print(f"Got {len(band_power)} frequency bands")
    """
    if bands is None:
        bands = DEFAULT_FREQS

    band_power = []
    for band_low, band_high in bands:
        band_mask = np.logical_and(freq_bins >= band_low, freq_bins <= band_high)
        if np.any(band_mask):
            band_power.append(np.mean(psd[band_mask]))
        else:
            logger.warning(
                f"No frequency bins found in range [{band_low}, {band_high}] Hz"
            )
            band_power.append(np.nan)

    return np.array(band_power)


def get_present_events(events: np.ndarray) -> Dict[str, int]:
    """Select events detected from a full list of events.

    Filters a predefined event dictionary to only include events that are
    present in the data. Used for gradCPT task events.

    Args:
        events: MNE-style events array, shape (n_events, 3).

    Returns:
        MNE-style event_id dictionary mapping event names to codes.

    Examples:
        >>> events = mne.find_events(raw)
        >>> event_id = get_present_events(events)
        >>> print(event_id.keys())
    """
    full_event_id = {
        "FreqHit": 211,
        "FreqMiss": 210,
        "RareHit": 311,
        "RareMiss": 310,
        "FreqIN": 2111,
        "FreqOUT": 2110,
    }

    event_id = {
        key: full_event_id[key]
        for key in full_event_id.keys()
        if full_event_id[key] in np.unique(events[:, 2])
    }

    logger.debug(f"Found {len(event_id)} event types: {list(event_id.keys())}")

    return event_id


def trim_events(
    events_noerr: np.ndarray,
    events_artrej: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find intersection of correct epochs and artifact-rejected epochs.

    Compares the events vectors of correct epochs (no behavioral errors) and
    kept epochs after artifact rejection to find the intersection.

    Args:
        events_noerr: Events array for correct trials, shape (n_correct, 3).
        events_artrej: Events array for artifact-free trials, shape (n_clean, 3).

    Returns:
        Tuple containing:
        - events_trimmed: Events in intersection, shape (n_kept, 3)
        - idx_trimmed: Indices of kept events in events_artrej

    Examples:
        >>> events_trimmed, idx = trim_events(correct_events, clean_events)
        >>> print(f"Kept {len(idx)} out of {len(clean_events)} epochs")
    """
    events_trimmed = []
    idx_trimmed = []

    for idx, event in enumerate(events_artrej):
        if event[0] in events_noerr[:, 0]:
            events_trimmed.append(event)
            idx_trimmed.append(idx)

    events_trimmed = np.array(events_trimmed)
    idx_trimmed = np.array(idx_trimmed)

    logger.info(
        f"N events in clean epochs: {len(events_artrej)}, "
        f"N events in correct epochs: {len(events_noerr)}, "
        f"N events in intersection: {len(idx_trimmed)}"
    )

    return events_trimmed, idx_trimmed


def trim_INOUT_idx(
    INidx: np.ndarray,
    OUTidx: np.ndarray,
    events_trimmed: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim IN/OUT indices to match artifact-rejected events.

    Updates IN and OUT epoch indices to refer to the trimmed (artifact-free)
    event list instead of the full event list.

    Args:
        INidx: Indices of IN epochs in full event list.
        OUTidx: Indices of OUT epochs in full event list.
        events_trimmed: Trimmed events array (artifact-free).
        events: Full events array (including artifacts).

    Returns:
        Tuple containing:
        - INidx_trimmed: IN indices in trimmed events
        - OUTidx_trimmed: OUT indices in trimmed events

    Examples:
        >>> IN_trim, OUT_trim = trim_INOUT_idx(IN_idx, OUT_idx, trimmed, all_events)
    """
    # Get events vector with only stimuli events (no response events)
    new_events = []
    for ev in events:
        if ev[2] != 99:  # 99 is response event code
            new_events.append(ev)
    events = np.array(new_events)

    INidx_trimmed = []
    OUTidx_trimmed = []

    # Compare trimmed events with all_events, store corresponding indices
    for idx, ev in enumerate(events):
        for idx_trim, ev_trim in enumerate(events_trimmed):
            if ev[0] == ev_trim[0]:
                if idx in INidx:
                    INidx_trimmed.append(idx_trim)
                if idx in OUTidx:
                    OUTidx_trimmed.append(idx_trim)

    INidx_trimmed = np.array(INidx_trimmed)
    OUTidx_trimmed = np.array(OUTidx_trimmed)

    logger.debug(
        f"Trimmed IN: {len(INidx)} -> {len(INidx_trimmed)}, "
        f"OUT: {len(OUTidx)} -> {len(OUTidx_trimmed)}"
    )

    return INidx_trimmed, OUTidx_trimmed


def compute_PSD_hilbert(
    raw: mne.io.Raw,
    ARlog: "autoreject.AutoReject",
    tmin: float = 0.0,
    tmax: float = 0.8,
    freqlist: Optional[List[List[float]]] = None,
) -> np.ndarray:
    """Compute band power using Hilbert transform.

    Filters raw data into frequency bands, applies Hilbert transform to get
    envelope, then segments and computes power.

    Args:
        raw: MNE Raw object with continuous data.
        ARlog: AutoReject log with bad epochs identified.
        tmin: Start time of epochs in seconds. Defaults to 0.0.
        tmax: End time of epochs in seconds. Defaults to 0.8.
        freqlist: List of [low, high] frequency bands. Defaults to standard bands.
            Load from config for project-specific bands.

    Returns:
        Epoched power data, shape (n_epochs, n_channels, n_bands).

    Examples:
        >>> from autoreject import AutoReject
        >>> ar = AutoReject()
        >>> epochs_power = compute_PSD_hilbert(raw, ar)
    """
    if freqlist is None:
        freqlist = DEFAULT_FREQS

    epochs_psds = []

    for low, high in freqlist:
        # Filter continuous data
        data = raw.copy().filter(low, high)
        hilbert = data.apply_hilbert(envelope=True)
        hilbert_pow = hilbert.copy()
        hilbert_pow._data = hilbert._data**2

        # Segment filtered data
        picks = mne.pick_types(
            raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False
        )

        try:
            events = mne.find_events(
                raw, min_duration=1 / raw.info["sfreq"], verbose=False
            )
        except ValueError:
            events = mne.find_events(
                raw, min_duration=2 / raw.info["sfreq"], verbose=False
            )

        event_id = {"Freq": 21, "Rare": 31}

        epochs = mne.Epochs(
            hilbert,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            picks=picks,
            preload=True,
        )

        # Drop bad epochs
        epochs.drop(ARlog.bad_epochs)
        epochs_psds.append(epochs.get_data())

    # Combine bands and average in time
    epochs_psds = np.array(epochs_psds)
    epochs_psds = np.mean(epochs_psds, axis=3).transpose(1, 2, 0)

    logger.info(
        f"Computed Hilbert power: {epochs_psds.shape[0]} epochs, "
        f"{epochs_psds.shape[1]} channels, {epochs_psds.shape[2]} bands"
    )

    return epochs_psds


def compute_envelopes_hilbert(
    raw: mne.io.Raw,
    ARlog: "autoreject.AutoReject",
    freqlist: Optional[List[List[float]]] = None,
    tmin: float = 0.0,
    tmax: float = 0.8,
) -> List[mne.Epochs]:
    """Compute band-limited envelopes using Hilbert transform.

    Filters raw data into frequency bands, segments into epochs, applies
    Hilbert transform, and returns list of epoch objects (one per band).

    Args:
        raw: MNE Raw object with continuous data.
        ARlog: AutoReject log with bad epochs identified.
        freqlist: List of [low, high] frequency bands. Defaults to standard bands.
            Load from config for project-specific bands.
        tmin: Start time of epochs in seconds. Defaults to 0.0.
        tmax: End time of epochs in seconds. Defaults to 0.8.

    Returns:
        List of MNE Epochs objects, one per frequency band.

    Examples:
        >>> epochs_list = compute_envelopes_hilbert(raw, ar)
        >>> print(f"Got {len(epochs_list)} frequency bands")
    """
    if freqlist is None:
        freqlist = DEFAULT_FREQS

    epochs_envelopes = []

    for low, high in freqlist:
        # Filter continuous data
        data = raw.copy().filter(low, high)

        # Segment data
        picks = mne.pick_types(
            raw.info, meg=True, ref_meg=False, eeg=False, eog=False, stim=False
        )

        try:
            events = mne.find_events(
                raw, min_duration=1 / raw.info["sfreq"], verbose=False
            )
        except ValueError:
            events = mne.find_events(
                raw, min_duration=2 / raw.info["sfreq"], verbose=False
            )

        event_id = {"Freq": 21, "Rare": 31}

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            picks=picks,
            preload=True,
        )

        # Drop bad epochs
        epochs.drop(ARlog.bad_epochs)

        # Apply Hilbert transform to obtain envelope
        hilbert = epochs.apply_hilbert(envelope=True)
        hilbert_pow = hilbert.copy()
        hilbert_pow._data = hilbert._data**2

        epochs_envelopes.append(hilbert_pow)

        # Clean up
        del hilbert_pow
        del hilbert
        del data
        del epochs

    logger.info(f"Computed {len(epochs_envelopes)} band-limited envelopes")

    return epochs_envelopes
