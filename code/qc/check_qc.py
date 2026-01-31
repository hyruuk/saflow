"""Data quality control checks for raw MEG sourcedata.

This module performs comprehensive QC checks on raw MEG data to assess
data quality and detect issues before BIDS conversion and preprocessing:

- Recording parameters (sampling rate, duration, channels)
- ISI (Inter-Stimulus Interval) validation from event triggers
- Event/trigger validation and counts
- Channel quality (flat, noisy, missing channels)
- Response detection and behavioral consistency

Usage:
    python -m code.qc.check_qc                    # All subjects
    python -m code.qc.check_qc --subject 04       # Single subject
    python -m code.qc.check_qc --verbose          # Detailed output
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes for QC Results
# ==============================================================================

@dataclass
class RecordingParams:
    """Recording parameters from MEG data."""
    sampling_rate: float = 0.0
    duration_sec: float = 0.0
    n_samples: int = 0
    n_channels_total: int = 0
    n_meg_channels: int = 0
    n_ref_channels: int = 0
    n_eeg_channels: int = 0
    n_stim_channels: int = 0
    n_other_channels: int = 0
    line_freq: float = 60.0
    highpass: float = 0.0
    lowpass: float = 0.0
    meg_system: str = ""
    passed: bool = True
    message: str = ""


@dataclass
class ISICheck:
    """Results of Inter-Stimulus Interval check from event triggers."""
    n_stimulus_events: int = 0
    mean_isi_ms: float = 0.0
    std_isi_ms: float = 0.0
    min_isi_ms: float = 0.0
    max_isi_ms: float = 0.0
    median_isi_ms: float = 0.0
    cv_isi: float = 0.0  # Coefficient of variation (std/mean) - measure of consistency
    n_outliers: int = 0  # ISIs outside ±3 std from mean
    isi_histogram: Dict[str, int] = field(default_factory=dict)  # Binned ISI counts
    passed: bool = True
    message: str = ""


@dataclass
class EventCheck:
    """Results of event/trigger validation."""
    n_events_total: int = 0
    event_types: Dict[str, int] = field(default_factory=dict)  # event_id -> count
    stim_channel: str = ""
    n_freq_stim: int = 0  # Frequent stimuli (event 21)
    n_rare_stim: int = 0  # Rare stimuli (event 31)
    n_responses: int = 0  # Button responses (event 99)
    n_block_starts: int = 0  # Block markers (event 10)
    passed: bool = True
    message: str = ""


@dataclass
class ChannelCheck:
    """Results of channel quality check."""
    n_meg_channels: int = 0
    n_expected_meg: int = 275  # CTF 275 system
    n_bad_total: int = 0  # Total bad channels (flat + noisy + from file)
    missing_channels: List[str] = field(default_factory=list)
    bad_channels_file: List[str] = field(default_factory=list)  # From BadChannels file
    n_flat_channels: int = 0
    flat_channels: List[str] = field(default_factory=list)
    n_noisy_channels: int = 0
    noisy_channels: List[str] = field(default_factory=list)
    pct_bad: float = 0.0  # Percentage of bad channels
    passed: bool = True
    message: str = ""


@dataclass
class EpochCheck:
    """Results of epoch quality check based on amplitude thresholds."""
    n_epochs_total: int = 0
    n_bad_epochs: int = 0
    pct_bad_epochs: float = 0.0
    bad_epoch_indices: List[int] = field(default_factory=list)
    # Threshold info
    peak_to_peak_threshold_t: float = 4000e-15  # 4000 fT for MEG
    # Per-epoch statistics
    mean_peak_to_peak_t: float = 0.0
    max_peak_to_peak_t: float = 0.0
    passed: bool = True
    message: str = ""


@dataclass
class ResponseCheck:
    """Results of response detection check."""
    # From MEG triggers (raw)
    n_stimuli: int = 0
    n_raw_button_presses: int = 0  # Raw event 99 count
    # From behavioral .mat file (ground truth)
    n_trials: int = 0
    n_attributed_responses: int = 0  # Properly attributed responses
    response_rate: float = 0.0  # Attributed responses / trials
    mean_rt_ms: float = 0.0
    std_rt_ms: float = 0.0
    min_rt_ms: float = 0.0
    max_rt_ms: float = 0.0
    # Performance breakdown
    n_commission_errors: int = 0
    n_correct_omissions: int = 0
    n_omission_errors: int = 0
    n_correct_commissions: int = 0
    passed: bool = True
    message: str = ""


@dataclass
class RunQCResult:
    """Complete QC results for a single run."""
    subject: str = ""
    run: str = ""
    file_path: str = ""
    timestamp: str = ""
    recording: RecordingParams = field(default_factory=RecordingParams)
    isi: ISICheck = field(default_factory=ISICheck)
    events: EventCheck = field(default_factory=EventCheck)
    channels: ChannelCheck = field(default_factory=ChannelCheck)
    epochs: EpochCheck = field(default_factory=EpochCheck)
    responses: ResponseCheck = field(default_factory=ResponseCheck)
    overall_passed: bool = True
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SubjectQCResult:
    """Complete QC results for a subject across all runs."""
    subject: str = ""
    date_folder: str = ""
    n_runs: int = 0
    n_passed: int = 0
    n_failed: int = 0
    runs: Dict[str, RunQCResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_subject_date_mapping(sourcedata_meg: Path) -> Dict[str, str]:
    """Build mapping of subjects to their date folders in sourcedata/meg."""
    mapping = {}
    if not sourcedata_meg.exists():
        return mapping

    for date_folder in sorted(sourcedata_meg.iterdir()):
        if not date_folder.is_dir():
            continue
        for ds_file in date_folder.glob("SA*_SAflow*.ds"):
            fname = ds_file.name
            if fname.startswith("SA") and "_SAflow" in fname:
                try:
                    subject = fname.split("SA")[1][:2]
                    if subject not in mapping:
                        mapping[subject] = date_folder.name
                except (IndexError, ValueError):
                    pass
    return mapping


def find_meg_files(sourcedata_meg: Path, subject: str, date_folder: str) -> Dict[str, Path]:
    """Find all MEG files for a subject.

    Returns dict mapping run number to file path.
    """
    files = {}
    folder = sourcedata_meg / date_folder
    if not folder.exists():
        return files

    for ds_file in sorted(folder.glob(f"SA{subject}_SAflow*.ds")):
        try:
            # Parse run from: SA04_SAflow-yharel_20190411_01.ds
            run = ds_file.name.replace(".ds", "").split("_")[-1]
            files[run] = ds_file
        except (IndexError, ValueError):
            pass
    return files


# ==============================================================================
# QC Check Functions
# ==============================================================================

def check_recording_params(raw) -> RecordingParams:
    """Extract and validate recording parameters.

    Args:
        raw: MNE Raw object.

    Returns:
        RecordingParams with recording details.
    """
    from mne import pick_types

    result = RecordingParams()

    # Basic parameters
    result.sampling_rate = raw.info['sfreq']
    result.n_samples = raw.n_times
    result.duration_sec = raw.n_times / raw.info['sfreq']
    result.n_channels_total = len(raw.ch_names)

    # Line frequency
    result.line_freq = raw.info.get('line_freq', 60.0) or 60.0

    # Filter settings
    result.highpass = raw.info.get('highpass', 0.0) or 0.0
    result.lowpass = raw.info.get('lowpass', 0.0) or 0.0

    # Count channel types
    try:
        meg_idx = pick_types(raw.info, meg=True, ref_meg=False, exclude=[])
        result.n_meg_channels = len(meg_idx)
    except:
        result.n_meg_channels = 0

    try:
        ref_idx = pick_types(raw.info, meg=False, ref_meg=True, exclude=[])
        result.n_ref_channels = len(ref_idx)
    except:
        result.n_ref_channels = 0

    try:
        eeg_idx = pick_types(raw.info, meg=False, eeg=True, exclude=[])
        result.n_eeg_channels = len(eeg_idx)
    except:
        result.n_eeg_channels = 0

    try:
        stim_idx = pick_types(raw.info, meg=False, stim=True, exclude=[])
        result.n_stim_channels = len(stim_idx)
    except:
        result.n_stim_channels = 0

    result.n_other_channels = (result.n_channels_total - result.n_meg_channels -
                                result.n_ref_channels - result.n_eeg_channels -
                                result.n_stim_channels)

    # MEG system info
    if hasattr(raw.info, 'get') and raw.info.get('device_info'):
        result.meg_system = raw.info['device_info'].get('type', 'CTF')
    else:
        result.meg_system = "CTF"  # Default for this dataset

    # Validation
    if result.sampling_rate < 100:
        result.passed = False
        result.message = f"Unexpected low sampling rate: {result.sampling_rate} Hz"
    elif result.n_meg_channels < 200:
        result.passed = False
        result.message = f"Low MEG channel count: {result.n_meg_channels}"
    else:
        result.message = (f"sfreq={result.sampling_rate}Hz, duration={result.duration_sec:.1f}s, "
                         f"MEG={result.n_meg_channels}, REF={result.n_ref_channels}, "
                         f"EEG={result.n_eeg_channels}, STIM={result.n_stim_channels}")

    return result


def check_isi_from_events(events: np.ndarray, sfreq: float,
                          stim_event_ids: List[int] = None) -> ISICheck:
    """Check Inter-Stimulus Interval consistency from event triggers.

    Focuses on ISI consistency (coefficient of variation) rather than
    comparing to an expected value.

    Args:
        events: MNE events array (n_events, 3).
        sfreq: Sampling frequency in Hz.
        stim_event_ids: Event IDs for stimulus events.

    Returns:
        ISICheck with ISI statistics.
    """
    result = ISICheck()

    # Default gradCPT stimulus event IDs
    if stim_event_ids is None:
        stim_event_ids = [21, 31]  # Freq and Rare stimuli

    # Get stimulus events
    stim_mask = np.isin(events[:, 2], stim_event_ids)
    stim_samples = events[stim_mask, 0]
    result.n_stimulus_events = len(stim_samples)

    if len(stim_samples) < 2:
        result.passed = False
        result.message = f"Insufficient stimulus events: {len(stim_samples)}"
        return result

    # Compute ISIs
    isis_samples = np.diff(stim_samples)
    isis_ms = (isis_samples / sfreq) * 1000

    result.mean_isi_ms = float(np.mean(isis_ms))
    result.std_isi_ms = float(np.std(isis_ms))
    result.min_isi_ms = float(np.min(isis_ms))
    result.max_isi_ms = float(np.max(isis_ms))
    result.median_isi_ms = float(np.median(isis_ms))

    # Coefficient of variation - measure of consistency (lower = more consistent)
    if result.mean_isi_ms > 0:
        result.cv_isi = result.std_isi_ms / result.mean_isi_ms
    else:
        result.cv_isi = 0.0

    # Find outliers using ±3 std from mean (data-driven)
    lower_bound = result.mean_isi_ms - 3 * result.std_isi_ms
    upper_bound = result.mean_isi_ms + 3 * result.std_isi_ms
    outlier_mask = (isis_ms < lower_bound) | (isis_ms > upper_bound)
    result.n_outliers = int(np.sum(outlier_mask))

    # Create histogram (100ms bins from 0 to 2000ms)
    bins = np.arange(0, 2100, 100)
    hist, _ = np.histogram(isis_ms, bins=bins)
    result.isi_histogram = {f"{int(bins[i])}-{int(bins[i+1])}ms": int(hist[i])
                           for i in range(len(hist)) if hist[i] > 0}

    # Validation based on consistency
    # CV > 0.1 (10%) indicates inconsistent timing
    if result.cv_isi > 0.1:
        result.passed = False
        result.message = f"Inconsistent ISI: CV={result.cv_isi:.3f} (std={result.std_isi_ms:.1f}ms)"
    elif result.n_outliers > len(isis_ms) * 0.01:
        result.passed = False
        result.message = f"ISI outliers: {result.n_outliers}/{len(isis_ms)} (>{1}%)"
    else:
        result.message = (f"ISI: mean={result.mean_isi_ms:.0f}ms, std={result.std_isi_ms:.1f}ms, "
                         f"CV={result.cv_isi:.4f}")

    return result


def check_events(events: np.ndarray, stim_channel: str = "UPPT001") -> EventCheck:
    """Validate events/triggers.

    Args:
        events: MNE events array.
        stim_channel: Name of stimulus channel.

    Returns:
        EventCheck with event statistics.
    """
    result = EventCheck(stim_channel=stim_channel)
    result.n_events_total = len(events)

    if len(events) == 0:
        result.passed = False
        result.message = "No events found!"
        return result

    # Count event types
    unique_events, counts = np.unique(events[:, 2], return_counts=True)
    result.event_types = {str(int(e)): int(c) for e, c in zip(unique_events, counts)}

    # gradCPT specific event codes
    result.n_freq_stim = result.event_types.get("21", 0)
    result.n_rare_stim = result.event_types.get("31", 0)
    result.n_responses = result.event_types.get("99", 0)
    result.n_block_starts = result.event_types.get("10", 0)

    total_stim = result.n_freq_stim + result.n_rare_stim

    # Validation
    if total_stim < 400:  # Expect ~525 stimuli per run
        result.passed = False
        result.message = f"Low stimulus count: {total_stim} (expected ~525)"
    else:
        result.message = (f"Events: Freq={result.n_freq_stim}, Rare={result.n_rare_stim}, "
                         f"Resp={result.n_responses}, Blocks={result.n_block_starts}")

    return result


def check_channels(raw, ds_path: Path) -> ChannelCheck:
    """Check channel quality and read BadChannels file.

    Args:
        raw: MNE Raw object.
        ds_path: Path to .ds directory.

    Returns:
        ChannelCheck with channel quality info.
    """
    from mne import pick_types

    result = ChannelCheck()

    # Get MEG channels
    try:
        meg_idx = pick_types(raw.info, meg=True, ref_meg=False, exclude=[])
        meg_channels = [raw.ch_names[i] for i in meg_idx]
        result.n_meg_channels = len(meg_channels)
    except:
        result.n_meg_channels = 0
        meg_channels = []

    # Check for missing channels (CTF 275 system)
    if result.n_meg_channels < result.n_expected_meg:
        result.missing_channels = [f"Missing {result.n_expected_meg - result.n_meg_channels} channels"]

    # Read BadChannels file from .ds directory
    bad_channels_file = ds_path / "BadChannels"
    if bad_channels_file.exists():
        try:
            with open(bad_channels_file, 'r') as f:
                bad_channels = [line.strip() for line in f if line.strip()]
                result.bad_channels_file = bad_channels
        except:
            pass

    # Check for flat channels (sample a portion of data)
    try:
        # Sample 10 seconds from the middle
        start_sample = max(0, raw.n_times // 2 - int(5 * raw.info['sfreq']))
        stop_sample = min(raw.n_times, start_sample + int(10 * raw.info['sfreq']))
        data = raw.get_data(picks=meg_idx, start=start_sample, stop=stop_sample)

        # Compute per-channel variance
        variances = np.var(data, axis=1)

        # Flat channels: near-zero variance
        flat_threshold = 1e-30
        flat_mask = variances < flat_threshold
        result.n_flat_channels = int(np.sum(flat_mask))
        result.flat_channels = [meg_channels[i] for i in np.where(flat_mask)[0]]

        # Noisy channels: high variance outliers
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        if std_var > 0:
            z_scores = (variances - mean_var) / std_var
            noisy_mask = z_scores > 5.0
            result.n_noisy_channels = int(np.sum(noisy_mask))
            result.noisy_channels = [meg_channels[i] for i in np.where(noisy_mask)[0]]
    except Exception as e:
        logger.warning(f"Could not check channel quality: {e}")

    # Compute totals
    # Get unique bad channels (some may be in multiple categories)
    all_bad = set(result.flat_channels) | set(result.noisy_channels) | set(result.bad_channels_file)
    result.n_bad_total = len(all_bad)
    result.pct_bad = (result.n_bad_total / result.n_meg_channels * 100) if result.n_meg_channels > 0 else 0

    # Validation
    if result.pct_bad > 10:
        result.passed = False
        result.message = f"High bad channel count: {result.n_bad_total}/{result.n_meg_channels} ({result.pct_bad:.1f}%)"
    else:
        result.message = (f"Channels: {result.n_meg_channels} MEG, "
                         f"{result.n_bad_total} bad ({result.pct_bad:.1f}%)")

    return result


def check_epochs(raw, events: np.ndarray, sfreq: float,
                 stim_event_ids: List[int] = None,
                 peak_to_peak_threshold: float = 4000e-15) -> EpochCheck:
    """Check epoch quality based on amplitude thresholds.

    Creates epochs around stimulus events and identifies bad epochs
    based on peak-to-peak amplitude threshold.

    Args:
        raw: MNE Raw object (must be preloaded).
        events: MNE events array.
        sfreq: Sampling frequency.
        stim_event_ids: Event IDs for stimulus events.
        peak_to_peak_threshold: Threshold in Tesla for MEG (default 4000 fT).

    Returns:
        EpochCheck with epoch quality info.
    """
    import mne
    from mne import pick_types

    result = EpochCheck()
    result.peak_to_peak_threshold_t = peak_to_peak_threshold

    # Default gradCPT stimulus event IDs
    if stim_event_ids is None:
        stim_event_ids = [21, 31]

    # Filter events to stimulus events only
    stim_mask = np.isin(events[:, 2], stim_event_ids)
    stim_events = events[stim_mask]

    if len(stim_events) < 10:
        result.message = f"Too few stimulus events for epoch check: {len(stim_events)}"
        return result

    try:
        # Create event_id dict
        event_id = {str(eid): eid for eid in stim_event_ids if eid in stim_events[:, 2]}

        # Create epochs (-0.2 to 0.8s around stimulus)
        epochs = mne.Epochs(
            raw, stim_events, event_id=event_id,
            tmin=-0.2, tmax=0.8,
            baseline=None,  # No baseline correction for QC
            preload=True,
            reject=None,  # We'll do our own rejection
            verbose=False
        )

        result.n_epochs_total = len(epochs)

        if result.n_epochs_total == 0:
            result.message = "No epochs created"
            return result

        # Get MEG data
        meg_picks = pick_types(epochs.info, meg=True, ref_meg=False, exclude=[])
        data = epochs.get_data(picks=meg_picks)  # (n_epochs, n_channels, n_times)

        # Compute peak-to-peak for each epoch (max across channels)
        ptp_per_epoch = np.ptp(data, axis=2).max(axis=1)  # (n_epochs,)

        result.mean_peak_to_peak_t = float(np.mean(ptp_per_epoch))
        result.max_peak_to_peak_t = float(np.max(ptp_per_epoch))

        # Find bad epochs
        bad_mask = ptp_per_epoch > peak_to_peak_threshold
        result.n_bad_epochs = int(np.sum(bad_mask))
        result.bad_epoch_indices = np.where(bad_mask)[0].tolist()
        result.pct_bad_epochs = (result.n_bad_epochs / result.n_epochs_total * 100)

        # Validation
        if result.pct_bad_epochs > 30:
            result.passed = False
            result.message = (f"High bad epoch rate: {result.n_bad_epochs}/{result.n_epochs_total} "
                            f"({result.pct_bad_epochs:.1f}%)")
        else:
            result.passed = True
            result.message = (f"Epochs: {result.n_bad_epochs}/{result.n_epochs_total} bad "
                            f"({result.pct_bad_epochs:.1f}%)")

    except Exception as e:
        result.passed = False
        result.message = f"Error checking epochs: {e}"
        logger.warning(result.message)

    return result


def check_responses_from_behavioral(
    behav_dir: Path, subject: str, run: str
) -> ResponseCheck:
    """Check response patterns from behavioral .mat file (ground truth).

    The .mat file contains properly attributed responses - one per trial at most.
    This gives the true response rate, unlike raw MEG button press counts.

    Args:
        behav_dir: Path to behavioral data directory.
        subject: Subject ID.
        run: Run number (02-07 for task runs).

    Returns:
        ResponseCheck with behavioral response statistics.
    """
    from scipy.io import loadmat

    result = ResponseCheck()

    # Map MEG run to behavioral run (MEG 02 -> behav 1, etc.)
    behav_run = str(int(run) - 1)

    # Find behavioral file for this subject/run
    behav_file = None
    for mat_file in sorted(behav_dir.glob(f"Data_0_{subject}_*.mat")):
        parts = mat_file.name.split("_")
        if len(parts) >= 4 and parts[3] == behav_run:
            behav_file = mat_file
            break

    if behav_file is None:
        result.message = f"Behavioral file not found for run {behav_run}"
        return result

    try:
        data = loadmat(behav_file)
        response = data['response']

        # Column 0: stimulus type (1=rare, 2=frequent)
        # Column 1: response indicator (0=no response, non-zero=response)
        # Column 4: RT in seconds

        stim_types = response[:, 0]
        resp_indicators = response[:, 1]
        rts = response[:, 4]

        # Filter out any non-stimulus trials (type 0)
        valid_mask = stim_types > 0
        stim_types = stim_types[valid_mask]
        resp_indicators = resp_indicators[valid_mask]
        rts = rts[valid_mask]

        result.n_trials = len(stim_types)
        result.n_attributed_responses = int(np.sum(resp_indicators != 0))
        result.response_rate = result.n_attributed_responses / result.n_trials if result.n_trials > 0 else 0

        # RT statistics (only for trials with responses)
        valid_rts = rts[rts > 0] * 1000  # Convert to ms
        if len(valid_rts) > 0:
            result.mean_rt_ms = float(np.mean(valid_rts))
            result.std_rt_ms = float(np.std(valid_rts))
            result.min_rt_ms = float(np.min(valid_rts))
            result.max_rt_ms = float(np.max(valid_rts))

        # Performance breakdown
        rare_mask = stim_types == 1
        freq_mask = stim_types == 2
        has_response = resp_indicators != 0

        result.n_commission_errors = int(np.sum(rare_mask & has_response))
        result.n_correct_omissions = int(np.sum(rare_mask & ~has_response))
        result.n_omission_errors = int(np.sum(freq_mask & ~has_response))
        result.n_correct_commissions = int(np.sum(freq_mask & has_response))

        # Validation
        if result.response_rate < 0.5:
            result.passed = False
            result.message = (
                f"Low response rate: {result.response_rate:.1%} "
                f"({result.n_attributed_responses}/{result.n_trials})"
            )
        elif result.response_rate > 1.0:
            result.passed = False
            result.message = (
                f"Invalid response rate: {result.response_rate:.1%} (>100%)"
            )
        else:
            result.passed = True
            result.message = (
                f"Resp: {result.n_attributed_responses}/{result.n_trials} "
                f"({result.response_rate:.1%}), RT={result.mean_rt_ms:.0f}±{result.std_rt_ms:.0f}ms"
            )

    except Exception as e:
        result.passed = False
        result.message = f"Error loading behavioral file: {e}"
        logger.warning(result.message)

    return result


def check_responses_from_triggers(events: np.ndarray, sfreq: float) -> Tuple[int, int]:
    """Count raw button presses and stimuli from MEG triggers.

    This is for comparison with behavioral file - raw counts include
    button bounces and multiple presses.

    Args:
        events: MNE events array.
        sfreq: Sampling frequency.

    Returns:
        Tuple of (n_stimuli, n_raw_button_presses).
    """
    stim_event_ids = [21, 31]  # Freq and Rare
    response_event_id = 99

    stim_mask = np.isin(events[:, 2], stim_event_ids)
    resp_mask = events[:, 2] == response_event_id

    return int(np.sum(stim_mask)), int(np.sum(resp_mask))


# ==============================================================================
# Main QC Runner
# ==============================================================================

def run_qc_for_run(ds_path: Path, subject: str, run: str, config: Dict,
                   behav_dir: Optional[Path] = None) -> RunQCResult:
    """Run all QC checks for a single run.

    Args:
        ds_path: Path to .ds file.
        subject: Subject ID.
        run: Run number.
        config: Configuration dictionary.
        behav_dir: Path to behavioral data directory (for response attribution).

    Returns:
        RunQCResult with all check results.
    """
    import mne

    result = RunQCResult(
        subject=subject,
        run=run,
        file_path=str(ds_path),
        timestamp=datetime.now().isoformat(),
    )

    if not ds_path.exists():
        result.critical_issues.append(f"File not found: {ds_path}")
        result.overall_passed = False
        return result

    logger.info(f"  Run {run}: {ds_path.name}")

    try:
        # Load raw data
        raw = mne.io.read_raw_ctf(str(ds_path), preload=False, verbose=False)

        # Recording parameters (always check)
        result.recording = check_recording_params(raw)
        logger.debug(f"    {result.recording.message}")

        # Get events
        try:
            events = mne.find_events(raw, stim_channel='UPPT001', verbose=False)
        except ValueError as e:
            # Try with min_duration if events too close
            min_dur = 2 / raw.info["sfreq"]
            events = mne.find_events(raw, stim_channel='UPPT001',
                                    min_duration=min_dur, verbose=False)

        # Event check
        result.events = check_events(events)
        logger.debug(f"    {result.events.message}")

        # ISI check (only for task runs with sufficient stimuli)
        if result.events.n_freq_stim + result.events.n_rare_stim > 100:
            result.isi = check_isi_from_events(events, raw.info['sfreq'])
            logger.debug(f"    {result.isi.message}")

        # Response check - use behavioral file for ground truth
        if behav_dir is not None and behav_dir.exists():
            result.responses = check_responses_from_behavioral(behav_dir, subject, run)
            # Also get raw trigger counts for comparison
            n_stim, n_raw_resp = check_responses_from_triggers(events, raw.info['sfreq'])
            result.responses.n_stimuli = n_stim
            result.responses.n_raw_button_presses = n_raw_resp
        else:
            # Fallback: just count raw triggers (will have inflated response rate)
            n_stim, n_raw_resp = check_responses_from_triggers(events, raw.info['sfreq'])
            result.responses.n_stimuli = n_stim
            result.responses.n_raw_button_presses = n_raw_resp
            result.responses.message = f"Raw triggers: {n_raw_resp} button presses / {n_stim} stimuli (behavioral file not available)"
        logger.debug(f"    {result.responses.message}")

        # Channel check (load minimal data)
        raw.load_data()
        result.channels = check_channels(raw, ds_path)
        logger.debug(f"    {result.channels.message}")

        # Epoch check (data already loaded)
        if result.events.n_freq_stim + result.events.n_rare_stim > 100:
            result.epochs = check_epochs(raw, events, raw.info['sfreq'])
            logger.debug(f"    {result.epochs.message}")

        # Collect issues
        if not result.recording.passed:
            result.warnings.append(f"Recording: {result.recording.message}")
        if not result.events.passed:
            result.critical_issues.append(f"Events: {result.events.message}")
        if not result.isi.passed:
            result.warnings.append(f"ISI: {result.isi.message}")
        if not result.responses.passed:
            if result.responses.n_attributed_responses == 0:
                result.critical_issues.append("No responses detected!")
            else:
                result.warnings.append(f"Responses: {result.responses.message}")
        if not result.channels.passed:
            result.warnings.append(f"Channels: {result.channels.message}")
        if not result.epochs.passed:
            result.warnings.append(f"Epochs: {result.epochs.message}")

        # Determine overall status
        result.overall_passed = len(result.critical_issues) == 0

    except Exception as e:
        result.critical_issues.append(f"Error processing file: {str(e)}")
        result.overall_passed = False
        logger.error(f"    Error: {e}")

    return result


def run_qc_for_subject(sourcedata_meg: Path, subject: str, date_folder: str,
                       runs: List[str], config: Dict,
                       behav_dir: Optional[Path] = None) -> SubjectQCResult:
    """Run QC for all runs of a subject.

    Args:
        sourcedata_meg: Path to sourcedata/meg.
        subject: Subject ID.
        date_folder: Date folder name.
        runs: List of run numbers to check.
        config: Configuration dictionary.
        behav_dir: Path to behavioral data directory.

    Returns:
        SubjectQCResult with all run results.
    """
    result = SubjectQCResult(subject=subject, date_folder=date_folder)

    # Find all MEG files for this subject
    meg_files = find_meg_files(sourcedata_meg, subject, date_folder)

    if not meg_files:
        logger.warning(f"No MEG files found for sub-{subject}")
        return result

    # Filter to requested runs
    runs_to_check = [r for r in runs if r in meg_files]
    result.n_runs = len(runs_to_check)

    for run in runs_to_check:
        ds_path = meg_files[run]
        run_result = run_qc_for_run(ds_path, subject, run, config, behav_dir)
        result.runs[run] = run_result

        if run_result.overall_passed:
            result.n_passed += 1
        else:
            result.n_failed += 1

    # Compute summary
    result.summary = compute_subject_summary(result)

    return result


def compute_subject_summary(subject_result: SubjectQCResult) -> Dict:
    """Compute summary statistics across runs for a subject."""
    summary = {
        "total_runs": subject_result.n_runs,
        "passed_runs": subject_result.n_passed,
        "failed_runs": subject_result.n_failed,
        "pass_rate": subject_result.n_passed / max(subject_result.n_runs, 1),
        "critical_issues": [],
        "warnings": [],
        # Behavioral
        "mean_response_rate": 0.0,
        "mean_rt_ms": 0.0,
        # ISI consistency
        "mean_isi_ms": 0.0,
        "mean_isi_std_ms": 0.0,
        "mean_isi_cv": 0.0,
        # Data quality
        "mean_pct_bad_channels": 0.0,
        "mean_pct_bad_epochs": 0.0,
        "total_bad_channels": 0,
        "total_bad_epochs": 0,
        # Recording
        "sampling_rate": 0.0,
        "n_meg_channels": 0,
    }

    response_rates = []
    rts = []
    isis = []
    isi_stds = []
    isi_cvs = []
    pct_bad_channels = []
    pct_bad_epochs = []
    total_bad_channels = 0
    total_bad_epochs = 0

    for run_id, run_result in subject_result.runs.items():
        summary["critical_issues"].extend(
            [f"Run {run_id}: {issue}" for issue in run_result.critical_issues]
        )
        summary["warnings"].extend(
            [f"Run {run_id}: {warn}" for warn in run_result.warnings]
        )

        if run_result.responses.response_rate > 0:
            response_rates.append(run_result.responses.response_rate)
        if run_result.responses.mean_rt_ms > 0:
            rts.append(run_result.responses.mean_rt_ms)
        if run_result.isi.mean_isi_ms > 0:
            isis.append(run_result.isi.mean_isi_ms)
            isi_stds.append(run_result.isi.std_isi_ms)
            isi_cvs.append(run_result.isi.cv_isi)

        # Data quality metrics
        pct_bad_channels.append(run_result.channels.pct_bad)
        total_bad_channels += run_result.channels.n_bad_total
        if run_result.epochs.n_epochs_total > 0:
            pct_bad_epochs.append(run_result.epochs.pct_bad_epochs)
            total_bad_epochs += run_result.epochs.n_bad_epochs

        # Take recording params from first run
        if summary["sampling_rate"] == 0:
            summary["sampling_rate"] = run_result.recording.sampling_rate
            summary["n_meg_channels"] = run_result.recording.n_meg_channels

    if response_rates:
        summary["mean_response_rate"] = float(np.mean(response_rates))
    if rts:
        summary["mean_rt_ms"] = float(np.mean(rts))
    if isis:
        summary["mean_isi_ms"] = float(np.mean(isis))
        summary["mean_isi_std_ms"] = float(np.mean(isi_stds))
        summary["mean_isi_cv"] = float(np.mean(isi_cvs))
    if pct_bad_channels:
        summary["mean_pct_bad_channels"] = float(np.mean(pct_bad_channels))
    if pct_bad_epochs:
        summary["mean_pct_bad_epochs"] = float(np.mean(pct_bad_epochs))
    summary["total_bad_channels"] = total_bad_channels
    summary["total_bad_epochs"] = total_bad_epochs

    return summary


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_text_report(results: Dict[str, SubjectQCResult]) -> str:
    """Generate a text summary report."""
    lines = [
        "=" * 80,
        "RAW MEG DATA QUALITY CONTROL REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
    ]

    # Overall summary
    total_subjects = len(results)
    total_runs = sum(r.n_runs for r in results.values())
    passed_runs = sum(r.n_passed for r in results.values())
    failed_runs = sum(r.n_failed for r in results.values())

    lines.extend([
        "OVERALL SUMMARY",
        "-" * 40,
        f"Subjects checked:  {total_subjects}",
        f"Total runs:        {total_runs}",
        f"Passed:            {passed_runs} ({passed_runs/max(total_runs,1)*100:.1f}%)",
        f"Failed:            {failed_runs} ({failed_runs/max(total_runs,1)*100:.1f}%)",
        "",
    ])

    # Recording parameters (from first result)
    for subj_result in results.values():
        for run_result in subj_result.runs.values():
            lines.extend([
                "RECORDING PARAMETERS",
                "-" * 40,
                f"Sampling rate:     {run_result.recording.sampling_rate} Hz",
                f"MEG channels:      {run_result.recording.n_meg_channels}",
                f"Reference chans:   {run_result.recording.n_ref_channels}",
                f"EEG channels:      {run_result.recording.n_eeg_channels}",
                f"Stim channels:     {run_result.recording.n_stim_channels}",
                f"Line frequency:    {run_result.recording.line_freq} Hz",
                f"Run duration:      {run_result.recording.duration_sec:.1f} sec",
                "",
            ])
            break
        break

    # ISI consistency summary
    all_isis = []
    all_isi_stds = []
    all_isi_cvs = []
    for subj_result in results.values():
        for run_result in subj_result.runs.values():
            if run_result.isi.mean_isi_ms > 0:
                all_isis.append(run_result.isi.mean_isi_ms)
                all_isi_stds.append(run_result.isi.std_isi_ms)
                all_isi_cvs.append(run_result.isi.cv_isi)

    if all_isis:
        lines.extend([
            "ISI CONSISTENCY SUMMARY",
            "-" * 40,
            f"Mean ISI:          {np.mean(all_isis):.1f} ms",
            f"ISI range:         [{np.min(all_isis):.0f}, {np.max(all_isis):.0f}] ms",
            f"Mean within-run std: {np.mean(all_isi_stds):.2f} ms",
            f"Mean CV:           {np.mean(all_isi_cvs):.5f} (lower = more consistent)",
            "",
        ])

    # Data quality summary
    all_pct_bad_ch = []
    all_pct_bad_ep = []
    total_bad_ch = 0
    total_bad_ep = 0
    for subj_result in results.values():
        for run_result in subj_result.runs.values():
            all_pct_bad_ch.append(run_result.channels.pct_bad)
            total_bad_ch += run_result.channels.n_bad_total
            if run_result.epochs.n_epochs_total > 0:
                all_pct_bad_ep.append(run_result.epochs.pct_bad_epochs)
                total_bad_ep += run_result.epochs.n_bad_epochs

    lines.extend([
        "DATA QUALITY SUMMARY",
        "-" * 40,
        f"Bad channels:      {np.mean(all_pct_bad_ch):.1f}% avg ({total_bad_ch} total across all runs)",
        f"Bad epochs:        {np.mean(all_pct_bad_ep):.1f}% avg ({total_bad_ep} total across all runs)",
        f"  (threshold: 4000 fT peak-to-peak)",
        "",
    ])

    # Critical issues
    all_critical = []
    for subj_id, subj_result in results.items():
        for issue in subj_result.summary.get("critical_issues", []):
            all_critical.append(f"sub-{subj_id}: {issue}")

    if all_critical:
        lines.extend([
            "CRITICAL ISSUES",
            "-" * 40,
        ])
        lines.extend(all_critical[:20])  # Limit to first 20
        if len(all_critical) > 20:
            lines.append(f"... and {len(all_critical) - 20} more issues")
        lines.append("")

    # Per-subject summary
    lines.extend([
        "PER-SUBJECT SUMMARY",
        "-" * 40,
    ])

    # Header for per-subject table
    lines.append(f"{'Subj':<8} {'Runs':^7} {'Resp%':>6} {'RT':>5} {'ISI':>5} {'ISI_std':>7} {'BadCh%':>6} {'BadEp%':>6}")
    lines.append("-" * 60)

    for subj_id, subj_result in sorted(results.items()):
        status = "✓" if subj_result.n_failed == 0 else "✗"
        resp_rate = subj_result.summary.get('mean_response_rate', 0) * 100
        rt = subj_result.summary.get('mean_rt_ms', 0)
        isi = subj_result.summary.get('mean_isi_ms', 0)
        isi_std = subj_result.summary.get('mean_isi_std_ms', 0)
        bad_ch = subj_result.summary.get('mean_pct_bad_channels', 0)
        bad_ep = subj_result.summary.get('mean_pct_bad_epochs', 0)

        lines.append(
            f"{status} {subj_id:<5} {subj_result.n_passed}/{subj_result.n_runs:>3}   "
            f"{resp_rate:5.1f}% {rt:5.0f} {isi:5.0f} {isi_std:6.2f}ms {bad_ch:5.1f}% {bad_ep:5.1f}%"
        )

    lines.extend(["", "=" * 80])

    return "\n".join(lines)


def save_report(results: Dict[str, SubjectQCResult], output_dir: Path):
    """Save QC report to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON report
    json_file = output_dir / f"qc_report_{timestamp}.json"
    json_data = {
        subj_id: {
            "subject": subj_result.subject,
            "date_folder": subj_result.date_folder,
            "n_runs": subj_result.n_runs,
            "n_passed": subj_result.n_passed,
            "n_failed": subj_result.n_failed,
            "summary": subj_result.summary,
            "runs": {
                run_id: asdict(run_result)
                for run_id, run_result in subj_result.runs.items()
            }
        }
        for subj_id, subj_result in results.items()
    }

    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"Saved JSON report: {json_file}")

    # Save text report
    text_file = output_dir / f"qc_report_{timestamp}.txt"
    text_report = generate_text_report(results)
    with open(text_file, "w") as f:
        f.write(text_report)
    logger.info(f"Saved text report: {text_file}")

    # Print summary to console
    print(text_report)


# ==============================================================================
# CLI
# ==============================================================================

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration."""
    if config_path is None:
        config_path = Path("config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run quality control checks on raw MEG sourcedata"
    )
    parser.add_argument(
        "--subject", "-s",
        type=str,
        help="Subject ID to check (e.g., '04')"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Check all subjects (default if no --subject specified)"
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        help="Specific runs to check (default: task runs 02-07)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="reports/qc",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(Path(args.config))

    # Get sourcedata paths
    data_root = Path(config["paths"]["data_root"])
    sourcedata_meg = data_root / "sourcedata" / "meg"
    sourcedata_behav = data_root / "sourcedata" / "behav"

    if not sourcedata_meg.exists():
        logger.error(f"Sourcedata MEG directory not found: {sourcedata_meg}")
        return 1

    if not sourcedata_behav.exists():
        logger.warning(f"Sourcedata behavioral directory not found: {sourcedata_behav}")
        logger.warning("Response rates will use raw trigger counts (may be inflated)")

    # Get subject-date mapping
    subj_date_map = get_subject_date_mapping(sourcedata_meg)

    # Get subjects
    if args.subject:
        subjects = [args.subject]
    else:
        # Default: all subjects
        subjects = config["bids"]["subjects"]

    # Get runs (default: task runs 02-07)
    runs = args.runs or ["02", "03", "04", "05", "06", "07"]

    logger.info(f"Sourcedata path: {sourcedata_meg}")
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Runs: {runs}")

    # Run QC
    results = {}
    for subject in subjects:
        if subject not in subj_date_map:
            logger.warning(f"No data found for sub-{subject}")
            continue

        logger.info(f"Processing subject {subject}...")
        date_folder = subj_date_map[subject]
        results[subject] = run_qc_for_subject(
            sourcedata_meg, subject, date_folder, runs, config,
            behav_dir=sourcedata_behav if sourcedata_behav.exists() else None
        )

    if not results:
        logger.error("No subjects processed!")
        return 1

    # Save reports
    output_dir = Path(args.output_dir)
    save_report(results, output_dir)

    # Return exit code based on results
    total_failed = sum(r.n_failed for r in results.values())
    if total_failed > 0:
        logger.warning(f"\n{total_failed} runs failed QC checks!")
        return 1
    else:
        logger.info("\nAll QC checks passed!")
        return 0


if __name__ == "__main__":
    exit(main())
