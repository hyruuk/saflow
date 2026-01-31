"""Data quality control checks for BIDS MEG data.

This module performs comprehensive QC checks on BIDS data to assess
data quality and detect issues before preprocessing:

- ISI (Inter-Stimulus Interval) validation and outlier detection
- Response detection and behavioral consistency
- Channel quality (flat, noisy, saturated channels)
- Event/trigger validation
- Data integrity checks (sampling rate, duration)
- Head motion (if HPI available)

Usage:
    python -m code.qc.check_qc --subject 04
    python -m code.qc.check_qc --all --output-dir reports/qc
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
class ISICheck:
    """Results of Inter-Stimulus Interval check."""
    expected_isi_ms: float = 800.0  # gradCPT default
    mean_isi_ms: float = 0.0
    std_isi_ms: float = 0.0
    min_isi_ms: float = 0.0
    max_isi_ms: float = 0.0
    n_outliers: int = 0  # ISIs outside expected range
    outlier_threshold_ms: float = 100.0  # ±100ms tolerance
    outlier_indices: List[int] = field(default_factory=list)
    passed: bool = True
    message: str = ""


@dataclass
class ResponseCheck:
    """Results of response detection check."""
    n_stimuli: int = 0
    n_responses: int = 0
    response_rate: float = 0.0
    n_commission_errors: int = 0  # Responses to non-targets
    n_omission_errors: int = 0    # No response to targets
    n_correct_commissions: int = 0
    n_correct_omissions: int = 0
    mean_rt_ms: float = 0.0
    std_rt_ms: float = 0.0
    no_responses: bool = False  # Critical: run has zero responses
    passed: bool = True
    message: str = ""


@dataclass
class ChannelCheck:
    """Results of channel quality check."""
    n_channels: int = 0
    n_meg_channels: int = 0
    n_flat_channels: int = 0
    n_noisy_channels: int = 0
    n_saturated_channels: int = 0
    flat_channels: List[str] = field(default_factory=list)
    noisy_channels: List[str] = field(default_factory=list)
    saturated_channels: List[str] = field(default_factory=list)
    flat_threshold: float = 1e-15  # Near-zero variance
    noise_threshold_std: float = 5.0  # Z-score threshold
    passed: bool = True
    message: str = ""


@dataclass
class EventCheck:
    """Results of event/trigger validation."""
    n_events_total: int = 0
    n_stimulus_events: int = 0
    n_response_events: int = 0
    n_other_events: int = 0
    expected_stimuli_per_run: int = 525  # gradCPT default
    event_types: Dict[str, int] = field(default_factory=dict)
    missing_events: bool = False
    duplicate_events: bool = False
    timing_gaps: List[Tuple[int, float]] = field(default_factory=list)  # (index, gap_sec)
    passed: bool = True
    message: str = ""


@dataclass
class DataIntegrityCheck:
    """Results of data integrity check."""
    sampling_rate: float = 0.0
    expected_sampling_rate: float = 1200.0  # CTF default
    duration_sec: float = 0.0
    expected_duration_sec: float = 450.0  # ~7.5 min per run
    n_samples: int = 0
    has_gaps: bool = False
    gap_locations: List[int] = field(default_factory=list)
    passed: bool = True
    message: str = ""


@dataclass
class MotionCheck:
    """Results of head motion check (if HPI available)."""
    hpi_available: bool = False
    max_movement_mm: float = 0.0
    mean_movement_mm: float = 0.0
    n_large_movements: int = 0  # Movements > threshold
    movement_threshold_mm: float = 5.0
    excessive_motion: bool = False
    passed: bool = True
    message: str = ""


@dataclass
class RunQCResult:
    """Complete QC results for a single run."""
    subject: str = ""
    run: str = ""
    timestamp: str = ""
    isi: ISICheck = field(default_factory=ISICheck)
    responses: ResponseCheck = field(default_factory=ResponseCheck)
    channels: ChannelCheck = field(default_factory=ChannelCheck)
    events: EventCheck = field(default_factory=EventCheck)
    data_integrity: DataIntegrityCheck = field(default_factory=DataIntegrityCheck)
    motion: MotionCheck = field(default_factory=MotionCheck)
    overall_passed: bool = True
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SubjectQCResult:
    """Complete QC results for a subject across all runs."""
    subject: str = ""
    n_runs: int = 0
    n_passed: int = 0
    n_failed: int = 0
    runs: Dict[str, RunQCResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# QC Check Functions
# ==============================================================================

def check_isi(events: np.ndarray, sfreq: float, expected_isi_ms: float = 800.0) -> ISICheck:
    """Check Inter-Stimulus Interval consistency.

    Args:
        events: MNE events array (n_events, 3).
        sfreq: Sampling frequency in Hz.
        expected_isi_ms: Expected ISI in milliseconds.

    Returns:
        ISICheck with results.
    """
    result = ISICheck(expected_isi_ms=expected_isi_ms)

    # Get stimulus events (assuming event_id for stimuli)
    # In gradCPT, stimuli have specific event codes
    stim_mask = events[:, 2] > 0  # Adjust based on actual event codes
    stim_samples = events[stim_mask, 0]

    if len(stim_samples) < 2:
        result.passed = False
        result.message = "Insufficient stimulus events to compute ISI"
        return result

    # Compute ISIs
    isis_samples = np.diff(stim_samples)
    isis_ms = (isis_samples / sfreq) * 1000

    result.mean_isi_ms = float(np.mean(isis_ms))
    result.std_isi_ms = float(np.std(isis_ms))
    result.min_isi_ms = float(np.min(isis_ms))
    result.max_isi_ms = float(np.max(isis_ms))

    # Find outliers
    lower_bound = expected_isi_ms - result.outlier_threshold_ms
    upper_bound = expected_isi_ms + result.outlier_threshold_ms
    outlier_mask = (isis_ms < lower_bound) | (isis_ms > upper_bound)
    result.n_outliers = int(np.sum(outlier_mask))
    result.outlier_indices = np.where(outlier_mask)[0].tolist()

    # Determine pass/fail
    if result.n_outliers > len(isis_ms) * 0.05:  # More than 5% outliers
        result.passed = False
        result.message = f"Excessive ISI outliers: {result.n_outliers}/{len(isis_ms)} ({result.n_outliers/len(isis_ms)*100:.1f}%)"
    elif abs(result.mean_isi_ms - expected_isi_ms) > 50:  # Mean off by >50ms
        result.passed = False
        result.message = f"Mean ISI ({result.mean_isi_ms:.1f}ms) deviates from expected ({expected_isi_ms}ms)"
    else:
        result.message = f"ISI OK: {result.mean_isi_ms:.1f} ± {result.std_isi_ms:.1f}ms"

    return result


def check_responses(events: np.ndarray, sfreq: float,
                    stim_event_ids: List[int] = None,
                    response_event_ids: List[int] = None) -> ResponseCheck:
    """Check response patterns and detect runs without responses.

    Args:
        events: MNE events array.
        sfreq: Sampling frequency.
        stim_event_ids: Event IDs for stimuli.
        response_event_ids: Event IDs for responses.

    Returns:
        ResponseCheck with results.
    """
    result = ResponseCheck()

    # Default event IDs (adjust based on actual gradCPT setup)
    if stim_event_ids is None:
        stim_event_ids = [1, 2]  # Example: 1=frequent, 2=rare
    if response_event_ids is None:
        response_event_ids = [256]  # Example: button press

    # Count events
    stim_mask = np.isin(events[:, 2], stim_event_ids)
    resp_mask = np.isin(events[:, 2], response_event_ids)

    result.n_stimuli = int(np.sum(stim_mask))
    result.n_responses = int(np.sum(resp_mask))

    if result.n_stimuli > 0:
        result.response_rate = result.n_responses / result.n_stimuli

    # Critical check: no responses at all
    if result.n_responses == 0:
        result.no_responses = True
        result.passed = False
        result.message = "CRITICAL: No responses detected in this run!"
        return result

    # Compute RT for responses following stimuli
    stim_times = events[stim_mask, 0]
    resp_times = events[resp_mask, 0]

    rts = []
    for resp_time in resp_times:
        # Find preceding stimulus
        preceding_stim = stim_times[stim_times < resp_time]
        if len(preceding_stim) > 0:
            rt_samples = resp_time - preceding_stim[-1]
            rt_ms = (rt_samples / sfreq) * 1000
            if 100 < rt_ms < 2000:  # Valid RT range
                rts.append(rt_ms)

    if rts:
        result.mean_rt_ms = float(np.mean(rts))
        result.std_rt_ms = float(np.std(rts))

    # Warnings
    if result.response_rate < 0.5:
        result.passed = False
        result.message = f"Low response rate: {result.response_rate:.1%}"
    elif result.response_rate > 0.95:
        result.warnings = ["Very high response rate - possible button stuck?"]
        result.message = f"High response rate: {result.response_rate:.1%}"
    else:
        result.message = f"Response rate: {result.response_rate:.1%}, RT: {result.mean_rt_ms:.0f}±{result.std_rt_ms:.0f}ms"

    return result


def check_channels(raw, flat_threshold: float = 1e-15,
                   noise_threshold_std: float = 5.0) -> ChannelCheck:
    """Check channel quality for flat, noisy, or saturated channels.

    Args:
        raw: MNE Raw object.
        flat_threshold: Variance threshold for flat channels.
        noise_threshold_std: Z-score threshold for noisy channels.

    Returns:
        ChannelCheck with results.
    """
    result = ChannelCheck(
        flat_threshold=flat_threshold,
        noise_threshold_std=noise_threshold_std,
    )

    # Get MEG channels
    meg_picks = raw.ch_names  # Adjust to pick only MEG
    try:
        from mne import pick_types
        meg_idx = pick_types(raw.info, meg=True, ref_meg=False, exclude='bads')
        meg_picks = [raw.ch_names[i] for i in meg_idx]
    except:
        pass

    result.n_channels = len(raw.ch_names)
    result.n_meg_channels = len(meg_picks)

    # Get data for analysis (sample a portion to save memory)
    try:
        # Sample 10 seconds from the middle
        start_sample = raw.n_times // 2 - int(5 * raw.info['sfreq'])
        stop_sample = start_sample + int(10 * raw.info['sfreq'])
        data = raw.get_data(picks=meg_picks, start=max(0, start_sample),
                           stop=min(raw.n_times, stop_sample))
    except:
        # Fallback: get all data
        data = raw.get_data(picks=meg_picks)

    # Compute per-channel statistics
    variances = np.var(data, axis=1)
    stds = np.std(data, axis=1)
    maxes = np.max(np.abs(data), axis=1)

    # Detect flat channels
    flat_mask = variances < flat_threshold
    result.n_flat_channels = int(np.sum(flat_mask))
    result.flat_channels = [meg_picks[i] for i in np.where(flat_mask)[0]]

    # Detect noisy channels (high variance outliers)
    mean_std = np.mean(stds)
    std_of_stds = np.std(stds)
    if std_of_stds > 0:
        z_scores = (stds - mean_std) / std_of_stds
        noisy_mask = z_scores > noise_threshold_std
        result.n_noisy_channels = int(np.sum(noisy_mask))
        result.noisy_channels = [meg_picks[i] for i in np.where(noisy_mask)[0]]

    # Detect saturated channels (near max ADC value)
    # CTF systems typically have max around 1e-11 to 1e-10 T
    saturation_threshold = np.percentile(maxes, 99) * 0.99
    saturated_mask = maxes > saturation_threshold
    result.n_saturated_channels = int(np.sum(saturated_mask))
    result.saturated_channels = [meg_picks[i] for i in np.where(saturated_mask)[0]]

    # Determine pass/fail
    bad_channel_pct = (result.n_flat_channels + result.n_noisy_channels) / max(result.n_meg_channels, 1)
    if bad_channel_pct > 0.1:  # More than 10% bad
        result.passed = False
        result.message = f"High bad channel count: {result.n_flat_channels} flat, {result.n_noisy_channels} noisy"
    else:
        result.message = f"Channels OK: {result.n_flat_channels} flat, {result.n_noisy_channels} noisy out of {result.n_meg_channels}"

    return result


def check_events(events: np.ndarray, expected_stimuli: int = 525) -> EventCheck:
    """Validate events/triggers.

    Args:
        events: MNE events array.
        expected_stimuli: Expected number of stimulus events per run.

    Returns:
        EventCheck with results.
    """
    result = EventCheck(expected_stimuli_per_run=expected_stimuli)

    result.n_events_total = len(events)

    # Count event types
    unique_events, counts = np.unique(events[:, 2], return_counts=True)
    result.event_types = {str(e): int(c) for e, c in zip(unique_events, counts)}

    # Estimate stimulus count (events with certain IDs)
    # This is task-specific; adjust based on actual event codes
    stim_ids = [e for e in unique_events if e > 0 and e < 100]  # Example
    result.n_stimulus_events = sum(result.event_types.get(str(e), 0) for e in stim_ids)

    # Response events (typically higher values)
    resp_ids = [e for e in unique_events if e >= 100]
    result.n_response_events = sum(result.event_types.get(str(e), 0) for e in resp_ids)

    result.n_other_events = result.n_events_total - result.n_stimulus_events - result.n_response_events

    # Check for missing events
    if result.n_stimulus_events < expected_stimuli * 0.9:
        result.missing_events = True
        result.passed = False
        result.message = f"Missing stimuli: found {result.n_stimulus_events}, expected ~{expected_stimuli}"

    # Check for duplicate events (same sample)
    if len(events) > 0:
        sample_diffs = np.diff(events[:, 0])
        n_duplicates = np.sum(sample_diffs == 0)
        if n_duplicates > 0:
            result.duplicate_events = True
            result.warnings = [f"{n_duplicates} duplicate events at same sample"]

    if result.passed:
        result.message = f"Events OK: {result.n_stimulus_events} stimuli, {result.n_response_events} responses"

    return result


def check_data_integrity(raw, expected_sfreq: float = 1200.0,
                         expected_duration_sec: float = 450.0) -> DataIntegrityCheck:
    """Check data integrity (sampling rate, duration, gaps).

    Args:
        raw: MNE Raw object.
        expected_sfreq: Expected sampling frequency.
        expected_duration_sec: Expected run duration in seconds.

    Returns:
        DataIntegrityCheck with results.
    """
    result = DataIntegrityCheck(
        expected_sampling_rate=expected_sfreq,
        expected_duration_sec=expected_duration_sec,
    )

    result.sampling_rate = raw.info['sfreq']
    result.n_samples = raw.n_times
    result.duration_sec = raw.n_times / raw.info['sfreq']

    # Check sampling rate
    if abs(result.sampling_rate - expected_sfreq) > 1:
        result.passed = False
        result.message = f"Unexpected sampling rate: {result.sampling_rate}Hz (expected {expected_sfreq}Hz)"
        return result

    # Check duration
    duration_diff = abs(result.duration_sec - expected_duration_sec)
    if duration_diff > 60:  # More than 1 minute off
        result.warnings = [f"Duration ({result.duration_sec:.0f}s) differs from expected ({expected_duration_sec}s)"]

    # Check for gaps (would need annotations or specific markers)
    # This is a simplified check
    if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
        gap_annots = [a for a in raw.annotations if 'gap' in a['description'].lower() or 'bad' in a['description'].lower()]
        if gap_annots:
            result.has_gaps = True
            result.warnings = [f"Found {len(gap_annots)} gap/bad annotations"]

    if result.passed:
        result.message = f"Data OK: {result.sampling_rate}Hz, {result.duration_sec:.0f}s"

    return result


def check_motion(raw) -> MotionCheck:
    """Check head motion using HPI coils (if available).

    Args:
        raw: MNE Raw object.

    Returns:
        MotionCheck with results.
    """
    result = MotionCheck()

    # Check if HPI channels exist
    try:
        from mne import pick_types
        hpi_picks = pick_types(raw.info, meg=False, ref_meg=False,
                               chpi=True, exclude=[])
        if len(hpi_picks) > 0:
            result.hpi_available = True
        else:
            result.message = "No HPI data available"
            return result
    except:
        result.message = "Could not check for HPI data"
        return result

    # If HPI available, try to compute head position
    try:
        from mne.chpi import compute_head_pos, read_head_pos
        # This requires continuous HPI recording
        # For CTF data, head position may be in different format
        result.message = "HPI available but motion extraction not implemented"
    except Exception as e:
        result.message = f"Could not compute head motion: {e}"

    return result


# ==============================================================================
# Main QC Runner
# ==============================================================================

def run_qc_for_run(bids_path: Path, subject: str, run: str,
                   config: Dict) -> RunQCResult:
    """Run all QC checks for a single run.

    Args:
        bids_path: Path to BIDS root.
        subject: Subject ID.
        run: Run number.
        config: Configuration dictionary.

    Returns:
        RunQCResult with all check results.
    """
    import mne

    result = RunQCResult(
        subject=subject,
        run=run,
        timestamp=datetime.now().isoformat(),
    )

    # Construct file path
    meg_path = bids_path / f"sub-{subject}" / "ses-recording" / "meg"
    fname_pattern = f"sub-{subject}_ses-recording_task-gradCPT_run-{run}_meg.ds"
    meg_file = meg_path / fname_pattern

    if not meg_file.exists():
        result.critical_issues.append(f"MEG file not found: {meg_file}")
        result.overall_passed = False
        return result

    logger.info(f"Running QC for sub-{subject} run-{run}")

    try:
        # Load raw data
        raw = mne.io.read_raw_ctf(str(meg_file), preload=False, verbose=False)

        # Get events
        events = mne.find_events(raw, stim_channel='UPPT001', verbose=False)

        # Run checks
        logger.info("  Checking ISI...")
        result.isi = check_isi(events, raw.info['sfreq'])
        if not result.isi.passed:
            result.warnings.append(f"ISI: {result.isi.message}")

        logger.info("  Checking responses...")
        result.responses = check_responses(events, raw.info['sfreq'])
        if result.responses.no_responses:
            result.critical_issues.append("No responses in run!")
        elif not result.responses.passed:
            result.warnings.append(f"Responses: {result.responses.message}")

        logger.info("  Checking channels...")
        raw.load_data()
        result.channels = check_channels(raw)
        if not result.channels.passed:
            result.warnings.append(f"Channels: {result.channels.message}")

        logger.info("  Checking events...")
        result.events = check_events(events)
        if not result.events.passed:
            result.warnings.append(f"Events: {result.events.message}")

        logger.info("  Checking data integrity...")
        result.data_integrity = check_data_integrity(raw)
        if not result.data_integrity.passed:
            result.critical_issues.append(f"Data: {result.data_integrity.message}")

        logger.info("  Checking motion...")
        result.motion = check_motion(raw)

        # Determine overall status
        result.overall_passed = (
            len(result.critical_issues) == 0 and
            result.isi.passed and
            result.responses.passed and
            result.channels.passed and
            result.events.passed and
            result.data_integrity.passed
        )

    except Exception as e:
        result.critical_issues.append(f"Error loading/processing data: {str(e)}")
        result.overall_passed = False
        logger.error(f"  Error: {e}")

    return result


def run_qc_for_subject(bids_path: Path, subject: str, runs: List[str],
                       config: Dict) -> SubjectQCResult:
    """Run QC for all runs of a subject.

    Args:
        bids_path: Path to BIDS root.
        subject: Subject ID.
        runs: List of run numbers.
        config: Configuration dictionary.

    Returns:
        SubjectQCResult with all run results.
    """
    result = SubjectQCResult(subject=subject, n_runs=len(runs))

    for run in runs:
        run_result = run_qc_for_run(bids_path, subject, run, config)
        result.runs[run] = run_result
        if run_result.overall_passed:
            result.n_passed += 1
        else:
            result.n_failed += 1

    # Compute summary statistics across runs
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
        "mean_response_rate": 0.0,
        "mean_rt_ms": 0.0,
        "total_bad_channels": set(),
    }

    response_rates = []
    rts = []

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

        summary["total_bad_channels"].update(run_result.channels.flat_channels)
        summary["total_bad_channels"].update(run_result.channels.noisy_channels)

    if response_rates:
        summary["mean_response_rate"] = float(np.mean(response_rates))
    if rts:
        summary["mean_rt_ms"] = float(np.mean(rts))

    summary["total_bad_channels"] = list(summary["total_bad_channels"])

    return summary


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_text_report(results: Dict[str, SubjectQCResult]) -> str:
    """Generate a text summary report."""
    lines = [
        "=" * 80,
        "BIDS DATA QUALITY CONTROL REPORT",
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
        lines.extend(all_critical)
        lines.append("")

    # Per-subject summary
    lines.extend([
        "PER-SUBJECT SUMMARY",
        "-" * 40,
    ])

    for subj_id, subj_result in sorted(results.items()):
        status = "✓" if subj_result.n_failed == 0 else "✗"
        lines.append(
            f"{status} sub-{subj_id}: {subj_result.n_passed}/{subj_result.n_runs} runs passed, "
            f"resp_rate={subj_result.summary.get('mean_response_rate', 0)*100:.1f}%, "
            f"RT={subj_result.summary.get('mean_rt_ms', 0):.0f}ms"
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
        description="Run quality control checks on BIDS MEG data"
    )
    parser.add_argument(
        "--subject", "-s",
        type=str,
        help="Subject ID to check (e.g., '04')"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Check all subjects"
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        help="Specific runs to check (default: all task runs)"
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

    # Get BIDS path
    data_root = Path(config["paths"]["data_root"])
    bids_path = data_root / "bids"

    # Get subjects
    if args.all:
        subjects = config["bids"]["subjects"]
    elif args.subject:
        subjects = [args.subject]
    else:
        parser.error("Must specify --subject or --all")
        return

    # Get runs
    runs = args.runs or config["bids"]["task_runs"]

    logger.info(f"BIDS path: {bids_path}")
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Runs: {runs}")

    # Run QC
    results = {}
    for subject in subjects:
        logger.info(f"\nProcessing subject {subject}...")
        results[subject] = run_qc_for_subject(bids_path, subject, runs, config)

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
