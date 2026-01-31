"""Stage 1: Compute Welch PSD from continuous data (UNIFIED: sensor/source/atlas).

This script implements the first stage of the PSD→FOOOF workflow:
1. Load continuous data (sensor, source, or atlas space)
2. Segment into trials based on events
3. Compute Welch PSD for each trial
4. Save per-trial PSDs with event metadata

**Unified Architecture**: This script works across all analysis spaces using the same
code with different inputs via the loader system (see code/features/loaders.py).

**Workflow**:
  Continuous data → Segment → Welch PSD → Save per-trial PSDs

Usage:
    # Sensor-level
    python -m code.features.compute_welch_psd --subject 04 --run 02 --space sensor

    # Source-level
    python -m code.features.compute_welch_psd --subject 04 --run 02 --space source

    # Atlas-level (ROI-based)
    python -m code.features.compute_welch_psd --subject 04 --run 02 --space atlas

Author: Claude (Anthropic)
Date: 2026-01-30
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import mne
import numpy as np
import pandas as pd

from code.features.loaders import load_data
from code.utils.config import load_config
from code.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def segment_spatial_temporal_data(
    data: np.ndarray,
    events_df: pd.DataFrame,
    sfreq: float,
    tmin: float = 0.426,
    tmax: float = 1.278,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Segment continuous data based on events.

    Works for any analysis space (sensor/source/atlas).

    Parameters
    ----------
    data : np.ndarray
        Continuous data array, shape (n_spatial, n_times)
    events_df : pd.DataFrame
        Events dataframe with 'onset', 'trial_type' columns
    sfreq : float
        Sampling frequency in Hz
    tmin : float
        Start time of epoch relative to event onset, in seconds
    tmax : float
        End time of epoch relative to event onset, in seconds

    Returns
    -------
    segmented_array : np.ndarray
        Segmented data, shape (n_trials, n_spatial, n_samples_per_trial)
    trial_metadata : pd.DataFrame
        Metadata for each segmented trial
    """
    logger.info(f"Segmenting continuous data (tmin={tmin}, tmax={tmax})")

    # Compute time samples
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)

    # Filter to stimulus trials only
    stim_trials = events_df[events_df['trial_type'].isin(['Freq', 'Rare'])].copy()
    logger.info(f"Found {len(stim_trials)} stimulus trials")

    segmented_array = []
    trial_metadata_list = []

    for idx, trial in stim_trials.iterrows():
        # Get event onset in samples
        onset_sample = int(trial['onset'] * sfreq)

        # Check boundaries
        if onset_sample + tmax_samples >= data.shape[1]:
            logger.debug(f"Skipping trial {idx}: extends beyond data")
            continue
        if onset_sample + tmin_samples < 0:
            logger.debug(f"Skipping trial {idx}: starts before data")
            continue

        # Extract segment
        segment = data[:, onset_sample + tmin_samples : onset_sample + tmax_samples]
        segmented_array.append(segment)
        trial_metadata_list.append(trial)

    segmented_array = np.array(segmented_array)
    trial_metadata = pd.DataFrame(trial_metadata_list).reset_index(drop=True)

    logger.info(f"Segmented {len(segmented_array)} trials from continuous data")
    logger.debug(f"Segmented array shape: {segmented_array.shape}")

    return segmented_array, trial_metadata


def compute_welch_psd_from_continuous(
    data: np.ndarray,
    events_df: pd.DataFrame,
    sfreq: float,
    n_fft: int = 1022,
    n_overlap: int = 959,
    tmin: float = 0.426,
    tmax: float = 1.278,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute Welch PSD from continuous data (space-agnostic).

    Parameters
    ----------
    data : np.ndarray
        Continuous data, shape (n_spatial, n_times)
    events_df : pd.DataFrame
        Events dataframe with trial metadata
    sfreq : float
        Sampling frequency
    n_fft : int
        FFT length for Welch's method
    n_overlap : int
        Number of overlapping samples
    tmin : float
        Epoch start time relative to event
    tmax : float
        Epoch end time relative to event
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    psds : np.ndarray
        Welch PSDs, shape (n_trials, n_spatial, n_freqs)
    freqs : np.ndarray
        Frequency bins
    trial_metadata : pd.DataFrame
        Metadata for each trial
    """
    logger.info("Computing Welch PSD from continuous data")

    # Segment continuous data into trials
    segmented_data, trial_metadata = segment_spatial_temporal_data(
        data=data,
        events_df=events_df,
        sfreq=sfreq,
        tmin=tmin,
        tmax=tmax,
    )

    logger.info(f"Segmented data shape: {segmented_data.shape}")
    logger.info(f"Computing Welch PSD (n_fft={n_fft}, n_overlap={n_overlap})...")

    # Compute Welch PSD for all trials
    # Input: (n_trials, n_spatial, n_times)
    # Output: (n_trials, n_spatial, n_freqs)
    psds, freqs = mne.time_frequency.psd_array_welch(
        segmented_data,
        sfreq=sfreq,
        n_fft=n_fft,
        n_overlap=n_overlap,
        average="median",
        n_jobs=n_jobs,
        verbose=False,
    )

    logger.info(f"Welch PSD shape: {psds.shape}")
    logger.info(f"Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz ({len(freqs)} bins)")

    return psds, freqs, trial_metadata


def save_welch_psds(
    psds: np.ndarray,
    freqs: np.ndarray,
    trial_metadata: pd.DataFrame,
    output_path: Path,
    subject: str,
    run: str,
    space: str,
    config: Dict,
    n_fft: int,
    n_overlap: int,
):
    """Save Welch PSDs to disk.

    Parameters
    ----------
    psds : np.ndarray
        Welch PSDs, shape (n_trials, n_spatial, n_freqs)
    freqs : np.ndarray
        Frequency bins
    trial_metadata : pd.DataFrame
        Trial metadata from events.tsv
    output_path : Path
        Output directory
    subject : str
        Subject ID
    run : str
        Run ID
    space : str
        Analysis space (sensor/source/atlas)
    config : Dict
        Configuration dictionary
    n_fft : int
        FFT length used
    n_overlap : int
        Overlap used
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Build output filename
    fname = (
        f"sub-{subject}_"
        f"ses-recording_"
        f"task-gradCPT_"
        f"run-{run}_"
        f"space-{space}_"
        f"desc-welch_"
        f"psds.npz"
    )

    output_file = output_path / fname

    # Get git hash for provenance
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    # Convert trial_metadata to dict format for numpy savez
    metadata_dict = trial_metadata.to_dict('list')

    # Save as compressed numpy array
    np.savez_compressed(
        output_file,
        psds=psds,
        freqs=freqs,
        trial_metadata=metadata_dict,
    )

    logger.info(f"Saved Welch PSDs to {output_file}")

    # Save parameters as JSON sidecar
    metadata_file = output_path / fname.replace(".npz", "_params.json")

    params = {
        "subject": subject,
        "run": run,
        "space": space,
        "method": "welch",
        "n_fft": n_fft,
        "n_overlap": n_overlap,
        "n_trials": psds.shape[0],
        "n_spatial": psds.shape[1],
        "n_freqs": psds.shape[2],
        "freq_range": [float(freqs[0]), float(freqs[-1])],
        "git_hash": git_hash,
    }

    with open(metadata_file, "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Welch PSD from continuous data (UNIFIED: sensor/source/atlas)"
    )

    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject ID",
    )

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run ID",
    )

    parser.add_argument(
        "--space",
        type=str,
        choices=["sensor", "source", "atlas"],
        required=True,
        help="Analysis space: sensor, source, or atlas",
    )

    parser.add_argument(
        "--bids-root",
        type=Path,
        default=None,
        help="BIDS dataset root (default: from config)",
    )

    parser.add_argument(
        "--n-fft",
        type=int,
        default=1022,
        help="FFT length for Welch's method (default: 1022)",
    )

    parser.add_argument(
        "--n-overlap",
        type=int,
        default=959,
        help="Number of overlapping samples (default: 959)",
    )

    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Epoch start time relative to event in seconds (default: from config)",
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Epoch end time relative to event in seconds (default: from config)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip if output already exists (default: True)",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Reprocess even if output exists",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1, use all CPUs)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_dir = Path("logs") / "features"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level=args.log_level, log_dir=log_dir, log_name="compute_welch_psd")

    logger.info("=" * 80)
    logger.info(f"Stage 1: Welch PSD Computation ({args.space.upper()}-LEVEL)")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Run: {args.run}")
    logger.info(f"Space: {args.space}")

    # Load configuration
    config = load_config()

    # Get epoch timing from config if not specified on CLI
    tmin = args.tmin if args.tmin is not None else config["analysis"]["epochs"]["tmin"]
    tmax = args.tmax if args.tmax is not None else config["analysis"]["epochs"]["tmax"]
    logger.info(f"Epoch timing: tmin={tmin}s, tmax={tmax}s")

    # Determine BIDS root
    bids_root = (
        Path(args.bids_root)
        if args.bids_root
        else Path(config["paths"]["data_root"]) / "bids"
    )
    derivatives_root = bids_root / "derivatives"

    # Build output path (simplified naming, no bounds or filter suffix)
    output_root = derivatives_root / f"welch_psds_{args.space}" / f"sub-{args.subject}" / "meg"

    # Check if output already exists
    output_file = (
        output_root
        / f"sub-{args.subject}_ses-recording_task-gradCPT_run-{args.run}_space-{args.space}_desc-welch_psds.npz"
    )

    if args.skip_existing and output_file.exists():
        logger.info(f"Output already exists: {output_file}")
        logger.info("Skipping (use --no-skip-existing to reprocess)")
        return 0

    # Load continuous data using unified loader
    logger.info(f"Loading {args.space}-level continuous data...")

    try:
        spatial_data = load_data(
            space=args.space,
            bids_root=bids_root,
            subject=args.subject,
            run=args.run,
            input_type="continuous",
            config=config,
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    logger.info(f"Loaded data shape: {spatial_data.data.shape}, sfreq: {spatial_data.sfreq} Hz")
    logger.info(f"Spatial dimension: {len(spatial_data.spatial_names)} {args.space} features")

    # Load events.tsv for trial metadata
    events_path = (
        bids_root
        / f"sub-{args.subject}"
        / "meg"
        / f"sub-{args.subject}_ses-recording_task-gradCPT_run-{args.run}_events.tsv"
    )

    if not events_path.exists():
        logger.error(f"Events file not found: {events_path}")
        return 1

    events_df = pd.read_csv(events_path, sep="\t")
    logger.info(f"Loaded {len(events_df)} events from {events_path}")

    # Compute Welch PSD
    psds, freqs, trial_metadata = compute_welch_psd_from_continuous(
        data=spatial_data.data,
        events_df=events_df,
        sfreq=spatial_data.sfreq,
        n_fft=args.n_fft,
        n_overlap=args.n_overlap,
        tmin=tmin,
        tmax=tmax,
        n_jobs=args.n_jobs,
    )

    # Save PSDs
    logger.info(f"Saving Welch PSDs to {output_root}...")
    save_welch_psds(
        psds=psds,
        freqs=freqs,
        trial_metadata=trial_metadata,
        output_path=output_root,
        subject=args.subject,
        run=args.run,
        space=args.space,
        config=config,
        n_fft=args.n_fft,
        n_overlap=args.n_overlap,
    )

    logger.info("✓ Stage 1 complete: Welch PSDs computed and saved!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
