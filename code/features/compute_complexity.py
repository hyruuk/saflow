#!/usr/bin/env python3
"""
Complexity Feature Extraction (Per-Epoch Pipeline)
===================================================

Compute various complexity measures (LZC, entropy, fractal dimension) on MEG data.

This script uses the same per-epoch pipeline as PSD and FOOOF:
1. Load continuous data
2. Segment based on Freq/Rare events
3. Compute complexity per epoch
4. Save results with trial metadata

This script supports multiple complexity metrics from the antropy library:
- Lempel-Ziv Complexity (LZC)
- Various entropy measures (permutation, sample, approximate, spectral, SVD)
- Fractal dimensions (Higuchi, Petrosian, Katz, DFA)

All parameters are configurable via config.yaml under features.complexity.

Usage:
    python code/features/compute_complexity.py --subject 04 --run 02
    python code/features/compute_complexity.py --subject all

    # Compute specific complexity types only
    python code/features/compute_complexity.py --subject 04 --complexity-type lzc
    python code/features/compute_complexity.py --subject 04 --complexity-type entropy fractal

Output:
    {processed}/complexity_{space}/
    ├── sub-04/
    │   ├── sub-04_..._desc-complexity.npz
    │   └── sub-04_..._desc-complexity_params.json

Output format (.npz):
    - lzc_median: shape (n_epochs, n_channels)
    - entropy_permutation: shape (n_epochs, n_channels)
    - entropy_spectral: shape (n_epochs, n_channels)
    - fractal_higuchi: shape (n_epochs, n_channels)
    - ... (all configured metrics)
    - trial_metadata: dict with trial info
    - ch_names: list of channel names
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Local imports
from code.features.utils import segment_spatial_temporal_data
from code.utils.config import load_config
from code.utils.logging_config import setup_logging
from code.utils.validation import validate_subject, validate_run

# Complexity library
import antropy

logger = logging.getLogger(__name__)


def compute_lzc_antropy(
    signal: np.ndarray,
    normalize: bool = True,
    symbolize: str = "median",
) -> float:
    """
    Compute Lempel-Ziv Complexity using antropy.

    Args:
        signal: 1D time series
        normalize: Whether to normalize by sequence length
        symbolize: Method to binarize signal ('median', 'mean', or threshold value)

    Returns:
        LZC value
    """
    # Binarize signal
    if symbolize == "median":
        threshold = np.median(signal)
    elif symbolize == "mean":
        threshold = np.mean(signal)
    else:
        threshold = float(symbolize)

    signal_binarized = np.array([0 if x < threshold else 1 for x in signal])

    # Compute LZC
    lzc = antropy.lziv_complexity(signal_binarized, normalize=normalize)

    return lzc


def compute_entropy_metric(
    signal: np.ndarray,
    metric: str,
    **kwargs,
) -> float:
    """
    Compute entropy metric using antropy.

    Args:
        signal: 1D time series
        metric: Entropy metric name ('perm_entropy', 'sample_entropy', etc.)
        **kwargs: Metric-specific parameters

    Returns:
        Entropy value
    """
    # Get the function from antropy
    entropy_func = getattr(antropy, metric)

    # Special handling for spectral entropy (needs sampling frequency)
    if metric == "spectral_entropy":
        if "sf" not in kwargs or kwargs["sf"] is None:
            # Try to infer from signal length (default to 1000 Hz)
            kwargs["sf"] = 1000.0
            logger.warning(
                f"Sampling frequency not specified for spectral entropy, using {kwargs['sf']} Hz"
            )

    # Compute entropy
    entropy_val = entropy_func(signal, **kwargs)

    return entropy_val


def compute_fractal_metric(
    signal: np.ndarray,
    metric: str,
    **kwargs,
) -> float:
    """
    Compute fractal dimension metric using antropy.

    Args:
        signal: 1D time series
        metric: Fractal metric name ('higuchi_fd', 'petrosian_fd', etc.)
        **kwargs: Metric-specific parameters

    Returns:
        Fractal dimension value
    """
    # Get the function from antropy
    fractal_func = getattr(antropy, metric)

    # Compute fractal dimension
    fractal_val = fractal_func(signal, **kwargs)

    return fractal_val


def compute_complexity_for_channel(
    signal: np.ndarray,
    chan_idx: int,
    complexity_type: str,
    params: Dict[str, Any],
    sfreq: float,
) -> float:
    """
    Compute complexity measure for a single channel.

    Args:
        signal: 1D time series for one channel
        chan_idx: Channel index
        complexity_type: Type of complexity ('lzc', 'entropy', 'fractal')
        params: Parameters for complexity computation
        sfreq: Sampling frequency

    Returns:
        Complexity value
    """
    try:
        if complexity_type == "lzc":
            value = compute_lzc_antropy(
                signal,
                normalize=params.get("normalize", True),
                symbolize=params.get("symbolize", "median"),
            )

        elif complexity_type == "entropy":
            metric = params["metric"]
            # Prepare kwargs (exclude metadata keys)
            kwargs = {k: v for k, v in params.items() if k not in ["metric", "name"]}
            # Add sampling frequency if needed
            if metric == "spectral_entropy":
                kwargs["sf"] = sfreq
            value = compute_entropy_metric(signal, metric, **kwargs)

        elif complexity_type == "fractal":
            metric = params["metric"]
            # Prepare kwargs (exclude metadata keys)
            kwargs = {k: v for k, v in params.items() if k not in ["metric", "name"]}
            value = compute_fractal_metric(signal, metric, **kwargs)

        else:
            raise ValueError(f"Unknown complexity type: {complexity_type}")

        logger.debug(
            f"Channel {chan_idx}: {complexity_type} ({params.get('name', 'unnamed')}) = {value:.6f}"
        )
        return value

    except Exception as e:
        logger.error(
            f"Error computing {complexity_type} for channel {chan_idx}: {e}"
        )
        return np.nan


def compute_complexity_all_channels(
    data: np.ndarray,
    complexity_type: str,
    params: Dict[str, Any],
    sfreq: float,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute complexity measure for all channels in parallel.

    Args:
        data: 2D array (n_channels, n_times)
        complexity_type: Type of complexity ('lzc', 'entropy', 'fractal')
        params: Parameters for complexity computation
        sfreq: Sampling frequency
        n_jobs: Number of parallel jobs

    Returns:
        1D array of complexity values (n_channels,)
    """
    n_channels = data.shape[0]

    # Parallel computation across channels
    complexity_values = Parallel(n_jobs=n_jobs)(
        delayed(compute_complexity_for_channel)(
            data[ch, :], ch, complexity_type, params, sfreq
        )
        for ch in range(n_channels)
    )

    return np.array(complexity_values)


def process_subject_run(
    subject: str,
    run: str,
    config: Dict[str, Any],
    complexity_types: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Process one subject/run: compute all complexity features per epoch.

    Uses the same per-epoch pipeline as PSD and FOOOF:
    1. Load continuous data
    2. Load events and segment into epochs
    3. Compute all complexity metrics per epoch
    4. Save in single .npz file with JSON sidecar

    Args:
        subject: Subject ID
        run: Run ID
        config: Configuration dictionary
        complexity_types: List of complexity types to compute (None = all)
        overwrite: Whether to overwrite existing files

    Returns:
        Path to output file, or None if skipped
    """
    logger.info(f"Processing subject {subject}, run {run}")

    # Get paths
    space = config["analysis"]["space"]
    data_root = Path(config["paths"]["data_root"])
    derivatives_root = Path(config["paths"]["derivatives"])
    bids_root = data_root / "bids"
    task_name = config["bids"]["task_name"]

    # Output directory and file
    output_root = Path(config["paths"]["features"]) / f"complexity_{space}"
    output_dir = output_root / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (
        f"sub-{subject}_ses-recording_task-{task_name}_run-{run}_"
        f"space-{space}_desc-complexity.npz"
    )

    if output_file.exists() and not overwrite:
        logger.info(f"Output exists, skipping: {output_file.name}")
        return None

    # Build path for preprocessed data (ICA-cleaned continuous data)
    preproc_dir = derivatives_root / "preprocessed" / f"sub-{subject}" / "meg"
    preproc_file = preproc_dir / f"sub-{subject}_task-{task_name}_run-{run}_proc-clean_meg.fif"

    if not preproc_file.exists():
        logger.error(f"Preprocessed file not found: {preproc_file}")
        return None

    # Load events.tsv for trial metadata
    events_path = (
        bids_root
        / f"sub-{subject}"
        / "meg"
        / f"sub-{subject}_task-{task_name}_run-{run}_events.tsv"
    )

    if not events_path.exists():
        logger.error(f"Events file not found: {events_path}")
        return None

    events_df = pd.read_csv(events_path, sep="\t")
    logger.info(f"Loaded {len(events_df)} events from {events_path}")

    # Load data
    logger.info(f"Loading data from {preproc_file}")
    raw = mne.io.read_raw_fif(preproc_file, preload=True)

    # Pick MEG channels only
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)
    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]
    ch_names = [raw.ch_names[i] for i in picks]

    logger.info(
        f"Data shape: {data.shape} ({data.shape[0]} channels, {data.shape[1]} samples)"
    )

    # Get epoch timing from config
    tmin = config["analysis"]["epochs"]["tmin"]
    tmax = config["analysis"]["epochs"]["tmax"]

    # Segment data into epochs
    segmented_data, trial_metadata = segment_spatial_temporal_data(
        data=data,
        events_df=events_df,
        sfreq=sfreq,
        tmin=tmin,
        tmax=tmax,
    )
    # Shape: (n_epochs, n_channels, n_samples)
    logger.info(f"Segmented data shape: {segmented_data.shape}")

    # Get complexity config
    complexity_config = config["features"].get("complexity", {})

    # Determine which complexity types to compute
    if complexity_types is None:
        complexity_types = list(complexity_config.keys())

    # Compute each complexity type
    n_jobs = config["computing"]["n_jobs"]
    n_epochs = segmented_data.shape[0]

    # Store all complexity results
    complexity_results = {}
    params_info = {}

    for comp_type in complexity_types:
        if comp_type not in complexity_config:
            logger.warning(f"Complexity type '{comp_type}' not in config, skipping")
            continue

        param_sets = complexity_config[comp_type]

        for param_set in param_sets:
            param_name = param_set.get("name", "unnamed")
            metric_key = f"{comp_type}_{param_name}"

            logger.info(f"Computing {comp_type} with parameters: {param_name}")

            # Compute complexity per epoch
            complexity_values = []
            for epoch_idx in range(n_epochs):
                epoch_data = segmented_data[epoch_idx]  # (n_channels, n_samples)
                values = compute_complexity_all_channels(
                    epoch_data, comp_type, param_set, sfreq, n_jobs=n_jobs
                )
                complexity_values.append(values)

            complexity_values = np.array(complexity_values)
            # Shape: (n_epochs, n_channels)
            logger.info(f"  {metric_key} shape: {complexity_values.shape}")
            logger.info(
                f"  {metric_key} range: "
                f"[{np.nanmin(complexity_values):.6f}, {np.nanmax(complexity_values):.6f}]"
            )

            complexity_results[metric_key] = complexity_values
            params_info[metric_key] = param_set

    # Get git hash for provenance
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    # Save all results to single .npz file
    logger.info(f"Saving complexity results: {output_file.name}")
    np.savez_compressed(
        output_file,
        **complexity_results,
        trial_metadata=trial_metadata.to_dict('list'),
        ch_names=ch_names,
    )

    # Save JSON sidecar with parameters
    params_file = output_file.with_name(output_file.name.replace(".npz", "_params.json"))
    params_data = {
        "subject": subject,
        "run": run,
        "space": space,
        "sfreq": float(sfreq),
        "tmin": tmin,
        "tmax": tmax,
        "n_epochs": int(n_epochs),
        "n_channels": len(ch_names),
        "metrics": list(complexity_results.keys()),
        "metric_params": params_info,
        "git_hash": git_hash,
    }

    with open(params_file, "w") as f:
        json.dump(params_data, f, indent=2)

    logger.info(f"Saved parameters to {params_file.name}")
    logger.info(f"Metrics computed: {list(complexity_results.keys())}")

    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute complexity features on MEG data"
    )
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        default="all",
        help="Subject ID (e.g., '04') or 'all'",
    )
    parser.add_argument(
        "-r", "--run", type=str, default="all", help="Run ID (e.g., '02') or 'all'"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: search standard locations)",
    )
    parser.add_argument(
        "--complexity-type",
        type=str,
        nargs="+",
        default=None,
        help="Complexity types to compute (lzc, entropy, fractal). Default: all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_dir = Path(config["paths"]["logs"]) / "features" / "complexity"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        __name__,
        log_file=log_dir / f"complexity_{args.subject}_{args.run}.log",
        level="DEBUG" if args.verbose else "INFO",
    )

    logger.info("=" * 80)
    logger.info("Complexity Feature Extraction")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Run: {args.run}")
    logger.info(f"Complexity types: {args.complexity_type or 'all'}")
    logger.info(f"Analysis space: {config['analysis']['space']}")

    # Get subjects and runs
    subjects = (
        config["bids"]["subjects"] if args.subject == "all" else [args.subject]
    )
    runs = config["bids"]["task_runs"] if args.run == "all" else [args.run]

    # Validate
    for subj in subjects:
        validate_subject(subj, config)
    for r in runs:
        validate_run(r, config)

    # Process
    for subject in subjects:
        for run in runs:
            try:
                process_subject_run(
                    subject,
                    run,
                    config,
                    complexity_types=args.complexity_type,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                logger.error(
                    f"Failed to process subject {subject}, run {run}: {e}",
                    exc_info=True,
                )
                continue

    logger.info("=" * 80)
    logger.info("Complexity feature extraction complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
