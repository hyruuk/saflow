#!/usr/bin/env python3
"""
Complexity Feature Extraction
==============================

Compute various complexity measures (LZC, entropy, fractal dimension) on MEG data.

This script supports multiple complexity metrics from antropy and neurokit2 libraries:
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
    {processed}/features_complexity_{space}/
    ├── sub-04/
    │   ├── sub-04_task-gradCPT_run-02_lzc-antropy_median.pkl
    │   ├── sub-04_task-gradCPT_run-02_entropy-permutation.pkl
    │   ├── sub-04_task-gradCPT_run-02_fractal-higuchi.pkl
    │   └── ...
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from joblib import Parallel, delayed
from mne_bids import BIDSPath, read_raw_bids

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from utils.logging_config import setup_logging
from utils.paths import get_bids_path, get_output_path
from utils.validation import validate_subject, validate_run

# Complexity libraries
import antropy
from neurokit2.complexity import (
    complexity_delay,
    complexity_dimension,
    complexity_lempelziv,
)

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


def compute_lzc_neurokit(
    signal: np.ndarray,
    permutation: bool = True,
    dimension: int = 7,
    delay: int = 2,
) -> float:
    """
    Compute Lempel-Ziv Complexity using neurokit2.

    Args:
        signal: 1D time series
        permutation: Whether to use permutation LZC
        dimension: Embedding dimension
        delay: Time delay

    Returns:
        LZC value
    """
    lzc = complexity_lempelziv(
        signal, permutation=permutation, dimension=dimension, delay=delay
    )[0]

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
            library = params.get("library", "antropy")
            if library == "antropy":
                value = compute_lzc_antropy(
                    signal,
                    normalize=params.get("normalize", True),
                    symbolize=params.get("symbolize", "median"),
                )
            elif library == "neurokit2":
                value = compute_lzc_neurokit(
                    signal,
                    permutation=params.get("permutation", True),
                    dimension=params.get("dimension", 7),
                    delay=params.get("delay", 2),
                )
            else:
                raise ValueError(f"Unknown library for LZC: {library}")

        elif complexity_type == "entropy":
            metric = params["metric"]
            # Prepare kwargs
            kwargs = {k: v for k, v in params.items() if k not in ["metric", "library", "name"]}
            # Add sampling frequency if needed
            if metric == "spectral_entropy":
                kwargs["sf"] = sfreq
            value = compute_entropy_metric(signal, metric, **kwargs)

        elif complexity_type == "fractal":
            metric = params["metric"]
            # Prepare kwargs
            kwargs = {k: v for k, v in params.items() if k not in ["metric", "library", "name"]}
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
) -> None:
    """
    Process one subject/run: compute complexity features.

    Args:
        subject: Subject ID
        run: Run ID
        config: Configuration dictionary
        complexity_types: List of complexity types to compute (None = all)
        overwrite: Whether to overwrite existing files
    """
    logger.info(f"Processing subject {subject}, run {run}")

    # Get paths
    space = config["analysis"]["space"]
    bids_root = Path(config["paths"]["derivatives"]) / "bids"

    # Build BIDS path for preprocessed data
    processing_track = config["analysis"]["processing_tracks"][
        0
    ]  # Use first track (continuous or epochs)

    bids_path = BIDSPath(
        subject=subject,
        task=config["bids"]["task_name"],
        run=run,
        session="recording",
        datatype="meg",
        root=bids_root,
        processing=processing_track,
        extension=".fif",
    )

    if not bids_path.fpath.exists():
        logger.error(f"Preprocessed file not found: {bids_path.fpath}")
        return

    # Load data
    logger.info(f"Loading data from {bids_path.fpath}")
    raw = read_raw_bids(bids_path)

    # Pick MEG channels only
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)
    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]
    ch_names = [raw.ch_names[i] for i in picks]

    logger.info(
        f"Data shape: {data.shape} ({data.shape[0]} channels, {data.shape[1]} samples)"
    )

    # Output directory
    output_root = Path(config["paths"]["processed"]) / f"features_complexity_{space}"
    output_dir = output_root / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get complexity config
    complexity_config = config["features"].get("complexity", {})

    # Determine which complexity types to compute
    if complexity_types is None:
        complexity_types = list(complexity_config.keys())

    # Compute each complexity type
    n_jobs = config["computing"]["n_jobs"]

    for comp_type in complexity_types:
        if comp_type not in complexity_config:
            logger.warning(f"Complexity type '{comp_type}' not in config, skipping")
            continue

        param_sets = complexity_config[comp_type]

        for param_set in param_sets:
            param_name = param_set.get("name", "unnamed")

            # Output filename
            output_fname = output_dir / (
                f"sub-{subject}_task-{config['bids']['task_name']}_run-{run}_"
                f"{comp_type}-{param_name}.pkl"
            )

            if output_fname.exists() and not overwrite:
                logger.info(f"Output exists, skipping: {output_fname.name}")
                continue

            logger.info(
                f"Computing {comp_type} with parameters: {param_name}"
            )

            # Compute complexity
            complexity_values = compute_complexity_all_channels(
                data, comp_type, param_set, sfreq, n_jobs=n_jobs
            )

            # Prepare output
            output_data = {
                "data": complexity_values,
                "ch_names": ch_names,
                "complexity_type": comp_type,
                "params": param_set,
                "sfreq": sfreq,
                "subject": subject,
                "run": run,
                "space": space,
                "processing_track": processing_track,
            }

            # Save
            with open(output_fname, "wb") as f:
                pickle.dump(output_data, f)

            logger.info(f"Saved to {output_fname}")
            logger.info(
                f"  {comp_type}-{param_name} range: "
                f"[{np.nanmin(complexity_values):.6f}, {np.nanmax(complexity_values):.6f}]"
            )


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
        log_dir / f"complexity_{args.subject}_{args.run}.log",
        level=logging.DEBUG if args.verbose else logging.INFO,
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
