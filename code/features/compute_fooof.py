"""FOOOF (Fitting Oscillations & One Over F) feature extraction.

This script fits FOOOF models to Welch PSDs across sensor/source/atlas spaces.
It supports per-trial fitting and produces:
1. FOOOF parameters (exponent, offset, r_squared) per trial
2. Aperiodic-corrected PSDs in the same format as original Welch PSDs

Usage:
    python -m code.features.compute_fooof --subject 04 --run 02 --space sensor

Output:
    {processed}/fooof_{space}/
    ├── sub-04/
    │   ├── sub-04_..._desc-fooof.npz           # FOOOF parameters
    │   └── sub-04_..._desc-fooof_params.json   # Metadata

    {processed}/welch_psds_corrected_{space}/
    ├── sub-04/
    │   ├── sub-04_..._desc-welch-corrected_psds.npz   # Corrected PSDs
    │   └── sub-04_..._desc-welch-corrected_psds_params.json

Author: Claude (Anthropic)
Date: 2026-01-31
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fooof import FOOOF, FOOOFGroup

from code.utils.config import load_config
from code.utils.logging_config import log_provenance, setup_logging
from code.utils.validation import validate_subject_run

logger = logging.getLogger(__name__)


def load_welch_psd(
    subject: str,
    run: str,
    space: str,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load pre-computed Welch PSD from processed/ directory.

    Args:
        subject: Subject ID
        run: Run ID
        space: Analysis space (sensor/source/atlas)
        config: Configuration dictionary

    Returns:
        Tuple of:
        - psd_array: PSD data, shape (n_trials, n_spatial, n_freqs)
        - freq_bins: Frequency bins in Hz
        - events_df: Event metadata with trial information

    Raises:
        FileNotFoundError: If Welch PSD file not found
    """
    data_root = Path(config["paths"]["data_root"])
    features_root = data_root / config["paths"]["features"]
    welch_dir = features_root / f"welch_psds_{space}" / f"sub-{subject}"

    # Find Welch PSD file (npz format from compute_welch_psd.py)
    task_name = config["bids"]["task_name"]
    pattern = f"sub-{subject}_*_run-{run}_space-{space}_desc-welch_psds.npz"
    welch_files = list(welch_dir.glob(pattern))

    if not welch_files:
        raise FileNotFoundError(
            f"Welch PSD not found for sub-{subject} run-{run} space-{space} in {welch_dir}"
        )

    welch_file = welch_files[0]
    logger.info(f"Loading Welch PSD: {welch_file.name}")

    # Load npz file
    welch_data = np.load(welch_file, allow_pickle=True)
    psd_array = welch_data["psds"]  # (n_trials, n_spatial, n_freqs)
    freq_bins = welch_data["freqs"]

    # Load trial metadata (saved as dict)
    trial_metadata_dict = welch_data["trial_metadata"].item()
    events_df = pd.DataFrame(trial_metadata_dict)

    logger.info(
        f"Loaded PSD: {psd_array.shape[0]} trials, {psd_array.shape[1]} spatial units, "
        f"{psd_array.shape[2]} frequency bins"
    )

    return psd_array, freq_bins, events_df


def fit_fooof_single(
    psd: np.ndarray,
    freq_bins: np.ndarray,
    freq_range: Tuple[float, float],
    fooof_params: Dict[str, Any],
) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit FOOOF to a single PSD and return parameters + corrected PSD.

    Args:
        psd: PSD data, shape (n_freqs,)
        freq_bins: Frequency bins in Hz
        freq_range: Tuple of (min_freq, max_freq) for fitting
        fooof_params: FOOOF configuration parameters

    Returns:
        Tuple of:
        - params: Dict with exponent, offset, r_squared
        - corrected_psd: Aperiodic-corrected PSD, shape (n_freqs,)
    """
    fm = FOOOF(
        peak_width_limits=fooof_params.get("peak_width_limits", [1, 8]),
        max_n_peaks=fooof_params.get("max_n_peaks", 4),
        min_peak_height=fooof_params.get("min_peak_height", 0.10),
        peak_threshold=fooof_params.get("peak_threshold", 2.0),
        aperiodic_mode=fooof_params.get("aperiodic_mode", "fixed"),
        verbose=False,
    )

    fm.fit(freq_bins, psd, freq_range=freq_range)

    # Extract parameters
    params = {
        "exponent": fm.aperiodic_params_[1] if fm.has_model else np.nan,
        "offset": fm.aperiodic_params_[0] if fm.has_model else np.nan,
        "r_squared": fm.r_squared_ if fm.has_model else np.nan,
    }

    # Get corrected (flattened) PSD - aperiodic component removed
    # This is the residual after removing the 1/f component
    if fm.has_model:
        # _spectrum_flat is the flattened spectrum (periodic only)
        # We need to get the full-length corrected spectrum
        # FOOOF only fits within freq_range, so we need to handle this carefully

        # Get the frequency mask for the fit range
        freq_mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])

        # Initialize corrected PSD with NaN outside fit range
        corrected_psd = np.full_like(psd, np.nan)

        # Get the flattened spectrum from FOOOF (only for fit range)
        if hasattr(fm, '_spectrum_flat') and fm._spectrum_flat is not None:
            corrected_psd[freq_mask] = fm._spectrum_flat
        else:
            # Fallback: compute manually
            # corrected = log(psd) - aperiodic_fit
            log_psd = np.log10(psd[freq_mask])
            aperiodic_fit = fm._ap_fit if hasattr(fm, '_ap_fit') else np.zeros_like(log_psd)
            corrected_psd[freq_mask] = log_psd - aperiodic_fit
    else:
        corrected_psd = np.full_like(psd, np.nan)

    return params, corrected_psd


def fit_fooof_group(
    psd_array: np.ndarray,
    freq_bins: np.ndarray,
    freq_range: Tuple[float, float],
    fooof_params: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Fit FOOOF to multi-channel PSD.

    Args:
        psd_array: PSD data, shape (n_spatial, n_freqs)
        freq_bins: Frequency bins in Hz
        freq_range: Tuple of (min_freq, max_freq) for fitting
        fooof_params: FOOOF configuration parameters

    Returns:
        Tuple of:
        - params: Dict with arrays of exponent, offset, r_squared (n_spatial,)
        - corrected_psds: Corrected PSDs, shape (n_spatial, n_freqs)
    """
    n_spatial = psd_array.shape[0]

    exponents = np.zeros(n_spatial)
    offsets = np.zeros(n_spatial)
    r_squareds = np.zeros(n_spatial)
    corrected_psds = np.zeros_like(psd_array)

    for i in range(n_spatial):
        params, corrected = fit_fooof_single(
            psd_array[i], freq_bins, freq_range, fooof_params
        )
        exponents[i] = params["exponent"]
        offsets[i] = params["offset"]
        r_squareds[i] = params["r_squared"]
        corrected_psds[i] = corrected

    return {
        "exponent": exponents,
        "offset": offsets,
        "r_squared": r_squareds,
    }, corrected_psds


def process_subject_run(
    subject: str,
    run: str,
    space: str,
    config: Dict[str, Any],
    skip_existing: bool = True,
) -> Optional[Path]:
    """Process one subject/run: fit FOOOF to Welch PSDs.

    Produces:
    1. FOOOF parameters (.npz) - exponent, offset, r_squared per trial/channel
    2. Corrected PSDs (.npz) - same format as original Welch PSDs

    Args:
        subject: Subject ID
        run: Run ID
        space: Analysis space (sensor/source/atlas)
        config: Configuration dictionary
        skip_existing: Skip if output already exists

    Returns:
        Path to saved FOOOF output file, or None if skipped
    """
    # Validate input
    validate_subject_run(subject, run, config)

    # Setup output directories
    data_root = Path(config["paths"]["data_root"])
    features_root = data_root / config["paths"]["features"]
    task_name = config["bids"]["task_name"]

    # FOOOF parameters output
    fooof_dir = features_root / f"fooof_{space}" / f"sub-{subject}"
    fooof_dir.mkdir(parents=True, exist_ok=True)
    fooof_file = (
        fooof_dir
        / f"sub-{subject}_ses-recording_task-{task_name}_run-{run}_space-{space}_desc-fooof.npz"
    )

    # Corrected PSDs output (same structure as welch_psds)
    corrected_dir = features_root / f"welch_psds_corrected_{space}" / f"sub-{subject}"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    corrected_file = (
        corrected_dir
        / f"sub-{subject}_ses-recording_task-{task_name}_run-{run}_space-{space}_desc-welch-corrected_psds.npz"
    )

    if skip_existing and fooof_file.exists() and corrected_file.exists():
        logger.info(f"Output exists, skipping: {fooof_file.name}")
        return fooof_file

    # Load Welch PSD
    logger.info(f"Processing sub-{subject} run-{run} space-{space}")
    psd_array, freq_bins, events_df = load_welch_psd(subject, run, space, config)

    # Get FOOOF parameters from config
    fooof_params_list = config["features"]["fooof"]
    fooof_params = fooof_params_list[0] if isinstance(fooof_params_list, list) else fooof_params_list
    freq_range = tuple(fooof_params["freq_range"])

    # Fit FOOOF to per-trial PSDs
    logger.info(f"Fitting FOOOF to {psd_array.shape[0]} trials")
    n_trials = psd_array.shape[0]
    n_spatial = psd_array.shape[1]

    # Initialize arrays
    exponents = np.zeros((n_trials, n_spatial))
    offsets = np.zeros((n_trials, n_spatial))
    r_squareds = np.zeros((n_trials, n_spatial))
    corrected_psds = np.zeros_like(psd_array)

    for trial_idx in range(n_trials):
        trial_psd = psd_array[trial_idx]  # (n_spatial, n_freqs)
        params, corrected = fit_fooof_group(trial_psd, freq_bins, freq_range, fooof_params)

        exponents[trial_idx] = params["exponent"]
        offsets[trial_idx] = params["offset"]
        r_squareds[trial_idx] = params["r_squared"]
        corrected_psds[trial_idx] = corrected

        if (trial_idx + 1) % 50 == 0:
            logger.info(f"  Processed {trial_idx + 1}/{n_trials} trials")

    logger.info(f"FOOOF fitting complete: mean r² = {np.nanmean(r_squareds):.3f}")

    # Get git hash for provenance
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    # Save FOOOF parameters
    logger.info(f"Saving FOOOF parameters: {fooof_file.name}")
    np.savez_compressed(
        fooof_file,
        exponent=exponents,  # (n_trials, n_spatial)
        offset=offsets,
        r_squared=r_squareds,
        trial_metadata=events_df.to_dict('list'),
        freqs=freq_bins,
    )

    # Save FOOOF metadata JSON
    fooof_params_file = fooof_file.with_name(fooof_file.name.replace(".npz", "_params.json"))
    with open(fooof_params_file, "w") as f:
        json.dump(
            {
                "subject": subject,
                "run": run,
                "space": space,
                "n_trials": int(n_trials),
                "n_spatial": int(n_spatial),
                "n_freqs": int(len(freq_bins)),
                "freq_range": list(freq_range),
                "fooof_params": fooof_params,
                "mean_r_squared": float(np.nanmean(r_squareds)),
                "mean_exponent": float(np.nanmean(exponents)),
                "git_hash": git_hash,
            },
            f,
            indent=2,
        )

    # Save corrected PSDs (same format as original Welch PSDs)
    logger.info(f"Saving corrected PSDs: {corrected_file.name}")
    np.savez_compressed(
        corrected_file,
        psds=corrected_psds,  # (n_trials, n_spatial, n_freqs) - same as original
        freqs=freq_bins,
        trial_metadata=events_df.to_dict('list'),
    )

    # Save corrected PSDs metadata JSON
    corrected_params_file = corrected_file.with_name(corrected_file.name.replace(".npz", "_params.json"))
    with open(corrected_params_file, "w") as f:
        json.dump(
            {
                "subject": subject,
                "run": run,
                "space": space,
                "description": "Aperiodic-corrected PSDs (1/f removed via FOOOF)",
                "n_trials": int(n_trials),
                "n_spatial": int(n_spatial),
                "n_freqs": int(len(freq_bins)),
                "freq_range": [float(freq_bins[0]), float(freq_bins[-1])],
                "fooof_freq_range": list(freq_range),
                "git_hash": git_hash,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved FOOOF parameters to {fooof_file}")
    logger.info(f"Saved corrected PSDs to {corrected_file}")

    return fooof_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute FOOOF features from Welch PSDs"
    )
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument(
        "--space",
        default="sensor",
        help="Analysis space: 'sensor', 'source', or atlas name (e.g., 'aparc.a2009s', 'schaefer_100')",
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=None,
        help="Override BIDS root from config",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip if output exists (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Reprocess even if output exists",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Setup logging
    logs_dir = Path(config["paths"]["logs"])
    log_dir = logs_dir / "features"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        name=__name__,
        log_file=log_dir / f"fooof_sub-{args.subject}_run-{args.run}_{args.space}.log",
        level=args.log_level,
    )

    logger.info("=" * 80)
    logger.info("FOOOF Feature Extraction")
    logger.info("=" * 80)

    # Log provenance
    log_provenance(logger, __name__, config=config)

    # Process subject/run
    try:
        output_file = process_subject_run(
            subject=args.subject,
            run=args.run,
            space=args.space,
            config=config,
            skip_existing=args.skip_existing,
        )

        logger.info("=" * 80)
        logger.info(f"FOOOF extraction complete")
        if output_file:
            logger.info(f"Output: {output_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"FOOOF extraction failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
