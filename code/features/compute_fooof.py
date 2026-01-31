"""FOOOF (Fitting Oscillations & One Over F) feature extraction.

This script fits FOOOF models to Welch PSDs across sensor/source/atlas spaces.
It supports per-trial fitting and IN/OUT zone averaging for comparative analysis.

Usage:
    python -m code.features.compute_fooof --subject 04 --run 02 --space sensor
    python -m code.features.compute_fooof --subject 04 --run 02 --space source --inout-bounds 25 75

Author: Claude (Anthropic)
Date: 2026-01-31
"""

import argparse
import json
import logging
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fooof import FOOOF, FOOOFGroup

from code.utils.behavioral import classify_trials_from_vtc, get_VTC_from_file
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
    """Load pre-computed Welch PSD from derivatives.

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
    processed_root = data_root / config["paths"]["processed"]
    welch_dir = processed_root / f"features_welch_{space}" / f"sub-{subject}"

    # Find Welch PSD file
    pattern = f"sub-{subject}_*_run-{run}_*_space-{space}_*welch*.pkl"
    welch_files = list(welch_dir.glob(pattern))

    if not welch_files:
        raise FileNotFoundError(
            f"Welch PSD not found for sub-{subject} run-{run} space-{space} in {welch_dir}"
        )

    welch_file = welch_files[0]
    logger.info(f"Loading Welch PSD: {welch_file.name}")

    with open(welch_file, "rb") as f:
        welch_data = pickle.load(f)

    psd_array = welch_data["psd"]  # (n_trials, n_spatial, n_freqs)
    freq_bins = welch_data["freqs"]
    events_df = welch_data["events"]

    logger.info(
        f"Loaded PSD: {psd_array.shape[0]} trials, {psd_array.shape[1]} spatial units, "
        f"{psd_array.shape[2]} frequency bins"
    )

    return psd_array, freq_bins, events_df


def fit_fooof_group(
    psd_array: np.ndarray,
    freq_bins: np.ndarray,
    freq_range: Tuple[float, float],
    fooof_params: Dict[str, Any],
) -> FOOOFGroup:
    """Fit FOOOF to multi-channel/multi-ROI PSD.

    Args:
        psd_array: PSD data, shape (n_spatial, n_freqs)
        freq_bins: Frequency bins in Hz
        freq_range: Tuple of (min_freq, max_freq) for fitting
        fooof_params: FOOOF configuration parameters

    Returns:
        FOOOFGroup object with fitted models
    """
    n_spatial = psd_array.shape[0]

    # Initialize FOOOFGroup
    fg = FOOOFGroup(
        peak_width_limits=fooof_params.get("peak_width_limits", [1, 8]),
        max_n_peaks=fooof_params.get("max_n_peaks", 4),
        min_peak_height=fooof_params.get("min_peak_height", 0.10),
        peak_threshold=fooof_params.get("peak_threshold", 2.0),
        aperiodic_mode=fooof_params.get("aperiodic_mode", "fixed"),
        verbose=False,
    )

    # Fit FOOOF
    logger.debug(
        f"Fitting FOOOF: {n_spatial} spatial units, "
        f"freq_range={freq_range}, mode={fooof_params.get('aperiodic_mode', 'fixed')}"
    )

    fg.fit(freq_bins, psd_array, freq_range=freq_range)

    logger.debug(
        f"FOOOF fitting complete: "
        f"{len(fg.group_results)} models fit, "
        f"mean r² = {np.mean([r.r_squared for r in fg.group_results]):.3f}"
    )

    return fg


def extract_fooof_params(fg: FOOOFGroup) -> Dict[str, np.ndarray]:
    """Extract aperiodic and periodic parameters from FOOOFGroup.

    Args:
        fg: Fitted FOOOFGroup object

    Returns:
        Dictionary containing:
        - 'exponent': Aperiodic exponent per spatial unit
        - 'offset': Aperiodic offset per spatial unit
        - 'r_squared': Model fit quality per spatial unit
        - 'peak_params': Peak parameters (CF, PW, BW) per spatial unit
    """
    n_spatial = len(fg.group_results)

    # Extract aperiodic parameters
    exponents = np.array([r.aperiodic_params[1] for r in fg.group_results])
    offsets = np.array([r.aperiodic_params[0] for r in fg.group_results])
    r_squared = np.array([r.r_squared for r in fg.group_results])

    # Extract peak parameters (CF, PW, BW for each peak)
    peak_params = []
    for r in fg.group_results:
        if len(r.peak_params) > 0:
            peak_params.append(r.peak_params)  # (n_peaks, 3)
        else:
            peak_params.append(np.array([]))  # No peaks

    return {
        "exponent": exponents,
        "offset": offsets,
        "r_squared": r_squared,
        "peak_params": peak_params,
    }


def average_psd_by_zone(
    psd_array: np.ndarray,
    zones: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Average PSD across trials within IN/OUT zones.

    Args:
        psd_array: PSD data, shape (n_trials, n_spatial, n_freqs)
        zones: Trial classification from classify_trials_from_vtc

    Returns:
        Dictionary with:
        - 'IN_psd': Averaged PSD for IN trials, shape (n_spatial, n_freqs)
        - 'OUT_psd': Averaged PSD for OUT trials, shape (n_spatial, n_freqs)
    """
    in_idx = zones["IN_idx"]
    out_idx = zones["OUT_idx"]

    in_psd = np.mean(psd_array[in_idx], axis=0)  # (n_spatial, n_freqs)
    out_psd = np.mean(psd_array[out_idx], axis=0)  # (n_spatial, n_freqs)

    logger.info(
        f"Averaged PSD: IN={len(in_idx)} trials, OUT={len(out_idx)} trials"
    )

    return {
        "IN_psd": in_psd,
        "OUT_psd": out_psd,
    }


def process_subject_run(
    subject: str,
    run: str,
    space: str,
    config: Dict[str, Any],
    inout_bounds: Tuple[int, int],
    skip_existing: bool = True,
) -> Path:
    """Process one subject/run: fit FOOOF to Welch PSDs with IN/OUT averaging.

    Args:
        subject: Subject ID
        run: Run ID
        space: Analysis space (sensor/source/atlas)
        config: Configuration dictionary
        inout_bounds: Percentile bounds for IN/OUT classification
        skip_existing: Skip if output already exists

    Returns:
        Path to saved output file
    """
    # Validate input
    validate_subject_run(subject, run, config)

    # Setup output directory
    data_root = Path(config["paths"]["data_root"])
    processed_root = data_root / config["paths"]["processed"]
    output_dir = processed_root / f"features_fooof_{space}" / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output exists
    output_file = (
        output_dir
        / f"sub-{subject}_task-gradCPT_run-{run}_space-{space}_desc-fooof.pkl"
    )

    if skip_existing and output_file.exists():
        logger.info(f"Output exists, skipping: {output_file.name}")
        return output_file

    # Load Welch PSD
    logger.info(f"Processing sub-{subject} run-{run} space-{space}")
    psd_array, freq_bins, events_df = load_welch_psd(subject, run, space, config)

    # Load VTC and classify trials
    logger.info(f"Loading VTC and classifying trials (bounds={inout_bounds})")
    bids_root = data_root / config["paths"]["bids"]
    logs_dir = Path(config["paths"]["logs"])

    try:
        vtc_data = get_VTC_from_file(
            subject=subject,
            run=run,
            bids_root=bids_root,
            logs_dir=logs_dir,
            inbound=inout_bounds[0],
            outbound=inout_bounds[1],
        )
        vtc_filtered = vtc_data[2]  # VTC_filtered is 3rd element
    except Exception as e:
        logger.error(f"Could not load VTC data: {e}")
        logger.warning("Proceeding without IN/OUT classification")
        vtc_filtered = None

    # Classify trials if VTC available
    if vtc_filtered is not None:
        zones = classify_trials_from_vtc(vtc_filtered, inout_bounds=inout_bounds)

        # Average PSD by zone
        avg_psds = average_psd_by_zone(psd_array, zones)
        in_psd = avg_psds["IN_psd"]
        out_psd = avg_psds["OUT_psd"]
    else:
        zones = None
        in_psd = None
        out_psd = None

    # Get FOOOF parameters from config
    fooof_params = config["features"]["fooof"]
    freq_range = tuple(fooof_params["freq_range"])

    # Fit FOOOF to per-trial PSDs
    logger.info("Fitting FOOOF to per-trial PSDs")
    trial_fooofs = []
    trial_params = {
        "exponent": [],
        "offset": [],
        "r_squared": [],
        "peak_params": [],
    }

    for trial_idx in range(psd_array.shape[0]):
        trial_psd = psd_array[trial_idx]  # (n_spatial, n_freqs)

        fg = fit_fooof_group(trial_psd, freq_bins, freq_range, fooof_params)
        trial_fooofs.append(fg)

        # Extract parameters
        params = extract_fooof_params(fg)
        trial_params["exponent"].append(params["exponent"])
        trial_params["offset"].append(params["offset"])
        trial_params["r_squared"].append(params["r_squared"])
        trial_params["peak_params"].append(params["peak_params"])

    # Convert to arrays
    trial_params["exponent"] = np.array(trial_params["exponent"])  # (n_trials, n_spatial)
    trial_params["offset"] = np.array(trial_params["offset"])
    trial_params["r_squared"] = np.array(trial_params["r_squared"])

    logger.info(f"Per-trial FOOOF complete: {len(trial_fooofs)} trials processed")

    # Fit FOOOF to IN/OUT averaged PSDs
    if in_psd is not None and out_psd is not None:
        logger.info("Fitting FOOOF to IN/OUT averaged PSDs")

        fg_in = fit_fooof_group(in_psd, freq_bins, freq_range, fooof_params)
        fg_out = fit_fooof_group(out_psd, freq_bins, freq_range, fooof_params)

        in_params = extract_fooof_params(fg_in)
        out_params = extract_fooof_params(fg_out)
    else:
        fg_in = None
        fg_out = None
        in_params = None
        out_params = None

    # Prepare output
    output_data = {
        # Per-trial results
        "trial_fooofs": trial_fooofs,
        "trial_params": trial_params,
        # IN/OUT averaged results
        "IN_fooof": fg_in,
        "OUT_fooof": fg_out,
        "IN_params": in_params,
        "OUT_params": out_params,
        # Metadata
        "freq_bins": freq_bins,
        "freq_range": freq_range,
        "zones": zones,
        "events": events_df,
        "inout_bounds": inout_bounds,
        # Provenance
        "config": {
            "subject": subject,
            "run": run,
            "space": space,
            "fooof_params": fooof_params,
            "n_trials": psd_array.shape[0],
            "n_spatial": psd_array.shape[1],
        },
    }

    # Save output
    logger.info(f"Saving FOOOF results: {output_file.name}")
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)

    # Save metadata JSON
    metadata_file = output_file.with_suffix(".json")
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "subject": subject,
                "run": run,
                "space": space,
                "n_trials": int(psd_array.shape[0]),
                "n_spatial": int(psd_array.shape[1]),
                "n_freqs": int(len(freq_bins)),
                "freq_range": freq_range,
                "inout_bounds": inout_bounds,
                "mean_r_squared": float(np.mean(trial_params["r_squared"])),
                "fooof_params": fooof_params,
            },
            f,
            indent=2,
        )

    logger.info(f"✓ FOOOF complete for sub-{subject} run-{run}")

    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute FOOOF features from Welch PSDs"
    )
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--run", required=True, help="Run ID")
    parser.add_argument(
        "--space",
        required=True,
        choices=["sensor", "source", "atlas"],
        help="Analysis space",
    )
    parser.add_argument(
        "--inout-bounds",
        nargs=2,
        type=int,
        default=[25, 75],
        metavar=("LOWER", "UPPER"),
        help="Percentile bounds for IN/OUT classification (default: 25 75)",
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
        log_file=log_dir / f"fooof_sub-{args.subject}_run-{args.run}_{args.space}.log",
        level=args.log_level,
    )

    logger.info("=" * 80)
    logger.info("FOOOF Feature Extraction")
    logger.info("=" * 80)

    # Log provenance
    log_provenance(config)

    # Process subject/run
    try:
        output_file = process_subject_run(
            subject=args.subject,
            run=args.run,
            space=args.space,
            config=config,
            inout_bounds=tuple(args.inout_bounds),
            skip_existing=args.skip_existing,
        )

        logger.info("=" * 80)
        logger.info(f"✓ FOOOF extraction complete")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"FOOOF extraction failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
