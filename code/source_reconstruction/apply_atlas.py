"""Apply atlas/parcellation to source estimates (Stage 2b).

This script applies cortical parcellations (e.g., aparc.a2009s, Schaefer) to morphed
source estimates, aggregating time series within each ROI/label using
``mne.extract_label_time_course``. Default mode is ``mean_flip`` (appropriate
for signed dSPM/MNE sources).

Outputs:
- ROI-averaged time series (.npz format with data + region names)
- JSON sidecar with metadata

Usage:
    # Single subject, all runs, default atlases
    python -m code.source_reconstruction.apply_atlas --subject 04

    # Specific runs
    python -m code.source_reconstruction.apply_atlas --subject 04 --runs "02 03 04"

    # Specific atlas
    python -m code.source_reconstruction.apply_atlas --subject 04 --atlas aparc.a2009s

    # Multiple atlases
    python -m code.source_reconstruction.apply_atlas --subject 04 --atlases "aparc.a2009s Schaefer2018_100Parcels_7Networks_order"
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np
from mne_bids import BIDSPath

from code.utils.config import load_config
from code.utils.logging_config import setup_logging
from code.utils.validation import validate_subject_run

# Default atlases to apply (short names)
DEFAULT_ATLASES = [
    "aparc.a2009s",
    "schaefer_100",
    "schaefer_200",
    "schaefer_400",
]

# Mapping from short names to full MNE/FreeSurfer atlas names
ATLAS_ALIASES = {
    "schaefer_100": "Schaefer2018_100Parcels_7Networks_order",
    "schaefer_200": "Schaefer2018_200Parcels_7Networks_order",
    "schaefer_400": "Schaefer2018_400Parcels_7Networks_order",
    # These stay as-is
    "aparc.a2009s": "aparc.a2009s",
    "aparc": "aparc",
}


def get_mne_atlas_name(short_name: str) -> str:
    """Convert short atlas name to full MNE/FreeSurfer name."""
    return ATLAS_ALIASES.get(short_name, short_name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply atlas/parcellation to source estimates (Stage 2b)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject ID (e.g., '04')",
    )

    parser.add_argument(
        "--run",
        type=str,
        help="Single run number (e.g., '02'). Mutually exclusive with --runs.",
    )

    parser.add_argument(
        "--runs",
        type=str,
        help="Space-separated run numbers (e.g., '02 03 04'). If neither --run nor --runs specified, processes all runs from config.",
    )

    parser.add_argument(
        "--atlas",
        type=str,
        help="Single atlas/parcellation name. Mutually exclusive with --atlases.",
    )

    parser.add_argument(
        "--atlases",
        type=str,
        help="Space-separated atlas names (e.g., 'aparc.a2009s Schaefer2018_100Parcels_7Networks_order'). If neither --atlas nor --atlases specified, uses default set.",
    )

    parser.add_argument(
        "--processing",
        type=str,
        default="clean",
        choices=["clean", "ica"],
        help="Processing state of input data",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["mean_flip", "mean", "pca_flip", "max", "auto"],
        help="ROI aggregation mode (passed to mne.extract_label_time_course). "
             "Default reads from config.source_reconstruction.label_mode (mean_flip).",
    )

    parser.add_argument(
        "--bids-root",
        type=Path,
        help="Override BIDS root directory from config",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip processing if output files already exist (default)",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Reprocess even if output files exist",
    )

    args = parser.parse_args()

    # Validate run arguments (mutually exclusive but both optional)
    if args.run and args.runs:
        parser.error("--run and --runs are mutually exclusive")

    # Validate atlas arguments (mutually exclusive but both optional)
    if args.atlas and args.atlases:
        parser.error("--atlas and --atlases are mutually exclusive")

    return args


def create_atlas_paths(
    subject: str,
    run: str,
    atlas: str,
    processing: str,
    derivatives_root: Path,
) -> Dict[str, Path]:
    """Create paths for atlas application.

    Args:
        subject: Subject ID
        run: Run number
        atlas: Atlas name
        processing: Processing state
        derivatives_root: Derivatives directory (for both input and output)

    Returns:
        Dictionary with 'input' and 'output' paths
    """
    # Input: morphed sources (in derivatives)
    input_dir = derivatives_root / "morphed_sources" / f"sub-{subject}" / "meg"
    input_file = input_dir / f"sub-{subject}_task-gradCPT_run-{run}_proc-{processing}_desc-morphed-stc.h5"

    # Output: atlas timeseries (in derivatives, part of source reconstruction pipeline)
    output_dir = derivatives_root / f"atlas_timeseries_{atlas}" / f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"sub-{subject}_ses-recording_task-gradCPT_run-{run}_space-{atlas}_timeseries.npz"

    return {"input": input_file, "output": output_file}


def apply_atlas_to_stc(
    stc: mne.SourceEstimate,
    atlas: str,
    subjects_dir: Path,
    mode: str = "mean_flip",
) -> Dict[str, np.ndarray]:
    """Apply atlas to source estimate and aggregate within ROIs.

    Uses MNE's ``extract_label_time_course``. For signed source estimates
    (dSPM, MNE), ``mean_flip`` projects each vertex onto its label's dominant
    cortical orientation before averaging, avoiding sign cancellation between
    vertices with opposing dipole orientations.

    Args:
        stc: Source estimate in fsaverage space
        atlas: Atlas/parcellation name (short name, will be converted to MNE name)
        subjects_dir: FreeSurfer subjects directory
        mode: Aggregation mode passed to ``extract_label_time_course``
            (e.g. ``mean_flip``, ``mean``, ``pca_flip``, ``max``).

    Returns:
        Dictionary mapping region names to aggregated time series
    """
    logger = logging.getLogger(__name__)

    mne_atlas_name = get_mne_atlas_name(atlas)

    logger.debug(f"Loading atlas '{atlas}' ({mne_atlas_name}) from fsaverage")
    labels = mne.read_labels_from_annot(
        "fsaverage", parc=mne_atlas_name, subjects_dir=subjects_dir, verbose=False
    )
    logger.info(f"Loaded {len(labels)} labels from atlas '{atlas}'")

    # mean_flip needs cortical surface normals → load the fsaverage source space
    fsaverage_src_path = subjects_dir / "fsaverage" / "bem" / "fsaverage-oct-6-src.fif"
    if not fsaverage_src_path.exists():
        raise FileNotFoundError(
            f"fsaverage source space not found at {fsaverage_src_path}. "
            "Run run_inverse_solution.py first to generate it."
        )
    src = mne.read_source_spaces(str(fsaverage_src_path), verbose=False)

    # Drop labels whose vertices don't intersect this STC's vertices (e.g. medial
    # wall / unknown). Preserves the previous behavior of only returning ROIs
    # that actually contain sources.
    vertices_lh = set(int(v) for v in stc.vertices[0])
    vertices_rh = set(int(v) for v in stc.vertices[1])
    nonempty_labels = []
    for lbl in labels:
        target = vertices_lh if lbl.hemi == "lh" else vertices_rh
        if set(int(v) for v in lbl.vertices) & target:
            nonempty_labels.append(lbl)
    n_dropped = len(labels) - len(nonempty_labels)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} label(s) with no vertices in source space")

    label_ts = mne.extract_label_time_course(
        stc, nonempty_labels, src, mode=mode, allow_empty=False, verbose=False
    )
    # label_ts shape: (n_labels, n_times)

    region_averages = {
        lbl.name: label_ts[i] for i, lbl in enumerate(nonempty_labels)
    }
    logger.info(f"Computed '{mode}' aggregates for {len(region_averages)} regions")

    return region_averages


def process_single_run(
    subject: str,
    run: str,
    atlas: str,
    processing: str,
    config: dict,
    derivatives_root: Path,
    skip_existing: bool,
    mode: str = "mean_flip",
) -> bool:
    """Process a single subject/run with one atlas.

    Args:
        subject: Subject ID
        run: Run number
        atlas: Atlas name
        processing: Processing state
        config: Configuration dictionary
        derivatives_root: Derivatives directory
        skip_existing: Skip if output exists

    Returns:
        True if processing succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    logger.info(f"  [{atlas}] Processing sub-{subject}, run-{run}")

    # Get FreeSurfer subjects directory (already expanded in config)
    fs_subjects_dir = Path(config["paths"]["freesurfer_subjects_dir"])

    if not fs_subjects_dir.exists():
        logger.error(f"FreeSurfer subjects directory not found: {fs_subjects_dir}")
        return False

    # Create paths
    try:
        filepaths = create_atlas_paths(
            subject, run, atlas, processing, derivatives_root
        )
    except Exception as e:
        logger.error(f"Failed to create paths: {e}")
        return False

    # Check if output exists
    output_path = filepaths["output"]
    if skip_existing and output_path.exists():
        logger.info(f"  [{atlas}] Output exists, skipping: {output_path.name}")
        return True

    # Load source estimate
    input_path = filepaths["input"]
    if not input_path.exists():
        logger.error(f"Input source estimate not found: {input_path}")
        logger.error("Run run_inverse_solution.py first to generate source estimates")
        return False

    logger.debug(f"Loading source estimate: {input_path}")
    try:
        stc = mne.read_source_estimate(str(input_path))
    except Exception as e:
        logger.error(f"Failed to load source estimate: {e}", exc_info=True)
        return False

    logger.debug(f"Loaded STC: {stc.data.shape[0]} sources, {stc.data.shape[1]} timepoints")

    # Apply atlas
    try:
        region_averages = apply_atlas_to_stc(stc, atlas, fs_subjects_dir, mode=mode)
    except Exception as e:
        logger.error(f"Failed to apply atlas '{atlas}': {e}", exc_info=True)
        return False

    # Convert to arrays - sort by region name for consistency
    sorted_regions = sorted(region_averages.keys())
    region_data = np.array([region_averages[r] for r in sorted_regions])
    region_names = sorted_regions

    logger.info(f"  [{atlas}] {len(region_names)} ROIs × {region_data.shape[1]} timepoints")

    # Get git hash for provenance
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    # Save as .npz (consistent with other features)
    logger.debug(f"Saving ROI-averaged data: {output_path}")
    try:
        np.savez_compressed(
            output_path,
            data=region_data,  # (n_rois, n_times)
            roi_names=region_names,
            sfreq=stc.sfreq,
        )
    except Exception as e:
        logger.error(f"Failed to save output: {e}", exc_info=True)
        return False

    # Save metadata JSON sidecar
    metadata = {
        "subject": subject,
        "run": run,
        "atlas": atlas,
        "processing": processing,
        "label_mode": mode,
        "n_rois": len(region_names),
        "n_timepoints": int(region_data.shape[1]),
        "sfreq": float(stc.sfreq),
        "roi_names": region_names,
        "git_hash": git_hash,
    }

    metadata_path = output_path.with_name(output_path.name.replace(".npz", "_params.json"))
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.debug(f"Saved metadata: {metadata_path.name}")

    return True


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        return 1

    # Determine atlases to process
    if args.atlas:
        atlases = [args.atlas]
    elif args.atlases:
        atlases = args.atlases.split()
    else:
        # Use atlases from config, or fall back to hardcoded defaults
        atlases = config.get("source_reconstruction", {}).get("atlases", DEFAULT_ATLASES)

    # Resolve ROI aggregation mode: CLI > config > default
    mode = (
        args.mode
        or config.get("source_reconstruction", {}).get("label_mode")
        or "mean_flip"
    )

    # Setup logging
    log_dir = Path(config["paths"]["logs"]) / "source_reconstruction"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        name=__name__,
        log_file=log_dir / f"apply_atlas_sub-{args.subject}.log",
        level=args.log_level,
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Atlas Application (Stage 2b)")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Atlases: {', '.join(atlases)}")
    logger.info(f"Label aggregation mode: {mode}")

    # Determine paths
    data_root = Path(config["paths"]["data_root"])
    if args.bids_root:
        bids_root = args.bids_root
    else:
        bids_root = Path(config["paths"]["bids"])

    derivatives_root = data_root / config["paths"]["derivatives"]

    logger.info(f"Derivatives root: {derivatives_root}")

    # Parse runs - default to all runs from config if not specified
    if args.run:
        runs = [args.run]
    elif args.runs:
        runs = args.runs.split()
    else:
        runs = config["bids"]["task_runs"]
        logger.info(f"No runs specified, using all runs from config: {runs}")

    logger.info(f"Processing {len(runs)} run(s): {', '.join(runs)}")

    # Process each atlas × run combination
    total_tasks = len(atlases) * len(runs)
    success_count = 0
    failed_tasks = []

    for atlas in atlases:
        logger.info(f"\n[Atlas: {atlas}]")

        for run in runs:
            # Validate subject/run
            if not validate_subject_run(args.subject, run, config):
                logger.error(f"Invalid subject/run combination: sub-{args.subject}, run-{run}")
                failed_tasks.append(f"{atlas}/run-{run}")
                continue

            # Process
            success = process_single_run(
                subject=args.subject,
                run=run,
                atlas=atlas,
                processing=args.processing,
                config=config,
                derivatives_root=derivatives_root,
                skip_existing=args.skip_existing,
                mode=mode,
            )

            if success:
                success_count += 1
            else:
                failed_tasks.append(f"{atlas}/run-{run}")

    # Summary
    logger.info("=" * 80)
    logger.info("Processing complete")
    logger.info(f"  Successful: {success_count}/{total_tasks}")
    if failed_tasks:
        logger.warning(f"  Failed: {', '.join(failed_tasks)}")
        return 1

    logger.info("✓ All atlas applications complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
