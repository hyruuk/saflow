"""Apply atlas/parcellation to source estimates (Stage 2b).

This script applies cortical parcellations (e.g., aparc.a2009s, Schaefer) to morphed
source estimates, averaging time series within each ROI/label.

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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np
from mne_bids import BIDSPath

from code.utils.config import load_config
from code.utils.logging_config import setup_logging
from code.utils.validation import validate_subject_run

# Default atlases to apply
DEFAULT_ATLASES = [
    "aparc.a2009s",
    "Schaefer2018_100Parcels_7Networks_order",
    "Schaefer2018_200Parcels_7Networks_order",
    "Schaefer2018_400Parcels_7Networks_order",
]


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
        choices=["clean", "ica", "icaar"],
        help="Processing state of input data",
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
) -> Dict[str, np.ndarray]:
    """Apply atlas to source estimate and average within ROIs.

    Args:
        stc: Source estimate (should be in fsaverage space)
        atlas: Atlas/parcellation name
        subjects_dir: FreeSurfer subjects directory

    Returns:
        Dictionary mapping region names to averaged time series
    """
    logger = logging.getLogger(__name__)

    # Load atlas labels
    logger.debug(f"Loading atlas '{atlas}' from fsaverage")
    labels = mne.read_labels_from_annot(
        "fsaverage", parc=atlas, subjects_dir=subjects_dir, verbose=False
    )
    logger.info(f"Loaded {len(labels)} labels from atlas '{atlas}'")

    # Get vertices for left and right hemispheres
    vertices_lh = stc.vertices[0]
    vertices_rh = stc.vertices[1]

    # Map vertices to regions
    vertex_to_region = {}

    for label in labels:
        label_vertices = label.vertices
        hemi = 0 if label.hemi == "lh" else 1

        if hemi == 0:
            common_vertices = np.intersect1d(vertices_lh, label_vertices)
        else:
            common_vertices = np.intersect1d(vertices_rh, label_vertices)

        for vert in common_vertices:
            vertex_to_region[vert] = label.name

    logger.debug(f"Mapped {len(vertex_to_region)} vertices to regions")

    # Collect data for each region
    region_data = defaultdict(list)

    for vert_idx, region in vertex_to_region.items():
        if vert_idx in vertices_lh:
            idx = np.where(vertices_lh == vert_idx)[0][0]
            region_data[region].append(stc.data[idx])
        elif vert_idx in vertices_rh:
            idx = np.where(vertices_rh == vert_idx)[0][0]
            region_data[region].append(stc.data[len(vertices_lh) + idx])

    # Average within each region
    region_averages = {
        region: np.mean(np.array(data), axis=0)
        for region, data in region_data.items()
    }

    logger.info(f"Computed averages for {len(region_averages)} regions")

    return region_averages


def process_single_run(
    subject: str,
    run: str,
    atlas: str,
    processing: str,
    config: dict,
    derivatives_root: Path,
    skip_existing: bool,
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
        stc = mne.read_source_estimate(str(input_path), verbose=False)
    except Exception as e:
        logger.error(f"Failed to load source estimate: {e}", exc_info=True)
        return False

    logger.debug(f"Loaded STC: {stc.data.shape[0]} sources, {stc.data.shape[1]} timepoints")

    # Apply atlas
    try:
        region_averages = apply_atlas_to_stc(stc, atlas, fs_subjects_dir)
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

    # Determine paths
    data_root = Path(config["paths"]["data_root"])
    if args.bids_root:
        bids_root = args.bids_root
    else:
        bids_root = data_root / "bids"

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
