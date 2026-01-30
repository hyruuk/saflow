"""Apply atlas/parcellation to source estimates (Stage 2b).

This script applies a cortical parcellation (e.g., aparc.a2009s) to morphed
source estimates, averaging time series within each ROI/label.

Outputs:
- ROI-averaged time series (pickled dictionary with data + region names)

Usage:
    # Single subject/run
    python -m code.source_reconstruction.apply_atlas --subject 04 --run 02

    # Multiple runs
    python -m code.source_reconstruction.apply_atlas --subject 04 --runs "02 03 04"

    # Specify atlas
    python -m code.source_reconstruction.apply_atlas --subject 04 --run 02 --atlas aparc.a2009s
"""

import argparse
import json
import logging
import pickle
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
        help="Space-separated run numbers (e.g., '02 03 04'). Mutually exclusive with --run.",
    )

    parser.add_argument(
        "--atlas",
        type=str,
        help="Atlas/parcellation name (e.g., 'aparc.a2009s', 'aparc'). Defaults to config.",
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

    # Validate run arguments
    if args.run and args.runs:
        parser.error("--run and --runs are mutually exclusive")

    if not args.run and not args.runs:
        parser.error("Either --run or --runs must be specified")

    return args


def create_atlas_paths(
    subject: str,
    run: str,
    atlas: str,
    processing: str,
    bids_root: Path,
) -> Dict[str, BIDSPath]:
    """Create BIDSPath objects for atlas application.

    Args:
        subject: Subject ID
        run: Run number
        atlas: Atlas name
        processing: Processing state
        bids_root: BIDS root directory

    Returns:
        Dictionary with 'input' and 'output' BIDSPath objects
    """
    derivatives_root = bids_root / "derivatives"

    # Input: morphed sources
    input_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing=processing,
        description="morphed",
        root=derivatives_root / "morphed_sources",
    )

    # Output: atlased sources
    output_bidspath = BIDSPath(
        subject=subject,
        task="gradCPT",
        run=run,
        datatype="meg",
        processing=processing,
        description=f"atlased_{atlas}",
        root=derivatives_root / f"atlased_sources_{atlas}",
    )
    output_bidspath.mkdir(exist_ok=True)

    return {"input": input_bidspath, "output": output_bidspath}


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
    bids_root: Path,
    skip_existing: bool,
) -> bool:
    """Process a single subject/run.

    Args:
        subject: Subject ID
        run: Run number
        atlas: Atlas name
        processing: Processing state
        config: Configuration dictionary
        bids_root: BIDS root directory
        skip_existing: Skip if output exists

    Returns:
        True if processing succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info(f"Processing sub-{subject}, run-{run}")
    logger.info(f"Atlas: {atlas}, Processing: {processing}")
    logger.info("=" * 80)

    # Get FreeSurfer subjects directory
    data_root = Path(config["paths"]["data_root"])
    fs_subjects_dir = data_root / config["paths"]["freesurfer_subjects_dir"]

    if not fs_subjects_dir.exists():
        logger.error(f"FreeSurfer subjects directory not found: {fs_subjects_dir}")
        return False

    # Create paths
    try:
        filepaths = create_atlas_paths(subject, run, atlas, processing, bids_root)
    except Exception as e:
        logger.error(f"Failed to create paths: {e}")
        return False

    # Check if output exists
    output_path = Path(str(filepaths["output"].fpath) + "-avg.pkl")
    if skip_existing and output_path.exists():
        logger.info(f"Output already exists, skipping: {output_path}")
        return True

    # Load source estimate
    input_path = Path(str(filepaths["input"].fpath) + "-stc.h5")
    if not input_path.exists():
        logger.error(f"Input source estimate not found: {input_path}")
        logger.error("Run run_inverse_solution.py first to generate source estimates")
        return False

    logger.info(f"Loading source estimate: {input_path}")
    try:
        stc = mne.read_source_estimate(str(input_path), verbose=False)
    except Exception as e:
        logger.error(f"Failed to load source estimate: {e}", exc_info=True)
        return False

    logger.info(f"Loaded STC: {stc.data.shape[0]} sources, {stc.data.shape[1]} timepoints")

    # Apply atlas
    logger.info(f"Applying atlas '{atlas}'...")
    try:
        region_averages = apply_atlas_to_stc(stc, atlas, fs_subjects_dir)
    except Exception as e:
        logger.error(f"Failed to apply atlas: {e}", exc_info=True)
        return False

    # Convert to arrays
    region_data = np.array(list(region_averages.values()))
    region_names = list(region_averages.keys())

    logger.info(f"Result: {len(region_names)} regions × {region_data.shape[1]} timepoints")

    # Save
    logger.info(f"Saving ROI-averaged data: {output_path}")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "data": region_data,
                    "region_names": region_names,
                    "sfreq": stc.sfreq,
                },
                f,
            )
    except Exception as e:
        logger.error(f"Failed to save output: {e}", exc_info=True)
        return False

    # Save metadata
    metadata = {
        "subject": subject,
        "run": run,
        "atlas": atlas,
        "processing": processing,
        "n_regions": len(region_names),
        "n_timepoints": region_data.shape[1],
        "sfreq": stc.sfreq,
        "region_names": region_names,
        "processing_successful": True,
    }

    metadata_path = Path(str(filepaths["output"].fpath) + "_params.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata: {metadata_path}")
    logger.info(f"✓ Successfully processed sub-{subject}, run-{run}")

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

    # Determine atlas
    if args.atlas:
        atlas = args.atlas
    else:
        atlas = config["source_reconstruction"]["atlas"]

    # Setup logging
    log_dir = Path(config["paths"]["logs"]) / "source_reconstruction"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_level=args.log_level,
        log_dir=log_dir,
        log_prefix=f"apply_atlas_sub-{args.subject}",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting atlas application (Stage 2b)")
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Atlas: {atlas}")

    # Determine BIDS root
    if args.bids_root:
        bids_root = args.bids_root
    else:
        data_root = Path(config["paths"]["data_root"])
        bids_root = data_root / config["paths"]["bids"]

    logger.info(f"BIDS root: {bids_root}")

    # Parse runs
    if args.run:
        runs = [args.run]
    else:
        runs = args.runs.split()

    logger.info(f"Processing {len(runs)} run(s): {', '.join(runs)}")

    # Process each run
    success_count = 0
    failed_runs = []

    for run in runs:
        # Validate subject/run
        if not validate_subject_run(args.subject, run, config):
            logger.error(f"Invalid subject/run combination: sub-{args.subject}, run-{run}")
            failed_runs.append(run)
            continue

        # Process
        success = process_single_run(
            subject=args.subject,
            run=run,
            atlas=atlas,
            processing=args.processing,
            config=config,
            bids_root=bids_root,
            skip_existing=args.skip_existing,
        )

        if success:
            success_count += 1
        else:
            failed_runs.append(run)

    # Summary
    logger.info("=" * 80)
    logger.info("Processing complete")
    logger.info(f"  Successful: {success_count}/{len(runs)}")
    if failed_runs:
        logger.warning(f"  Failed runs: {', '.join(failed_runs)}")
        return 1

    logger.info("✓ All runs processed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
