"""Run inverse solution computation (Stage 2).

This script performs source reconstruction from preprocessed MEG data:
1. Coregistration (MEG to MRI coordinate system)
2. Source space setup
3. BEM model creation
4. Forward solution computation
5. Noise covariance estimation
6. Inverse solution application
7. Morphing to fsaverage template

Outputs:
- Coregistration transform (trans/*.fif)
- Forward solution (fwd/*.fif)
- Noise covariance (noise_cov/*.fif)
- Source estimates (minimum-norm-estimate/*.h5)
- Morphed sources (morphed_sources/*.h5)

Usage:
    # Single subject/run
    python -m code.source_reconstruction.run_inverse_solution --subject 04 --run 02

    # Multiple runs
    python -m code.source_reconstruction.run_inverse_solution --subject 04 --runs "02 03 04"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import mne

from code.source_reconstruction import utils
from code.utils.config import load_config
from code.utils.logging_config import setup_logging
from code.utils.validation import validate_subject_run


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MEG source reconstruction (Stage 2)",
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
        "--bids-root",
        type=Path,
        help="Override BIDS root directory from config",
    )

    parser.add_argument(
        "--input-type",
        type=str,
        choices=["continuous", "epochs"],
        default="continuous",
        help="Input data type: 'continuous' (ICA-cleaned) or 'epochs' (ICA+AutoReject)",
    )

    parser.add_argument(
        "--processing",
        type=str,
        choices=["clean", "ica", "icaar"],
        default="clean",
        help="Processing state: 'clean' (continuous ICA-cleaned), 'ica' (epochs ICA-only), or 'icaar' (epochs ICA+AR)",
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

    # Validate run arguments (mutually exclusive but both optional - defaults to all runs)
    if args.run and args.runs:
        parser.error("--run and --runs are mutually exclusive")

    return args


def process_single_run(
    subject: str,
    run: str,
    config: dict,
    bids_root: Path,
    derivatives_root: Path,
    skip_existing: bool,
    input_type: str = "continuous",
    processing: str = "clean",
) -> bool:
    """Process a single subject/run.

    Args:
        subject: Subject ID
        run: Run number
        config: Configuration dictionary
        bids_root: BIDS root directory
        derivatives_root: Derivatives root directory
        skip_existing: Skip if output exists
        input_type: "continuous" or "epochs"
        processing: Processing state ("clean", "ica", or "icaar")

    Returns:
        True if processing succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info(f"Processing sub-{subject}, run-{run}")
    logger.info("=" * 80)

    # Get source reconstruction parameters
    src_config = config["source_reconstruction"]
    method = src_config["method"]
    snr = src_config["snr"]

    # Get computing parameters
    n_jobs = config["computing"]["n_jobs"]

    # Get FreeSurfer subjects directory (already expanded in config)
    fs_subjects_dir = Path(config["paths"]["freesurfer_subjects_dir"])

    if not fs_subjects_dir.exists():
        logger.error(f"FreeSurfer subjects directory not found: {fs_subjects_dir}")
        return False

    logger.info(f"FreeSurfer subjects directory: {fs_subjects_dir}")

    # Create output paths
    try:
        filepaths = utils.create_output_paths(
            subject, run, bids_root, derivatives_root
        )
    except Exception as e:
        logger.error(f"Failed to create output paths: {e}")
        return False

    # Check if output exists and skip if requested
    morph_output = Path(str(filepaths["morph"].fpath) + "-stc.h5")
    if skip_existing and morph_output.exists():
        logger.info(f"Output already exists, skipping: {morph_output}")
        return True

    # Check MRI availability
    mri_available = utils.check_mri_availability(subject, fs_subjects_dir)

    # Step 1: Coregistration
    trans_fpath = Path(str(filepaths["trans"].fpath) + ".fif")
    if not trans_fpath.exists():
        logger.info("[1/7] Computing coregistration...")
        try:
            utils.compute_coregistration(
                filepaths["raw"],
                filepaths["trans"],
                subject,
                fs_subjects_dir,
                mri_available=mri_available,
            )
        except Exception as e:
            logger.error(f"Coregistration failed: {e}", exc_info=True)
            return False
    else:
        logger.info(f"[1/7] Coregistration exists, loading: {trans_fpath}")

    # Step 2: Source space
    logger.info("[2/7] Setting up source space...")
    try:
        src = utils.setup_source_space(
            subject, fs_subjects_dir, mri_available=mri_available
        )
    except Exception as e:
        logger.error(f"Source space setup failed: {e}", exc_info=True)
        return False

    # Step 3: BEM model
    logger.info("[3/7] Creating BEM model...")
    try:
        bem = utils.create_bem_model(
            subject, fs_subjects_dir, mri_available=mri_available
        )
    except Exception as e:
        logger.error(f"BEM creation failed: {e}", exc_info=True)
        return False

    # Step 4: Forward solution
    fwd_fpath = Path(str(filepaths["fwd"].fpath) + ".fif")
    if not fwd_fpath.exists():
        logger.info("[4/7] Computing forward solution...")
        try:
            fwd = utils.compute_forward_solution(
                filepaths["preproc"],
                filepaths["trans"],
                src,
                bem,
                n_jobs=n_jobs,
            )
            mne.write_forward_solution(str(fwd_fpath), fwd, overwrite=True)
            logger.info(f"Saved forward solution: {fwd_fpath}")
        except Exception as e:
            logger.error(f"Forward solution computation failed: {e}", exc_info=True)
            return False
    else:
        logger.info(f"[4/7] Forward solution exists, loading: {fwd_fpath}")
        try:
            fwd = mne.read_forward_solution(fwd_fpath, verbose=False)
        except Exception as e:
            logger.error(f"Failed to load forward solution: {e}", exc_info=True)
            return False

    # Step 5: Noise covariance
    noise_cov_fpath = Path(str(filepaths["noise_cov"].fpath))
    if not noise_cov_fpath.exists():
        logger.info("[5/7] Computing noise covariance...")
        try:
            noise_cov = utils.compute_noise_covariance(filepaths["noise"])
            noise_cov.save(str(noise_cov_fpath), overwrite=True)
            logger.info(f"Saved noise covariance: {noise_cov_fpath}")
        except Exception as e:
            logger.error(f"Noise covariance computation failed: {e}", exc_info=True)
            return False
    else:
        logger.info(f"[5/7] Noise covariance exists, loading: {noise_cov_fpath}")
        try:
            noise_cov = mne.read_cov(str(noise_cov_fpath), verbose=False)
        except Exception as e:
            logger.error(f"Failed to load noise covariance: {e}", exc_info=True)
            return False

    # Step 6: Apply inverse solution
    logger.info(f"[6/7] Applying inverse solution (input_type={input_type}, processing={processing})...")

    try:
        if input_type == "continuous":
            # Apply inverse to continuous data
            stc = utils.apply_inverse_continuous(
                filepaths["preproc"],
                fwd,
                noise_cov,
                method=method,
                snr=snr,
            )
            stcs = [stc]  # Wrap in list for consistency

        elif input_type == "epochs":
            # Load epochs file instead of continuous
            from mne_bids import BIDSPath

            epochs_root = derivatives_root / "epochs" / f"sub-{subject}" / "meg"
            epochs_path = BIDSPath(
                subject=subject,
                session="recording",
                task="gradCPT",
                run=run,
                processing=processing,
                suffix="epo",
                extension=".fif",
                datatype="meg",
                root=epochs_root,
            )

            logger.info(f"Loading epochs from: {epochs_path}")

            # Apply inverse to epochs
            stcs = utils.apply_inverse_epochs(
                epochs_path,
                fwd,
                noise_cov,
                method=method,
                snr=snr,
            )

        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    except Exception as e:
        logger.error(f"Inverse solution failed: {e}", exc_info=True)
        return False

    # Step 7: Morph to fsaverage
    logger.info(f"[7/7] Morphing {len(stcs)} source estimate(s) to fsaverage...")
    try:
        morphed_stcs = utils.morph_to_fsaverage(
            stcs,
            fwd,
            subject,
            fs_subjects_dir,
            mri_available=mri_available,
        )

        # Save morphed source estimates
        if input_type == "continuous":
            # Save single continuous STC
            saved_path = utils.save_source_estimate(
                morphed_stcs[0],
                filepaths["morph"],
                epoch_idx=None,
            )
            logger.info(f"Saved morphed source estimate: {saved_path}")

        elif input_type == "epochs":
            # Save each epoch separately
            saved_paths = []
            for idx, morphed_stc in enumerate(morphed_stcs):
                saved_path = utils.save_source_estimate(
                    morphed_stc,
                    filepaths["morph"],
                    epoch_idx=idx,
                )
                saved_paths.append(saved_path)
            logger.info(f"Saved {len(saved_paths)} epoched source estimates")

    except Exception as e:
        logger.error(f"Morphing failed: {e}", exc_info=True)
        return False

    # Save metadata
    metadata = {
        "subject": subject,
        "run": run,
        "method": method,
        "snr": snr,
        "input_type": input_type,
        "processing": processing,
        "mri_available": mri_available,
        "n_sources": stcs[0].data.shape[0],
        "n_epochs" if input_type == "epochs" else "n_timepoints": len(stcs) if input_type == "epochs" else stcs[0].data.shape[1],
        "sfreq": 1.0 / stcs[0].tstep,
        "processing_successful": True,
    }

    metadata_path = Path(str(filepaths["morph"].fpath) + "_params.json")
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

    # Setup logging
    log_dir = Path(config["paths"]["logs"]) / "source_reconstruction"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        name=__name__,
        log_file=log_dir / f"source_recon_sub-{args.subject}.log",
        level=args.log_level,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting source reconstruction (Stage 2)")
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Input type: {args.input_type}")
    logger.info(f"Processing: {args.processing}")

    # Determine BIDS root
    data_root = Path(config["paths"]["data_root"])
    if args.bids_root:
        bids_root = args.bids_root
    else:
        bids_root = data_root / "bids"

    derivatives_root = data_root / config["paths"]["derivatives"]

    logger.info(f"BIDS root: {bids_root}")
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
            config=config,
            bids_root=bids_root,
            derivatives_root=derivatives_root,
            skip_existing=args.skip_existing,
            input_type=args.input_type,
            processing=args.processing,
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
